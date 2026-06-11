from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_graspnet_split_od_baselines import flatten_split_config  # noqa: E402
from scripts.validate_graspnet_mujoco_grasp import validate_graspnet_mujoco_grasp  # noqa: E402
from stacked_grasping.gripper.mujoco_grasp_validation import LiteGraspValidationConfig  # noqa: E402


Validator = Callable[..., dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run top-K GraspNet dynamic MuJoCo validation for selected split frames."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "graspnet_candidate_split.json")
    parser.add_argument("--scene-root", type=Path, help="Override scene root. Defaults to config scene_root.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "graspnet_mujoco_dynamic_topk")
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--mesh-file", default="textured.obj")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Run with a temporary scratch directory and do not write outputs.")
    parser.add_argument("--no-align-to-table", dest="align_to_table", action="store_false")
    parser.set_defaults(align_to_table=True)
    parser.add_argument("--gripper-opening-margin", type=float, default=0.004)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--approach-steps", type=int, default=40)
    parser.add_argument("--close-steps", type=int, default=40)
    parser.add_argument("--lift-steps", type=int, default=60)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--pregrasp-distance", type=float, default=0.06)
    parser.add_argument("--lift-distance", type=float, default=0.08)
    parser.add_argument("--lift-success-threshold-m", type=float, default=0.02)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_graspnet_split_dynamic_topk(
        config_path=args.config,
        scene_root=args.scene_root,
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        out_dir=args.out_dir,
        camera=args.camera,
        mesh_file=args.mesh_file,
        top_k=args.top_k,
        stop_on_success=args.stop_on_success,
        save_outputs=not args.no_save,
        align_to_table=args.align_to_table,
        gripper_opening_margin=args.gripper_opening_margin,
        validation_config=LiteGraspValidationConfig(
            settle_steps=args.settle_steps,
            approach_steps=args.approach_steps,
            close_steps=args.close_steps,
            lift_steps=args.lift_steps,
            hold_steps=args.hold_steps,
            pregrasp_distance=args.pregrasp_distance,
            lift_distance=args.lift_distance,
            lift_success_threshold_m=args.lift_success_threshold_m,
        ),
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("GraspNet split dynamic top-K validation finished")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  output_saved: {summary['output_saved']}")
    print(f"  frames: {summary['processed_frame_count']}/{summary['frame_count']}")
    print(f"  top_k: {summary['top_k']}")
    print(f"  candidate_evaluations: {summary['candidate_evaluation_count']}")
    print(f"  top1_success_rate: {summary['top1_success_rate']}")
    print(f"  topk_success_rate: {summary['topk_success_rate']}")
    print(f"  failure_reason_counts: {summary['failure_reason_counts']}")


def run_graspnet_split_dynamic_topk(
    *,
    config_path: str | Path,
    scene_root: str | Path | None = None,
    dataset_root: str | Path,
    prediction_root: str | Path,
    out_dir: str | Path,
    camera: str = "realsense",
    mesh_file: str = "textured.obj",
    top_k: int = 10,
    stop_on_success: bool = False,
    save_outputs: bool = True,
    align_to_table: bool = True,
    gripper_opening_margin: float = 0.004,
    validation_config: LiteGraspValidationConfig | None = None,
    validator: Validator = validate_graspnet_mujoco_grasp,
) -> dict[str, object]:
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive.")

    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    resolved_scene_root = _resolve_scene_root(scene_root, config, base_dir=config_file.parent.parent)
    requested_target = Path(out_dir)

    if save_outputs:
        return _run_graspnet_split_dynamic_topk_core(
            config_file=config_file,
            resolved_scene_root=resolved_scene_root,
            dataset_root=Path(dataset_root),
            prediction_root=Path(prediction_root),
            requested_target=requested_target,
            scratch_target=requested_target,
            camera=camera,
            mesh_file=mesh_file,
            top_k=int(top_k),
            stop_on_success=stop_on_success,
            save_outputs=True,
            align_to_table=align_to_table,
            gripper_opening_margin=gripper_opening_margin,
            validation_config=validation_config,
            validator=validator,
        )

    with tempfile.TemporaryDirectory(prefix="graspnet_dynamic_topk_") as scratch_dir:
        return _run_graspnet_split_dynamic_topk_core(
            config_file=config_file,
            resolved_scene_root=resolved_scene_root,
            dataset_root=Path(dataset_root),
            prediction_root=Path(prediction_root),
            requested_target=requested_target,
            scratch_target=Path(scratch_dir),
            camera=camera,
            mesh_file=mesh_file,
            top_k=int(top_k),
            stop_on_success=stop_on_success,
            save_outputs=False,
            align_to_table=align_to_table,
            gripper_opening_margin=gripper_opening_margin,
            validation_config=validation_config,
            validator=validator,
        )


def _run_graspnet_split_dynamic_topk_core(
    *,
    config_file: Path,
    resolved_scene_root: Path,
    dataset_root: Path,
    prediction_root: Path,
    requested_target: Path,
    scratch_target: Path,
    camera: str,
    mesh_file: str,
    top_k: int,
    stop_on_success: bool,
    save_outputs: bool,
    align_to_table: bool,
    gripper_opening_margin: float,
    validation_config: LiteGraspValidationConfig | None,
    validator: Validator,
) -> dict[str, object]:
    if save_outputs:
        scratch_target.mkdir(parents=True, exist_ok=True)
    entries = flatten_split_config(json.loads(config_file.read_text(encoding="utf-8")))
    frame_results: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    missing_frames: list[dict[str, object]] = []
    for entry in entries:
        try:
            frame_result, rows = run_entry_dynamic_topk(
                entry,
                scene_root=resolved_scene_root,
                dataset_root=dataset_root,
                prediction_root=prediction_root,
                out_dir=scratch_target,
                camera=camera,
                mesh_file=mesh_file,
                top_k=top_k,
                stop_on_success=stop_on_success,
                align_to_table=align_to_table,
                gripper_opening_margin=gripper_opening_margin,
                validation_config=validation_config,
                validator=validator,
            )
        except FileNotFoundError as exc:
            missing_frames.append({**entry, "error": str(exc)})
            continue
        frame_results.append(frame_result)
        candidate_rows.extend(rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(config_file),
        "scene_root": str(resolved_scene_root),
        "dataset_root": str(dataset_root),
        "prediction_root": str(prediction_root),
        "output_dir": str(requested_target),
        "output_saved": bool(save_outputs),
        "top_k": top_k,
        "stop_on_success": bool(stop_on_success),
        "frame_count": len(entries),
        "processed_frame_count": len(frame_results),
        "missing_frame_count": len(missing_frames),
        "missing_frames": missing_frames,
        "candidate_evaluation_count": len(candidate_rows),
        "top1_lift_success_count": sum(1 for row in frame_results if row.get("top1_lift_success")),
        "topk_lift_success_count": sum(1 for row in frame_results if row.get("topk_lift_success")),
        "top1_success_rate": _rate(sum(1 for row in frame_results if row.get("top1_lift_success")), len(frame_results)),
        "topk_success_rate": _rate(sum(1 for row in frame_results if row.get("topk_lift_success")), len(frame_results)),
        "simulation_unstable_count": sum(1 for row in candidate_rows if row.get("simulation_unstable")),
        "failure_reason_counts": dict(Counter(str(row.get("failure_reason")) for row in candidate_rows)),
        "frame_results": frame_results,
    }
    if save_outputs:
        (scratch_target / "split_dynamic_topk_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_frame_results_csv(scratch_target / "split_dynamic_topk_frame_results.csv", frame_results)
        write_candidate_results_csv(scratch_target / "split_dynamic_topk_candidate_results.csv", candidate_rows)
        write_missing_frames_csv(scratch_target / "missing_dynamic_topk_frames.csv", missing_frames)
    return summary


def run_entry_dynamic_topk(
    entry: dict[str, object],
    *,
    scene_root: Path,
    dataset_root: Path,
    prediction_root: Path,
    out_dir: Path,
    camera: str,
    mesh_file: str,
    top_k: int,
    stop_on_success: bool,
    align_to_table: bool,
    gripper_opening_margin: float,
    validation_config: LiteGraspValidationConfig | None,
    validator: Validator,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    scene = str(entry["scene"])
    frame = str(entry["frame"])
    candidate_summaries: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for rank in range(int(top_k)):
        try:
            summary = validator(
                scene_root=scene_root,
                dataset_root=dataset_root,
                prediction_root=prediction_root,
                out_dir=out_dir,
                scene=scene,
                frame=frame,
                camera=camera,
                mesh_file=mesh_file,
                candidate_rank=rank,
                align_to_table=align_to_table,
                gripper_opening_margin=gripper_opening_margin,
                validation_config=validation_config,
            )
        except ValueError as exc:
            if "candidate_rank" in str(exc) and "out of range" in str(exc):
                break
            raise
        candidate_summaries.append(summary)
        row = candidate_row_from_summary(entry, summary)
        candidate_rows.append(row)
        validation = summary.get("validation", {})
        if stop_on_success and bool(validation.get("lift_success")):
            break

    best_candidate = choose_best_candidate(candidate_summaries)
    top1 = candidate_summaries[0] if candidate_summaries else None
    top1_validation = top1.get("validation", {}) if top1 else {}
    best_validation = best_candidate.get("validation", {}) if best_candidate else {}
    frame_result = {
        **_entry_prefix(entry),
        "evaluated_candidate_count": len(candidate_summaries),
        "top1_lift_success": bool(top1_validation.get("lift_success")),
        "top1_failure_reason": top1_validation.get("failure_reason"),
        "top1_target_lift_delta_m": top1_validation.get("target_lift_delta_m"),
        "topk_lift_success": any(bool(item.get("validation", {}).get("lift_success")) for item in candidate_summaries),
        "best_candidate_rank": best_candidate.get("candidate_rank") if best_candidate else None,
        "best_selected_grasp_score": best_candidate.get("selected_grasp_score") if best_candidate else None,
        "best_target_object_id": best_candidate.get("target_object_id") if best_candidate else None,
        "best_target_object_name": best_candidate.get("target_object_name") if best_candidate else None,
        "best_lift_success": bool(best_validation.get("lift_success")),
        "best_failure_reason": best_validation.get("failure_reason"),
        "best_simulation_unstable": bool(best_validation.get("simulation_unstable")),
        "best_target_lift_delta_m": best_validation.get("target_lift_delta_m"),
        "best_max_target_lift_delta_m": best_validation.get("max_target_lift_delta_m"),
        "best_lift_contact_step_count": best_validation.get("lift_contact_step_count"),
        "candidate_results": candidate_rows,
    }
    return frame_result, candidate_rows


def choose_best_candidate(candidate_summaries: Sequence[dict[str, object]]) -> dict[str, object] | None:
    if not candidate_summaries:
        return None
    return max(candidate_summaries, key=_candidate_sort_key)


def candidate_row_from_summary(entry: dict[str, object], summary: dict[str, object]) -> dict[str, object]:
    validation = dict(summary.get("validation", {}))
    return {
        **_entry_prefix(entry),
        "candidate_rank": summary.get("candidate_rank"),
        "selected_grasp_score": summary.get("selected_grasp_score"),
        "target_object_id": summary.get("target_object_id"),
        "target_object_name": summary.get("target_object_name"),
        "target_body_name": summary.get("target_body_name"),
        "compile_success": validation.get("compile_success"),
        "lift_success": validation.get("lift_success"),
        "failure_reason": validation.get("failure_reason"),
        "simulation_unstable": validation.get("simulation_unstable"),
        "target_contact_step_count": validation.get("target_contact_step_count"),
        "lift_contact_step_count": validation.get("lift_contact_step_count"),
        "target_lift_delta_m": validation.get("target_lift_delta_m"),
        "max_target_lift_delta_m": validation.get("max_target_lift_delta_m"),
        "xml_path": summary.get("xml_path"),
        "summary_path": summary.get("summary_path"),
    }


def write_frame_results_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "group_id",
        "split",
        "scene",
        "frame",
        "role",
        "evaluated_candidate_count",
        "top1_lift_success",
        "top1_failure_reason",
        "top1_target_lift_delta_m",
        "topk_lift_success",
        "best_candidate_rank",
        "best_selected_grasp_score",
        "best_target_object_id",
        "best_target_object_name",
        "best_lift_success",
        "best_failure_reason",
        "best_simulation_unstable",
        "best_target_lift_delta_m",
        "best_max_target_lift_delta_m",
        "best_lift_contact_step_count",
    ]
    _write_csv(path, fieldnames, rows)


def write_candidate_results_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "group_id",
        "split",
        "scene",
        "frame",
        "role",
        "candidate_rank",
        "selected_grasp_score",
        "target_object_id",
        "target_object_name",
        "compile_success",
        "lift_success",
        "failure_reason",
        "simulation_unstable",
        "target_contact_step_count",
        "lift_contact_step_count",
        "target_lift_delta_m",
        "max_target_lift_delta_m",
        "xml_path",
        "summary_path",
    ]
    _write_csv(path, fieldnames, rows)


def write_missing_frames_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    _write_csv(path, ["group_id", "split", "scene", "frame", "role", "error"], rows)


def _candidate_sort_key(summary: dict[str, object]) -> tuple[float, ...]:
    validation = dict(summary.get("validation", {}))
    rank = int(summary.get("candidate_rank", 10**6))
    return (
        1.0 if validation.get("lift_success") else 0.0,
        0.0 if validation.get("simulation_unstable") else 1.0,
        _float_value(validation.get("target_lift_delta_m")),
        _float_value(validation.get("max_target_lift_delta_m")),
        _float_value(validation.get("lift_contact_step_count")),
        _float_value(validation.get("target_contact_step_count")),
        _float_value(summary.get("selected_grasp_score")),
        -float(rank),
    )


def _resolve_scene_root(scene_root: str | Path | None, config: dict[str, object], *, base_dir: Path) -> Path:
    raw = Path(scene_root) if scene_root is not None else Path(str(config["scene_root"]))
    return raw if raw.is_absolute() else base_dir / raw


def _entry_prefix(entry: dict[str, object]) -> dict[str, object]:
    return {
        "group_id": entry.get("group_id"),
        "group_name": entry.get("group_name"),
        "split": entry.get("split"),
        "scene": entry.get("scene"),
        "frame": entry.get("frame"),
        "role": entry.get("role"),
        "tags": list(entry.get("tags", [])),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _rate(count: int, total: int) -> float | None:
    return round(float(count) / float(total), 6) if total else None


def _float_value(value: object) -> float:
    if value is None:
        return float("-inf")
    return float(value)


if __name__ == "__main__":
    main()
