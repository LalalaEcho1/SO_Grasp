from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_external_graspnet_pointcloud_episode import sample_points  # noqa: E402
from scripts.run_graspnet_split_od_baselines import flatten_split_config  # noqa: E402
from scripts.run_graspnet_split_pointcloud_episodes import prediction_path_for_scene, realsense_path_for_scene  # noqa: E402
from stacked_grasping.gripper.external_graspnet_data import (  # noqa: E402
    GraspNetRealSenseSource,
    RealSenseFrame,
    depth_to_point_cloud,
)
from stacked_grasping.gripper.external_graspnet_scene import (  # noqa: E402
    ExternalGraspNetFrameScene,
    build_external_graspnet_episode_inputs,
)
from stacked_grasping.gripper.graspnet_binding import (  # noqa: E402
    GraspNetPredictionSource,
    bind_graspnet_records_to_frame_labels,
)
from stacked_grasping.gripper.pointcloud_feasibility import (  # noqa: E402
    PointCloudCollisionConfig,
    assess_scene_bound_graspnet_pointcloud_feasibility,
)
from stacked_grasping.planning.episode import EpisodeResult, run_policy_episode  # noqa: E402
from stacked_grasping.planning.policies import VALID_POLICIES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare grasp ordering policies on selected GraspNet split frames."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "graspnet_candidate_split.json")
    parser.add_argument("--scene-root", type=Path, help="Override scene root. Defaults to config scene_root.")
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "graspnet_split_policy_comparison")
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--policies", nargs="+", default=("adaptive-score-v2-graspnet", "od-only"))
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--factor-depth", type=int, default=1000)
    parser.add_argument("--binding-pixel-radius", type=int, default=3)
    parser.add_argument("--binding-depth-tolerance-m", type=float, default=0.12)
    parser.add_argument("--min-points-per-object", type=int, default=20)
    parser.add_argument("--min-half-extent", type=float, default=0.01)
    parser.add_argument("--object-padding", type=float, default=0.002)
    parser.add_argument("--min-boundary-pixels", type=int, default=50)
    parser.add_argument("--point-sample-limit", type=int, default=60000)
    parser.add_argument("--point-sample-seed", type=int, default=1234)
    parser.add_argument("--risk-threshold", type=float, default=0.45)
    parser.add_argument("--max-opening", type=float, default=0.085)
    parser.add_argument("--collision-threshold", type=float, default=0.01)
    parser.add_argument("--empty-threshold", type=float, default=0.01)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_graspnet_split_policy_comparison(
        config_path=args.config,
        scene_root=args.scene_root,
        prediction_root=args.prediction_root,
        out_dir=args.out_dir,
        camera=args.camera,
        policies=tuple(args.policies),
        max_steps=args.max_steps,
        factor_depth=args.factor_depth,
        binding_pixel_radius=args.binding_pixel_radius,
        binding_depth_tolerance_m=args.binding_depth_tolerance_m,
        min_points_per_object=args.min_points_per_object,
        min_half_extent=args.min_half_extent,
        object_padding=args.object_padding,
        min_boundary_pixels=args.min_boundary_pixels,
        point_sample_limit=args.point_sample_limit,
        point_sample_seed=args.point_sample_seed,
        risk_threshold=args.risk_threshold,
        pointcloud_config=PointCloudCollisionConfig(
            max_opening=args.max_opening,
            collision_threshold=args.collision_threshold,
            empty_threshold=args.empty_threshold,
        ),
        save_outputs=not args.no_save,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("GraspNet split policy comparison finished")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  output_saved: {summary['output_saved']}")
    print(f"  frames: {summary['processed_frame_count']}/{summary['frame_count']}")
    print(f"  policies: {', '.join(summary['policies'])}")
    print("  success rates:")
    for policy, metrics in summary["aggregate"]["by_policy"].items():
        print(f"   {policy}: {metrics['success_count']}/{metrics['frame_count']} = {metrics['success_rate']}")


def run_graspnet_split_policy_comparison(
    *,
    config_path: str | Path,
    scene_root: str | Path | None = None,
    prediction_root: str | Path,
    out_dir: str | Path,
    policies: Sequence[str] = ("adaptive-score-v2-graspnet", "od-only"),
    camera: str = "realsense",
    max_steps: int | None = 1,
    factor_depth: int = 1000,
    binding_pixel_radius: int = 3,
    binding_depth_tolerance_m: float | None = 0.12,
    min_points_per_object: int = 20,
    min_half_extent: float = 0.01,
    object_padding: float = 0.002,
    min_boundary_pixels: int = 50,
    point_sample_limit: int | None = 60000,
    point_sample_seed: int = 1234,
    risk_threshold: float = 0.45,
    pointcloud_config: PointCloudCollisionConfig | None = None,
    save_outputs: bool = True,
) -> dict[str, object]:
    _validate_policies(policies)
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    resolved_scene_root = _resolve_scene_root(scene_root, config, base_dir=config_file.parent.parent)
    resolved_prediction_root = Path(prediction_root)
    target = Path(out_dir)
    if save_outputs:
        target.mkdir(parents=True, exist_ok=True)

    entries = flatten_split_config(config)
    results: list[dict[str, object]] = []
    missing_predictions: list[dict[str, object]] = []
    for entry in entries:
        frame_result = run_split_entry_policy_comparison(
            entry,
            scene_root=resolved_scene_root,
            prediction_root=resolved_prediction_root,
            camera=camera,
            policies=policies,
            max_steps=max_steps,
            factor_depth=factor_depth,
            binding_pixel_radius=binding_pixel_radius,
            binding_depth_tolerance_m=binding_depth_tolerance_m,
            min_points_per_object=min_points_per_object,
            min_half_extent=min_half_extent,
            object_padding=object_padding,
            min_boundary_pixels=min_boundary_pixels,
            point_sample_limit=point_sample_limit,
            point_sample_seed=point_sample_seed,
            risk_threshold=risk_threshold,
            pointcloud_config=pointcloud_config,
        )
        if frame_result.get("missing_prediction"):
            missing_predictions.append(frame_result["missing_prediction"])
            continue
        results.extend(frame_result["policy_results"])

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(config_file),
        "scene_root": str(resolved_scene_root),
        "prediction_root": str(resolved_prediction_root),
        "output_dir": str(target),
        "output_saved": bool(save_outputs),
        "policies": list(policies),
        "policy_count": len(policies),
        "frame_count": len(entries),
        "processed_frame_count": len({(row["scene"], row["frame"]) for row in results}),
        "result_count": len(results),
        "missing_prediction_count": len(missing_predictions),
        "missing_predictions": missing_predictions,
        "aggregate": aggregate_policy_results(results),
        "results": results,
    }
    if save_outputs:
        (target / "split_policy_comparison_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_policy_results_csv(target / "split_policy_comparison_results.csv", results)
        write_missing_predictions_csv(target / "missing_policy_comparison_predictions.csv", missing_predictions)
    return summary


def run_split_entry_policy_comparison(
    entry: dict[str, object],
    *,
    scene_root: Path,
    prediction_root: Path,
    camera: str,
    policies: Sequence[str],
    max_steps: int | None,
    factor_depth: int,
    binding_pixel_radius: int,
    binding_depth_tolerance_m: float | None,
    min_points_per_object: int,
    min_half_extent: float,
    object_padding: float,
    min_boundary_pixels: int,
    point_sample_limit: int | None,
    point_sample_seed: int,
    risk_threshold: float,
    pointcloud_config: PointCloudCollisionConfig | None,
) -> dict[str, object]:
    scene_name = str(entry["scene"])
    frame_id = str(entry["frame"])
    prediction_path = prediction_path_for_scene(prediction_root, scene_name, camera)
    prediction_file = prediction_path / f"{frame_id}.npy" if prediction_path.is_dir() else prediction_path
    if not prediction_file.exists():
        return {"missing_prediction": {**_entry_prefix(entry), "prediction_file": str(prediction_file)}}

    realsense_path = realsense_path_for_scene(scene_root, scene_name, camera)
    with GraspNetRealSenseSource.open(realsense_path) as realsense_source, GraspNetPredictionSource.open(prediction_path) as prediction_source:
        loaded_frame = realsense_source.load_frame(frame_id)
        frame = RealSenseFrame(
            frame=loaded_frame.frame,
            color=loaded_frame.color,
            depth_raw=loaded_frame.depth_raw,
            label=loaded_frame.label,
            intrinsic_matrix=loaded_frame.intrinsic_matrix,
            camera_pose=loaded_frame.camera_pose,
            cam0_wrt_table=loaded_frame.cam0_wrt_table,
            factor_depth=factor_depth,
        )
        annotations = realsense_source.load_annotation_objects(frame.frame)
        records = prediction_source.load_records(frame.frame)

    bindings = bind_graspnet_records_to_frame_labels(
        records,
        frame,
        annotations,
        pixel_radius=binding_pixel_radius,
        depth_tolerance_m=binding_depth_tolerance_m,
    )
    episode_inputs = build_external_graspnet_episode_inputs(
        frame,
        annotations,
        bindings,
        min_points_per_object=min_points_per_object,
        min_half_extent=min_half_extent,
        padding=object_padding,
        min_boundary_pixels=min_boundary_pixels,
    )
    objects = episode_inputs.scene.read_objects()
    contact_pairs = episode_inputs.scene.read_object_contact_pairs()
    points, _ = depth_to_point_cloud(frame.depth_raw, frame.intrinsic_matrix, factor_depth=frame.factor_depth)
    points = sample_points(points, limit=point_sample_limit, seed=point_sample_seed)
    common = {
        **_entry_prefix(entry),
        "total_candidates": len(records),
        "bound_count": sum(1 for binding in bindings if binding.status == "bound"),
        "object_count": len(objects),
        "contact_pair_count": len(contact_pairs),
        "prediction_path": str(prediction_path),
    }
    common["binding_ratio"] = (
        round(float(common["bound_count"]) / float(common["total_candidates"]), 6)
        if common["total_candidates"]
        else None
    )

    policy_results = []
    for policy in policies:
        scene_copy = ExternalGraspNetFrameScene(objects, contact_pairs)

        def gripper_provider(active_objects):
            return assess_scene_bound_graspnet_pointcloud_feasibility(
                active_objects,
                points,
                bindings,
                config=pointcloud_config,
            )

        result = run_policy_episode(
            scene_copy,
            policy=policy,
            max_steps=max_steps,
            post_grasp_settle_steps=0,
            seed=0,
            failure_mode="risk-threshold",
            risk_threshold=risk_threshold,
            gripper_feasibility_provider=gripper_provider,
        )
        policy_results.append(policy_result_row(common, policy, result, initial_object_count=len(objects)))
    return {"policy_results": policy_results}


def policy_result_row(
    common: dict[str, object],
    policy: str,
    result: EpisodeResult,
    *,
    initial_object_count: int,
) -> dict[str, object]:
    first_step = result.steps[0] if result.steps else None
    successful_steps = sum(1 for step in result.steps if step.grasp_success)
    clearance_rate = round(float(successful_steps) / float(max(initial_object_count, 1)), 6)
    selected_pose = first_step.selected_grasp_pose if first_step and first_step.selected_grasp_pose else None
    return {
        **common,
        "policy": policy,
        "num_steps": len(result.steps),
        "grasp_sequence": list(result.grasp_sequence),
        "first_selected_object": first_step.selected_object if first_step else None,
        "first_grasp_success": bool(first_step.grasp_success) if first_step else False,
        "grasp_success": bool(result.steps and all(step.grasp_success for step in result.steps)),
        "first_grasp_risk": round(float(first_step.grasp_risk), 6) if first_step else None,
        "failure_mode": result.failure_mode,
        "failure_reason": first_step.failure_reason if first_step else result.failure_reason,
        "clearance_rate": clearance_rate,
        "remaining_object_count": len(result.final_objects),
        "gripper_feasible": first_step.gripper_feasible if first_step else None,
        "gripper_feasible_grasp_count": first_step.gripper_feasible_grasp_count if first_step else None,
        "gripper_collision_risk": round(float(first_step.gripper_collision_risk), 6) if first_step else None,
        "selected_pose_generator": selected_pose.generator if selected_pose else None,
        "selected_pose_score": round(float(selected_pose.score), 6) if selected_pose else None,
    }


def aggregate_policy_results(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy"])].append(dict(row))
    return {
        "by_policy": {
            policy: {
                "frame_count": len(items),
                "success_count": sum(1 for item in items if item.get("grasp_success")),
                "success_rate": _rate(sum(1 for item in items if item.get("grasp_success")), len(items)),
                "mean_first_grasp_risk": _mean(
                    [float(item["first_grasp_risk"]) for item in items if item.get("first_grasp_risk") is not None]
                ),
                "mean_clearance_rate": _mean(
                    [float(item["clearance_rate"]) for item in items if item.get("clearance_rate") is not None]
                ),
                "failure_reason_counts": dict(
                    Counter(str(item.get("failure_reason")) for item in items if item.get("failure_reason") is not None)
                ),
            }
            for policy, items in sorted(by_policy.items())
        }
    }


def write_policy_results_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "group_id",
        "split",
        "scene",
        "frame",
        "role",
        "policy",
        "total_candidates",
        "bound_count",
        "binding_ratio",
        "object_count",
        "contact_pair_count",
        "num_steps",
        "grasp_sequence",
        "first_selected_object",
        "first_grasp_success",
        "grasp_success",
        "first_grasp_risk",
        "failure_mode",
        "failure_reason",
        "clearance_rate",
        "remaining_object_count",
        "gripper_feasible",
        "gripper_feasible_grasp_count",
        "gripper_collision_risk",
        "selected_pose_generator",
        "selected_pose_score",
        "prediction_path",
    ]
    _write_csv(path, fieldnames, rows)


def write_missing_predictions_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    _write_csv(path, ["group_id", "split", "scene", "frame", "role", "prediction_file"], rows)


def _validate_policies(policies: Sequence[str]) -> None:
    invalid = [policy for policy in policies if policy not in VALID_POLICIES]
    if invalid:
        raise ValueError(f"Unknown policies: {', '.join(invalid)}")


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


def _resolve_scene_root(scene_root: str | Path | None, config: dict[str, object], *, base_dir: Path) -> Path:
    raw = Path(scene_root) if scene_root is not None else Path(str(config["scene_root"]))
    return raw if raw.is_absolute() else base_dir / raw


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _rate(count: int, total: int) -> float | None:
    return round(float(count) / float(total), 6) if total else None


if __name__ == "__main__":
    main()
