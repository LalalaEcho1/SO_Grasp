from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stacked_grasping.gripper.external_graspnet_data import (  # noqa: E402
    GraspNetRealSenseSource,
    RealSenseFrame,
    assess_single_view_od_sufficiency,
    normalize_frame_id,
)
from stacked_grasping.gripper.external_graspnet_scene import build_external_graspnet_episode_inputs  # noqa: E402
from stacked_grasping.planning.episode import run_policy_episode  # noqa: E402
from stacked_grasping.planning.policies import VALID_POLICIES  # noqa: E402


DEFAULT_POLICIES = (
    "adaptive-score-v2",
    "adaptive-score-v2-gripper",
    "od-only",
    "highest-first",
    "lowest-blocked",
    "random",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run config-driven OD baseline prechecks on selected GraspNet frames.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "graspnet_candidate_split.json")
    parser.add_argument("--scene-root", type=Path, help="Override scene root. Defaults to config scene_root.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "graspnet_split_od_baselines")
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--factor-depth", type=int, default=1000)
    parser.add_argument("--min-points-per-object", type=int, default=20)
    parser.add_argument("--min-half-extent", type=float, default=0.01)
    parser.add_argument("--object-padding", type=float, default=0.002)
    parser.add_argument("--min-boundary-pixels", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_graspnet_split_od_baselines(
        config_path=args.config,
        scene_root=args.scene_root,
        out_dir=args.out_dir,
        camera=args.camera,
        policies=tuple(args.policies),
        max_steps=args.max_steps,
        seed=args.seed,
        factor_depth=args.factor_depth,
        min_points_per_object=args.min_points_per_object,
        min_half_extent=args.min_half_extent,
        object_padding=args.object_padding,
        min_boundary_pixels=args.min_boundary_pixels,
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    aggregate = summary["aggregate"]
    print("GraspNet split OD baseline precheck finished")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  frames: {summary['frame_count']}")
    print(f"  policies: {', '.join(summary['policies'])}")
    print(f"  results: {summary['result_count']}")
    print(f"  mean_hidden_object_count: {aggregate['mean_hidden_object_count']}")
    print(f"  mean_unobservable_pair_count: {aggregate['mean_unobservable_pair_count']}")
    print("  policy_mean_clearance_rate:")
    for policy, value in aggregate["policy_mean_clearance_rate"].items():
        print(f"   {policy}: {value}")


def run_graspnet_split_od_baselines(
    *,
    config_path: str | Path,
    scene_root: str | Path | None = None,
    out_dir: str | Path,
    camera: str = "realsense",
    policies: Sequence[str] = DEFAULT_POLICIES,
    max_steps: int | None = 3,
    seed: int = 0,
    factor_depth: int = 1000,
    min_points_per_object: int = 20,
    min_half_extent: float = 0.01,
    object_padding: float = 0.002,
    min_boundary_pixels: int = 50,
) -> dict[str, object]:
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    root = _resolve_scene_root(scene_root, config, base_dir=config_file.parent.parent)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    selected_policies = _validate_policies(policies)
    entries = flatten_split_config(config)

    results: list[dict[str, object]] = []
    for entry in entries:
        results.extend(
            run_entry_policy_rows(
                entry,
                scene_root=root,
                camera=camera,
                policies=selected_policies,
                max_steps=max_steps,
                seed=seed,
                factor_depth=factor_depth,
                min_points_per_object=min_points_per_object,
                min_half_extent=min_half_extent,
                object_padding=object_padding,
                min_boundary_pixels=min_boundary_pixels,
            )
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(config_file),
        "scene_root": str(root),
        "output_dir": str(target),
        "split_policy": config.get("split_policy"),
        "policies": list(selected_policies),
        "max_steps": max_steps,
        "frame_count": len(entries),
        "policy_count": len(selected_policies),
        "result_count": len(results),
        "aggregate": aggregate_results(results),
        "entries": entries,
        "results": results,
    }
    (target / "split_od_baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_results_csv(target / "split_od_baseline_results.csv", results)
    write_entries_csv(target / "split_entries.csv", entries)
    return summary


def flatten_split_config(config: dict[str, object]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for group in config.get("groups", []):
        group_dict = dict(group)
        for scene in group_dict.get("scenes", []):
            scene_dict = dict(scene)
            for frame in scene_dict.get("selected_frames", []):
                entries.append(
                    {
                        "group_id": str(group_dict["group_id"]),
                        "group_name": str(group_dict.get("name", "")),
                        "scene": str(scene_dict["scene"]),
                        "frame": normalize_frame_id(frame),
                        "split": str(scene_dict["split"]),
                        "role": str(scene_dict.get("role", "")),
                        "tags": list(scene_dict.get("tags", [])),
                    }
                )
    return entries


def run_entry_policy_rows(
    entry: dict[str, object],
    *,
    scene_root: Path,
    camera: str,
    policies: Sequence[str],
    max_steps: int | None,
    seed: int,
    factor_depth: int,
    min_points_per_object: int,
    min_half_extent: float,
    object_padding: float,
    min_boundary_pixels: int,
) -> list[dict[str, object]]:
    scene_name = str(entry["scene"])
    frame_id = normalize_frame_id(entry["frame"])
    source_path = scene_root / scene_name / camera
    if not source_path.is_dir():
        source_path = scene_root / scene_name

    with GraspNetRealSenseSource.open(source_path) as source:
        loaded = source.load_frame(frame_id)
        frame = RealSenseFrame(
            frame=loaded.frame,
            color=loaded.color,
            depth_raw=loaded.depth_raw,
            label=loaded.label,
            intrinsic_matrix=loaded.intrinsic_matrix,
            camera_pose=loaded.camera_pose,
            cam0_wrt_table=loaded.cam0_wrt_table,
            factor_depth=factor_depth,
        )
        annotations = source.load_annotation_objects(frame.frame)
        od_report = assess_single_view_od_sufficiency(frame, annotations, min_boundary_pixels=min_boundary_pixels)

    rows: list[dict[str, object]] = []
    for policy in policies:
        episode_inputs = build_external_graspnet_episode_inputs(
            frame,
            annotations,
            [],
            min_points_per_object=min_points_per_object,
            min_half_extent=min_half_extent,
            padding=object_padding,
            min_boundary_pixels=min_boundary_pixels,
        )
        initial_object_count = len(episode_inputs.scene.read_objects())
        contact_pair_count = len(episode_inputs.scene.read_object_contact_pairs())
        result = run_policy_episode(
            episode_inputs.scene,
            policy=policy,
            max_steps=max_steps,
            post_grasp_settle_steps=0,
            seed=seed,
            failure_mode="none",
        )
        final_object_count = len(result.final_objects)
        rows.append(
            {
                **_entry_prefix(entry),
                "policy": policy,
                "object_count": initial_object_count,
                "contact_pair_count": contact_pair_count,
                "complete_object_count": int(od_report["complete_object_count"]),
                "visible_object_count": int(od_report["visible_object_count"]),
                "hidden_object_count": int(od_report["hidden_object_count"]),
                "direct_visible_boundary_pair_count": int(od_report["direct_visible_boundary_pair_count"]),
                "unobservable_pair_count": int(od_report["unobservable_pair_count"]),
                "direct_pair_observability_ratio": float(od_report["direct_pair_observability_ratio"]),
                "step_count": len(result.steps),
                "selected_sequence": result.grasp_sequence,
                "first_selected_object": result.grasp_sequence[0] if result.grasp_sequence else None,
                "final_object_count": final_object_count,
                "clearance_rate": float((initial_object_count - final_object_count) / initial_object_count)
                if initial_object_count
                else None,
                "failure_reason": result.failure_reason,
            }
        )
    return rows


def aggregate_results(results: Sequence[dict[str, object]]) -> dict[str, object]:
    hidden_values = [float(row["hidden_object_count"]) for row in results]
    unobservable_values = [float(row["unobservable_pair_count"]) for row in results]
    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    first_selected_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        policy = str(row["policy"])
        by_policy[policy].append(dict(row))
        if row.get("first_selected_object") is not None:
            first_selected_counts[policy][str(row["first_selected_object"])] += 1

    return {
        "mean_hidden_object_count": _mean(hidden_values),
        "mean_unobservable_pair_count": _mean(unobservable_values),
        "policy_mean_clearance_rate": {
            policy: _mean([float(row["clearance_rate"]) for row in rows if row.get("clearance_rate") is not None])
            for policy, rows in sorted(by_policy.items())
        },
        "policy_first_selected_counts": {
            policy: dict(sorted(counter.items())) for policy, counter in sorted(first_selected_counts.items())
        },
    }


def write_results_csv(path: Path, results: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "group_id",
        "split",
        "scene",
        "frame",
        "role",
        "policy",
        "object_count",
        "contact_pair_count",
        "complete_object_count",
        "visible_object_count",
        "hidden_object_count",
        "direct_visible_boundary_pair_count",
        "unobservable_pair_count",
        "direct_pair_observability_ratio",
        "step_count",
        "first_selected_object",
        "selected_sequence",
        "final_object_count",
        "clearance_rate",
        "failure_reason",
    ]
    _write_csv(path, fieldnames, results)


def write_entries_csv(path: Path, entries: Sequence[dict[str, object]]) -> None:
    _write_csv(path, ["group_id", "group_name", "split", "scene", "frame", "role", "tags"], entries)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _resolve_scene_root(scene_root: str | Path | None, config: dict[str, object], *, base_dir: Path) -> Path:
    raw = Path(scene_root) if scene_root is not None else Path(str(config["scene_root"]))
    return raw if raw.is_absolute() else base_dir / raw


def _validate_policies(policies: Sequence[str]) -> tuple[str, ...]:
    valid = set(VALID_POLICIES)
    selected = tuple(str(policy) for policy in policies)
    invalid = [policy for policy in selected if policy not in valid]
    if invalid:
        raise ValueError(f"Invalid policies: {invalid}. Valid policies: {sorted(valid)}")
    return selected


def _entry_prefix(entry: dict[str, object]) -> dict[str, object]:
    copied = deepcopy(entry)
    copied["tags"] = list(copied.get("tags", []))
    return copied


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


if __name__ == "__main__":
    main()
