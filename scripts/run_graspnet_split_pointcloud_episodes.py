from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_external_graspnet_pointcloud_episode import (  # noqa: E402
    aggregate_frame_summaries,
    run_frame_episode_summary,
)
from scripts.run_graspnet_split_od_baselines import flatten_split_config  # noqa: E402
from stacked_grasping.gripper.external_graspnet_data import GraspNetRealSenseSource  # noqa: E402
from stacked_grasping.gripper.graspnet_binding import GraspNetPredictionSource  # noqa: E402
from stacked_grasping.gripper.pointcloud_feasibility import PointCloudCollisionConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GraspNet prediction binding and point-cloud feasibility episodes for a fixed split config."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "graspnet_candidate_split.json")
    parser.add_argument("--scene-root", type=Path, help="Override scene root. Defaults to config scene_root.")
    parser.add_argument("--prediction-root", type=Path, required=True, help="GraspNet dump root: scene_xxxx/realsense/0000.npy.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "graspnet_split_pointcloud_episode")
    parser.add_argument("--camera", default="realsense")
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
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_graspnet_split_pointcloud_episodes(
        config_path=args.config,
        scene_root=args.scene_root,
        prediction_root=args.prediction_root,
        out_dir=args.out_dir,
        camera=args.camera,
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
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    aggregate = summary["aggregate"]
    print("GraspNet split point-cloud episodes finished")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  frames: {summary['frame_count']}")
    print(f"  prediction_root: {summary['prediction_root']}")
    print(f"  binding_ratio: {aggregate['binding_ratio']}")
    print(f"  pointcloud_feasible_candidate_count: {aggregate['pointcloud_feasible_candidate_count']}")
    print(f"  success_rate: {aggregate['success_rate']}")
    print(f"  failure_reason_counts: {aggregate['failure_reason_counts']}")


def run_graspnet_split_pointcloud_episodes(
    *,
    config_path: str | Path,
    scene_root: str | Path | None = None,
    prediction_root: str | Path,
    out_dir: str | Path,
    camera: str = "realsense",
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
) -> dict[str, object]:
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    resolved_scene_root = _resolve_scene_root(scene_root, config, base_dir=config_file.parent.parent)
    resolved_prediction_root = Path(prediction_root)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    entries = flatten_split_config(config)
    frame_results: list[dict[str, object]] = []
    missing_predictions: list[dict[str, object]] = []
    for entry in entries:
        scene_name = str(entry["scene"])
        frame_id = str(entry["frame"])
        prediction_path = prediction_path_for_scene(resolved_prediction_root, scene_name, camera)
        prediction_file = prediction_path / f"{frame_id}.npy" if prediction_path.is_dir() else prediction_path
        if not prediction_file.exists():
            missing_predictions.append({**entry, "prediction_file": str(prediction_file)})
            continue
        realsense_path = realsense_path_for_scene(resolved_scene_root, scene_name, camera)
        with GraspNetRealSenseSource.open(realsense_path) as realsense_source, GraspNetPredictionSource.open(prediction_path) as prediction_source:
            result = run_frame_episode_summary(
                realsense_source=realsense_source,
                prediction_source=prediction_source,
                frame_id=frame_id,
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
        result.update(
            {
                "group_id": entry["group_id"],
                "group_name": entry["group_name"],
                "split": entry["split"],
                "scene": scene_name,
                "role": entry["role"],
                "tags": list(entry.get("tags", [])),
                "prediction_path": str(prediction_path),
            }
        )
        frame_results.append(result)

    summary = {
        "config": str(config_file),
        "scene_root": str(resolved_scene_root),
        "prediction_root": str(resolved_prediction_root),
        "output_dir": str(target),
        "frame_count": len(entries),
        "processed_frame_count": len(frame_results),
        "missing_prediction_count": len(missing_predictions),
        "missing_predictions": missing_predictions,
        "frame_results": frame_results,
        "aggregate": aggregate_frame_summaries(frame_results),
    }
    (target / "split_graspnet_pointcloud_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_split_frame_results_csv(target / "split_graspnet_pointcloud_frame_results.csv", frame_results)
    write_missing_predictions_csv(target / "missing_predictions.csv", missing_predictions)
    return summary


def prediction_path_for_scene(prediction_root: str | Path, scene: str, camera: str = "realsense") -> Path:
    root = Path(prediction_root)
    candidates = [
        root / scene / camera,
        root / scene,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return root / scene / camera


def realsense_path_for_scene(scene_root: str | Path, scene: str, camera: str = "realsense") -> Path:
    root = Path(scene_root)
    candidate = root / scene / camera
    return candidate if candidate.exists() else root / scene


def write_split_frame_results_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "group_id",
        "split",
        "scene",
        "frame",
        "role",
        "object_count",
        "contact_pair_count",
        "total_candidates",
        "bound_count",
        "binding_ratio",
        "pointcloud_feasible_candidate_count",
        "pointcloud_feasible_object_count",
        "selected_object",
        "grasp_success",
        "grasp_risk",
        "failure_reason",
        "gripper_feasible",
        "gripper_feasible_grasp_count",
        "selected_pose_generator",
        "selected_pose_score",
        "prediction_path",
    ]
    _write_csv(path, fieldnames, rows)


def write_missing_predictions_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    _write_csv(path, ["group_id", "split", "scene", "frame", "role", "prediction_file"], rows)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _resolve_scene_root(scene_root: str | Path | None, config: dict[str, object], *, base_dir: Path) -> Path:
    raw = Path(scene_root) if scene_root is not None else Path(str(config["scene_root"]))
    return raw if raw.is_absolute() else base_dir / raw


def _csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


if __name__ == "__main__":
    main()
