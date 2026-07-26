from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.gripper.external_graspnet_data import (  # noqa: E402
    GraspNetRealSenseSource,
    RealSenseFrame,
    depth_to_point_cloud,
    normalize_frame_id,
)
from stacked_grasping.gripper.external_graspnet_scene import build_external_graspnet_episode_inputs  # noqa: E402
from stacked_grasping.gripper.graspnet_binding import (  # noqa: E402
    BINDING_MODES,
    GraspNetPredictionSource,
    bind_graspnet_records,
)
from stacked_grasping.gripper.pointcloud_feasibility import (  # noqa: E402
    PointCloudCollisionConfig,
    assess_scene_bound_graspnet_pointcloud_feasibility,
)
from stacked_grasping.planning.episode import run_policy_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run point-cloud GraspNet candidate smoke episodes on external frames.")
    parser.add_argument("--realsense", type=Path, required=True, help="GraspNet-style realsense directory or zip.")
    parser.add_argument("--prediction", type=Path, required=True, help="GraspNet prediction .npy directory, file, or zip.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for JSON/CSV summaries.")
    parser.add_argument("--frames", nargs="*", default=(), help="Frame ids to sample, e.g. 14 75 156.")
    parser.add_argument("--max-frames", type=int, default=5, help="Evenly sample this many common frames when --frames is omitted.")
    parser.add_argument("--factor-depth", type=int, default=1000)
    parser.add_argument("--binding-pixel-radius", type=int, default=3)
    parser.add_argument("--binding-depth-tolerance-m", type=float, default=0.12)
    parser.add_argument("--binding-mode", choices=BINDING_MODES, default="pixel")
    parser.add_argument("--binding-3d-max-distance-m", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=1, help="Steps per frame episode; <=0 clears until failure or empty.")
    parser.add_argument("--policy", default="adaptive-score-v2-graspnet")
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
    parser.add_argument(
        "--clamp-width",
        action="store_true",
        help="Clamp candidate opening to --max-opening and re-check instead of rejecting as opening-too-small.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _resolve_output_dir(args.out_dir)
    summary = run_external_graspnet_pointcloud_episodes(
        realsense_path=args.realsense,
        prediction_path=args.prediction,
        out_dir=out_dir,
        requested_frames=tuple(args.frames),
        max_frames=args.max_frames,
        factor_depth=args.factor_depth,
        binding_pixel_radius=args.binding_pixel_radius,
        binding_depth_tolerance_m=args.binding_depth_tolerance_m,
        binding_mode=args.binding_mode,
        binding_3d_max_distance_m=args.binding_3d_max_distance_m,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else None,
        policy=args.policy,
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
            clamp_width_to_max_opening=args.clamp_width,
        ),
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    aggregate = summary["aggregate"]
    print("External GraspNet point-cloud episode smoke finished")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  frames: {summary['selected_frames']}")
    print(f"  binding_ratio: {aggregate['binding_ratio']}")
    print(f"  pointcloud_feasible_candidates: {aggregate['pointcloud_feasible_candidate_count']}")
    print(f"  success_rate: {aggregate['success_rate']}")
    print(f"  failure_reason_counts: {aggregate['failure_reason_counts']}")


def run_external_graspnet_pointcloud_episodes(
    *,
    realsense_path: Path,
    prediction_path: Path,
    out_dir: Path,
    requested_frames: Sequence[int | str] = (),
    max_frames: int | None = 5,
    factor_depth: int = 1000,
    binding_pixel_radius: int = 3,
    binding_depth_tolerance_m: float | None = 0.12,
    binding_mode: str = "pixel",
    binding_3d_max_distance_m: float = 0.05,
    max_steps: int | None = 1,
    policy: str = "adaptive-score-v2-graspnet",
    min_points_per_object: int = 20,
    min_half_extent: float = 0.01,
    object_padding: float = 0.002,
    min_boundary_pixels: int = 50,
    point_sample_limit: int | None = 60000,
    point_sample_seed: int = 1234,
    risk_threshold: float = 0.45,
    pointcloud_config: PointCloudCollisionConfig | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_results: list[dict[str, object]] = []
    with GraspNetRealSenseSource.open(realsense_path) as realsense_source, GraspNetPredictionSource.open(prediction_path) as prediction_source:
        selected_frames = select_matching_frame_ids(
            realsense_source.list_frames(),
            prediction_source.list_frames(),
            requested=requested_frames,
            max_frames=max_frames,
        )
        for frame_id in selected_frames:
            frame_results.append(
                run_frame_episode_summary(
                    realsense_source=realsense_source,
                    prediction_source=prediction_source,
                    frame_id=frame_id,
                    factor_depth=factor_depth,
                    binding_pixel_radius=binding_pixel_radius,
                    binding_depth_tolerance_m=binding_depth_tolerance_m,
                    binding_mode=binding_mode,
                    binding_3d_max_distance_m=binding_3d_max_distance_m,
                    max_steps=max_steps,
                    policy=policy,
                    min_points_per_object=min_points_per_object,
                    min_half_extent=min_half_extent,
                    object_padding=object_padding,
                    min_boundary_pixels=min_boundary_pixels,
                    point_sample_limit=point_sample_limit,
                    point_sample_seed=point_sample_seed,
                    risk_threshold=risk_threshold,
                    pointcloud_config=pointcloud_config,
                )
            )

    summary = {
        "output_dir": str(out_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "realsense_path": str(realsense_path),
        "prediction_path": str(prediction_path),
        "selected_frames": selected_frames,
        "frame_results": frame_results,
        "aggregate": aggregate_frame_summaries(frame_results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_frame_results_csv(out_dir / "frame_results.csv", frame_results)
    return summary


def run_frame_episode_summary(
    *,
    realsense_source: GraspNetRealSenseSource,
    prediction_source: GraspNetPredictionSource,
    frame_id: int | str,
    factor_depth: int = 1000,
    binding_pixel_radius: int = 3,
    binding_depth_tolerance_m: float | None = 0.12,
    binding_mode: str = "pixel",
    binding_3d_max_distance_m: float = 0.05,
    max_steps: int | None = 1,
    policy: str = "adaptive-score-v2-graspnet",
    min_points_per_object: int = 20,
    min_half_extent: float = 0.01,
    object_padding: float = 0.002,
    min_boundary_pixels: int = 50,
    point_sample_limit: int | None = 60000,
    point_sample_seed: int = 1234,
    risk_threshold: float = 0.45,
    pointcloud_config: PointCloudCollisionConfig | None = None,
) -> dict[str, object]:
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
    bindings = bind_graspnet_records(
        records,
        frame,
        annotations,
        mode=binding_mode,
        pixel_radius=binding_pixel_radius,
        depth_tolerance_m=binding_depth_tolerance_m,
        max_distance_m=binding_3d_max_distance_m,
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
    points, valid_mask = depth_to_point_cloud(frame.depth_raw, frame.intrinsic_matrix, factor_depth=frame.factor_depth)
    point_labels = np.asarray(frame.label).reshape(-1)[np.asarray(valid_mask).reshape(-1)] if frame.label is not None else np.zeros(len(points), dtype=int)
    points, point_labels = sample_points_with_labels(points, point_labels, limit=point_sample_limit, seed=point_sample_seed)
    label_by_object_name = {annotation.name: int(annotation.label_id) for annotation in annotations}
    initial_labels = {
        label_by_object_name[obj.name]
        for obj in episode_inputs.scene.read_objects()
        if obj.name in label_by_object_name
    }

    def gripper_provider(objects):
        # Grasped objects vanish from the collision cloud: points carrying the label
        # of a removed object are filtered out, while background/table points and
        # labels never present in the episode stay untouched.
        active_labels = {label_by_object_name[obj.name] for obj in objects if obj.name in label_by_object_name}
        removed_labels = sorted(initial_labels - active_labels)
        step_points = points if not removed_labels else points[~np.isin(point_labels, removed_labels)]
        return assess_scene_bound_graspnet_pointcloud_feasibility(
            objects,
            step_points,
            bindings,
            config=pointcloud_config,
        )

    gripper_feasibilities = list(gripper_provider(episode_inputs.scene.read_objects()))
    result = run_policy_episode(
        episode_inputs.scene,
        policy=policy,
        max_steps=max_steps,
        post_grasp_settle_steps=0,
        failure_mode="risk-threshold",
        risk_threshold=risk_threshold,
        gripper_feasibility_provider=gripper_provider,
    )
    return compact_frame_summary(
        frame=frame.frame,
        total_candidates=len(records),
        bindings=bindings,
        object_count=len(gripper_feasibilities),
        contact_pair_count=len(episode_inputs.scene.read_object_contact_pairs()),
        gripper_feasibilities=gripper_feasibilities,
        result=result,
    )


def compact_frame_summary(
    *,
    frame: str,
    total_candidates: int,
    bindings: Sequence[object],
    object_count: int,
    contact_pair_count: int,
    gripper_feasibilities: Sequence[object],
    result: object,
) -> dict[str, object]:
    bound_count = sum(1 for binding in bindings if getattr(binding, "status") == "bound")
    feasible_candidate_count = sum(int(item.feasible_grasp_count) for item in gripper_feasibilities)
    feasible_object_count = sum(1 for item in gripper_feasibilities if int(item.feasible_grasp_count) > 0)
    candidate_reason_counts = Counter(
        candidate.reason or "feasible"
        for item in gripper_feasibilities
        for candidate in item.candidates
    )
    step = result.steps[0] if result.steps else None
    selected_pose = step.selected_grasp_pose if step and step.selected_grasp_pose else None
    num_successes = sum(1 for item in result.steps if item.grasp_success)
    return {
        "frame": frame,
        "object_count": object_count,
        "contact_pair_count": contact_pair_count,
        "total_candidates": total_candidates,
        "bound_count": bound_count,
        "binding_ratio": float(bound_count / total_candidates) if total_candidates else None,
        "pointcloud_feasible_candidate_count": feasible_candidate_count,
        "pointcloud_feasible_object_count": feasible_object_count,
        "pointcloud_reason_counts": dict(sorted(candidate_reason_counts.items())),
        "selected_object": step.selected_object if step else None,
        "grasp_success": bool(step.grasp_success) if step else False,
        "grasp_risk": round(float(step.grasp_risk), 6) if step else None,
        "failure_reason": step.failure_reason if step else getattr(result, "failure_reason", None),
        "gripper_feasible": step.gripper_feasible if step else None,
        "gripper_feasible_grasp_count": step.gripper_feasible_grasp_count if step else None,
        "selected_pose_generator": selected_pose.generator if selected_pose else None,
        "selected_pose_score": round(float(selected_pose.score), 6) if selected_pose else None,
        "num_steps": len(result.steps),
        "num_successes": num_successes,
        "cleared_objects": num_successes,
        "clearance_rate": float(num_successes / object_count) if object_count else None,
        "episode_failure_reason": getattr(result, "failure_reason", None),
    }


def aggregate_frame_summaries(frame_results: Sequence[dict[str, object]]) -> dict[str, object]:
    total_candidates = sum(int(item["total_candidates"]) for item in frame_results)
    bound_count = sum(int(item["bound_count"]) for item in frame_results)
    feasible_candidates = sum(int(item["pointcloud_feasible_candidate_count"]) for item in frame_results)
    success_count = sum(1 for item in frame_results if bool(item["grasp_success"]))
    selected_counter = Counter(str(item["selected_object"]) for item in frame_results if item.get("selected_object") is not None)
    failure_counter = Counter(str(item["failure_reason"]) for item in frame_results if item.get("failure_reason") is not None)
    return {
        "frame_count": len(frame_results),
        "total_candidates": total_candidates,
        "bound_count": bound_count,
        "binding_ratio": float(bound_count / total_candidates) if total_candidates else None,
        "pointcloud_feasible_candidate_count": feasible_candidates,
        "pointcloud_feasible_frame_count": sum(
            1 for item in frame_results if int(item["pointcloud_feasible_candidate_count"]) > 0
        ),
        "mean_pointcloud_feasible_candidate_count": float(feasible_candidates / len(frame_results)) if frame_results else None,
        "successful_frame_count": success_count,
        "success_rate": float(success_count / len(frame_results)) if frame_results else None,
        "failure_reason_counts": dict(sorted(failure_counter.items())),
        "selected_object_counts": dict(sorted(selected_counter.items())),
        "mean_clearance_rate": _mean_optional([item.get("clearance_rate") for item in frame_results]),
        "full_clear_frame_count": sum(1 for item in frame_results if item.get("clearance_rate") == 1.0),
        "mean_steps": _mean_optional([item.get("num_steps") for item in frame_results]),
    }


def _mean_optional(values: Sequence[object]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return float(sum(numbers) / len(numbers)) if numbers else None


def select_matching_frame_ids(
    realsense_frames: Sequence[str],
    prediction_frames: Sequence[str],
    *,
    requested: Sequence[int | str] = (),
    max_frames: int | None = 5,
) -> list[str]:
    realsense_set = {normalize_frame_id(frame) for frame in realsense_frames}
    prediction_set = {normalize_frame_id(frame) for frame in prediction_frames}
    common = sorted(realsense_set & prediction_set)
    if requested:
        requested_ids = [normalize_frame_id(frame) for frame in requested]
        common_set = set(common)
        return [frame for frame in requested_ids if frame in common_set]
    if max_frames is None or max_frames <= 0 or max_frames >= len(common):
        return common
    indices = np.linspace(0, len(common) - 1, num=max_frames, dtype=int)
    return [common[int(index)] for index in indices]


def sample_points(points: np.ndarray, *, limit: int | None, seed: int) -> np.ndarray:
    arr = np.asarray(points, dtype=float).reshape(-1, 3)
    if limit is None or limit <= 0 or arr.shape[0] <= limit:
        return arr
    rng = np.random.default_rng(seed)
    return arr[rng.choice(arr.shape[0], int(limit), replace=False)]


def sample_points_with_labels(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    limit: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Jointly subsample a point cloud and its per-point labels with shared indices."""
    arr = np.asarray(points, dtype=float).reshape(-1, 3)
    label_arr = np.asarray(labels).reshape(-1)
    if arr.shape[0] != label_arr.shape[0]:
        raise ValueError("points and labels must have matching lengths.")
    if limit is None or limit <= 0 or arr.shape[0] <= limit:
        return arr, label_arr
    rng = np.random.default_rng(seed)
    indices = rng.choice(arr.shape[0], int(limit), replace=False)
    return arr[indices], label_arr[indices]


def write_frame_results_csv(path: Path, frame_results: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "frame",
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
        "num_steps",
        "num_successes",
        "cleared_objects",
        "clearance_rate",
        "episode_failure_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(frame_results)


def _resolve_output_dir(out_dir: Path | None) -> Path:
    if out_dir is None:
        return PROJECT_ROOT / "results" / "external_graspnet_pointcloud_episode" / (
            "run_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    return out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir


if __name__ == "__main__":
    main()
