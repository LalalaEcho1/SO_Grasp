from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_external_graspnet_pointcloud_episode import (  # noqa: E402
    sample_points,
    select_matching_frame_ids,
)
from stacked_grasping.gripper.external_graspnet_data import (  # noqa: E402
    GraspNetRealSenseSource,
    RealSenseFrame,
    depth_to_point_cloud,
)
from stacked_grasping.gripper.external_graspnet_scene import build_external_graspnet_episode_inputs  # noqa: E402
from stacked_grasping.gripper.graspnet_binding import (  # noqa: E402
    GraspNetPredictionSource,
    bind_graspnet_records_to_frame_labels,
)
from stacked_grasping.gripper.pointcloud_feasibility import (  # noqa: E402
    PointCloudCollisionConfig,
    assess_scene_bound_graspnet_pointcloud_feasibility,
)
from stacked_grasping.planning.episode import run_policy_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep point-cloud feasibility parameters (width clamp on/off x collision/empty thresholds) "
            "over external GraspNet frames and report the candidate-supply funnel per cell."
        )
    )
    parser.add_argument("--realsense", type=Path, required=True, help="GraspNet-style realsense directory or zip.")
    parser.add_argument("--prediction", type=Path, required=True, help="GraspNet prediction .npy directory, file, or zip.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for JSON/CSV summaries.")
    parser.add_argument("--frames", nargs="*", default=(), help="Frame ids to sample, e.g. 14 75 156.")
    parser.add_argument("--max-frames", type=int, default=0, help="Evenly sample this many frames; <=0 means all common frames.")
    parser.add_argument("--clamp-modes", nargs="+", default=("off", "on"), choices=("off", "on"))
    parser.add_argument("--collision-thresholds", nargs="+", type=float, default=(0.005, 0.01, 0.02, 0.03))
    parser.add_argument("--empty-thresholds", nargs="+", type=float, default=(0.01,))
    parser.add_argument("--policy", default="adaptive-score-v2-graspnet")
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
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args()


def build_grid_cells(
    *,
    clamp_modes: Sequence[str] = ("off", "on"),
    collision_thresholds: Sequence[float] = (0.005, 0.01, 0.02, 0.03),
    empty_thresholds: Sequence[float] = (0.01,),
    max_opening: float = 0.085,
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for clamp_mode in clamp_modes:
        clamp = str(clamp_mode).lower() in {"on", "true", "1", "yes"}
        for collision_threshold in collision_thresholds:
            for empty_threshold in empty_thresholds:
                cells.append(
                    {
                        "cell_id": f"clamp-{'on' if clamp else 'off'}_ct-{float(collision_threshold):g}_et-{float(empty_threshold):g}",
                        "clamp_width": clamp,
                        "collision_threshold": float(collision_threshold),
                        "empty_threshold": float(empty_threshold),
                        "config": PointCloudCollisionConfig(
                            max_opening=float(max_opening),
                            collision_threshold=float(collision_threshold),
                            empty_threshold=float(empty_threshold),
                            clamp_width_to_max_opening=clamp,
                        ),
                    }
                )
    return cells


def evaluate_frame_cell(
    *,
    frame: RealSenseFrame,
    annotations: Sequence[object],
    records: Sequence[dict[str, object]],
    bindings: Sequence[object],
    points,
    cell_config: PointCloudCollisionConfig,
    risk_threshold: float = 0.45,
    policy: str = "adaptive-score-v2-graspnet",
    min_points_per_object: int = 20,
    min_half_extent: float = 0.01,
    object_padding: float = 0.002,
    min_boundary_pixels: int = 50,
) -> dict[str, object]:
    # The episode mutates its scene on success, so inputs are rebuilt per cell.
    episode_inputs = build_external_graspnet_episode_inputs(
        frame,
        annotations,
        bindings,
        min_points_per_object=min_points_per_object,
        min_half_extent=min_half_extent,
        padding=object_padding,
        min_boundary_pixels=min_boundary_pixels,
    )
    gripper_feasibilities = list(
        assess_scene_bound_graspnet_pointcloud_feasibility(
            episode_inputs.scene.read_objects(),
            points,
            bindings,
            config=cell_config,
        )
    )
    result = run_policy_episode(
        episode_inputs.scene,
        policy=policy,
        max_steps=1,
        post_grasp_settle_steps=0,
        failure_mode="risk-threshold",
        risk_threshold=risk_threshold,
        gripper_feasibility_provider=lambda objects: gripper_feasibilities,
    )
    step = result.steps[0] if result.steps else None

    feasible_candidate_count = sum(int(item.feasible_grasp_count) for item in gripper_feasibilities)
    feasible_object_count = sum(1 for item in gripper_feasibilities if int(item.feasible_grasp_count) > 0)
    clamp_recovered = sum(
        1
        for item in gripper_feasibilities
        for candidate in item.candidates
        if candidate.feasible and float(candidate.required_opening) > float(cell_config.max_opening)
    )
    reason_counts = Counter(
        candidate.reason or "feasible"
        for item in gripper_feasibilities
        for candidate in item.candidates
    )
    return {
        "frame": frame.frame,
        "total_candidates": len(records),
        "bound_count": sum(1 for binding in bindings if getattr(binding, "status", None) == "bound"),
        "pointcloud_feasible_candidate_count": feasible_candidate_count,
        "pointcloud_feasible_object_count": feasible_object_count,
        "clamp_recovered_candidate_count": clamp_recovered,
        "pointcloud_reason_counts": dict(sorted(reason_counts.items())),
        "selected_object": step.selected_object if step else None,
        "grasp_success": bool(step.grasp_success) if step else False,
        "grasp_risk": round(float(step.grasp_risk), 6) if step else None,
        "failure_reason": step.failure_reason if step else getattr(result, "failure_reason", None),
    }


def aggregate_cell_rows(frame_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    frame_count = len(frame_rows)
    feasible_candidates = sum(int(row["pointcloud_feasible_candidate_count"]) for row in frame_rows)
    success_count = sum(1 for row in frame_rows if bool(row["grasp_success"]))
    failure_counter = Counter(
        str(row["failure_reason"]) for row in frame_rows if row.get("failure_reason") is not None
    )
    reason_totals: Counter[str] = Counter()
    for row in frame_rows:
        reason_totals.update({str(k): int(v) for k, v in dict(row.get("pointcloud_reason_counts", {})).items()})
    return {
        "frame_count": frame_count,
        "pointcloud_feasible_candidate_count": feasible_candidates,
        "mean_pointcloud_feasible_candidate_count": float(feasible_candidates / frame_count) if frame_count else None,
        "pointcloud_feasible_frame_count": sum(
            1 for row in frame_rows if int(row["pointcloud_feasible_candidate_count"]) > 0
        ),
        "clamp_recovered_candidate_count": sum(int(row["clamp_recovered_candidate_count"]) for row in frame_rows),
        "successful_frame_count": success_count,
        "success_rate": float(success_count / frame_count) if frame_count else None,
        "failure_reason_counts": dict(sorted(failure_counter.items())),
        "pointcloud_reason_totals": dict(sorted(reason_totals.items())),
    }


def run_feasibility_grid(
    *,
    realsense_path: Path,
    prediction_path: Path,
    out_dir: Path,
    requested_frames: Sequence[int | str] = (),
    max_frames: int | None = None,
    clamp_modes: Sequence[str] = ("off", "on"),
    collision_thresholds: Sequence[float] = (0.005, 0.01, 0.02, 0.03),
    empty_thresholds: Sequence[float] = (0.01,),
    policy: str = "adaptive-score-v2-graspnet",
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
    max_opening: float = 0.085,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = build_grid_cells(
        clamp_modes=clamp_modes,
        collision_thresholds=collision_thresholds,
        empty_thresholds=empty_thresholds,
        max_opening=max_opening,
    )
    rows_by_cell: dict[str, list[dict[str, object]]] = {str(cell["cell_id"]): [] for cell in cells}

    with GraspNetRealSenseSource.open(realsense_path) as realsense_source, GraspNetPredictionSource.open(prediction_path) as prediction_source:
        selected_frames = select_matching_frame_ids(
            realsense_source.list_frames(),
            prediction_source.list_frames(),
            requested=requested_frames,
            max_frames=max_frames,
        )
        for frame_id in selected_frames:
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
            points, _ = depth_to_point_cloud(frame.depth_raw, frame.intrinsic_matrix, factor_depth=frame.factor_depth)
            points = sample_points(points, limit=point_sample_limit, seed=point_sample_seed)
            for cell in cells:
                rows_by_cell[str(cell["cell_id"])].append(
                    evaluate_frame_cell(
                        frame=frame,
                        annotations=annotations,
                        records=records,
                        bindings=bindings,
                        points=points,
                        cell_config=cell["config"],  # type: ignore[arg-type]
                        risk_threshold=risk_threshold,
                        policy=policy,
                        min_points_per_object=min_points_per_object,
                        min_half_extent=min_half_extent,
                        object_padding=object_padding,
                        min_boundary_pixels=min_boundary_pixels,
                    )
                )

    cell_summaries = []
    for cell in cells:
        cell_id = str(cell["cell_id"])
        cell_summaries.append(
            {
                "cell_id": cell_id,
                "clamp_width": cell["clamp_width"],
                "collision_threshold": cell["collision_threshold"],
                "empty_threshold": cell["empty_threshold"],
                "aggregate": aggregate_cell_rows(rows_by_cell[cell_id]),
                "frame_results": rows_by_cell[cell_id],
            }
        )

    summary = {
        "output_dir": str(out_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "realsense_path": str(realsense_path),
        "prediction_path": str(prediction_path),
        "selected_frames": selected_frames,
        "policy": policy,
        "risk_threshold": risk_threshold,
        "max_opening": max_opening,
        "cell_count": len(cell_summaries),
        "cells": cell_summaries,
    }
    (out_dir / "grid_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_grid_summary_csv(out_dir / "grid_summary.csv", cell_summaries)
    return summary


def write_grid_summary_csv(path: Path, cell_summaries: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "cell_id",
        "clamp_width",
        "collision_threshold",
        "empty_threshold",
        "frame_count",
        "pointcloud_feasible_candidate_count",
        "mean_pointcloud_feasible_candidate_count",
        "pointcloud_feasible_frame_count",
        "clamp_recovered_candidate_count",
        "successful_frame_count",
        "success_rate",
        "failure_reason_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for cell in cell_summaries:
            aggregate = dict(cell["aggregate"])  # type: ignore[arg-type]
            writer.writerow(
                {
                    "cell_id": cell["cell_id"],
                    "clamp_width": cell["clamp_width"],
                    "collision_threshold": cell["collision_threshold"],
                    "empty_threshold": cell["empty_threshold"],
                    "frame_count": aggregate.get("frame_count"),
                    "pointcloud_feasible_candidate_count": aggregate.get("pointcloud_feasible_candidate_count"),
                    "mean_pointcloud_feasible_candidate_count": aggregate.get("mean_pointcloud_feasible_candidate_count"),
                    "pointcloud_feasible_frame_count": aggregate.get("pointcloud_feasible_frame_count"),
                    "clamp_recovered_candidate_count": aggregate.get("clamp_recovered_candidate_count"),
                    "successful_frame_count": aggregate.get("successful_frame_count"),
                    "success_rate": aggregate.get("success_rate"),
                    "failure_reason_counts": json.dumps(aggregate.get("failure_reason_counts", {}), ensure_ascii=False),
                }
            )


def _resolve_output_dir(out_dir: Path | None) -> Path:
    if out_dir is None:
        return PROJECT_ROOT / "results" / "pointcloud_feasibility_grid" / (
            "sweep_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    return out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir


def main() -> None:
    args = parse_args()
    summary = run_feasibility_grid(
        realsense_path=args.realsense,
        prediction_path=args.prediction,
        out_dir=_resolve_output_dir(args.out_dir),
        requested_frames=tuple(args.frames),
        max_frames=args.max_frames if args.max_frames and args.max_frames > 0 else None,
        clamp_modes=tuple(args.clamp_modes),
        collision_thresholds=tuple(args.collision_thresholds),
        empty_thresholds=tuple(args.empty_thresholds),
        policy=args.policy,
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
        max_opening=args.max_opening,
    )
    if args.json:
        compact = dict(summary)
        compact["cells"] = [
            {key: value for key, value in cell.items() if key != "frame_results"} for cell in summary["cells"]
        ]
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        return

    print("Point-cloud feasibility grid sweep finished")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  frames: {len(summary['selected_frames'])}")
    for cell in summary["cells"]:
        aggregate = cell["aggregate"]
        print(
            f"   {cell['cell_id']}: feasible/frame={aggregate['mean_pointcloud_feasible_candidate_count']:.2f} "
            f"success_rate={aggregate['success_rate']} recovered={aggregate['clamp_recovered_candidate_count']}"
        )


if __name__ == "__main__":
    main()
