from __future__ import annotations

import argparse
import csv
import json
import sys
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

from scripts.run_external_graspnet_pointcloud_episode import sample_points  # noqa: E402
from scripts.run_graspnet_split_pointcloud_episodes import prediction_path_for_scene, realsense_path_for_scene  # noqa: E402
from stacked_grasping.gripper.external_graspnet_data import (  # noqa: E402
    GraspNetRealSenseSource,
    RealSenseFrame,
    depth_to_point_cloud,
)
from stacked_grasping.gripper.graspnet_binding import (  # noqa: E402
    GraspNetPredictionSource,
    bind_graspnet_records_to_frame_labels,
)
from stacked_grasping.gripper.pointcloud_feasibility import (  # noqa: E402
    PointCloudCollisionConfig,
    diagnose_graspnet_pointcloud_collisions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export point-cloud binding and gripper diagnostic features for saved dynamic top-K candidates."
    )
    parser.add_argument("--summary", type=Path, required=True, help="Path to split_dynamic_topk_summary.json.")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, help="Defaults to <summary parent>/pointcloud_features.")
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--factor-depth", type=int, default=1000)
    parser.add_argument("--binding-pixel-radius", type=int, default=3)
    parser.add_argument("--binding-depth-tolerance-m", type=float, default=0.12)
    parser.add_argument("--point-sample-limit", type=int, default=60000)
    parser.add_argument("--point-sample-seed", type=int, default=1234)
    parser.add_argument("--max-opening", type=float, default=0.085)
    parser.add_argument("--collision-threshold", type=float, default=0.01)
    parser.add_argument("--empty-threshold", type=float, default=0.01)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enriched = export_dynamic_topk_pointcloud_features(
        summary_path=args.summary,
        scene_root=args.scene_root,
        prediction_root=args.prediction_root,
        out_dir=args.out_dir or args.summary.parent / "pointcloud_features",
        camera=args.camera,
        factor_depth=args.factor_depth,
        binding_pixel_radius=args.binding_pixel_radius,
        binding_depth_tolerance_m=args.binding_depth_tolerance_m,
        point_sample_limit=args.point_sample_limit,
        point_sample_seed=args.point_sample_seed,
        pointcloud_config=PointCloudCollisionConfig(
            max_opening=args.max_opening,
            collision_threshold=args.collision_threshold,
            empty_threshold=args.empty_threshold,
        ),
    )
    if args.json:
        print(json.dumps(enriched, ensure_ascii=False, indent=2))
        return

    aggregate = enriched["pointcloud_feature_aggregate"]
    print("GraspNet dynamic top-K point-cloud features exported")
    print(f"  output_dir: {enriched['output_dir']}")
    print(f"  frames: {aggregate['frame_count']}")
    print(f"  candidates: {aggregate['candidate_count']}")
    print(f"  bound_ratio: {aggregate['bound_ratio']}")
    print(f"  pointcloud_feasible_ratio: {aggregate['pointcloud_feasible_ratio']}")


def export_dynamic_topk_pointcloud_features(
    *,
    summary_path: str | Path,
    scene_root: str | Path,
    prediction_root: str | Path,
    out_dir: str | Path,
    camera: str = "realsense",
    factor_depth: int = 1000,
    binding_pixel_radius: int = 3,
    binding_depth_tolerance_m: float | None = 0.12,
    point_sample_limit: int | None = 60000,
    point_sample_seed: int = 1234,
    pointcloud_config: PointCloudCollisionConfig | None = None,
) -> dict[str, object]:
    source = Path(summary_path)
    summary = json.loads(source.read_text(encoding="utf-8"))
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    cfg = pointcloud_config or PointCloudCollisionConfig()

    enriched = deepcopy(summary)
    enriched["created_at"] = datetime.now().isoformat(timespec="seconds")
    enriched["source_summary_path"] = str(source)
    enriched["scene_root"] = str(scene_root)
    enriched["prediction_root"] = str(prediction_root)
    enriched["output_dir"] = str(target)

    candidate_rows: list[dict[str, object]] = []
    for frame_result in enriched.get("frame_results", []):
        features_by_rank = compute_frame_pointcloud_features(
            frame_result,
            scene_root=Path(scene_root),
            prediction_root=Path(prediction_root),
            camera=camera,
            factor_depth=factor_depth,
            binding_pixel_radius=binding_pixel_radius,
            binding_depth_tolerance_m=binding_depth_tolerance_m,
            point_sample_limit=point_sample_limit,
            point_sample_seed=point_sample_seed,
            pointcloud_config=cfg,
        )
        for candidate in frame_result.get("candidate_results", []):
            rank = int(candidate.get("candidate_rank", -1))
            candidate.update(features_by_rank.get(rank, _missing_feature(rank)))
            candidate_rows.append(_candidate_output_row(frame_result, candidate))

    enriched["pointcloud_feature_aggregate"] = aggregate_candidate_features(candidate_rows)
    (target / "dynamic_topk_pointcloud_features_summary.json").write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_candidate_features_csv(target / "dynamic_topk_pointcloud_features_candidates.csv", candidate_rows)
    return enriched


def compute_frame_pointcloud_features(
    frame_result: dict[str, object],
    *,
    scene_root: Path,
    prediction_root: Path,
    camera: str,
    factor_depth: int,
    binding_pixel_radius: int,
    binding_depth_tolerance_m: float | None,
    point_sample_limit: int | None,
    point_sample_seed: int,
    pointcloud_config: PointCloudCollisionConfig,
) -> dict[int, dict[str, object]]:
    scene = str(frame_result["scene"])
    frame_id = str(frame_result["frame"])
    ranks = sorted(
        {
            int(candidate.get("candidate_rank", -1))
            for candidate in frame_result.get("candidate_results", [])
            if candidate.get("candidate_rank") is not None
        }
    )
    if not ranks:
        return {}

    realsense_path = realsense_path_for_scene(scene_root, scene, camera)
    prediction_path = prediction_path_for_scene(prediction_root, scene, camera)
    with GraspNetRealSenseSource.open(realsense_path) as realsense_source, GraspNetPredictionSource.open(prediction_path) as prediction_source:
        loaded = realsense_source.load_frame(frame_id)
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
        annotations = realsense_source.load_annotation_objects(frame.frame)
        records = sorted(prediction_source.load_records(frame.frame), key=lambda item: float(item.get("score", 0.0)), reverse=True)
        bindings = bind_graspnet_records_to_frame_labels(
            records,
            frame,
            annotations,
            pixel_radius=binding_pixel_radius,
            depth_tolerance_m=binding_depth_tolerance_m,
        )
        points, _ = depth_to_point_cloud(frame.depth_raw, frame.intrinsic_matrix, factor_depth=frame.factor_depth)
        points = sample_points(points, limit=point_sample_limit, seed=point_sample_seed)

    selected_records = [records[rank] for rank in ranks if 0 <= rank < len(records)]
    selected_ranks = [rank for rank in ranks if 0 <= rank < len(records)]
    diagnostics = diagnose_graspnet_pointcloud_collisions(points, selected_records, config=pointcloud_config)
    features: dict[int, dict[str, object]] = {}
    for rank, diagnostic in zip(selected_ranks, diagnostics):
        binding = bindings[rank]
        pointcloud_feasible = bool(binding.status == "bound" and diagnostic.feasible)
        features[rank] = {
            "binding_status": binding.status,
            "binding_pixel": list(binding.pixel) if binding.pixel is not None else None,
            "binding_label_id": binding.label_id,
            "binding_object_id": binding.object_id,
            "binding_object_name": binding.object_name,
            "binding_depth_error_m": _rounded_or_none(binding.depth_error_m),
            "pointcloud_feasible": pointcloud_feasible,
            "pointcloud_failure_reason": None
            if pointcloud_feasible
            else (diagnostic.reason if binding.status == "bound" else f"binding-{binding.status}"),
            "pointcloud_collision_iou": round(float(diagnostic.collision_iou), 6),
            "pointcloud_empty_ratio": round(float(diagnostic.empty_ratio), 6),
            "pointcloud_collision": bool(diagnostic.collision),
            "pointcloud_empty": bool(diagnostic.empty),
            "pointcloud_opening_too_small": bool(diagnostic.opening_too_small),
        }
    return features


def aggregate_candidate_features(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    candidate_count = len(rows)
    bound_count = sum(1 for row in rows if row.get("binding_status") == "bound")
    feasible_count = sum(1 for row in rows if row.get("pointcloud_feasible"))
    return {
        "frame_count": len({(row.get("scene"), row.get("frame")) for row in rows}),
        "candidate_count": candidate_count,
        "bound_count": bound_count,
        "bound_ratio": _rate(bound_count, candidate_count),
        "pointcloud_feasible_count": feasible_count,
        "pointcloud_feasible_ratio": _rate(feasible_count, candidate_count),
    }


def write_candidate_features_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "group_id",
        "split",
        "scene",
        "frame",
        "candidate_rank",
        "selected_grasp_score",
        "target_object_id",
        "target_object_name",
        "lift_success",
        "failure_reason",
        "binding_status",
        "binding_label_id",
        "binding_object_id",
        "binding_depth_error_m",
        "pointcloud_feasible",
        "pointcloud_failure_reason",
        "pointcloud_collision_iou",
        "pointcloud_empty_ratio",
        "pointcloud_collision",
        "pointcloud_empty",
        "pointcloud_opening_too_small",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _candidate_output_row(frame_result: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    return {
        "group_id": frame_result.get("group_id"),
        "split": frame_result.get("split"),
        "scene": frame_result.get("scene"),
        "frame": frame_result.get("frame"),
        **dict(candidate),
    }


def _missing_feature(rank: int) -> dict[str, object]:
    return {
        "binding_status": "missing-record",
        "pointcloud_feasible": False,
        "pointcloud_failure_reason": f"missing-record-rank-{rank}",
    }


def _rate(count: int, total: int) -> float | None:
    return round(float(count) / float(total), 6) if total else None


def _rounded_or_none(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


if __name__ == "__main__":
    main()
