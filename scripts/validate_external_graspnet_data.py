from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.gripper.external_graspnet_data import (  # noqa: E402
    GraspNetRealSenseSource,
    RealSenseFrame,
    assess_single_view_od_sufficiency,
    depth_to_point_cloud,
    normalize_frame_id,
    summarize_graspnet_prediction_package,
    summarize_realsense_frame,
)
from stacked_grasping.gripper.graspnet_binding import (  # noqa: E402
    GraspNetPredictionSource,
    bind_graspnet_records_to_frame_labels,
    bound_candidates_to_grasp_poses_by_object,
    summarize_bound_graspnet_candidates,
)
from stacked_grasping.gripper.graspnet_input import write_graspnet_input_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate external GraspNet RealSense data and prediction packages.")
    parser.add_argument("--realsense", type=Path, help="GraspNet-style realsense directory or zip.")
    parser.add_argument("--prediction", type=Path, help="GraspNet prediction .npy directory, file, or zip.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for summaries, previews, and demo bundle.")
    parser.add_argument("--frames", nargs="*", default=(), help="Frame ids to sample, e.g. 0 64 128.")
    parser.add_argument("--max-frames", type=int, default=5, help="Evenly sample this many frames when --frames is omitted.")
    parser.add_argument("--factor-depth", type=int, default=1000)
    parser.add_argument("--min-boundary-pixels", type=int, default=50)
    parser.add_argument("--binding-pixel-radius", type=int, default=3)
    parser.add_argument("--binding-depth-tolerance-m", type=float, default=0.12)
    parser.add_argument("--no-previews", action="store_true", help="Skip PNG preview generation.")
    parser.add_argument("--no-demo-bundle", action="store_true", help="Skip GraspNet demo input export for the first frame.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args()
    if args.realsense is None and args.prediction is None:
        parser.error("At least one of --realsense or --prediction is required.")
    return args


def main() -> None:
    args = parse_args()
    out_dir = _resolve_output_dir(args.out_dir)
    summary = validate_external_graspnet_data(
        realsense_path=args.realsense,
        prediction_path=args.prediction,
        out_dir=out_dir,
        requested_frames=tuple(args.frames),
        max_frames=args.max_frames,
        factor_depth=args.factor_depth,
        min_boundary_pixels=args.min_boundary_pixels,
        binding_pixel_radius=args.binding_pixel_radius,
        binding_depth_tolerance_m=args.binding_depth_tolerance_m,
        save_previews=not args.no_previews,
        export_demo_bundle=not args.no_demo_bundle,
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("External GraspNet data validation finished")
    print(f"  output_dir: {summary['output_dir']}")
    if "realsense" in summary:
        real = summary["realsense"]
        print(f"  realsense_frames: {real['selected_frames']}")
        print(f"  frame_summaries: {len(real['frame_results'])}")
        if real.get("demo_bundle"):
            print(f"  demo_bundle: {real['demo_bundle']}")
    if "prediction" in summary:
        pred = summary["prediction"]
        print(f"  prediction_files: {pred['file_count']}")
        print(f"  total_grasps: {pred['total_grasps']}")
    if "binding" in summary:
        binding = summary["binding"]["aggregate"]
        print(f"  binding_frames: {summary['binding']['selected_frames']}")
        print(f"  binding_ratio: {binding['binding_ratio']}")


def validate_external_graspnet_data(
    *,
    realsense_path: Path | None,
    prediction_path: Path | None,
    out_dir: Path,
    requested_frames: Sequence[int | str] = (),
    max_frames: int | None = 5,
    factor_depth: int = 1000,
    min_boundary_pixels: int = 50,
    binding_pixel_radius: int = 3,
    binding_depth_tolerance_m: float | None = 0.12,
    save_previews: bool = True,
    export_demo_bundle: bool = True,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "output_dir": str(out_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    if realsense_path is not None:
        summary["realsense"] = validate_realsense_data(
            realsense_path,
            out_dir=out_dir,
            requested_frames=requested_frames,
            max_frames=max_frames,
            factor_depth=factor_depth,
            min_boundary_pixels=min_boundary_pixels,
            save_previews=save_previews,
            export_demo_bundle=export_demo_bundle,
        )

    if prediction_path is not None:
        prediction_summary = summarize_graspnet_prediction_package(prediction_path)
        prediction_summary["path"] = str(prediction_path)
        summary["prediction"] = prediction_summary

    if realsense_path is not None and prediction_path is not None:
        summary["binding"] = validate_prediction_bindings(
            realsense_path=realsense_path,
            prediction_path=prediction_path,
            requested_frames=requested_frames,
            max_frames=max_frames,
            factor_depth=factor_depth,
            pixel_radius=binding_pixel_radius,
            depth_tolerance_m=binding_depth_tolerance_m,
        )

    (out_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def validate_prediction_bindings(
    *,
    realsense_path: Path,
    prediction_path: Path,
    requested_frames: Sequence[int | str] = (),
    max_frames: int | None = 5,
    factor_depth: int = 1000,
    pixel_radius: int = 3,
    depth_tolerance_m: float | None = 0.12,
) -> dict[str, object]:
    frame_results: list[dict[str, object]] = []
    with GraspNetRealSenseSource.open(realsense_path) as realsense_source, GraspNetPredictionSource.open(prediction_path) as prediction_source:
        selected_frames = select_matching_frame_ids(
            realsense_source.list_frames(),
            prediction_source.list_frames(),
            requested=requested_frames,
            max_frames=max_frames,
        )
        for frame_id in selected_frames:
            frame = realsense_source.load_frame(frame_id)
            frame = RealSenseFrame(
                frame=frame.frame,
                color=frame.color,
                depth_raw=frame.depth_raw,
                label=frame.label,
                intrinsic_matrix=frame.intrinsic_matrix,
                camera_pose=frame.camera_pose,
                cam0_wrt_table=frame.cam0_wrt_table,
                factor_depth=factor_depth,
            )
            objects = realsense_source.load_annotation_objects(frame.frame) if realsense_source.exists(f"annotations/{frame.frame}.xml") else []
            records = prediction_source.load_records(frame.frame)
            bindings = bind_graspnet_records_to_frame_labels(
                records,
                frame,
                objects,
                pixel_radius=pixel_radius,
                depth_tolerance_m=depth_tolerance_m,
            )
            summary = summarize_bound_graspnet_candidates(bindings, objects)
            poses_by_object = bound_candidates_to_grasp_poses_by_object(bindings)
            pose_counts = {name: len(poses) for name, poses in poses_by_object.items()}
            summary["grasp_pose_candidate_count"] = sum(pose_counts.values())
            summary["grasp_pose_candidate_count_by_object"] = pose_counts
            summary["frame"] = frame.frame
            frame_results.append(summary)

    return {
        "realsense_path": str(realsense_path),
        "prediction_path": str(prediction_path),
        "selected_frames": selected_frames,
        "pixel_radius": pixel_radius,
        "depth_tolerance_m": depth_tolerance_m,
        "frame_results": frame_results,
        "aggregate": aggregate_binding_summaries(frame_results),
    }


def validate_realsense_data(
    realsense_path: Path,
    *,
    out_dir: Path,
    requested_frames: Sequence[int | str] = (),
    max_frames: int | None = 5,
    factor_depth: int = 1000,
    min_boundary_pixels: int = 50,
    save_previews: bool = True,
    export_demo_bundle: bool = True,
) -> dict[str, object]:
    frame_results: list[dict[str, object]] = []
    od_reports: list[dict[str, object]] = []
    demo_bundle_path = None
    with GraspNetRealSenseSource.open(realsense_path) as source:
        available_frames = source.list_frames()
        selected_frames = select_frame_ids(available_frames, requested=requested_frames, max_frames=max_frames)
        for frame_id in selected_frames:
            frame = source.load_frame(frame_id)
            frame = RealSenseFrame(
                frame=frame.frame,
                color=frame.color,
                depth_raw=frame.depth_raw,
                label=frame.label,
                intrinsic_matrix=frame.intrinsic_matrix,
                camera_pose=frame.camera_pose,
                cam0_wrt_table=frame.cam0_wrt_table,
                factor_depth=factor_depth,
            )
            result = summarize_realsense_frame(frame, min_boundary_pixels=min_boundary_pixels)
            if source.exists(f"annotations/{frame.frame}.xml"):
                od_report = assess_single_view_od_sufficiency(
                    frame,
                    source.load_annotation_objects(frame.frame),
                    min_boundary_pixels=min_boundary_pixels,
                )
                result["single_view_od_sufficiency"] = od_report
                od_reports.append(od_report)
            if save_previews:
                preview_path = out_dir / f"frame_{frame.frame}_single_view_validation.png"
                make_frame_preview(frame).save(preview_path)
                result["preview"] = str(preview_path)
            frame_results.append(result)

        if export_demo_bundle and selected_frames:
            first_frame = source.load_frame(selected_frames[0])
            first_frame = RealSenseFrame(
                frame=first_frame.frame,
                color=first_frame.color,
                depth_raw=first_frame.depth_raw,
                label=first_frame.label,
                intrinsic_matrix=first_frame.intrinsic_matrix,
                camera_pose=first_frame.camera_pose,
                cam0_wrt_table=first_frame.cam0_wrt_table,
                factor_depth=factor_depth,
            )
            demo_bundle_path = out_dir / f"graspnet_demo_frame_{first_frame.frame}"
            write_graspnet_input_bundle(
                demo_bundle_path,
                color=first_frame.color,
                depth_meters=first_frame.depth_meters,
                intrinsic_matrix=first_frame.intrinsic_matrix,
                factor_depth=factor_depth,
                workspace_mask=first_frame.depth_raw > 0,
                camera="realsense",
                scene=f"external_realsense/{first_frame.frame}",
                camera_to_world_matrix=_camera_to_table_matrix(first_frame),
            )

    return {
        "path": str(realsense_path),
        "root_prefix": source.root_prefix,
        "available_frame_count": len(available_frames),
        "selected_frames": selected_frames,
        "frame_results": frame_results,
        "od_sufficiency_aggregate": aggregate_od_sufficiency_reports(od_reports) if od_reports else None,
        "demo_bundle": str(demo_bundle_path) if demo_bundle_path is not None else None,
    }


def select_frame_ids(
    available_frames: Sequence[str],
    *,
    requested: Sequence[int | str] = (),
    max_frames: int | None = 5,
) -> list[str]:
    available = sorted({normalize_frame_id(frame) for frame in available_frames})
    available_set = set(available)
    if not available:
        return []

    if requested:
        selected = [normalize_frame_id(frame) for frame in requested]
        missing = [frame for frame in selected if frame not in available_set]
        if missing:
            raise ValueError(f"Requested frames are not available: {missing}")
        return selected

    if max_frames is None or max_frames <= 0 or max_frames >= len(available):
        return available
    indices = np.linspace(0, len(available) - 1, num=max_frames, dtype=int)
    return [available[int(index)] for index in indices]


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
    common_set = set(common)
    if requested:
        return [normalize_frame_id(frame) for frame in requested if normalize_frame_id(frame) in common_set]
    return select_frame_ids(common, requested=(), max_frames=max_frames)


def aggregate_od_sufficiency_reports(reports: Sequence[dict[str, object]]) -> dict[str, object]:
    if not reports:
        return {
            "frame_count": 0,
            "sufficient_frame_count": 0,
            "mean_direct_pair_observability_ratio": None,
            "max_hidden_object_count": 0,
            "max_unobservable_pair_count": 0,
            "insufficiency_reason_counts": {},
        }

    ratios = [float(report["direct_pair_observability_ratio"]) for report in reports]
    reason_counts: dict[str, int] = {}
    for report in reports:
        for reason in report.get("insufficiency_reasons", []):
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "frame_count": len(reports),
        "sufficient_frame_count": sum(1 for report in reports if bool(report["single_view_sufficient_for_complete_od"])),
        "mean_direct_pair_observability_ratio": float(np.mean(ratios)),
        "max_hidden_object_count": max(int(report["hidden_object_count"]) for report in reports),
        "max_unobservable_pair_count": max(int(report["unobservable_pair_count"]) for report in reports),
        "insufficiency_reason_counts": dict(sorted(reason_counts.items())),
    }


def aggregate_binding_summaries(summaries: Sequence[dict[str, object]]) -> dict[str, object]:
    status_counter: Counter[str] = Counter()
    object_stats: dict[int, dict[str, object]] = {}
    grasp_pose_objects: set[str] = set()
    total_candidates = 0
    bound_count = 0
    unbound_count = 0
    grasp_pose_candidate_count = 0

    for summary in summaries:
        total_candidates += int(summary["total_candidates"])
        bound_count += int(summary["bound_count"])
        unbound_count += int(summary["unbound_count"])
        per_object_pose_counts = dict(summary.get("grasp_pose_candidate_count_by_object", {}))
        if "grasp_pose_candidate_count" in summary:
            grasp_pose_candidate_count += int(summary["grasp_pose_candidate_count"])
        else:
            grasp_pose_candidate_count += sum(int(count) for count in per_object_pose_counts.values())
        grasp_pose_objects.update(str(name) for name, count in per_object_pose_counts.items() if int(count) > 0)
        status_counter.update({str(key): int(value) for key, value in dict(summary["status_counts"]).items()})
        for obj in summary.get("objects", []):
            label_id = int(obj["label_id"])
            item = object_stats.setdefault(
                label_id,
                {
                    "label_id": label_id,
                    "object_id": obj.get("object_id"),
                    "object_name": obj.get("object_name"),
                    "total_candidate_count": 0,
                    "frames_with_candidates": 0,
                    "best_score": None,
                },
            )
            count = int(obj.get("candidate_count", 0))
            item["total_candidate_count"] = int(item["total_candidate_count"]) + count
            if count > 0:
                item["frames_with_candidates"] = int(item["frames_with_candidates"]) + 1
            best_score = obj.get("best_score")
            if best_score is not None:
                item["best_score"] = max(float(best_score), float(item["best_score"]) if item["best_score"] is not None else float(best_score))

    return {
        "frame_count": len(summaries),
        "total_candidates": total_candidates,
        "bound_count": bound_count,
        "unbound_count": unbound_count,
        "binding_ratio": float(bound_count / total_candidates) if total_candidates else None,
        "status_counts": dict(sorted(status_counter.items())),
        "grasp_pose_candidate_count": grasp_pose_candidate_count,
        "grasp_pose_object_count": len(grasp_pose_objects),
        "grasp_pose_objects": sorted(grasp_pose_objects),
        "objects": sorted(object_stats.values(), key=lambda item: int(item["label_id"])),
    }


def make_frame_preview(frame: RealSenseFrame) -> Image.Image:
    points, valid_mask = depth_to_point_cloud(
        frame.depth_raw,
        frame.intrinsic_matrix,
        factor_depth=frame.factor_depth,
    )
    pc_preview = _point_cloud_xz_preview(points, frame.color, valid_mask)
    panels = [
        _with_title(Image.fromarray(frame.color), f"RGB frame {frame.frame}"),
        _with_title(Image.fromarray(_depth_visualization(frame.depth_raw)), "Depth visualization"),
        _with_title(Image.fromarray(_label_visualization(frame.label, frame.depth_raw.shape)), "Visible label mask"),
        _with_title(Image.fromarray(pc_preview), "Single-view point cloud XZ"),
    ]
    panel_w, panel_h = 426, 240
    canvas = Image.new("RGB", (panel_w * 2, panel_h * 2), (255, 255, 255))
    for index, panel in enumerate(panels):
        panel = panel.resize((panel_w, panel_h))
        canvas.paste(panel, ((index % 2) * panel_w, (index // 2) * panel_h))
    return canvas


def _resolve_output_dir(out_dir: Path | None) -> Path:
    if out_dir is None:
        return PROJECT_ROOT / "results" / "external_graspnet_validation" / (
            "validation_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    return out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir


def _camera_to_table_matrix(frame: RealSenseFrame) -> np.ndarray | None:
    if frame.cam0_wrt_table is None or frame.camera_pose is None:
        return None
    return np.asarray(frame.cam0_wrt_table, dtype=float) @ np.asarray(frame.camera_pose, dtype=float)


def _with_title(image: Image.Image, title: str) -> Image.Image:
    rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    draw.rectangle([0, 0, rgb.width, 24], fill=(255, 255, 255))
    draw.text((8, 5), title, fill=(0, 0, 0))
    return rgb


def _depth_visualization(depth_raw: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_raw)
    valid = depth > 0
    output = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        values = depth[valid].astype(np.float32)
        lo = float(np.percentile(values, 1))
        hi = float(np.percentile(values, 99))
        if hi <= lo:
            hi = float(values.max())
            lo = float(values.min())
        scaled = (depth.astype(np.float32) - lo) / max(hi - lo, 1.0)
        output[valid] = np.clip(scaled[valid] * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(output[:, :, None], 3, axis=2)


def _label_visualization(label: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if label is None:
        return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    label_array = np.asarray(label)
    output = np.zeros((*label_array.shape, 3), dtype=np.uint8)
    for label_id in np.unique(label_array):
        if label_id == 0:
            continue
        output[label_array == label_id] = np.array(
            [
                (int(label_id) * 37) % 255,
                (int(label_id) * 67 + 50) % 255,
                (int(label_id) * 97 + 100) % 255,
            ],
            dtype=np.uint8,
        )
    return output


def _point_cloud_xz_preview(points: np.ndarray, color: np.ndarray, valid_mask: np.ndarray, size: int = 512) -> np.ndarray:
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    if len(points) == 0:
        return canvas
    valid_indices = np.flatnonzero(valid_mask.reshape(-1))
    sample_n = min(60000, len(points))
    rng = np.random.default_rng(1234)
    choice = rng.choice(len(points), sample_n, replace=False)
    sampled = points[choice]
    sampled_color = color.reshape(-1, 3)[valid_indices[choice]]
    x = sampled[:, 0]
    z = sampled[:, 2]
    x_lo, x_hi = np.percentile(x, [1, 99])
    z_lo, z_hi = np.percentile(z, [1, 99])
    if x_hi <= x_lo or z_hi <= z_lo:
        return canvas
    px = np.clip(((x - x_lo) / (x_hi - x_lo) * (size - 1)).astype(int), 0, size - 1)
    py = np.clip(((1.0 - (z - z_lo) / (z_hi - z_lo)) * (size - 1)).astype(int), 0, size - 1)
    canvas[py, px] = sampled_color
    return canvas


if __name__ == "__main__":
    main()
