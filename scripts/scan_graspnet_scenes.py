from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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

from scripts.validate_external_graspnet_data import select_frame_ids  # noqa: E402
from stacked_grasping.gripper.external_graspnet_data import (  # noqa: E402
    GraspNetRealSenseSource,
    RealSenseFrame,
    assess_single_view_od_sufficiency,
    summarize_realsense_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan GraspNet official scenes and rank frames/scenes by single-view OD difficulty."
    )
    parser.add_argument("--scene-root", type=Path, required=True, help="GraspNet scenes root, e.g. .../graspnet/scenes.")
    parser.add_argument("--camera", default="realsense", help="Camera folder inside each scene. Default: realsense.")
    parser.add_argument("--scenes", nargs="*", default=(), help="Optional scene names, e.g. scene_0007 scene_0100.")
    parser.add_argument("--frames", nargs="*", default=(), help="Optional frame ids to sample from every scene.")
    parser.add_argument("--max-scenes", type=int, default=20, help="Limit number of scenes when --scenes is omitted.")
    parser.add_argument("--max-frames-per-scene", type=int, default=8, help="Evenly sample this many frames per scene.")
    parser.add_argument("--factor-depth", type=int, default=1000)
    parser.add_argument("--min-boundary-pixels", type=int, default=50)
    parser.add_argument("--top", type=int, default=10, help="Print this many highest difficulty scenes.")
    parser.add_argument("--out-dir", type=Path, help="Optional output directory for JSON/CSV scan results.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = scan_graspnet_scenes(
        args.scene_root,
        camera=args.camera,
        requested_scenes=tuple(args.scenes),
        requested_frames=tuple(args.frames),
        max_scenes=args.max_scenes,
        max_frames_per_scene=args.max_frames_per_scene,
        factor_depth=args.factor_depth,
        min_boundary_pixels=args.min_boundary_pixels,
    )

    if args.out_dir is not None:
        write_scan_outputs(result, args.out_dir)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("GraspNet scene scan finished")
    print(f"  scene_root: {result['scene_root']}")
    print(f"  scanned_scene_count: {result['scanned_scene_count']}")
    print(f"  scanned_frame_count: {result['scanned_frame_count']}")
    if args.out_dir is not None:
        print(f"  output_dir: {args.out_dir}")
    print("  top difficult scenes:")
    for item in result["scene_summaries"][: max(args.top, 0)]:
        print(
            "   "
            f"{item['scene']}: score={item['difficulty_score']:.3f}, "
            f"frames={item['frame_count']}, "
            f"hidden_mean={item['mean_hidden_object_count']:.2f}, "
            f"unobservable_mean={item['mean_unobservable_pair_count']:.2f}, "
            f"boundary_mean={item['mean_visible_boundary_edge_count']:.2f}, "
            f"od_ratio_mean={item['mean_direct_pair_observability_ratio']:.3f}"
        )


def scan_graspnet_scenes(
    scene_root: str | Path,
    *,
    camera: str = "realsense",
    requested_scenes: Sequence[str] = (),
    requested_frames: Sequence[int | str] = (),
    max_scenes: int | None = 20,
    max_frames_per_scene: int | None = 8,
    factor_depth: int = 1000,
    min_boundary_pixels: int = 50,
) -> dict[str, object]:
    root = Path(scene_root)
    scene_names = list(requested_scenes) if requested_scenes else select_scene_names(root, max_scenes=max_scenes)
    frame_records: list[dict[str, object]] = []
    failed_scenes: list[dict[str, str]] = []

    for scene_name in scene_names:
        scene_dir = root / scene_name
        try:
            frame_records.extend(
                scan_graspnet_scene(
                    scene_dir,
                    scene_name=scene_name,
                    camera=camera,
                    requested_frames=requested_frames,
                    max_frames_per_scene=max_frames_per_scene,
                    factor_depth=factor_depth,
                    min_boundary_pixels=min_boundary_pixels,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised on real incomplete datasets.
            failed_scenes.append({"scene": scene_name, "error": str(exc)})

    scene_summaries = aggregate_scene_records(frame_records)
    return {
        "scene_root": str(root),
        "camera": camera,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "requested_scenes": list(requested_scenes),
        "selected_scenes": scene_names,
        "scanned_scene_count": len({str(record["scene"]) for record in frame_records}),
        "scanned_frame_count": len(frame_records),
        "failed_scenes": failed_scenes,
        "scene_summaries": scene_summaries,
        "frame_records": frame_records,
    }


def scan_graspnet_scene(
    scene_dir: Path,
    *,
    scene_name: str | None = None,
    camera: str = "realsense",
    requested_frames: Sequence[int | str] = (),
    max_frames_per_scene: int | None = 8,
    factor_depth: int = 1000,
    min_boundary_pixels: int = 50,
) -> list[dict[str, object]]:
    source_path = scene_dir / camera if (scene_dir / camera).is_dir() else scene_dir
    name = scene_name or scene_dir.name
    records: list[dict[str, object]] = []
    with GraspNetRealSenseSource.open(source_path) as source:
        selected_frames = select_frame_ids(source.list_frames(), requested=requested_frames, max_frames=max_frames_per_scene)
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
            if not source.exists(f"annotations/{frame.frame}.xml"):
                continue
            frame_summary = summarize_realsense_frame(frame, min_boundary_pixels=min_boundary_pixels)
            od_report = assess_single_view_od_sufficiency(
                frame,
                source.load_annotation_objects(frame.frame),
                min_boundary_pixels=min_boundary_pixels,
            )
            records.append(make_frame_record(name, source.root_prefix, frame_summary, od_report))
    return records


def make_frame_record(
    scene_name: str,
    source_root_prefix: str,
    frame_summary: dict[str, object],
    od_report: dict[str, object],
) -> dict[str, object]:
    return {
        "scene": scene_name,
        "source_root_prefix": source_root_prefix,
        "frame": str(frame_summary["frame"]),
        "complete_object_count": int(od_report["complete_object_count"]),
        "visible_object_count": int(od_report["visible_object_count"]),
        "hidden_object_count": int(od_report["hidden_object_count"]),
        "complete_pair_count": int(od_report["complete_pair_count"]),
        "direct_visible_boundary_pair_count": int(od_report["direct_visible_boundary_pair_count"]),
        "hidden_object_pair_count": int(od_report["hidden_object_pair_count"]),
        "visible_nonboundary_pair_count": int(od_report["visible_nonboundary_pair_count"]),
        "unobservable_pair_count": int(od_report["unobservable_pair_count"]),
        "direct_pair_observability_ratio": float(od_report["direct_pair_observability_ratio"]),
        "single_view_sufficient_for_complete_od": bool(od_report["single_view_sufficient_for_complete_od"]),
        "insufficiency_reasons": list(od_report.get("insufficiency_reasons", [])),
        "visible_label_count": int(frame_summary.get("visible_label_count", 0)),
        "visible_boundary_edge_count": int(frame_summary.get("visible_boundary_edge_count", 0)),
        "valid_depth_pixels": int(frame_summary["valid_depth_pixels"]),
        "point_cloud_points": int(frame_summary["point_cloud_points"]),
    }


def aggregate_scene_records(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["scene"])].append(dict(record))

    summaries = []
    for scene_name, scene_records in grouped.items():
        hidden = _numeric_values(scene_records, "hidden_object_count")
        unobservable = _numeric_values(scene_records, "unobservable_pair_count")
        boundary = _numeric_values(scene_records, "visible_boundary_edge_count")
        ratios = _numeric_values(scene_records, "direct_pair_observability_ratio")
        objects = _numeric_values(scene_records, "complete_object_count")
        difficulty_score = (
            _mean(hidden) * 2.0
            + _mean(unobservable)
            + _mean(boundary) * 0.1
            + _mean(objects) * 0.05
            + (1.0 - _mean(ratios)) * 2.0
        )
        summaries.append(
            {
                "scene": scene_name,
                "frame_count": len(scene_records),
                "difficulty_score": float(difficulty_score),
                "mean_complete_object_count": float(_mean(objects)),
                "max_complete_object_count": int(max(objects)) if objects else 0,
                "mean_hidden_object_count": float(_mean(hidden)),
                "max_hidden_object_count": int(max(hidden)) if hidden else 0,
                "mean_unobservable_pair_count": float(_mean(unobservable)),
                "max_unobservable_pair_count": int(max(unobservable)) if unobservable else 0,
                "mean_visible_boundary_edge_count": float(_mean(boundary)),
                "max_visible_boundary_edge_count": int(max(boundary)) if boundary else 0,
                "mean_direct_pair_observability_ratio": float(_mean(ratios)),
                "sufficient_frame_count": sum(
                    1 for record in scene_records if bool(record.get("single_view_sufficient_for_complete_od", False))
                ),
            }
        )
    return sorted(summaries, key=lambda item: (-float(item["difficulty_score"]), str(item["scene"])))


def select_scene_names(scene_root: str | Path, *, max_scenes: int | None = 20) -> list[str]:
    root = Path(scene_root)
    if not root.exists():
        raise FileNotFoundError(f"Scene root does not exist: {root}")
    scene_names = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("scene_"))
    if max_scenes is None or max_scenes <= 0:
        return scene_names
    return scene_names[:max_scenes]


def write_scan_outputs(result: dict[str, object], out_dir: str | Path) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "graspnet_scene_scan_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(target / "graspnet_scene_scan_scenes.csv", result["scene_summaries"])
    _write_csv(target / "graspnet_scene_scan_frames.csv", result["frame_records"])


def _write_csv(path: Path, rows: object) -> None:
    row_list = [dict(row) for row in rows]  # type: ignore[arg-type]
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(row_list[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _numeric_values(records: Sequence[dict[str, object]], key: str) -> list[float]:
    return [float(record[key]) for record in records]


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


if __name__ == "__main__":
    main()
