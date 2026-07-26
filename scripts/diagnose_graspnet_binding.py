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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_external_graspnet_pointcloud_episode import select_matching_frame_ids  # noqa: E402
from stacked_grasping.gripper.external_graspnet_data import GraspNetRealSenseSource, RealSenseFrame  # noqa: E402
from stacked_grasping.gripper.graspnet_binding import (  # noqa: E402
    GraspNetPredictionSource,
    bind_graspnet_records_to_frame_labels,
    project_camera_points_to_pixels,
)


UNBOUND_STATUSES = ("no-label", "invalid-translation", "out-of-frame", "background", "depth-mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why GraspNet candidates fail to bind to labeled objects, and sweep "
            "pixel-radius / depth-tolerance to show how much each relaxation recovers."
        )
    )
    parser.add_argument("--realsense", type=Path, required=True, help="GraspNet-style realsense directory or zip.")
    parser.add_argument("--prediction", type=Path, required=True, help="GraspNet prediction .npy directory, file, or zip.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for JSON/CSV summaries.")
    parser.add_argument("--frames", nargs="*", default=(), help="Frame ids to sample, e.g. 14 75 156.")
    parser.add_argument("--max-frames", type=int, default=0, help="Evenly sample this many frames; <=0 means all common frames.")
    parser.add_argument("--factor-depth", type=int, default=1000)
    parser.add_argument("--pixel-radii", nargs="+", type=int, default=(0, 2, 3, 5, 8))
    parser.add_argument(
        "--depth-tolerances",
        nargs="+",
        type=float,
        default=(0.08, 0.12, 0.2, float("inf")),
        help="Depth tolerances in meters; use inf to disable the depth check.",
    )
    parser.add_argument("--base-pixel-radius", type=int, default=3, help="Cell used for the detailed reason breakdown.")
    parser.add_argument("--base-depth-tolerance-m", type=float, default=0.12)
    parser.add_argument("--needed-radius-cap", type=int, default=15, help="Max radius probed for background candidates.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args()


def needed_label_radius(label: np.ndarray, u: int, v: int, max_radius: int) -> int | None:
    """Smallest Chebyshev radius (<= max_radius) at which a positive label pixel exists around (u, v)."""
    label_arr = np.asarray(label)
    height, width = label_arr.shape
    for radius in range(0, int(max_radius) + 1):
        y0, y1 = max(0, v - radius), min(height, v + radius + 1)
        x0, x1 = max(0, u - radius), min(width, u + radius + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        if (label_arr[y0:y1, x0:x1] > 0).any():
            return radius
    return None


def sweep_binding_cells(
    records: Sequence[dict[str, object]],
    frame: RealSenseFrame,
    annotations: Sequence[object],
    *,
    pixel_radii: Sequence[int] = (0, 2, 3, 5, 8),
    depth_tolerances: Sequence[float | None] = (0.08, 0.12, 0.2, None),
) -> list[dict[str, object]]:
    cells = []
    for radius in pixel_radii:
        for tolerance in depth_tolerances:
            bindings = bind_graspnet_records_to_frame_labels(
                records,
                frame,
                annotations,
                pixel_radius=int(radius),
                depth_tolerance_m=tolerance,
            )
            status_counts = Counter(binding.status for binding in bindings)
            bound = int(status_counts.get("bound", 0))
            cells.append(
                {
                    "pixel_radius": int(radius),
                    "depth_tolerance_m": None if tolerance is None else float(tolerance),
                    "total_candidates": len(bindings),
                    "bound_count": bound,
                    "bound_ratio": float(bound / len(bindings)) if bindings else None,
                    "status_counts": dict(sorted(status_counts.items())),
                }
            )
    return cells


def frame_binding_diagnostics(
    records: Sequence[dict[str, object]],
    frame: RealSenseFrame,
    annotations: Sequence[object],
    *,
    base_pixel_radius: int = 3,
    base_depth_tolerance_m: float | None = 0.12,
    needed_radius_cap: int = 15,
) -> dict[str, object]:
    bindings = bind_graspnet_records_to_frame_labels(
        records,
        frame,
        annotations,
        pixel_radius=base_pixel_radius,
        depth_tolerance_m=base_depth_tolerance_m,
    )
    status_counts = Counter(binding.status for binding in bindings)

    needed_radius_counter: Counter[str] = Counter()
    depth_errors: list[float] = []
    for binding in bindings:
        if binding.status == "background" and binding.pixel is not None and frame.label is not None:
            radius = needed_label_radius(frame.label, binding.pixel[0], binding.pixel[1], needed_radius_cap)
            needed_radius_counter[str(radius) if radius is not None else f">{needed_radius_cap}"] += 1
        elif binding.status == "depth-mismatch" and binding.depth_error_m is not None:
            depth_errors.append(float(binding.depth_error_m))

    return {
        "frame": frame.frame,
        "total_candidates": len(bindings),
        "bound_count": int(status_counts.get("bound", 0)),
        "status_counts": dict(sorted(status_counts.items())),
        "background_needed_radius_counts": dict(sorted(needed_radius_counter.items(), key=_radius_sort_key)),
        "depth_mismatch_errors_m": depth_errors,
    }


def aggregate_diagnostics(frame_diagnostics: Sequence[dict[str, object]]) -> dict[str, object]:
    status_totals: Counter[str] = Counter()
    needed_radius_totals: Counter[str] = Counter()
    depth_errors: list[float] = []
    total = 0
    bound = 0
    for item in frame_diagnostics:
        total += int(item["total_candidates"])
        bound += int(item["bound_count"])
        status_totals.update({str(k): int(v) for k, v in dict(item["status_counts"]).items()})
        needed_radius_totals.update({str(k): int(v) for k, v in dict(item["background_needed_radius_counts"]).items()})
        depth_errors.extend(float(value) for value in item["depth_mismatch_errors_m"])

    quantiles = {}
    if depth_errors:
        arr = np.asarray(depth_errors, dtype=float)
        quantiles = {
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }
    return {
        "frame_count": len(frame_diagnostics),
        "total_candidates": total,
        "bound_count": bound,
        "bound_ratio": float(bound / total) if total else None,
        "status_totals": dict(sorted(status_totals.items())),
        "background_needed_radius_totals": dict(sorted(needed_radius_totals.items(), key=_radius_sort_key)),
        "depth_mismatch_error_quantiles_m": quantiles,
        "depth_mismatch_count": len(depth_errors),
    }


def aggregate_sweep_cells(cells_per_frame: Sequence[list[dict[str, object]]]) -> list[dict[str, object]]:
    merged: dict[tuple[int, float | None], dict[str, object]] = {}
    for frame_cells in cells_per_frame:
        for cell in frame_cells:
            key = (int(cell["pixel_radius"]), cell["depth_tolerance_m"])
            slot = merged.setdefault(
                key,
                {
                    "pixel_radius": key[0],
                    "depth_tolerance_m": key[1],
                    "total_candidates": 0,
                    "bound_count": 0,
                    "status_counts": Counter(),
                },
            )
            slot["total_candidates"] += int(cell["total_candidates"])
            slot["bound_count"] += int(cell["bound_count"])
            slot["status_counts"].update({str(k): int(v) for k, v in dict(cell["status_counts"]).items()})

    rows = []
    for slot in merged.values():
        total = int(slot["total_candidates"])
        rows.append(
            {
                "pixel_radius": slot["pixel_radius"],
                "depth_tolerance_m": slot["depth_tolerance_m"],
                "total_candidates": total,
                "bound_count": int(slot["bound_count"]),
                "bound_ratio": float(slot["bound_count"] / total) if total else None,
                "status_counts": dict(sorted(dict(slot["status_counts"]).items())),
            }
        )
    rows.sort(key=lambda row: (row["pixel_radius"], float("inf") if row["depth_tolerance_m"] is None else row["depth_tolerance_m"]))
    return rows


def run_binding_diagnosis(
    *,
    realsense_path: Path,
    prediction_path: Path,
    out_dir: Path,
    requested_frames: Sequence[int | str] = (),
    max_frames: int | None = None,
    factor_depth: int = 1000,
    pixel_radii: Sequence[int] = (0, 2, 3, 5, 8),
    depth_tolerances: Sequence[float | None] = (0.08, 0.12, 0.2, None),
    base_pixel_radius: int = 3,
    base_depth_tolerance_m: float | None = 0.12,
    needed_radius_cap: int = 15,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_diagnostics: list[dict[str, object]] = []
    sweep_cells_per_frame: list[list[dict[str, object]]] = []
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
            frame_diagnostics.append(
                frame_binding_diagnostics(
                    records,
                    frame,
                    annotations,
                    base_pixel_radius=base_pixel_radius,
                    base_depth_tolerance_m=base_depth_tolerance_m,
                    needed_radius_cap=needed_radius_cap,
                )
            )
            sweep_cells_per_frame.append(
                sweep_binding_cells(
                    records,
                    frame,
                    annotations,
                    pixel_radii=pixel_radii,
                    depth_tolerances=depth_tolerances,
                )
            )

    summary = {
        "output_dir": str(out_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "realsense_path": str(realsense_path),
        "prediction_path": str(prediction_path),
        "selected_frames": selected_frames,
        "base_pixel_radius": base_pixel_radius,
        "base_depth_tolerance_m": base_depth_tolerance_m,
        "aggregate": aggregate_diagnostics(frame_diagnostics),
        "sweep_cells": aggregate_sweep_cells(sweep_cells_per_frame),
        "frame_diagnostics": frame_diagnostics,
    }
    (out_dir / "binding_diagnosis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sweep_csv(out_dir / "binding_sweep.csv", summary["sweep_cells"])
    return summary


def write_sweep_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = ["pixel_radius", "depth_tolerance_m", "total_candidates", "bound_count", "bound_ratio", "status_counts"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["status_counts"] = json.dumps(item.get("status_counts", {}), ensure_ascii=False)
            writer.writerow(item)


def _radius_sort_key(item: tuple[str, int]) -> tuple[int, float]:
    key = item[0]
    if key.startswith(">"):
        return (1, float("inf"))
    try:
        return (0, float(key))
    except ValueError:
        return (1, float("inf"))


def _resolve_output_dir(out_dir: Path | None) -> Path:
    if out_dir is None:
        return PROJECT_ROOT / "results" / "graspnet_binding_diagnosis" / (
            "diagnosis_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    return out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir


def main() -> None:
    args = parse_args()
    tolerances: list[float | None] = [None if not np.isfinite(value) else float(value) for value in args.depth_tolerances]
    summary = run_binding_diagnosis(
        realsense_path=args.realsense,
        prediction_path=args.prediction,
        out_dir=_resolve_output_dir(args.out_dir),
        requested_frames=tuple(args.frames),
        max_frames=args.max_frames if args.max_frames and args.max_frames > 0 else None,
        factor_depth=args.factor_depth,
        pixel_radii=tuple(args.pixel_radii),
        depth_tolerances=tuple(tolerances),
        base_pixel_radius=args.base_pixel_radius,
        base_depth_tolerance_m=args.base_depth_tolerance_m,
        needed_radius_cap=args.needed_radius_cap,
    )
    if args.json:
        compact = {key: value for key, value in summary.items() if key != "frame_diagnostics"}
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        return

    aggregate = summary["aggregate"]
    print("GraspNet binding diagnosis finished")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  frames: {len(summary['selected_frames'])}")
    print(f"  base bound_ratio: {aggregate['bound_ratio']}")
    print(f"  status_totals: {aggregate['status_totals']}")
    print(f"  background needed-radius: {aggregate['background_needed_radius_totals']}")
    print(f"  depth-mismatch quantiles: {aggregate['depth_mismatch_error_quantiles_m']}")
    print("  sweep:")
    for row in summary["sweep_cells"]:
        tolerance = "none" if row["depth_tolerance_m"] is None else f"{row['depth_tolerance_m']:g}"
        print(f"   radius={row['pixel_radius']} tol={tolerance}: bound_ratio={row['bound_ratio']}")


if __name__ == "__main__":
    main()
