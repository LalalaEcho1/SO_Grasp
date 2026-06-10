from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stacked_grasping.gripper.external_graspnet_data import (  # noqa: E402
    AnnotationObject,
    GraspNetRealSenseSource,
    RealSenseFrame,
    assess_single_view_od_sufficiency,
    normalize_frame_id,
    visible_boundary_edges,
)


DEFAULT_FRAMES_BY_SCENE: dict[str, list[str]] = {
    "scene_0015": ["0000", "0036", "0072", "0145", "0218", "0255"],
    "scene_0007": ["0000", "0036", "0072", "0109", "0218", "0255"],
    "scene_0009": ["0000", "0036", "0072", "0109", "0182", "0255"],
    "scene_0011": ["0000", "0036", "0072", "0109", "0218", "0255"],
    "scene_0017": ["0000", "0036", "0072", "0145", "0182", "0218"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RGB/depth/label/OD previews for selected GraspNet frames.")
    parser.add_argument("--scene-root", type=Path, default=PROJECT_ROOT / "data" / "graspnet" / "scenes")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "graspnet_candidate_previews")
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--scenes", nargs="*", default=tuple(DEFAULT_FRAMES_BY_SCENE), help="Scene names to export.")
    parser.add_argument(
        "--frames-by-scene",
        nargs="*",
        default=(),
        help="Override frames, e.g. scene_0009:0000,0036 scene_0015:72.",
    )
    parser.add_argument("--factor-depth", type=int, default=1000)
    parser.add_argument("--min-boundary-pixels", type=int, default=50)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=420)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_by_scene = parse_frames_by_scene(args.frames_by_scene) if args.frames_by_scene else {
        scene: DEFAULT_FRAMES_BY_SCENE[scene] for scene in args.scenes if scene in DEFAULT_FRAMES_BY_SCENE
    }
    missing = [scene for scene in args.scenes if scene not in frames_by_scene]
    if missing:
        raise ValueError(f"No frames configured for scenes: {missing}")

    summary = export_candidate_previews(
        args.scene_root,
        out_dir=args.out_dir,
        frames_by_scene=frames_by_scene,
        camera=args.camera,
        factor_depth=args.factor_depth,
        min_boundary_pixels=args.min_boundary_pixels,
        panel_size=(args.panel_width, args.panel_height),
    )
    print("GraspNet candidate previews exported")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  preview_count: {summary['preview_count']}")
    print("  object_sets:")
    for item in summary["object_sets"]:
        print(f"   set_{item['set_id']}: scenes={','.join(item['scenes'])}, objects={item['object_count']}")


def parse_frames_by_scene(items: Sequence[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Frame mapping must use scene:frame,frame format: {item}")
        scene, frames_text = item.split(":", 1)
        frames = [normalize_frame_id(frame.strip()) for frame in frames_text.split(",") if frame.strip()]
        if not scene or not frames:
            raise ValueError(f"Frame mapping must include a scene and at least one frame: {item}")
        parsed[scene] = frames
    return parsed


def export_candidate_previews(
    scene_root: str | Path,
    *,
    out_dir: str | Path,
    frames_by_scene: dict[str, list[str]],
    camera: str = "realsense",
    factor_depth: int = 1000,
    min_boundary_pixels: int = 50,
    panel_size: tuple[int, int] = (640, 420),
) -> dict[str, object]:
    root = Path(scene_root)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    previews: list[dict[str, object]] = []
    objects_by_scene: dict[str, list[tuple[int, str]]] = {}

    for scene_name, frame_ids in frames_by_scene.items():
        scene_dir = root / scene_name
        with GraspNetRealSenseSource.open(scene_dir / camera if (scene_dir / camera).is_dir() else scene_dir) as source:
            first_objects = source.load_annotation_objects(frame_ids[0])
            objects_by_scene[scene_name] = [(obj.object_id, obj.name) for obj in first_objects]
            scene_out = target / scene_name
            scene_out.mkdir(parents=True, exist_ok=True)
            for frame_id in frame_ids:
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
                objects = source.load_annotation_objects(frame.frame)
                od_report = assess_single_view_od_sufficiency(
                    frame,
                    objects,
                    min_boundary_pixels=min_boundary_pixels,
                )
                preview = make_candidate_preview(
                    frame,
                    objects,
                    od_report,
                    min_boundary_pixels=min_boundary_pixels,
                    panel_size=panel_size,
                )
                preview_path = scene_out / f"{scene_name}_frame_{frame.frame}_preview.png"
                preview.save(preview_path)
                previews.append(
                    {
                        "scene": scene_name,
                        "frame": frame.frame,
                        "preview": str(preview_path),
                        "complete_object_count": int(od_report["complete_object_count"]),
                        "visible_object_count": int(od_report["visible_object_count"]),
                        "hidden_object_count": int(od_report["hidden_object_count"]),
                        "unobservable_pair_count": int(od_report["unobservable_pair_count"]),
                        "direct_visible_boundary_pair_count": int(od_report["direct_visible_boundary_pair_count"]),
                        "direct_pair_observability_ratio": float(od_report["direct_pair_observability_ratio"]),
                        "insufficiency_reasons": list(od_report.get("insufficiency_reasons", [])),
                    }
                )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scene_root": str(root),
        "output_dir": str(target),
        "preview_count": len(previews),
        "frames_by_scene": frames_by_scene,
        "object_sets": summarize_object_sets(objects_by_scene),
        "previews": previews,
    }
    (target / "candidate_preview_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarize_object_sets(objects_by_scene: dict[str, list[tuple[int, str]]]) -> list[dict[str, object]]:
    grouped: dict[tuple[tuple[int, str], ...], list[str]] = defaultdict(list)
    for scene, objects in objects_by_scene.items():
        grouped[tuple(sorted((int(object_id), str(name)) for object_id, name in objects))].append(scene)

    summaries = []
    for set_id, (objects, scenes) in enumerate(
        sorted(grouped.items(), key=lambda item: (-len(item[1]), item[1][0] if item[1] else "")),
        start=1,
    ):
        summaries.append(
            {
                "set_id": set_id,
                "scenes": sorted(scenes),
                "object_count": len(objects),
                "objects": [{"object_id": object_id, "name": name} for object_id, name in objects],
            }
        )
    return summaries


def make_candidate_preview(
    frame: RealSenseFrame,
    objects: Sequence[AnnotationObject],
    od_report: dict[str, object],
    *,
    min_boundary_pixels: int = 50,
    panel_size: tuple[int, int] = (640, 420),
) -> Image.Image:
    panels = [
        _with_title(Image.fromarray(frame.color), f"RGB {frame.frame}"),
        _with_title(Image.fromarray(_depth_visualization(frame.depth_raw)), "Depth"),
        _with_title(Image.fromarray(_label_visualization(frame.label, frame.depth_raw.shape)), "Label"),
        _od_graph_panel(frame, objects, od_report, min_boundary_pixels=min_boundary_pixels),
    ]
    width, height = panel_size
    canvas = Image.new("RGB", (width * 2, height * 2), (255, 255, 255))
    for index, panel in enumerate(panels):
        canvas.paste(panel.convert("RGB").resize((width, height)), ((index % 2) * width, (index // 2) * height))
    return canvas


def _od_graph_panel(
    frame: RealSenseFrame,
    objects: Sequence[AnnotationObject],
    od_report: dict[str, object],
    *,
    min_boundary_pixels: int,
) -> Image.Image:
    visible_ids = {int(value) for value in od_report["visible_label_ids"]}
    hidden_ids = {int(value) for value in od_report["hidden_label_ids"]}
    direct_pairs = {tuple(int(value) for value in pair) for pair in od_report["direct_visible_boundary_pairs"]}
    hidden_pairs = {tuple(int(value) for value in pair) for pair in od_report["hidden_object_pairs"]}
    visible_nonboundary_pairs = {tuple(int(value) for value in pair) for pair in od_report["visible_nonboundary_pairs"]}
    label_to_object = {obj.label_id: obj for obj in objects}
    positions = _object_positions(objects)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120)
    ax.set_title("OD relation graph", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    _draw_pairs(ax, positions, visible_nonboundary_pairs, color="#b0b0b0", linestyle=":", alpha=0.35, max_edges=24)
    _draw_pairs(ax, positions, hidden_pairs, color="#c62828", linestyle="--", alpha=0.45, max_edges=24)
    _draw_pairs(ax, positions, direct_pairs, color="#ef8a00", linestyle="-", alpha=0.9, max_edges=999)

    for label_id, obj in sorted(label_to_object.items()):
        x, y = positions[label_id]
        color = "#9ad29b" if label_id in visible_ids else "#d9d9d9"
        edge_color = "#c62828" if label_id in hidden_ids else "#333333"
        ax.scatter([x], [y], s=520, c=[color], edgecolors=edge_color, linewidths=1.6, zorder=4)
        ax.text(
            x,
            y,
            f"{label_id}\n{_short_name(obj.name)}",
            ha="center",
            va="center",
            fontsize=7,
            color="#111111",
            zorder=5,
        )

    _fit_axes(ax, positions)
    summary = (
        f"objects {od_report['visible_object_count']}/{od_report['complete_object_count']} visible\n"
        f"direct edges {od_report['direct_visible_boundary_pair_count']} | "
        f"unobservable {od_report['unobservable_pair_count']}\n"
        f"OD observable ratio {float(od_report['direct_pair_observability_ratio']):.3f}"
    )
    ax.text(
        0.02,
        0.02,
        "orange: visible boundary\nred dashed: hidden-related\ngray dotted: non-boundary\n" + summary,
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#dddddd", "alpha": 0.92},
    )

    if frame.label is not None:
        edge_count = len(visible_boundary_edges(frame.label, frame.depth_meters, min_boundary_pixels=min_boundary_pixels))
        ax.text(0.98, 0.98, f"visible boundary edges: {edge_count}", transform=ax.transAxes, ha="right", va="top", fontsize=7)

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _draw_pairs(
    ax: plt.Axes,
    positions: dict[int, np.ndarray],
    pairs: set[tuple[int, int]],
    *,
    color: str,
    linestyle: str,
    alpha: float,
    max_edges: int,
) -> None:
    for index, pair in enumerate(sorted(pairs)):
        if index >= max_edges:
            break
        if pair[0] not in positions or pair[1] not in positions:
            continue
        start = positions[pair[0]]
        end = positions[pair[1]]
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linestyle=linestyle, linewidth=1.2, alpha=alpha, zorder=1)


def _object_positions(objects: Sequence[AnnotationObject]) -> dict[int, np.ndarray]:
    raw = {obj.label_id: np.asarray(obj.position[:2], dtype=float) for obj in objects}
    values = np.array(list(raw.values()), dtype=float)
    if len(values) == 0:
        return {}
    if np.max(np.ptp(values, axis=0)) < 1e-5:
        angles = np.linspace(0, 2 * np.pi, len(raw), endpoint=False)
        return {label_id: np.array([np.cos(angle), np.sin(angle)], dtype=float) for (label_id, _), angle in zip(raw.items(), angles)}
    return raw


def _fit_axes(ax: plt.Axes, positions: dict[int, np.ndarray]) -> None:
    if not positions:
        return
    arr = np.array(list(positions.values()), dtype=float)
    center = (arr.min(axis=0) + arr.max(axis=0)) * 0.5
    span = max(float(np.max(arr.max(axis=0) - arr.min(axis=0))), 0.3)
    margin = span * 0.65
    ax.set_xlim(center[0] - margin, center[0] + margin)
    ax.set_ylim(center[1] - margin, center[1] + margin)


def _with_title(image: Image.Image, title: str) -> Image.Image:
    rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    draw.rectangle([0, 0, rgb.width, 28], fill=(255, 255, 255))
    draw.text((8, 7), title, fill=(0, 0, 0))
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


def _short_name(name: str, *, max_len: int = 14) -> str:
    cleaned = name.replace(".ply", "").replace("072-", "").replace("_", " ")
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 1] + "."


if __name__ == "__main__":
    main()
