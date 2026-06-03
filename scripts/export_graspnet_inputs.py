from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.export_graspnet_input import export_scene_graspnet_input
from stacked_grasping.gripper.graspnet_input import graspnet_input_dir_for_scene
from stacked_grasping.utils.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch export MuJoCo scenes as GraspNet demo RGB-D input bundles.")
    parser.add_argument("--scene", type=Path, nargs="*", default=(), help="Explicit scene XML paths, kept in order.")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=PROJECT_ROOT / "assets" / "scenes" / "generated_main_v1",
        help="Directory of scene XML files used when --scene is omitted.",
    )
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "results" / "graspnet_inputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--settle-steps", type=int, default=800)
    parser.add_argument("--factor-depth", type=int, default=1000)
    parser.add_argument("--min-depth-m", type=float, default=0.0)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    return parser.parse_args()


def resolve_batch_scene_paths(
    *,
    scene_paths: Sequence[Path],
    scene_dir: Path | None,
    limit: int | None,
) -> list[Path]:
    if scene_paths:
        paths = list(scene_paths)
    else:
        if scene_dir is None:
            raise ValueError("scene_dir is required when scene_paths is empty.")
        paths = sorted(Path(scene_dir).glob("*.xml"))

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        paths = paths[:limit]
    return paths


def main() -> None:
    args = parse_args()
    scene_paths = resolve_batch_scene_paths(
        scene_paths=tuple(args.scene),
        scene_dir=resolve_project_path(args.scene_dir) if args.scene_dir is not None else None,
        limit=args.limit,
    )
    out_root = resolve_project_path(args.out_root)
    summaries = []

    for scene_path in scene_paths:
        resolved_scene = resolve_project_path(scene_path)
        out_dir = graspnet_input_dir_for_scene(out_root, resolved_scene)
        summary = export_scene_graspnet_input(
            scene_path=resolved_scene,
            out_dir=out_dir,
            camera=args.camera,
            width=args.width,
            height=args.height,
            settle_steps=args.settle_steps,
            factor_depth=args.factor_depth,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        summaries.append(summary)

    payload = {
        "scene_count": len(summaries),
        "out_root": str(out_root),
        "camera": args.camera,
        "summaries": summaries,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Batch GraspNet input export")
    print(f"  scenes: {len(summaries)}")
    print(f"  out_root: {out_root}")
    for summary in summaries:
        print(
            "  - "
            f"{summary['scene']} -> {summary['output_dir']} "
            f"mask={float(summary['workspace_mask_ratio']):.3f}"
        )


if __name__ == "__main__":
    main()
