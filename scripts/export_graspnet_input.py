from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.env.mujoco_scene import MujocoStackedScene
from stacked_grasping.gripper.graspnet_input import workspace_mask_from_depth, write_graspnet_input_bundle
from stacked_grasping.utils.paths import to_project_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one MuJoCo scene as a GraspNet demo RGB-D input bundle.")
    parser.add_argument("--scene", type=Path, required=True, help="MuJoCo scene XML path.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for color/depth/mask/meta files.")
    parser.add_argument("--camera", default="overview", help="MuJoCo camera name.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--settle-steps", type=int, default=800)
    parser.add_argument("--factor-depth", type=int, default=1000)
    parser.add_argument("--min-depth-m", type=float, default=0.0)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = _resolve_project_path(args.scene)
    out_dir = _resolve_project_path(args.out_dir)

    summary = export_scene_graspnet_input(
        scene_path=scene_path,
        out_dir=out_dir,
        camera=args.camera,
        width=args.width,
        height=args.height,
        settle_steps=args.settle_steps,
        factor_depth=args.factor_depth,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("GraspNet input bundle exported")
    print(f"  scene: {summary['scene']}")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  camera: {summary['camera']}")
    print(f"  size: {summary['width']} x {summary['height']}")
    print(f"  files: color.png, depth.png, workspace_mask.png, meta.mat, metadata.json")
    print(f"  depth_min_m: {summary['depth_min_m']}")
    print(f"  depth_max_m: {summary['depth_max_m']}")
    print(f"  workspace_mask_pixels: {summary['workspace_mask_pixels']}")
    print(f"  workspace_mask_ratio: {summary['workspace_mask_ratio']}")


def export_scene_graspnet_input(
    *,
    scene_path: Path,
    out_dir: Path,
    camera: str = "overview",
    width: int = 1280,
    height: int = 720,
    settle_steps: int = 800,
    factor_depth: int = 1000,
    min_depth_m: float = 0.0,
    max_depth_m: float = 3.0,
) -> dict[str, object]:
    scene = MujocoStackedScene(scene_path)
    scene.reset_and_settle(steps=settle_steps)
    color, depth = scene.render_rgbd(width=width, height=height, camera=camera)
    intrinsic = scene.camera_intrinsic_matrix(width=width, height=height, camera=camera)
    camera_to_world = scene.camera_to_world_matrix(camera=camera)
    workspace_mask = workspace_mask_from_depth(
        depth,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )

    summary = write_graspnet_input_bundle(
        out_dir,
        color=color,
        depth_meters=depth,
        intrinsic_matrix=intrinsic,
        factor_depth=factor_depth,
        workspace_mask=workspace_mask,
        camera=camera,
        scene=to_project_relative(scene_path, root=PROJECT_ROOT),
        camera_to_world_matrix=camera_to_world,
    )
    summary["output_dir"] = to_project_relative(out_dir, root=PROJECT_ROOT)
    summary["depth_min_m"] = float(depth[depth > 0].min()) if (depth > 0).any() else None
    summary["depth_max_m"] = float(depth.max())
    summary["workspace_mask_pixels"] = int(workspace_mask.sum())
    summary["workspace_mask_ratio"] = float(workspace_mask.mean())
    summary["workspace_depth_range_m"] = [min_depth_m, max_depth_m]
    return summary


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    main()
