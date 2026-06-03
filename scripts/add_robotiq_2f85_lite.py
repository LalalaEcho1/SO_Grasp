from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.gripper.robotiq_2f85_lite import (
    Robotiq2F85LiteConfig,
    attach_gripper_to_scene_xml,
    rewrite_asset_file_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add a lightweight Robotiq 2F-85 gripper to a MuJoCo scene.")
    parser.add_argument("--scene", type=Path, required=True, help="Input MuJoCo XML scene.")
    parser.add_argument("--out", type=Path, required=True, help="Output MuJoCo XML scene with gripper.")
    parser.add_argument("--target-object", required=True, help="Object body name to place the top-down gripper above.")
    parser.add_argument("--opening", type=float, default=0.085, help="Initial finger opening in meters.")
    parser.add_argument("--approach-height", type=float, default=0.20, help="Height above target object top surface.")
    parser.add_argument(
        "--asset-path-mode",
        choices=["relative-to-output", "absolute"],
        default="relative-to-output",
        help="How mesh/texture file paths should be rewritten in the output XML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = args.scene if args.scene.is_absolute() else PROJECT_ROOT / args.scene
    out_path = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    xml_text = scene_path.read_text(encoding="utf-8")
    result = attach_gripper_to_scene_xml(
        xml_text,
        target_object=args.target_object,
        config=Robotiq2F85LiteConfig(opening=args.opening, approach_height=args.approach_height),
    )
    result = rewrite_asset_file_paths(result, scene_path, output_path=out_path, mode=args.asset_path_mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")
    print(f"saved gripper scene: {out_path}")


if __name__ == "__main__":
    main()
