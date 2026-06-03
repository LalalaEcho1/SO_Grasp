from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.env.mujoco_scene import MujocoStackedScene
from stacked_grasping.planning.episode import format_episode_summary, run_adaptive_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an abstract sequential-grasp episode.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=PROJECT_ROOT / "assets" / "scenes" / "ycb_mesh_stacked.xml",
        help="Path to a MuJoCo XML scene.",
    )
    parser.add_argument("--settle-steps", type=int, default=1500, help="Initial physics steps before planning.")
    parser.add_argument("--post-grasp-steps", type=int, default=500, help="Physics steps after each abstract grasp.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many grasps.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "episodes")
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of the text summary.")
    parser.add_argument("--no-save", action="store_true", help="Do not save an episode JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.scene.exists():
        raise SystemExit(f"Scene not found: {args.scene}")

    scene = MujocoStackedScene(args.scene)
    scene.reset_and_settle(args.settle_steps)
    result = run_adaptive_episode(
        scene,
        max_steps=args.max_steps,
        post_grasp_settle_steps=args.post_grasp_steps,
    )

    payload = {
        "scene": str(args.scene),
        "settle_steps": args.settle_steps,
        "post_grasp_steps": args.post_grasp_steps,
        **result.to_dict(),
    }

    if not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.out_dir / f"episode_{timestamp}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved episode JSON: {json_path}")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_episode_summary(result))


if __name__ == "__main__":
    main()

