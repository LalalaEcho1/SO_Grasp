from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.env.mujoco_scene import MujocoStackedScene
from stacked_grasping.gripper.feasibility import (
    ObjectGraspFeasibility,
    TopDownGraspConfig,
    assess_scene_topdown_grasps,
)
from stacked_grasping.utils.paths import resolve_project_path, to_project_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate top-down Robotiq 2F-85 Lite grasp feasibility.")
    parser.add_argument("--scene", type=Path, required=True, help="MuJoCo scene XML path.")
    parser.add_argument("--settle-steps", type=int, default=800)
    parser.add_argument("--max-opening", type=float, default=0.085)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "gripper_feasibility")
    parser.add_argument("--save", action="store_true", help="Save JSON feasibility details.")
    return parser.parse_args()


def summarize_feasibility_results(results: Sequence[ObjectGraspFeasibility]) -> Dict[str, object]:
    object_count = len(results)
    feasible_objects = sum(1 for result in results if result.feasible)
    total_feasible_grasps = sum(result.feasible_grasp_count for result in results)
    denominator = max(object_count, 1)
    return {
        "object_count": object_count,
        "feasible_objects": feasible_objects,
        "feasible_object_rate": round(feasible_objects / denominator, 6),
        "total_feasible_grasps": total_feasible_grasps,
        "mean_feasible_grasps": round(total_feasible_grasps / denominator, 6),
        "mean_gripper_collision_risk": round(
            sum(result.gripper_collision_risk for result in results) / denominator,
            6,
        ),
    }


def main() -> None:
    args = parse_args()
    scene_path = resolve_project_path(args.scene)
    scene = MujocoStackedScene(scene_path)
    scene.reset_and_settle(args.settle_steps)
    objects = scene.read_objects()
    results = assess_scene_topdown_grasps(
        objects,
        config=TopDownGraspConfig(max_opening=args.max_opening),
    )
    summary = summarize_feasibility_results(results)
    payload = {
        "scene": to_project_relative(scene_path),
        "settle_steps": args.settle_steps,
        "max_opening": args.max_opening,
        "summary": summary,
        "objects": [result.to_dict() for result in results],
    }

    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"gripper_feasibility_{timestamp}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved feasibility JSON: {out_path}")

    print("Top-down gripper feasibility")
    print(f"  scene: {payload['scene']}")
    print(
        "  "
        f"objects={summary['object_count']}, "
        f"feasible_objects={summary['feasible_objects']}, "
        f"feasible_rate={summary['feasible_object_rate']:.3f}, "
        f"total_feasible_grasps={summary['total_feasible_grasps']}, "
        f"mean_collision_risk={summary['mean_gripper_collision_risk']:.3f}"
    )
    for result in results:
        print(
            "  "
            f"- {result.object_name}: feasible={result.feasible}, "
            f"count={result.feasible_grasp_count}, "
            f"collision_risk={result.gripper_collision_risk:.3f}"
        )


if __name__ == "__main__":
    main()
