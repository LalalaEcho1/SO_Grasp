from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.env.mujoco_scene import MujocoStackedScene
from stacked_grasping.planning.adaptive_score import rank_objects
from stacked_grasping.relations.graph import build_relation_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MuJoCo stacked-object relation graph prototype.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=PROJECT_ROOT / "assets" / "scenes" / "stacked_blocks.xml",
        help="Path to a MuJoCo XML scene.",
    )
    parser.add_argument("--settle-steps", type=int, default=1500, help="Physics steps before reading relations.")
    parser.add_argument("--headless", action="store_true", help="Only compute and print relations.")
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer after computing relations.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = MujocoStackedScene(args.scene)
    scene.reset_and_settle(args.settle_steps)

    objects = scene.read_objects()
    contact_pairs = scene.read_object_contact_pairs()
    graph = build_relation_graph(objects, contact_pairs)
    ranking = rank_objects(graph)

    payload = {
        "objects": [obj.to_dict() for obj in objects],
        "contact_pairs": sorted([list(pair) for pair in contact_pairs]),
        "edges": [edge.to_dict() for edge in graph.edges],
        "ranking": [score.to_dict() for score in ranking],
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("\nObjects")
        for obj in objects:
            pos = ", ".join(f"{v:.3f}" for v in obj.position)
            print(f"  - {obj.name}: pos=({pos}), geom={obj.geom_type}")

        print("\nContact pairs")
        if contact_pairs:
            for a, b in sorted(contact_pairs):
                print(f"  - {a} <-> {b}")
        else:
            print("  - none")

        print("\nDirected relation edges")
        for edge in graph.edges:
            if edge.contact or edge.support_source_to_target or edge.support_target_to_source or edge.xy_overlap_ratio > 0.05:
                print(
                    "  - "
                    f"{edge.source} -> {edge.target}: "
                    f"contact={int(edge.contact)}, "
                    f"support={int(edge.support_source_to_target)}, "
                    f"height_diff={edge.height_diff:.3f}, "
                    f"overlap={edge.xy_overlap_ratio:.3f}, "
                    f"od={edge.od:.3f}, "
                    f"blocked={edge.blocked_grasp_ratio:.2f}, "
                    f"blocked_kinds={edge.blocked_grasp_kinds}"
                )

        print("\nAdaptive baseline ranking")
        for idx, score in enumerate(ranking, start=1):
            print(
                f"  {idx}. {score.name}: score={score.score:.3f}, "
                f"blocked={score.blocked_by_od:.3f}, "
                f"support_risk={score.support_risk:.3f}, "
                f"clearance_gain={score.clearance_gain:.3f}"
            )

    if args.viewer:
        try:
            import mujoco
            import mujoco.viewer
        except ImportError as exc:
            raise SystemExit("MuJoCo viewer is unavailable. Install dependencies with: pip install -r requirements.txt") from exc

        with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(scene.model, scene.data)
                viewer.sync()
                time.sleep(scene.model.opt.timestep)


if __name__ == "__main__":
    main()
