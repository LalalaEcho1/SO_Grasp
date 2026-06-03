from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export headless scene and relation-graph visualizations.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=PROJECT_ROOT / "assets" / "scenes" / "stacked_blocks.xml",
        help="Path to a MuJoCo XML scene.",
    )
    parser.add_argument("--settle-steps", type=int, default=1500, help="Physics steps before exporting.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "visual_debug")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--camera", default="overview", help="MuJoCo camera name. Use empty string for default camera.")
    parser.add_argument(
        "--gl-backend",
        default="egl",
        choices=["egl", "osmesa", "glfw"],
        help="MuJoCo OpenGL backend. Use egl first on WSL with NVIDIA; use osmesa for CPU offscreen rendering.",
    )
    parser.add_argument("--skip-scene", action="store_true", help="Only export relation graph and JSON.")
    parser.add_argument("--skip-graph", action="store_true", help="Only export scene image and JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["MUJOCO_GL"] = args.gl_backend

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from PIL import Image

    from stacked_grasping.env.mujoco_scene import MujocoStackedScene
    from stacked_grasping.planning.adaptive_score import rank_objects
    from stacked_grasping.relations.graph import build_relation_graph
    from stacked_grasping.visualization.relation_plot import save_relation_graph_png

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)

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
        "gl_backend": args.gl_backend,
    }

    json_path = args.out_dir / f"relations_{timestamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved JSON: {json_path}")

    if not args.skip_scene:
        scene_path = args.out_dir / f"scene_{timestamp}.png"
        try:
            rgb = scene.render_rgb(width=args.width, height=args.height, camera=args.camera or None)
            Image.fromarray(rgb).save(scene_path)
            print(f"saved scene image: {scene_path}")
        except Exception as exc:
            print(f"scene image export failed with MUJOCO_GL={args.gl_backend}: {exc}")
            print("Try: sudo apt install -y libosmesa6 && python scripts/export_visuals.py --gl-backend osmesa")

    if not args.skip_graph:
        graph_path = args.out_dir / f"relation_graph_{timestamp}.png"
        save_relation_graph_png(graph, ranking, graph_path)
        print(f"saved relation graph: {graph_path}")


if __name__ == "__main__":
    main()

