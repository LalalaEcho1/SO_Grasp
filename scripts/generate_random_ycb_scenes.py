from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_ycb_mesh_scene import generate_scene
from stacked_grasping.assets.random_scene import sample_random_layout, scene_seeds
from stacked_grasping.assets.ycb_starter import STARTER_OBJECTS
from stacked_grasping.utils.paths import to_project_relative


TABLE_HALF_EXTENTS = (0.50, 0.38)
TABLE_TOP_Z = 0.385


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate randomized YCB MuJoCo stacked-object scenes.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "assets" / "objects" / "ycb",
        help="Directory containing normalized YCB object folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "assets" / "scenes" / "generated",
        help="Output directory for generated scene XML files.",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of scenes to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--object-count", type=int, default=None, help="Fixed number of objects per scene.")
    parser.add_argument("--min-objects", type=int, default=5, help="Minimum objects per scene if object count varies.")
    parser.add_argument("--stack-probability", type=float, default=0.65, help="Probability that an object is placed on a previous object.")
    parser.add_argument("--max-stack-depth", type=int, default=3, help="Maximum number of objects in one support chain.")
    parser.add_argument("--require-valid", action="store_true", help="Reject scenes without enough contacts and OD relation edges.")
    parser.add_argument("--min-contacts", type=int, default=1, help="Minimum object-object contact pairs for a valid scene.")
    parser.add_argument("--min-visible-edges", type=int, default=3, help="Minimum drawable OD relation edges for a valid scene.")
    parser.add_argument("--max-attempts", type=int, default=30, help="Maximum sampling attempts per accepted scene.")
    parser.add_argument("--settle-steps", type=int, default=1500, help="Physics settle steps used by validation.")
    parser.add_argument("--table-margin", type=float, default=0.03, help="Allowed tabletop overhang margin in meters.")
    parser.add_argument("--table-z-margin", type=float, default=0.04, help="Allowed vertical tolerance below tabletop in meters.")
    parser.add_argument("--max-object-top-z", type=float, default=0.82, help="Reject scenes whose highest object top exceeds this z value.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated scenes.")
    return parser.parse_args()


@dataclass(frozen=True)
class ValidationMetrics:
    object_count: int
    contact_pairs: int
    visible_edges: int
    out_of_bounds_objects: int = 0
    below_table_objects: int = 0
    max_top_z: float = 0.0


def count_visible_relation_edges(graph, min_od_to_draw: float = 0.18) -> int:
    return sum(
        1
        for edge in graph.edges
        if edge.contact
        or edge.support_source_to_target
        or edge.support_target_to_source
        or edge.od >= min_od_to_draw
        or edge.xy_overlap_ratio >= 0.08
    )


def is_valid_scene_metrics(
    metrics: ValidationMetrics,
    min_contacts: int,
    min_visible_edges: int,
    min_objects: int = 1,
    max_out_of_bounds_objects: int = 0,
    max_below_table_objects: int = 0,
    max_object_top_z: float | None = 0.82,
) -> bool:
    return bool(
        metrics.object_count >= min_objects
        and metrics.contact_pairs >= min_contacts
        and metrics.visible_edges >= min_visible_edges
        and metrics.out_of_bounds_objects <= max_out_of_bounds_objects
        and metrics.below_table_objects <= max_below_table_objects
        and (max_object_top_z is None or metrics.max_top_z <= max_object_top_z)
    )


def count_out_of_bounds_objects(objects, table_margin: float) -> int:
    half_x, half_y = TABLE_HALF_EXTENTS
    count = 0
    for obj in objects:
        x, y = obj.position[:2]
        size_x, size_y = obj.half_extents[:2]
        if abs(float(x)) + float(size_x) > half_x + table_margin:
            count += 1
        elif abs(float(y)) + float(size_y) > half_y + table_margin:
            count += 1
    return count


def count_below_table_objects(objects, table_z_margin: float) -> int:
    count = 0
    for obj in objects:
        bottom_z = float(obj.position[2] - obj.half_extents[2])
        if bottom_z < TABLE_TOP_Z - table_z_margin:
            count += 1
    return count


def max_object_top_z(objects) -> float:
    if not objects:
        return 0.0
    return max(float(obj.position[2] + obj.half_extents[2]) for obj in objects)


def validate_generated_scene(
    scene_path: Path,
    settle_steps: int,
    table_margin: float = 0.03,
    table_z_margin: float = 0.04,
) -> ValidationMetrics:
    from stacked_grasping.env.mujoco_scene import MujocoStackedScene
    from stacked_grasping.relations.graph import build_relation_graph

    scene = MujocoStackedScene(scene_path)
    scene.reset_and_settle(settle_steps)
    objects = scene.read_objects()
    contact_pairs = scene.read_object_contact_pairs()
    graph = build_relation_graph(objects, contact_pairs)
    return ValidationMetrics(
        object_count=len(objects),
        contact_pairs=len(contact_pairs),
        visible_edges=count_visible_relation_edges(graph),
        out_of_bounds_objects=count_out_of_bounds_objects(objects, table_margin=table_margin),
        below_table_objects=count_below_table_objects(objects, table_z_margin=table_z_margin),
        max_top_z=round(max_object_top_z(objects), 6),
    )


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if not 0.0 <= args.stack_probability <= 1.0:
        raise SystemExit("--stack-probability must be between 0 and 1")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")
    if args.max_stack_depth < 1:
        raise SystemExit("--max-stack-depth must be at least 1")
    if args.min_contacts < 0:
        raise SystemExit("--min-contacts must not be negative")
    if args.min_visible_edges < 0:
        raise SystemExit("--min-visible-edges must not be negative")
    if args.table_margin < 0.0:
        raise SystemExit("--table-margin must not be negative")
    if args.table_z_margin < 0.0:
        raise SystemExit("--table-z-margin must not be negative")
    if args.max_object_top_z <= TABLE_TOP_Z:
        raise SystemExit("--max-object-top-z must be above the tabletop height")

    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "count": args.count,
        "seed": args.seed,
        "object_count": args.object_count,
        "min_objects": args.min_objects,
        "stack_probability": args.stack_probability,
        "max_stack_depth": args.max_stack_depth,
        "require_valid": args.require_valid,
        "min_contacts": args.min_contacts,
        "min_visible_edges": args.min_visible_edges,
        "table_margin": args.table_margin,
        "table_z_margin": args.table_z_margin,
        "max_object_top_z": args.max_object_top_z,
        "scenes": [],
    }

    next_seed = args.seed
    for index in range(args.count):
        scene_path = out_dir / f"scene_{index:04d}.xml"
        if scene_path.exists() and not args.overwrite:
            raise SystemExit(f"Scene already exists: {scene_path}. Use --overwrite to replace it.")

        accepted = False
        for attempt in range(1, args.max_attempts + 1):
            seed = next_seed if args.require_valid else args.seed + index
            next_seed = seed + 1

            layout = sample_random_layout(
                STARTER_OBJECTS,
                seed=seed,
                object_count=args.object_count,
                min_objects=args.min_objects,
                stack_probability=args.stack_probability,
                max_stack_depth=args.max_stack_depth,
            )
            candidate_path = scene_path.with_name(f".{scene_path.stem}.candidate.xml")
            scene_xml = generate_scene(
                dataset_root=args.dataset_root,
                output=candidate_path,
                specs=layout,
                model_name=f"ycb_random_scene_{index:04d}",
            )
            candidate_path.write_text(scene_xml, encoding="utf-8")

            metrics = None
            if args.require_valid:
                metrics = validate_generated_scene(
                    candidate_path,
                    settle_steps=args.settle_steps,
                    table_margin=args.table_margin,
                    table_z_margin=args.table_z_margin,
                )
                if not is_valid_scene_metrics(
                    metrics,
                    min_contacts=args.min_contacts,
                    min_visible_edges=args.min_visible_edges,
                    min_objects=args.min_objects,
                    max_object_top_z=args.max_object_top_z,
                ):
                    print(
                        "rejected candidate: "
                        f"scene={index:04d}, seed={seed}, attempt={attempt}, metrics={asdict(metrics)}"
                    )
                    continue

            candidate_path.replace(scene_path)
            accepted = True
            print(f"saved random scene: {scene_path}")

            manifest["scenes"].append(
                {
                    "index": index,
                    "seed": seed,
                    "attempt": attempt,
                    "path": to_project_relative(scene_path),
                    "validation": asdict(metrics) if metrics is not None else None,
                    "objects": [
                        {
                            "name": spec.name,
                            "pos_xy": [spec.pos[0], spec.pos[1]],
                            "support": spec.support,
                        }
                        for spec in layout
                    ],
                }
            )
            break

        if not accepted:
            raise SystemExit(
                f"Could not generate valid scene {index:04d} after {args.max_attempts} attempts. "
                "Lower --min-contacts/--min-visible-edges or increase --max-attempts."
            )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
