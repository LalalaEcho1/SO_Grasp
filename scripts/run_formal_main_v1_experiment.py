from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_experiments import EPISODE_FIELDS, run_experiments
from stacked_grasping.utils.paths import path_key, resolve_project_path, to_project_relative


DEFAULT_POLICIES = [
    "adaptive-score-v2-gripper",
    "adaptive-score-v2",
    "adaptive-score",
    "highest-first",
    "od-only",
    "lowest-blocked",
    "random",
]
FORMAL_METADATA_FIELDS = [
    "scene_index",
    "scene_name",
    "difficulty",
    "contact_pairs",
    "visible_edges",
    "max_top_z",
    "risk_threshold",
]
FORMAL_EPISODE_FIELDS = FORMAL_METADATA_FIELDS + EPISODE_FIELDS
FORMAL_SUMMARY_FIELDS = [
    "difficulty",
    "candidate_source",
    "policy",
    "episodes",
    "avg_clearance_rate",
    "avg_success_rate",
    "total_failures",
    "avg_failures",
    "avg_mean_grasp_risk",
    "avg_max_grasp_risk",
    "total_gripper_infeasible_steps",
    "avg_mean_gripper_collision_risk",
    "avg_max_gripper_collision_risk",
    "avg_steps",
    "avg_planning_time_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal 100-scene main-v1 grasp-order experiment.")
    parser.add_argument(
        "--difficulty-splits",
        type=Path,
        default=PROJECT_ROOT / "assets" / "scenes" / "generated_main_v1" / "difficulty_splits.json",
        help="Difficulty split JSON generated with the scene set.",
    )
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--settle-steps", type=int, default=800)
    parser.add_argument("--post-grasp-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--risk-threshold", type=float, default=0.35)
    parser.add_argument("--candidate-source", default="rule", help="Candidate source label recorded in outputs.")
    parser.add_argument("--graspnet-prediction-root", type=Path, default=None, help="Root containing GraspNet prediction files.")
    parser.add_argument("--graspnet-input-root", type=Path, default=None, help="Root containing exported GraspNet input metadata.")
    parser.add_argument("--graspnet-camera", default="realsense")
    parser.add_argument("--graspnet-view-id", default=0)
    parser.add_argument("--graspnet-assign-margin", type=float, default=0.0)
    parser.add_argument("--limit-scenes", type=int, default=None, help="Optional smoke-test limit over the ordered split.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "experiments" / "formal_main_v1")
    parser.add_argument("--no-save", action="store_true", help="Run and print summaries without writing result files.")
    return parser.parse_args()


def load_difficulty_scene_records(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    groups = payload.get("groups", {})
    ordered_difficulties = [name for name in ["easy", "medium", "hard"] if name in groups]
    ordered_difficulties.extend(name for name in groups if name not in ordered_difficulties)

    records: List[Dict[str, object]] = []
    for difficulty in ordered_difficulties:
        for item in groups[difficulty].get("scenes", []):
            records.append(
                {
                    "scene_index": int(item["index"]),
                    "scene_name": str(item["scene"]),
                    "path": to_project_relative(item["path"]),
                    "difficulty": difficulty,
                    "contact_pairs": int(item["contact_pairs"]),
                    "visible_edges": int(item["visible_edges"]),
                    "max_top_z": float(item["max_top_z"]),
                }
            )

    if not records:
        raise ValueError(f"No scene records found in: {path}")
    return records


def annotate_episode_rows(
    rows: Sequence[Dict[str, object]],
    scene_records: Sequence[Dict[str, object]],
    risk_threshold: float,
) -> List[Dict[str, object]]:
    metadata_by_path = {path_key(record["path"]): record for record in scene_records}
    annotated = []
    for row in rows:
        metadata = metadata_by_path[path_key(row["scene"])]
        combined = dict(row)
        combined.update(
            {
                "scene_index": metadata["scene_index"],
                "scene_name": metadata.get("scene_name", metadata.get("scene")),
                "difficulty": metadata["difficulty"],
                "contact_pairs": metadata["contact_pairs"],
                "visible_edges": metadata["visible_edges"],
                "max_top_z": metadata["max_top_z"],
                "risk_threshold": risk_threshold,
            }
        )
        annotated.append(combined)
    return annotated


def aggregate_formal_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows = []
    summary_rows.extend(_aggregate_group(rows, difficulty="overall"))
    for difficulty in ["easy", "medium", "hard"]:
        group_rows = [row for row in rows if row.get("difficulty") == difficulty]
        if group_rows:
            summary_rows.extend(_aggregate_group(group_rows, difficulty=difficulty))
    return summary_rows


def run_formal_experiment(
    scene_records: Sequence[Dict[str, object]],
    policies: Sequence[str],
    trials: int,
    seed: int,
    settle_steps: int,
    post_grasp_steps: int,
    max_steps: int | None,
    risk_threshold: float,
    candidate_source: str = "rule",
    graspnet_prediction_root: Path | None = None,
    graspnet_input_root: Path | None = None,
    graspnet_camera: str = "realsense",
    graspnet_view_id: int | str = 0,
    graspnet_assign_margin: float = 0.0,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    scene_paths = [resolve_project_path(str(record["path"])) for record in scene_records]
    episode_rows, _, _ = run_experiments(
        scene_paths=scene_paths,
        policies=policies,
        trials=trials,
        seed=seed,
        settle_steps=settle_steps,
        post_grasp_steps=post_grasp_steps,
        max_steps=max_steps,
        failure_mode="risk-threshold",
        risk_threshold=risk_threshold,
        candidate_source=candidate_source,
        graspnet_prediction_root=graspnet_prediction_root,
        graspnet_input_root=graspnet_input_root,
        graspnet_camera=graspnet_camera,
        graspnet_view_id=graspnet_view_id,
        graspnet_assign_margin=graspnet_assign_margin,
    )
    annotated_rows = annotate_episode_rows(episode_rows, scene_records, risk_threshold=risk_threshold)
    return annotated_rows, aggregate_formal_rows(annotated_rows)


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scene_records = load_difficulty_scene_records(args.difficulty_splits)
    if args.limit_scenes is not None:
        scene_records = scene_records[: args.limit_scenes]
    candidate_source = args.candidate_source
    if args.graspnet_prediction_root is not None and candidate_source == "rule":
        candidate_source = "graspnet-prediction"

    episode_rows, summary_rows = run_formal_experiment(
        scene_records=scene_records,
        policies=args.policies,
        trials=args.trials,
        seed=args.seed,
        settle_steps=args.settle_steps,
        post_grasp_steps=args.post_grasp_steps,
        max_steps=args.max_steps,
        risk_threshold=args.risk_threshold,
        candidate_source=candidate_source,
        graspnet_prediction_root=args.graspnet_prediction_root,
        graspnet_input_root=args.graspnet_input_root,
        graspnet_camera=args.graspnet_camera,
        graspnet_view_id=args.graspnet_view_id,
        graspnet_assign_margin=args.graspnet_assign_margin,
    )

    if not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
        episodes_path = out_dir / f"formal_main_v1_{timestamp}_episodes.csv"
        summary_path = out_dir / f"formal_main_v1_{timestamp}_summary.csv"
        json_path = out_dir / f"formal_main_v1_{timestamp}.json"
        write_csv(episodes_path, episode_rows, FORMAL_EPISODE_FIELDS)
        write_csv(summary_path, summary_rows, FORMAL_SUMMARY_FIELDS)
        json_path.write_text(
            json.dumps(
                {
                    "difficulty_splits": to_project_relative(args.difficulty_splits),
                    "scene_count": len(scene_records),
                    "policies": list(args.policies),
                    "trials": args.trials,
                    "seed": args.seed,
                    "settle_steps": args.settle_steps,
                    "post_grasp_steps": args.post_grasp_steps,
                    "max_steps": args.max_steps,
                    "failure_mode": "risk-threshold",
                    "risk_threshold": args.risk_threshold,
                    "candidate_source": candidate_source,
                    "episodes": episode_rows,
                    "summary": summary_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved formal episodes CSV: {episodes_path}")
        print(f"saved formal summary CSV: {summary_path}")
        print(f"saved formal JSON: {json_path}")

    print_formal_summary(summary_rows)


def print_formal_summary(rows: Sequence[Dict[str, object]]) -> None:
    print("\nFormal main-v1 experiment summary")
    for row in rows:
        print(
            f"  difficulty={row['difficulty']} "
            f"source={row.get('candidate_source', 'unknown')} "
            f"policy={row['policy']} "
            f"episodes={row['episodes']} "
            f"clearance={float(row['avg_clearance_rate']):.3f} "
            f"success={float(row['avg_success_rate']):.3f} "
            f"failures={int(row['total_failures'])} "
            f"gripper_infeasible={int(row['total_gripper_infeasible_steps'])}"
        )


def _aggregate_group(rows: Sequence[Dict[str, object]], difficulty: str) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("candidate_source", "rule")), str(row["policy"]))].append(row)

    summary_rows = []
    for candidate_source, policy in sorted(grouped):
        policy_rows = grouped[(candidate_source, policy)]
        count = len(policy_rows)
        failures = sum(int(row["num_failures"]) for row in policy_rows)
        gripper_infeasible = sum(int(row.get("num_gripper_infeasible_steps", 0)) for row in policy_rows)
        summary_rows.append(
            {
                "difficulty": difficulty,
                "candidate_source": candidate_source,
                "policy": policy,
                "episodes": count,
                "avg_clearance_rate": _mean(policy_rows, "clearance_rate"),
                "avg_success_rate": _mean(policy_rows, "success_rate"),
                "total_failures": failures,
                "avg_failures": round(failures / count, 6),
                "avg_mean_grasp_risk": _mean(policy_rows, "mean_grasp_risk"),
                "avg_max_grasp_risk": _mean(policy_rows, "max_grasp_risk"),
                "total_gripper_infeasible_steps": gripper_infeasible,
                "avg_mean_gripper_collision_risk": _mean(policy_rows, "mean_selected_gripper_collision_risk"),
                "avg_max_gripper_collision_risk": _mean(policy_rows, "max_selected_gripper_collision_risk"),
                "avg_steps": _mean(policy_rows, "num_steps"),
                "avg_planning_time_sec": _mean(policy_rows, "planning_time_sec"),
            }
        )
    return summary_rows


def _mean(rows: Sequence[Dict[str, object]], key: str) -> float:
    return round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)
if __name__ == "__main__":
    main()
