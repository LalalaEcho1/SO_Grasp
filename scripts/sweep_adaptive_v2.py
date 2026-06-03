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

from scripts.run_experiments import run_experiments
from stacked_grasping.utils.paths import resolve_project_path, to_project_relative


DEFAULT_POLICIES = [
    "adaptive-score-v2-gripper",
    "adaptive-score-v2",
    "adaptive-score",
    "highest-first",
    "od-only",
    "lowest-blocked",
    "random",
]
SUMMARY_FIELDS = [
    "risk_threshold",
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep risk thresholds for adaptive-score-v2 evaluation.")
    parser.add_argument(
        "--selected-scenes",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "experiments"
        / "preliminary_main_v1"
        / "preliminary_20260527-105515_selected_scenes.json",
        help="JSON list of selected scenes. Supports the previous preliminary selected-scene format.",
    )
    parser.add_argument("--thresholds", nargs="+", default=["0.25", "0.30", "0.35", "0.40", "0.45"])
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--settle-steps", type=int, default=800)
    parser.add_argument("--post-grasp-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "experiments" / "adaptive_v2_sweep")
    parser.add_argument("--save", action="store_true", help="Write sweep CSV/JSON files.")
    parser.add_argument("--no-save", action="store_true", help="Compatibility flag; sweep results are not saved by default.")
    return parser.parse_args()


def parse_thresholds(values: Sequence[str]) -> List[float]:
    thresholds = [float(value) for value in values]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def load_selected_scene_paths(path: Path) -> List[Path]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        items = payload.get("selected_scenes", [])
    else:
        items = payload

    paths = []
    for item in items:
        if isinstance(item, dict):
            paths.append(resolve_project_path(str(item["path"])))
        else:
            paths.append(resolve_project_path(str(item)))
    if not paths:
        raise ValueError(f"No scene paths found in: {path}")
    return paths


def aggregate_policy_rows(rows: Sequence[Dict[str, object]], risk_threshold: float) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)

    summaries = []
    for policy in sorted(grouped):
        policy_rows = grouped[policy]
        count = len(policy_rows)
        failures = sum(int(row["num_failures"]) for row in policy_rows)
        gripper_infeasible = sum(int(row.get("num_gripper_infeasible_steps", 0)) for row in policy_rows)
        summaries.append(
            {
                "risk_threshold": risk_threshold,
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
            }
        )
    return summaries


def run_threshold_sweep(
    scene_paths: Sequence[Path],
    thresholds: Sequence[float],
    policies: Sequence[str],
    trials: int,
    seed: int,
    settle_steps: int,
    post_grasp_steps: int,
    max_steps: int | None,
) -> List[Dict[str, object]]:
    summary_rows = []
    for threshold in thresholds:
        episode_rows, _, _ = run_experiments(
            scene_paths=scene_paths,
            policies=policies,
            trials=trials,
            seed=seed,
            settle_steps=settle_steps,
            post_grasp_steps=post_grasp_steps,
            max_steps=max_steps,
            failure_mode="risk-threshold",
            risk_threshold=threshold,
        )
        summary_rows.extend(aggregate_policy_rows(episode_rows, risk_threshold=threshold))
    return summary_rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    scene_paths = load_selected_scene_paths(args.selected_scenes)
    rows = run_threshold_sweep(
        scene_paths=scene_paths,
        thresholds=thresholds,
        policies=args.policies,
        trials=args.trials,
        seed=args.seed,
        settle_steps=args.settle_steps,
        post_grasp_steps=args.post_grasp_steps,
        max_steps=args.max_steps,
    )

    if args.save and args.no_save:
        raise SystemExit("--save and --no-save cannot be used together.")

    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
        csv_path = out_dir / f"adaptive_v2_sweep_{timestamp}.csv"
        json_path = out_dir / f"adaptive_v2_sweep_{timestamp}.json"
        write_csv(csv_path, rows)
        json_path.write_text(
            json.dumps(
                {
                    "selected_scenes": [to_project_relative(path) for path in scene_paths],
                    "thresholds": thresholds,
                    "policies": list(args.policies),
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved sweep CSV: {csv_path}")
        print(f"saved sweep JSON: {json_path}")

    print_sweep_summary(rows)


def print_sweep_summary(rows: Sequence[Dict[str, object]]) -> None:
    print("\nAdaptive-score-v2 threshold sweep")
    for row in rows:
        print(
            f"  threshold={float(row['risk_threshold']):.2f} "
            f"policy={row['policy']} "
            f"clearance={float(row['avg_clearance_rate']):.3f} "
            f"success={float(row['avg_success_rate']):.3f} "
            f"failures={int(row['total_failures'])} "
            f"gripper_infeasible={int(row['total_gripper_infeasible_steps'])}"
        )


def _mean(rows: Sequence[Dict[str, object]], key: str) -> float:
    return round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)


if __name__ == "__main__":
    main()
