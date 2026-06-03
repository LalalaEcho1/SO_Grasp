from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.env.mujoco_scene import MujocoStackedScene
from stacked_grasping.gripper.graspnet_predictions import load_scene_prediction_candidates
from stacked_grasping.planning.episode import EpisodeResult, EpisodeStep, run_policy_episode
from stacked_grasping.planning.policies import VALID_POLICIES
from stacked_grasping.utils.paths import to_project_relative


EPISODE_FIELDS = [
    "policy",
    "trial_index",
    "scene",
    "candidate_source",
    "seed",
    "failure_mode",
    "num_initial_objects",
    "num_steps",
    "num_successes",
    "num_failures",
    "first_failure_step",
    "success_rate",
    "cleared_objects",
    "remaining_objects",
    "clearance_rate",
    "mean_grasp_risk",
    "max_grasp_risk",
    "num_gripper_infeasible_steps",
    "mean_selected_gripper_collision_risk",
    "max_selected_gripper_collision_risk",
    "episode_failure_reason",
    "total_selected_blocked_by_od",
    "mean_selected_blocked_by_od",
    "total_selected_support_risk",
    "mean_selected_support_risk",
    "total_selected_contact_risk",
    "mean_selected_contact_risk",
    "total_selected_clearance_gain",
    "planning_time_sec",
]

STEP_FIELDS = [
    "policy",
    "trial_index",
    "candidate_source",
    "step_index",
    "selected_object",
    "remaining_before",
    "selected_score",
    "selected_blocked_by_od",
    "selected_support_risk",
    "selected_contact_risk",
    "selected_clearance_gain",
    "gripper_feasible",
    "gripper_candidate_count",
    "gripper_feasible_grasp_count",
    "gripper_collision_risk",
    "selected_grasp_generator",
    "selected_grasp_closing_axis",
    "selected_grasp_score",
    "selected_grasp_required_opening",
    "grasp_success",
    "grasp_risk",
    "failure_reason",
    "contact_pair_count",
    "edge_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch grasp-order baseline experiments.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=PROJECT_ROOT / "assets" / "scenes" / "ycb_mesh_stacked.xml",
        help="Path to a MuJoCo XML scene.",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=None,
        help="Directory containing generated MuJoCo XML scenes. If set, all *.xml files are evaluated.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "adaptive-score",
            "adaptive-score-v2",
            "adaptive-score-v2-gripper",
            "od-only",
            "highest-first",
            "lowest-blocked",
            "random",
        ],
        help=f"Policies to evaluate. Valid: {', '.join(VALID_POLICIES)}",
    )
    parser.add_argument("--trials", type=int, default=1, help="Trials per policy.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--settle-steps", type=int, default=1500, help="Initial physics steps before planning.")
    parser.add_argument("--post-grasp-steps", type=int, default=500, help="Physics steps after each abstract grasp.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop each episode after this many grasps.")
    parser.add_argument(
        "--failure-mode",
        choices=["none", "risk-threshold"],
        default="none",
        help="Optional abstract failure model for high-risk grasps.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.45,
        help="Risk threshold used when --failure-mode risk-threshold is enabled.",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "experiments")
    parser.add_argument("--no-save", action="store_true", help="Do not save experiment files.")
    parser.add_argument("--candidate-source", default="rule", help="Candidate source label recorded in outputs.")
    parser.add_argument("--graspnet-prediction-root", type=Path, default=None, help="Root containing GraspNet prediction files.")
    parser.add_argument("--graspnet-input-root", type=Path, default=None, help="Root containing exported GraspNet input metadata.")
    parser.add_argument("--graspnet-camera", default="realsense")
    parser.add_argument("--graspnet-view-id", default=0)
    parser.add_argument("--graspnet-assign-margin", type=float, default=0.0)
    return parser.parse_args()


def summarize_episode(
    result: EpisodeResult,
    scene: str,
    policy: str,
    trial_index: int,
    seed: int,
    planning_time_sec: float,
    candidate_source: str = "rule",
) -> Dict[str, object]:
    selected_scores = [step.selected_score for step in result.steps]
    successful_steps = [step for step in result.steps if step.grasp_success]
    failed_steps = [step for step in result.steps if not step.grasp_success]
    risks = [step.grasp_risk for step in result.steps]
    gripper_risks = [step.gripper_collision_risk for step in result.steps if step.gripper_feasible is not None]
    gripper_infeasible = [step for step in result.steps if step.gripper_feasible is False]
    num_initial_objects = len(result.steps[0].remaining_objects_before) if result.steps else len(result.final_objects)
    cleared_objects = len(successful_steps)
    remaining_objects = len(result.final_objects)
    denominator = max(num_initial_objects, 1)
    first_failure_step = failed_steps[0].step_index if failed_steps else None

    total_blocked = sum(score.blocked_by_od for score in selected_scores)
    total_support = sum(score.support_risk for score in selected_scores)
    total_contact = sum(score.contact_risk for score in selected_scores)
    total_clearance = sum(score.clearance_gain for score in selected_scores)
    step_count = max(len(selected_scores), 1)
    risk_count = max(len(risks), 1)
    gripper_risk_count = max(len(gripper_risks), 1)

    return {
        "policy": policy,
        "trial_index": trial_index,
        "scene": scene,
        "candidate_source": candidate_source,
        "seed": seed,
        "failure_mode": result.failure_mode,
        "num_initial_objects": num_initial_objects,
        "num_steps": len(result.steps),
        "num_successes": len(successful_steps),
        "num_failures": len(failed_steps),
        "first_failure_step": first_failure_step,
        "success_rate": round(len(successful_steps) / max(len(result.steps), 1), 6),
        "cleared_objects": cleared_objects,
        "remaining_objects": remaining_objects,
        "clearance_rate": round(cleared_objects / denominator, 6),
        "mean_grasp_risk": round(sum(risks) / risk_count, 6),
        "max_grasp_risk": round(max(risks) if risks else 0.0, 6),
        "num_gripper_infeasible_steps": len(gripper_infeasible),
        "mean_selected_gripper_collision_risk": round(sum(gripper_risks) / gripper_risk_count, 6),
        "max_selected_gripper_collision_risk": round(max(gripper_risks) if gripper_risks else 0.0, 6),
        "episode_failure_reason": result.failure_reason,
        "total_selected_blocked_by_od": round(total_blocked, 6),
        "mean_selected_blocked_by_od": round(total_blocked / step_count, 6),
        "total_selected_support_risk": round(total_support, 6),
        "mean_selected_support_risk": round(total_support / step_count, 6),
        "total_selected_contact_risk": round(total_contact, 6),
        "mean_selected_contact_risk": round(total_contact / step_count, 6),
        "total_selected_clearance_gain": round(total_clearance, 6),
        "planning_time_sec": round(planning_time_sec, 6),
    }


def step_rows(result: EpisodeResult, policy: str, trial_index: int, candidate_source: str = "rule") -> List[Dict[str, object]]:
    return [_step_row(step, policy, trial_index, candidate_source) for step in result.steps]


def resolve_scene_paths(scene: Path | None, scene_dir: Path | None) -> List[Path]:
    if scene_dir is not None:
        root = scene_dir if scene_dir.is_absolute() else PROJECT_ROOT / scene_dir
        if not root.exists():
            raise SystemExit(f"Scene directory not found: {root}")
        paths = sorted(path for path in root.glob("*.xml") if path.is_file())
        if not paths:
            raise SystemExit(f"No XML scenes found in: {root}")
        return paths

    if scene is None:
        raise SystemExit("Either --scene or --scene-dir must be provided.")
    scene_path = scene if scene.is_absolute() else PROJECT_ROOT / scene
    if not scene_path.exists():
        raise SystemExit(f"Scene not found: {scene_path}")
    return [scene_path]


def run_experiments(
    scene_paths: Sequence[Path],
    policies: Sequence[str],
    trials: int,
    seed: int,
    settle_steps: int,
    post_grasp_steps: int,
    max_steps: int | None,
    failure_mode: str = "none",
    risk_threshold: float = 0.45,
    candidate_source: str = "rule",
    graspnet_prediction_root: Path | None = None,
    graspnet_input_root: Path | None = None,
    graspnet_camera: str = "realsense",
    graspnet_view_id: int | str = 0,
    graspnet_assign_margin: float = 0.0,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    _validate_policies(policies)
    episode_rows: List[Dict[str, object]] = []
    all_step_rows: List[Dict[str, object]] = []
    full_runs: List[Dict[str, object]] = []

    for scene_index, scene_path in enumerate(scene_paths):
        for policy in policies:
            for trial_index in range(trials):
                trial_seed = seed + scene_index * 10000 + trial_index
                scene = MujocoStackedScene(scene_path)
                scene.reset_and_settle(settle_steps)
                grasp_poses_by_object = None
                if graspnet_prediction_root is not None:
                    grasp_poses_by_object = load_scene_prediction_candidates(
                        graspnet_prediction_root,
                        scene_path,
                        objects=scene.read_objects(),
                        metadata_root=graspnet_input_root,
                        camera=graspnet_camera,
                        view_id=graspnet_view_id,
                        assign_margin=graspnet_assign_margin,
                    )

                started = time.perf_counter()
                result = run_policy_episode(
                    scene,
                    policy=policy,
                    max_steps=max_steps,
                    post_grasp_settle_steps=post_grasp_steps,
                    seed=trial_seed,
                    failure_mode=failure_mode,
                    risk_threshold=risk_threshold,
                    grasp_poses_by_object=grasp_poses_by_object,
                )
                elapsed = time.perf_counter() - started

                episode_row = summarize_episode(
                    result=result,
                    scene=to_project_relative(scene_path),
                    policy=policy,
                    trial_index=trial_index,
                    seed=trial_seed,
                    planning_time_sec=elapsed,
                    candidate_source=candidate_source,
                )
                episode_rows.append(episode_row)
                all_step_rows.extend(step_rows(result, policy, trial_index, candidate_source=candidate_source))
                full_runs.append(
                    {
                        "scene_index": scene_index,
                        "scene": to_project_relative(scene_path),
                        "policy": policy,
                        "trial_index": trial_index,
                        "seed": trial_seed,
                        "candidate_source": candidate_source,
                        "failure_mode": failure_mode,
                        "risk_threshold": risk_threshold,
                        "episode": result.to_dict(),
                        "summary": episode_row,
                    }
                )

    return episode_rows, all_step_rows, full_runs


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scene_paths = resolve_scene_paths(args.scene, args.scene_dir)
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    candidate_source = args.candidate_source
    if args.graspnet_prediction_root is not None and candidate_source == "rule":
        candidate_source = "graspnet-prediction"

    episode_rows, rows_by_step, full_runs = run_experiments(
        scene_paths=scene_paths,
        policies=args.policies,
        trials=args.trials,
        seed=args.seed,
        settle_steps=args.settle_steps,
        post_grasp_steps=args.post_grasp_steps,
        max_steps=args.max_steps,
        failure_mode=args.failure_mode,
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
        episodes_path = out_dir / f"experiment_{timestamp}_episodes.csv"
        steps_path = out_dir / f"experiment_{timestamp}_steps.csv"
        json_path = out_dir / f"experiment_{timestamp}.json"

        write_csv(episodes_path, episode_rows, EPISODE_FIELDS)
        write_csv(steps_path, rows_by_step, STEP_FIELDS)
        json_path.write_text(
            json.dumps(
                {
                    "scenes": [to_project_relative(path) for path in scene_paths],
                    "policies": list(args.policies),
                    "trials": args.trials,
                    "seed": args.seed,
                    "candidate_source": candidate_source,
                    "settle_steps": args.settle_steps,
                    "post_grasp_steps": args.post_grasp_steps,
                    "max_steps": args.max_steps,
                    "failure_mode": args.failure_mode,
                    "risk_threshold": args.risk_threshold,
                    "episodes": episode_rows,
                    "steps": rows_by_step,
                    "runs": full_runs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved episodes CSV: {episodes_path}")
        print(f"saved steps CSV: {steps_path}")
        print(f"saved JSON: {json_path}")

    print_experiment_summary(episode_rows)


def print_experiment_summary(episode_rows: Sequence[Dict[str, object]]) -> None:
    by_policy: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in episode_rows:
        by_policy[str(row["policy"])].append(row)

    print("\nBatch experiment summary")
    for policy in sorted(by_policy):
        rows = by_policy[policy]
        clearance = sum(float(row["clearance_rate"]) for row in rows) / len(rows)
        success = sum(float(row["success_rate"]) for row in rows) / len(rows)
        failures = sum(float(row["num_failures"]) for row in rows) / len(rows)
        steps = sum(float(row["num_steps"]) for row in rows) / len(rows)
        blocked = sum(float(row["mean_selected_blocked_by_od"]) for row in rows) / len(rows)
        gripper_infeasible = sum(float(row["num_gripper_infeasible_steps"]) for row in rows) / len(rows)
        gripper_risk = sum(float(row["mean_selected_gripper_collision_risk"]) for row in rows) / len(rows)
        print(
            f"  - {policy}: trials={len(rows)}, "
            f"avg_clearance={clearance:.3f}, avg_success={success:.3f}, "
            f"avg_failures={failures:.2f}, avg_steps={steps:.2f}, avg_selected_blocked_od={blocked:.3f}, "
            f"avg_gripper_infeasible={gripper_infeasible:.2f}, avg_gripper_risk={gripper_risk:.3f}"
        )


def _step_row(step: EpisodeStep, policy: str, trial_index: int, candidate_source: str) -> Dict[str, object]:
    selected = step.selected_score
    selected_pose = step.selected_grasp_pose
    candidate_count = len(step.gripper_feasibility.candidates) if step.gripper_feasibility else None
    return {
        "policy": policy,
        "trial_index": trial_index,
        "candidate_source": candidate_source,
        "step_index": step.step_index,
        "selected_object": step.selected_object,
        "remaining_before": len(step.remaining_objects_before),
        "selected_score": round(selected.score, 6),
        "selected_blocked_by_od": round(selected.blocked_by_od, 6),
        "selected_support_risk": round(selected.support_risk, 6),
        "selected_contact_risk": round(selected.contact_risk, 6),
        "selected_clearance_gain": round(selected.clearance_gain, 6),
        "gripper_feasible": step.gripper_feasible,
        "gripper_candidate_count": candidate_count,
        "gripper_feasible_grasp_count": step.gripper_feasible_grasp_count,
        "gripper_collision_risk": round(step.gripper_collision_risk, 6),
        "selected_grasp_generator": selected_pose.generator if selected_pose else None,
        "selected_grasp_closing_axis": selected_pose.closing_axis if selected_pose else None,
        "selected_grasp_score": round(selected_pose.score, 6) if selected_pose else None,
        "selected_grasp_required_opening": round(selected_pose.required_opening, 6) if selected_pose else None,
        "grasp_success": step.grasp_success,
        "grasp_risk": round(step.grasp_risk, 6),
        "failure_reason": step.failure_reason,
        "contact_pair_count": len(step.contact_pairs_before),
        "edge_count": len(step.edges_before),
    }


def _validate_policies(policies: Sequence[str]) -> None:
    unknown = [policy for policy in policies if policy not in VALID_POLICIES]
    if unknown:
        valid = ", ".join(VALID_POLICIES)
        raise SystemExit(f"Unknown policies: {', '.join(unknown)}. Valid policies: {valid}")


if __name__ == "__main__":
    main()
