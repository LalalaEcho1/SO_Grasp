from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests import conftest  # noqa: F401
from scripts.run_experiments import resolve_scene_paths, run_experiments, step_rows, summarize_episode
from stacked_grasping.gripper.feasibility import GraspCandidate, ObjectGraspFeasibility
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.planning.adaptive_score import ObjectScore
from stacked_grasping.planning.episode import EpisodeResult, EpisodeStep
from stacked_grasping.relations.geometry import ObjectState


def _score(name: str, blocked: float = 0.2) -> ObjectScore:
    return ObjectScore(
        name=name,
        score=1.0 - blocked,
        graspability_prior=1.0 - blocked,
        blocked_by_od=blocked,
        support_risk=0.1,
        contact_risk=0.2,
        clearance_gain=0.3,
    )


def _feasibility(name: str, feasible: bool, feasible_count: int) -> ObjectGraspFeasibility:
    def _pose(axis: str, candidate_feasible: bool, reason: str | None) -> GraspPoseCandidate:
        return GraspPoseCandidate(
            object_name=name,
            generator="rule-topdown",
            position=np.array([0.0, 0.0, 0.5], dtype=float),
            pregrasp_position=np.array([0.0, 0.0, 0.62], dtype=float),
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=float),
            closing_axis=axis,
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            required_opening=0.04,
            feasible=candidate_feasible,
            failure_reason=reason,
        )

    return ObjectGraspFeasibility(
        object_name=name,
        feasible=feasible,
        feasible_grasp_count=feasible_count,
        candidates=[
            GraspCandidate(
                name,
                "x",
                0.04,
                feasible_count >= 1,
                None if feasible_count >= 1 else "opening-too-small",
                [],
                pose=_pose("x", feasible_count >= 1, None if feasible_count >= 1 else "opening-too-small"),
            ),
            GraspCandidate(
                name,
                "y",
                0.04,
                feasible_count >= 2,
                None if feasible_count >= 2 else "finger-collision",
                ["blocker"],
                pose=_pose("y", feasible_count >= 2, None if feasible_count >= 2 else "finger-collision"),
            ),
        ],
    )


class ExperimentSummaryTests(unittest.TestCase):
    def test_summarize_episode_aggregates_selected_metrics(self):
        step = EpisodeStep(
            step_index=1,
            selected_object="obj_a",
            remaining_objects_before=["obj_a", "obj_b"],
            contact_pairs_before=[("obj_a", "obj_b")],
            ranking_before=[_score("obj_a", blocked=0.25), _score("obj_b", blocked=0.5)],
            edges_before=[],
            gripper_feasibility=_feasibility("obj_a", feasible=True, feasible_count=1),
        )
        result = EpisodeResult(policy="adaptive-score", steps=[step], final_objects=[])

        row = summarize_episode(
            result=result,
            scene="scene.xml",
            policy="adaptive-score",
            trial_index=0,
            seed=42,
            planning_time_sec=0.125,
            candidate_source="rule",
        )

        self.assertEqual(row["policy"], "adaptive-score")
        self.assertEqual(row["candidate_source"], "rule")
        self.assertEqual(row["num_initial_objects"], 2)
        self.assertEqual(row["num_steps"], 1)
        self.assertEqual(row["num_failures"], 0)
        self.assertEqual(row["success_rate"], 1.0)
        self.assertEqual(row["cleared_objects"], 1)
        self.assertEqual(row["remaining_objects"], 0)
        self.assertEqual(row["clearance_rate"], 0.5)
        self.assertEqual(row["mean_selected_blocked_by_od"], 0.25)
        self.assertEqual(row["num_gripper_infeasible_steps"], 0)
        self.assertEqual(row["mean_selected_gripper_collision_risk"], 0.5)
        self.assertEqual(row["max_selected_gripper_collision_risk"], 0.5)
        self.assertEqual(row["planning_time_sec"], 0.125)

        step_row = step_rows(result, policy="adaptive-score", trial_index=0, candidate_source="rule")[0]
        self.assertEqual(step_row["candidate_source"], "rule")
        self.assertTrue(step_row["gripper_feasible"])
        self.assertEqual(step_row["gripper_candidate_count"], 2)
        self.assertEqual(step_row["gripper_feasible_grasp_count"], 1)
        self.assertEqual(step_row["gripper_collision_risk"], 0.5)
        self.assertEqual(step_row["selected_grasp_generator"], "rule-topdown")
        self.assertEqual(step_row["selected_grasp_closing_axis"], "x")

    def test_summarize_episode_reports_failed_step_and_risk_metrics(self):
        step = EpisodeStep(
            step_index=1,
            selected_object="obj_a",
            remaining_objects_before=["obj_a", "obj_b"],
            contact_pairs_before=[],
            ranking_before=[_score("obj_a", blocked=0.5), _score("obj_b", blocked=0.1)],
            edges_before=[],
            grasp_success=False,
            grasp_risk=0.62,
            failure_reason="risk-threshold",
            gripper_feasibility=_feasibility("obj_a", feasible=False, feasible_count=0),
        )
        result = EpisodeResult(policy="random", steps=[step], final_objects=[])

        row = summarize_episode(
            result=result,
            scene="scene.xml",
            policy="random",
            trial_index=0,
            seed=42,
            planning_time_sec=0.125,
            candidate_source="rule",
        )

        self.assertEqual(row["num_failures"], 1)
        self.assertEqual(row["first_failure_step"], 1)
        self.assertEqual(row["success_rate"], 0.0)
        self.assertEqual(row["cleared_objects"], 0)
        self.assertEqual(row["mean_grasp_risk"], 0.62)
        self.assertEqual(row["max_grasp_risk"], 0.62)
        self.assertEqual(row["num_gripper_infeasible_steps"], 1)
        self.assertEqual(row["mean_selected_gripper_collision_risk"], 1.0)

    def test_resolve_scene_paths_reads_sorted_scene_directory(self):
        paths = resolve_scene_paths(scene=None, scene_dir=conftest.PROJECT_ROOT / "assets" / "scenes")

        self.assertEqual(
            [path.name for path in paths],
            ["stacked_blocks.xml", "ycb_lite_stacked.xml", "ycb_mesh_stacked.xml"],
        )

    def test_run_experiments_can_use_graspnet_prediction_candidates(self):
        class Scene:
            def __init__(self, path):
                self.objects = [
                    ObjectState(
                        name="obj_a",
                        body_id=1,
                        geom_id=1,
                        geom_type="box",
                        position=np.array([0.0, 0.0, 0.1], dtype=float),
                        half_extents=np.array([0.02, 0.02, 0.05], dtype=float),
                    )
                ]

            def reset_and_settle(self, steps):
                pass

            def read_objects(self):
                return list(self.objects)

            def read_object_contact_pairs(self):
                return set()

            def remove_object(self, name):
                self.objects = []

            def settle(self, steps):
                pass

        pose = GraspPoseCandidate(
            object_name="obj_a",
            generator="graspnet-prediction",
            position=np.array([0.0, 0.0, 0.15], dtype=float),
            pregrasp_position=np.array([0.0, 0.0, 0.27], dtype=float),
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=float),
            closing_axis="x",
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            required_opening=0.04,
            score=0.9,
        )

        with patch("scripts.run_experiments.MujocoStackedScene", Scene), patch(
            "scripts.run_experiments.load_scene_prediction_candidates",
            return_value={"obj_a": [pose]},
        ) as loader:
            episode_rows, step_records, _ = run_experiments(
                scene_paths=[Path("assets/scenes/generated_main_v1/scene_0000.xml")],
                policies=["highest-first"],
                trials=1,
                seed=1,
                settle_steps=1,
                post_grasp_steps=0,
                max_steps=1,
                candidate_source="graspnet-prediction",
                graspnet_prediction_root=Path("results/graspnet_predictions"),
                graspnet_input_root=Path("results/graspnet_inputs"),
            )

        loader.assert_called_once()
        self.assertEqual(episode_rows[0]["candidate_source"], "graspnet-prediction")
        self.assertEqual(step_records[0]["selected_grasp_generator"], "graspnet-prediction")


if __name__ == "__main__":
    unittest.main()
