from __future__ import annotations

import random
import unittest

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.feasibility import GraspCandidate, ObjectGraspFeasibility
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.planning.episode import run_policy_episode
from stacked_grasping.planning.policies import VALID_POLICIES, select_object
from stacked_grasping.relations.geometry import ObjectState
from stacked_grasping.relations.graph import RelationGraph


def _obj(name: str, z: float) -> ObjectState:
    return ObjectState(
        name=name,
        body_id=0,
        geom_id=0,
        geom_type="box",
        position=np.array([0.0, 0.0, z], dtype=float),
        half_extents=np.array([0.02, 0.02, 0.05], dtype=float),
    )


def _candidate(name: str, candidate_index: int, *, score: float, tolerance: float) -> GraspCandidate:
    pose = GraspPoseCandidate(
        object_name=name,
        generator="graspnet-pointcloud",
        position=np.array([0.0, 0.0, 0.1], dtype=float),
        pregrasp_position=np.array([0.0, 0.0, 0.2], dtype=float),
        approach_direction=np.array([0.0, 0.0, -1.0], dtype=float),
        closing_axis="6d",
        orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        required_opening=0.04,
        score=score,
        grasp_tolerance=tolerance,
        candidate_index=candidate_index,
    )
    return GraspCandidate(name, "6d", 0.04, True, None, [], pose=pose)


def _feasibility(name: str, candidates: list[GraspCandidate]) -> ObjectGraspFeasibility:
    return ObjectGraspFeasibility(
        object_name=name,
        feasible=bool(candidates),
        feasible_grasp_count=len(candidates),
        candidates=candidates,
    )


class OneObjectScene:
    def __init__(self):
        self.objects = [_obj("obj", 0.3)]

    def read_objects(self):
        return list(self.objects)

    def read_object_contact_pairs(self):
        return set()

    def remove_object(self, name: str):
        self.objects = [obj for obj in self.objects if obj.name != name]

    def settle(self, steps: int):
        pass


class V3CandidatePolicyTests(unittest.TestCase):
    def test_adaptive_score_v3_exports_ranked_candidate_indices_by_score(self):
        graph = RelationGraph(objects=[_obj("obj", 0.3)], edges=[])
        low_score_high_tolerance = _candidate("obj", 1, score=0.2, tolerance=0.9)
        high_score_low_tolerance = _candidate("obj", 2, score=0.8, tolerance=0.1)

        decision = select_object(
            "adaptive-score-v3-candidate",
            graph,
            random.Random(1),
            gripper_feasibilities=[
                _feasibility("obj", [low_score_high_tolerance, high_score_low_tolerance])
            ],
        )

        self.assertIn("adaptive-score-v3-candidate", VALID_POLICIES)
        self.assertEqual(decision.selected_candidate_index, 2)
        self.assertEqual(decision.ranking[0]["selected_candidate_index"], 2)
        self.assertEqual(decision.ranking[0]["candidate_indices"], [2, 1])

    def test_adaptive_score_v2_graspnet_prefers_higher_tolerance(self):
        graph = RelationGraph(objects=[_obj("low_quality_high", 0.4), _obj("high_quality_low", 0.2)], edges=[])

        decision = select_object(
            "adaptive-score-v2-graspnet",
            graph,
            random.Random(1),
            gripper_feasibilities=[
                _feasibility("low_quality_high", [_candidate("low_quality_high", 1, score=0.9, tolerance=0.01)]),
                _feasibility("high_quality_low", [_candidate("high_quality_low", 2, score=0.3, tolerance=0.08)]),
            ],
        )

        self.assertEqual(decision.selected_object, "high_quality_low")
        self.assertEqual(decision.ranking[0]["graspnet_best_tolerance"], 0.08)
        self.assertEqual(decision.ranking[0]["graspnet_quality_score"], 0.08)

    def test_graspnet_score_first_uses_prediction_score_before_height(self):
        graph = RelationGraph(objects=[_obj("high_low_score", 0.5), _obj("low_high_score", 0.1)], edges=[])

        decision = select_object(
            "graspnet-score-first",
            graph,
            random.Random(1),
            gripper_feasibilities=[
                _feasibility("high_low_score", [_candidate("high_low_score", 1, score=0.2, tolerance=0.2)]),
                _feasibility("low_high_score", [_candidate("low_high_score", 2, score=0.9, tolerance=0.1)]),
            ],
        )

        self.assertEqual(decision.selected_object, "low_high_score")

    def test_episode_selected_pose_follows_policy_candidate_index(self):
        scene = OneObjectScene()
        low_score_high_tolerance = _candidate("obj", 1, score=0.2, tolerance=0.9)
        high_score_low_tolerance = _candidate("obj", 2, score=0.8, tolerance=0.1)

        def provider(objects):
            return [_feasibility("obj", [low_score_high_tolerance, high_score_low_tolerance])]

        result = run_policy_episode(
            scene,
            policy="adaptive-score-v3-candidate",
            max_steps=1,
            post_grasp_settle_steps=0,
            seed=1,
            failure_mode="risk-threshold",
            risk_threshold=0.45,
            gripper_feasibility_provider=provider,
        )

        self.assertEqual(result.steps[0].policy_decision.selected_candidate_index, 2)
        self.assertEqual(result.steps[0].selected_grasp_pose.candidate_index, 2)


if __name__ == "__main__":
    unittest.main()
