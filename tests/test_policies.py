from __future__ import annotations

import random
import unittest

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.planning.adaptive_score import ObjectScore
from stacked_grasping.planning.episode import run_policy_episode
from stacked_grasping.planning.policies import VALID_POLICIES, select_object
from stacked_grasping.planning.grasp_risk import GraspRiskConfig, assess_grasp_risk
from stacked_grasping.gripper.feasibility import GraspCandidate, ObjectGraspFeasibility
from stacked_grasping.relations.geometry import ObjectState
from stacked_grasping.relations.graph import EdgeFeatures, RelationGraph


def _obj(name: str, z: float, half=(0.02, 0.02, 0.05), xy=(0.0, 0.0)) -> ObjectState:
    return ObjectState(
        name=name,
        body_id=0,
        geom_id=0,
        geom_type="box",
        position=np.array([xy[0], xy[1], z], dtype=float),
        half_extents=np.array(half, dtype=float),
    )


def _edge(source: str, target: str, od: float, blocked: float = 0.0) -> EdgeFeatures:
    return EdgeFeatures(
        source=source,
        target=target,
        contact=False,
        support_source_to_target=False,
        support_target_to_source=False,
        height_diff=0.0,
        xy_distance=0.0,
        xy_overlap_ratio=0.0,
        vertical_gap=0.0,
        od=od,
        blocked_grasp_ratio=blocked,
        od_mean=od,
        od_max=od,
        blocked_grasp_kinds=[],
        od_prior=0.0,
    )


def _gripper(name: str, feasible: bool, feasible_count: int) -> ObjectGraspFeasibility:
    return ObjectGraspFeasibility(
        object_name=name,
        feasible=feasible,
        feasible_grasp_count=feasible_count,
        candidates=[
            GraspCandidate(name, "x", 0.04, feasible_count >= 1, None if feasible_count >= 1 else "opening-too-small", []),
            GraspCandidate(name, "y", 0.04, feasible_count >= 2, None if feasible_count >= 2 else "finger-collision", ["blocker"]),
        ],
    )


class PolicySelectionTests(unittest.TestCase):
    def test_adaptive_score_v2_prefers_safe_object_over_high_risk_clearance_gain(self):
        graph = RelationGraph(
            objects=[
                _obj("safe", 0.2),
                _obj("risky", 0.3),
                _obj("blocker_a", 0.4),
                _obj("blocker_b", 0.5),
            ],
            edges=[
                _edge("blocker_a", "risky", 0.9),
                _edge("blocker_b", "risky", 0.9),
                _edge("risky", "blocker_a", 1.0),
                _edge("risky", "blocker_b", 1.0),
            ],
        )

        decision = select_object("adaptive-score-v2", graph, random.Random(1))

        self.assertNotEqual(decision.selected_object, "risky")
        self.assertFalse(decision.ranking[0]["high_risk"])
        self.assertTrue(next(item for item in decision.ranking if item["name"] == "risky")["high_risk"])
        self.assertTrue(any(item["high_risk"] for item in decision.ranking))

    def test_adaptive_score_v2_prefers_higher_object_when_risk_is_equal(self):
        graph = RelationGraph(objects=[_obj("low_safe", 0.1), _obj("high_safe", 0.4)], edges=[])

        decision = select_object("adaptive-score-v2", graph, random.Random(1))

        self.assertEqual(decision.selected_object, "high_safe")
        self.assertGreater(decision.ranking[0]["height_priority"], decision.ranking[1]["height_priority"])

    def test_adaptive_score_v2_uses_height_before_clearance_gain_inside_safe_set(self):
        graph = RelationGraph(
            objects=[
                _obj("low_clearance", 0.1),
                _obj("mid_safe", 0.2),
                _obj("top_a", 0.5),
                _obj("top_b", 0.5),
            ],
            edges=[
                _edge("low_clearance", "top_a", 1.0),
                _edge("low_clearance", "top_b", 1.0),
            ],
        )

        decision = select_object("adaptive-score-v2", graph, random.Random(1))

        self.assertNotEqual(decision.selected_object, "low_clearance")
        self.assertGreater(
            decision.ranking[0]["height_priority"],
            next(item for item in decision.ranking if item["name"] == "low_clearance")["height_priority"],
        )

    def test_adaptive_score_v2_gripper_prefers_feasible_object_over_infeasible_high_object(self):
        graph = RelationGraph(
            objects=[
                _obj("narrow_low", 0.1),
                _obj("wide_high", 0.4),
            ],
            edges=[],
        )

        decision = select_object(
            "adaptive-score-v2-gripper",
            graph,
            random.Random(1),
            gripper_feasibilities=[
                _gripper("narrow_low", feasible=True, feasible_count=2),
                _gripper("wide_high", feasible=False, feasible_count=0),
            ],
        )

        self.assertEqual(decision.selected_object, "narrow_low")
        self.assertTrue(decision.ranking[0]["gripper_feasible"])
        self.assertFalse(decision.ranking[1]["gripper_feasible"])
        self.assertEqual(decision.ranking[1]["name"], "wide_high")

    def test_adaptive_score_v2_graspnet_uses_gripper_aware_ranking(self):
        graph = RelationGraph(
            objects=[
                _obj("narrow_low", 0.1),
                _obj("wide_high", 0.4),
            ],
            edges=[],
        )

        decision = select_object(
            "adaptive-score-v2-graspnet",
            graph,
            random.Random(1),
            gripper_feasibilities=[
                _gripper("narrow_low", feasible=True, feasible_count=1),
                _gripper("wide_high", feasible=False, feasible_count=0),
            ],
        )

        self.assertIn("adaptive-score-v2-graspnet", VALID_POLICIES)
        self.assertEqual(decision.policy, "adaptive-score-v2-graspnet")
        self.assertEqual(decision.selected_object, "narrow_low")
        self.assertTrue(decision.ranking[0]["gripper_feasible"])
        self.assertFalse(decision.ranking[1]["gripper_feasible"])

    def test_highest_first_selects_highest_object(self):
        graph = RelationGraph(objects=[_obj("low", 0.1), _obj("high", 0.3)], edges=[])

        decision = select_object("highest-first", graph, random.Random(1))

        self.assertEqual(decision.selected_object, "high")
        self.assertEqual([item["name"] for item in decision.ranking], ["high", "low"])

    def test_od_only_selects_lowest_incoming_od(self):
        graph = RelationGraph(
            objects=[_obj("a", 0.1), _obj("b", 0.2), _obj("c", 0.3)],
            edges=[
                _edge("b", "a", 0.8),
                _edge("c", "a", 0.2),
                _edge("a", "b", 0.1),
                _edge("c", "b", 0.1),
                _edge("a", "c", 0.3),
                _edge("b", "c", 0.3),
            ],
        )

        decision = select_object("od-only", graph, random.Random(1))

        self.assertEqual(decision.selected_object, "b")
        self.assertEqual(decision.ranking[0]["incoming_od"], 0.1)

    def test_lowest_blocked_selects_lowest_incoming_blocked_ratio(self):
        graph = RelationGraph(
            objects=[_obj("a", 0.1), _obj("b", 0.2)],
            edges=[
                _edge("b", "a", 0.2, blocked=0.6),
                _edge("a", "b", 0.2, blocked=0.0),
            ],
        )

        decision = select_object("lowest-blocked", graph, random.Random(1))

        self.assertEqual(decision.selected_object, "b")
        self.assertEqual(decision.ranking[0]["incoming_blocked_grasp_ratio"], 0.0)

    def test_random_policy_is_reproducible_with_seeded_rng(self):
        graph = RelationGraph(objects=[_obj("a", 0.1), _obj("b", 0.2), _obj("c", 0.3)], edges=[])

        first = select_object("random", graph, random.Random(7))
        second = select_object("random", graph, random.Random(7))

        self.assertEqual(first.selected_object, second.selected_object)
        self.assertEqual(first.ranking, second.ranking)


class FakeScene:
    def __init__(self):
        self.objects = [_obj("low", 0.1, xy=(0.0, 0.0)), _obj("high", 0.3, xy=(0.12, 0.0))]
        self.removed = []

    def read_objects(self):
        return list(self.objects)

    def read_object_contact_pairs(self):
        return set()

    def remove_object(self, name: str):
        self.removed.append(name)
        self.objects = [obj for obj in self.objects if obj.name != name]

    def settle(self, steps: int):
        pass


class WideSelectedScene(FakeScene):
    def __init__(self):
        self.objects = [
            _obj("narrow_low", 0.1),
            _obj("wide_high", 0.3, half=(0.06, 0.06, 0.05), xy=(0.12, 0.0)),
        ]
        self.removed = []


class PolicyEpisodeTests(unittest.TestCase):
    def test_policy_episode_uses_requested_policy(self):
        scene = FakeScene()

        result = run_policy_episode(scene, policy="highest-first", max_steps=1, post_grasp_settle_steps=0, seed=3)

        self.assertEqual(result.policy, "highest-first")
        self.assertEqual(result.grasp_sequence, ["high"])
        self.assertEqual(scene.removed, ["high"])
        self.assertEqual(result.steps[0].selected_grasp_pose.closing_axis, "x")
        self.assertEqual(result.steps[0].to_dict()["selected_grasp_pose"]["closing_axis"], "x")

    def test_grasp_risk_assessment_marks_high_risk_selection_as_failure(self):
        score = ObjectScore(
            name="blocked",
            score=0.0,
            graspability_prior=0.2,
            blocked_by_od=0.42,
            support_risk=0.20,
            contact_risk=0.10,
            clearance_gain=0.0,
        )

        assessment = assess_grasp_risk(score, config=GraspRiskConfig(threshold=0.45))

        self.assertFalse(assessment.success)
        self.assertGreaterEqual(assessment.risk, 0.45)
        self.assertEqual(assessment.reason, "risk-threshold")

    def test_grasp_risk_assessment_marks_gripper_infeasible_selection_as_failure(self):
        score = ObjectScore(
            name="wide",
            score=1.0,
            graspability_prior=1.0,
            blocked_by_od=0.0,
            support_risk=0.0,
            contact_risk=0.0,
            clearance_gain=0.0,
        )
        feasibility = ObjectGraspFeasibility(
            object_name="wide",
            feasible=False,
            feasible_grasp_count=0,
            candidates=[
                GraspCandidate("wide", "x", 0.12, False, "opening-too-small", []),
                GraspCandidate("wide", "y", 0.12, False, "opening-too-small", []),
            ],
        )

        assessment = assess_grasp_risk(
            score,
            config=GraspRiskConfig(threshold=0.45),
            gripper_feasibility=feasibility,
        )

        self.assertFalse(assessment.success)
        self.assertEqual(assessment.risk, 1.0)
        self.assertEqual(assessment.reason, "gripper-infeasible")

    def test_policy_episode_stops_before_removal_when_risk_threshold_fails(self):
        scene = FakeScene()

        result = run_policy_episode(
            scene,
            policy="random",
            max_steps=1,
            post_grasp_settle_steps=0,
            seed=1,
            failure_mode="risk-threshold",
            risk_threshold=0.0,
        )

        self.assertEqual(len(result.steps), 1)
        self.assertFalse(result.steps[0].grasp_success)
        self.assertEqual(result.failure_reason, "risk-threshold")
        self.assertEqual(scene.removed, [])

    def test_policy_episode_stops_before_removal_when_gripper_cannot_enter(self):
        scene = WideSelectedScene()

        result = run_policy_episode(
            scene,
            policy="highest-first",
            max_steps=1,
            post_grasp_settle_steps=0,
            seed=1,
            failure_mode="risk-threshold",
            risk_threshold=0.45,
        )

        self.assertEqual(len(result.steps), 1)
        self.assertFalse(result.steps[0].grasp_success)
        self.assertFalse(result.steps[0].gripper_feasible)
        self.assertEqual(result.steps[0].gripper_feasible_grasp_count, 0)
        self.assertEqual(result.steps[0].failure_reason, "gripper-infeasible")
        self.assertEqual(result.failure_reason, "gripper-infeasible")
        self.assertEqual(scene.removed, [])

    def test_gripper_aware_episode_selects_feasible_object_before_infeasible_high_object(self):
        scene = WideSelectedScene()

        result = run_policy_episode(
            scene,
            policy="adaptive-score-v2-gripper",
            max_steps=1,
            post_grasp_settle_steps=0,
            seed=1,
            failure_mode="risk-threshold",
            risk_threshold=0.45,
        )

        self.assertEqual(result.grasp_sequence, ["narrow_low"])
        self.assertTrue(result.steps[0].grasp_success)
        self.assertTrue(result.steps[0].gripper_feasible)
        self.assertEqual(scene.removed, ["narrow_low"])

    def test_policy_episode_can_use_external_graspnet_candidates(self):
        scene = WideSelectedScene()
        pose = GraspPoseCandidate(
            object_name="wide_high",
            generator="graspnet-prediction",
            position=np.array([0.12, 0.0, 0.35], dtype=float),
            pregrasp_position=np.array([0.12, 0.0, 0.47], dtype=float),
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=float),
            closing_axis="6d",
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            required_opening=0.04,
            score=0.93,
        )

        result = run_policy_episode(
            scene,
            policy="highest-first",
            max_steps=1,
            post_grasp_settle_steps=0,
            seed=1,
            failure_mode="risk-threshold",
            risk_threshold=0.45,
            grasp_poses_by_object={"wide_high": [pose]},
        )

        self.assertEqual(result.grasp_sequence, ["wide_high"])
        self.assertTrue(result.steps[0].gripper_feasible)
        self.assertEqual(result.steps[0].selected_grasp_pose.generator, "graspnet-prediction")
        self.assertEqual(scene.removed, ["wide_high"])

    def test_graspnet_policy_episode_uses_external_candidates(self):
        scene = WideSelectedScene()
        pose = GraspPoseCandidate(
            object_name="wide_high",
            generator="graspnet-bound",
            position=np.array([0.12, 0.0, 0.35], dtype=float),
            pregrasp_position=np.array([0.12, 0.0, 0.47], dtype=float),
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=float),
            closing_axis="6d",
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            required_opening=0.04,
            score=0.93,
        )

        result = run_policy_episode(
            scene,
            policy="adaptive-score-v2-graspnet",
            max_steps=1,
            post_grasp_settle_steps=0,
            seed=1,
            failure_mode="risk-threshold",
            risk_threshold=0.45,
            grasp_poses_by_object={"wide_high": [pose]},
        )

        self.assertEqual(result.policy, "adaptive-score-v2-graspnet")
        self.assertEqual(result.grasp_sequence, ["wide_high"])
        self.assertTrue(result.steps[0].gripper_feasible)
        self.assertEqual(result.steps[0].selected_grasp_pose.generator, "graspnet-bound")
        self.assertEqual(scene.removed, ["wide_high"])

    def test_policy_episode_can_use_custom_gripper_feasibility_provider(self):
        scene = WideSelectedScene()

        def provider(objects):
            return [
                _gripper("narrow_low", feasible=False, feasible_count=0),
                _gripper("wide_high", feasible=True, feasible_count=1),
            ]

        result = run_policy_episode(
            scene,
            policy="adaptive-score-v2-graspnet",
            max_steps=1,
            post_grasp_settle_steps=0,
            seed=1,
            failure_mode="risk-threshold",
            risk_threshold=0.45,
            gripper_feasibility_provider=provider,
        )

        self.assertEqual(result.grasp_sequence, ["wide_high"])
        self.assertTrue(result.steps[0].gripper_feasible)
        self.assertEqual(scene.removed, ["wide_high"])


if __name__ == "__main__":
    unittest.main()
