from __future__ import annotations

import random
import unittest

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.feasibility import GraspCandidate, ObjectGraspFeasibility
from stacked_grasping.planning.grasp_risk import GraspRiskConfig
from stacked_grasping.planning.policies import VALID_POLICIES, select_object
from stacked_grasping.relations.geometry import ObjectState
from stacked_grasping.relations.graph import EdgeFeatures, RelationGraph


def _obj(name: str, z: float) -> ObjectState:
    return ObjectState(
        name=name,
        body_id=0,
        geom_id=0,
        geom_type="box",
        position=np.array([0.0, 0.0, z], dtype=float),
        half_extents=np.array([0.02, 0.02, 0.05], dtype=float),
    )


def _edge(source: str, target: str, od: float) -> EdgeFeatures:
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
        blocked_grasp_ratio=0.0,
        od_mean=od,
        od_max=od,
        blocked_grasp_kinds=[],
        od_prior=0.0,
    )


def _gripper(name: str, feasible_count: int, total: int = 2) -> ObjectGraspFeasibility:
    candidates = [
        GraspCandidate(name, "x", 0.04, index < feasible_count, None if index < feasible_count else "finger-collision", [])
        for index in range(total)
    ]
    return ObjectGraspFeasibility(
        object_name=name,
        feasible=feasible_count > 0,
        feasible_grasp_count=feasible_count,
        candidates=candidates,
    )


class RiskAwarePolicyTests(unittest.TestCase):
    def test_prefers_sub_threshold_object_that_graspnet_ranking_misses(self):
        # tall_risky: base risk 0.32 (below v2's internal 0.35 gate) + 0.35 * 0.5
        # gripper term = 0.495 full risk -> judged a failure at threshold 0.45.
        # low_safe: base 0, same gripper term -> full risk 0.175.
        graph = RelationGraph(
            objects=[_obj("low_safe", 0.1), _obj("tall_risky", 0.4)],
            edges=[_edge("low_safe", "tall_risky", od=0.32)],
        )
        feasibilities = [_gripper("low_safe", 1), _gripper("tall_risky", 1)]

        graspnet = select_object(
            "adaptive-score-v2-graspnet", graph, random.Random(0), gripper_feasibilities=feasibilities
        )
        riskaware = select_object(
            "adaptive-score-v2-riskaware",
            graph,
            random.Random(0),
            gripper_feasibilities=feasibilities,
            risk_config=GraspRiskConfig(threshold=0.45),
        )

        self.assertIn("adaptive-score-v2-riskaware", VALID_POLICIES)
        self.assertEqual(graspnet.selected_object, "tall_risky")  # the mismatch this policy fixes
        self.assertEqual(riskaware.selected_object, "low_safe")
        self.assertTrue(riskaware.ranking[0]["predicted_success"])
        self.assertFalse(riskaware.ranking[1]["predicted_success"])
        self.assertLess(riskaware.ranking[0]["full_grasp_risk"], 0.45)

    def test_prefers_higher_object_within_predicted_success_set(self):
        graph = RelationGraph(objects=[_obj("low", 0.1), _obj("high", 0.4)], edges=[])
        feasibilities = [_gripper("low", 1), _gripper("high", 1)]

        decision = select_object(
            "adaptive-score-v2-riskaware", graph, random.Random(0), gripper_feasibilities=feasibilities
        )

        self.assertEqual(decision.selected_object, "high")

    def test_falls_back_to_lowest_full_risk_when_no_predicted_success(self):
        graph = RelationGraph(
            objects=[_obj("tall_worst", 0.4), _obj("low_least_bad", 0.1)],
            edges=[
                _edge("low_least_bad", "tall_worst", od=0.42),
                _edge("tall_worst", "low_least_bad", od=0.40),
            ],
        )
        feasibilities = [_gripper("tall_worst", 1), _gripper("low_least_bad", 1)]

        decision = select_object(
            "adaptive-score-v2-riskaware",
            graph,
            random.Random(0),
            gripper_feasibilities=feasibilities,
            risk_config=GraspRiskConfig(threshold=0.45),
        )

        self.assertEqual(decision.selected_object, "low_least_bad")
        self.assertFalse(decision.ranking[0]["predicted_success"])
        self.assertLess(
            decision.ranking[0]["full_grasp_risk"], decision.ranking[1]["full_grasp_risk"]
        )

    def test_infeasible_objects_rank_last(self):
        graph = RelationGraph(objects=[_obj("feasible_low", 0.1), _obj("infeasible_high", 0.4)], edges=[])
        feasibilities = [_gripper("feasible_low", 1), _gripper("infeasible_high", 0)]

        decision = select_object(
            "adaptive-score-v2-riskaware", graph, random.Random(0), gripper_feasibilities=feasibilities
        )

        self.assertEqual(decision.selected_object, "feasible_low")
        self.assertFalse(decision.ranking[-1]["gripper_feasible"])


if __name__ == "__main__":
    unittest.main()
