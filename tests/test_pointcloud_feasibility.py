from __future__ import annotations

import unittest

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.graspnet_binding import BoundGraspNetCandidate
from stacked_grasping.gripper.pointcloud_feasibility import (
    PointCloudCollisionConfig,
    assess_bound_graspnet_candidates_with_point_cloud,
    assess_scene_bound_graspnet_pointcloud_feasibility,
)
from stacked_grasping.relations.geometry import ObjectState


def _binding(
    *,
    width: float = 0.04,
    object_name: str = "target.ply",
    score: float = 0.9,
) -> BoundGraspNetCandidate:
    return BoundGraspNetCandidate(
        record={
            "score": score,
            "width": width,
            "height": 0.02,
            "depth": 0.03,
            "rotation_matrix": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 0.0],
            "object_id": -1,
        },
        frame="0000",
        status="bound",
        pixel=(0, 0),
        label_id=1,
        object_id=0,
        object_name=object_name,
        depth_error_m=0.0,
    )


class PointCloudFeasibilityTests(unittest.TestCase):
    def test_bound_graspnet_candidate_is_feasible_when_inner_points_exist_without_collision(self):
        points = np.array([[0.0, 0.0, 0.0]] * 10, dtype=float)

        results = assess_bound_graspnet_candidates_with_point_cloud(points, [_binding()])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].object_name, "target.ply")
        self.assertTrue(results[0].feasible)
        self.assertEqual(results[0].feasible_grasp_count, 1)
        self.assertEqual(results[0].selected_grasp_pose.generator, "graspnet-pointcloud")

    def test_pointcloud_collision_rejects_candidate(self):
        inner_points = np.array([[0.0, 0.0, 0.0]] * 10, dtype=float)
        finger_collision_points = np.array([[0.0, -0.025, 0.0]] * 10, dtype=float)
        points = np.vstack([inner_points, finger_collision_points])

        results = assess_bound_graspnet_candidates_with_point_cloud(points, [_binding()])

        self.assertFalse(results[0].feasible)
        self.assertEqual(results[0].candidates[0].reason, "pointcloud-collision")

    def test_opening_limit_rejects_too_wide_candidate_before_collision(self):
        points = np.array([[0.0, 0.0, 0.0]] * 10, dtype=float)
        config = PointCloudCollisionConfig(max_opening=0.085)

        results = assess_bound_graspnet_candidates_with_point_cloud(points, [_binding(width=0.12)], config=config)

        self.assertFalse(results[0].feasible)
        self.assertEqual(results[0].candidates[0].reason, "opening-too-small")

    def test_scene_assessment_reports_objects_without_bound_candidates_as_infeasible(self):
        points = np.array([[0.0, 0.0, 0.0]] * 10, dtype=float)
        objects = [
            ObjectState("target.ply", 0, 1, "box", np.zeros(3), np.ones(3) * 0.01),
            ObjectState("missing.ply", 1, 2, "box", np.ones(3), np.ones(3) * 0.01),
        ]

        results = assess_scene_bound_graspnet_pointcloud_feasibility(objects, points, [_binding()])

        self.assertEqual([item.object_name for item in results], ["target.ply", "missing.ply"])
        self.assertTrue(results[0].feasible)
        self.assertFalse(results[1].feasible)
        self.assertEqual(results[1].feasible_grasp_count, 0)
        self.assertEqual(results[1].candidates, [])


class PointCloudWidthClampTests(unittest.TestCase):
    def test_clamp_recovers_wide_candidate_on_thin_object(self):
        points = np.array([[0.0, 0.0, 0.0]] * 10, dtype=float)
        config = PointCloudCollisionConfig(max_opening=0.085, clamp_width_to_max_opening=True)

        results = assess_bound_graspnet_candidates_with_point_cloud(points, [_binding(width=0.12)], config=config)

        self.assertTrue(results[0].feasible)
        self.assertEqual(results[0].feasible_grasp_count, 1)
        self.assertIsNone(results[0].candidates[0].reason)

    def test_clamp_still_rejects_object_thicker_than_max_opening(self):
        inner_points = np.array([[0.0, 0.0, 0.0]] * 10, dtype=float)
        # Points at y=0.05 sit inside a 0.12-wide opening but land in the finger
        # volume once the opening is clamped to 0.085 (half width 0.0425).
        finger_points = np.array([[0.0, 0.05, 0.0]] * 30, dtype=float)
        points = np.vstack([inner_points, finger_points])
        config = PointCloudCollisionConfig(max_opening=0.085, clamp_width_to_max_opening=True)

        results = assess_bound_graspnet_candidates_with_point_cloud(points, [_binding(width=0.12)], config=config)

        self.assertFalse(results[0].feasible)
        self.assertEqual(results[0].candidates[0].reason, "pointcloud-collision")

    def test_clamp_disabled_keeps_hard_reject_behaviour(self):
        points = np.array([[0.0, 0.0, 0.0]] * 10, dtype=float)
        config = PointCloudCollisionConfig(max_opening=0.085)

        results = assess_bound_graspnet_candidates_with_point_cloud(points, [_binding(width=0.12)], config=config)

        self.assertFalse(results[0].feasible)
        self.assertEqual(results[0].candidates[0].reason, "opening-too-small")

    def test_clamp_does_not_change_candidates_within_opening_limit(self):
        points = np.array([[0.0, 0.0, 0.0]] * 10, dtype=float)
        clamped = assess_bound_graspnet_candidates_with_point_cloud(
            points,
            [_binding(width=0.04)],
            config=PointCloudCollisionConfig(clamp_width_to_max_opening=True),
        )
        default = assess_bound_graspnet_candidates_with_point_cloud(
            points,
            [_binding(width=0.04)],
            config=PointCloudCollisionConfig(),
        )

        self.assertTrue(clamped[0].feasible)
        self.assertTrue(default[0].feasible)
        self.assertEqual(clamped[0].feasible_grasp_count, default[0].feasible_grasp_count)


if __name__ == "__main__":
    unittest.main()
