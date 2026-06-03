from __future__ import annotations

import unittest

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.grasp_pose import (
    MockGraspNetPoseGenerator,
    RuleTopDownGraspPoseGenerator,
    assign_candidates_to_objects,
    generate_side_grasp_pose_candidates,
    generate_topdown_grasp_pose_candidates,
    graspnet_outputs_to_candidates,
)
from stacked_grasping.relations.geometry import ObjectState


def _obj(name: str, pos, half) -> ObjectState:
    return ObjectState(
        name=name,
        body_id=0,
        geom_id=0,
        geom_type="box",
        position=np.array(pos, dtype=float),
        half_extents=np.array(half, dtype=float),
    )


class GraspPoseCandidateTests(unittest.TestCase):
    def test_rule_topdown_generator_returns_two_standard_pose_candidates(self):
        target = _obj("target", (0.10, -0.20, 0.50), (0.02, 0.03, 0.04))

        candidates = generate_topdown_grasp_pose_candidates(
            target,
            lateral_clearance=0.002,
            approach_height=0.12,
        )

        self.assertEqual([candidate.closing_axis for candidate in candidates], ["x", "y"])
        x_candidate = candidates[0]
        self.assertEqual(x_candidate.object_name, "target")
        self.assertEqual(x_candidate.generator, "rule-topdown")
        np.testing.assert_allclose(x_candidate.position, np.array([0.10, -0.20, 0.54]))
        np.testing.assert_allclose(x_candidate.pregrasp_position, np.array([0.10, -0.20, 0.66]))
        np.testing.assert_allclose(x_candidate.approach_direction, np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(x_candidate.orientation_quat_wxyz, np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(x_candidate.required_opening, 0.044)

        payload = x_candidate.to_dict()
        self.assertEqual(payload["object_name"], "target")
        self.assertEqual(payload["generator"], "rule-topdown")
        self.assertEqual(payload["position"], [0.1, -0.2, 0.54])
        self.assertEqual(payload["pregrasp_position"], [0.1, -0.2, 0.66])
        self.assertEqual(payload["approach_direction"], [0.0, 0.0, -1.0])
        self.assertEqual(payload["closing_axis"], "x")
        self.assertEqual(payload["required_opening"], 0.044)

    def test_rule_topdown_generator_can_emit_span_offset_pose_candidates(self):
        target = _obj("target", (0.10, -0.20, 0.50), (0.02, 0.03, 0.04))

        candidates = generate_topdown_grasp_pose_candidates(
            target,
            lateral_clearance=0.002,
            approach_height=0.12,
            span_offsets=(0.0, 0.015),
        )

        self.assertEqual([candidate.closing_axis for candidate in candidates], ["x", "x", "y", "y"])
        np.testing.assert_allclose(candidates[0].position, np.array([0.10, -0.20, 0.54]))
        np.testing.assert_allclose(candidates[1].position, np.array([0.10, -0.185, 0.54]))
        np.testing.assert_allclose(candidates[2].position, np.array([0.10, -0.20, 0.54]))
        np.testing.assert_allclose(candidates[3].position, np.array([0.115, -0.20, 0.54]))

    def test_rule_side_generator_returns_four_axis_aligned_pose_candidates(self):
        target = _obj("target", (0.10, -0.20, 0.50), (0.02, 0.03, 0.04))

        candidates = generate_side_grasp_pose_candidates(
            target,
            lateral_clearance=0.002,
            pregrasp_distance=0.12,
        )

        self.assertEqual([candidate.closing_axis for candidate in candidates], ["y", "y", "x", "x"])
        np.testing.assert_allclose(candidates[0].position, np.array([0.10, -0.20, 0.50]))
        np.testing.assert_allclose(candidates[0].approach_direction, np.array([-1.0, 0.0, 0.0]))
        np.testing.assert_allclose(candidates[0].pregrasp_position, np.array([0.22, -0.20, 0.50]))
        self.assertEqual(candidates[0].required_opening, 0.064)
        np.testing.assert_allclose(candidates[1].approach_direction, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(candidates[2].approach_direction, np.array([0.0, -1.0, 0.0]))
        np.testing.assert_allclose(candidates[3].approach_direction, np.array([0.0, 1.0, 0.0]))

    def test_rule_topdown_generator_class_matches_function_api(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.02, 0.03, 0.04))

        generator = RuleTopDownGraspPoseGenerator(lateral_clearance=0.002, approach_height=0.12)
        candidates = generator.generate_for_object(target)

        self.assertEqual([candidate.closing_axis for candidate in candidates], ["x", "y"])
        self.assertTrue(all(candidate.generator == "rule-topdown" for candidate in candidates))

    def test_graspnet_outputs_convert_to_standard_pose_candidates(self):
        candidates = graspnet_outputs_to_candidates(
            [
                {
                    "translation": [0.10, -0.20, 0.54],
                    "rotation_matrix": np.eye(3),
                    "width": 0.05,
                    "score": 0.87,
                }
            ],
            pregrasp_distance=0.12,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.object_name, "unassigned")
        self.assertEqual(candidate.generator, "graspnet")
        self.assertEqual(candidate.closing_axis, "6d")
        self.assertEqual(candidate.required_opening, 0.05)
        self.assertEqual(candidate.score, 0.87)
        np.testing.assert_allclose(candidate.position, np.array([0.10, -0.20, 0.54]))
        np.testing.assert_allclose(candidate.pregrasp_position, np.array([-0.02, -0.20, 0.54]))
        np.testing.assert_allclose(candidate.approach_direction, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(candidate.orientation_quat_wxyz, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_assign_candidates_to_objects_uses_inflated_aabb(self):
        obj_a = _obj("a", (0.0, 0.0, 0.5), (0.02, 0.02, 0.04))
        obj_b = _obj("b", (0.10, 0.0, 0.5), (0.02, 0.02, 0.04))
        candidates = graspnet_outputs_to_candidates(
            [
                {
                    "translation": [0.10, 0.0, 0.50],
                    "rotation_matrix": np.eye(3),
                    "width": 0.04,
                    "score": 0.90,
                },
                {
                    "translation": [0.40, 0.0, 0.50],
                    "rotation_matrix": np.eye(3),
                    "width": 0.04,
                    "score": 0.20,
                },
            ]
        )

        assigned = assign_candidates_to_objects([obj_a, obj_b], candidates, margin=0.001)

        self.assertEqual(assigned["a"], [])
        self.assertEqual(len(assigned["b"]), 1)
        self.assertEqual(assigned["b"][0].object_name, "b")
        self.assertEqual(assigned["b"][0].generator, "graspnet")

    def test_mock_graspnet_generator_returns_scene_candidates_by_object(self):
        obj_a = _obj("a", (0.0, 0.0, 0.5), (0.02, 0.02, 0.04))
        obj_b = _obj("b", (0.10, 0.0, 0.5), (0.02, 0.02, 0.04))
        generator = MockGraspNetPoseGenerator(
            records=[
                {
                    "translation": [0.0, 0.0, 0.50],
                    "rotation_matrix": np.eye(3),
                    "width": 0.04,
                    "score": 0.81,
                },
                {
                    "translation": [0.10, 0.0, 0.50],
                    "rotation_matrix": np.eye(3),
                    "width": 0.05,
                    "score": 0.91,
                },
            ],
            margin=0.001,
        )

        assigned = generator.generate_for_scene([obj_a, obj_b])

        self.assertEqual([candidate.object_name for candidate in assigned["a"]], ["a"])
        self.assertEqual([candidate.object_name for candidate in assigned["b"]], ["b"])
        self.assertEqual(assigned["b"][0].score, 0.91)


if __name__ == "__main__":
    unittest.main()
