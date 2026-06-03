from __future__ import annotations

import unittest

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.feasibility import (
    TopDownGraspConfig,
    assess_object_grasp_candidates,
    assess_object_topdown_grasps,
    assess_scene_grasp_candidates,
    assess_scene_topdown_grasps,
)
from stacked_grasping.gripper.grasp_pose import (
    GraspPoseCandidate,
    generate_side_grasp_pose_candidates,
    generate_topdown_grasp_pose_candidates,
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


class GripperFeasibilityTests(unittest.TestCase):
    def test_topdown_candidate_exports_explicit_pose(self):
        target = _obj("target", (0.10, -0.20, 0.50), (0.02, 0.03, 0.04))

        result = assess_object_topdown_grasps([target], target_name="target", config=TopDownGraspConfig())
        x_candidate = next(candidate for candidate in result.candidates if candidate.closing_axis == "x")
        y_candidate = next(candidate for candidate in result.candidates if candidate.closing_axis == "y")

        self.assertIsNotNone(x_candidate.pose)
        self.assertIsNotNone(y_candidate.pose)
        np.testing.assert_allclose(x_candidate.pose.position, np.array([0.10, -0.20, 0.54]))
        np.testing.assert_allclose(x_candidate.pose.pregrasp_position, np.array([0.10, -0.20, 0.66]))
        np.testing.assert_allclose(x_candidate.pose.approach_direction, np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(x_candidate.pose.orientation_quat_wxyz, np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(
            y_candidate.pose.orientation_quat_wxyz,
            np.array([0.707107, 0.0, 0.0, 0.707107]),
            atol=1e-6,
        )
        self.assertEqual(x_candidate.pose.generator, "rule-topdown")
        self.assertEqual(x_candidate.pose.required_opening, x_candidate.required_opening)
        self.assertTrue(x_candidate.pose.feasible)
        self.assertIsNone(x_candidate.pose.failure_reason)

    def test_small_isolated_object_has_center_offset_and_side_feasible_candidates(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.02, 0.03, 0.04))

        result = assess_object_topdown_grasps([target], target_name="target", config=TopDownGraspConfig())

        self.assertTrue(result.feasible)
        self.assertEqual(result.feasible_grasp_count, 10)
        self.assertEqual({candidate.closing_axis for candidate in result.candidates}, {"x", "y"})
        self.assertEqual(result.selected_candidate.closing_axis, "x")
        self.assertEqual(result.selected_grasp_pose.closing_axis, "x")
        np.testing.assert_allclose(result.selected_grasp_pose.position, np.array([0.0, 0.0, 0.54]))
        self.assertEqual(result.to_dict()["selected_grasp_pose"]["closing_axis"], "x")

    def test_candidate_fails_when_object_is_wider_than_gripper_opening(self):
        target = _obj("wide", (0.0, 0.0, 0.5), (0.06, 0.02, 0.04))

        result = assess_object_topdown_grasps([target], target_name="wide", config=TopDownGraspConfig(max_opening=0.085))
        x_candidate = next(candidate for candidate in result.candidates if candidate.closing_axis == "x")
        y_candidate = next(candidate for candidate in result.candidates if candidate.closing_axis == "y")

        self.assertFalse(x_candidate.feasible)
        self.assertEqual(x_candidate.reason, "opening-too-small")
        self.assertTrue(y_candidate.feasible)
        self.assertEqual(result.selected_candidate.closing_axis, "y")
        self.assertEqual(result.selected_grasp_pose.closing_axis, "y")

    def test_default_clearance_allows_small_ycb_can_width(self):
        target = _obj("can", (0.0, 0.0, 0.5), (0.0339, 0.0339, 0.05))

        result = assess_object_topdown_grasps([target], target_name="can")

        self.assertTrue(result.feasible)
        self.assertGreaterEqual(result.feasible_grasp_count, 2)

    def test_side_obstacle_blocks_matching_finger_candidate(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.02, 0.02, 0.04))
        side_obstacle = _obj("side", (0.036, 0.0, 0.5), (0.012, 0.01, 0.04))

        result = assess_object_topdown_grasps([target, side_obstacle], target_name="target")
        x_candidate = next(candidate for candidate in result.candidates if candidate.closing_axis == "x")
        y_candidate = next(candidate for candidate in result.candidates if candidate.closing_axis == "y")

        self.assertFalse(x_candidate.feasible)
        self.assertEqual(x_candidate.reason, "finger-collision")
        self.assertEqual(x_candidate.collision_objects, ["side"])
        self.assertFalse(x_candidate.pose.feasible)
        self.assertEqual(x_candidate.pose.failure_reason, "finger-collision")
        self.assertEqual(x_candidate.pose.collision_objects, ("side",))
        self.assertTrue(y_candidate.feasible)
        self.assertEqual(result.selected_candidate.closing_axis, "y")

    def test_span_offset_candidate_can_avoid_a_local_finger_collision(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.015, 0.03, 0.04))
        side_obstacle = _obj("side", (0.027, -0.028, 0.5), (0.004, 0.002, 0.04))
        config = TopDownGraspConfig(
            max_opening=0.05,
            finger_depth_margin=0.001,
            span_offsets=(0.0, 0.025),
            include_side_grasps=False,
        )

        result = assess_object_topdown_grasps([target, side_obstacle], target_name="target", config=config)
        x_candidates = [candidate for candidate in result.candidates if candidate.closing_axis == "x"]
        y_candidates = [candidate for candidate in result.candidates if candidate.closing_axis == "y"]

        self.assertEqual(len(x_candidates), 2)
        self.assertFalse(x_candidates[0].feasible)
        self.assertEqual(x_candidates[0].reason, "finger-collision")
        self.assertTrue(x_candidates[1].feasible)
        np.testing.assert_allclose(x_candidates[1].pose.position, np.array([0.0, 0.025, 0.54]))
        self.assertFalse(any(candidate.feasible for candidate in y_candidates))
        self.assertIs(result.selected_candidate, x_candidates[1])
        self.assertIs(result.selected_grasp_pose, x_candidates[1].pose)

    def test_assesses_explicit_candidate_list_for_future_generators(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.015, 0.03, 0.04))
        side_obstacle = _obj("side", (0.027, -0.028, 0.5), (0.004, 0.002, 0.04))
        config = TopDownGraspConfig(
            max_opening=0.05,
            finger_depth_margin=0.001,
            span_offsets=(0.0,),
        )
        poses = generate_topdown_grasp_pose_candidates(
            target,
            lateral_clearance=config.vertical_clearance,
            approach_height=config.approach_height,
            span_offsets=(0.0, 0.025),
        )

        result = assess_object_grasp_candidates(
            [target, side_obstacle],
            target_name="target",
            poses=poses,
            config=config,
        )

        self.assertEqual(len(result.candidates), 3)
        self.assertTrue(result.feasible)
        self.assertEqual(result.selected_grasp_pose.position[1], 0.025)

    def test_side_candidate_can_avoid_a_blocked_approach_side(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.015, 0.02, 0.04))
        approach_blocker = _obj("approach_blocker", (0.075, 0.0, 0.5), (0.01, 0.01, 0.04))
        poses = [
            pose
            for pose in generate_side_grasp_pose_candidates(target, pregrasp_distance=0.12)
            if pose.closing_axis == "y"
        ]

        result = assess_object_grasp_candidates(
            [target, approach_blocker],
            target_name="target",
            poses=poses,
            config=TopDownGraspConfig(include_side_grasps=True),
        )

        blocked_from_positive_x = result.candidates[0]
        feasible_from_negative_x = result.candidates[1]
        self.assertFalse(blocked_from_positive_x.feasible)
        self.assertEqual(blocked_from_positive_x.reason, "approach-collision")
        self.assertEqual(blocked_from_positive_x.collision_objects, ["approach_blocker"])
        self.assertTrue(feasible_from_negative_x.feasible)
        np.testing.assert_allclose(feasible_from_negative_x.pose.approach_direction, np.array([1.0, 0.0, 0.0]))
        self.assertIs(result.selected_candidate, feasible_from_negative_x)

    def test_default_rule_candidates_can_fall_back_to_side_grasp_when_topdown_is_blocked(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.015, 0.015, 0.04))
        high_blockers = [
            _obj("top_x_pos", (0.024, 0.0, 0.62), (0.003, 0.003, 0.01)),
            _obj("top_x_neg", (-0.024, 0.0, 0.62), (0.003, 0.003, 0.01)),
            _obj("top_y_pos", (0.0, 0.024, 0.62), (0.003, 0.003, 0.01)),
            _obj("top_y_neg", (0.0, -0.024, 0.62), (0.003, 0.003, 0.01)),
        ]

        result = assess_object_topdown_grasps([target, *high_blockers], target_name="target")

        self.assertTrue(result.feasible)
        self.assertEqual(result.selected_grasp_pose.generator, "rule-side")
        self.assertTrue(all(not candidate.feasible for candidate in result.candidates if candidate.pose.generator == "rule-topdown"))
        self.assertTrue(any(candidate.feasible for candidate in result.candidates if candidate.pose.generator == "rule-side"))

    def test_6d_candidate_is_feasible_for_isolated_target(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.015, 0.03, 0.04))
        pose = GraspPoseCandidate(
            object_name="target",
            generator="graspnet",
            position=np.array([0.0, 0.0, 0.5], dtype=float),
            pregrasp_position=np.array([0.0, 0.0, 0.62], dtype=float),
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=float),
            closing_axis="6d",
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            required_opening=0.04,
            score=0.9,
        )

        result = assess_object_grasp_candidates([target], target_name="target", poses=[pose])

        self.assertTrue(result.feasible)
        self.assertTrue(result.candidates[0].feasible)
        self.assertIsNone(result.candidates[0].reason)
        self.assertEqual(result.selected_grasp_pose.generator, "graspnet")

    def test_6d_candidate_fails_when_approach_channel_is_blocked(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.015, 0.03, 0.04))
        blocker = _obj("blocker", (0.0, 0.0, 0.58), (0.01, 0.01, 0.01))
        pose = GraspPoseCandidate(
            object_name="target",
            generator="graspnet",
            position=np.array([0.0, 0.0, 0.5], dtype=float),
            pregrasp_position=np.array([0.0, 0.0, 0.66], dtype=float),
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=float),
            closing_axis="6d",
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            required_opening=0.04,
            score=0.9,
        )

        result = assess_object_grasp_candidates([target, blocker], target_name="target", poses=[pose])

        self.assertFalse(result.feasible)
        self.assertEqual(result.candidates[0].reason, "approach-collision")
        self.assertEqual(result.candidates[0].collision_objects, ["blocker"])
        self.assertEqual(result.candidates[0].pose.failure_reason, "approach-collision")

    def test_6d_candidate_fails_when_grasp_center_is_not_on_target(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.015, 0.03, 0.04))
        pose = GraspPoseCandidate(
            object_name="target",
            generator="graspnet",
            position=np.array([0.20, 0.0, 0.5], dtype=float),
            pregrasp_position=np.array([0.20, 0.0, 0.62], dtype=float),
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=float),
            closing_axis="6d",
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            required_opening=0.04,
            score=0.9,
        )

        result = assess_object_grasp_candidates([target], target_name="target", poses=[pose])

        self.assertFalse(result.feasible)
        self.assertEqual(result.candidates[0].reason, "target-mismatch")

    def test_6d_graspnet_identity_rotation_uses_y_axis_for_fingers(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.015, 0.015, 0.04))
        finger_blocker = _obj("finger_blocker", (0.0, 0.034, 0.5), (0.003, 0.003, 0.006))
        pose = GraspPoseCandidate(
            object_name="target",
            generator="graspnet",
            position=np.array([0.0, 0.0, 0.5], dtype=float),
            pregrasp_position=np.array([-0.12, 0.0, 0.5], dtype=float),
            approach_direction=np.array([1.0, 0.0, 0.0], dtype=float),
            closing_axis="6d",
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            required_opening=0.04,
            score=0.9,
        )

        result = assess_object_grasp_candidates([target, finger_blocker], target_name="target", poses=[pose])

        self.assertFalse(result.feasible)
        self.assertEqual(result.candidates[0].reason, "approach-collision")
        self.assertEqual(result.candidates[0].collision_objects, ["finger_blocker"])

    def test_6d_candidate_uses_orientation_for_collision(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.015, 0.015, 0.04))
        offset = 0.031 / np.sqrt(2.0)
        blocker = _obj("diagonal_blocker", (-offset, offset, 0.5), (0.006, 0.006, 0.004))
        yaw_45 = np.array([np.cos(np.pi / 8.0), 0.0, 0.0, np.sin(np.pi / 8.0)], dtype=float)
        pose = GraspPoseCandidate(
            object_name="target",
            generator="graspnet",
            position=np.array([0.0, 0.0, 0.5], dtype=float),
            pregrasp_position=np.array([0.0, 0.0, 0.62], dtype=float),
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=float),
            closing_axis="6d",
            orientation_quat_wxyz=yaw_45,
            required_opening=0.04,
            score=0.9,
        )

        result = assess_object_grasp_candidates([target, blocker], target_name="target", poses=[pose])

        self.assertFalse(result.feasible)
        self.assertEqual(result.candidates[0].reason, "approach-collision")
        self.assertEqual(result.candidates[0].collision_objects, ["diagonal_blocker"])

    def test_assesses_scene_candidate_mapping_by_object_name(self):
        obj_a = _obj("a", (0.0, 0.0, 0.5), (0.015, 0.02, 0.04))
        obj_b = _obj("b", (0.12, 0.0, 0.5), (0.015, 0.02, 0.04))
        poses_by_object = {
            "a": generate_topdown_grasp_pose_candidates(obj_a),
            "b": [],
        }

        results = assess_scene_grasp_candidates([obj_a, obj_b], poses_by_object)

        self.assertEqual([result.object_name for result in results], ["a", "b"])
        self.assertTrue(results[0].feasible)
        self.assertFalse(results[1].feasible)
        self.assertEqual(results[1].feasible_grasp_count, 0)

    def test_selected_candidate_is_none_when_all_candidates_fail(self):
        target = _obj("wide", (0.0, 0.0, 0.5), (0.06, 0.06, 0.04))

        result = assess_object_topdown_grasps([target], target_name="wide", config=TopDownGraspConfig(max_opening=0.085))

        self.assertFalse(result.feasible)
        self.assertIsNone(result.selected_candidate)
        self.assertIsNone(result.selected_grasp_pose)
        self.assertIsNone(result.to_dict()["selected_grasp_pose"])

    def test_scene_assessment_reports_all_objects(self):
        objects = [
            _obj("a", (0.0, 0.0, 0.5), (0.02, 0.02, 0.04)),
            _obj("b", (0.12, 0.0, 0.5), (0.02, 0.02, 0.04)),
        ]

        results = assess_scene_topdown_grasps(objects)

        self.assertEqual([result.object_name for result in results], ["a", "b"])
        self.assertTrue(all(result.feasible for result in results))


if __name__ == "__main__":
    unittest.main()
