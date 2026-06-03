from __future__ import annotations

import unittest

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.external_graspnet_data import AnnotationObject, RealSenseFrame
from stacked_grasping.gripper.external_graspnet_scene import (
    ExternalGraspNetFrameScene,
    build_external_graspnet_episode_inputs,
    objects_from_labeled_point_cloud,
    visible_boundary_contact_pairs,
)
from stacked_grasping.gripper.graspnet_binding import BoundGraspNetCandidate
from stacked_grasping.planning.episode import run_policy_episode


def _frame(label: np.ndarray, depth_raw: np.ndarray | None = None) -> RealSenseFrame:
    depth = depth_raw if depth_raw is not None else np.full(label.shape, 1000, dtype=np.uint16)
    return RealSenseFrame(
        frame="0000",
        color=np.zeros((*label.shape, 3), dtype=np.uint8),
        depth_raw=np.asarray(depth, dtype=np.uint16),
        label=np.asarray(label, dtype=np.uint8),
        intrinsic_matrix=np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
    )


def _annotations() -> list[AnnotationObject]:
    return [
        AnnotationObject(object_id=0, label_id=1, name="a.ply", position=np.zeros(3)),
        AnnotationObject(object_id=1, label_id=2, name="b.ply", position=np.zeros(3)),
    ]


class ExternalGraspNetSceneTests(unittest.TestCase):
    def test_objects_from_labeled_point_cloud_builds_visible_object_states(self):
        label = np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
            ],
            dtype=np.uint8,
        )

        objects = objects_from_labeled_point_cloud(
            _frame(label),
            _annotations(),
            min_points_per_object=1,
            min_half_extent=0.001,
            padding=0.0,
        )

        self.assertEqual([obj.name for obj in objects], ["a.ply", "b.ply"])
        self.assertEqual([obj.body_id for obj in objects], [0, 1])
        self.assertEqual([obj.geom_id for obj in objects], [1, 2])
        self.assertTrue(all(obj.geom_type == "external-label-aabb" for obj in objects))
        self.assertLess(objects[0].position[0], objects[1].position[0])
        self.assertTrue(all(np.all(obj.half_extents >= 0.001) for obj in objects))

    def test_visible_boundary_pairs_feed_lightweight_scene_contacts(self):
        label = np.array(
            [
                [1, 1, 2],
                [1, 1, 2],
            ],
            dtype=np.uint8,
        )
        frame = _frame(label)
        objects = objects_from_labeled_point_cloud(frame, _annotations(), min_points_per_object=1)
        contact_pairs = visible_boundary_contact_pairs(frame, _annotations(), min_boundary_pixels=1)
        scene = ExternalGraspNetFrameScene(objects, contact_pairs)

        self.assertEqual(contact_pairs, {("a.ply", "b.ply")})
        self.assertEqual(scene.read_object_contact_pairs(), {("a.ply", "b.ply")})

        scene.remove_object("a.ply")

        self.assertEqual([obj.name for obj in scene.read_objects()], ["b.ply"])
        self.assertEqual(scene.read_object_contact_pairs(), set())

    def test_external_graspnet_episode_inputs_run_graspnet_policy(self):
        label = np.array([[1]], dtype=np.uint8)
        frame = _frame(label)
        record = {
            "score": 0.9,
            "width": 0.04,
            "height": 0.02,
            "depth": 0.03,
            "rotation_matrix": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 1.0],
            "object_id": -1,
        }
        binding = BoundGraspNetCandidate(
            record=record,
            frame="0000",
            status="bound",
            pixel=(0, 0),
            label_id=1,
            object_id=0,
            object_name="a.ply",
            depth_error_m=0.0,
        )

        inputs = build_external_graspnet_episode_inputs(
            frame,
            _annotations(),
            [binding],
            min_points_per_object=1,
            min_half_extent=0.02,
        )
        result = run_policy_episode(
            inputs.scene,
            policy="adaptive-score-v2-graspnet",
            max_steps=1,
            post_grasp_settle_steps=0,
            failure_mode="risk-threshold",
            risk_threshold=0.45,
            grasp_poses_by_object=inputs.grasp_poses_by_object,
        )

        self.assertEqual(result.grasp_sequence, ["a.ply"])
        self.assertTrue(result.steps[0].gripper_feasible)
        self.assertEqual(result.steps[0].selected_grasp_pose.generator, "graspnet-bound")
        self.assertEqual(inputs.scene.removed, ["a.ply"])


if __name__ == "__main__":
    unittest.main()
