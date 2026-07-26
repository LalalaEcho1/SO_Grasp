from __future__ import annotations

import unittest

import numpy as np

from tests import conftest  # noqa: F401
from scripts.run_external_graspnet_pointcloud_episode import (
    run_frame_episode_summary,
    sample_points_with_labels,
)
from stacked_grasping.gripper.external_graspnet_data import AnnotationObject, RealSenseFrame
from stacked_grasping.gripper.pointcloud_feasibility import PointCloudCollisionConfig


class _FakeRealSenseSource:
    def __init__(self, frame: RealSenseFrame, annotations: list[AnnotationObject]):
        self._frame = frame
        self._annotations = annotations

    def load_frame(self, frame_id):
        return self._frame

    def load_annotation_objects(self, frame_id):
        return list(self._annotations)


class _FakePredictionSource:
    def __init__(self, records: list[dict[str, object]]):
        self._records = records

    def load_records(self, frame_id):
        return [dict(record) for record in self._records]


def _two_object_frame() -> tuple[RealSenseFrame, list[AnnotationObject]]:
    # 16x16 frame, fx=fy=100, cx=cy=8, depth 1 m: pixel (u, v) -> ((u-8)/100, (v-8)/100, 1).
    # Object A (label 2) occupies row v=5  -> points at y=-0.03.
    # Object B (label 3) occupies row v=8  -> points at y=0.
    label = np.zeros((16, 16), dtype=np.uint8)
    label[5, 6:11] = 2
    label[8, 6:11] = 3
    # Background sits 0.3 m behind the objects so table points stay clear of the
    # gripper volume around the z=1.0 grasps.
    depth_raw = np.full((16, 16), 1300, dtype=np.uint16)
    depth_raw[5, 6:11] = 1000
    depth_raw[8, 6:11] = 1000
    frame = RealSenseFrame(
        frame="0000",
        color=np.zeros((16, 16, 3), dtype=np.uint8),
        depth_raw=depth_raw,
        label=label,
        intrinsic_matrix=np.array([[100.0, 0.0, 8.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]]),
    )
    annotations = [
        AnnotationObject(object_id=0, label_id=2, name="blocker_a.ply", position=np.array([0.0, -0.03, 1.0])),
        AnnotationObject(object_id=1, label_id=3, name="target_b.ply", position=np.array([0.0, 0.0, 1.0])),
    ]
    return frame, annotations


def _record(translation: list[float], width: float) -> dict[str, object]:
    return {
        "score": 0.9,
        "width": width,
        "height": 0.02,
        "depth": 0.03,
        "rotation_matrix": np.eye(3).tolist(),
        "translation": translation,
        "object_id": -1,
    }


class ExternalClearingEpisodeTests(unittest.TestCase):
    def test_sample_points_with_labels_keeps_pairs_aligned(self):
        points = np.array([[float(i), 0.0, 0.0] for i in range(10)])
        labels = np.arange(10)

        sampled_points, sampled_labels = sample_points_with_labels(points, labels, limit=4, seed=7)

        self.assertEqual(sampled_points.shape, (4, 3))
        self.assertEqual(sampled_labels.shape, (4,))
        for point, label in zip(sampled_points, sampled_labels):
            self.assertEqual(int(point[0]), int(label))

        full_points, full_labels = sample_points_with_labels(points, labels, limit=None, seed=7)
        self.assertEqual(full_points.shape[0], 10)
        self.assertEqual(full_labels.shape[0], 10)

    def test_multi_step_clearing_removes_grasped_points_and_unblocks_target(self):
        frame, annotations = _two_object_frame()
        # A's grasp (width 0.02): fingers at |y+0.03| in (0.01, 0.02) -> B's points at
        # y=0 are 0.03 away, outside the finger band -> always feasible.
        # B's grasp (width 0.05): fingers at |y| in (0.025, 0.035) -> A's points at
        # y=-0.03 collide while A is still on the table.
        records = [
            _record([0.0, -0.03, 1.0], width=0.02),
            _record([0.0, 0.0, 1.0], width=0.05),
        ]
        shared = dict(
            realsense_source=_FakeRealSenseSource(frame, annotations),
            prediction_source=_FakePredictionSource(records),
            frame_id="0000",
            binding_mode="3d",
            binding_3d_max_distance_m=0.05,
            min_points_per_object=1,
            min_half_extent=0.001,
            object_padding=0.0,
            min_boundary_pixels=1,
            point_sample_limit=None,
            risk_threshold=0.45,
            pointcloud_config=PointCloudCollisionConfig(voxel_size=0.01, clamp_width_to_max_opening=True),
        )

        single = run_frame_episode_summary(max_steps=1, policy="adaptive-score-v2-riskaware", **shared)
        self.assertEqual(single["num_steps"], 1)
        self.assertEqual(single["pointcloud_feasible_object_count"], 1)
        self.assertEqual(single["selected_object"], "blocker_a.ply")
        self.assertTrue(single["grasp_success"])

        clearing = run_frame_episode_summary(max_steps=None, policy="adaptive-score-v2-riskaware", **shared)
        self.assertEqual(clearing["num_steps"], 2)
        self.assertEqual(clearing["num_successes"], 2)
        self.assertAlmostEqual(clearing["clearance_rate"], 1.0)
        self.assertIsNone(clearing["episode_failure_reason"])


if __name__ == "__main__":
    unittest.main()
