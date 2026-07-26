from __future__ import annotations

import unittest

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.external_graspnet_data import AnnotationObject, RealSenseFrame
from stacked_grasping.gripper.graspnet_binding import (
    bind_graspnet_records,
    bind_graspnet_records_to_frame_labels,
    bind_graspnet_records_to_objects_3d,
)


def _frame(label: np.ndarray) -> RealSenseFrame:
    size = label.shape[0]
    return RealSenseFrame(
        frame="0000",
        color=np.zeros((size, size, 3), dtype=np.uint8),
        depth_raw=np.full((size, size), 1000, dtype=np.uint16),
        label=label,
        intrinsic_matrix=np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]),
    )


def _record(translation: list[float]) -> dict[str, object]:
    return {
        "score": 0.5,
        "width": 0.06,
        "height": 0.02,
        "depth": 0.03,
        "rotation_matrix": np.eye(3).tolist(),
        "translation": translation,
        "object_id": -1,
    }


class GraspNetBinding3DTests(unittest.TestCase):
    def test_binds_to_nearest_labeled_points_within_max_distance(self):
        label = np.zeros((4, 4), dtype=np.uint8)
        label[1:3, 1:3] = 2
        frame = _frame(label)
        annotations = [AnnotationObject(object_id=1, label_id=2, name="banana.ply", position=np.zeros(3))]
        # Label pixels map to 3D points around (0.0-0.01, 0.0-0.01, 1.0).
        records = [_record([0.005, 0.005, 0.98])]

        bound = bind_graspnet_records_to_objects_3d(records, frame, annotations, max_distance_m=0.05, point_stride=1)

        self.assertEqual(bound[0].status, "bound")
        self.assertEqual(bound[0].label_id, 2)
        self.assertEqual(bound[0].object_name, "banana.ply")
        self.assertLessEqual(bound[0].depth_error_m, 0.05)

    def test_rejects_candidate_beyond_max_distance(self):
        label = np.zeros((4, 4), dtype=np.uint8)
        label[1:3, 1:3] = 2
        frame = _frame(label)

        bound = bind_graspnet_records_to_objects_3d(
            [_record([0.0, 0.0, 0.5])], frame, (), max_distance_m=0.05, point_stride=1
        )

        self.assertEqual(bound[0].status, "no-nearby-points")
        self.assertIsNone(bound[0].label_id)
        self.assertGreater(bound[0].depth_error_m, 0.05)

    def test_invalid_translation_and_missing_label_are_reported(self):
        label = np.zeros((4, 4), dtype=np.uint8)
        label[1:3, 1:3] = 2
        frame = _frame(label)

        invalid = bind_graspnet_records_to_objects_3d([_record([np.nan, 0.0, 1.0])], frame, (), point_stride=1)
        self.assertEqual(invalid[0].status, "invalid-translation")

        no_label_frame = RealSenseFrame(
            frame="0000",
            color=frame.color,
            depth_raw=frame.depth_raw,
            label=None,
            intrinsic_matrix=frame.intrinsic_matrix,
        )
        missing = bind_graspnet_records_to_objects_3d([_record([0.0, 0.0, 1.0])], no_label_frame, ())
        self.assertEqual(missing[0].status, "no-label")

    def test_3d_binding_recovers_parallax_case_that_pixel_binding_loses(self):
        # Object label occupies pixels (2..3, 2..3): 3D points near (0.01-0.02, 0.01-0.02, 1.0).
        label = np.zeros((4, 4), dtype=np.uint8)
        label[2:4, 2:4] = 5
        frame = _frame(label)
        annotations = [AnnotationObject(object_id=4, label_id=5, name="pear.ply", position=np.zeros(3))]
        # Grasp center is 3D-close to the object (0.042 m) but projects to pixel (6, 6),
        # outside the image entirely — the parallax failure mode from the diagnosis.
        records = [_record([0.05, 0.05, 1.0])]

        pixel_bound = bind_graspnet_records_to_frame_labels(records, frame, annotations, pixel_radius=1)
        bound_3d = bind_graspnet_records_to_objects_3d(records, frame, annotations, max_distance_m=0.05, point_stride=1)

        self.assertNotEqual(pixel_bound[0].status, "bound")
        self.assertEqual(bound_3d[0].status, "bound")
        self.assertEqual(bound_3d[0].object_name, "pear.ply")

    def test_dispatcher_selects_mode_and_validates(self):
        label = np.zeros((4, 4), dtype=np.uint8)
        label[1:3, 1:3] = 2
        frame = _frame(label)
        annotations = [AnnotationObject(object_id=1, label_id=2, name="banana.ply", position=np.zeros(3))]
        records = [_record([0.005, 0.005, 1.0])]

        via_pixel = bind_graspnet_records(records, frame, annotations, mode="pixel", pixel_radius=1, depth_tolerance_m=0.05)
        via_3d = bind_graspnet_records(records, frame, annotations, mode="3d", max_distance_m=0.05, point_stride=1)

        self.assertEqual(via_pixel[0].status, "bound")
        self.assertEqual(via_3d[0].status, "bound")
        with self.assertRaises(ValueError):
            bind_graspnet_records(records, frame, annotations, mode="voxel")


if __name__ == "__main__":
    unittest.main()
