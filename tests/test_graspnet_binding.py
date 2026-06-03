from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.external_graspnet_data import AnnotationObject, RealSenseFrame
from stacked_grasping.gripper.graspnet_binding import (
    BoundGraspNetCandidate,
    GraspNetPredictionSource,
    bind_graspnet_records_to_frame_labels,
    bound_candidates_to_grasp_poses_by_object,
    project_camera_points_to_pixels,
    summarize_bound_graspnet_candidates,
)


def _grasp_row(score: float, translation: tuple[float, float, float], object_id: float = -1.0) -> np.ndarray:
    row = np.zeros(17, dtype=np.float32)
    row[0] = score
    row[1] = 0.08
    row[2] = 0.02
    row[3] = 0.03
    row[4:13] = np.eye(3, dtype=np.float32).reshape(-1)
    row[13:16] = np.array(translation, dtype=np.float32)
    row[16] = object_id
    return row


def _npy_bytes(array: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


class GraspNetBindingTests(unittest.TestCase):
    def test_project_camera_points_to_pixels_uses_intrinsic_matrix(self):
        intrinsic = np.array([[100.0, 0.0, 10.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]])
        points = np.array([[0.0, 0.0, 1.0], [0.2, -0.1, 2.0]], dtype=float)

        pixels, valid = project_camera_points_to_pixels(points, intrinsic)

        np.testing.assert_allclose(pixels, np.array([[10.0, 20.0], [20.0, 10.0]]))
        self.assertEqual(valid.tolist(), [True, True])

    def test_bind_records_to_label_and_annotation_object(self):
        frame = RealSenseFrame(
            frame="0000",
            color=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_raw=np.full((4, 4), 1000, dtype=np.uint16),
            label=np.array(
                [
                    [0, 0, 0, 0],
                    [0, 2, 2, 0],
                    [0, 2, 2, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            intrinsic_matrix=np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]),
        )
        records = [
            {
                "score": 0.7,
                "width": 0.08,
                "height": 0.02,
                "depth": 0.03,
                "rotation_matrix": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 1.0],
                "object_id": -1,
            }
        ]
        objects = [
            AnnotationObject(
                object_id=1,
                label_id=2,
                name="banana.ply",
                position=np.array([0.0, 0.0, 1.0]),
            )
        ]

        bound = bind_graspnet_records_to_frame_labels(records, frame, objects, pixel_radius=0, depth_tolerance_m=0.02)

        self.assertEqual(len(bound), 1)
        self.assertEqual(bound[0].status, "bound")
        self.assertEqual(bound[0].label_id, 2)
        self.assertEqual(bound[0].object_id, 1)
        self.assertEqual(bound[0].object_name, "banana.ply")
        self.assertEqual(bound[0].pixel, (1, 1))
        self.assertAlmostEqual(bound[0].depth_error_m, 0.0)

    def test_binding_uses_neighborhood_and_rejects_depth_mismatch(self):
        frame = RealSenseFrame(
            frame="0000",
            color=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_raw=np.full((4, 4), 1000, dtype=np.uint16),
            label=np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 3, 0],
                    [0, 3, 3, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            intrinsic_matrix=np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]),
        )
        objects = [AnnotationObject(object_id=2, label_id=3, name="pear.ply", position=np.zeros(3))]
        records = [
            {
                "score": 0.6,
                "width": 0.08,
                "height": 0.02,
                "depth": 0.03,
                "rotation_matrix": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 1.0],
            },
            {
                "score": 0.4,
                "width": 0.08,
                "height": 0.02,
                "depth": 0.03,
                "rotation_matrix": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 1.3],
            },
        ]

        bound = bind_graspnet_records_to_frame_labels(records, frame, objects, pixel_radius=1, depth_tolerance_m=0.05)

        self.assertEqual(bound[0].status, "bound")
        self.assertEqual(bound[0].label_id, 3)
        self.assertEqual(bound[1].status, "depth-mismatch")
        self.assertIsNone(bound[1].label_id)

    def test_summarize_bound_candidates_groups_by_object_and_status(self):
        frame = RealSenseFrame(
            frame="0000",
            color=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_raw=np.full((2, 2), 1000, dtype=np.uint16),
            label=np.array([[1, 2], [0, 0]], dtype=np.uint8),
            intrinsic_matrix=np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        records = [
            {"score": 0.5, "width": 0.08, "translation": [0.0, 0.0, 1.0], "rotation_matrix": np.eye(3).tolist()},
            {"score": 0.8, "width": 0.08, "translation": [0.01, 0.0, 1.0], "rotation_matrix": np.eye(3).tolist()},
            {"score": 0.2, "width": 0.08, "translation": [10.0, 0.0, 1.0], "rotation_matrix": np.eye(3).tolist()},
        ]
        objects = [
            AnnotationObject(object_id=0, label_id=1, name="a.ply", position=np.zeros(3)),
            AnnotationObject(object_id=1, label_id=2, name="b.ply", position=np.zeros(3)),
        ]
        bound = bind_graspnet_records_to_frame_labels(records, frame, objects, pixel_radius=0, depth_tolerance_m=0.02)

        summary = summarize_bound_graspnet_candidates(bound, objects)

        self.assertEqual(summary["total_candidates"], 3)
        self.assertEqual(summary["bound_count"], 2)
        self.assertEqual(summary["status_counts"], {"bound": 2, "out-of-frame": 1})
        self.assertEqual(summary["objects"][0]["candidate_count"], 1)
        self.assertEqual(summary["objects"][1]["best_score"], 0.8)

    def test_prediction_source_reads_frame_records_from_zip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scene_0007.zip"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("scene_0007/realsense/0064.npy", _npy_bytes(_grasp_row(0.5, (0.0, 0.0, 1.0)).reshape(1, 17)))

            with GraspNetPredictionSource.open(path) as source:
                self.assertEqual(source.list_frames(), ["0064"])
                records = source.load_records("64")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["score"], 0.5)

    def test_bound_candidates_convert_to_grasp_poses_by_object(self):
        record = {
            "score": 0.9,
            "width": 0.04,
            "height": 0.02,
            "depth": 0.03,
            "rotation_matrix": np.eye(3).tolist(),
            "translation": [0.1, 0.2, 0.3],
            "object_id": -1,
        }
        bindings = [
            BoundGraspNetCandidate(
                record=record,
                frame="0000",
                status="bound",
                pixel=(10, 20),
                label_id=6,
                object_id=5,
                object_name="011_banana.ply",
                depth_error_m=0.0,
            ),
            BoundGraspNetCandidate(
                record=record,
                frame="0000",
                status="background",
                pixel=(0, 0),
                label_id=None,
                object_id=None,
                object_name=None,
                depth_error_m=None,
            ),
        ]

        poses_by_object = bound_candidates_to_grasp_poses_by_object(bindings)

        self.assertEqual(list(poses_by_object), ["011_banana.ply"])
        pose = poses_by_object["011_banana.ply"][0]
        self.assertEqual(pose.object_name, "011_banana.ply")
        self.assertEqual(pose.object_id, 5)
        self.assertEqual(pose.generator, "graspnet-bound")
        self.assertEqual(pose.closing_axis, "6d")
        self.assertAlmostEqual(pose.required_opening, 0.04)
        self.assertAlmostEqual(pose.score, 0.9)
        np.testing.assert_allclose(pose.position, np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(pose.pregrasp_position, np.array([-0.02, 0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()
