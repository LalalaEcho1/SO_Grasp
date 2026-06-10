from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.external_graspnet_data import (
    GraspNetRealSenseSource,
    assess_single_view_od_sufficiency,
    depth_to_point_cloud,
    parse_graspnet_annotation_xml,
    summarize_graspnet_prediction_package,
    summarize_realsense_frame,
)


def _png_bytes(array: np.ndarray, mode: str | None = None) -> bytes:
    buffer = io.BytesIO()
    image = Image.fromarray(array, mode=mode) if mode else Image.fromarray(array)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


class ExternalGraspNetDataTests(unittest.TestCase):
    def test_realsense_source_reads_common_frames_from_zip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "realsense.zip"
            with zipfile.ZipFile(path, "w") as zf:
                rgb = np.zeros((2, 3, 3), dtype=np.uint8)
                depth = np.array([[1000, 0, 2000], [500, 1000, 1500]], dtype=np.uint16)
                label = np.array([[1, 0, 2], [1, 2, 2]], dtype=np.uint8)
                intrinsic = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]])

                zf.writestr("scene_0000/realsense/rgb/0000.png", _png_bytes(rgb))
                zf.writestr("scene_0000/realsense/depth/0000.png", _png_bytes(depth, mode="I;16"))
                zf.writestr("scene_0000/realsense/label/0000.png", _png_bytes(label, mode="L"))
                zf.writestr("scene_0000/realsense/camK.npy", _npy_bytes(intrinsic))

            with GraspNetRealSenseSource.open(path) as source:
                self.assertEqual(source.root_prefix, "scene_0000/realsense")
                self.assertEqual(source.list_frames(), ["0000"])
                frame = source.load_frame("0000")

        self.assertEqual(frame.color.shape, (2, 3, 3))
        self.assertEqual(frame.depth_raw.dtype, np.uint16)
        np.testing.assert_allclose(frame.intrinsic_matrix, intrinsic)
        self.assertEqual(frame.label.tolist(), [[1, 0, 2], [1, 2, 2]])

    def test_depth_to_point_cloud_uses_factor_depth_and_intrinsic(self):
        depth = np.array([[1000, 0], [2000, 1000]], dtype=np.uint16)
        intrinsic = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])

        points, mask = depth_to_point_cloud(depth, intrinsic, factor_depth=1000)

        self.assertEqual(mask.tolist(), [[True, False], [True, True]])
        np.testing.assert_allclose(
            points,
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.02, 2.0],
                    [0.01, 0.01, 1.0],
                ],
                dtype=np.float32,
            ),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_summarize_realsense_frame_reports_labels_and_visible_edges(self):
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.array([[1000, 1200, 0], [1000, 1300, 900]], dtype=np.uint16)
        label = np.array([[1, 2, 0], [1, 2, 3]], dtype=np.uint8)
        intrinsic = np.eye(3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "realsense.zip"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("realsense/rgb/0000.png", _png_bytes(color))
                zf.writestr("realsense/depth/0000.png", _png_bytes(depth, mode="I;16"))
                zf.writestr("realsense/label/0000.png", _png_bytes(label, mode="L"))
                zf.writestr("realsense/camK.npy", _npy_bytes(intrinsic))
            with GraspNetRealSenseSource.open(path) as source:
                summary = summarize_realsense_frame(source.load_frame(0), min_boundary_pixels=1)

        self.assertEqual(summary["frame"], "0000")
        self.assertEqual(summary["point_cloud_points"], 5)
        self.assertEqual(summary["visible_label_count"], 3)
        self.assertEqual(summary["top_visible_labels"][0], {"label": 1, "pixels": 2})
        self.assertGreaterEqual(summary["visible_boundary_edge_count"], 2)

    def test_summarize_graspnet_prediction_package_reads_zip_graspgroup_arrays(self):
        row = np.array(
            [
                0.5,
                0.08,
                0.02,
                0.03,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.1,
                0.2,
                0.3,
                -1.0,
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scene_0007.zip"
            lower_score_row = row.copy()
            lower_score_row[0] = 0.25
            lower_score_row[16] = 2.0
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("scene_0007/realsense/0000.npy", _npy_bytes(np.stack([row, lower_score_row])))
                zf.writestr("scene_0007/realsense/0001.npy", _npy_bytes(row.reshape(1, 17)))

            summary = summarize_graspnet_prediction_package(path)

        self.assertEqual(summary["file_count"], 2)
        self.assertEqual(summary["total_grasps"], 3)
        self.assertEqual(summary["shape_each_unique"], [[1, 17], [2, 17]])
        self.assertEqual(summary["object_ids"], [-1, 2])
        self.assertAlmostEqual(summary["score"]["max"], 0.5)

    def test_parse_graspnet_annotation_xml_maps_object_ids_to_label_ids(self):
        xml = """
        <scene>
            <obj>
                <obj_id>0</obj_id>
                <obj_name>003_cracker_box.ply</obj_name>
                <obj_path>models/003_cracker_box.ply</obj_path>
                <pos_in_world>0.1 0.2 0.3</pos_in_world>
                <ori_in_world>0.5 0.5 0.5 0.5</ori_in_world>
            </obj>
            <obj>
                <obj_id>14</obj_id>
                <obj_name>015_peach.ply</obj_name>
                <pos_in_world>-0.1 0.0 0.4</pos_in_world>
            </obj>
        </scene>
        """

        objects = parse_graspnet_annotation_xml(xml)

        self.assertEqual([obj.object_id for obj in objects], [0, 14])
        self.assertEqual([obj.label_id for obj in objects], [1, 15])
        self.assertEqual(objects[0].name, "003_cracker_box.ply")
        self.assertEqual(objects[0].model_path, "models/003_cracker_box.ply")
        np.testing.assert_allclose(objects[0].orientation_quat_wxyz, np.array([0.5, 0.5, 0.5, 0.5]))
        np.testing.assert_allclose(objects[1].orientation_quat_wxyz, np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(objects[1].position, np.array([-0.1, 0.0, 0.4]))

    def test_assess_single_view_od_sufficiency_reports_hidden_and_unobservable_pairs(self):
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.array([[1000, 1100, 0], [1000, 1100, 0]], dtype=np.uint16)
        label = np.array([[1, 2, 0], [1, 2, 0]], dtype=np.uint8)
        frame = RealSenseFrameForTest(color=color, depth_raw=depth, label=label)
        objects = parse_graspnet_annotation_xml(
            """
            <scene>
                <obj><obj_id>0</obj_id><obj_name>a.ply</obj_name><pos_in_world>0 0 0</pos_in_world></obj>
                <obj><obj_id>1</obj_id><obj_name>b.ply</obj_name><pos_in_world>0 0 0.1</pos_in_world></obj>
                <obj><obj_id>2</obj_id><obj_name>c.ply</obj_name><pos_in_world>0 0 0.2</pos_in_world></obj>
            </scene>
            """
        )

        report = assess_single_view_od_sufficiency(frame, objects, min_boundary_pixels=1)

        self.assertEqual(report["complete_object_count"], 3)
        self.assertEqual(report["visible_object_count"], 2)
        self.assertEqual(report["hidden_label_ids"], [3])
        self.assertEqual(report["complete_pair_count"], 3)
        self.assertEqual(report["direct_visible_boundary_pair_count"], 1)
        self.assertEqual(report["hidden_object_pair_count"], 2)
        self.assertEqual(report["unobservable_pair_count"], 2)
        self.assertAlmostEqual(report["direct_pair_observability_ratio"], 1 / 3)
        self.assertFalse(report["single_view_sufficient_for_complete_od"])


class RealSenseFrameForTest:
    def __init__(self, color: np.ndarray, depth_raw: np.ndarray, label: np.ndarray):
        self.frame = "0000"
        self.color = color
        self.depth_raw = depth_raw
        self.label = label
        self.intrinsic_matrix = np.eye(3)
        self.factor_depth = 1000

    @property
    def depth_meters(self) -> np.ndarray:
        return self.depth_raw.astype(np.float32) / float(self.factor_depth)


if __name__ == "__main__":
    unittest.main()
