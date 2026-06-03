from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests import conftest  # noqa: F401
from stacked_grasping.env.mujoco_scene import MujocoStackedScene
from stacked_grasping.gripper.graspnet_input import (
    depth_meters_to_uint16,
    graspnet_input_dir_for_scene,
    workspace_mask_from_depth,
    write_graspnet_input_bundle,
    write_mat_v4_numeric,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GraspNetInputExportTests(unittest.TestCase):
    def test_graspnet_input_dir_for_scene_uses_scene_stem(self):
        root = Path("outputs")

        out_dir = graspnet_input_dir_for_scene(root, "assets/scenes/generated_main_v1/scene_0034.xml")

        self.assertEqual(out_dir, root / "scene_0034")

    def test_depth_meters_to_uint16_uses_factor_depth_and_clips_invalid_values(self):
        depth = np.array([[0.0, 0.1234], [np.inf, 70.0]], dtype=float)

        encoded = depth_meters_to_uint16(depth, factor_depth=1000)

        self.assertEqual(encoded.dtype, np.uint16)
        self.assertEqual(encoded.tolist(), [[0, 123], [0, 65535]])

    def test_writes_graspnet_demo_input_bundle(self):
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        color[:, :, 0] = 255
        depth = np.array([[0.0, 0.5, 0.7], [0.3, np.inf, 1.2]], dtype=float)
        intrinsic = np.array([[500.0, 0.0, 1.0], [0.0, 500.0, 0.5], [0.0, 0.0, 1.0]], dtype=float)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            result = write_graspnet_input_bundle(
                out_dir,
                color=color,
                depth_meters=depth,
                intrinsic_matrix=intrinsic,
                factor_depth=1000,
                camera="overview",
                scene="scene_0000.xml",
                camera_to_world_matrix=np.eye(4, dtype=float),
            )

            self.assertTrue((out_dir / "color.png").exists())
            self.assertTrue((out_dir / "depth.png").exists())
            self.assertTrue((out_dir / "workspace_mask.png").exists())
            self.assertTrue((out_dir / "meta.mat").exists())
            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))

            depth_png = np.array(Image.open(out_dir / "depth.png"))
            mask_png = np.array(Image.open(out_dir / "workspace_mask.png"))

        self.assertEqual(result["width"], 3)
        self.assertEqual(result["height"], 2)
        self.assertEqual(metadata["camera"], "overview")
        self.assertEqual(metadata["scene"], "scene_0000.xml")
        self.assertEqual(metadata["factor_depth"], 1000)
        self.assertEqual(metadata["camera_frame"], "opencv")
        self.assertEqual(metadata["camera_to_world_matrix"], np.eye(4, dtype=float).tolist())
        self.assertEqual(depth_png.dtype, np.uint16)
        self.assertEqual(depth_png.tolist(), [[0, 500, 700], [300, 0, 1200]])
        self.assertEqual(mask_png.tolist(), [[0, 255, 255], [255, 0, 255]])

    def test_workspace_mask_from_depth_can_exclude_far_background(self):
        depth = np.array([[0.0, 0.5, 4.0], [np.nan, 2.9, 3.1]], dtype=float)

        mask = workspace_mask_from_depth(depth, max_depth_m=3.0)

        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.tolist(), [[False, True, False], [False, True, False]])

    def test_mat_v4_writer_contains_requested_variable_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "meta.mat"
            write_mat_v4_numeric(
                path,
                {
                    "intrinsic_matrix": np.eye(3, dtype=float),
                    "factor_depth": np.array([[1000.0]], dtype=float),
                },
            )
            data = path.read_bytes()

        self.assertIn(b"intrinsic_matrix\x00", data)
        self.assertIn(b"factor_depth\x00", data)

    def test_scene_camera_intrinsic_matrix_uses_named_camera_fovy(self):
        scene_path = PROJECT_ROOT / "assets" / "scenes" / "generated_main_v1" / "scene_0000.xml"
        if not scene_path.exists():
            self.skipTest("generated scene_0000.xml is not available")

        scene = MujocoStackedScene(scene_path)
        scene.reset_and_settle(steps=1)

        intrinsic = scene.camera_intrinsic_matrix(width=1280, height=720, camera="overview")
        camera_to_world = scene.camera_to_world_matrix(camera="overview")

        expected_focal = 0.5 * 720.0 / np.tan(np.deg2rad(62.0) * 0.5)
        np.testing.assert_allclose(
            intrinsic,
            np.array(
                [
                    [expected_focal, 0.0, 639.5],
                    [0.0, expected_focal, 359.5],
                    [0.0, 0.0, 1.0],
                ]
            ),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual(camera_to_world.shape, (4, 4))
        np.testing.assert_allclose(camera_to_world[3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.assertTrue(np.isfinite(camera_to_world).all())


if __name__ == "__main__":
    unittest.main()
