from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.graspnet_predictions import (
    find_scene_prediction_file,
    load_scene_prediction_candidates,
    scene_key_from_path,
)
from stacked_grasping.relations.geometry import ObjectState


class GraspNetPredictionTests(unittest.TestCase):
    def test_scene_key_from_path_uses_xml_stem(self):
        self.assertEqual(scene_key_from_path("assets/scenes/generated_main_v1/scene_0034.xml"), "scene_0034")

    def test_prediction_file_locator_supports_official_dump_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prediction_path = root / "scene_0000" / "realsense" / "0000.npy"
            prediction_path.parent.mkdir(parents=True)
            np.save(prediction_path, np.zeros((0, 17), dtype=float))

            found = find_scene_prediction_file(root, "assets/scenes/generated_main_v1/scene_0000.xml")

        self.assertEqual(found, prediction_path)

    def test_load_scene_prediction_candidates_transforms_and_assigns_to_objects(self):
        record = np.zeros((1, 17), dtype=float)
        record[0, 0] = 0.92
        record[0, 1] = 0.046
        record[0, 4:13] = np.eye(3).reshape(-1)
        record[0, 13:16] = [0.0, 0.0, 0.0]
        record[0, 16] = -1
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = [0.10, -0.20, 0.50]
        objects = [
            ObjectState(
                name="obj_003_cracker_box",
                body_id=1,
                geom_id=1,
                geom_type="box",
                position=np.array([0.10, -0.20, 0.50], dtype=float),
                half_extents=np.array([0.05, 0.05, 0.05], dtype=float),
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prediction_path = root / "predictions" / "scene_0000" / "realsense" / "0000.npy"
            prediction_path.parent.mkdir(parents=True)
            np.save(prediction_path, record)
            metadata_dir = root / "inputs" / "scene_0000"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "metadata.json").write_text(
                json.dumps({"camera_to_world_matrix": transform.tolist()}),
                encoding="utf-8",
            )

            grouped = load_scene_prediction_candidates(
                root / "predictions",
                "assets/scenes/generated_main_v1/scene_0000.xml",
                objects=objects,
                metadata_root=root / "inputs",
            )

        self.assertEqual(list(grouped), ["obj_003_cracker_box"])
        self.assertEqual(len(grouped["obj_003_cracker_box"]), 1)
        candidate = grouped["obj_003_cracker_box"][0]
        self.assertEqual(candidate.generator, "graspnet-prediction")
        self.assertEqual(candidate.object_name, "obj_003_cracker_box")
        np.testing.assert_allclose(candidate.position, np.array([0.10, -0.20, 0.50]))
        self.assertEqual(candidate.score, 0.92)
        self.assertEqual(candidate.required_opening, 0.046)


if __name__ == "__main__":
    unittest.main()
