from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from tests import conftest  # noqa: F401
from scripts.export_graspnet_mujoco_scene import export_graspnet_mujoco_scene


def _write_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")


def _write_annotation(scene_root: Path, scene: str, frame: str) -> None:
    annotations = scene_root / scene / "realsense" / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    (annotations / f"{frame}.xml").write_text(
        """
        <scene>
          <obj>
            <obj_id>0</obj_id>
            <obj_name>003_cracker_box.ply</obj_name>
            <obj_path>models/003_cracker_box.ply</obj_path>
            <pos_in_world>0.1 0.2 0.3</pos_in_world>
            <ori_in_world>1 0 0 0</ori_in_world>
          </obj>
        </scene>
        """,
        encoding="utf-8",
    )


def _write_prediction(prediction_root: Path, scene: str, frame: str) -> None:
    prediction_dir = prediction_root / scene / "realsense"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    row = np.zeros((1, 17), dtype=np.float32)
    row[0, 0] = 0.9
    row[0, 1] = 0.04
    row[0, 2] = 0.02
    row[0, 3] = 0.03
    row[0, 4:13] = np.eye(3).reshape(-1)
    row[0, 13:16] = [0.11, 0.21, 0.51]
    row[0, 16] = -1
    np.save(prediction_dir / f"{frame}.npy", row)


class ExportGraspNetMujocoSceneTests(unittest.TestCase):
    def test_export_scene_writes_xml_with_top_prediction_gripper_pose(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_root = root / "scenes"
            dataset_root = root / "graspnet"
            prediction_root = root / "predictions"
            out_dir = root / "results"
            _write_obj(dataset_root / "models" / "000" / "textured.obj")
            _write_annotation(scene_root, "scene_0000", "0000")
            _write_prediction(prediction_root, "scene_0000", "0000")

            summary = export_graspnet_mujoco_scene(
                scene_root=scene_root,
                dataset_root=dataset_root,
                prediction_root=prediction_root,
                out_dir=out_dir,
                scene="scene_0000",
                frame="0000",
                compile_mujoco=False,
            )

            xml_path = Path(summary["xml_path"])
            self.assertTrue(xml_path.exists())
            self.assertEqual(summary["object_count"], 1)
            self.assertEqual(summary["selected_grasp_score"], 0.9)
            xml_root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
            gripper = xml_root.find(".//body[@name='robotiq_2f85_lite']")
            self.assertIsNotNone(gripper)
            self.assertEqual(gripper.attrib["pos"], "0.110000 0.210000 0.510000")

    def test_script_help_runs(self):
        script = conftest.PROJECT_ROOT / "scripts" / "export_graspnet_mujoco_scene.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=conftest.PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--prediction-root", result.stdout)
        self.assertIn("--scene", result.stdout)


if __name__ == "__main__":
    unittest.main()
