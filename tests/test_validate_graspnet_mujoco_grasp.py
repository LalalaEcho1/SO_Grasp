from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import conftest  # noqa: F401
from scripts.validate_graspnet_mujoco_grasp import (
    select_target_annotation_for_grasp,
    validate_graspnet_mujoco_grasp,
)
from stacked_grasping.gripper.external_graspnet_data import AnnotationObject
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.gripper.mujoco_grasp_validation import LiteGraspValidationConfig


def _write_cube_obj(path: Path, half_extent: float = 0.02) -> None:
    h = half_extent
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"v {-h} {-h} {-h}",
                f"v {h} {-h} {-h}",
                f"v {h} {h} {-h}",
                f"v {-h} {h} {-h}",
                f"v {-h} {-h} {h}",
                f"v {h} {-h} {h}",
                f"v {h} {h} {h}",
                f"v {-h} {h} {h}",
                "f 1 2 3 4",
                "f 5 8 7 6",
                "f 1 5 6 2",
                "f 2 6 7 3",
                "f 3 7 8 4",
                "f 4 8 5 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


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
            <pos_in_world>0 0 0.03</pos_in_world>
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
    row[0, 13:16] = [0.0, 0.0, 0.08]
    row[0, 16] = -1
    np.save(prediction_dir / f"{frame}.npy", row)


def _write_alignment(scene_root: Path, scene: str) -> None:
    realsense = scene_root / scene / "realsense"
    realsense.mkdir(parents=True, exist_ok=True)
    camera_poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 1, axis=0)
    align = np.eye(4, dtype=np.float32)
    align[:3, 3] = [0.1, 0.0, 0.2]
    np.save(realsense / "camera_poses.npy", camera_poses)
    np.save(realsense / "cam0_wrt_table.npy", align)


class ValidateGraspNetMujocoGraspTests(unittest.TestCase):
    def test_select_target_annotation_prefers_object_id_when_valid(self):
        annotations = [
            AnnotationObject(object_id=0, label_id=1, name="a.ply", position=np.array([0.0, 0.0, 0.0])),
            AnnotationObject(object_id=2, label_id=3, name="b.ply", position=np.array([1.0, 0.0, 0.0])),
        ]
        candidate = GraspPoseCandidate(
            object_name="unknown",
            generator="test",
            position=np.array([0.0, 0.0, 0.0]),
            pregrasp_position=np.array([0.0, 0.0, 0.1]),
            approach_direction=np.array([1.0, 0.0, 0.0]),
            closing_axis="6d",
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            required_opening=0.04,
            object_id=2,
        )

        selected = select_target_annotation_for_grasp(annotations, candidate)

        self.assertEqual(selected.object_id, 2)

    def test_validate_graspnet_mujoco_grasp_exports_dynamic_xml_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_root = root / "scenes"
            dataset_root = root / "graspnet"
            prediction_root = root / "predictions"
            out_dir = root / "results"
            _write_cube_obj(dataset_root / "models" / "000" / "textured.obj")
            _write_annotation(scene_root, "scene_0000", "0000")
            _write_prediction(prediction_root, "scene_0000", "0000")

            summary = validate_graspnet_mujoco_grasp(
                scene_root=scene_root,
                dataset_root=dataset_root,
                prediction_root=prediction_root,
                out_dir=out_dir,
                scene="scene_0000",
                frame="0000",
                align_to_table=False,
                validation_config=LiteGraspValidationConfig(
                    settle_steps=2,
                    approach_steps=2,
                    close_steps=2,
                    lift_steps=2,
                    hold_steps=1,
                    pregrasp_distance=0.01,
                    lift_distance=0.02,
                ),
            )

            self.assertTrue(Path(summary["xml_path"]).exists())
            self.assertTrue(Path(summary["summary_path"]).exists())
            self.assertEqual(summary["target_object_id"], 0)
            self.assertEqual(summary["validation"]["compile_success"], True)
            self.assertIn("target_lift_delta_m", summary["validation"])

    def test_validate_graspnet_mujoco_grasp_can_align_to_table_frame(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_root = root / "scenes"
            dataset_root = root / "graspnet"
            prediction_root = root / "predictions"
            out_dir = root / "results"
            _write_cube_obj(dataset_root / "models" / "000" / "textured.obj")
            _write_annotation(scene_root, "scene_0000", "0000")
            _write_prediction(prediction_root, "scene_0000", "0000")
            _write_alignment(scene_root, "scene_0000")

            summary = validate_graspnet_mujoco_grasp(
                scene_root=scene_root,
                dataset_root=dataset_root,
                prediction_root=prediction_root,
                out_dir=out_dir,
                scene="scene_0000",
                frame="0000",
                align_to_table=True,
                validation_config=LiteGraspValidationConfig(
                    settle_steps=0,
                    approach_steps=0,
                    close_steps=0,
                    lift_steps=0,
                    hold_steps=0,
                ),
            )

        self.assertEqual(summary["coordinate_frame"], "table_aligned")
        self.assertEqual(summary["selected_grasp_position"], [0.13, 0.0, 0.28])
        self.assertEqual(summary["selected_grasp_depth"], 0.03)
        self.assertEqual(summary["target_position"], [0.1, 0.0, 0.23])

    def test_script_help_runs(self):
        script = conftest.PROJECT_ROOT / "scripts" / "validate_graspnet_mujoco_grasp.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=conftest.PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--lift-distance", result.stdout)
        self.assertIn("--gripper-opening-margin", result.stdout)
        self.assertIn("--gripper-backend", result.stdout)
        self.assertIn("--franka-hand-xml", result.stdout)
        self.assertIn("--robotiq-2f85-xml", result.stdout)


if __name__ == "__main__":
    unittest.main()
