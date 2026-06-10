from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.external_graspnet_data import AnnotationObject
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.gripper.graspnet_mujoco_scene import (
    build_graspnet_mujoco_scene_xml,
    mujoco_body_name_for_annotation,
    resolve_graspnet_model_mesh,
)


def _write_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
                "",
            ]
        ),
        encoding="utf-8",
    )


class GraspNetMujocoSceneTests(unittest.TestCase):
    def test_resolve_graspnet_model_mesh_uses_official_object_id_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "graspnet"
            mesh_path = dataset_root / "models" / "007" / "textured.obj"
            _write_obj(mesh_path)
            annotation = AnnotationObject(
                object_id=7,
                label_id=8,
                name="025_mug.ply",
                model_path="models/025_mug.ply",
                position=np.array([0.1, 0.2, 0.3]),
                orientation_quat_wxyz=np.array([0.5, 0.5, 0.5, 0.5]),
            )

            resolved = resolve_graspnet_model_mesh(annotation, dataset_root)

        self.assertEqual(resolved, mesh_path)

    def test_build_scene_xml_contains_annotated_objects_and_selected_gripper_pose(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "graspnet"
            _write_obj(dataset_root / "models" / "000" / "textured.obj")
            _write_obj(dataset_root / "models" / "001" / "textured.obj")
            output_path = root / "results" / "scene_0000_0000.xml"
            annotations = [
                AnnotationObject(
                    object_id=0,
                    label_id=1,
                    name="003_cracker_box.ply",
                    model_path="models/003_cracker_box.ply",
                    position=np.array([0.1, 0.2, 0.3]),
                    orientation_quat_wxyz=np.array([0.5, 0.5, 0.5, 0.5]),
                ),
                AnnotationObject(
                    object_id=1,
                    label_id=2,
                    name="004_sugar_box.ply",
                    model_path="models/004_sugar_box.ply",
                    position=np.array([-0.1, 0.0, 0.4]),
                    orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                ),
            ]
            candidate = GraspPoseCandidate(
                object_name="003_cracker_box.ply",
                generator="graspnet-bound",
                position=np.array([0.11, 0.21, 0.51]),
                pregrasp_position=np.array([0.01, 0.21, 0.51]),
                approach_direction=np.array([1.0, 0.0, 0.0]),
                closing_axis="6d",
                orientation_quat_wxyz=np.array([0.0, 0.0, 0.0, 1.0]),
                required_opening=0.04,
                score=0.9,
                object_id=0,
            )

            xml_text = build_graspnet_mujoco_scene_xml(
                annotations,
                dataset_root=dataset_root,
                output_path=output_path,
                selected_grasp=candidate,
            )

        xml_root = ET.fromstring(xml_text)
        first_body = xml_root.find(f".//body[@name='{mujoco_body_name_for_annotation(annotations[0])}']")
        first_mesh = xml_root.find(".//mesh[@name='obj_000_mesh']")
        gripper = xml_root.find(".//body[@name='robotiq_2f85_lite']")

        self.assertIsNotNone(first_body)
        self.assertEqual(first_body.attrib["pos"], "0.100000 0.200000 0.300000")
        self.assertEqual(first_body.attrib["quat"], "0.500000 0.500000 0.500000 0.500000")
        self.assertIsNotNone(first_body.find("freejoint"))
        self.assertIsNotNone(first_mesh)
        self.assertEqual(first_mesh.attrib["file"], "../graspnet/models/000/textured.obj")
        self.assertIsNotNone(gripper)
        self.assertEqual(gripper.attrib["pos"], "0.110000 0.210000 0.510000")
        self.assertEqual(gripper.attrib["quat"], "0.000000 0.000000 0.000000 1.000000")


if __name__ == "__main__":
    unittest.main()
