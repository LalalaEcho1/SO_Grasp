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
    robotiq_lite_config_from_graspnet_candidate,
)


def _write_minimal_hand_xml(path: Path) -> None:
    assets = path.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "hand.stl").write_text(
        "\n".join(
            [
                "solid hand",
                "facet normal 0 0 1",
                "outer loop",
                "vertex 0 0 0",
                "vertex 0.01 0 0",
                "vertex 0 0.01 0",
                "endloop",
                "endfacet",
                "endsolid hand",
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.write_text(
        """\
<mujoco model="panda hand">
  <compiler angle="radian" meshdir="assets"/>
  <default>
    <default class="finger">
      <joint axis="0 1 0" type="slide" range="0 0.04"/>
    </default>
  </default>
  <asset>
    <mesh name="hand_c" file="hand.stl"/>
  </asset>
  <worldbody>
    <body name="hand">
      <geom mesh="hand_c"/>
      <body name="left_finger"><joint name="finger_joint1" class="finger"/></body>
      <body name="right_finger"><joint name="finger_joint2" class="finger"/></body>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
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
        first_geom = first_body.find("geom")
        self.assertIsNotNone(first_geom)
        self.assertEqual(first_geom.attrib["friction"], "3.0 0.05 0.001")
        self.assertEqual(first_geom.attrib["condim"], "6")
        self.assertEqual(first_geom.attrib["solref"], "0.004 1")
        self.assertEqual(first_geom.attrib["solimp"], "0.95 0.99 0.001")
        self.assertIsNotNone(gripper)
        self.assertEqual(gripper.attrib["pos"], "0.110000 0.210000 0.510000")
        self.assertEqual(gripper.attrib["quat"], "0.000000 0.000000 0.000000 1.000000")

    def test_robotiq_lite_config_from_graspnet_candidate_maps_graspnet_axes_to_lite_geometry(self):
        candidate = GraspPoseCandidate(
            object_name="box",
            generator="graspnet",
            position=np.array([0.1, 0.2, 0.3]),
            pregrasp_position=np.array([0.0, 0.2, 0.3]),
            approach_direction=np.array([1.0, 0.0, 0.0]),
            closing_axis="6d",
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            required_opening=0.04,
        )

        config = robotiq_lite_config_from_graspnet_candidate(candidate, include_freejoint=True)

        self.assertTrue(config.include_freejoint)
        self.assertAlmostEqual(config.opening, 0.044)
        np.testing.assert_allclose(config.pos, np.array([-0.043, 0.2, 0.3]), atol=1e-6)
        rotation = _quat_wxyz_to_rotation_for_test(config.quat)
        np.testing.assert_allclose(rotation[:, 1], np.array([0.0, 1.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(-rotation[:, 2], np.array([1.0, 0.0, 0.0]), atol=1e-6)

    def test_build_scene_xml_can_attach_franka_hand_backend(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "graspnet"
            hand_xml = root / "franka" / "hand.xml"
            output_path = root / "results" / "scene_0000_0000.xml"
            _write_obj(dataset_root / "models" / "000" / "textured.obj")
            _write_minimal_hand_xml(hand_xml)
            annotations = [
                AnnotationObject(
                    object_id=0,
                    label_id=1,
                    name="003_cracker_box.ply",
                    model_path="models/003_cracker_box.ply",
                    position=np.array([0.1, 0.2, 0.3]),
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
                orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                required_opening=0.04,
                score=0.9,
                object_id=0,
            )

            xml_text = build_graspnet_mujoco_scene_xml(
                annotations,
                dataset_root=dataset_root,
                output_path=output_path,
                selected_grasp=candidate,
                gripper_backend="franka-hand",
                franka_hand_xml=hand_xml,
            )

        xml_root = ET.fromstring(xml_text)
        franka_hand = xml_root.find(".//body[@name='franka_hand']")
        hand_mesh = xml_root.find(".//mesh[@name='hand_c']")

        self.assertIsNotNone(franka_hand)
        self.assertIsNotNone(franka_hand.find("freejoint[@name='franka_hand_freejoint']"))
        self.assertIsNotNone(hand_mesh)
        self.assertEqual(hand_mesh.attrib["file"], "../franka/assets/hand.stl")


if __name__ == "__main__":
    unittest.main()


def _quat_wxyz_to_rotation_for_test(quat: np.ndarray | tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=float)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
