from __future__ import annotations

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.gripper.robotiq_2f85 import (
    ROBOTIQ_2F85_BODY_NAME,
    ROBOTIQ_2F85_ROOT_JOINT_NAME,
    Robotiq2F85Config,
    append_robotiq_2f85_to_mujoco_root,
    resolve_robotiq_2f85_xml,
    robotiq_2f85_config_from_graspnet_candidate,
    _asset_file_for_xml,
)


def _write_minimal_2f85_xml(path: Path) -> None:
    assets = path.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in (
        "base_mount.stl",
        "base.stl",
        "driver.stl",
        "coupler.stl",
        "follower.stl",
        "pad.stl",
        "silicone_pad.stl",
        "spring_link.stl",
    ):
        (assets / name).write_text(
            "\n".join(
                [
                    f"solid {Path(name).stem}",
                    "facet normal 0 0 1",
                    "outer loop",
                    "vertex 0 0 0",
                    "vertex 0.01 0 0",
                    "vertex 0 0.01 0",
                    "endloop",
                    "endfacet",
                    f"endsolid {Path(name).stem}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    path.write_text(
        """\
<mujoco model="robotiq_2f85">
  <compiler angle="radian" meshdir="assets"/>
  <default>
    <default class="2f85">
      <mesh scale="0.001 0.001 0.001"/>
      <default class="driver">
        <joint type="hinge" axis="1 0 0" range="0 0.8"/>
      </default>
    </default>
  </default>
  <asset>
    <material name="black" rgba="0.149 0.149 0.149 1"/>
    <mesh name="base_mount" class="2f85" file="base_mount.stl"/>
  </asset>
  <worldbody>
    <body name="base_mount" pos="0 0 0.007" childclass="2f85">
      <geom mesh="base_mount" material="black"/>
      <body name="base" pos="0 0 0.0038">
        <site name="pinch" pos="0 0 0.145" size="0.005"/>
        <body name="right_driver">
          <joint name="right_driver_joint" class="driver"/>
        </body>
        <body name="left_driver">
          <joint name="left_driver_joint" class="driver"/>
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="base_mount" body2="right_driver"/>
  </contact>
  <tendon>
    <fixed name="split">
      <joint joint="right_driver_joint" coef="0.5"/>
      <joint joint="left_driver_joint" coef="0.5"/>
    </fixed>
  </tendon>
</mujoco>
""",
        encoding="utf-8",
    )


class Robotiq2F85Tests(unittest.TestCase):
    def test_resolve_robotiq_2f85_xml_prefers_project_menagerie_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            xml_path = root / "mujoco_menagerie" / "robotiq_2f85" / "2f85.xml"
            _write_minimal_2f85_xml(xml_path)

            resolved = resolve_robotiq_2f85_xml(root=root)

        self.assertEqual(resolved, xml_path)

    def test_robotiq_2f85_config_maps_graspnet_approach_to_local_positive_z(self):
        candidate = GraspPoseCandidate(
            object_name="box",
            generator="graspnet",
            position=np.array([0.2, 0.3, 0.4]),
            pregrasp_position=np.array([0.1, 0.3, 0.4]),
            approach_direction=np.array([1.0, 0.0, 0.0]),
            closing_axis="6d",
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            required_opening=0.04,
        )

        config = robotiq_2f85_config_from_graspnet_candidate(candidate, include_freejoint=True)

        rotation = _quat_wxyz_to_rotation_for_test(config.quat)
        np.testing.assert_allclose(rotation[:, 1], np.array([0.0, 1.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(rotation[:, 2], candidate.approach_direction, atol=1e-6)
        np.testing.assert_allclose(config.pos, np.array([0.0442, 0.3, 0.4]), atol=1e-6)
        self.assertTrue(config.include_freejoint)

    def test_append_robotiq_2f85_to_scene_rewrites_assets_and_adds_freejoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            gripper_xml = root_dir / "mujoco_menagerie" / "robotiq_2f85" / "2f85.xml"
            output_path = root_dir / "results" / "scene.xml"
            _write_minimal_2f85_xml(gripper_xml)
            scene_root = ET.fromstring(
                """\
<mujoco>
  <asset/>
  <worldbody/>
</mujoco>
"""
            )

            append_robotiq_2f85_to_mujoco_root(
                scene_root,
                gripper_xml_path=gripper_xml,
                output_path=output_path,
                config=Robotiq2F85Config(
                    pos=(0.1, 0.2, 0.3),
                    quat=(1.0, 0.0, 0.0, 0.0),
                    include_freejoint=True,
                ),
            )

        body = scene_root.find(f".//body[@name='{ROBOTIQ_2F85_BODY_NAME}']")
        mesh = scene_root.find(".//mesh[@name='base_mount']")
        exclude = scene_root.find(".//exclude")

        self.assertIsNotNone(body)
        self.assertEqual(body.attrib["pos"], "0.100000 0.200000 0.300000")
        self.assertEqual(body.attrib["quat"], "1.000000 0.000000 0.000000 0.000000")
        self.assertIsNotNone(body.find(f"freejoint[@name='{ROBOTIQ_2F85_ROOT_JOINT_NAME}']"))
        self.assertIsNotNone(mesh)
        self.assertEqual(mesh.attrib["file"], "../mujoco_menagerie/robotiq_2f85/assets/base_mount.stl")
        self.assertIsNotNone(exclude)
        self.assertEqual(exclude.attrib["body1"], ROBOTIQ_2F85_BODY_NAME)

    @unittest.skipUnless(os.name == "nt", "Windows drive handling regression test")
    def test_asset_path_rewrite_falls_back_to_absolute_when_output_is_on_different_drive(self):
        rewritten = _asset_file_for_xml(
            "base_mount.stl",
            source_path=Path("D:/menagerie/robotiq_2f85/2f85.xml"),
            source_meshdir="assets",
            output_path=Path("C:/tmp/scene.xml"),
        )

        self.assertEqual(rewritten, "D:/menagerie/robotiq_2f85/assets/base_mount.stl")


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


if __name__ == "__main__":
    unittest.main()
