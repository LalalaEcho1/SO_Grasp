from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
import os
from pathlib import Path

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.franka_hand import (
    FRANKA_HAND_BODY_NAME,
    FRANKA_HAND_ROOT_JOINT_NAME,
    FrankaHandConfig,
    append_franka_hand_to_mujoco_root,
    franka_hand_config_from_graspnet_candidate,
    resolve_franka_hand_xml,
    _asset_file_for_xml,
)
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate


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
    <material name="white" rgba="1 1 1 1"/>
    <mesh name="hand_c" file="hand.stl"/>
  </asset>
  <worldbody>
    <body name="hand" quat="0 0 0 1">
      <inertial mass="0.73" pos="0 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
      <geom mesh="hand_c"/>
      <body name="left_finger" pos="0 0 0.0584">
        <joint name="finger_joint1" class="finger"/>
        <geom type="box" size="0.004 0.004 0.02"/>
      </body>
      <body name="right_finger" pos="0 0 0.0584">
        <joint name="finger_joint2" class="finger"/>
        <geom type="box" size="0.004 0.004 0.02"/>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="hand" body2="left_finger"/>
    <exclude body1="hand" body2="right_finger"/>
  </contact>
</mujoco>
""",
        encoding="utf-8",
    )


class FrankaHandTests(unittest.TestCase):
    def test_resolve_franka_hand_xml_prefers_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            explicit = Path(tmp_dir) / "franka" / "hand.xml"
            _write_minimal_hand_xml(explicit)

            resolved = resolve_franka_hand_xml(explicit_path=explicit)

        self.assertEqual(resolved, explicit)

    def test_franka_hand_config_maps_graspnet_approach_to_local_positive_z(self):
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

        config = franka_hand_config_from_graspnet_candidate(candidate, include_freejoint=True)

        rotation = _quat_wxyz_to_rotation_for_test(config.quat)
        np.testing.assert_allclose(rotation[:, 1], np.array([0.0, 1.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(rotation[:, 2], candidate.approach_direction, atol=1e-6)
        np.testing.assert_allclose(config.pos, np.array([0.097, 0.3, 0.4]), atol=1e-6)
        self.assertTrue(config.include_freejoint)

    def test_append_franka_hand_to_scene_rewrites_assets_and_adds_freejoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            hand_xml = root_dir / "franka" / "hand.xml"
            output_path = root_dir / "results" / "scene.xml"
            _write_minimal_hand_xml(hand_xml)
            scene_root = ET.fromstring(
                """\
<mujoco>
  <asset/>
  <worldbody/>
</mujoco>
"""
            )

            append_franka_hand_to_mujoco_root(
                scene_root,
                hand_xml_path=hand_xml,
                output_path=output_path,
                config=FrankaHandConfig(
                    pos=(0.1, 0.2, 0.3),
                    quat=(1.0, 0.0, 0.0, 0.0),
                    include_freejoint=True,
                ),
            )

        body = scene_root.find(f".//body[@name='{FRANKA_HAND_BODY_NAME}']")
        mesh = scene_root.find(".//mesh[@name='hand_c']")
        exclude = scene_root.find(".//exclude")

        self.assertIsNotNone(body)
        self.assertEqual(body.attrib["pos"], "0.100000 0.200000 0.300000")
        self.assertEqual(body.attrib["quat"], "1.000000 0.000000 0.000000 0.000000")
        self.assertIsNotNone(body.find(f"freejoint[@name='{FRANKA_HAND_ROOT_JOINT_NAME}']"))
        self.assertIsNotNone(mesh)
        self.assertEqual(mesh.attrib["file"], "../franka/assets/hand.stl")
        self.assertIsNotNone(exclude)
        self.assertEqual(exclude.attrib["body1"], FRANKA_HAND_BODY_NAME)

    @unittest.skipUnless(os.name == "nt", "Windows drive handling regression test")
    def test_asset_path_rewrite_falls_back_to_absolute_when_output_is_on_different_drive(self):
        rewritten = _asset_file_for_xml(
            "hand.stl",
            source_path=Path("D:/franka/hand.xml"),
            source_meshdir="assets",
            output_path=Path("C:/tmp/scene.xml"),
        )

        self.assertEqual(rewritten, "D:/franka/assets/hand.stl")


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
