from __future__ import annotations

import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.robotiq_2f85_lite import (
    Robotiq2F85LiteConfig,
    attach_gripper_to_scene_xml,
    build_gripper_body,
    rewrite_asset_file_paths,
)


MINIMAL_SCENE = """\
<mujoco model="minimal">
  <asset/>
  <worldbody>
    <body name="table" pos="0 0 0.35">
      <geom name="table_top" type="box" size="0.50 0.38 0.035"/>
    </body>
    <body name="obj_box" pos="0.10 0.20 0.50">
      <freejoint/>
      <geom name="obj_box_geom" type="box" size="0.04 0.05 0.06"/>
    </body>
  </worldbody>
</mujoco>
"""


class Robotiq2F85LiteTests(unittest.TestCase):
    def test_build_gripper_body_contains_two_sliding_fingers(self):
        body = build_gripper_body(Robotiq2F85LiteConfig(pos=(0.0, 0.0, 0.8), opening=0.07))

        self.assertEqual(body.attrib["name"], "robotiq_2f85_lite")
        self.assertFalse(body.attrib["name"].startswith("obj_"))
        self.assertIsNotNone(body.find(".//body[@name='robotiq_left_finger']"))
        self.assertIsNotNone(body.find(".//body[@name='robotiq_right_finger']"))
        self.assertIsNotNone(body.find(".//joint[@name='robotiq_left_slide']"))
        self.assertIsNotNone(body.find(".//joint[@name='robotiq_right_slide']"))

    def test_build_gripper_body_can_include_root_freejoint_for_dynamic_validation(self):
        body = build_gripper_body(Robotiq2F85LiteConfig(include_freejoint=True))

        root_freejoint = body.find("freejoint")

        self.assertIsNotNone(root_freejoint)
        self.assertEqual(root_freejoint.attrib["name"], "robotiq_root_freejoint")
        self.assertEqual(list(body)[0].tag, "freejoint")

    def test_build_gripper_body_adds_high_friction_contact_params_to_fingertips(self):
        body = build_gripper_body(
            Robotiq2F85LiteConfig(
                pad_friction="4.0 0.08 0.005",
                contact_condim=6,
                contact_solref="0.004 1",
                contact_solimp="0.95 0.99 0.001",
            )
        )

        left_pad = body.find(".//geom[@name='robotiq_left_finger_pad']")
        right_pad = body.find(".//geom[@name='robotiq_right_finger_pad']")

        self.assertIsNotNone(left_pad)
        self.assertIsNotNone(right_pad)
        for pad in (left_pad, right_pad):
            self.assertEqual(pad.attrib["friction"], "4.0 0.08 0.005")
            self.assertEqual(pad.attrib["condim"], "6")
            self.assertEqual(pad.attrib["solref"], "0.004 1")
            self.assertEqual(pad.attrib["solimp"], "0.95 0.99 0.001")

    def test_attach_gripper_to_scene_xml_places_gripper_above_target_object(self):
        xml_text = attach_gripper_to_scene_xml(
            MINIMAL_SCENE,
            target_object="obj_box",
            config=Robotiq2F85LiteConfig(approach_height=0.20, opening=0.08),
        )
        root = ET.fromstring(xml_text)
        gripper = root.find(".//body[@name='robotiq_2f85_lite']")

        self.assertIsNotNone(gripper)
        self.assertEqual(gripper.attrib["pos"], "0.100000 0.200000 0.760000")
        self.assertIsNotNone(root.find(".//material[@name='robotiq_dark']"))

    def test_add_gripper_script_help_runs_when_invoked_by_file_path(self):
        script_path = conftest.PROJECT_ROOT / "scripts" / "add_robotiq_2f85_lite.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=conftest.PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--target-object", result.stdout)

    def test_rewrite_asset_file_paths_makes_relative_meshes_relocatable(self):
        xml_text = """\
<mujoco>
  <asset>
    <mesh name="mesh_box" file="../../objects/ycb/box/textured.obj"/>
    <texture name="tex_box" type="2d" file="../../objects/ycb/box/texture.png"/>
  </asset>
</mujoco>
"""
        source_path = conftest.PROJECT_ROOT / "assets" / "scenes" / "generated_main_v1" / "scene.xml"
        output_path = conftest.PROJECT_ROOT / "results" / "gripper_check" / "scene.xml"

        relocated = rewrite_asset_file_paths(xml_text, source_path, output_path=output_path, mode="relative-to-output")
        root = ET.fromstring(relocated)

        mesh_file = root.find(".//mesh").attrib["file"]
        texture_file = root.find(".//texture").attrib["file"]
        self.assertEqual(mesh_file, "../../assets/objects/ycb/box/textured.obj")
        self.assertEqual(texture_file, "../../assets/objects/ycb/box/texture.png")


if __name__ == "__main__":
    unittest.main()
