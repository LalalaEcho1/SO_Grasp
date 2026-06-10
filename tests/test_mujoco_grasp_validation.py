from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.mujoco_grasp_validation import (
    LiteGraspValidationConfig,
    validate_lite_grasp_xml,
)
from stacked_grasping.gripper.robotiq_2f85_lite import Robotiq2F85LiteConfig, build_gripper_body


def _minimal_validation_xml() -> str:
    gripper = build_gripper_body(
        Robotiq2F85LiteConfig(
            pos=(0.20, 0.0, 0.18),
            quat=(1.0, 0.0, 0.0, 0.0),
            include_freejoint=True,
            opening=0.06,
        )
    )
    import xml.etree.ElementTree as ET

    return f"""\
<mujoco model="lite_validation">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <material name="robotiq_dark" rgba="0.08 0.08 0.09 1"/>
    <material name="robotiq_pad" rgba="0.02 0.02 0.02 1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0.5 0.5 0.01"/>
    <body name="target_box" pos="0 0 0.04">
      <freejoint name="target_box_freejoint"/>
      <geom name="target_box_geom" type="box" size="0.03 0.03 0.03" density="500"/>
    </body>
    {ET.tostring(gripper, encoding="unicode")}
  </worldbody>
</mujoco>
"""


class MujocoGraspValidationTests(unittest.TestCase):
    def test_validate_lite_grasp_xml_reports_motion_and_lift_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "validation.xml"
            xml_path.write_text(_minimal_validation_xml(), encoding="utf-8")

            result = validate_lite_grasp_xml(
                xml_path,
                target_body_name="target_box",
                config=LiteGraspValidationConfig(
                    settle_steps=2,
                    approach_steps=3,
                    close_steps=3,
                    lift_steps=3,
                    hold_steps=2,
                    pregrasp_distance=0.02,
                    lift_distance=0.04,
                ),
            )

        payload = result.to_dict()
        self.assertTrue(result.compile_success)
        self.assertEqual(payload["target_body_name"], "target_box")
        self.assertEqual(payload["phase_step_count"], 13)
        self.assertIn("target_contact_step_count", payload)
        self.assertIn("lift_contact_step_count", payload)
        self.assertIn("target_lift_delta_m", payload)
        self.assertIn("max_target_lift_delta_m", payload)
        self.assertIn("max_target_z", payload)
        self.assertGreater(payload["gripper_lift_delta_m"], 0.03)
        self.assertIsInstance(payload["lift_success"], bool)

    def test_validate_lite_grasp_xml_reports_missing_gripper_freejoint(self):
        xml = """\
<mujoco>
  <worldbody>
    <body name="target_box"><geom type="box" size="0.02 0.02 0.02"/></body>
    <body name="robotiq_2f85_lite"><geom type="box" size="0.01 0.01 0.01"/></body>
  </worldbody>
</mujoco>
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "missing_freejoint.xml"
            xml_path.write_text(xml, encoding="utf-8")

            result = validate_lite_grasp_xml(xml_path, target_body_name="target_box")

        self.assertTrue(result.compile_success)
        self.assertEqual(result.failure_reason, "missing_gripper_freejoint")


if __name__ == "__main__":
    unittest.main()
