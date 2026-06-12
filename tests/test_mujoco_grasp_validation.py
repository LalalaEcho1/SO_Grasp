from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.gripper.graspnet_mujoco_scene import robotiq_lite_config_from_graspnet_candidate
from stacked_grasping.gripper.mujoco_grasp_validation import (
    FRANKA_HAND_GRIPPER_SPEC,
    LiteGraspValidationConfig,
    _gripper_approach_axis,
    _is_unstable_lift_delta,
    validate_grasp_xml,
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

    def test_unstable_lift_delta_detects_implausible_target_jump(self):
        cfg = LiteGraspValidationConfig(lift_distance=0.08, instability_lift_multiplier=3.0)

        self.assertFalse(_is_unstable_lift_delta(0.12, cfg))
        self.assertTrue(_is_unstable_lift_delta(0.30, cfg))

    def test_gripper_approach_axis_uses_lite_fingertip_direction(self):
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
        config = robotiq_lite_config_from_graspnet_candidate(candidate)

        np.testing.assert_allclose(
            _gripper_approach_axis(config.quat),
            candidate.approach_direction,
            atol=1e-6,
        )

    def test_validate_grasp_xml_supports_franka_hand_joint_names(self):
        xml = """\
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="0.5 0.5 0.01"/>
    <body name="target_box" pos="0 0 0.03">
      <freejoint name="target_box_freejoint"/>
      <geom name="target_box_geom" type="box" size="0.02 0.02 0.02" density="500"/>
    </body>
    <body name="franka_hand" pos="0 0 0.14">
      <freejoint name="franka_hand_freejoint"/>
      <geom name="franka_palm" type="box" size="0.03 0.01 0.02"/>
      <body name="left_finger" pos="0 0 0.04">
        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
        <geom name="left_finger_geom" type="box" size="0.004 0.004 0.03"/>
      </body>
      <body name="right_finger" pos="0 0 0.04">
        <joint name="finger_joint2" type="slide" axis="0 1 0" range="0 0.04"/>
        <geom name="right_finger_geom" type="box" size="0.004 0.004 0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "franka_hand_validation.xml"
            xml_path.write_text(xml, encoding="utf-8")

            result = validate_grasp_xml(
                xml_path,
                target_body_name="target_box",
                gripper_spec=FRANKA_HAND_GRIPPER_SPEC,
                config=LiteGraspValidationConfig(
                    settle_steps=1,
                    approach_steps=1,
                    close_steps=1,
                    lift_steps=1,
                    hold_steps=1,
                ),
            )

        self.assertTrue(result.compile_success)
        self.assertEqual(result.target_body_name, "target_box")
        self.assertNotEqual(result.failure_reason, "missing_gripper_freejoint")
        self.assertNotEqual(result.failure_reason, "missing_finger_slide_joints")


if __name__ == "__main__":
    unittest.main()
