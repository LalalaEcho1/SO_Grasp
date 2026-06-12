from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


GRIPPER_BODY_NAME = "robotiq_2f85_lite"
GRIPPER_ROOT_JOINT_NAME = "robotiq_root_freejoint"
LEFT_SLIDE_JOINT_NAME = "robotiq_left_slide"
RIGHT_SLIDE_JOINT_NAME = "robotiq_right_slide"


@dataclass(frozen=True)
class GripperValidationSpec:
    root_body_name: str
    root_joint_name: str
    left_slide_joint_name: str
    right_slide_joint_name: str
    approach_axis_column: int
    approach_axis_sign: float
    left_open_limit: str = "upper"
    right_open_limit: str = "lower"
    left_closed_qpos: float = 0.0
    right_closed_qpos: float = 0.0


LITE_GRIPPER_SPEC = GripperValidationSpec(
    root_body_name=GRIPPER_BODY_NAME,
    root_joint_name=GRIPPER_ROOT_JOINT_NAME,
    left_slide_joint_name=LEFT_SLIDE_JOINT_NAME,
    right_slide_joint_name=RIGHT_SLIDE_JOINT_NAME,
    approach_axis_column=2,
    approach_axis_sign=-1.0,
    left_open_limit="upper",
    right_open_limit="lower",
)

FRANKA_HAND_GRIPPER_SPEC = GripperValidationSpec(
    root_body_name="franka_hand",
    root_joint_name="franka_hand_freejoint",
    left_slide_joint_name="finger_joint1",
    right_slide_joint_name="finger_joint2",
    approach_axis_column=2,
    approach_axis_sign=1.0,
    left_open_limit="upper",
    right_open_limit="upper",
)

ROBOTIQ_2F85_GRIPPER_SPEC = GripperValidationSpec(
    root_body_name="robotiq_2f85",
    root_joint_name="robotiq_2f85_freejoint",
    left_slide_joint_name="left_driver_joint",
    right_slide_joint_name="right_driver_joint",
    approach_axis_column=2,
    approach_axis_sign=1.0,
    left_open_limit="lower",
    right_open_limit="lower",
    left_closed_qpos=0.8,
    right_closed_qpos=0.8,
)


@dataclass(frozen=True)
class LiteGraspValidationConfig:
    settle_steps: int = 20
    approach_steps: int = 40
    close_steps: int = 40
    lift_steps: int = 60
    hold_steps: int = 20
    pregrasp_distance: float = 0.06
    lift_distance: float = 0.08
    lift_success_threshold_m: float = 0.02
    instability_lift_multiplier: float = 3.0


@dataclass(frozen=True)
class LiteGraspValidationResult:
    compile_success: bool
    target_body_name: str
    failure_reason: str | None = None
    phase_step_count: int = 0
    target_contact_step_count: int = 0
    lift_contact_step_count: int = 0
    initial_target_z: float | None = None
    final_target_z: float | None = None
    max_target_z: float | None = None
    target_lift_delta_m: float | None = None
    max_target_lift_delta_m: float | None = None
    initial_gripper_z: float | None = None
    final_gripper_z: float | None = None
    gripper_lift_delta_m: float | None = None
    simulation_unstable: bool = False
    lift_success: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "compile_success": self.compile_success,
            "target_body_name": self.target_body_name,
            "failure_reason": self.failure_reason,
            "phase_step_count": self.phase_step_count,
            "target_contact_step_count": self.target_contact_step_count,
            "lift_contact_step_count": self.lift_contact_step_count,
            "initial_target_z": _rounded_or_none(self.initial_target_z),
            "final_target_z": _rounded_or_none(self.final_target_z),
            "max_target_z": _rounded_or_none(self.max_target_z),
            "target_lift_delta_m": _rounded_or_none(self.target_lift_delta_m),
            "max_target_lift_delta_m": _rounded_or_none(self.max_target_lift_delta_m),
            "initial_gripper_z": _rounded_or_none(self.initial_gripper_z),
            "final_gripper_z": _rounded_or_none(self.final_gripper_z),
            "gripper_lift_delta_m": _rounded_or_none(self.gripper_lift_delta_m),
            "simulation_unstable": self.simulation_unstable,
            "lift_success": self.lift_success,
        }


def validate_lite_grasp_xml(
    xml_path: str | Path,
    *,
    target_body_name: str,
    config: LiteGraspValidationConfig | None = None,
) -> LiteGraspValidationResult:
    return validate_grasp_xml(
        xml_path,
        target_body_name=target_body_name,
        gripper_spec=LITE_GRIPPER_SPEC,
        config=config,
    )


def validate_grasp_xml(
    xml_path: str | Path,
    *,
    target_body_name: str,
    gripper_spec: GripperValidationSpec,
    config: LiteGraspValidationConfig | None = None,
) -> LiteGraspValidationResult:
    cfg = config or LiteGraspValidationConfig()
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
    except Exception as exc:
        return LiteGraspValidationResult(
            compile_success=False,
            target_body_name=target_body_name,
            failure_reason=f"compile_failed: {exc}",
        )

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    if target_body_id < 0:
        return LiteGraspValidationResult(
            compile_success=True,
            target_body_name=target_body_name,
            failure_reason="missing_target_body",
        )

    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gripper_spec.root_body_name)
    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gripper_spec.root_joint_name)
    if gripper_body_id < 0 or root_joint_id < 0:
        return LiteGraspValidationResult(
            compile_success=True,
            target_body_name=target_body_name,
            failure_reason="missing_gripper_freejoint",
        )

    left_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gripper_spec.left_slide_joint_name)
    right_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gripper_spec.right_slide_joint_name)
    if left_joint_id < 0 or right_joint_id < 0:
        return LiteGraspValidationResult(
            compile_success=True,
            target_body_name=target_body_name,
            failure_reason="missing_finger_slide_joints",
        )

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    qpos_addr = int(model.jnt_qposadr[root_joint_id])
    initial_grasp_pos = np.asarray(data.qpos[qpos_addr : qpos_addr + 3], dtype=float).copy()
    grasp_quat = _normalized_quat(data.qpos[qpos_addr + 3 : qpos_addr + 7])
    approach_axis = _gripper_approach_axis(grasp_quat, gripper_spec=gripper_spec)
    pregrasp_pos = initial_grasp_pos - approach_axis * float(cfg.pregrasp_distance)
    lift_pos = initial_grasp_pos + np.array([0.0, 0.0, float(cfg.lift_distance)], dtype=float)

    gripper_body_ids = _body_subtree_ids(model, gripper_body_id)
    open_qpos = _finger_open_qpos(model, left_joint_id, right_joint_id, gripper_spec=gripper_spec)
    closed_qpos = _finger_closed_qpos(model, left_joint_id, right_joint_id, gripper_spec=gripper_spec)

    initial_target_z = float(data.xpos[target_body_id, 2])
    max_target_z = initial_target_z
    initial_gripper_z = float(initial_grasp_pos[2])
    contact_steps = 0
    lift_contact_steps = 0
    phase_steps = 0

    for _ in range(int(cfg.settle_steps)):
        _set_gripper_pose(model, data, root_joint_id, pregrasp_pos, grasp_quat)
        _set_finger_qpos(model, data, left_joint_id, right_joint_id, open_qpos)
        mujoco.mj_step(model, data)
        phase_steps += 1
        contact_steps += _has_target_gripper_contact(model, data, target_body_id, gripper_body_ids)
        max_target_z = max(max_target_z, float(data.xpos[target_body_id, 2]))

    for alpha in _linspace01(cfg.approach_steps):
        pos = pregrasp_pos * (1.0 - alpha) + initial_grasp_pos * alpha
        _set_gripper_pose(model, data, root_joint_id, pos, grasp_quat)
        _set_finger_qpos(model, data, left_joint_id, right_joint_id, open_qpos)
        mujoco.mj_step(model, data)
        phase_steps += 1
        contact_steps += _has_target_gripper_contact(model, data, target_body_id, gripper_body_ids)
        max_target_z = max(max_target_z, float(data.xpos[target_body_id, 2]))

    for alpha in _linspace01(cfg.close_steps):
        finger_qpos = tuple(open_qpos[index] * (1.0 - alpha) + closed_qpos[index] * alpha for index in range(2))
        _set_gripper_pose(model, data, root_joint_id, initial_grasp_pos, grasp_quat)
        _set_finger_qpos(model, data, left_joint_id, right_joint_id, finger_qpos)
        mujoco.mj_step(model, data)
        phase_steps += 1
        contact_steps += _has_target_gripper_contact(model, data, target_body_id, gripper_body_ids)
        max_target_z = max(max_target_z, float(data.xpos[target_body_id, 2]))

    for alpha in _linspace01(cfg.lift_steps):
        pos = initial_grasp_pos * (1.0 - alpha) + lift_pos * alpha
        _set_gripper_pose(model, data, root_joint_id, pos, grasp_quat)
        _set_finger_qpos(model, data, left_joint_id, right_joint_id, closed_qpos)
        mujoco.mj_step(model, data)
        phase_steps += 1
        has_contact = _has_target_gripper_contact(model, data, target_body_id, gripper_body_ids)
        contact_steps += has_contact
        lift_contact_steps += has_contact
        max_target_z = max(max_target_z, float(data.xpos[target_body_id, 2]))

    for _ in range(int(cfg.hold_steps)):
        _set_gripper_pose(model, data, root_joint_id, lift_pos, grasp_quat)
        _set_finger_qpos(model, data, left_joint_id, right_joint_id, closed_qpos)
        mujoco.mj_step(model, data)
        phase_steps += 1
        has_contact = _has_target_gripper_contact(model, data, target_body_id, gripper_body_ids)
        contact_steps += has_contact
        lift_contact_steps += has_contact
        max_target_z = max(max_target_z, float(data.xpos[target_body_id, 2]))

    final_target_z = float(data.xpos[target_body_id, 2])
    final_gripper_z = float(data.xpos[gripper_body_id, 2])
    target_delta = final_target_z - initial_target_z
    max_target_delta = max_target_z - initial_target_z
    gripper_delta = final_gripper_z - initial_gripper_z
    simulation_unstable = _is_unstable_lift_delta(max_target_delta, cfg)
    lift_success = bool(
        not simulation_unstable and lift_contact_steps > 0 and target_delta >= float(cfg.lift_success_threshold_m)
    )
    failure_reason = None
    if not lift_success:
        if simulation_unstable:
            failure_reason = "simulation_unstable"
        else:
            failure_reason = "no_target_contact" if contact_steps == 0 else "insufficient_lift"

    return LiteGraspValidationResult(
        compile_success=True,
        target_body_name=target_body_name,
        failure_reason=failure_reason,
        phase_step_count=phase_steps,
        target_contact_step_count=contact_steps,
        lift_contact_step_count=lift_contact_steps,
        initial_target_z=initial_target_z,
        final_target_z=final_target_z,
        max_target_z=max_target_z,
        target_lift_delta_m=target_delta,
        max_target_lift_delta_m=max_target_delta,
        initial_gripper_z=initial_gripper_z,
        final_gripper_z=final_gripper_z,
        gripper_lift_delta_m=gripper_delta,
        simulation_unstable=simulation_unstable,
        lift_success=lift_success,
    )


def _set_gripper_pose(model, data, joint_id: int, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    qpos_addr = int(model.jnt_qposadr[joint_id])
    qvel_addr = int(model.jnt_dofadr[joint_id])
    current_pos = np.asarray(data.qpos[qpos_addr : qpos_addr + 3], dtype=float).copy()
    linear_velocity = _clipped_velocity(np.asarray(pos, dtype=float).reshape(3) - current_pos, _model_timestep(model))
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = _normalized_quat(quat_wxyz)
    data.qvel[qvel_addr : qvel_addr + 3] = linear_velocity
    data.qvel[qvel_addr + 3 : qvel_addr + 6] = 0.0


def _set_finger_qpos(model, data, left_joint_id: int, right_joint_id: int, values: tuple[float, float]) -> None:
    for joint_id, value in ((left_joint_id, values[0]), (right_joint_id, values[1])):
        qpos_addr = int(model.jnt_qposadr[joint_id])
        qvel_addr = int(model.jnt_dofadr[joint_id])
        velocity = _clipped_scalar_velocity(float(value) - float(data.qpos[qpos_addr]), _model_timestep(model))
        data.qpos[qpos_addr] = float(value)
        data.qvel[qvel_addr] = velocity


def _finger_open_qpos(
    model,
    left_joint_id: int,
    right_joint_id: int,
    *,
    gripper_spec: GripperValidationSpec,
) -> tuple[float, float]:
    return (
        _joint_limit_value(model, left_joint_id, gripper_spec.left_open_limit),
        _joint_limit_value(model, right_joint_id, gripper_spec.right_open_limit),
    )


def _finger_closed_qpos(
    model,
    left_joint_id: int,
    right_joint_id: int,
    *,
    gripper_spec: GripperValidationSpec,
) -> tuple[float, float]:
    left_range = model.jnt_range[left_joint_id]
    right_range = model.jnt_range[right_joint_id]
    return (_clamp(gripper_spec.left_closed_qpos, left_range), _clamp(gripper_spec.right_closed_qpos, right_range))


def _has_target_gripper_contact(model, data, target_body_id: int, gripper_body_ids: set[int]) -> int:
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        body_a = int(model.geom_bodyid[contact.geom1])
        body_b = int(model.geom_bodyid[contact.geom2])
        if (body_a == target_body_id and body_b in gripper_body_ids) or (
            body_b == target_body_id and body_a in gripper_body_ids
        ):
            return 1
    return 0


def _gripper_approach_axis(
    quat_wxyz: Iterable[float],
    *,
    gripper_spec: GripperValidationSpec = LITE_GRIPPER_SPEC,
) -> np.ndarray:
    rotation = _quat_wxyz_to_rotation(quat_wxyz)
    return float(gripper_spec.approach_axis_sign) * rotation[:, int(gripper_spec.approach_axis_column)]


def _body_subtree_ids(model, root_body_id: int) -> set[int]:
    body_ids = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            parent = int(model.body_parentid[body_id])
            if parent in body_ids and body_id not in body_ids:
                body_ids.add(int(body_id))
                changed = True
    return body_ids


def _quat_wxyz_to_rotation(quat: Iterable[float]) -> np.ndarray:
    w, x, y, z = _normalized_quat(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _normalized_quat(values: Iterable[float]) -> np.ndarray:
    quat = np.asarray(list(values), dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def _linspace01(step_count: int) -> np.ndarray:
    count = max(int(step_count), 0)
    if count == 0:
        return np.empty(0, dtype=float)
    if count == 1:
        return np.array([1.0], dtype=float)
    return np.linspace(0.0, 1.0, count, dtype=float)


def _clamp(value: float, limits: np.ndarray) -> float:
    return float(min(max(value, float(limits[0])), float(limits[1])))


def _joint_limit_value(model, joint_id: int, side: str) -> float:
    if side == "lower":
        return float(model.jnt_range[joint_id, 0])
    if side == "upper":
        return float(model.jnt_range[joint_id, 1])
    raise ValueError(f"Unsupported joint limit side: {side}")


def _model_timestep(model) -> float:
    timestep = float(model.opt.timestep)
    return timestep if timestep > 1e-9 else 1.0


def _clipped_velocity(delta: np.ndarray, timestep: float, max_speed: float = 1.0) -> np.ndarray:
    velocity = np.asarray(delta, dtype=float).reshape(3) / float(timestep)
    speed = float(np.linalg.norm(velocity))
    if speed <= float(max_speed) or speed <= 1e-9:
        return velocity
    return velocity / speed * float(max_speed)


def _clipped_scalar_velocity(delta: float, timestep: float, max_speed: float = 1.0) -> float:
    velocity = float(delta) / float(timestep)
    return float(min(max(velocity, -float(max_speed)), float(max_speed)))


def _is_unstable_lift_delta(max_target_lift_delta_m: float, config: LiteGraspValidationConfig) -> bool:
    threshold = abs(float(config.lift_distance)) * max(float(config.instability_lift_multiplier), 1.0)
    return abs(float(max_target_lift_delta_m)) > threshold


def _rounded_or_none(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)
