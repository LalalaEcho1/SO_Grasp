from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np

from stacked_grasping.relations.geometry import ObjectState


@dataclass(frozen=True)
class GraspPoseCandidate:
    object_name: str
    generator: str
    position: np.ndarray
    pregrasp_position: np.ndarray
    approach_direction: np.ndarray
    closing_axis: str
    orientation_quat_wxyz: np.ndarray
    required_opening: float
    score: float = 1.0
    feasible: bool = True
    failure_reason: str | None = None
    collision_objects: Tuple[str, ...] = ()
    object_id: int | str | None = None
    candidate_index: int | None = None
    pa_grasp_score: float | None = None
    original_score: float | None = None
    grasp_tolerance: float | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "object_name": self.object_name,
            "object_id": self.object_id,
            "candidate_index": self.candidate_index,
            "generator": self.generator,
            "position": self.position.round(6).tolist(),
            "pregrasp_position": self.pregrasp_position.round(6).tolist(),
            "approach_direction": self.approach_direction.round(6).tolist(),
            "closing_axis": self.closing_axis,
            "orientation_quat_wxyz": self.orientation_quat_wxyz.round(6).tolist(),
            "required_opening": round(self.required_opening, 6),
            "score": round(self.score, 6),
            "pa_grasp_score": _round_optional(self.pa_grasp_score),
            "original_score": _round_optional(self.original_score),
            "grasp_tolerance": _round_optional(self.grasp_tolerance),
            "feasible": self.feasible,
            "failure_reason": self.failure_reason,
            "collision_objects": list(self.collision_objects),
        }


class GraspPoseGenerator(Protocol):
    def generate_for_object(self, target: ObjectState) -> List[GraspPoseCandidate]:
        ...


class SceneGraspPoseGenerator(Protocol):
    def generate_for_scene(self, objects: Sequence[ObjectState]) -> Dict[str, List[GraspPoseCandidate]]:
        ...


@dataclass(frozen=True)
class RuleTopDownGraspPoseGenerator:
    lateral_clearance: float = 0.002
    approach_height: float = 0.12
    span_offsets: Tuple[float, ...] = (0.0,)

    def generate_for_object(self, target: ObjectState) -> List[GraspPoseCandidate]:
        return generate_topdown_grasp_pose_candidates(
            target,
            lateral_clearance=self.lateral_clearance,
            approach_height=self.approach_height,
            span_offsets=self.span_offsets,
        )

    def generate_for_scene(self, objects: Sequence[ObjectState]) -> Dict[str, List[GraspPoseCandidate]]:
        return {obj.name: self.generate_for_object(obj) for obj in objects}


@dataclass(frozen=True)
class MockGraspNetPoseGenerator:
    records: Sequence[Dict[str, object]]
    pregrasp_distance: float = 0.12
    margin: float = 0.0

    def generate_for_scene(self, objects: Sequence[ObjectState]) -> Dict[str, List[GraspPoseCandidate]]:
        candidates = graspnet_outputs_to_candidates(
            self.records,
            pregrasp_distance=self.pregrasp_distance,
            generator="graspnet-mock",
        )
        return assign_candidates_to_objects(objects, candidates, margin=self.margin)


def generate_topdown_grasp_pose_candidates(
    target: ObjectState,
    lateral_clearance: float = 0.002,
    approach_height: float = 0.12,
    span_offsets: Sequence[float] = (0.0,),
) -> List[GraspPoseCandidate]:
    approach_direction = np.array([0.0, 0.0, -1.0], dtype=float)
    offsets = tuple(float(offset) for offset in span_offsets) or (0.0,)
    candidates = []
    for closing_axis, yaw in (("x", 0.0), ("y", math.pi / 2.0)):
        close_axis = 0 if closing_axis == "x" else 1
        span_axis = 1 - close_axis
        for span_offset in offsets:
            if abs(span_offset) > float(target.half_extents[span_axis]) + 1e-9:
                continue
            position = np.array([target.position[0], target.position[1], target.top_z], dtype=float)
            position[span_axis] += span_offset
            pregrasp_position = position - approach_direction * approach_height
            candidates.append(
                _topdown_pose(
                    target=target,
                    closing_axis=closing_axis,
                    yaw=yaw,
                    position=position,
                    pregrasp_position=pregrasp_position,
                    approach_direction=approach_direction,
                    lateral_clearance=lateral_clearance,
                )
            )
    return candidates


def generate_side_grasp_pose_candidates(
    target: ObjectState,
    lateral_clearance: float = 0.002,
    pregrasp_distance: float = 0.12,
) -> List[GraspPoseCandidate]:
    position = np.array([target.position[0], target.position[1], target.position[2]], dtype=float)
    specs = (
        ("y", np.array([-1.0, 0.0, 0.0], dtype=float)),
        ("y", np.array([1.0, 0.0, 0.0], dtype=float)),
        ("x", np.array([0.0, -1.0, 0.0], dtype=float)),
        ("x", np.array([0.0, 1.0, 0.0], dtype=float)),
    )
    return [
        _side_pose(
            target=target,
            closing_axis=closing_axis,
            position=position,
            approach_direction=approach_direction,
            lateral_clearance=lateral_clearance,
            pregrasp_distance=pregrasp_distance,
        )
        for closing_axis, approach_direction in specs
    ]


def _topdown_pose(
    target: ObjectState,
    closing_axis: str,
    yaw: float,
    position: np.ndarray,
    pregrasp_position: np.ndarray,
    approach_direction: np.ndarray,
    lateral_clearance: float,
) -> GraspPoseCandidate:
    width_axis = 0 if closing_axis == "x" else 1
    required_opening = float(2.0 * target.half_extents[width_axis] + 2.0 * lateral_clearance)
    return GraspPoseCandidate(
        object_name=target.name,
        generator="rule-topdown",
        position=position.copy(),
        pregrasp_position=pregrasp_position.copy(),
        approach_direction=approach_direction.copy(),
        closing_axis=closing_axis,
        orientation_quat_wxyz=_yaw_quat_wxyz(yaw),
        required_opening=required_opening,
    )


def _side_pose(
    target: ObjectState,
    closing_axis: str,
    position: np.ndarray,
    approach_direction: np.ndarray,
    lateral_clearance: float,
    pregrasp_distance: float,
) -> GraspPoseCandidate:
    width_axis = 0 if closing_axis == "x" else 1
    required_opening = float(2.0 * target.half_extents[width_axis] + 2.0 * lateral_clearance)
    pregrasp_position = position - approach_direction * pregrasp_distance
    return GraspPoseCandidate(
        object_name=target.name,
        generator="rule-side",
        position=position.copy(),
        pregrasp_position=pregrasp_position.copy(),
        approach_direction=approach_direction.copy(),
        closing_axis=closing_axis,
        orientation_quat_wxyz=_orientation_quat_from_axes(closing_axis, approach_direction),
        required_opening=required_opening,
    )


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    half_yaw = yaw / 2.0
    return np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=float)


def _orientation_quat_from_axes(closing_axis: str, approach_direction: np.ndarray) -> np.ndarray:
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float) if closing_axis == "x" else np.array([0.0, 1.0, 0.0], dtype=float)
    z_axis = -np.array(approach_direction, dtype=float)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    y_axis = y_axis / y_norm
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return _rotation_matrix_to_quat_wxyz(rotation)


def graspnet_outputs_to_candidates(
    records: Sequence[Dict[str, object]],
    pregrasp_distance: float = 0.12,
    generator: str = "graspnet",
    default_object_name: str = "unassigned",
) -> List[GraspPoseCandidate]:
    candidates = []
    for record in records:
        rotation = np.array(record["rotation_matrix"], dtype=float).reshape(3, 3)
        position = np.array(record["translation"], dtype=float).reshape(3)
        approach_direction = rotation[:, 0]
        norm = float(np.linalg.norm(approach_direction))
        if norm > 1e-9:
            approach_direction = approach_direction / norm
        pregrasp_position = position - approach_direction * pregrasp_distance
        candidates.append(
            GraspPoseCandidate(
                object_name=str(record.get("object_name", default_object_name)),
                generator=generator,
                position=position,
                pregrasp_position=pregrasp_position,
                approach_direction=approach_direction,
                closing_axis=str(record.get("closing_axis", "6d")),
                orientation_quat_wxyz=_rotation_matrix_to_quat_wxyz(rotation),
                required_opening=float(record["width"]),
                score=float(record.get("score", 1.0)),
                object_id=record.get("object_id"),
                candidate_index=_optional_int(record.get("candidate_index")),
                pa_grasp_score=_optional_float(record.get("pa_grasp_score")),
                original_score=_optional_float(record.get("original_score")),
                grasp_tolerance=_optional_float(record.get("grasp_tolerance")),
            )
        )
    return candidates


def assign_candidates_to_objects(
    objects: Sequence[ObjectState],
    candidates: Sequence[GraspPoseCandidate],
    margin: float = 0.0,
) -> Dict[str, List[GraspPoseCandidate]]:
    assigned: Dict[str, List[GraspPoseCandidate]] = {obj.name: [] for obj in objects}
    for candidate in candidates:
        owner = _candidate_owner(objects, candidate, margin=margin)
        if owner is not None:
            assigned[owner.name].append(replace(candidate, object_name=owner.name))
    return assigned


def _candidate_owner(
    objects: Sequence[ObjectState],
    candidate: GraspPoseCandidate,
    margin: float,
) -> ObjectState | None:
    containing = []
    point = candidate.position
    for obj in objects:
        min_corner = obj.min_corner - margin
        max_corner = obj.max_corner + margin
        if bool(np.all(point >= min_corner) and np.all(point <= max_corner)):
            containing.append(obj)
    if not containing:
        return None
    return min(containing, key=lambda obj: float(np.linalg.norm(point - obj.position)))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _rotation_matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=float,
        )
    else:
        diag = np.diag(rotation)
        axis = int(np.argmax(diag))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=float,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=float,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=float,
            )

    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm
