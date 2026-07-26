from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from stacked_grasping.gripper.grasp_pose import (
    GraspPoseCandidate,
    generate_side_grasp_pose_candidates,
    generate_topdown_grasp_pose_candidates,
)
from stacked_grasping.relations.geometry import ObjectState, aabb_intersection_volume, find_object


@dataclass(frozen=True)
class TopDownGraspConfig:
    max_opening: float = 0.085
    finger_thickness: float = 0.018
    finger_depth_margin: float = 0.012
    vertical_clearance: float = 0.002
    approach_height: float = 0.12
    span_offsets: Tuple[float, ...] = (0.0, -0.015, 0.015)
    include_side_grasps: bool = True
    target_assignment_margin: float = 0.03


@dataclass(frozen=True)
class OrientedBox:
    center: np.ndarray
    axes: np.ndarray
    half_extents: np.ndarray


@dataclass
class GraspCandidate:
    object_name: str
    closing_axis: str
    required_opening: float
    feasible: bool
    reason: str | None
    collision_objects: List[str]
    pose: GraspPoseCandidate | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "object_name": self.object_name,
            "closing_axis": self.closing_axis,
            "required_opening": round(self.required_opening, 6),
            "feasible": self.feasible,
            "reason": self.reason,
            "collision_objects": self.collision_objects,
            "pose": self.pose.to_dict() if self.pose else None,
        }


@dataclass
class ObjectGraspFeasibility:
    object_name: str
    feasible: bool
    feasible_grasp_count: int
    candidates: List[GraspCandidate]

    @property
    def gripper_collision_risk(self) -> float:
        if not self.candidates:
            return 1.0
        failed = sum(1 for candidate in self.candidates if not candidate.feasible)
        return failed / len(self.candidates)

    @property
    def selected_candidate(self) -> GraspCandidate | None:
        feasible_candidates = [candidate for candidate in self.candidates if candidate.feasible]
        if not feasible_candidates:
            return None
        return sorted(
            feasible_candidates,
            key=lambda candidate: (
                -_candidate_quality_score(candidate),
                float(candidate.required_opening),
                candidate.closing_axis,
            ),
        )[0]

    @property
    def best_grasp_score(self) -> float | None:
        return _max_feasible_pose_value(self.candidates, "score")

    @property
    def best_grasp_tolerance(self) -> float | None:
        return _max_feasible_pose_value(self.candidates, "grasp_tolerance")

    @property
    def graspnet_quality_score(self) -> float | None:
        return self.best_grasp_tolerance if self.best_grasp_tolerance is not None else self.best_grasp_score

    @property
    def selected_grasp_pose(self) -> GraspPoseCandidate | None:
        candidate = self.selected_candidate
        if candidate is None:
            return None
        return candidate.pose

    def to_dict(self) -> Dict[str, object]:
        return {
            "object_name": self.object_name,
            "feasible": self.feasible,
            "feasible_grasp_count": self.feasible_grasp_count,
            "gripper_collision_risk": round(self.gripper_collision_risk, 6),
            "best_grasp_score": _round_optional(self.best_grasp_score),
            "best_grasp_tolerance": _round_optional(self.best_grasp_tolerance),
            "graspnet_quality_score": _round_optional(self.graspnet_quality_score),
            "selected_candidate": self.selected_candidate.to_dict() if self.selected_candidate else None,
            "selected_grasp_pose": self.selected_grasp_pose.to_dict() if self.selected_grasp_pose else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _candidate_quality_score(candidate: GraspCandidate) -> float:
    pose = candidate.pose
    if pose is None:
        return 0.0
    if pose.grasp_tolerance is not None:
        return float(pose.grasp_tolerance)
    return float(pose.score)


def _max_feasible_pose_value(candidates: Sequence[GraspCandidate], attr: str) -> float | None:
    values = []
    for candidate in candidates:
        if not candidate.feasible or candidate.pose is None:
            continue
        value = getattr(candidate.pose, attr)
        if value is not None:
            values.append(float(value))
    return max(values) if values else None


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def assess_scene_topdown_grasps(
    objects: Sequence[ObjectState],
    config: TopDownGraspConfig | None = None,
) -> List[ObjectGraspFeasibility]:
    return [assess_object_topdown_grasps(objects, obj.name, config=config) for obj in objects]


def assess_scene_grasp_candidates(
    objects: Sequence[ObjectState],
    poses_by_object: Mapping[str, Sequence[GraspPoseCandidate]],
    config: TopDownGraspConfig | None = None,
) -> List[ObjectGraspFeasibility]:
    return [
        assess_object_grasp_candidates(
            objects,
            target_name=obj.name,
            poses=poses_by_object.get(obj.name, ()),
            config=config,
        )
        for obj in objects
    ]


def assess_object_topdown_grasps(
    objects: Sequence[ObjectState],
    target_name: str,
    config: TopDownGraspConfig | None = None,
) -> ObjectGraspFeasibility:
    cfg = config or TopDownGraspConfig()
    target = find_object(list(objects), target_name)
    poses = generate_topdown_grasp_pose_candidates(
        target,
        lateral_clearance=cfg.vertical_clearance,
        approach_height=cfg.approach_height,
        span_offsets=cfg.span_offsets,
    )
    if cfg.include_side_grasps:
        poses.extend(
            generate_side_grasp_pose_candidates(
                target,
                lateral_clearance=cfg.vertical_clearance,
                pregrasp_distance=cfg.approach_height,
            )
        )
    return assess_object_grasp_candidates(objects, target_name=target_name, poses=poses, config=cfg)


def assess_object_grasp_candidates(
    objects: Sequence[ObjectState],
    target_name: str,
    poses: Sequence[GraspPoseCandidate],
    config: TopDownGraspConfig | None = None,
) -> ObjectGraspFeasibility:
    cfg = config or TopDownGraspConfig()
    target = find_object(list(objects), target_name)
    candidates = [_assess_candidate(objects, target, pose=replace(pose, object_name=target.name), config=cfg) for pose in poses]
    feasible_count = sum(1 for candidate in candidates if candidate.feasible)
    return ObjectGraspFeasibility(
        object_name=target.name,
        feasible=feasible_count > 0,
        feasible_grasp_count=feasible_count,
        candidates=candidates,
    )


def _assess_candidate(
    objects: Sequence[ObjectState],
    target: ObjectState,
    pose: GraspPoseCandidate,
    config: TopDownGraspConfig,
) -> GraspCandidate:
    closing_axis = pose.closing_axis
    required_opening = pose.required_opening
    if closing_axis == "6d":
        return _assess_6d_candidate(objects, target, pose, config)

    if closing_axis not in {"x", "y"}:
        assessed_pose = replace(pose, feasible=False, failure_reason="unsupported-closing-axis")
        return GraspCandidate(
            object_name=target.name,
            closing_axis=closing_axis,
            required_opening=required_opening,
            feasible=False,
            reason="unsupported-closing-axis",
            collision_objects=[],
            pose=assessed_pose,
        )

    if required_opening > config.max_opening:
        assessed_pose = replace(pose, feasible=False, failure_reason="opening-too-small")
        return GraspCandidate(
            object_name=target.name,
            closing_axis=closing_axis,
            required_opening=required_opening,
            feasible=False,
            reason="opening-too-small",
            collision_objects=[],
            pose=assessed_pose,
        )

    reason, collision_objects = _candidate_collision(objects, target, pose, config)
    if reason == "unsupported-approach-direction":
        assessed_pose = replace(pose, feasible=False, failure_reason=reason)
        return GraspCandidate(
            object_name=target.name,
            closing_axis=closing_axis,
            required_opening=required_opening,
            feasible=False,
            reason=reason,
            collision_objects=[],
            pose=assessed_pose,
        )

    feasible = not collision_objects
    reason = None if feasible else reason
    assessed_pose = replace(
        pose,
        feasible=feasible,
        failure_reason=reason,
        collision_objects=tuple(collision_objects),
    )
    return GraspCandidate(
        object_name=target.name,
        closing_axis=closing_axis,
        required_opening=required_opening,
        feasible=feasible,
        reason=reason,
        collision_objects=collision_objects,
        pose=assessed_pose,
    )


def _assess_6d_candidate(
    objects: Sequence[ObjectState],
    target: ObjectState,
    pose: GraspPoseCandidate,
    config: TopDownGraspConfig,
) -> GraspCandidate:
    required_opening = pose.required_opening
    if required_opening > config.max_opening:
        assessed_pose = replace(pose, feasible=False, failure_reason="opening-too-small")
        return GraspCandidate(
            object_name=target.name,
            closing_axis=pose.closing_axis,
            required_opening=required_opening,
            feasible=False,
            reason="opening-too-small",
            collision_objects=[],
            pose=assessed_pose,
        )

    if not _pose_center_targets_object(target, pose, margin=config.target_assignment_margin):
        assessed_pose = replace(pose, feasible=False, failure_reason="target-mismatch")
        return GraspCandidate(
            object_name=target.name,
            closing_axis=pose.closing_axis,
            required_opening=required_opening,
            feasible=False,
            reason="target-mismatch",
            collision_objects=[],
            pose=assessed_pose,
        )

    reason, collision_objects = _sixd_candidate_collision(objects, target, pose, config)
    feasible = not collision_objects
    reason = None if feasible else reason
    assessed_pose = replace(
        pose,
        feasible=feasible,
        failure_reason=reason,
        collision_objects=tuple(collision_objects),
    )
    return GraspCandidate(
        object_name=target.name,
        closing_axis=pose.closing_axis,
        required_opening=required_opening,
        feasible=feasible,
        reason=reason,
        collision_objects=collision_objects,
        pose=assessed_pose,
    )


def _target_width(target: ObjectState, closing_axis: str) -> float:
    axis = 0 if closing_axis == "x" else 1
    return float(2.0 * target.half_extents[axis])


def _candidate_collision(
    objects: Sequence[ObjectState],
    target: ObjectState,
    pose: GraspPoseCandidate,
    config: TopDownGraspConfig,
) -> Tuple[str, List[str]]:
    if _is_topdown_pose(pose):
        return "finger-collision", _collision_objects(objects, target, _finger_boxes(target, pose, config))
    if _is_side_pose(pose):
        approach_collisions = _collision_objects(objects, target, [_side_approach_box(target, pose, config)])
        if approach_collisions:
            return "approach-collision", approach_collisions
        return "finger-collision", _collision_objects(objects, target, _side_finger_boxes(target, pose, config))
    return "unsupported-approach-direction", []


def _collision_objects(
    objects: Sequence[ObjectState],
    target: ObjectState,
    boxes: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> List[str]:
    collision_objects = []
    for obj in objects:
        if obj.name == target.name:
            continue
        for min_corner, max_corner in boxes:
            if aabb_intersection_volume(min_corner, max_corner, obj.min_corner, obj.max_corner) > 0.0:
                collision_objects.append(obj.name)
                break
    return sorted(collision_objects)


def _sixd_candidate_collision(
    objects: Sequence[ObjectState],
    target: ObjectState,
    pose: GraspPoseCandidate,
    config: TopDownGraspConfig,
) -> Tuple[str, List[str]]:
    approach_collisions = _collision_objects_with_oriented_boxes(
        objects,
        target,
        [_sixd_approach_box(target, pose, config)],
    )
    if approach_collisions:
        return "approach-collision", approach_collisions
    return "finger-collision", _collision_objects_with_oriented_boxes(
        objects,
        target,
        _sixd_finger_boxes(target, pose, config),
    )


def _collision_objects_with_oriented_boxes(
    objects: Sequence[ObjectState],
    target: ObjectState,
    boxes: Sequence[OrientedBox],
) -> List[str]:
    collision_objects = []
    for obj in objects:
        if obj.name == target.name:
            continue
        for box in boxes:
            if _oriented_box_intersects_aabb(box, obj.min_corner, obj.max_corner):
                collision_objects.append(obj.name)
                break
    return sorted(collision_objects)


def _pose_center_targets_object(target: ObjectState, pose: GraspPoseCandidate, margin: float) -> bool:
    point = np.asarray(pose.position, dtype=float)
    return bool(np.all(point >= target.min_corner - margin) and np.all(point <= target.max_corner + margin))


def _sixd_finger_boxes(
    target: ObjectState,
    pose: GraspPoseCandidate,
    config: TopDownGraspConfig,
) -> List[OrientedBox]:
    axes = _sixd_gripper_axes(pose)
    closing_axis = axes[:, 0]
    span_axis = axes[:, 1]
    approach_axis = axes[:, 2]
    target_span_radius = _aabb_project_radius(target, span_axis)
    target_approach_radius = _aabb_project_radius(target, approach_axis)
    finger_half_extents = np.array(
        [
            config.finger_thickness / 2.0,
            target_span_radius + config.finger_depth_margin,
            max(target_approach_radius + config.finger_depth_margin, config.finger_thickness / 2.0),
        ],
        dtype=float,
    )
    offset = pose.required_opening / 2.0 + config.finger_thickness / 2.0
    return [
        OrientedBox(
            center=np.asarray(pose.position, dtype=float) + sign * closing_axis * offset,
            axes=axes,
            half_extents=finger_half_extents,
        )
        for sign in (-1.0, 1.0)
    ]


def _sixd_approach_box(
    target: ObjectState,
    pose: GraspPoseCandidate,
    config: TopDownGraspConfig,
) -> OrientedBox:
    axes = _sixd_gripper_axes(pose)
    span_axis = axes[:, 1]
    start = np.asarray(pose.pregrasp_position, dtype=float)
    end = np.asarray(pose.position, dtype=float)
    distance = float(np.linalg.norm(end - start))
    return OrientedBox(
        center=(start + end) * 0.5,
        axes=axes,
        half_extents=np.array(
            [
                pose.required_opening / 2.0 + config.finger_thickness,
                _aabb_project_radius(target, span_axis) + config.finger_depth_margin,
                max(distance / 2.0, config.finger_thickness / 2.0),
            ],
            dtype=float,
        ),
    )


def _sixd_gripper_axes(pose: GraspPoseCandidate) -> np.ndarray:
    rotation = _quat_wxyz_to_rotation(np.asarray(pose.orientation_quat_wxyz, dtype=float))
    approach = _normalized(np.asarray(pose.approach_direction, dtype=float))
    if float(np.linalg.norm(approach)) <= 1e-9:
        approach = _normalized(rotation[:, 0])
    closing = rotation[:, 1] - approach * float(np.dot(rotation[:, 1], approach))
    closing = _normalized(closing)
    if float(np.linalg.norm(closing)) <= 1e-9:
        closing = _any_perpendicular(approach)
    span = _normalized(np.cross(approach, closing))
    if float(np.linalg.norm(span)) <= 1e-9:
        span = _any_perpendicular(closing)
    return np.column_stack((closing, span, approach))


def _aabb_project_radius(obj: ObjectState, axis: np.ndarray) -> float:
    return float(np.dot(np.abs(np.asarray(axis, dtype=float)), obj.half_extents))


def _oriented_box_intersects_aabb(box: OrientedBox, min_corner: np.ndarray, max_corner: np.ndarray) -> bool:
    aabb_center = (np.asarray(min_corner, dtype=float) + np.asarray(max_corner, dtype=float)) * 0.5
    aabb_half_extents = (np.asarray(max_corner, dtype=float) - np.asarray(min_corner, dtype=float)) * 0.5
    return _oriented_boxes_intersect(
        box,
        OrientedBox(center=aabb_center, axes=np.eye(3, dtype=float), half_extents=aabb_half_extents),
    )


def _oriented_boxes_intersect(a: OrientedBox, b: OrientedBox) -> bool:
    axes_a = np.asarray(a.axes, dtype=float)
    axes_b = np.asarray(b.axes, dtype=float)
    half_a = np.asarray(a.half_extents, dtype=float)
    half_b = np.asarray(b.half_extents, dtype=float)
    rotation = axes_a.T @ axes_b
    translation = axes_a.T @ (np.asarray(b.center, dtype=float) - np.asarray(a.center, dtype=float))
    abs_rotation = np.abs(rotation) + 1e-9

    for axis in range(3):
        radius_a = half_a[axis]
        radius_b = float(np.dot(half_b, abs_rotation[axis, :]))
        if abs(float(translation[axis])) > radius_a + radius_b:
            return False

    for axis in range(3):
        radius_a = float(np.dot(half_a, abs_rotation[:, axis]))
        radius_b = half_b[axis]
        if abs(float(np.dot(translation, rotation[:, axis]))) > radius_a + radius_b:
            return False

    for axis_a in range(3):
        for axis_b in range(3):
            other_a = [(axis_a + 1) % 3, (axis_a + 2) % 3]
            other_b = [(axis_b + 1) % 3, (axis_b + 2) % 3]
            radius_a = (
                half_a[other_a[0]] * abs_rotation[other_a[1], axis_b]
                + half_a[other_a[1]] * abs_rotation[other_a[0], axis_b]
            )
            radius_b = (
                half_b[other_b[0]] * abs_rotation[axis_a, other_b[1]]
                + half_b[other_b[1]] * abs_rotation[axis_a, other_b[0]]
            )
            distance = abs(
                float(
                    translation[other_a[1]] * rotation[other_a[0], axis_b]
                    - translation[other_a[0]] * rotation[other_a[1], axis_b]
                )
            )
            if distance > radius_a + radius_b:
                return False

    return True


def _quat_wxyz_to_rotation(quat: np.ndarray) -> np.ndarray:
    arr = np.asarray(quat, dtype=float).reshape(4)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.eye(3, dtype=float)
    w, x, y, z = arr / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _any_perpendicular(axis: np.ndarray) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(_normalized(vector), reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    return _normalized(np.cross(vector, reference))


def _is_topdown_pose(pose: GraspPoseCandidate) -> bool:
    direction = _normalized(pose.approach_direction)
    return bool(np.allclose(direction, np.array([0.0, 0.0, -1.0], dtype=float), atol=1e-6))


def _is_side_pose(pose: GraspPoseCandidate) -> bool:
    direction = _normalized(pose.approach_direction)
    horizontal = abs(float(direction[2])) <= 1e-6
    approach_axis = int(np.argmax(np.abs(direction[:2])))
    close_axis = 0 if pose.closing_axis == "x" else 1
    axis_aligned = abs(float(direction[approach_axis])) >= 1.0 - 1e-6
    return horizontal and axis_aligned and approach_axis != close_axis


def _normalized(vector: np.ndarray) -> np.ndarray:
    arr = np.array(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return arr
    return arr / norm


def _finger_boxes(
    target: ObjectState,
    pose: GraspPoseCandidate,
    config: TopDownGraspConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    closing_axis = pose.closing_axis
    close_axis = 0 if closing_axis == "x" else 1
    span_axis = 1 - close_axis

    boxes = []
    for sign in (-1.0, 1.0):
        min_corner = np.array(target.min_corner, dtype=float)
        max_corner = np.array(target.max_corner, dtype=float)
        center = float(pose.position[close_axis] + sign * (target.half_extents[close_axis] + config.finger_thickness / 2.0))
        min_corner[close_axis] = center - config.finger_thickness / 2.0
        max_corner[close_axis] = center + config.finger_thickness / 2.0
        min_corner[span_axis] = float(pose.position[span_axis] - target.half_extents[span_axis] - config.finger_depth_margin)
        max_corner[span_axis] = float(pose.position[span_axis] + target.half_extents[span_axis] + config.finger_depth_margin)
        min_corner[2] = float(target.bottom_z)
        max_corner[2] = float(target.top_z + config.approach_height)
        boxes.append((min_corner, max_corner))
    return boxes


def _side_finger_boxes(
    target: ObjectState,
    pose: GraspPoseCandidate,
    config: TopDownGraspConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    close_axis = 0 if pose.closing_axis == "x" else 1
    approach_axis = 1 - close_axis

    boxes = []
    for sign in (-1.0, 1.0):
        min_corner = np.array(target.min_corner, dtype=float)
        max_corner = np.array(target.max_corner, dtype=float)
        center = float(pose.position[close_axis] + sign * (target.half_extents[close_axis] + config.finger_thickness / 2.0))
        min_corner[close_axis] = center - config.finger_thickness / 2.0
        max_corner[close_axis] = center + config.finger_thickness / 2.0
        min_corner[approach_axis] = float(pose.position[approach_axis] - target.half_extents[approach_axis] - config.finger_depth_margin)
        max_corner[approach_axis] = float(pose.position[approach_axis] + target.half_extents[approach_axis] + config.finger_depth_margin)
        min_corner[2] = float(pose.position[2] - target.half_extents[2] - config.vertical_clearance)
        max_corner[2] = float(pose.position[2] + target.half_extents[2] + config.vertical_clearance)
        boxes.append((min_corner, max_corner))
    return boxes


def _side_approach_box(
    target: ObjectState,
    pose: GraspPoseCandidate,
    config: TopDownGraspConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    direction = _normalized(pose.approach_direction)
    approach_axis = int(np.argmax(np.abs(direction[:2])))
    close_axis = 0 if pose.closing_axis == "x" else 1
    target_side = target.position[approach_axis] - np.sign(direction[approach_axis]) * target.half_extents[approach_axis]

    min_corner = np.array(target.min_corner, dtype=float)
    max_corner = np.array(target.max_corner, dtype=float)
    min_corner[approach_axis] = float(min(target_side, pose.pregrasp_position[approach_axis]))
    max_corner[approach_axis] = float(max(target_side, pose.pregrasp_position[approach_axis]))
    min_corner[close_axis] = float(pose.position[close_axis] - target.half_extents[close_axis] - config.finger_thickness)
    max_corner[close_axis] = float(pose.position[close_axis] + target.half_extents[close_axis] + config.finger_thickness)
    min_corner[2] = float(pose.position[2] - target.half_extents[2] - config.vertical_clearance)
    max_corner[2] = float(pose.position[2] + target.half_extents[2] + config.vertical_clearance)
    return min_corner, max_corner
