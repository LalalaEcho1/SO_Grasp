from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from stacked_grasping.gripper.feasibility import GraspCandidate, ObjectGraspFeasibility
from stacked_grasping.gripper.grasp_pose import graspnet_outputs_to_candidates
from stacked_grasping.gripper.graspnet_binding import BoundGraspNetCandidate
from stacked_grasping.relations.geometry import ObjectState


@dataclass(frozen=True)
class PointCloudCollisionConfig:
    max_opening: float = 0.085
    finger_width: float = 0.01
    finger_length: float = 0.06
    approach_distance: float = 0.05
    voxel_size: float = 0.005
    collision_threshold: float = 0.01
    empty_threshold: float = 0.01
    # When enabled, candidates whose predicted opening exceeds max_opening are not
    # rejected outright: their opening is clamped to max_opening and the clamped
    # geometry is re-checked for collision/empty instead. Default keeps the
    # historical hard-reject behaviour.
    clamp_width_to_max_opening: bool = False


@dataclass(frozen=True)
class PointCloudCandidateDiagnostic:
    collision_iou: float
    empty_ratio: float
    collision: bool
    empty: bool
    opening_too_small: bool
    opening_clamped: bool = False

    @property
    def feasible(self) -> bool:
        return not (self.collision or self.empty or self.opening_too_small)

    @property
    def reason(self) -> str | None:
        if self.opening_too_small:
            return "opening-too-small"
        if self.collision:
            return "pointcloud-collision"
        if self.empty:
            return "empty-grasp"
        return None


def assess_bound_graspnet_candidates_with_point_cloud(
    points: np.ndarray,
    bindings: Sequence[BoundGraspNetCandidate],
    *,
    config: PointCloudCollisionConfig | None = None,
    pregrasp_distance: float = 0.12,
    generator: str = "graspnet-pointcloud",
) -> list[ObjectGraspFeasibility]:
    cfg = config or PointCloudCollisionConfig()
    bound = [binding for binding in bindings if binding.status == "bound" and _binding_object_name(binding) is not None]
    if not bound:
        return []

    records = [_record_with_binding_object(binding) for binding in bound]
    diagnostics = diagnose_graspnet_pointcloud_collisions(points, records, config=cfg)

    grouped: dict[str, list[GraspCandidate]] = {}
    for binding, record, diagnostic in zip(bound, records, diagnostics):
        object_name = str(record["object_name"])
        pose = graspnet_outputs_to_candidates(
            [record],
            pregrasp_distance=pregrasp_distance,
            generator=generator,
        )[0]
        assessed_pose = replace(
            pose,
            feasible=diagnostic.feasible,
            failure_reason=diagnostic.reason,
            collision_objects=("pointcloud-scene",) if diagnostic.reason == "pointcloud-collision" else (),
        )
        grouped.setdefault(object_name, []).append(
            GraspCandidate(
                object_name=object_name,
                closing_axis="6d",
                required_opening=float(record["width"]),
                feasible=diagnostic.feasible,
                reason=diagnostic.reason,
                collision_objects=["pointcloud-scene"] if diagnostic.reason == "pointcloud-collision" else [],
                pose=assessed_pose,
            )
        )

    return [
        ObjectGraspFeasibility(
            object_name=object_name,
            feasible=sum(1 for candidate in candidates if candidate.feasible) > 0,
            feasible_grasp_count=sum(1 for candidate in candidates if candidate.feasible),
            candidates=candidates,
        )
        for object_name, candidates in grouped.items()
    ]


def assess_scene_bound_graspnet_pointcloud_feasibility(
    objects: Sequence[ObjectState],
    points: np.ndarray,
    bindings: Sequence[BoundGraspNetCandidate],
    *,
    config: PointCloudCollisionConfig | None = None,
    pregrasp_distance: float = 0.12,
    generator: str = "graspnet-pointcloud",
) -> list[ObjectGraspFeasibility]:
    assessed = assess_bound_graspnet_candidates_with_point_cloud(
        points,
        bindings,
        config=config,
        pregrasp_distance=pregrasp_distance,
        generator=generator,
    )
    by_name = {item.object_name: item for item in assessed}
    return [
        by_name.get(
            obj.name,
            ObjectGraspFeasibility(
                object_name=obj.name,
                feasible=False,
                feasible_grasp_count=0,
                candidates=[],
            ),
        )
        for obj in objects
    ]


def diagnose_graspnet_pointcloud_collisions(
    points: np.ndarray,
    records: Sequence[Mapping[str, object]],
    *,
    config: PointCloudCollisionConfig | None = None,
) -> list[PointCloudCandidateDiagnostic]:
    cfg = config or PointCloudCollisionConfig()
    if not records:
        return []

    scene_points = np.asarray(points, dtype=float).reshape(-1, 3)
    translations = np.array([record["translation"] for record in records], dtype=float)
    rotations = np.array([record["rotation_matrix"] for record in records], dtype=float).reshape(-1, 3, 3)
    heights = np.array([float(record.get("height", 0.02)) for record in records], dtype=float)[:, None]
    depths = np.array([float(record.get("depth", 0.03)) for record in records], dtype=float)[:, None]
    widths = np.array([float(record["width"]) for record in records], dtype=float)[:, None]
    over_opening = widths.reshape(-1) > cfg.max_opening
    if cfg.clamp_width_to_max_opening:
        # Re-check the clamped gripper geometry instead of hard-rejecting wide
        # predictions: the predicted width is a pre-close opening, so a gripper
        # approaching at its own max opening can still be valid when the object
        # itself fits between the fingers.
        widths = np.minimum(widths, float(cfg.max_opening))

    targets = scene_points[None, :, :] - translations[:, None, :]
    targets = np.matmul(targets, rotations)
    approach_dist = max(float(cfg.approach_distance), float(cfg.finger_width))

    mask_height = (targets[:, :, 2] > -heights / 2.0) & (targets[:, :, 2] < heights / 2.0)
    mask_depth = (targets[:, :, 0] > depths - cfg.finger_length) & (targets[:, :, 0] < depths)
    mask_left_outer = targets[:, :, 1] > -(widths / 2.0 + cfg.finger_width)
    mask_left_inner = targets[:, :, 1] < -widths / 2.0
    mask_right_outer = targets[:, :, 1] < widths / 2.0 + cfg.finger_width
    mask_right_inner = targets[:, :, 1] > widths / 2.0
    mask_bottom = (targets[:, :, 0] <= depths - cfg.finger_length) & (
        targets[:, :, 0] > depths - cfg.finger_length - cfg.finger_width
    )
    mask_shifting = (targets[:, :, 0] <= depths - cfg.finger_length - cfg.finger_width) & (
        targets[:, :, 0] > depths - cfg.finger_length - cfg.finger_width - approach_dist
    )

    left_mask = mask_height & mask_depth & mask_left_outer & mask_left_inner
    right_mask = mask_height & mask_depth & mask_right_outer & mask_right_inner
    bottom_mask = mask_height & mask_left_outer & mask_right_outer & mask_bottom
    shifting_mask = mask_height & mask_left_outer & mask_right_outer & mask_shifting
    global_mask = left_mask | right_mask | bottom_mask | shifting_mask

    voxel_volume = float(cfg.voxel_size) ** 3
    left_right_volume = (heights * cfg.finger_length * cfg.finger_width / voxel_volume).reshape(-1)
    bottom_volume = (heights * (widths + 2.0 * cfg.finger_width) * cfg.finger_width / voxel_volume).reshape(-1)
    shifting_volume = (heights * (widths + 2.0 * cfg.finger_width) * approach_dist / voxel_volume).reshape(-1)
    collision_volume = left_right_volume * 2.0 + bottom_volume + shifting_volume
    collision_iou = global_mask.sum(axis=1) / (collision_volume + 1e-6)

    inner_mask = mask_height & mask_depth & (~mask_left_inner) & (~mask_right_inner)
    inner_volume = (heights * cfg.finger_length * widths / voxel_volume).reshape(-1)
    empty_ratio = inner_mask.sum(axis=1) / (inner_volume + 1e-6)

    if cfg.clamp_width_to_max_opening:
        opening_bad = np.zeros(len(records), dtype=bool)
        opening_clamped = over_opening
    else:
        opening_bad = over_opening
        opening_clamped = np.zeros(len(records), dtype=bool)
    return [
        PointCloudCandidateDiagnostic(
            collision_iou=float(iou),
            empty_ratio=float(empty),
            collision=bool(iou > cfg.collision_threshold),
            empty=bool(empty < cfg.empty_threshold),
            opening_too_small=bool(is_opening_bad),
            opening_clamped=bool(is_clamped),
        )
        for iou, empty, is_opening_bad, is_clamped in zip(collision_iou, empty_ratio, opening_bad, opening_clamped)
    ]


def _record_with_binding_object(binding: BoundGraspNetCandidate) -> dict[str, object]:
    record = dict(binding.record)
    record["object_name"] = _binding_object_name(binding)
    if binding.object_id is not None:
        record["object_id"] = binding.object_id
    record.setdefault("closing_axis", "6d")
    return record


def _binding_object_name(binding: BoundGraspNetCandidate) -> str | None:
    if binding.object_name:
        return binding.object_name
    if binding.object_id is not None:
        return f"object_{binding.object_id}"
    if binding.label_id is not None:
        return f"label_{binding.label_id}"
    return None
