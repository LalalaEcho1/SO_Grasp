from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from stacked_grasping.gripper.external_graspnet_data import (
    AnnotationObject,
    RealSenseFrame,
    depth_to_point_cloud,
    visible_boundary_edges,
)
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.gripper.graspnet_binding import BoundGraspNetCandidate, bound_candidates_to_grasp_poses_by_object
from stacked_grasping.relations.geometry import ObjectState


@dataclass(frozen=True)
class ExternalGraspNetEpisodeInputs:
    scene: "ExternalGraspNetFrameScene"
    grasp_poses_by_object: dict[str, list[GraspPoseCandidate]]


class ExternalGraspNetFrameScene:
    """Minimal episode-compatible scene backed by one labeled GraspNet frame."""

    def __init__(self, objects: Sequence[ObjectState], contact_pairs: Sequence[tuple[str, str]] | set[tuple[str, str]] = ()):
        self._objects = list(objects)
        self._contact_pairs = {_sorted_pair(pair) for pair in contact_pairs}
        self.removed: list[str] = []

    def read_objects(self) -> list[ObjectState]:
        return list(self._objects)

    def read_object_contact_pairs(self) -> set[tuple[str, str]]:
        active = {obj.name for obj in self._objects}
        return {pair for pair in self._contact_pairs if pair[0] in active and pair[1] in active}

    def remove_object(self, name: str) -> None:
        self.removed.append(name)
        self._objects = [obj for obj in self._objects if obj.name != name]

    def settle(self, steps: int) -> None:
        return None


def build_external_graspnet_episode_inputs(
    frame: RealSenseFrame,
    annotation_objects: Sequence[AnnotationObject],
    bindings: Sequence[BoundGraspNetCandidate],
    *,
    min_points_per_object: int = 20,
    min_half_extent: float = 0.005,
    padding: float = 0.002,
    min_boundary_pixels: int = 50,
) -> ExternalGraspNetEpisodeInputs:
    objects = objects_from_labeled_point_cloud(
        frame,
        annotation_objects,
        min_points_per_object=min_points_per_object,
        min_half_extent=min_half_extent,
        padding=padding,
    )
    contact_pairs = visible_boundary_contact_pairs(
        frame,
        annotation_objects,
        min_boundary_pixels=min_boundary_pixels,
    )
    active_names = {obj.name for obj in objects}
    poses_by_object = {
        name: list(poses)
        for name, poses in bound_candidates_to_grasp_poses_by_object(bindings).items()
        if name in active_names
    }
    return ExternalGraspNetEpisodeInputs(
        scene=ExternalGraspNetFrameScene(objects, contact_pairs),
        grasp_poses_by_object=poses_by_object,
    )


def objects_from_labeled_point_cloud(
    frame: RealSenseFrame,
    annotation_objects: Sequence[AnnotationObject],
    *,
    min_points_per_object: int = 20,
    min_half_extent: float = 0.005,
    padding: float = 0.002,
) -> list[ObjectState]:
    if frame.label is None:
        raise ValueError("External GraspNet scene bridge requires a label image.")

    points, valid_mask = depth_to_point_cloud(
        frame.depth_raw,
        frame.intrinsic_matrix,
        factor_depth=frame.factor_depth,
    )
    labels = np.asarray(frame.label).reshape(-1)[valid_mask.reshape(-1)]

    objects: list[ObjectState] = []
    min_extent = np.full(3, float(min_half_extent), dtype=float)
    for annotation in annotation_objects:
        object_points = points[labels == annotation.label_id]
        if object_points.shape[0] < int(min_points_per_object):
            continue
        min_corner = object_points.min(axis=0)
        max_corner = object_points.max(axis=0)
        center = (min_corner + max_corner) * 0.5
        half_extents = np.maximum((max_corner - min_corner) * 0.5 + float(padding), min_extent)
        objects.append(
            ObjectState(
                name=annotation.name,
                body_id=annotation.object_id,
                geom_id=annotation.label_id,
                geom_type="external-label-aabb",
                position=center.astype(float),
                half_extents=half_extents.astype(float),
            )
        )
    return objects


def visible_boundary_contact_pairs(
    frame: RealSenseFrame,
    annotation_objects: Sequence[AnnotationObject],
    *,
    min_boundary_pixels: int = 50,
) -> set[tuple[str, str]]:
    if frame.label is None:
        return set()

    label_to_name = {obj.label_id: obj.name for obj in annotation_objects}
    pairs: set[tuple[str, str]] = set()
    for edge in visible_boundary_edges(
        frame.label,
        frame.depth_meters,
        min_boundary_pixels=min_boundary_pixels,
    ):
        label_a, label_b = [int(value) for value in edge["pair"]]
        name_a = label_to_name.get(label_a)
        name_b = label_to_name.get(label_b)
        if name_a is not None and name_b is not None:
            pairs.add(_sorted_pair((name_a, name_b)))
    return pairs


def _sorted_pair(pair: tuple[str, str]) -> tuple[str, str]:
    return tuple(sorted(pair))  # type: ignore[return-value]
