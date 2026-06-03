from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

from stacked_grasping.relations.geometry import ObjectState, xy_distance, xy_overlap_area, xy_overlap_ratio
from stacked_grasping.relations.obstruction_degree import compute_obstruction_degree


@dataclass
class EdgeFeatures:
    source: str
    target: str
    contact: bool
    support_source_to_target: bool
    support_target_to_source: bool
    height_diff: float
    xy_distance: float
    xy_overlap_ratio: float
    vertical_gap: float
    od: float
    blocked_grasp_ratio: float
    od_mean: float
    od_max: float
    blocked_grasp_kinds: List[str]
    od_prior: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "contact": self.contact,
            "support_source_to_target": self.support_source_to_target,
            "support_target_to_source": self.support_target_to_source,
            "height_diff": round(self.height_diff, 6),
            "xy_distance": round(self.xy_distance, 6),
            "xy_overlap_ratio": round(self.xy_overlap_ratio, 6),
            "vertical_gap": round(self.vertical_gap, 6),
            "od": round(self.od, 6),
            "blocked_grasp_ratio": round(self.blocked_grasp_ratio, 6),
            "od_mean": round(self.od_mean, 6),
            "od_max": round(self.od_max, 6),
            "blocked_grasp_kinds": self.blocked_grasp_kinds,
            "od_prior": round(self.od_prior, 6),
        }


@dataclass
class RelationGraph:
    objects: List[ObjectState]
    edges: List[EdgeFeatures]

    @property
    def object_names(self) -> List[str]:
        return [obj.name for obj in self.objects]

    def incoming(self, name: str) -> List[EdgeFeatures]:
        return [edge for edge in self.edges if edge.target == name]

    def outgoing(self, name: str) -> List[EdgeFeatures]:
        return [edge for edge in self.edges if edge.source == name]


def build_relation_graph(
    objects: List[ObjectState],
    contact_pairs: Set[Tuple[str, str]],
    support_z_tolerance: float = 0.025,
    overlap_threshold: float = 0.04,
    support_area_threshold: float = 5e-4,
) -> RelationGraph:
    edges: List[EdgeFeatures] = []

    for source in objects:
        for target in objects:
            if source.name == target.name:
                continue

            pair = tuple(sorted((source.name, target.name)))
            contact = pair in contact_pairs
            overlap_area = xy_overlap_area(source, target)
            overlap = xy_overlap_ratio(source, target)
            distance_xy = xy_distance(source, target)
            height_diff = float(source.position[2] - target.position[2])

            source_supports_target = _supports(
                lower=source,
                upper=target,
                contact=contact,
                overlap_area=overlap_area,
                overlap=overlap,
                support_z_tolerance=support_z_tolerance,
                overlap_threshold=overlap_threshold,
                support_area_threshold=support_area_threshold,
            )
            target_supports_source = _supports(
                lower=target,
                upper=source,
                contact=contact,
                overlap_area=overlap_area,
                overlap=overlap,
                support_z_tolerance=support_z_tolerance,
                overlap_threshold=overlap_threshold,
                support_area_threshold=support_area_threshold,
            )

            vertical_gap = _vertical_gap(source, target)
            obstruction = compute_obstruction_degree(
                source,
                target,
                contact=contact,
                xy_distance=distance_xy,
            )
            od_prior = _obstruction_prior(
                source=source,
                target=target,
                contact=contact,
                overlap=overlap,
                distance_xy=distance_xy,
            )

            edges.append(
                EdgeFeatures(
                    source=source.name,
                    target=target.name,
                    contact=contact,
                    support_source_to_target=source_supports_target,
                    support_target_to_source=target_supports_source,
                    height_diff=height_diff,
                    xy_distance=distance_xy,
                    xy_overlap_ratio=overlap,
                    vertical_gap=vertical_gap,
                    od=obstruction.od,
                    blocked_grasp_ratio=obstruction.blocked_grasp_ratio,
                    od_mean=obstruction.od_mean,
                    od_max=obstruction.od_max,
                    blocked_grasp_kinds=obstruction.blocked_grasp_kinds,
                    od_prior=od_prior,
                )
            )

    return RelationGraph(objects=objects, edges=edges)


def _supports(
    lower: ObjectState,
    upper: ObjectState,
    contact: bool,
    overlap_area: float,
    overlap: float,
    support_z_tolerance: float,
    overlap_threshold: float,
    support_area_threshold: float,
) -> bool:
    has_enough_overlap = overlap >= overlap_threshold or overlap_area >= support_area_threshold
    if not has_enough_overlap:
        return False
    if lower.position[2] >= upper.position[2]:
        return False
    close_in_z = abs(lower.top_z - upper.bottom_z) <= support_z_tolerance
    return bool(contact or close_in_z)


def _vertical_gap(source: ObjectState, target: ObjectState) -> float:
    if source.position[2] <= target.position[2]:
        return float(target.bottom_z - source.top_z)
    return float(source.bottom_z - target.top_z)


def _obstruction_prior(
    source: ObjectState,
    target: ObjectState,
    contact: bool,
    overlap: float,
    distance_xy: float,
) -> float:
    """Temporary OD-like prior before grasp-candidate corridor checking exists."""
    near_factor = max(0.0, 1.0 - distance_xy / 0.25)
    above_factor = 1.0 if source.position[2] > target.position[2] else 0.0
    contact_factor = 1.0 if contact else 0.0
    value = 0.45 * overlap + 0.25 * contact_factor + 0.30 * above_factor * near_factor
    return max(0.0, min(1.0, float(value)))
