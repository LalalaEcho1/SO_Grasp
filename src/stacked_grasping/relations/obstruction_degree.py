from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from stacked_grasping.grasp.grasp_candidates import GraspCandidate, generate_grasp_candidates
from stacked_grasping.relations.geometry import ObjectState, aabb_intersection_volume, aabb_volume


@dataclass
class ObstructionDegree:
    source: str
    target: str
    od: float
    blocked_grasp_ratio: float
    od_mean: float
    od_max: float
    candidate_scores: Dict[str, float]

    @property
    def blocked_grasp_kinds(self) -> List[str]:
        return [kind for kind, score in self.candidate_scores.items() if score > 0.05]


def compute_obstruction_degree(
    source: ObjectState,
    target: ObjectState,
    contact: bool = False,
    xy_distance: float | None = None,
    block_threshold: float = 0.05,
    influence_distance: float = 0.18,
) -> ObstructionDegree:
    candidates = generate_grasp_candidates(target)
    relation_weight = _relation_influence_weight(
        source=source,
        target=target,
        contact=contact,
        xy_distance=xy_distance,
        influence_distance=influence_distance,
    )
    scores = {
        candidate.kind: _candidate_obstruction_score(source, candidate) * relation_weight
        for candidate in candidates
    }

    values = np.array(list(scores.values()), dtype=float)
    blocked_ratio = float(np.mean(values > block_threshold)) if len(values) else 0.0
    od_mean = float(np.mean(values)) if len(values) else 0.0
    od_max = float(np.max(values)) if len(values) else 0.0

    # OD should reflect both how many grasps are blocked and how strongly blocked they are.
    od = float(np.clip(0.65 * blocked_ratio + 0.35 * od_mean, 0.0, 1.0))
    return ObstructionDegree(
        source=source.name,
        target=target.name,
        od=od,
        blocked_grasp_ratio=blocked_ratio,
        od_mean=od_mean,
        od_max=od_max,
        candidate_scores=scores,
    )


def _candidate_obstruction_score(source: ObjectState, candidate: GraspCandidate) -> float:
    overlap_volume = aabb_intersection_volume(
        source.min_corner,
        source.max_corner,
        candidate.corridor_min,
        candidate.corridor_max,
    )
    if overlap_volume <= 0.0:
        return 0.0

    source_volume = aabb_volume(source.min_corner, source.max_corner)
    corridor_volume = aabb_volume(candidate.corridor_min, candidate.corridor_max)
    denominator = max(min(source_volume, corridor_volume), 1e-9)
    return float(np.clip(overlap_volume / denominator, 0.0, 1.0))


def _relation_influence_weight(
    source: ObjectState,
    target: ObjectState,
    contact: bool,
    xy_distance: float | None,
    influence_distance: float,
) -> float:
    if contact:
        return 1.0

    if xy_distance is None:
        xy_distance = float(np.linalg.norm(source.position[:2] - target.position[:2]))

    distance_weight = max(0.0, 1.0 - xy_distance / influence_distance)
    vertical_overlap = min(source.top_z, target.top_z) - max(source.bottom_z, target.bottom_z)
    vertical_overlap_weight = 1.0 if vertical_overlap > 0.0 else 0.65
    return float(np.clip(distance_weight * vertical_overlap_weight, 0.0, 1.0))
