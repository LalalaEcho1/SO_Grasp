from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from stacked_grasping.relations.geometry import ObjectState


@dataclass
class GraspCandidate:
    target: str
    kind: str
    center: np.ndarray
    approach_direction: np.ndarray
    corridor_min: np.ndarray
    corridor_max: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "target": self.target,
            "kind": self.kind,
            "center": self.center.round(6).tolist(),
            "approach_direction": self.approach_direction.round(6).tolist(),
            "corridor_min": self.corridor_min.round(6).tolist(),
            "corridor_max": self.corridor_max.round(6).tolist(),
        }


def generate_grasp_candidates(
    obj: ObjectState,
    approach_length: float = 0.22,
    lateral_clearance: float = 0.025,
    vertical_clearance: float = 0.020,
) -> List[GraspCandidate]:
    """Generate simple top and side grasp approach corridors around an object AABB."""
    x, y, z = obj.position
    hx, hy, hz = obj.half_extents
    candidates: List[GraspCandidate] = []

    candidates.append(
        GraspCandidate(
            target=obj.name,
            kind="top",
            center=np.array([x, y, obj.top_z], dtype=float),
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=float),
            corridor_min=np.array([x - hx - lateral_clearance, y - hy - lateral_clearance, obj.top_z], dtype=float),
            corridor_max=np.array(
                [x + hx + lateral_clearance, y + hy + lateral_clearance, obj.top_z + approach_length],
                dtype=float,
            ),
        )
    )

    side_specs = [
        ("side_pos_x", np.array([-1.0, 0.0, 0.0]), np.array([obj.max_corner[0], y - hy - lateral_clearance, z - hz - vertical_clearance]), np.array([obj.max_corner[0] + approach_length, y + hy + lateral_clearance, z + hz + vertical_clearance])),
        ("side_neg_x", np.array([1.0, 0.0, 0.0]), np.array([obj.min_corner[0] - approach_length, y - hy - lateral_clearance, z - hz - vertical_clearance]), np.array([obj.min_corner[0], y + hy + lateral_clearance, z + hz + vertical_clearance])),
        ("side_pos_y", np.array([0.0, -1.0, 0.0]), np.array([x - hx - lateral_clearance, obj.max_corner[1], z - hz - vertical_clearance]), np.array([x + hx + lateral_clearance, obj.max_corner[1] + approach_length, z + hz + vertical_clearance])),
        ("side_neg_y", np.array([0.0, 1.0, 0.0]), np.array([x - hx - lateral_clearance, obj.min_corner[1] - approach_length, z - hz - vertical_clearance]), np.array([x + hx + lateral_clearance, obj.min_corner[1], z + hz + vertical_clearance])),
    ]
    for kind, direction, corridor_min, corridor_max in side_specs:
        candidates.append(
            GraspCandidate(
                target=obj.name,
                kind=kind,
                center=obj.position.copy(),
                approach_direction=direction.astype(float),
                corridor_min=corridor_min.astype(float),
                corridor_max=corridor_max.astype(float),
            )
        )

    return candidates

