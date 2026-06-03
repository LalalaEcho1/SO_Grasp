from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ObjectState:
    name: str
    body_id: int
    geom_id: int
    geom_type: str
    position: np.ndarray
    half_extents: np.ndarray

    @property
    def min_xy(self) -> np.ndarray:
        return self.position[:2] - self.half_extents[:2]

    @property
    def max_xy(self) -> np.ndarray:
        return self.position[:2] + self.half_extents[:2]

    @property
    def min_corner(self) -> np.ndarray:
        return self.position - self.half_extents

    @property
    def max_corner(self) -> np.ndarray:
        return self.position + self.half_extents

    @property
    def bottom_z(self) -> float:
        return float(self.position[2] - self.half_extents[2])

    @property
    def top_z(self) -> float:
        return float(self.position[2] + self.half_extents[2])

    @property
    def xy_area(self) -> float:
        size_xy = self.half_extents[:2] * 2.0
        return float(size_xy[0] * size_xy[1])

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "body_id": self.body_id,
            "geom_id": self.geom_id,
            "geom_type": self.geom_type,
            "position": self.position.round(6).tolist(),
            "half_extents": self.half_extents.round(6).tolist(),
            "bottom_z": round(self.bottom_z, 6),
            "top_z": round(self.top_z, 6),
        }


def xy_overlap_area(a: ObjectState, b: ObjectState) -> float:
    min_xy = np.maximum(a.min_xy, b.min_xy)
    max_xy = np.minimum(a.max_xy, b.max_xy)
    overlap = np.maximum(max_xy - min_xy, 0.0)
    return float(overlap[0] * overlap[1])


def xy_overlap_ratio(a: ObjectState, b: ObjectState) -> float:
    denominator = max(min(a.xy_area, b.xy_area), 1e-9)
    return xy_overlap_area(a, b) / denominator


def xy_distance(a: ObjectState, b: ObjectState) -> float:
    return float(np.linalg.norm(a.position[:2] - b.position[:2]))


def aabb_intersection_volume(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> float:
    overlap = np.maximum(np.minimum(max_a, max_b) - np.maximum(min_a, min_b), 0.0)
    return float(overlap[0] * overlap[1] * overlap[2])


def aabb_volume(min_corner: np.ndarray, max_corner: np.ndarray) -> float:
    size = np.maximum(max_corner - min_corner, 0.0)
    return float(size[0] * size[1] * size[2])


def find_object(objects: List[ObjectState], name: str) -> ObjectState:
    for obj in objects:
        if obj.name == name:
            return obj
    raise KeyError(name)
