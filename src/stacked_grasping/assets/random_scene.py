from __future__ import annotations

import random
from dataclasses import replace
from typing import Iterable, Sequence

from stacked_grasping.assets.ycb_starter import YcbObjectSpec


def sample_random_layout(
    objects: Sequence[YcbObjectSpec],
    seed: int,
    object_count: int | None = None,
    min_objects: int = 5,
    stack_probability: float = 0.55,
    workspace_x: tuple[float, float] = (-0.12, 0.12),
    workspace_y: tuple[float, float] = (-0.09, 0.09),
    support_jitter: float = 0.03,
    max_stack_depth: int = 3,
) -> tuple[YcbObjectSpec, ...]:
    """Sample a deterministic random object layout for YCB stacked scenes."""
    if not objects:
        raise ValueError("objects must not be empty")
    if max_stack_depth < 1:
        raise ValueError("max_stack_depth must be at least 1")
    rng = random.Random(seed)
    count = _resolve_count(len(objects), object_count, min_objects, rng)
    selected = rng.sample(list(objects), count)

    placed: list[YcbObjectSpec] = []
    stack_depth_by_name: dict[str, int] = {}
    layout: list[YcbObjectSpec] = []
    for spec in selected:
        support = None
        support_candidates = [
            candidate
            for candidate in placed
            if stack_depth_by_name[candidate.name] < max_stack_depth and can_stack_on(candidate, spec)
        ]
        if support_candidates and rng.random() < stack_probability:
            support_spec = rng.choice(support_candidates)
            support = support_spec.name
            x_bound, y_bound = support_offset_bounds(support_spec, spec, support_jitter=support_jitter)
            x = support_spec.pos[0] + rng.uniform(-x_bound, x_bound)
            y = support_spec.pos[1] + rng.uniform(-y_bound, y_bound)
        else:
            x = rng.uniform(*workspace_x)
            y = rng.uniform(*workspace_y)

        randomized = replace(spec, pos=(round(x, 5), round(y, 5), 0.0), support=support)
        layout.append(randomized)
        placed.append(randomized)
        stack_depth_by_name[randomized.name] = 1 if support is None else stack_depth_by_name[support] + 1

    return tuple(layout)


def scene_seeds(base_seed: int, count: int) -> Iterable[int]:
    for offset in range(count):
        yield base_seed + offset


def footprint_half_extents(spec: YcbObjectSpec) -> tuple[float, float]:
    if spec.geom_type == "cylinder":
        radius = spec.size[0]
        return radius, radius
    return spec.size[0], spec.size[1]


def can_stack_on(support: YcbObjectSpec, child: YcbObjectSpec) -> bool:
    if support.geom_type != "box":
        return False

    support_x, support_y = footprint_half_extents(support)
    child_x, child_y = footprint_half_extents(child)
    support_area = support_x * support_y
    child_area = child_x * child_y

    return bool(
        support_x >= child_x * 0.45
        and support_y >= child_y * 0.45
        and support_area >= child_area * 0.35
    )


def support_offset_bounds(
    support: YcbObjectSpec,
    child: YcbObjectSpec,
    support_jitter: float,
) -> tuple[float, float]:
    support_x, support_y = footprint_half_extents(support)
    child_x, child_y = footprint_half_extents(child)

    x_bound = min(support_jitter, support_x * 0.65, max(0.0, support_x - child_x * 0.35))
    y_bound = min(support_jitter, support_y * 0.65, max(0.0, support_y - child_y * 0.35))
    return x_bound, y_bound


def _resolve_count(total: int, object_count: int | None, min_objects: int, rng: random.Random) -> int:
    if object_count is not None:
        if object_count < 1:
            raise ValueError("object_count must be at least 1")
        if object_count > total:
            raise ValueError("object_count cannot exceed available objects")
        return object_count

    lower = max(1, min(min_objects, total))
    return rng.randint(lower, total)
