from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class YcbObjectSpec:
    name: str
    geom_type: str
    size: tuple[float, float, float]
    mass: float
    pos: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    support: str | None = None


STARTER_OBJECTS: tuple[YcbObjectSpec, ...] = (
    YcbObjectSpec(
        name="003_cracker_box",
        geom_type="box",
        size=(0.030, 0.079, 0.105),
        mass=0.411,
        pos=(-0.080, 0.000, 0.490),
        rgba=(0.86, 0.44, 0.18, 1.0),
    ),
    YcbObjectSpec(
        name="004_sugar_box",
        geom_type="box",
        size=(0.032, 0.045, 0.088),
        mass=0.514,
        pos=(0.015, 0.010, 0.475),
        rgba=(0.92, 0.86, 0.62, 1.0),
    ),
    YcbObjectSpec(
        name="008_pudding_box",
        geom_type="box",
        size=(0.018, 0.055, 0.045),
        mass=0.187,
        pos=(-0.040, 0.005, 0.615),
        rgba=(0.95, 0.76, 0.22, 1.0),
        support="003_cracker_box",
    ),
    YcbObjectSpec(
        name="009_gelatin_box",
        geom_type="box",
        size=(0.014, 0.043, 0.037),
        mass=0.097,
        pos=(0.040, 0.015, 0.585),
        rgba=(0.78, 0.16, 0.22, 1.0),
        support="004_sugar_box",
    ),
    YcbObjectSpec(
        name="005_tomato_soup_can",
        geom_type="cylinder",
        size=(0.033, 0.051, 0.0),
        mass=0.349,
        pos=(0.130, -0.075, 0.435),
        rgba=(0.82, 0.14, 0.10, 1.0),
    ),
    YcbObjectSpec(
        name="007_tuna_fish_can",
        geom_type="cylinder",
        size=(0.043, 0.017, 0.0),
        mass=0.171,
        pos=(0.178, -0.020, 0.405),
        rgba=(0.20, 0.42, 0.72, 1.0),
    ),
    YcbObjectSpec(
        name="061_foam_brick",
        geom_type="box",
        size=(0.038, 0.025, 0.025),
        mass=0.028,
        pos=(-0.150, -0.090, 0.410),
        rgba=(0.35, 0.72, 0.45, 1.0),
    ),
    YcbObjectSpec(
        name="010_potted_meat_can",
        geom_type="box",
        size=(0.049, 0.041, 0.025),
        mass=0.370,
        pos=(0.090, 0.085, 0.420),
        rgba=(0.63, 0.38, 0.25, 1.0),
    ),
)


STARTER_OBJECT_NAMES = tuple(spec.name for spec in STARTER_OBJECTS)


def resolve_object_names(values: Iterable[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        if value == "starter":
            names.extend(STARTER_OBJECT_NAMES)
        else:
            names.append(value)

    seen: set[str] = set()
    unique_names: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique_names.append(name)
    return unique_names


def starter_specs_by_name() -> dict[str, YcbObjectSpec]:
    return {spec.name: spec for spec in STARTER_OBJECTS}
