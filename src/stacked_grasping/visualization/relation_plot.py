from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from stacked_grasping.planning.adaptive_score import ObjectScore
from stacked_grasping.relations.graph import EdgeFeatures, RelationGraph


def save_relation_graph_png(
    graph: RelationGraph,
    ranking: Sequence[ObjectScore],
    output_path: str | Path,
    min_od_to_draw: float = 0.18,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    positions = {obj.name: obj.position[:2] for obj in graph.objects}
    rank_by_name = {score.name: idx + 1 for idx, score in enumerate(ranking)}
    score_by_name = {score.name: score.score for score in ranking}

    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=160)
    ax.set_title("Directed Object Relation Graph", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e5e5", linewidth=0.8)

    for edge in graph.edges:
        if _should_draw(edge, min_od_to_draw):
            _draw_edge(ax, positions[edge.source], positions[edge.target], edge)

    for obj in graph.objects:
        x, y = positions[obj.name]
        score = score_by_name.get(obj.name, 0.0)
        color = _score_color(score)
        ax.scatter([x], [y], s=900, c=[color], edgecolors="#222222", linewidths=1.5, zorder=4)
        ax.text(
            x,
            y,
            f"#{rank_by_name.get(obj.name, '?')}\n{_short_name(obj.name)}",
            ha="center",
            va="center",
            fontsize=8,
            color="#111111",
            zorder=5,
        )

    _fit_axes(ax, positions.values())
    _add_legend(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _should_draw(edge: EdgeFeatures, min_od_to_draw: float) -> bool:
    return bool(
        edge.contact
        or edge.support_source_to_target
        or edge.support_target_to_source
        or edge.od >= min_od_to_draw
        or edge.xy_overlap_ratio >= 0.08
    )


def _draw_edge(ax: plt.Axes, source_xy: np.ndarray, target_xy: np.ndarray, edge: EdgeFeatures) -> None:
    start = np.array(source_xy, dtype=float)
    end = np.array(target_xy, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return

    unit = delta / length
    offset = np.array([-unit[1], unit[0]]) * 0.012
    start = start + unit * 0.018 + offset
    end = end - unit * 0.018 + offset

    color = "#2f7d32" if edge.support_source_to_target else "#c45100"
    alpha = 0.85 if edge.contact or edge.support_source_to_target else 0.55
    width = 1.0 + 3.0 * max(edge.od, 0.05)
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=width,
        color=color,
        alpha=alpha,
        shrinkA=12,
        shrinkB=12,
        zorder=2,
        connectionstyle="arc3,rad=0.12",
    )
    ax.add_patch(arrow)

    midpoint = (start + end) / 2.0
    ax.text(
        midpoint[0],
        midpoint[1],
        f"{edge.od:.2f}",
        fontsize=7,
        color=color,
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
        zorder=3,
    )


def _score_color(score: float) -> str:
    if score >= 0.75:
        return "#8fd175"
    if score >= 0.45:
        return "#ffd166"
    return "#f28b82"


def _short_name(name: str) -> str:
    return name.replace("obj_", "").replace("_box", "").replace("_cylinder", "")


def _fit_axes(ax: plt.Axes, points: Iterable[np.ndarray]) -> None:
    arr = np.array(list(points), dtype=float)
    min_xy = arr.min(axis=0)
    max_xy = arr.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    span = max(float(np.max(max_xy - min_xy)), 0.22)
    margin = span * 0.75
    ax.set_xlim(center[0] - margin, center[0] + margin)
    ax.set_ylim(center[1] - margin, center[1] + margin)


def _add_legend(ax: plt.Axes) -> None:
    ax.text(
        0.02,
        0.02,
        "green: support edge\norange: obstruction/contact edge\nedge label: OD prior",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#dddddd", "alpha": 0.9},
    )
