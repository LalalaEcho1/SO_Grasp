from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from stacked_grasping.planning.adaptive_score import ObjectScore, rank_objects
from stacked_grasping.planning.grasp_risk import GraspRiskConfig, assess_grasp_risk
from stacked_grasping.relations.graph import RelationGraph


@dataclass(frozen=True)
class AdaptiveScoreV2Config:
    risk_threshold: float = 0.35
    base_score_weight: float = 0.25
    graspability_weight: float = 0.30
    height_weight: float = 1.10
    clearance_gain_weight: float = 0.55
    risk_weight: float = 2.20
    support_weight: float = 0.45
    contact_weight: float = 0.15


@dataclass
class AdaptiveScoreV2:
    name: str
    adaptive_v2_score: float
    grasp_risk: float
    high_risk: bool
    height_priority: float
    base_score: ObjectScore

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "adaptive_v2_score": round(self.adaptive_v2_score, 6),
            "grasp_risk": round(self.grasp_risk, 6),
            "high_risk": self.high_risk,
            "height_priority": round(self.height_priority, 6),
            "base_score": round(self.base_score.score, 6),
            "graspability_prior": round(self.base_score.graspability_prior, 6),
            "blocked_by_od": round(self.base_score.blocked_by_od, 6),
            "support_risk": round(self.base_score.support_risk, 6),
            "contact_risk": round(self.base_score.contact_risk, 6),
            "clearance_gain": round(self.base_score.clearance_gain, 6),
        }


def rank_objects_v2(
    graph: RelationGraph,
    config: AdaptiveScoreV2Config | None = None,
) -> List[AdaptiveScoreV2]:
    cfg = config or AdaptiveScoreV2Config()
    risk_cfg = GraspRiskConfig(threshold=cfg.risk_threshold)
    height_priorities = _height_priorities(graph)
    scores = []
    for base_score in rank_objects(graph):
        assessment = assess_grasp_risk(base_score, risk_cfg)
        height_priority = height_priorities[base_score.name]
        score = _score_v2(base_score, assessment.risk, height_priority, cfg)
        scores.append(
            AdaptiveScoreV2(
                name=base_score.name,
                adaptive_v2_score=score,
                grasp_risk=assessment.risk,
                high_risk=not assessment.success,
                height_priority=height_priority,
                base_score=base_score,
            )
        )

    return sorted(
        scores,
        key=lambda item: (
            item.high_risk,
            -item.height_priority,
            item.grasp_risk,
            -item.adaptive_v2_score,
            item.name,
        ),
    )


def _score_v2(
    base_score: ObjectScore,
    grasp_risk: float,
    height_priority: float,
    cfg: AdaptiveScoreV2Config,
) -> float:
    return (
        cfg.base_score_weight * base_score.score
        + cfg.graspability_weight * base_score.graspability_prior
        + cfg.height_weight * height_priority
        + cfg.clearance_gain_weight * base_score.clearance_gain
        - cfg.risk_weight * grasp_risk
        - cfg.support_weight * base_score.support_risk
        - cfg.contact_weight * base_score.contact_risk
    )


def _height_priorities(graph: RelationGraph) -> Dict[str, float]:
    heights = {obj.name: float(obj.position[2]) for obj in graph.objects}
    if not heights:
        return {}
    min_height = min(heights.values())
    max_height = max(heights.values())
    span = max(max_height - min_height, 1e-9)
    return {name: (height - min_height) / span for name, height in heights.items()}
