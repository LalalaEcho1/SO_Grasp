from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from stacked_grasping.relations.graph import RelationGraph


@dataclass
class ObjectScore:
    name: str
    score: float
    graspability_prior: float
    blocked_by_od: float
    support_risk: float
    contact_risk: float
    clearance_gain: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "score": round(self.score, 6),
            "graspability_prior": round(self.graspability_prior, 6),
            "blocked_by_od": round(self.blocked_by_od, 6),
            "support_risk": round(self.support_risk, 6),
            "contact_risk": round(self.contact_risk, 6),
            "clearance_gain": round(self.clearance_gain, 6),
        }


def rank_objects(graph: RelationGraph) -> List[ObjectScore]:
    scores = [_score_object(graph, name) for name in graph.object_names]
    return sorted(scores, key=lambda item: item.score, reverse=True)


def _score_object(graph: RelationGraph, name: str) -> ObjectScore:
    incoming = graph.incoming(name)
    outgoing = graph.outgoing(name)
    normalizer = max(len(graph.object_names) - 1, 1)

    blocked_by_od = sum(edge.od for edge in incoming) / normalizer
    graspability_prior = max(0.0, 1.0 - blocked_by_od)
    support_risk = sum(1.0 for edge in outgoing if edge.support_source_to_target) / normalizer
    contact_risk = sum(1.0 for edge in outgoing if edge.contact) / normalizer
    clearance_gain = sum(edge.od for edge in outgoing) / normalizer

    score = (
        1.00 * graspability_prior
        - 0.85 * blocked_by_od
        - 0.75 * support_risk
        - 0.20 * contact_risk
        + 0.65 * clearance_gain
    )

    return ObjectScore(
        name=name,
        score=score,
        graspability_prior=graspability_prior,
        blocked_by_od=blocked_by_od,
        support_risk=support_risk,
        contact_risk=contact_risk,
        clearance_gain=clearance_gain,
    )
