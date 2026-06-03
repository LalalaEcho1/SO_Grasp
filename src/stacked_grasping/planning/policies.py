from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List, Sequence

from stacked_grasping.gripper.feasibility import ObjectGraspFeasibility
from stacked_grasping.planning.adaptive_score import rank_objects
from stacked_grasping.planning.adaptive_score_v2 import rank_objects_v2
from stacked_grasping.relations.graph import RelationGraph


VALID_POLICIES = (
    "adaptive-score",
    "adaptive-score-v2",
    "adaptive-score-v2-gripper",
    "adaptive-score-v2-graspnet",
    "od-only",
    "highest-first",
    "lowest-blocked",
    "random",
)


@dataclass
class PolicyDecision:
    policy: str
    selected_object: str
    ranking: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "policy": self.policy,
            "selected_object": self.selected_object,
            "ranking": self.ranking,
        }


def select_object(
    policy: str,
    graph: RelationGraph,
    rng: Random,
    gripper_feasibilities: Sequence[ObjectGraspFeasibility] | None = None,
) -> PolicyDecision:
    if policy == "adaptive-score":
        ranking = [score.to_dict() for score in rank_objects(graph)]
        return _decision_from_ranking(policy, ranking)
    if policy == "adaptive-score-v2":
        ranking = [score.to_dict() for score in rank_objects_v2(graph)]
        return _decision_from_ranking(policy, ranking)
    if policy in {"adaptive-score-v2-gripper", "adaptive-score-v2-graspnet"}:
        return _rank_adaptive_score_v2_gripper(policy, graph, gripper_feasibilities)
    if policy == "od-only":
        return _rank_by_incoming_od(policy, graph)
    if policy == "highest-first":
        return _rank_by_height(policy, graph)
    if policy == "lowest-blocked":
        return _rank_by_blocked_ratio(policy, graph)
    if policy == "random":
        return _random_decision(policy, graph, rng)

    valid = ", ".join(VALID_POLICIES)
    raise ValueError(f"Unknown policy {policy!r}. Valid policies: {valid}")


def _decision_from_ranking(policy: str, ranking: List[Dict[str, object]]) -> PolicyDecision:
    if not ranking:
        raise ValueError("Policy cannot select from an empty graph.")
    return PolicyDecision(policy=policy, selected_object=str(ranking[0]["name"]), ranking=ranking)


def _rank_adaptive_score_v2_gripper(
    policy: str,
    graph: RelationGraph,
    gripper_feasibilities: Sequence[ObjectGraspFeasibility] | None,
) -> PolicyDecision:
    feasibility_by_name = {item.object_name: item for item in gripper_feasibilities or []}
    ranking = []
    for score in rank_objects_v2(graph):
        item = score.to_dict()
        feasibility = feasibility_by_name.get(score.name)
        if feasibility is None:
            item.update(
                {
                    "gripper_feasible": True,
                    "gripper_feasible_grasp_count": None,
                    "gripper_collision_risk": 0.0,
                }
            )
        else:
            item.update(
                {
                    "gripper_feasible": feasibility.feasible,
                    "gripper_feasible_grasp_count": feasibility.feasible_grasp_count,
                    "gripper_collision_risk": round(feasibility.gripper_collision_risk, 6),
                }
            )
        ranking.append(item)

    ranking.sort(
        key=lambda item: (
            not bool(item["gripper_feasible"]),
            float(item["gripper_collision_risk"]),
            bool(item["high_risk"]),
            -float(item["height_priority"]),
            float(item["grasp_risk"]),
            -float(item["adaptive_v2_score"]),
            str(item["name"]),
        )
    )
    return _decision_from_ranking(policy, ranking)


def _rank_by_incoming_od(policy: str, graph: RelationGraph) -> PolicyDecision:
    normalizer = max(len(graph.object_names) - 1, 1)
    ranking = []
    for obj in graph.objects:
        incoming_od = sum(edge.od for edge in graph.incoming(obj.name)) / normalizer
        ranking.append({"name": obj.name, "incoming_od": round(incoming_od, 6)})
    ranking.sort(key=lambda item: (float(item["incoming_od"]), str(item["name"])))
    return _decision_from_ranking(policy, ranking)


def _rank_by_height(policy: str, graph: RelationGraph) -> PolicyDecision:
    ranking = [{"name": obj.name, "height": round(float(obj.position[2]), 6)} for obj in graph.objects]
    ranking.sort(key=lambda item: (-float(item["height"]), str(item["name"])))
    return _decision_from_ranking(policy, ranking)


def _rank_by_blocked_ratio(policy: str, graph: RelationGraph) -> PolicyDecision:
    normalizer = max(len(graph.object_names) - 1, 1)
    ranking = []
    for obj in graph.objects:
        blocked = sum(edge.blocked_grasp_ratio for edge in graph.incoming(obj.name)) / normalizer
        ranking.append({"name": obj.name, "incoming_blocked_grasp_ratio": round(blocked, 6)})
    ranking.sort(key=lambda item: (float(item["incoming_blocked_grasp_ratio"]), str(item["name"])))
    return _decision_from_ranking(policy, ranking)


def _random_decision(policy: str, graph: RelationGraph, rng: Random) -> PolicyDecision:
    names = list(graph.object_names)
    if not names:
        raise ValueError("Policy cannot select from an empty graph.")
    selected = rng.choice(names)
    ranking = [{"name": selected, "random_selected": True}]
    ranking.extend({"name": name, "random_selected": False} for name in names if name != selected)
    return PolicyDecision(policy=policy, selected_object=selected, ranking=ranking)
