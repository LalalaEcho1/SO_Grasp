from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List, Sequence

from stacked_grasping.gripper.feasibility import GraspCandidate, ObjectGraspFeasibility
from stacked_grasping.planning.adaptive_score import rank_objects
from stacked_grasping.planning.adaptive_score_v2 import rank_objects_v2
from stacked_grasping.planning.grasp_risk import GraspRiskConfig, assess_grasp_risk
from stacked_grasping.relations.graph import RelationGraph


VALID_POLICIES = (
    "adaptive-score",
    "adaptive-score-v2",
    "adaptive-score-v2-gripper",
    "adaptive-score-v2-graspnet",
    "adaptive-score-v2-graspnet-prior",
    "adaptive-score-v2-riskaware",
    "adaptive-score-v3-candidate",
    "adaptive-score-v3-candidate-reserve",
    "graspnet-quality-first",
    "graspnet-score-first",
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
    selected_candidate_index: int | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "policy": self.policy,
            "selected_object": self.selected_object,
            "selected_candidate_index": self.selected_candidate_index,
            "ranking": self.ranking,
        }


def select_object(
    policy: str,
    graph: RelationGraph,
    rng: Random,
    gripper_feasibilities: Sequence[ObjectGraspFeasibility] | None = None,
    risk_config: GraspRiskConfig | None = None,
) -> PolicyDecision:
    if policy == "adaptive-score":
        ranking = [score.to_dict() for score in rank_objects(graph)]
        return _decision_from_ranking(policy, ranking)
    if policy == "adaptive-score-v2":
        ranking = [score.to_dict() for score in rank_objects_v2(graph)]
        return _decision_from_ranking(policy, ranking)
    if policy in {
        "adaptive-score-v2-gripper",
        "adaptive-score-v2-graspnet",
        "adaptive-score-v2-graspnet-prior",
        "adaptive-score-v3-candidate",
        "adaptive-score-v3-candidate-reserve",
        "graspnet-quality-first",
        "graspnet-score-first",
    }:
        return _rank_adaptive_score_v2_gripper(policy, graph, gripper_feasibilities)
    if policy == "adaptive-score-v2-riskaware":
        return _rank_adaptive_score_v2_riskaware(policy, graph, gripper_feasibilities, risk_config)
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
    return PolicyDecision(
        policy=policy,
        selected_object=str(ranking[0]["name"]),
        ranking=ranking,
        selected_candidate_index=_optional_int(ranking[0].get("selected_candidate_index")),
    )


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
                    "graspnet_best_score": None,
                    "graspnet_best_tolerance": None,
                    "graspnet_quality_score": None,
                    "selected_candidate_index": None,
                    "selected_candidate_score": None,
                    "selected_candidate_tolerance": None,
                    "candidate_priority_score": None,
                    "candidate_reserve_bonus": None,
                    "candidate_reserve_priority_score": None,
                    "candidate_indices": [],
                }
            )
        else:
            candidate_score_field = (
                "score"
                if policy in {
                    "adaptive-score-v3-candidate",
                    "adaptive-score-v3-candidate-reserve",
                    "graspnet-score-first",
                }
                else "quality"
            )
            selected_candidate = _best_feasible_candidate(feasibility, score_field=candidate_score_field)
            selected_score = _candidate_pose_value(selected_candidate, "score")
            selected_tolerance = _candidate_pose_value(selected_candidate, "grasp_tolerance")
            candidate_priority = _candidate_priority(
                selected_score,
                gripper_collision_risk=feasibility.gripper_collision_risk,
                grasp_risk=float(item["grasp_risk"]),
                high_risk=bool(item["high_risk"]),
            )
            reserve_bonus = _candidate_reserve_bonus(feasibility.feasible_grasp_count)
            reserve_priority = candidate_priority + reserve_bonus if candidate_priority is not None else None
            item.update(
                {
                    "gripper_feasible": feasibility.feasible,
                    "gripper_feasible_grasp_count": feasibility.feasible_grasp_count,
                    "gripper_collision_risk": round(feasibility.gripper_collision_risk, 6),
                    "graspnet_best_score": _round_optional(feasibility.best_grasp_score),
                    "graspnet_best_tolerance": _round_optional(feasibility.best_grasp_tolerance),
                    "graspnet_quality_score": _round_optional(feasibility.graspnet_quality_score),
                    "selected_candidate_index": _candidate_index(selected_candidate),
                    "selected_candidate_score": _round_optional(selected_score),
                    "selected_candidate_tolerance": _round_optional(selected_tolerance),
                    "candidate_priority_score": _round_optional(candidate_priority),
                    "candidate_reserve_bonus": _round_optional(reserve_bonus),
                    "candidate_reserve_priority_score": _round_optional(reserve_priority),
                    "candidate_indices": _ranked_feasible_candidate_indices(
                        feasibility,
                        score_field=candidate_score_field,
                    ),
                }
            )
        ranking.append(item)

    if policy == "adaptive-score-v3-candidate-reserve":
        ranking.sort(
            key=lambda item: (
                not bool(item["gripper_feasible"]),
                -_optional_score(item.get("candidate_reserve_priority_score")),
                -_optional_score(item.get("candidate_priority_score")),
                -_optional_score(item.get("selected_candidate_score")),
                -int(item["gripper_feasible_grasp_count"] or 0),
                float(item["gripper_collision_risk"]),
                bool(item["high_risk"]),
                float(item["grasp_risk"]),
                -float(item["height_priority"]),
                -float(item["adaptive_v2_score"]),
                str(item["name"]),
            )
        )
    elif policy == "adaptive-score-v3-candidate":
        ranking.sort(
            key=lambda item: (
                not bool(item["gripper_feasible"]),
                -_optional_score(item.get("candidate_priority_score")),
                -_optional_score(item.get("selected_candidate_score")),
                float(item["gripper_collision_risk"]),
                bool(item["high_risk"]),
                float(item["grasp_risk"]),
                -float(item["height_priority"]),
                -float(item["adaptive_v2_score"]),
                str(item["name"]),
            )
        )
    elif policy == "graspnet-score-first":
        ranking.sort(
            key=lambda item: (
                not bool(item["gripper_feasible"]),
                -_optional_score(item.get("graspnet_best_score")),
                float(item["gripper_collision_risk"]),
                bool(item["high_risk"]),
                float(item["grasp_risk"]),
                -float(item["height_priority"]),
                -float(item["adaptive_v2_score"]),
                str(item["name"]),
            )
        )
    elif policy == "graspnet-quality-first":
        ranking.sort(
            key=lambda item: (
                not bool(item["gripper_feasible"]),
                -_optional_score(item.get("graspnet_quality_score")),
                float(item["gripper_collision_risk"]),
                bool(item["high_risk"]),
                float(item["grasp_risk"]),
                -float(item["height_priority"]),
                -float(item["adaptive_v2_score"]),
                str(item["name"]),
            )
        )
    elif policy == "adaptive-score-v2-graspnet-prior":
        ranking.sort(
            key=lambda item: (
                not bool(item["gripper_feasible"]),
                bool(item["high_risk"]),
                -_optional_score(item.get("graspnet_quality_score")),
                float(item["gripper_collision_risk"]),
                float(item["grasp_risk"]),
                -float(item["height_priority"]),
                -float(item["adaptive_v2_score"]),
                str(item["name"]),
            )
        )
    elif policy == "adaptive-score-v2-graspnet":
        ranking.sort(
            key=lambda item: (
                not bool(item["gripper_feasible"]),
                float(item["gripper_collision_risk"]),
                bool(item["high_risk"]),
                -_optional_score(item.get("graspnet_quality_score")),
                -float(item["height_priority"]),
                float(item["grasp_risk"]),
                -float(item["adaptive_v2_score"]),
                str(item["name"]),
            )
        )
    else:
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


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _optional_score(value: object) -> float:
    return float(value) if value is not None else float("-inf")


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _best_feasible_candidate(feasibility: ObjectGraspFeasibility, *, score_field: str) -> GraspCandidate | None:
    candidates = _ranked_feasible_candidates(feasibility, score_field=score_field)
    return candidates[0] if candidates else None


def _ranked_feasible_candidate_indices(feasibility: ObjectGraspFeasibility, *, score_field: str) -> List[int]:
    indices = []
    for candidate in _ranked_feasible_candidates(feasibility, score_field=score_field):
        candidate_index = _candidate_index(candidate)
        if candidate_index is not None:
            indices.append(candidate_index)
    return indices


def _ranked_feasible_candidates(feasibility: ObjectGraspFeasibility, *, score_field: str) -> List[GraspCandidate]:
    candidates = [
        candidate
        for candidate in feasibility.candidates
        if candidate.feasible and candidate.pose is not None
    ]
    return sorted(
        candidates,
        key=lambda candidate: (
            -_pose_quality(candidate.pose, score_field=score_field),
            float(candidate.required_opening),
            candidate.closing_axis,
            _candidate_index(candidate) if _candidate_index(candidate) is not None else 10**9,
        ),
    )


def _pose_quality(pose: object, *, score_field: str) -> float:
    if score_field == "score":
        return float(getattr(pose, "score", 0.0))
    tolerance = getattr(pose, "grasp_tolerance", None)
    if tolerance is not None:
        return float(tolerance)
    return float(getattr(pose, "score", 0.0))


def _candidate_pose_value(candidate: GraspCandidate | None, attr: str) -> float | None:
    if candidate is None or candidate.pose is None:
        return None
    value = getattr(candidate.pose, attr)
    return float(value) if value is not None else None


def _candidate_index(candidate: GraspCandidate | None) -> int | None:
    if candidate is None or candidate.pose is None or candidate.pose.candidate_index is None:
        return None
    return int(candidate.pose.candidate_index)


def _candidate_priority(
    score: float | None,
    *,
    gripper_collision_risk: float,
    grasp_risk: float,
    high_risk: bool,
) -> float | None:
    if score is None:
        return None
    return (
        float(score)
        - 0.15 * float(gripper_collision_risk)
        - 0.10 * float(grasp_risk)
        - (0.05 if high_risk else 0.0)
    )


def _candidate_reserve_bonus(feasible_grasp_count: int | None) -> float:
    if feasible_grasp_count is None:
        return 0.0
    capped_count = max(min(int(feasible_grasp_count), 5), 0)
    return 0.015 * float(capped_count)


def _rank_adaptive_score_v2_riskaware(
    policy: str,
    graph: RelationGraph,
    gripper_feasibilities: Sequence[ObjectGraspFeasibility] | None,
    risk_config: GraspRiskConfig | None = None,
) -> PolicyDecision:
    """Gripper-aware v2 ranking gated by the same full risk the episode judge uses.

    adaptive-score-v2-graspnet sorts lexicographically by gripper collision risk
    before any aggregate risk, while episode success is judged on the FULL risk
    (base + gripper term) against the threshold. This policy closes that gap:
    objects predicted to succeed under the judge's own metric come first (highest
    first, keeping the v2 safety/height spirit); if no object is predicted to
    succeed, it falls back to the lowest full risk.
    """
    cfg = risk_config or GraspRiskConfig()
    feasibility_by_name = {item.object_name: item for item in gripper_feasibilities or []}
    ranking = []
    for score in rank_objects_v2(graph):
        item = score.to_dict()
        feasibility = feasibility_by_name.get(score.name)
        assessment = assess_grasp_risk(score.base_score, cfg, gripper_feasibility=feasibility)
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
        item.update(
            {
                "full_grasp_risk": round(assessment.risk, 6),
                "predicted_success": bool(assessment.success),
            }
        )
        ranking.append(item)

    ranking.sort(
        key=lambda item: (
            not bool(item["gripper_feasible"]),
            not bool(item["predicted_success"]),
            -float(item["height_priority"]) if item["predicted_success"] else 0.0,
            float(item["full_grasp_risk"]),
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
