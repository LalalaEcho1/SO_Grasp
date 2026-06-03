from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from stacked_grasping.gripper.feasibility import ObjectGraspFeasibility


def summarize_candidate_quality(
    feasibilities: Sequence[ObjectGraspFeasibility],
    candidate_source: str = "rule",
) -> List[Dict[str, object]]:
    return [_summarize_object(item, candidate_source=candidate_source) for item in feasibilities]


def _summarize_object(item: ObjectGraspFeasibility, candidate_source: str) -> Dict[str, object]:
    candidates = item.candidates
    num_candidates = len(candidates)
    feasible_candidates = [candidate for candidate in candidates if candidate.feasible]
    openings = [candidate.required_opening for candidate in candidates]
    scores = [candidate.pose.score for candidate in candidates if candidate.pose is not None]
    generators = Counter(candidate.pose.generator for candidate in candidates if candidate.pose is not None)
    failure_reasons = Counter(candidate.reason for candidate in candidates if candidate.reason)
    selected_pose = item.selected_grasp_pose

    return {
        "candidate_source": candidate_source,
        "object_name": item.object_name,
        "num_candidates": num_candidates,
        "num_feasible": len(feasible_candidates),
        "num_infeasible": num_candidates - len(feasible_candidates),
        "feasible_rate": round(len(feasible_candidates) / max(num_candidates, 1), 6),
        "mean_required_opening": round(sum(openings) / max(len(openings), 1), 6),
        "max_score": round(max(scores) if scores else 0.0, 6),
        "selected_grasp_generator": selected_pose.generator if selected_pose else None,
        "selected_grasp_score": round(selected_pose.score, 6) if selected_pose else None,
        "generator_counts": dict(sorted(generators.items())),
        "failure_reasons": dict(sorted(failure_reasons.items())),
    }
