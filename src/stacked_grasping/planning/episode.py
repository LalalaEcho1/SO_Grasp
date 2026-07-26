from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

from stacked_grasping.env.mujoco_scene import MujocoStackedScene
from stacked_grasping.gripper.feasibility import (
    ObjectGraspFeasibility,
    TopDownGraspConfig,
    assess_scene_grasp_candidates,
    assess_scene_topdown_grasps,
)
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.planning.adaptive_score import ObjectScore, rank_objects
from stacked_grasping.planning.grasp_risk import GraspRiskConfig, assess_grasp_risk
from stacked_grasping.planning.policies import PolicyDecision, select_object
from stacked_grasping.relations.graph import EdgeFeatures, build_relation_graph
from stacked_grasping.relations.geometry import ObjectState


@dataclass
class EpisodeStep:
    step_index: int
    selected_object: str
    remaining_objects_before: List[str]
    contact_pairs_before: List[Tuple[str, str]]
    ranking_before: List[ObjectScore]
    edges_before: List[EdgeFeatures]
    policy_decision: PolicyDecision | None = None
    grasp_success: bool = True
    grasp_risk: float = 0.0
    failure_reason: str | None = None
    gripper_feasibility: ObjectGraspFeasibility | None = None

    @property
    def selected_score(self) -> ObjectScore:
        for score in self.ranking_before:
            if score.name == self.selected_object:
                return score
        raise KeyError(self.selected_object)

    @property
    def gripper_feasible(self) -> bool | None:
        if self.gripper_feasibility is None:
            return None
        return self.gripper_feasibility.feasible

    @property
    def gripper_feasible_grasp_count(self) -> int | None:
        if self.gripper_feasibility is None:
            return None
        return self.gripper_feasibility.feasible_grasp_count

    @property
    def gripper_collision_risk(self) -> float:
        if self.gripper_feasibility is None:
            return 0.0
        return self.gripper_feasibility.gripper_collision_risk

    @property
    def selected_grasp_pose(self) -> GraspPoseCandidate | None:
        if self.gripper_feasibility is None:
            return None
        return self.gripper_feasibility.selected_grasp_pose

    def to_dict(self) -> Dict[str, object]:
        return {
            "step_index": self.step_index,
            "selected_object": self.selected_object,
            "selected_score": self.selected_score.to_dict(),
            "remaining_objects_before": self.remaining_objects_before,
            "contact_pairs_before": [list(pair) for pair in self.contact_pairs_before],
            "ranking_before": [score.to_dict() for score in self.ranking_before],
            "edges_before": [edge.to_dict() for edge in self.edges_before],
            "policy_decision": self.policy_decision.to_dict() if self.policy_decision else None,
            "grasp_success": self.grasp_success,
            "grasp_risk": round(self.grasp_risk, 6),
            "failure_reason": self.failure_reason,
            "gripper_feasible": self.gripper_feasible,
            "gripper_feasible_grasp_count": self.gripper_feasible_grasp_count,
            "gripper_collision_risk": round(self.gripper_collision_risk, 6),
            "selected_grasp_pose": self.selected_grasp_pose.to_dict() if self.selected_grasp_pose else None,
            "gripper_feasibility": self.gripper_feasibility.to_dict() if self.gripper_feasibility else None,
        }


@dataclass
class EpisodeResult:
    policy: str
    steps: List[EpisodeStep]
    final_objects: List[ObjectState]
    failure_mode: str = "none"
    failure_reason: str | None = None

    @property
    def grasp_sequence(self) -> List[str]:
        return [step.selected_object for step in self.steps]

    def to_dict(self) -> Dict[str, object]:
        return {
            "policy": self.policy,
            "num_steps": len(self.steps),
            "grasp_sequence": self.grasp_sequence,
            "failure_mode": self.failure_mode,
            "failure_reason": self.failure_reason,
            "steps": [step.to_dict() for step in self.steps],
            "final_objects": [obj.to_dict() for obj in self.final_objects],
        }


def run_adaptive_episode(
    scene: MujocoStackedScene,
    max_steps: int | None = None,
    post_grasp_settle_steps: int = 500,
    failure_mode: str = "none",
    risk_threshold: float = 0.45,
    gripper_config: TopDownGraspConfig | None = None,
    grasp_poses_by_object: Mapping[str, Sequence[GraspPoseCandidate]] | None = None,
    gripper_feasibility_provider: Callable[[Sequence[ObjectState]], Sequence[ObjectGraspFeasibility]] | None = None,
) -> EpisodeResult:
    """Run an abstract sequential-grasp episode with the adaptive baseline policy."""
    return run_policy_episode(
        scene,
        policy="adaptive-score",
        max_steps=max_steps,
        post_grasp_settle_steps=post_grasp_settle_steps,
        seed=0,
        failure_mode=failure_mode,
        risk_threshold=risk_threshold,
        gripper_config=gripper_config,
        grasp_poses_by_object=grasp_poses_by_object,
        gripper_feasibility_provider=gripper_feasibility_provider,
    )


def run_policy_episode(
    scene: MujocoStackedScene,
    policy: str,
    max_steps: int | None = None,
    post_grasp_settle_steps: int = 500,
    seed: int = 0,
    failure_mode: str = "none",
    risk_threshold: float = 0.45,
    gripper_config: TopDownGraspConfig | None = None,
    grasp_poses_by_object: Mapping[str, Sequence[GraspPoseCandidate]] | None = None,
    gripper_feasibility_provider: Callable[[Sequence[ObjectState]], Sequence[ObjectGraspFeasibility]] | None = None,
) -> EpisodeResult:
    """Run an abstract sequential-grasp episode with a selectable high-level policy."""
    _validate_failure_mode(failure_mode)
    steps: List[EpisodeStep] = []
    rng = random.Random(seed)
    episode_failure_reason: str | None = None
    risk_config = GraspRiskConfig(threshold=risk_threshold)

    while True:
        objects = scene.read_objects()
        if not objects:
            break
        if max_steps is not None and len(steps) >= max_steps:
            break

        contact_pairs = sorted(scene.read_object_contact_pairs())
        graph = build_relation_graph(objects, set(contact_pairs))
        ranking = rank_objects(graph)
        if not ranking:
            break

        if gripper_feasibility_provider is not None:
            gripper_feasibilities = list(gripper_feasibility_provider(objects))
        elif grasp_poses_by_object is None:
            gripper_feasibilities = assess_scene_topdown_grasps(objects, config=gripper_config)
        else:
            gripper_feasibilities = assess_scene_grasp_candidates(
                objects,
                grasp_poses_by_object,
                config=gripper_config,
            )
        gripper_feasibility_by_name = {item.object_name: item for item in gripper_feasibilities}
        decision = select_object(
            policy,
            graph,
            rng,
            gripper_feasibilities=gripper_feasibilities,
            risk_config=risk_config,
        )
        selected_object = decision.selected_object
        selected_score = _selected_score(ranking, selected_object)
        gripper_feasibility = gripper_feasibility_by_name[selected_object]
        assessment = assess_grasp_risk(selected_score, risk_config, gripper_feasibility=gripper_feasibility)
        failed = failure_mode == "risk-threshold" and not assessment.success
        steps.append(
            EpisodeStep(
                step_index=len(steps) + 1,
                selected_object=selected_object,
                remaining_objects_before=[obj.name for obj in objects],
                contact_pairs_before=contact_pairs,
                ranking_before=ranking,
                edges_before=graph.edges,
                policy_decision=decision,
                grasp_success=not failed,
                grasp_risk=assessment.risk,
                failure_reason=assessment.reason if failed else None,
                gripper_feasibility=gripper_feasibility,
            )
        )

        if failed:
            episode_failure_reason = assessment.reason
            break

        scene.remove_object(selected_object)
        scene.settle(post_grasp_settle_steps)

    return EpisodeResult(
        policy=policy,
        steps=steps,
        final_objects=scene.read_objects(),
        failure_mode=failure_mode,
        failure_reason=episode_failure_reason,
    )


def format_episode_summary(result: EpisodeResult) -> str:
    lines = [
        "Sequential grasp episode",
        f"  policy: {result.policy}",
        f"  steps: {len(result.steps)}",
        "",
        "Grasp sequence",
    ]
    for step in result.steps:
        score = step.selected_score
        lines.append(
            "  "
            f"{step.step_index}. {step.selected_object}: "
            f"score={score.score:.3f}, blocked={score.blocked_by_od:.3f}, "
            f"support_risk={score.support_risk:.3f}, clearance_gain={score.clearance_gain:.3f}, "
            f"gripper_feasible={step.gripper_feasible}, gripper_risk={step.gripper_collision_risk:.3f}, "
            f"grasp_risk={step.grasp_risk:.3f}, success={step.grasp_success}"
        )
        if step.failure_reason:
            lines.append(f"     failure_reason={step.failure_reason}")

    if result.final_objects:
        lines.append("")
        lines.append("Final objects still active")
        for obj in result.final_objects:
            lines.append(f"  - {obj.name}")

    return "\n".join(lines)


def compact_step_records(result: EpisodeResult) -> Sequence[Dict[str, object]]:
    return [
        {
            "step": step.step_index,
            "selected": step.selected_object,
            "score": round(step.selected_score.score, 6),
            "grasp_success": step.grasp_success,
            "grasp_risk": round(step.grasp_risk, 6),
            "gripper_feasible": step.gripper_feasible,
            "gripper_collision_risk": round(step.gripper_collision_risk, 6),
            "selected_grasp_pose": step.selected_grasp_pose.to_dict() if step.selected_grasp_pose else None,
            "remaining_before": step.remaining_objects_before,
        }
        for step in result.steps
    ]


def _selected_score(ranking: Sequence[ObjectScore], selected_object: str) -> ObjectScore:
    for score in ranking:
        if score.name == selected_object:
            return score
    raise KeyError(selected_object)


def _validate_failure_mode(failure_mode: str) -> None:
    if failure_mode not in {"none", "risk-threshold"}:
        raise ValueError(f"Unsupported failure mode: {failure_mode}")
