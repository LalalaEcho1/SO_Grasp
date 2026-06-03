from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from stacked_grasping.gripper.feasibility import ObjectGraspFeasibility
from stacked_grasping.planning.adaptive_score import ObjectScore


@dataclass(frozen=True)
class GraspRiskConfig:
    threshold: float = 0.45
    blocked_weight: float = 1.0
    support_weight: float = 0.6
    contact_weight: float = 0.25
    gripper_weight: float = 0.35
    gripper_blocking_risk: float = 1.0


@dataclass(frozen=True)
class GraspRiskAssessment:
    risk: float
    success: bool
    reason: str | None

    def to_dict(self) -> Dict[str, object]:
        return {
            "risk": round(self.risk, 6),
            "success": self.success,
            "reason": self.reason,
        }


def assess_grasp_risk(
    score: ObjectScore,
    config: GraspRiskConfig | None = None,
    gripper_feasibility: ObjectGraspFeasibility | None = None,
) -> GraspRiskAssessment:
    cfg = config or GraspRiskConfig()
    risk = (
        cfg.blocked_weight * score.blocked_by_od
        + cfg.support_weight * score.support_risk
        + cfg.contact_weight * score.contact_risk
    )
    risk = max(0.0, float(risk))
    if gripper_feasibility is not None:
        if not gripper_feasibility.feasible:
            risk = max(risk, cfg.gripper_blocking_risk)
            return GraspRiskAssessment(
                risk=risk,
                success=False,
                reason="gripper-infeasible",
            )
        risk += cfg.gripper_weight * gripper_feasibility.gripper_collision_risk

    success = risk < cfg.threshold
    return GraspRiskAssessment(
        risk=risk,
        success=success,
        reason=None if success else "risk-threshold",
    )
