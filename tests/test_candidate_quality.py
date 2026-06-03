from __future__ import annotations

import unittest

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.candidate_quality import summarize_candidate_quality
from stacked_grasping.gripper.feasibility import assess_object_grasp_candidates
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate, generate_topdown_grasp_pose_candidates
from stacked_grasping.relations.geometry import ObjectState


def _obj(name: str, pos, half) -> ObjectState:
    return ObjectState(
        name=name,
        body_id=0,
        geom_id=0,
        geom_type="box",
        position=np.array(pos, dtype=float),
        half_extents=np.array(half, dtype=float),
    )


class CandidateQualityTests(unittest.TestCase):
    def test_summarizes_candidate_quality_for_one_object(self):
        target = _obj("target", (0.0, 0.0, 0.5), (0.02, 0.02, 0.04))
        feasible_pose = generate_topdown_grasp_pose_candidates(target)[0]
        graspnet_pose = GraspPoseCandidate(
            object_name="target",
            generator="graspnet",
            position=np.array([0.0, 0.0, 0.54], dtype=float),
            pregrasp_position=np.array([0.0, 0.0, 0.66], dtype=float),
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=float),
            closing_axis="6d",
            orientation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            required_opening=0.05,
            score=0.91,
        )
        feasibility = assess_object_grasp_candidates(
            [target],
            target_name="target",
            poses=[feasible_pose, graspnet_pose],
        )

        summary = summarize_candidate_quality([feasibility], candidate_source="mixed")[0]

        self.assertEqual(summary["candidate_source"], "mixed")
        self.assertEqual(summary["object_name"], "target")
        self.assertEqual(summary["num_candidates"], 2)
        self.assertEqual(summary["num_feasible"], 2)
        self.assertEqual(summary["num_infeasible"], 0)
        self.assertEqual(summary["feasible_rate"], 1.0)
        self.assertEqual(summary["mean_required_opening"], 0.047)
        self.assertEqual(summary["max_score"], 1.0)
        self.assertEqual(summary["selected_grasp_generator"], "rule-topdown")
        self.assertEqual(summary["generator_counts"], {"rule-topdown": 1, "graspnet": 1})
        self.assertEqual(summary["failure_reasons"], {})


if __name__ == "__main__":
    unittest.main()
