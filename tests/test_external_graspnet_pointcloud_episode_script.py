from __future__ import annotations

import unittest

from tests import conftest  # noqa: F401
from scripts.run_external_graspnet_pointcloud_episode import aggregate_frame_summaries


class ExternalGraspNetPointCloudEpisodeScriptTests(unittest.TestCase):
    def test_aggregate_frame_summaries_reports_binding_feasibility_and_policy_outcomes(self):
        aggregate = aggregate_frame_summaries(
            [
                {
                    "frame": "0001",
                    "total_candidates": 100,
                    "bound_count": 40,
                    "pointcloud_feasible_candidate_count": 3,
                    "pointcloud_feasible_object_count": 2,
                    "selected_object": "a.ply",
                    "grasp_success": True,
                    "failure_reason": None,
                },
                {
                    "frame": "0002",
                    "total_candidates": 100,
                    "bound_count": 20,
                    "pointcloud_feasible_candidate_count": 1,
                    "pointcloud_feasible_object_count": 1,
                    "selected_object": "b.ply",
                    "grasp_success": False,
                    "failure_reason": "risk-threshold",
                },
            ]
        )

        self.assertEqual(aggregate["frame_count"], 2)
        self.assertEqual(aggregate["total_candidates"], 200)
        self.assertEqual(aggregate["bound_count"], 60)
        self.assertAlmostEqual(aggregate["binding_ratio"], 0.3)
        self.assertEqual(aggregate["pointcloud_feasible_candidate_count"], 4)
        self.assertEqual(aggregate["pointcloud_feasible_frame_count"], 2)
        self.assertEqual(aggregate["successful_frame_count"], 1)
        self.assertAlmostEqual(aggregate["success_rate"], 0.5)
        self.assertEqual(aggregate["failure_reason_counts"], {"risk-threshold": 1})
        self.assertEqual(aggregate["selected_object_counts"], {"a.ply": 1, "b.ply": 1})


if __name__ == "__main__":
    unittest.main()
