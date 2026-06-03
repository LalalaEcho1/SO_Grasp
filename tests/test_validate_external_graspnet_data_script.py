from __future__ import annotations

import unittest

from tests import conftest  # noqa: F401
from scripts.validate_external_graspnet_data import (
    aggregate_binding_summaries,
    aggregate_od_sufficiency_reports,
    select_frame_ids,
    select_matching_frame_ids,
)


class ValidateExternalGraspNetDataScriptTests(unittest.TestCase):
    def test_select_frame_ids_uses_even_samples_when_no_frames_requested(self):
        selected = select_frame_ids([f"{idx:04d}" for idx in range(10)], requested=(), max_frames=4)

        self.assertEqual(selected, ["0000", "0003", "0006", "0009"])

    def test_select_frame_ids_keeps_requested_existing_order(self):
        selected = select_frame_ids(["0000", "0064", "0128"], requested=("128", "0"), max_frames=None)

        self.assertEqual(selected, ["0128", "0000"])

    def test_aggregate_od_sufficiency_reports_summarizes_observability(self):
        aggregate = aggregate_od_sufficiency_reports(
            [
                {
                    "single_view_sufficient_for_complete_od": False,
                    "direct_pair_observability_ratio": 0.25,
                    "hidden_object_count": 2,
                    "unobservable_pair_count": 10,
                    "insufficiency_reasons": ["hidden_objects_present"],
                },
                {
                    "single_view_sufficient_for_complete_od": False,
                    "direct_pair_observability_ratio": 0.5,
                    "hidden_object_count": 0,
                    "unobservable_pair_count": 5,
                    "insufficiency_reasons": ["bottom_backside_contact_not_observable_from_single_view"],
                },
            ]
        )

        self.assertEqual(aggregate["frame_count"], 2)
        self.assertEqual(aggregate["sufficient_frame_count"], 0)
        self.assertAlmostEqual(aggregate["mean_direct_pair_observability_ratio"], 0.375)
        self.assertEqual(aggregate["max_hidden_object_count"], 2)
        self.assertEqual(aggregate["max_unobservable_pair_count"], 10)
        self.assertEqual(
            aggregate["insufficiency_reason_counts"],
            {
                "bottom_backside_contact_not_observable_from_single_view": 1,
                "hidden_objects_present": 1,
            },
        )

    def test_select_matching_frame_ids_uses_intersection_and_requested_order(self):
        selected = select_matching_frame_ids(
            ["0000", "0064", "0128"],
            ["0064", "0255"],
            requested=("0", "64", "255"),
            max_frames=5,
        )

        self.assertEqual(selected, ["0064"])

    def test_aggregate_binding_summaries_combines_status_and_object_counts(self):
        aggregate = aggregate_binding_summaries(
            [
                {
                    "total_candidates": 3,
                    "bound_count": 2,
                    "unbound_count": 1,
                    "status_counts": {"bound": 2, "out-of-frame": 1},
                    "grasp_pose_candidate_count": 2,
                    "grasp_pose_candidate_count_by_object": {"a.ply": 2},
                    "objects": [
                        {"label_id": 1, "object_id": 0, "object_name": "a.ply", "candidate_count": 2, "best_score": 0.8}
                    ],
                },
                {
                    "total_candidates": 2,
                    "bound_count": 1,
                    "unbound_count": 1,
                    "status_counts": {"bound": 1, "background": 1},
                    "grasp_pose_candidate_count": 1,
                    "grasp_pose_candidate_count_by_object": {"a.ply": 1},
                    "objects": [
                        {"label_id": 1, "object_id": 0, "object_name": "a.ply", "candidate_count": 1, "best_score": 0.6},
                        {"label_id": 2, "object_id": 1, "object_name": "b.ply", "candidate_count": 0, "best_score": None},
                    ],
                },
            ]
        )

        self.assertEqual(aggregate["frame_count"], 2)
        self.assertEqual(aggregate["total_candidates"], 5)
        self.assertEqual(aggregate["bound_count"], 3)
        self.assertAlmostEqual(aggregate["binding_ratio"], 0.6)
        self.assertEqual(aggregate["status_counts"], {"background": 1, "bound": 3, "out-of-frame": 1})
        self.assertEqual(aggregate["grasp_pose_candidate_count"], 3)
        self.assertEqual(aggregate["grasp_pose_object_count"], 1)
        self.assertEqual(aggregate["grasp_pose_objects"], ["a.ply"])
        self.assertEqual(aggregate["objects"][0]["total_candidate_count"], 3)
        self.assertEqual(aggregate["objects"][0]["best_score"], 0.8)


if __name__ == "__main__":
    unittest.main()
