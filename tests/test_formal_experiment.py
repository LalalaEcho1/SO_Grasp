from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests import conftest  # noqa: F401
from scripts.run_formal_main_v1_experiment import (
    aggregate_formal_rows,
    annotate_episode_rows,
    load_difficulty_scene_records,
)


class FormalExperimentTests(unittest.TestCase):
    def test_load_difficulty_scene_records_flattens_groups_with_difficulty(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "difficulty_splits.json"
            path.write_text(
                json.dumps(
                    {
                        "groups": {
                            "easy": {
                                "scenes": [
                                    {
                                        "index": 1,
                                        "scene": "scene_0001.xml",
                                        "path": r"D:\Stacked-Object Grasping\assets\scene_0001.xml",
                                        "contact_pairs": 2,
                                        "visible_edges": 5,
                                        "max_top_z": 0.55,
                                    }
                                ]
                            },
                            "hard": {
                                "scenes": [
                                    {
                                        "index": 2,
                                        "scene": "scene_0002.xml",
                                        "path": r"D:\Stacked-Object Grasping\assets\scene_0002.xml",
                                        "contact_pairs": 5,
                                        "visible_edges": 20,
                                        "max_top_z": 0.80,
                                    }
                                ]
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            records = load_difficulty_scene_records(path)

        self.assertEqual([record["difficulty"] for record in records], ["easy", "hard"])
        self.assertEqual(records[0]["scene_index"], 1)
        self.assertEqual(records[0]["path"], "assets/scene_0001.xml")
        self.assertEqual(records[1]["visible_edges"], 20)

    def test_annotate_episode_rows_adds_scene_metadata_and_threshold(self):
        records = [
            {
                "scene_index": 1,
                "scene": "scene_0001.xml",
                "path": r"D:\Stacked-Object Grasping\assets\scene_0001.xml",
                "difficulty": "easy",
                "contact_pairs": 2,
                "visible_edges": 5,
                "max_top_z": 0.55,
            }
        ]
        rows = [
            {
                "policy": "adaptive-score-v2",
                "scene": r"D:\Stacked-Object Grasping\assets\scene_0001.xml",
                "clearance_rate": 1.0,
                "success_rate": 1.0,
                "num_failures": 0,
                "mean_grasp_risk": 0.1,
                "max_grasp_risk": 0.2,
                "num_gripper_infeasible_steps": 0,
                "mean_selected_gripper_collision_risk": 0.25,
                "max_selected_gripper_collision_risk": 0.5,
            }
        ]

        annotated = annotate_episode_rows(rows, records, risk_threshold=0.35)

        self.assertEqual(annotated[0]["difficulty"], "easy")
        self.assertEqual(annotated[0]["scene_index"], 1)
        self.assertEqual(annotated[0]["risk_threshold"], 0.35)

    def test_aggregate_formal_rows_reports_overall_and_difficulty_summaries(self):
        rows = [
            {
                "policy": "adaptive-score-v2",
                "candidate_source": "rule",
                "difficulty": "easy",
                "clearance_rate": 1.0,
                "success_rate": 1.0,
                "num_failures": 0,
                "mean_grasp_risk": 0.1,
                "max_grasp_risk": 0.2,
                "num_gripper_infeasible_steps": 0,
                "mean_selected_gripper_collision_risk": 0.25,
                "max_selected_gripper_collision_risk": 0.5,
            },
            {
                "policy": "adaptive-score-v2",
                "candidate_source": "graspnet",
                "difficulty": "hard",
                "clearance_rate": 0.5,
                "success_rate": 0.8,
                "num_failures": 1,
                "mean_grasp_risk": 0.3,
                "max_grasp_risk": 0.5,
                "num_gripper_infeasible_steps": 1,
                "mean_selected_gripper_collision_risk": 0.75,
                "max_selected_gripper_collision_risk": 1.0,
            },
        ]

        summary = aggregate_formal_rows(rows)

        overall = next(row for row in summary if row["difficulty"] == "overall" and row["candidate_source"] == "rule")
        hard = next(row for row in summary if row["difficulty"] == "hard")
        self.assertEqual({row["candidate_source"] for row in summary if row["difficulty"] == "overall"}, {"rule", "graspnet"})
        self.assertEqual(overall["episodes"], 1)
        self.assertEqual(overall["avg_clearance_rate"], 1.0)
        self.assertEqual(overall["total_failures"], 0)
        self.assertEqual(overall["total_gripper_infeasible_steps"], 0)
        self.assertEqual(overall["avg_mean_gripper_collision_risk"], 0.25)
        self.assertEqual(overall["avg_max_gripper_collision_risk"], 0.5)
        self.assertEqual(hard["episodes"], 1)
        self.assertEqual(hard["avg_success_rate"], 0.8)
        self.assertEqual(hard["candidate_source"], "graspnet")


if __name__ == "__main__":
    unittest.main()
