from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import conftest  # noqa: F401
from scripts.sweep_adaptive_v2 import aggregate_policy_rows, load_selected_scene_paths, parse_thresholds


class AdaptiveV2SweepTests(unittest.TestCase):
    def test_parse_thresholds_returns_floats_in_cli_order(self):
        self.assertEqual(parse_thresholds(["0.25", "0.35", "0.45"]), [0.25, 0.35, 0.45])

    def test_script_help_runs_when_invoked_by_file_path(self):
        script_path = conftest.PROJECT_ROOT / "scripts" / "sweep_adaptive_v2.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=conftest.PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--thresholds", result.stdout)

    def test_load_selected_scene_paths_accepts_previous_selected_scene_format(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "selected.json"
            path.write_text(
                json.dumps(
                    [
                        {"path": r"D:\Stacked-Object Grasping\assets\scenes\generated_main_v1\scene_0001.xml"},
                        {"path": r"D:\Stacked-Object Grasping\assets\scenes\generated_main_v1\scene_0002.xml"},
                    ]
                ),
                encoding="utf-8",
            )

            paths = load_selected_scene_paths(path)

        self.assertEqual([item.name for item in paths], ["scene_0001.xml", "scene_0002.xml"])
        self.assertEqual(paths[0], conftest.PROJECT_ROOT / "assets" / "scenes" / "generated_main_v1" / "scene_0001.xml")

    def test_aggregate_policy_rows_summarizes_failure_metrics(self):
        rows = [
            {
                "policy": "adaptive-score-v2",
                "clearance_rate": 1.0,
                "success_rate": 1.0,
                "num_failures": 0,
                "mean_grasp_risk": 0.10,
                "max_grasp_risk": 0.20,
                "num_gripper_infeasible_steps": 0,
                "mean_selected_gripper_collision_risk": 0.25,
                "max_selected_gripper_collision_risk": 0.5,
            },
            {
                "policy": "adaptive-score-v2",
                "clearance_rate": 0.5,
                "success_rate": 0.8,
                "num_failures": 1,
                "mean_grasp_risk": 0.20,
                "max_grasp_risk": 0.40,
                "num_gripper_infeasible_steps": 1,
                "mean_selected_gripper_collision_risk": 0.75,
                "max_selected_gripper_collision_risk": 1.0,
            },
        ]

        summary = aggregate_policy_rows(rows, risk_threshold=0.35)

        self.assertEqual(summary[0]["risk_threshold"], 0.35)
        self.assertEqual(summary[0]["policy"], "adaptive-score-v2")
        self.assertEqual(summary[0]["episodes"], 2)
        self.assertEqual(summary[0]["avg_clearance_rate"], 0.75)
        self.assertEqual(summary[0]["total_failures"], 1)
        self.assertEqual(summary[0]["avg_mean_grasp_risk"], 0.15)
        self.assertEqual(summary[0]["total_gripper_infeasible_steps"], 1)
        self.assertEqual(summary[0]["avg_mean_gripper_collision_risk"], 0.5)


if __name__ == "__main__":
    unittest.main()
