from __future__ import annotations

import subprocess
import sys
import unittest

from tests import conftest  # noqa: F401
from scripts.evaluate_gripper_feasibility import summarize_feasibility_results
from stacked_grasping.gripper.feasibility import GraspCandidate, ObjectGraspFeasibility


class EvaluateGripperFeasibilityTests(unittest.TestCase):
    def test_script_help_runs_when_invoked_by_file_path(self):
        script_path = conftest.PROJECT_ROOT / "scripts" / "evaluate_gripper_feasibility.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=conftest.PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--scene", result.stdout)

    def test_summarize_feasibility_results_counts_feasible_objects(self):
        results = [
            ObjectGraspFeasibility(
                object_name="a",
                feasible=True,
                feasible_grasp_count=2,
                candidates=[
                    GraspCandidate("a", "x", 0.04, True, None, []),
                    GraspCandidate("a", "y", 0.04, True, None, []),
                ],
            ),
            ObjectGraspFeasibility(
                object_name="b",
                feasible=False,
                feasible_grasp_count=0,
                candidates=[
                    GraspCandidate("b", "x", 0.12, False, "opening-too-small", []),
                    GraspCandidate("b", "y", 0.04, False, "finger-collision", ["a"]),
                ],
            ),
        ]

        summary = summarize_feasibility_results(results)

        self.assertEqual(summary["object_count"], 2)
        self.assertEqual(summary["feasible_objects"], 1)
        self.assertEqual(summary["feasible_object_rate"], 0.5)
        self.assertEqual(summary["total_feasible_grasps"], 2)
        self.assertEqual(summary["mean_gripper_collision_risk"], 0.5)


if __name__ == "__main__":
    unittest.main()
