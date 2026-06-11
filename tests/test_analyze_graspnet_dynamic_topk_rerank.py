from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import conftest  # noqa: F401
from scripts.analyze_graspnet_dynamic_topk_rerank import (
    analyze_dynamic_topk_rerank,
    choose_object_consensus_candidate,
)


def _candidate(
    rank: int,
    *,
    target_id: int,
    score: float,
    success: bool,
) -> dict[str, object]:
    return {
        "group_id": "A",
        "split": "final_test",
        "scene": "scene_0000",
        "frame": "0000",
        "role": "unit",
        "candidate_rank": rank,
        "selected_grasp_score": score,
        "target_object_id": target_id,
        "target_object_name": f"object_{target_id:03d}.ply",
        "lift_success": success,
        "failure_reason": None if success else "insufficient_lift",
        "simulation_unstable": False,
        "target_lift_delta_m": 0.03 if success else 0.002,
        "max_target_lift_delta_m": 0.035 if success else 0.01,
    }


def _summary() -> dict[str, object]:
    frame_a_candidates = [
        _candidate(0, target_id=1, score=0.95, success=False),
        _candidate(1, target_id=2, score=0.82, success=True),
        _candidate(2, target_id=2, score=0.74, success=False),
    ]
    frame_b_candidates = [
        _candidate(0, target_id=3, score=0.90, success=True),
        _candidate(1, target_id=4, score=0.80, success=False),
    ]
    return {
        "top_k": 3,
        "processed_frame_count": 2,
        "frame_results": [
            {
                "group_id": "A",
                "group_name": "test",
                "split": "final_test",
                "scene": "scene_0000",
                "frame": "0000",
                "role": "unit",
                "candidate_results": frame_a_candidates,
            },
            {
                "group_id": "A",
                "group_name": "test",
                "split": "final_test",
                "scene": "scene_0000",
                "frame": "0001",
                "role": "unit",
                "candidate_results": [
                    {**candidate, "frame": "0001"} for candidate in frame_b_candidates
                ],
            },
        ],
    }


class AnalyzeGraspNetDynamicTopKRerankTests(unittest.TestCase):
    def test_choose_object_consensus_candidate_uses_target_repeat_without_dynamic_labels(self):
        candidates = [
            _candidate(0, target_id=1, score=0.95, success=False),
            _candidate(1, target_id=2, score=0.82, success=True),
            _candidate(2, target_id=2, score=0.74, success=False),
        ]

        selected = choose_object_consensus_candidate(candidates)

        self.assertEqual(selected["candidate_rank"], 1)
        self.assertEqual(selected["target_object_id"], 2)

    def test_analyze_dynamic_topk_rerank_reports_policy_success_rates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "split_dynamic_topk_summary.json"
            out_dir = root / "analysis"
            summary_path.write_text(json.dumps(_summary(), ensure_ascii=False), encoding="utf-8")

            result = analyze_dynamic_topk_rerank(summary_path=summary_path, out_dir=out_dir)

            top1 = result["policy_aggregate"]["top1"]
            consensus = result["policy_aggregate"]["object-consensus"]
            self.assertEqual(result["frame_count"], 2)
            self.assertEqual(top1["success_count"], 1)
            self.assertEqual(top1["success_rate"], 0.5)
            self.assertEqual(consensus["success_count"], 2)
            self.assertEqual(consensus["success_rate"], 1.0)
            self.assertTrue((out_dir / "dynamic_topk_rerank_analysis.json").exists())
            self.assertTrue((out_dir / "dynamic_topk_rerank_policy_results.csv").exists())
            self.assertTrue((out_dir / "dynamic_topk_candidate_label_analysis.csv").exists())

    def test_script_help_runs(self):
        script = conftest.PROJECT_ROOT / "scripts" / "analyze_graspnet_dynamic_topk_rerank.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=conftest.PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--summary", result.stdout)
        self.assertIn("--out-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
