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
    attach_object_prior_features,
    choose_od_pointcloud_compact_candidate,
    choose_object_consensus_candidate,
    choose_pointcloud_feasible_score_candidate,
    choose_pointcloud_soft_score_candidate,
)


def _candidate(
    rank: int,
    *,
    target_id: int,
    score: float,
    success: bool,
    pointcloud_feasible: bool | None = None,
    opening_over_limit_m: float | None = None,
) -> dict[str, object]:
    row = {
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
    if pointcloud_feasible is not None:
        row["pointcloud_feasible"] = pointcloud_feasible
        row["pointcloud_collision_iou"] = 0.001 if pointcloud_feasible else 0.2
        row["pointcloud_empty_ratio"] = 0.5 if pointcloud_feasible else 0.001
        row["pointcloud_failure_reason"] = None if pointcloud_feasible else "opening-too-small"
        row["opening_over_limit_m"] = 0.0 if opening_over_limit_m is None else opening_over_limit_m
        row["grasp_width_m"] = 0.085 + row["opening_over_limit_m"]
    return row


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

    def test_choose_pointcloud_feasible_score_candidate_prefers_feasible_candidate_without_dynamic_labels(self):
        candidates = [
            _candidate(0, target_id=1, score=0.95, success=False, pointcloud_feasible=False),
            _candidate(1, target_id=2, score=0.82, success=True, pointcloud_feasible=True),
            _candidate(2, target_id=3, score=0.60, success=False, pointcloud_feasible=True),
        ]

        selected = choose_pointcloud_feasible_score_candidate(candidates)

        self.assertEqual(selected["candidate_rank"], 1)
        self.assertEqual(selected["target_object_id"], 2)

    def test_choose_pointcloud_soft_score_candidate_tolerates_small_opening_excess(self):
        candidates = [
            _candidate(
                0,
                target_id=1,
                score=0.92,
                success=True,
                pointcloud_feasible=False,
                opening_over_limit_m=0.003,
            ),
            _candidate(1, target_id=2, score=0.80, success=False, pointcloud_feasible=True),
            _candidate(
                2,
                target_id=3,
                score=0.95,
                success=False,
                pointcloud_feasible=False,
                opening_over_limit_m=0.04,
            ),
        ]

        selected = choose_pointcloud_soft_score_candidate(candidates)

        self.assertEqual(selected["candidate_rank"], 0)
        self.assertEqual(selected["target_object_id"], 1)

    def test_attach_object_prior_features_adds_adaptive_v2_fields_to_candidate_targets(self):
        candidates = [
            _candidate(0, target_id=1, score=0.95, success=False),
            _candidate(1, target_id=2, score=0.75, success=True),
        ]
        prior_by_object = {
            "object_001.ply": {
                "object_rank": 3,
                "object_adaptive_v2_score": 0.2,
                "object_adaptive_v2_score_norm": 0.1,
                "object_grasp_risk": 0.4,
                "object_high_risk": True,
                "object_height_priority": 0.2,
            },
            "object_002.ply": {
                "object_rank": 0,
                "object_adaptive_v2_score": 1.2,
                "object_adaptive_v2_score_norm": 0.9,
                "object_grasp_risk": 0.1,
                "object_high_risk": False,
                "object_height_priority": 0.8,
            },
        }

        attach_object_prior_features(candidates, prior_by_object)

        self.assertEqual(candidates[0]["object_rank"], 3)
        self.assertAlmostEqual(candidates[1]["object_adaptive_v2_score_norm"], 0.9)
        self.assertEqual(candidates[1]["object_high_risk"], False)

    def test_choose_od_pointcloud_compact_candidate_combines_object_prior_and_compact_grasp(self):
        candidates = [
            _candidate(0, target_id=1, score=0.95, success=False, pointcloud_feasible=True),
            _candidate(1, target_id=2, score=0.72, success=True, pointcloud_feasible=True),
        ]
        candidates[0].update(
            {
                "pointcloud_empty_ratio": 0.82,
                "object_adaptive_v2_score_norm": 0.1,
                "object_grasp_risk": 0.35,
                "object_high_risk": True,
            }
        )
        candidates[1].update(
            {
                "pointcloud_empty_ratio": 0.16,
                "object_adaptive_v2_score_norm": 0.9,
                "object_grasp_risk": 0.05,
                "object_high_risk": False,
            }
        )

        selected = choose_od_pointcloud_compact_candidate(candidates)

        self.assertEqual(selected["candidate_rank"], 1)
        self.assertEqual(selected["target_object_id"], 2)

    def test_choose_od_pointcloud_compact_candidate_penalizes_unbound_candidate(self):
        candidates = [
            _candidate(0, target_id=1, score=0.50, success=True, pointcloud_feasible=True),
            _candidate(1, target_id=2, score=0.50, success=False, pointcloud_feasible=False),
        ]
        candidates[0].update(
            {
                "binding_status": "bound",
                "pointcloud_empty_ratio": 0.20,
                "object_adaptive_v2_score_norm": 0.40,
                "object_high_risk": False,
            }
        )
        candidates[1].update(
            {
                "binding_status": "binding-background",
                "pointcloud_empty_ratio": 0.10,
                "object_adaptive_v2_score_norm": 1.00,
                "object_high_risk": False,
            }
        )

        selected = choose_od_pointcloud_compact_candidate(candidates)

        self.assertEqual(selected["candidate_rank"], 0)
        self.assertEqual(selected["target_object_id"], 1)

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

    def test_analyze_dynamic_topk_rerank_can_run_without_saving_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "split_dynamic_topk_summary.json"
            out_dir = root / "analysis"
            summary_path.write_text(json.dumps(_summary(), ensure_ascii=False), encoding="utf-8")

            result = analyze_dynamic_topk_rerank(
                summary_path=summary_path,
                out_dir=out_dir,
                save_outputs=False,
            )

        self.assertEqual(result["output_saved"], False)
        self.assertFalse(out_dir.exists())

    def test_analyze_dynamic_topk_rerank_can_attach_object_priors_from_provider(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "split_dynamic_topk_summary.json"
            out_dir = root / "analysis"
            summary_path.write_text(json.dumps(_summary(), ensure_ascii=False), encoding="utf-8")

            result = analyze_dynamic_topk_rerank(
                summary_path=summary_path,
                out_dir=out_dir,
                save_outputs=False,
                object_prior_provider=lambda frame: {
                    "object_001.ply": {
                        "object_rank": 1,
                        "object_adaptive_v2_score": 0.1,
                        "object_adaptive_v2_score_norm": 0.2,
                        "object_grasp_risk": 0.4,
                        "object_high_risk": True,
                        "object_height_priority": 0.1,
                    },
                    "object_002.ply": {
                        "object_rank": 0,
                        "object_adaptive_v2_score": 1.0,
                        "object_adaptive_v2_score_norm": 0.9,
                        "object_grasp_risk": 0.1,
                        "object_high_risk": False,
                        "object_height_priority": 0.9,
                    },
                },
            )

            first_policy_row = result["policy_results"][0]
            self.assertIn("od-pointcloud-compact", result["policy_aggregate"])
            self.assertIn("object_adaptive_v2_score_norm", first_policy_row)

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
        self.assertIn("--no-save", result.stdout)
        self.assertIn("--scene-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
