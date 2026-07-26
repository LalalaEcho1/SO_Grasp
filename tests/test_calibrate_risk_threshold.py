from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests import conftest  # noqa: F401
from scripts.calibrate_risk_threshold import (
    collect_pairs_from_validation,
    compute_auc,
    compute_threshold_table,
    reliability_bins,
    run_calibration,
    suggest_thresholds,
)


class CalibrateRiskThresholdTests(unittest.TestCase):
    def test_separable_risks_give_perfect_auc_and_threshold_between_groups(self):
        risks = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        successes = [True, True, True, False, False, False]

        auc = compute_auc(risks, successes)
        table = compute_threshold_table(risks, successes)
        suggestions = suggest_thresholds(table, target_precision=0.9)

        self.assertAlmostEqual(auc, 1.0)
        best = suggestions["max_youden"]
        self.assertGreater(best["threshold"], 0.3)
        self.assertLess(best["threshold"], 0.7)
        self.assertAlmostEqual(best["tpr"], 1.0)
        self.assertAlmostEqual(best["fpr"], 0.0)
        precise = suggestions["max_threshold_at_target_precision"]
        self.assertIsNotNone(precise)
        self.assertGreaterEqual(precise["precision"], 0.9)

    def test_threshold_table_counts_confusion_matrix(self):
        table = compute_threshold_table([0.2, 0.6], [True, False], thresholds=[0.4])

        self.assertEqual(table[0]["tp"], 1)
        self.assertEqual(table[0]["fp"], 0)
        self.assertEqual(table[0]["tn"], 1)
        self.assertEqual(table[0]["fn"], 0)
        self.assertAlmostEqual(table[0]["youden_j"], 1.0)

    def test_reliability_bins_report_success_rate_per_risk_band(self):
        rows = reliability_bins([0.05, 0.15, 0.95], [True, True, False], bins=10)

        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["count"], 1)
        self.assertAlmostEqual(rows[0]["success_rate"], 1.0)
        self.assertEqual(rows[9]["count"], 1)
        self.assertAlmostEqual(rows[9]["success_rate"], 0.0)
        self.assertEqual(rows[5]["count"], 0)
        self.assertIsNone(rows[5]["success_rate"])

    def test_run_calibration_writes_outputs(self):
        pairs = [
            {"risk": 0.1, "success": True},
            {"risk": 0.2, "success": True},
            {"risk": 0.8, "success": False},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "calibration"
            summary = run_calibration(pairs, out_dir=out_dir, make_plot=False)

            self.assertEqual(summary["pair_count"], 3)
            self.assertAlmostEqual(summary["auc"], 1.0)
            self.assertTrue((out_dir / "calibration_table.csv").exists())
            self.assertTrue((out_dir / "reliability_bins.csv").exists())
            self.assertTrue((out_dir / "calibration_summary.json").exists())

    def test_collect_pairs_joins_frame_results_with_validation_summaries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            frame_results = tmp / "frame_results.csv"
            frame_results.write_text(
                "frame,selected_object,grasp_risk\n"
                "0014,tape.ply,0.52\n"
                "0023,banana.ply,0.21\n"
                "0033,banana.ply,0.30\n",
                encoding="utf-8",
            )
            validation_dir = tmp / "validation"
            validation_dir.mkdir()
            (validation_dir / "a.json").write_text(
                json.dumps({"frame": "0014", "target_object_name": "tape.ply", "validation": {"lift_success": True}}),
                encoding="utf-8",
            )
            (validation_dir / "b.json").write_text(
                json.dumps({"target_object_name": "banana.ply", "validation": {"lift_success": False}}),
                encoding="utf-8",
            )
            (validation_dir / "c.json").write_text(
                json.dumps({"target_object_name": "missing.ply", "validation": {"lift_success": True}}),
                encoding="utf-8",
            )

            pairs, skipped = collect_pairs_from_validation(frame_results, validation_dir)

        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["risk"], 0.52)
        self.assertTrue(pairs[0]["success"])
        reasons = {item["reason"] for item in skipped}
        self.assertIn("ambiguous-object-match", reasons)
        self.assertIn("no-episode-match", reasons)


if __name__ == "__main__":
    unittest.main()
