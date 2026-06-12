from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import conftest  # noqa: F401
from scripts.run_graspnet_split_dynamic_topk import (
    choose_best_candidate,
    run_graspnet_split_dynamic_topk,
)


def _split_config() -> dict[str, object]:
    return {
        "version": "test",
        "scene_root": "scenes",
        "groups": [
            {
                "group_id": "A",
                "name": "test group",
                "scenes": [
                    {
                        "scene": "scene_0000",
                        "split": "final_test",
                        "role": "test",
                        "tags": ["unit"],
                        "selected_frames": ["0000"],
                    }
                ],
            }
        ],
    }


def _candidate_summary(rank: int, *, success: bool, lift: float, unstable: bool = False) -> dict[str, object]:
    return {
        "scene": "scene_0000",
        "camera": "realsense",
        "frame": "0000",
        "candidate_rank": rank,
        "selected_grasp_score": 1.0 - rank * 0.1,
        "target_object_id": 0,
        "target_object_name": "object_000.ply",
        "target_body_name": "obj_000_object_000",
        "validation": {
            "compile_success": True,
            "lift_success": success,
            "failure_reason": None if success else ("simulation_unstable" if unstable else "insufficient_lift"),
            "simulation_unstable": unstable,
            "target_contact_step_count": 10 + rank,
            "lift_contact_step_count": 5 + rank,
            "target_lift_delta_m": lift,
            "max_target_lift_delta_m": max(lift, 0.0),
        },
    }


class RunGraspNetSplitDynamicTopKTests(unittest.TestCase):
    def test_choose_best_candidate_prefers_successful_non_unstable_candidate(self):
        rows = [
            _candidate_summary(0, success=False, lift=0.04, unstable=True),
            _candidate_summary(1, success=False, lift=0.018),
            _candidate_summary(2, success=True, lift=0.022),
        ]

        best = choose_best_candidate(rows)

        self.assertEqual(best["candidate_rank"], 2)
        self.assertEqual(best["validation"]["lift_success"], True)

    def test_run_graspnet_split_dynamic_topk_reports_top1_and_topk_success_rates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "split.json"
            config_path.write_text(json.dumps(_split_config(), ensure_ascii=False), encoding="utf-8")
            out_dir = root / "results"

            def fake_validator(**kwargs):
                rank = int(kwargs["candidate_rank"])
                return _candidate_summary(rank, success=(rank == 1), lift=0.024 if rank == 1 else 0.002)

            summary = run_graspnet_split_dynamic_topk(
                config_path=config_path,
                scene_root=root / "scenes",
                dataset_root=root / "dataset",
                prediction_root=root / "predictions",
                out_dir=out_dir,
                top_k=3,
                stop_on_success=True,
                validator=fake_validator,
            )

            self.assertEqual(summary["frame_count"], 1)
            self.assertEqual(summary["processed_frame_count"], 1)
            self.assertEqual(summary["top1_lift_success_count"], 0)
            self.assertEqual(summary["topk_lift_success_count"], 1)
            self.assertEqual(summary["candidate_evaluation_count"], 2)
            frame = summary["frame_results"][0]
            self.assertEqual(frame["best_candidate_rank"], 1)
            self.assertEqual(frame["evaluated_candidate_count"], 2)
            self.assertTrue(frame["topk_lift_success"])
            self.assertTrue((out_dir / "split_dynamic_topk_summary.json").exists())
            self.assertTrue((out_dir / "split_dynamic_topk_frame_results.csv").exists())
            self.assertTrue((out_dir / "split_dynamic_topk_candidate_results.csv").exists())

    def test_run_graspnet_split_dynamic_topk_can_run_without_saving_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "split.json"
            config_path.write_text(json.dumps(_split_config(), ensure_ascii=False), encoding="utf-8")
            out_dir = root / "results"
            validator_out_dirs: list[Path] = []

            def fake_validator(**kwargs):
                validator_out_dirs.append(Path(kwargs["out_dir"]))
                return _candidate_summary(int(kwargs["candidate_rank"]), success=False, lift=0.001)

            summary = run_graspnet_split_dynamic_topk(
                config_path=config_path,
                scene_root=root / "scenes",
                dataset_root=root / "dataset",
                prediction_root=root / "predictions",
                out_dir=out_dir,
                top_k=1,
                save_outputs=False,
                validator=fake_validator,
            )

            self.assertEqual(summary["output_saved"], False)
            self.assertFalse(out_dir.exists())
            self.assertEqual(len(validator_out_dirs), 1)
            self.assertNotEqual(validator_out_dirs[0], out_dir)

    def test_run_graspnet_split_dynamic_topk_passes_gripper_backend_to_validator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "split.json"
            config_path.write_text(json.dumps(_split_config(), ensure_ascii=False), encoding="utf-8")
            gripper_xml = root / "mujoco_menagerie" / "robotiq_2f85" / "2f85.xml"
            seen_kwargs: list[dict[str, object]] = []

            def fake_validator(**kwargs):
                seen_kwargs.append(dict(kwargs))
                return {
                    **_candidate_summary(int(kwargs["candidate_rank"]), success=False, lift=0.001),
                    "gripper_backend": kwargs["gripper_backend"],
                    "robotiq_2f85_xml": str(kwargs["robotiq_2f85_xml"]),
                }

            summary = run_graspnet_split_dynamic_topk(
                config_path=config_path,
                scene_root=root / "scenes",
                dataset_root=root / "dataset",
                prediction_root=root / "predictions",
                out_dir=root / "results",
                top_k=1,
                save_outputs=False,
                gripper_backend="robotiq-2f85",
                robotiq_2f85_xml=gripper_xml,
                validator=fake_validator,
            )

        self.assertEqual(summary["gripper_backend"], "robotiq-2f85")
        self.assertEqual(summary["robotiq_2f85_xml"], str(gripper_xml))
        self.assertEqual(len(seen_kwargs), 1)
        self.assertEqual(seen_kwargs[0]["gripper_backend"], "robotiq-2f85")
        self.assertEqual(seen_kwargs[0]["robotiq_2f85_xml"], gripper_xml)

    def test_script_help_runs(self):
        script = conftest.PROJECT_ROOT / "scripts" / "run_graspnet_split_dynamic_topk.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=conftest.PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--top-k", result.stdout)
        self.assertIn("--stop-on-success", result.stdout)
        self.assertIn("--no-save", result.stdout)
        self.assertIn("--gripper-backend", result.stdout)
        self.assertIn("--robotiq-2f85-xml", result.stdout)


if __name__ == "__main__":
    unittest.main()
