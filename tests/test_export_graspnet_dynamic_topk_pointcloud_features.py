from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests import conftest  # noqa: F401
from scripts.export_graspnet_dynamic_topk_pointcloud_features import (
    export_dynamic_topk_pointcloud_features,
)


def _png_bytes(array: np.ndarray, mode: str | None = None) -> bytes:
    buffer = io.BytesIO()
    image = Image.fromarray(array, mode=mode) if mode else Image.fromarray(array)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _write_realsense_frame(scene_dir: Path, frame: str = "0000") -> None:
    realsense = scene_dir / "realsense"
    for folder in ("rgb", "depth", "label", "annotations"):
        (realsense / folder).mkdir(parents=True, exist_ok=True)
    label = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    intrinsic = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
    (realsense / "rgb" / f"{frame}.png").write_bytes(_png_bytes(color))
    (realsense / "depth" / f"{frame}.png").write_bytes(_png_bytes(depth, mode="I;16"))
    (realsense / "label" / f"{frame}.png").write_bytes(_png_bytes(label, mode="L"))
    (realsense / "camK.npy").write_bytes(_npy_bytes(intrinsic))
    (realsense / "annotations" / f"{frame}.xml").write_text(
        """
        <scene>
          <obj>
            <obj_id>0</obj_id>
            <obj_name>object_000.ply</obj_name>
            <pos_in_world>0 0 1</pos_in_world>
            <ori_in_world>1 0 0 0</ori_in_world>
          </obj>
        </scene>
        """,
        encoding="utf-8",
    )


def _write_prediction(prediction_root: Path, frame: str = "0000") -> None:
    prediction_dir = prediction_root / "scene_0000" / "realsense"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    row = np.zeros((1, 17), dtype=np.float32)
    row[0, 0] = 0.9
    row[0, 1] = 0.04
    row[0, 2] = 0.02
    row[0, 3] = 0.03
    row[0, 4:13] = np.eye(3).reshape(-1)
    row[0, 13:16] = [0.0, 0.0, 1.0]
    row[0, 16] = -1
    np.save(prediction_dir / f"{frame}.npy", row)


def _summary() -> dict[str, object]:
    return {
        "top_k": 1,
        "frame_results": [
            {
                "group_id": "A",
                "group_name": "test",
                "split": "final_test",
                "scene": "scene_0000",
                "frame": "0000",
                "role": "unit",
                "candidate_results": [
                    {
                        "group_id": "A",
                        "split": "final_test",
                        "scene": "scene_0000",
                        "frame": "0000",
                        "candidate_rank": 0,
                        "selected_grasp_score": 0.9,
                        "target_object_id": 0,
                        "target_object_name": "object_000.ply",
                        "lift_success": True,
                    }
                ],
            }
        ],
    }


class ExportGraspNetDynamicTopKPointCloudFeaturesTests(unittest.TestCase):
    def test_export_dynamic_topk_pointcloud_features_adds_candidate_feature_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_root = root / "scenes"
            prediction_root = root / "predictions"
            out_dir = root / "out"
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(_summary(), ensure_ascii=False), encoding="utf-8")
            _write_realsense_frame(scene_root / "scene_0000")
            _write_prediction(prediction_root)

            enriched = export_dynamic_topk_pointcloud_features(
                summary_path=summary_path,
                scene_root=scene_root,
                prediction_root=prediction_root,
                out_dir=out_dir,
                point_sample_limit=None,
            )

            candidate = enriched["frame_results"][0]["candidate_results"][0]
            self.assertEqual(candidate["binding_status"], "bound")
            self.assertEqual(candidate["binding_object_id"], 0)
            self.assertAlmostEqual(candidate["grasp_width_m"], 0.04)
            self.assertAlmostEqual(candidate["opening_limit_m"], 0.085)
            self.assertAlmostEqual(candidate["opening_over_limit_m"], 0.0)
            self.assertIn("pointcloud_feasible", candidate)
            self.assertIn("pointcloud_collision_iou", candidate)
            self.assertIn("pointcloud_empty_ratio", candidate)
            aggregate = enriched["pointcloud_feature_aggregate"]
            self.assertEqual(aggregate["opening_exceeded_count"], 0)
            self.assertEqual(aggregate["opening_exceeded_ratio"], 0.0)
            self.assertTrue((out_dir / "dynamic_topk_pointcloud_features_summary.json").exists())
            self.assertTrue((out_dir / "dynamic_topk_pointcloud_features_candidates.csv").exists())

    def test_script_help_runs(self):
        script = conftest.PROJECT_ROOT / "scripts" / "export_graspnet_dynamic_topk_pointcloud_features.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=conftest.PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--summary", result.stdout)
        self.assertIn("--prediction-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
