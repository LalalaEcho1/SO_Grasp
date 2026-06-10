from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests import conftest  # noqa: F401
from scripts.run_graspnet_split_pointcloud_episodes import (
    prediction_path_for_scene,
    run_graspnet_split_pointcloud_episodes,
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


def _annotation_xml() -> str:
    return """
    <scene>
      <obj>
        <obj_id>0</obj_id>
        <obj_name>object_000.ply</obj_name>
        <pos_in_world>0 0 0</pos_in_world>
      </obj>
    </scene>
    """


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
    (realsense / "annotations" / f"{frame}.xml").write_text(_annotation_xml(), encoding="utf-8")


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


def _split_config() -> dict[str, object]:
    return {
        "version": "test",
        "scene_root": "scenes",
        "split_policy": "scene_level",
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
                        "preview_frames": ["0000"],
                        "selected_frames": ["0000"],
                    }
                ],
            }
        ],
    }


class RunGraspNetSplitPointCloudEpisodesTests(unittest.TestCase):
    def test_prediction_path_for_scene_prefers_scene_camera_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            expected = root / "scene_0000" / "realsense"
            expected.mkdir(parents=True)

            self.assertEqual(prediction_path_for_scene(root, "scene_0000", "realsense"), expected)

    def test_run_graspnet_split_pointcloud_episodes_reports_binding_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_root = root / "scenes"
            prediction_root = root / "predictions"
            out_dir = root / "results"
            config_path = root / "split.json"
            config_path.write_text(json.dumps(_split_config(), ensure_ascii=False), encoding="utf-8")
            _write_realsense_frame(scene_root / "scene_0000")
            _write_prediction(prediction_root)

            summary = run_graspnet_split_pointcloud_episodes(
                config_path=config_path,
                scene_root=scene_root,
                prediction_root=prediction_root,
                out_dir=out_dir,
                min_points_per_object=1,
                point_sample_limit=10,
                binding_pixel_radius=1,
                binding_depth_tolerance_m=0.1,
            )

            self.assertEqual(summary["frame_count"], 1)
            self.assertEqual(summary["aggregate"]["total_candidates"], 1)
            self.assertEqual(summary["aggregate"]["bound_count"], 1)
            self.assertEqual(summary["frame_results"][0]["scene"], "scene_0000")
            self.assertEqual(summary["frame_results"][0]["frame"], "0000")
            self.assertTrue((out_dir / "split_graspnet_pointcloud_summary.json").exists())
            self.assertTrue((out_dir / "split_graspnet_pointcloud_frame_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
