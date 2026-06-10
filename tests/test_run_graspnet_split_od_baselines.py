from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests import conftest  # noqa: F401
from scripts.run_graspnet_split_od_baselines import (
    flatten_split_config,
    run_graspnet_split_od_baselines,
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


def _annotation_xml(object_count: int) -> str:
    objects = []
    for object_id in range(object_count):
        objects.append(
            f"""
            <obj>
                <obj_id>{object_id}</obj_id>
                <obj_name>object_{object_id:03d}.ply</obj_name>
                <pos_in_world>{object_id / 10:.2f} 0 {object_id / 20:.2f}</pos_in_world>
            </obj>
            """
        )
    return "<scene>" + "\n".join(objects) + "</scene>"


def _write_frame(scene_dir: Path, frame: str, *, label: np.ndarray, object_count: int) -> None:
    realsense = scene_dir / "realsense"
    for folder in ("rgb", "depth", "label", "annotations"):
        (realsense / folder).mkdir(parents=True, exist_ok=True)
    color = np.zeros((*label.shape, 3), dtype=np.uint8)
    depth = np.where(label > 0, 1000 + label.astype(np.uint16) * 20, 0).astype(np.uint16)
    intrinsic = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    (realsense / "rgb" / f"{frame}.png").write_bytes(_png_bytes(color))
    (realsense / "depth" / f"{frame}.png").write_bytes(_png_bytes(depth, mode="I;16"))
    (realsense / "label" / f"{frame}.png").write_bytes(_png_bytes(label, mode="L"))
    (realsense / "camK.npy").write_bytes(_npy_bytes(intrinsic))
    (realsense / "annotations" / f"{frame}.xml").write_text(_annotation_xml(object_count), encoding="utf-8")


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


class RunGraspNetSplitOdBaselinesTests(unittest.TestCase):
    def test_flatten_split_config_returns_scene_level_entries(self):
        entries = flatten_split_config(_split_config())

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["group_id"], "A")
        self.assertEqual(entries[0]["scene"], "scene_0000")
        self.assertEqual(entries[0]["frame"], "0000")
        self.assertEqual(entries[0]["split"], "final_test")

    def test_run_graspnet_split_od_baselines_writes_policy_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_root = root / "scenes"
            out_dir = root / "results"
            config_path = root / "split.json"
            config_path.write_text(json.dumps(_split_config(), ensure_ascii=False), encoding="utf-8")
            _write_frame(
                scene_root / "scene_0000",
                "0000",
                label=np.array([[1, 2, 0], [1, 2, 0]], dtype=np.uint8),
                object_count=3,
            )

            summary = run_graspnet_split_od_baselines(
                config_path=config_path,
                scene_root=scene_root,
                out_dir=out_dir,
                policies=("od-only", "adaptive-score-v2"),
                max_steps=2,
                min_boundary_pixels=1,
            )

            self.assertEqual(summary["frame_count"], 1)
            self.assertEqual(summary["policy_count"], 2)
            self.assertEqual(summary["result_count"], 2)
            self.assertTrue((out_dir / "split_od_baseline_summary.json").exists())
            self.assertTrue((out_dir / "split_od_baseline_results.csv").exists())
            self.assertEqual(summary["results"][0]["hidden_object_count"], 1)
            self.assertIn("selected_sequence", summary["results"][0])


if __name__ == "__main__":
    unittest.main()
