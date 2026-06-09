from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests import conftest  # noqa: F401
from scripts.scan_graspnet_scenes import (
    aggregate_scene_records,
    scan_graspnet_scenes,
    select_scene_names,
    write_scan_outputs,
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
                <pos_in_world>0 0 {object_id / 10:.2f}</pos_in_world>
            </obj>
            """
        )
    return "<scene>" + "\n".join(objects) + "</scene>"


def _write_frame(scene_dir: Path, frame: str, *, label: np.ndarray, object_count: int) -> None:
    realsense = scene_dir / "realsense"
    for folder in ("rgb", "depth", "label", "annotations"):
        (realsense / folder).mkdir(parents=True, exist_ok=True)

    color = np.zeros((*label.shape, 3), dtype=np.uint8)
    depth = np.where(label > 0, 1000, 0).astype(np.uint16)
    intrinsic = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    (realsense / "rgb" / f"{frame}.png").write_bytes(_png_bytes(color))
    (realsense / "depth" / f"{frame}.png").write_bytes(_png_bytes(depth, mode="I;16"))
    (realsense / "label" / f"{frame}.png").write_bytes(_png_bytes(label, mode="L"))
    (realsense / "camK.npy").write_bytes(_npy_bytes(intrinsic))
    (realsense / "annotations" / f"{frame}.xml").write_text(_annotation_xml(object_count), encoding="utf-8")


class ScanGraspNetScenesScriptTests(unittest.TestCase):
    def test_select_scene_names_prefers_scene_directories_and_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "scene_0002").mkdir()
            (root / "scene_0000").mkdir()
            (root / "notes").mkdir()

            self.assertEqual(select_scene_names(root, max_scenes=1), ["scene_0000"])

    def test_scan_graspnet_scenes_ranks_more_occluded_scene_higher(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_frame(
                root / "scene_0000",
                "0000",
                label=np.array([[1, 2, 0], [1, 2, 0]], dtype=np.uint8),
                object_count=3,
            )
            _write_frame(
                root / "scene_0001",
                "0000",
                label=np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8),
                object_count=3,
            )

            result = scan_graspnet_scenes(
                root,
                max_scenes=None,
                max_frames_per_scene=1,
                min_boundary_pixels=1,
            )

        self.assertEqual([record["scene"] for record in result["frame_records"]], ["scene_0000", "scene_0001"])
        self.assertEqual(result["frame_records"][0]["hidden_object_count"], 1)
        self.assertGreaterEqual(result["frame_records"][0]["visible_boundary_edge_count"], 1)
        self.assertEqual(result["scene_summaries"][0]["scene"], "scene_0000")
        self.assertGreater(result["scene_summaries"][0]["difficulty_score"], result["scene_summaries"][1]["difficulty_score"])

    def test_aggregate_scene_records_reports_mean_metrics(self):
        summaries = aggregate_scene_records(
            [
                {
                    "scene": "scene_0000",
                    "hidden_object_count": 2,
                    "unobservable_pair_count": 5,
                    "visible_boundary_edge_count": 3,
                    "direct_pair_observability_ratio": 0.25,
                    "complete_object_count": 4,
                },
                {
                    "scene": "scene_0000",
                    "hidden_object_count": 0,
                    "unobservable_pair_count": 1,
                    "visible_boundary_edge_count": 1,
                    "direct_pair_observability_ratio": 0.75,
                    "complete_object_count": 2,
                },
            ]
        )

        self.assertEqual(summaries[0]["frame_count"], 2)
        self.assertAlmostEqual(summaries[0]["mean_hidden_object_count"], 1.0)
        self.assertAlmostEqual(summaries[0]["mean_direct_pair_observability_ratio"], 0.5)
        self.assertGreater(summaries[0]["difficulty_score"], 0.0)

    def test_write_scan_outputs_saves_json_and_csv_tables(self):
        result = {
            "scene_summaries": [{"scene": "scene_0000", "difficulty_score": 1.25}],
            "frame_records": [{"scene": "scene_0000", "frame": "0000", "insufficiency_reasons": ["hidden"]}],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            write_scan_outputs(result, Path(tmp_dir))

            self.assertTrue((Path(tmp_dir) / "graspnet_scene_scan_summary.json").exists())
            self.assertIn("scene_0000", (Path(tmp_dir) / "graspnet_scene_scan_scenes.csv").read_text(encoding="utf-8-sig"))
            self.assertIn("hidden", (Path(tmp_dir) / "graspnet_scene_scan_frames.csv").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
