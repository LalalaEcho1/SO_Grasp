from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests import conftest  # noqa: F401
from scripts.export_graspnet_candidate_previews import (
    export_candidate_previews,
    parse_frames_by_scene,
    summarize_object_sets,
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


def _annotation_xml(object_ids: list[int]) -> str:
    objects = []
    for object_id in object_ids:
        objects.append(
            f"""
            <obj>
                <obj_id>{object_id}</obj_id>
                <obj_name>object_{object_id:03d}.ply</obj_name>
                <pos_in_world>{object_id / 10:.2f} 0 0</pos_in_world>
            </obj>
            """
        )
    return "<scene>" + "\n".join(objects) + "</scene>"


def _write_frame(scene_dir: Path, frame: str, *, label: np.ndarray, object_ids: list[int]) -> None:
    realsense = scene_dir / "realsense"
    for folder in ("rgb", "depth", "label", "annotations"):
        (realsense / folder).mkdir(parents=True, exist_ok=True)
    color = np.zeros((*label.shape, 3), dtype=np.uint8)
    color[..., 0] = np.where(label > 0, 150, 0).astype(np.uint8)
    depth = np.where(label > 0, 1000 + label.astype(np.uint16) * 50, 0).astype(np.uint16)
    intrinsic = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    (realsense / "rgb" / f"{frame}.png").write_bytes(_png_bytes(color))
    (realsense / "depth" / f"{frame}.png").write_bytes(_png_bytes(depth, mode="I;16"))
    (realsense / "label" / f"{frame}.png").write_bytes(_png_bytes(label, mode="L"))
    (realsense / "camK.npy").write_bytes(_npy_bytes(intrinsic))
    (realsense / "annotations" / f"{frame}.xml").write_text(_annotation_xml(object_ids), encoding="utf-8")


class ExportGraspNetCandidatePreviewsTests(unittest.TestCase):
    def test_parse_frames_by_scene_accepts_scene_colon_frames(self):
        parsed = parse_frames_by_scene(["scene_0009:0000,0036", "scene_0015:72"])

        self.assertEqual(parsed, {"scene_0009": ["0000", "0036"], "scene_0015": ["0072"]})

    def test_summarize_object_sets_groups_scenes_with_same_objects(self):
        grouped = summarize_object_sets(
            {
                "scene_a": [(1, "a.ply"), (2, "b.ply")],
                "scene_b": [(2, "b.ply"), (1, "a.ply")],
                "scene_c": [(3, "c.ply")],
            }
        )

        self.assertEqual(grouped[0]["scenes"], ["scene_a", "scene_b"])
        self.assertEqual(grouped[0]["object_count"], 2)
        self.assertEqual(grouped[1]["scenes"], ["scene_c"])

    def test_export_candidate_previews_writes_four_panel_png_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "scenes"
            out_dir = Path(tmp_dir) / "previews"
            _write_frame(
                root / "scene_0000",
                "0000",
                label=np.array([[1, 2, 0], [1, 2, 0]], dtype=np.uint8),
                object_ids=[0, 1, 2],
            )

            summary = export_candidate_previews(
                root,
                out_dir=out_dir,
                frames_by_scene={"scene_0000": ["0000"]},
                min_boundary_pixels=1,
            )

            preview_path = out_dir / "scene_0000" / "scene_0000_frame_0000_preview.png"
            self.assertTrue(preview_path.exists())
            with Image.open(preview_path) as image:
                self.assertGreater(image.width, 800)
            self.assertEqual(summary["preview_count"], 1)
            self.assertEqual(summary["previews"][0]["hidden_object_count"], 1)
            saved = json.loads((out_dir / "candidate_preview_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["object_sets"][0]["object_count"], 3)


if __name__ == "__main__":
    unittest.main()
