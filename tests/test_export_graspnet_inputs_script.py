from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests import conftest  # noqa: F401
from scripts.export_graspnet_inputs import resolve_batch_scene_paths


class ExportGraspNetInputsScriptTests(unittest.TestCase):
    def test_resolve_batch_scene_paths_reads_sorted_directory_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "b_scene.xml").write_text("<mujoco/>", encoding="utf-8")
            (root / "a_scene.xml").write_text("<mujoco/>", encoding="utf-8")
            (root / "ignore.txt").write_text("", encoding="utf-8")

            paths = resolve_batch_scene_paths(scene_paths=(), scene_dir=root, limit=1)

        self.assertEqual([path.name for path in paths], ["a_scene.xml"])

    def test_resolve_batch_scene_paths_keeps_explicit_order(self):
        paths = resolve_batch_scene_paths(
            scene_paths=(Path("second.xml"), Path("first.xml")),
            scene_dir=None,
            limit=None,
        )

        self.assertEqual(paths, [Path("second.xml"), Path("first.xml")])


if __name__ == "__main__":
    unittest.main()
