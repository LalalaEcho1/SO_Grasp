from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests import conftest  # noqa: F401
from scripts.normalize_project_paths import normalize_json_project_paths


class NormalizeProjectPathsTests(unittest.TestCase):
    def test_normalize_json_project_paths_rewrites_nested_windows_paths(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "source_manifest": r"D:\Stacked-Object Grasping\assets\scenes\generated_main_v1\manifest.json",
                        "scenes": [
                            {
                                "path": r"D:\Stacked-Object Grasping\assets\scenes\generated_main_v1\scene_0001.xml",
                                "keep": "plain text",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            changed = normalize_json_project_paths(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(changed, 2)
        self.assertEqual(payload["source_manifest"], "assets/scenes/generated_main_v1/manifest.json")
        self.assertEqual(payload["scenes"][0]["path"], "assets/scenes/generated_main_v1/scene_0001.xml")
        self.assertEqual(payload["scenes"][0]["keep"], "plain text")


if __name__ == "__main__":
    unittest.main()
