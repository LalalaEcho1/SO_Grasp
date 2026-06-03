from __future__ import annotations

import unittest
from pathlib import Path

from tests import conftest  # noqa: F401
from stacked_grasping.utils.paths import (
    is_windows_absolute_path,
    resolve_project_path,
    to_project_relative,
)


class ProjectPathTests(unittest.TestCase):
    def test_resolve_project_path_accepts_project_relative_path(self):
        resolved = resolve_project_path("assets/scenes/generated_main_v1/scene_0001.xml")

        self.assertEqual(resolved, conftest.PROJECT_ROOT / "assets" / "scenes" / "generated_main_v1" / "scene_0001.xml")

    def test_resolve_project_path_recovers_legacy_windows_project_path(self):
        resolved = resolve_project_path(r"D:\Stacked-Object Grasping\assets\scenes\generated_main_v1\scene_0001.xml")

        self.assertEqual(resolved, conftest.PROJECT_ROOT / "assets" / "scenes" / "generated_main_v1" / "scene_0001.xml")

    def test_to_project_relative_outputs_posix_project_path(self):
        absolute = conftest.PROJECT_ROOT / "assets" / "scenes" / "generated_main_v1" / "scene_0001.xml"

        self.assertEqual(to_project_relative(absolute), "assets/scenes/generated_main_v1/scene_0001.xml")

    def test_to_project_relative_normalizes_legacy_windows_path(self):
        self.assertEqual(
            to_project_relative(r"D:\Stacked-Object Grasping\assets\scenes\generated_main_v1\scene_0001.xml"),
            "assets/scenes/generated_main_v1/scene_0001.xml",
        )

    def test_is_windows_absolute_path_detects_drive_prefix(self):
        self.assertTrue(is_windows_absolute_path(r"D:\Stacked-Object Grasping\assets\scene.xml"))
        self.assertFalse(is_windows_absolute_path("assets/scenes/scene.xml"))
        self.assertFalse(is_windows_absolute_path(str(Path("/tmp/scene.xml"))))


if __name__ == "__main__":
    unittest.main()
