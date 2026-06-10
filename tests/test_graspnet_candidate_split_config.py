from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests import conftest  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "graspnet_candidate_split.json"


class GraspNetCandidateSplitConfigTests(unittest.TestCase):
    def test_candidate_split_config_defines_expected_groups_and_scenes(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

        groups = {group["group_id"]: group for group in config["groups"]}
        self.assertEqual(set(groups), {"A", "B", "C"})
        self.assertEqual([scene["scene"] for scene in groups["A"]["scenes"]], ["scene_0009", "scene_0011"])
        self.assertEqual([scene["scene"] for scene in groups["B"]["scenes"]], ["scene_0015", "scene_0017"])
        self.assertEqual([scene["scene"] for scene in groups["C"]["scenes"]], ["scene_0007"])

    def test_candidate_split_config_keeps_scene_level_splits_and_valid_frames(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        selected_count = 0
        seen_scenes = set()
        for group in config["groups"]:
            for scene in group["scenes"]:
                scene_name = scene["scene"]
                self.assertNotIn(scene_name, seen_scenes)
                seen_scenes.add(scene_name)
                preview_frames = set(scene["preview_frames"])
                selected_frames = scene["selected_frames"]
                self.assertTrue(selected_frames)
                self.assertTrue(set(selected_frames).issubset(preview_frames))
                selected_count += len(selected_frames)
                self.assertIn(scene["split"], {"validation", "final_test", "supplementary_test"})

        self.assertEqual(selected_count, 15)
        self.assertEqual(config["split_policy"], "scene_level")


if __name__ == "__main__":
    unittest.main()
