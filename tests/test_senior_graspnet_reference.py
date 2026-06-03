from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.senior_graspnet import (
    SeniorGraspNetPaths,
    load_reference_grasp_candidates,
    summarize_reference_grasp_file,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


class SeniorGraspNetReferenceTests(unittest.TestCase):
    def test_validates_reference_tree_and_loads_sample_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _touch(root / "graspnet-baseline-main" / "demo.py")
            _touch(root / "graspnetAPI-master" / "graspnetAPI" / "__init__.py")
            _touch(root / "checkpoints" / "checkpoint-rs.tar")
            _touch(root / "checkpoints" / "checkpoint-kn.tar")
            _touch(root / "mujoco" / "mujoco_sim_final.py")
            _touch(root / "utils" / "utils.py")
            _touch(root / "simulation" / "dataset" / "franka" / "panda.xml")
            _touch(root / "simulation" / "dataset" / "coacd_models" / "003" / "part_0.obj")
            _touch(root / "simulation" / "dataset" / "offical_models" / "003" / "textured.obj")

            sample = np.zeros((1, 17), dtype=float)
            sample[0, 0] = 0.88
            sample[0, 1] = 0.041
            sample[0, 2] = 0.02
            sample[0, 3] = 0.04
            sample[0, 4:13] = np.eye(3).reshape(-1)
            sample[0, 13:16] = [0.1, -0.2, 0.5]
            sample[0, 16] = 3
            sample_path = root / "simulation" / "dataset" / "grasp_poses" / "train" / "scene_0000" / "realsense" / "0000.npy"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(sample_path, sample)

            paths = SeniorGraspNetPaths(root)
            validation = paths.validate(required_object_ids=("003",))
            candidates = load_reference_grasp_candidates(
                split="train",
                scene_id=0,
                view_id=0,
                root=root,
                object_id_to_name={3: "003_cracker_box"},
            )
            summary = summarize_reference_grasp_file(paths.grasp_pose_path("train", 0, 0))

        self.assertTrue(validation.ok)
        self.assertEqual(validation.missing_coacd_models, [])
        self.assertEqual(validation.missing_official_models, [])
        self.assertEqual(candidates[0].object_name, "003_cracker_box")
        self.assertEqual(candidates[0].object_id, 3)
        self.assertEqual(candidates[0].generator, "graspnet-reference")
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["object_ids"], [3])
        self.assertEqual(summary["score_max"], 0.88)


if __name__ == "__main__":
    unittest.main()
