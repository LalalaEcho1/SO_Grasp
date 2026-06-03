from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import conftest  # noqa: F401
from stacked_grasping.gripper.candidate_io import (
    load_graspnet_candidates,
    load_graspnet_records,
)


class GraspCandidateIOTests(unittest.TestCase):
    def test_loads_graspnet_candidates_from_json_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scene_grasps.json"
            path.write_text(
                json.dumps(
                    {
                        "grasps": [
                            {
                                "score": 0.91,
                                "width": 0.052,
                                "rotation_matrix": np.eye(3).tolist(),
                                "translation": [0.10, -0.20, 0.54],
                                "object_id": 5,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            candidates = load_graspnet_candidates(path, object_id_to_name={5: "obj_005"})

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.object_name, "obj_005")
        self.assertEqual(candidate.generator, "graspnet")
        self.assertEqual(candidate.object_id, 5)
        self.assertEqual(candidate.score, 0.91)
        self.assertEqual(candidate.required_opening, 0.052)
        np.testing.assert_allclose(candidate.position, np.array([0.10, -0.20, 0.54]))
        np.testing.assert_allclose(candidate.approach_direction, np.array([1.0, 0.0, 0.0]))
        self.assertEqual(candidate.to_dict()["object_id"], 5)

    def test_loads_graspnet_records_from_official_npy_array_layout(self):
        record = np.zeros((1, 17), dtype=float)
        record[0, 0] = 0.87
        record[0, 1] = 0.045
        record[0, 2] = 0.02
        record[0, 3] = 0.04
        record[0, 4:13] = np.eye(3).reshape(-1)
        record[0, 13:16] = [0.11, -0.21, 0.55]
        record[0, 16] = 7

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "0000.npy"
            np.save(path, record)
            records = load_graspnet_records(path, object_id_to_name={7: "obj_007"})

        self.assertEqual(
            records,
            [
                {
                    "score": 0.87,
                    "width": 0.045,
                    "height": 0.02,
                    "depth": 0.04,
                    "rotation_matrix": np.eye(3).tolist(),
                    "translation": [0.11, -0.21, 0.55],
                    "object_id": 7,
                    "object_name": "obj_007",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
