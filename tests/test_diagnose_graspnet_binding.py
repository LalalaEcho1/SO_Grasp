from __future__ import annotations

import unittest

import numpy as np

from tests import conftest  # noqa: F401
from scripts.diagnose_graspnet_binding import (
    aggregate_diagnostics,
    aggregate_sweep_cells,
    frame_binding_diagnostics,
    needed_label_radius,
    sweep_binding_cells,
)
from stacked_grasping.gripper.external_graspnet_data import AnnotationObject, RealSenseFrame


def _frame() -> RealSenseFrame:
    label = np.zeros((6, 6), dtype=np.uint8)
    label[3:5, 3:5] = 2
    return RealSenseFrame(
        frame="0000",
        color=np.zeros((6, 6, 3), dtype=np.uint8),
        depth_raw=np.full((6, 6), 1000, dtype=np.uint16),
        label=label,
        intrinsic_matrix=np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]),
    )


def _record(translation: list[float]) -> dict[str, object]:
    return {
        "score": 0.5,
        "width": 0.06,
        "height": 0.02,
        "depth": 0.03,
        "rotation_matrix": np.eye(3).tolist(),
        "translation": translation,
        "object_id": -1,
    }


class DiagnoseGraspNetBindingTests(unittest.TestCase):
    def test_needed_label_radius_finds_nearest_positive_pixel(self):
        label = np.zeros((6, 6), dtype=np.uint8)
        label[3:5, 3:5] = 2

        self.assertEqual(needed_label_radius(label, 3, 3, 5), 0)
        self.assertEqual(needed_label_radius(label, 1, 1, 5), 2)
        self.assertIsNone(needed_label_radius(label, 0, 0, 2))

    def test_sweep_cells_show_recovery_with_larger_radius_and_tolerance(self):
        frame = _frame()
        annotations = [AnnotationObject(object_id=1, label_id=2, name="pear.ply", position=np.zeros(3))]
        records = [
            _record([0.02, 0.02, 1.0]),  # projects to (3, 3): on the label
            _record([0.0, 0.0, 1.0]),  # projects to (1, 1): background, needs radius 2
            _record([0.028, 0.028, 1.4]),  # projects to (3, 3) but 0.4 m off in depth
        ]

        cells = sweep_binding_cells(
            records,
            frame,
            annotations,
            pixel_radii=(0, 2),
            depth_tolerances=(0.05, None),
        )

        by_key = {(cell["pixel_radius"], cell["depth_tolerance_m"]): cell for cell in cells}
        self.assertEqual(by_key[(0, 0.05)]["bound_count"], 1)
        self.assertEqual(by_key[(0, None)]["bound_count"], 2)
        self.assertEqual(by_key[(2, 0.05)]["bound_count"], 2)
        self.assertEqual(by_key[(2, None)]["bound_count"], 3)
        self.assertEqual(by_key[(0, 0.05)]["status_counts"].get("background"), 1)
        self.assertEqual(by_key[(0, 0.05)]["status_counts"].get("depth-mismatch"), 1)

    def test_frame_diagnostics_report_needed_radius_and_depth_errors(self):
        frame = _frame()
        annotations = [AnnotationObject(object_id=1, label_id=2, name="pear.ply", position=np.zeros(3))]
        records = [
            _record([0.0, 0.0, 1.0]),  # background at radius 1, needs 2
            _record([0.02, 0.02, 1.4]),  # depth mismatch of ~0.4 m
        ]

        diagnostics = frame_binding_diagnostics(
            records,
            frame,
            annotations,
            base_pixel_radius=1,
            base_depth_tolerance_m=0.05,
            needed_radius_cap=5,
        )

        self.assertEqual(diagnostics["bound_count"], 0)
        self.assertEqual(diagnostics["status_counts"].get("background"), 1)
        self.assertEqual(diagnostics["status_counts"].get("depth-mismatch"), 1)
        self.assertEqual(diagnostics["background_needed_radius_counts"], {"2": 1})
        self.assertEqual(len(diagnostics["depth_mismatch_errors_m"]), 1)
        self.assertAlmostEqual(diagnostics["depth_mismatch_errors_m"][0], 0.4, places=6)

    def test_aggregate_diagnostics_and_sweep_cells_merge_frames(self):
        frame_diagnostics = [
            {
                "frame": "0000",
                "total_candidates": 2,
                "bound_count": 1,
                "status_counts": {"bound": 1, "background": 1},
                "background_needed_radius_counts": {"2": 1},
                "depth_mismatch_errors_m": [],
            },
            {
                "frame": "0001",
                "total_candidates": 2,
                "bound_count": 0,
                "status_counts": {"background": 1, "depth-mismatch": 1},
                "background_needed_radius_counts": {"2": 1},
                "depth_mismatch_errors_m": [0.2],
            },
        ]

        aggregate = aggregate_diagnostics(frame_diagnostics)
        self.assertEqual(aggregate["total_candidates"], 4)
        self.assertEqual(aggregate["bound_count"], 1)
        self.assertEqual(aggregate["status_totals"], {"background": 2, "bound": 1, "depth-mismatch": 1})
        self.assertEqual(aggregate["background_needed_radius_totals"], {"2": 2})
        self.assertEqual(aggregate["depth_mismatch_count"], 1)
        self.assertAlmostEqual(aggregate["depth_mismatch_error_quantiles_m"]["p50"], 0.2)

        merged = aggregate_sweep_cells(
            [
                [{"pixel_radius": 2, "depth_tolerance_m": 0.05, "total_candidates": 2, "bound_count": 1, "status_counts": {"bound": 1, "background": 1}}],
                [{"pixel_radius": 2, "depth_tolerance_m": 0.05, "total_candidates": 2, "bound_count": 2, "status_counts": {"bound": 2}}],
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["total_candidates"], 4)
        self.assertEqual(merged[0]["bound_count"], 3)
        self.assertAlmostEqual(merged[0]["bound_ratio"], 0.75)
        self.assertEqual(merged[0]["status_counts"], {"background": 1, "bound": 3})


if __name__ == "__main__":
    unittest.main()
