from __future__ import annotations

import unittest

import numpy as np

from tests import conftest  # noqa: F401
from scripts.sweep_pointcloud_feasibility_grid import (
    aggregate_cell_rows,
    build_grid_cells,
    evaluate_frame_cell,
)
from stacked_grasping.gripper.external_graspnet_data import (
    AnnotationObject,
    RealSenseFrame,
    depth_to_point_cloud,
)
from stacked_grasping.gripper.graspnet_binding import bind_graspnet_records_to_frame_labels


def _frame() -> RealSenseFrame:
    return RealSenseFrame(
        frame="0000",
        color=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_raw=np.full((4, 4), 1000, dtype=np.uint16),
        label=np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 2, 0],
                [0, 2, 2, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        intrinsic_matrix=np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]),
    )


def _record(width: float, score: float = 0.8) -> dict[str, object]:
    return {
        "score": score,
        "width": width,
        "height": 0.02,
        "depth": 0.03,
        "rotation_matrix": np.eye(3).tolist(),
        "translation": [0.0, 0.0, 1.0],
        "object_id": -1,
    }


class SweepPointCloudFeasibilityGridTests(unittest.TestCase):
    def test_build_grid_cells_produces_cartesian_product(self):
        cells = build_grid_cells(
            clamp_modes=("off", "on"),
            collision_thresholds=(0.01, 0.02),
            empty_thresholds=(0.01,),
            max_opening=0.085,
        )

        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0]["cell_id"], "clamp-off_ct-0.01_et-0.01")
        self.assertFalse(cells[0]["config"].clamp_width_to_max_opening)
        self.assertTrue(cells[2]["config"].clamp_width_to_max_opening)
        self.assertAlmostEqual(cells[1]["config"].collision_threshold, 0.02)

    def test_clamp_cell_recovers_wide_candidate_and_reports_funnel(self):
        frame = _frame()
        annotations = [AnnotationObject(object_id=1, label_id=2, name="banana.ply", position=np.array([0.0, 0.0, 1.0]))]
        records = [_record(width=0.12), _record(width=0.04, score=0.6)]
        bindings = bind_graspnet_records_to_frame_labels(records, frame, annotations, pixel_radius=1, depth_tolerance_m=0.05)
        points, _ = depth_to_point_cloud(frame.depth_raw, frame.intrinsic_matrix, factor_depth=frame.factor_depth)
        cells = build_grid_cells(clamp_modes=("off", "on"), collision_thresholds=(0.01,), empty_thresholds=(0.01,))

        shared = dict(
            frame=frame,
            annotations=annotations,
            records=records,
            bindings=bindings,
            points=points,
            risk_threshold=0.45,
            min_points_per_object=1,
            min_half_extent=0.001,
            object_padding=0.0,
            min_boundary_pixels=1,
        )
        row_off = evaluate_frame_cell(cell_config=cells[0]["config"], **shared)
        row_on = evaluate_frame_cell(cell_config=cells[1]["config"], **shared)

        self.assertEqual(row_off["bound_count"], 2)
        self.assertEqual(row_off["pointcloud_feasible_candidate_count"], 1)
        self.assertEqual(row_off["clamp_recovered_candidate_count"], 0)
        self.assertEqual(row_off["pointcloud_reason_counts"].get("opening-too-small"), 1)

        self.assertEqual(row_on["pointcloud_feasible_candidate_count"], 2)
        self.assertEqual(row_on["clamp_recovered_candidate_count"], 1)
        self.assertNotIn("opening-too-small", row_on["pointcloud_reason_counts"])
        self.assertTrue(row_on["grasp_success"])

    def test_aggregate_cell_rows_reports_rates_and_reason_totals(self):
        aggregate = aggregate_cell_rows(
            [
                {
                    "pointcloud_feasible_candidate_count": 3,
                    "clamp_recovered_candidate_count": 2,
                    "grasp_success": True,
                    "failure_reason": None,
                    "pointcloud_reason_counts": {"feasible": 3, "pointcloud-collision": 5},
                },
                {
                    "pointcloud_feasible_candidate_count": 0,
                    "clamp_recovered_candidate_count": 0,
                    "grasp_success": False,
                    "failure_reason": "gripper-infeasible",
                    "pointcloud_reason_counts": {"pointcloud-collision": 4},
                },
            ]
        )

        self.assertEqual(aggregate["frame_count"], 2)
        self.assertAlmostEqual(aggregate["mean_pointcloud_feasible_candidate_count"], 1.5)
        self.assertEqual(aggregate["pointcloud_feasible_frame_count"], 1)
        self.assertEqual(aggregate["clamp_recovered_candidate_count"], 2)
        self.assertAlmostEqual(aggregate["success_rate"], 0.5)
        self.assertEqual(aggregate["failure_reason_counts"], {"gripper-infeasible": 1})
        self.assertEqual(aggregate["pointcloud_reason_totals"], {"feasible": 3, "pointcloud-collision": 9})


if __name__ == "__main__":
    unittest.main()
