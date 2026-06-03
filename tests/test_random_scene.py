from __future__ import annotations

import unittest

from tests import conftest  # noqa: F401
from scripts.generate_random_ycb_scenes import ValidationMetrics, is_valid_scene_metrics
from stacked_grasping.assets.random_scene import footprint_half_extents, sample_random_layout
from stacked_grasping.assets.ycb_starter import STARTER_OBJECTS


class RandomSceneLayoutTests(unittest.TestCase):
    def test_layout_is_reproducible_for_same_seed(self):
        first = sample_random_layout(STARTER_OBJECTS, seed=11, object_count=5)
        second = sample_random_layout(STARTER_OBJECTS, seed=11, object_count=5)

        self.assertEqual(first, second)

    def test_layout_respects_requested_object_count(self):
        layout = sample_random_layout(STARTER_OBJECTS, seed=3, object_count=4)

        self.assertEqual(len(layout), 4)

    def test_support_references_only_previously_placed_objects(self):
        layout = sample_random_layout(STARTER_OBJECTS, seed=21, object_count=8, stack_probability=1.0)
        placed = set()

        for spec in layout:
            if spec.support is not None:
                self.assertIn(spec.support, placed)
            placed.add(spec.name)

    def test_layout_uses_only_box_objects_as_stack_supports(self):
        layout = sample_random_layout(STARTER_OBJECTS, seed=1, object_count=8, stack_probability=1.0)
        by_name = {spec.name: spec for spec in layout}

        for spec in layout:
            if spec.support is not None:
                self.assertEqual(by_name[spec.support].geom_type, "box")

    def test_stacked_offsets_stay_inside_support_footprint(self):
        layout = sample_random_layout(STARTER_OBJECTS, seed=1, object_count=8, stack_probability=1.0)
        by_name = {spec.name: spec for spec in layout}

        for spec in layout:
            if spec.support is None:
                continue
            support = by_name[spec.support]
            support_half_x, support_half_y = footprint_half_extents(support)
            dx = abs(spec.pos[0] - support.pos[0])
            dy = abs(spec.pos[1] - support.pos[1])

            self.assertLessEqual(dx, support_half_x * 0.75)
            self.assertLessEqual(dy, support_half_y * 0.75)

    def test_layout_limits_stack_depth(self):
        layout = sample_random_layout(
            STARTER_OBJECTS,
            seed=1,
            object_count=8,
            stack_probability=1.0,
            max_stack_depth=2,
        )
        by_name = {spec.name: spec for spec in layout}

        for spec in layout:
            depth = 1
            current = spec
            while current.support is not None:
                depth += 1
                current = by_name[current.support]
            self.assertLessEqual(depth, 2)

    def test_scene_validation_requires_contacts_and_relation_edges(self):
        no_contacts = ValidationMetrics(object_count=6, contact_pairs=0, visible_edges=4)
        no_edges = ValidationMetrics(object_count=6, contact_pairs=2, visible_edges=0)
        enough = ValidationMetrics(object_count=6, contact_pairs=2, visible_edges=4)

        self.assertFalse(is_valid_scene_metrics(no_contacts, min_contacts=1, min_visible_edges=3))
        self.assertFalse(is_valid_scene_metrics(no_edges, min_contacts=1, min_visible_edges=3))
        self.assertTrue(is_valid_scene_metrics(enough, min_contacts=1, min_visible_edges=3))

    def test_scene_validation_rejects_out_of_bounds_and_below_table_objects(self):
        out_of_bounds = ValidationMetrics(
            object_count=6,
            contact_pairs=2,
            visible_edges=4,
            out_of_bounds_objects=1,
        )
        below_table = ValidationMetrics(
            object_count=6,
            contact_pairs=2,
            visible_edges=4,
            below_table_objects=1,
        )
        natural = ValidationMetrics(
            object_count=6,
            contact_pairs=2,
            visible_edges=4,
            out_of_bounds_objects=0,
            below_table_objects=0,
        )

        self.assertFalse(is_valid_scene_metrics(out_of_bounds, min_contacts=1, min_visible_edges=3))
        self.assertFalse(is_valid_scene_metrics(below_table, min_contacts=1, min_visible_edges=3))
        self.assertTrue(is_valid_scene_metrics(natural, min_contacts=1, min_visible_edges=3))

    def test_scene_validation_rejects_too_tall_stacks(self):
        too_tall = ValidationMetrics(
            object_count=6,
            contact_pairs=3,
            visible_edges=8,
            max_top_z=0.91,
        )
        natural = ValidationMetrics(
            object_count=6,
            contact_pairs=3,
            visible_edges=8,
            max_top_z=0.72,
        )

        self.assertFalse(
            is_valid_scene_metrics(
                too_tall,
                min_contacts=2,
                min_visible_edges=5,
                max_object_top_z=0.82,
            )
        )
        self.assertTrue(
            is_valid_scene_metrics(
                natural,
                min_contacts=2,
                min_visible_edges=5,
                max_object_top_z=0.82,
            )
        )


if __name__ == "__main__":
    unittest.main()
