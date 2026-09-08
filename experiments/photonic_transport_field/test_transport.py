from __future__ import annotations

import unittest

import numpy as np

from .scenes import facing_patch_grids, two_patch_color_bleed
from .budgeted import BudgetedTransportGeometry, solve_budgeted_transport
from .transport import (
    compile_dense_centroid_field,
    compile_hierarchical_field,
    dense_fixed_point,
    solve_transport,
)


class PhotonicTransportFieldTests(unittest.TestCase):
    def test_centroid_form_factor_reciprocity(self) -> None:
        patches, _ = facing_patch_grids(side=2, separation=4.0, extent=1.0)
        field = compile_dense_centroid_field(patches)
        operator = field.dense_matrix()
        area = np.array([patch.area for patch in patches])
        exchange = area[None, :] * operator
        self.assertTrue(np.allclose(exchange, exchange.T, atol=1.0e-15))

    def test_hierarchy_compresses_coherent_open_exchange(self) -> None:
        patches, _ = facing_patch_grids(side=8, separation=8.0, extent=2.0)
        dense = compile_dense_centroid_field(patches)
        hierarchical = compile_hierarchical_field(
            patches, admissibility=0.10
        )
        reference = dense.dense_matrix()
        approximation = hierarchical.dense_matrix()
        relative_error = np.linalg.norm(approximation - reference) / np.linalg.norm(reference)
        self.assertLess(relative_error, 0.03)
        self.assertLess(
            hierarchical.stored_coefficients,
            dense.stored_coefficients // 8,
        )
        self.assertGreater(
            int(hierarchical.diagnostics["coherent_pair_count"]), 0
        )

    def test_transport_scale_preserves_every_pair_ratio(self) -> None:
        patches, _ = facing_patch_grids(side=4, separation=6.0, extent=2.0)
        ordinary = compile_hierarchical_field(patches, admissibility=0.2)
        scaled = compile_hierarchical_field(
            patches, admissibility=0.2, transport_scale=0.73
        )
        self.assertTrue(np.allclose(
            scaled.dense_matrix(), 0.73 * ordinary.dense_matrix()
        ))
        self.assertTrue(np.allclose(
            scaled.outgoing_fraction, 0.73 * ordinary.outgoing_fraction
        ))

    def test_occlusion_uncertainty_forces_local_refinement(self) -> None:
        patches, blockers = facing_patch_grids(
            side=6,
            separation=6.0,
            extent=3.0,
            occluder_radius=0.55,
        )
        open_field = compile_hierarchical_field(
            patches, admissibility=0.35
        )
        blocked_field = compile_hierarchical_field(
            patches,
            occluders=blockers,
            admissibility=0.35,
        )
        dense_blocked = compile_dense_centroid_field(
            patches, occluders=blockers
        ).dense_matrix()
        hierarchical_blocked = blocked_field.dense_matrix()
        self.assertGreater(
            int(blocked_field.diagnostics["occlusion_refinements"]), 0
        )
        self.assertGreater(
            int(blocked_field.diagnostics["exact_visibility_tests"]),
            int(open_field.diagnostics["exact_visibility_tests"]),
        )
        self.assertGreater(
            int(blocked_field.diagnostics["occluded_leaf_pairs"]), 0
        )
        self.assertLess(
            float(np.sum(hierarchical_blocked)),
            float(np.sum(open_field.dense_matrix())),
        )
        # A certified-clear aggregate may approximate a visible coefficient,
        # but it must never bridge a centroid link the exact blocker test hides.
        self.assertEqual(
            int(np.count_nonzero((dense_blocked == 0.0) & (hierarchical_blocked != 0.0))),
            0,
        )

    def test_residual_march_matches_dense_fixed_point(self) -> None:
        patches, _ = facing_patch_grids(side=3, separation=3.0, extent=1.5)
        field = compile_hierarchical_field(patches, admissibility=0.2)
        emission = np.stack([patch.emission for patch in patches])
        albedo = np.stack([patch.albedo for patch in patches])
        marched = solve_transport(
            field,
            emission,
            albedo,
            relative_tolerance=1.0e-13,
            absolute_tolerance=1.0e-15,
        )
        reference = dense_fixed_point(field, emission, albedo)
        self.assertTrue(np.allclose(
            marched.outgoing, reference, rtol=2.0e-12, atol=2.0e-14
        ))
        self.assertLess(float(np.max(np.abs(marched.conservation_error))), 1.0e-12)

    def test_surface_albedo_filters_received_color(self) -> None:
        patches = two_patch_color_bleed()
        field = compile_dense_centroid_field(patches)
        emission = np.stack([patch.emission for patch in patches])
        albedo = np.stack([patch.albedo for patch in patches])
        result = solve_transport(field, emission, albedo)
        reflected = result.outgoing[1]
        self.assertGreater(reflected[0], 7.5 * reflected[1])
        self.assertGreater(reflected[1], 1.9 * reflected[2])
        self.assertTrue(np.all(result.incident[1] > 0.0))

    def test_zero_budget_lazy_discovery_matches_compiled_field(self) -> None:
        patches, blockers = facing_patch_grids(
            side=3,
            separation=4.0,
            extent=1.5,
            occluder_radius=0.25,
        )
        field = compile_hierarchical_field(
            patches, occluders=blockers, admissibility=0.1
        )
        emission = np.stack([patch.emission for patch in patches])
        albedo = np.stack([patch.albedo for patch in patches])
        reference = dense_fixed_point(field, emission, albedo)
        geometry = BudgetedTransportGeometry(
            patches, occluders=blockers, admissibility=0.1
        )
        result = solve_budgeted_transport(
            geometry,
            emission,
            albedo,
            outgoing_error_budget=0.0,
            frontier_tolerance=1.0e-14,
        )
        self.assertTrue(np.allclose(
            result.outgoing, reference, rtol=2.0e-13, atol=2.0e-15
        ))
        self.assertLessEqual(
            float(np.sum(np.abs(result.outgoing - reference))),
            result.outgoing_error_bound + 2.0e-15,
        )

    def test_diffusion_budget_collapses_late_geometry_work(self) -> None:
        patches, blockers = facing_patch_grids(
            side=3,
            separation=4.0,
            extent=1.5,
            occluder_radius=0.25,
        )
        field = compile_hierarchical_field(
            patches, occluders=blockers, admissibility=0.1
        )
        emission = np.stack([patch.emission for patch in patches])
        albedo = np.stack([patch.albedo for patch in patches])
        reference = dense_fixed_point(field, emission, albedo)
        geometry = BudgetedTransportGeometry(
            patches, occluders=blockers, admissibility=0.1
        )
        result = solve_budgeted_transport(
            geometry,
            emission,
            albedo,
            outgoing_error_budget=1.0e-3,
            tail_budget_fraction=0.0,
        )
        actual_error = float(np.sum(np.abs(result.outgoing - reference)))
        self.assertLessEqual(actual_error, result.outgoing_error_bound + 2.0e-15)
        self.assertLessEqual(result.outgoing_error_bound, 1.0e-3)
        visits = [
            int(record["cluster_pair_visits"])
            for record in result.depth_diagnostics
        ]
        self.assertGreater(max(visits[:-1]), visits[-1])
        self.assertEqual(int(result.depth_diagnostics[-1]["block_count"]), 0)
        self.assertEqual(visits[-1], 1)


if __name__ == "__main__":
    unittest.main()
