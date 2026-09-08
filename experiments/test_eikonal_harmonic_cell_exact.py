"""Exact invariant tests for Riemannian harmonic cell interpolation."""

from __future__ import annotations

import unittest

import sympy as sp

from experiments.eikonal_harmonic_cell_exact import (
    exact_fidelity_sweep,
    exact_harmonic_cell_weights,
    exact_metric_flip_certificate,
    exact_round_trip_and_stability_certificate,
    exact_two_dimensional_certificate,
)


class HarmonicCellExactTests(unittest.TestCase):
    def setUp(self) -> None:
        metric = sp.Matrix(((2, 1), (1, 1)))
        spacing = sp.diag(1, sp.Rational(17, 10))
        self.diffusion = spacing.inv() * metric.inv() * spacing.inv().T

    def test_positive_partition_and_affine_precision(self) -> None:
        weights = exact_harmonic_cell_weights(self.diffusion, side=4)
        vertex_x = (0, 4, 4, 0)
        vertex_y = (0, 0, 4, 4)
        for (x, y), row in weights.items():
            self.assertEqual(sum(row), 1)
            self.assertTrue(all(value >= 0 for value in row))
            self.assertEqual(sum(w * value for w, value in zip(row, vertex_x)), x)
            self.assertEqual(sum(w * value for w, value in zip(row, vertex_y)), y)

    def test_two_dimensional_certificate(self) -> None:
        result = exact_two_dimensional_certificate()
        self.assertEqual(
            result["mse"],
            "129920509520847408108232460083/182664799049274102636914110959360",
        )
        self.assertFalse(result["harmonic_beats_p1"])
        self.assertEqual(result["maximum_sample_range_overshoot"], "0")
        self.assertEqual(result["strict_unobserved_extrema_on_common_lattice"], 0)
        self.assertTrue(result["cell_energy"]["harmonic_strictly_minimal_against_both"])
        self.assertTrue(all(result["certificates"].values()))

    def test_exact_fidelity_sweep(self) -> None:
        result = exact_fidelity_sweep()
        self.assertEqual(
            result["summary"],
            {"harmonic_wins": 3, "p1_wins": 1, "ties": 4},
        )
        self.assertEqual(
            result["cases"]["metric_harmonic_quadratic"]["winner"],
            "harmonic",
        )
        self.assertEqual(result["cases"]["oblique_step"]["winner"], "p1")

    def test_round_trip_and_lanczos_norms(self) -> None:
        result = exact_round_trip_and_stability_certificate()
        self.assertEqual(
            result["induced_l_infinity_norms"],
            {
                "positive_harmonic_or_p1": "1",
                "normalized_half_shift_lanczos2": "5/4",
                "normalized_half_shift_lanczos3": "71/46",
            },
        )
        self.assertEqual(
            result["two_dimensional_quarter_shift"]["summary"],
            {
                "harmonic_higher_retention": 2,
                "p1_higher_retention": 1,
                "ties": 4,
            },
        )
        self.assertEqual(
            result["one_dimensional_step_edge"]["harmonic_equals_p1_mse"], "0"
        )
        self.assertEqual(
            result["one_dimensional_step_edge"]["normalized_lanczos2_mse"],
            "1/256",
        )

    def test_metric_flip_is_removed(self) -> None:
        result = exact_metric_flip_certificate()
        self.assertFalse(
            result["delaunay_p1_at_query_one_quarter"]["continuous_in_metric"]
        )
        self.assertTrue(
            result["harmonic_metric_continuity"]["continuous_at_zero"]
        )
        self.assertEqual(
            result["harmonic_cell_top_right_coordinate_at_query_one_quarter"]["0"],
            "1/16",
        )


if __name__ == "__main__":
    unittest.main()
