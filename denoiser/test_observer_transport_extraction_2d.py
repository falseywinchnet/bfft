"""Invariants for observer-space transport extraction."""

from __future__ import annotations

import unittest

import numpy as np

from .causal_information_lineage_2d import causal_information_lineage_law_2d
from .observer_transport_extraction_2d import (
    denoise_observer_transport_extraction_2d,
    observer_chart_operators_2d,
    observer_transport_extraction_readout_2d,
)


class ObserverTransportExtraction2DTests(unittest.TestCase):
    def test_constant_is_exact(self):
        image = np.full((8, 8), 0.37)
        estimate, diagnostic = denoise_observer_transport_extraction_2d(
            image, angular_count=4, quantile_count=8)
        np.testing.assert_allclose(estimate, image, atol=2e-8, rtol=0.0)
        self.assertLess(
            diagnostic["extraction"]["maximum_chart_row_mass_error"], 2e-15)

    def test_chart_operators_are_affine_and_target_free(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.2 + 0.4 * xx / 7.0 + 0.1 * np.sin(yy)
        law, _ = causal_information_lineage_law_2d(
            image, angular_count=4, quantile_count=8)
        charts, ownership, diagnostic = observer_chart_operators_2d(law)
        for chart, operator in enumerate(charts):
            supported = ownership[..., chart].reshape(-1) > 0.0
            np.testing.assert_allclose(
                np.asarray(operator.sum(axis=1)).reshape(-1)[supported],
                1.0, atol=2e-15, rtol=0.0)
            np.testing.assert_allclose(
                operator.diagonal(), 0.0, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            np.sum(ownership, axis=-1), 1.0, atol=2e-15, rtol=0.0)
        self.assertEqual(diagnostic["maximum_target_self_coefficient"], 0.0)

    def test_positive_screened_solution_contracts_observer_residual(self):
        yy, xx = np.mgrid[:8, :8]
        image = np.clip(
            0.2 + 0.5 * (xx > 3) + 0.08 * np.sin(2.3 * yy + xx),
            0.0, 1.0)
        law, _ = causal_information_lineage_law_2d(
            image, angular_count=4, quantile_count=8)
        forms, diagnostic = observer_transport_extraction_readout_2d(
            image, law)
        self.assertLessEqual(diagnostic["objective_after"],
                             diagnostic["objective_before"] + 1e-10)
        self.assertLessEqual(diagnostic["observer_residual_contraction"], 1.0)
        self.assertLess(diagnostic["normal_equation_maximum_error"], 2e-7)
        recomposed = (
            forms["observer_transport_structure"]
            + forms["observer_adjoint_recomposition"])
        np.testing.assert_allclose(recomposed, image, atol=2e-7, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
