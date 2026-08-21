"""Invariants for cavity-certified residual erosion and re-entry."""

import unittest

import numpy as np

from .continual_eikonal_noise_transport_2d import (
    _continual_flux_laplacian,
    continual_transport_metric,
)
from .residual_erosion_transport_2d import (
    _cavity_residual_relation,
    denoise_cavity_residual_erosion_2d,
)


class ResidualErosionTransport2DTests(unittest.TestCase):
    def test_cavity_relation_excludes_target_residual_from_its_authority(self):
        yy, xx = np.mgrid[:8, :8]
        state = 0.3 + 0.1 * np.sin(0.7 * xx) + 0.05 * yy
        residual = 0.03 * np.cos(0.9 * xx + 0.4 * yy)
        metric = continual_transport_metric(state, residual * residual)
        laplacian, _transport, stencil = _continual_flux_laplacian(
            metric, np.ones_like(state))
        _first, first = _cavity_residual_relation(
            state, residual, laplacian, float(stencil["maximum_degree"]))
        changed = residual.copy()
        changed[4, 4] += 100.0
        _second, second = _cavity_residual_relation(
            state, changed, laplacian, float(stencil["maximum_degree"]))
        for key in (
            "cross_moment", "curvature_moment", "residual_moment",
            "explained_action", "schur_innovation",
        ):
            self.assertEqual(first[key][4, 4], second[key][4, 4])

    def test_every_accepted_continuation_strictly_contracts_action(self):
        rng = np.random.default_rng(331)
        image = rng.uniform(0.0, 1.0, size=(8, 8))
        estimate, diagnostic = denoise_cavity_residual_erosion_2d(
            image, continuation_guard=8)
        for record in diagnostic["continuations"]:
            if record["accepted"]:
                self.assertLess(
                    record["residual_action_after"],
                    record["residual_action_before"],
                )
        residual = image - estimate
        self.assertLess(diagnostic["observation_graph_maximum_error"], 1e-15)
        self.assertAlmostEqual(
            diagnostic["final_residual_action"],
            float(np.mean(residual * residual)),
            places=15,
        )

    def test_constant_initial_fixed_point_is_unchanged(self):
        image = np.full((8, 8), 0.37)
        estimate, diagnostic = denoise_cavity_residual_erosion_2d(
            image, initial_state=image)
        np.testing.assert_array_equal(estimate, image)
        self.assertEqual(diagnostic["accepted_continuations"], 0)
        self.assertFalse(diagnostic["continuation_guard_hit"])


if __name__ == "__main__":
    unittest.main()
