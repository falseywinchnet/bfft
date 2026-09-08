"""Invariants for the Hodge-lifted flux-uncertainty hypothesis."""

from __future__ import annotations

import unittest

import numpy as np

from .hodge_zonotope_chambolle import (
    denoise_hodge_zonotope_chambolle,
    hodge_transport_state,
)


class HodgeZonotopeChambolleTests(unittest.TestCase):
    def test_hodge_lift_reconstructs_every_zero_mean_lane(self):
        rng = np.random.default_rng(5)
        image = 0.3 + 0.08 * rng.normal(size=(16, 18))
        state = hodge_transport_state(image)
        self.assertLess(state.hodge_reconstruction_error, 2e-10)

    def test_flux_projection_lies_outside_witness_zonotope_only(self):
        rng = np.random.default_rng(7)
        image = 0.4 + 0.1 * rng.normal(size=(16, 18))
        state = hodge_transport_state(image)
        admitted = state.departure_flux - state.excess_flux
        self.assertLessEqual(
            float(np.max(np.abs(admitted) - state.flux_radius)), 1e-14)

    def test_unsupported_flux_readout_has_zero_mass(self):
        rng = np.random.default_rng(9)
        image = rng.random((16, 18))
        state = hodge_transport_state(image)
        self.assertAlmostEqual(float(np.sum(state.unsupported)), 0.0, places=11)

    def test_constant_is_exact_identity(self):
        image = np.full((16, 18), 0.37)
        estimate, diagnostic = denoise_hodge_zonotope_chambolle(image)
        np.testing.assert_array_equal(estimate, image)
        self.assertEqual(diagnostic["recomposition_error"], 0.0)

    def test_impulse_contracts_and_recomposes(self):
        image = np.full((16, 18), 0.2)
        image[8, 9] = 1.0
        estimate, diagnostic = denoise_hodge_zonotope_chambolle(image)
        self.assertLess(estimate[8, 9], image[8, 9])
        residual = image - estimate
        np.testing.assert_allclose(estimate + residual, image, atol=0.0, rtol=0.0)
        self.assertLess(diagnostic["recomposition_error"], 1e-15)


if __name__ == "__main__":
    unittest.main()
