"""Invariants for the set-valued conservative Selling-edge flux state."""

import unittest

import numpy as np
from scipy import sparse

from .continual_eikonal_noise_transport_2d import (
    _continual_flux_laplacian,
    continual_transport_metric,
)
from .zonotopic_edge_flux_2d import (
    _contract_linear_zonotope_box,
    _selling_edge_flux_zonotope,
    zonotopic_edge_flux_state_2d,
)


class ZonotopicEdgeFlux2DTests(unittest.TestCase):
    def test_selling_flux_generators_reconstruct_complete_proposal(self):
        yy, xx = np.mgrid[:8, :8]
        posterior = 0.3 + 0.1 * np.sin(0.7 * xx) + 0.04 * yy
        residual = 0.03 * np.cos(0.9 * xx + 0.4 * yy)
        proposal = 0.02 + 0.05 * np.sin(0.6 * xx - 0.3 * yy)
        metric = continual_transport_metric(posterior, residual * residual)
        laplacian, _markov, _stencil = _continual_flux_laplacian(
            metric, np.ones_like(posterior))
        generator, diagnostic = _selling_edge_flux_zonotope(
            laplacian, proposal)
        reconstructed = np.asarray(
            generator @ np.ones(generator.shape[1])).reshape(proposal.shape)
        np.testing.assert_allclose(
            reconstructed, proposal, atol=2e-13, rtol=0.0)
        self.assertLess(
            diagnostic["edge_antisymmetry_column_sum_error"], 2e-15)
        self.assertEqual(diagnostic["connected_component_count"], 1)

    def test_linear_contractor_retains_intervals_not_probabilities(self):
        generator = sparse.eye(2, format="csc")
        lower, upper, diagnostic = _contract_linear_zonotope_box(
            generator,
            np.asarray((0.25, 0.50)),
            np.asarray((0.75, 0.90)),
        )
        np.testing.assert_allclose(lower, (0.25, 0.50), atol=2e-15)
        np.testing.assert_allclose(upper, (0.75, 0.90), atol=2e-15)
        self.assertTrue(diagnostic["feasible_outer_component"])
        self.assertGreater(diagnostic["mean_coefficient_width"], 0.0)

    def test_linear_contractor_conclusively_falsifies_empty_box(self):
        _lower, _upper, diagnostic = _contract_linear_zonotope_box(
            sparse.eye(2, format="csc"),
            np.asarray((2.0, 0.0)),
            np.asarray((3.0, 1.0)),
        )
        self.assertFalse(diagnostic["feasible_outer_component"])

    def test_mixture_midpoints_preserve_observation_exactly(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.4 + 0.14 * np.sin(0.8 * xx) + 0.06 * np.cos(0.5 * yy)
        state = zonotopic_edge_flux_state_2d(image)
        self.assertLess(state["observation_recomposition_error"], 2e-15)
        for error in state["midpoint_recomposition_errors"].values():
            self.assertLess(error, 2e-15)
        for component in state["components"]:
            self.assertLess(
                component["representation"][
                    "proposal_reconstruction_maximum_error"],
                5e-13,
            )
            self.assertTrue(component["zero_transfer_in_contracted_box"])
            self.assertTrue(
                component["contraction"]["feasible_outer_component"])

    def test_constant_scene_has_only_zero_transfer_proposals(self):
        image = np.full((8, 8), 0.37)
        state = zonotopic_edge_flux_state_2d(image)
        for component in state["components"]:
            self.assertLess(component["proposal_action"], 1e-27)
            self.assertTrue(
                component["contraction"]["feasible_outer_component"])


if __name__ == "__main__":
    unittest.main()
