"""Invariants for characteristic branch-probability transport."""

from __future__ import annotations

import unittest

import numpy as np

from .branch_posterior_transport_2d import (
    _backtransport_branch_action,
    _branch_energy_score,
    denoise_causal_branch_action_2d,
    denoise_branch_posterior_transport_2d,
)


class BranchPosteriorTransport2DTests(unittest.TestCase):
    def test_branch_action_pullback_follows_exact_affine_incidence(self):
        action = np.array([[[1.0], [3.0]], [[5.0], [7.0]]])
        conductance = np.ones_like(action)
        identity = np.array([
            [[[1, 2]], [[0, 3]]],
            [[[0, 3]], [[1, 2]]],
        ])
        coefficient = np.full(identity.shape, 0.5)
        transported, diagnostic = _backtransport_branch_action(
            action, conductance, identity, coefficient)
        np.testing.assert_allclose(
            transported[..., 0], np.array([[4.0, 4.0], [4.0, 4.0]]))
        self.assertEqual(diagnostic["represented_chart_fraction"], 1.0)

    def test_exact_point_witness_has_zero_energy(self):
        probability = np.eye(3)
        prediction = np.array([
            [0.1, 0.5, 0.9],
            [0.1, 0.5, 0.9],
            [0.1, 0.5, 0.9],
        ])
        point_crps = np.abs(prediction - np.array([[0.1], [0.5], [0.9]]))
        action = _branch_energy_score(
            probability, prediction, point_crps, np.zeros(3))
        self.assertEqual(action, 0.0)

    def test_constant_is_exact_and_reaches_equilibrium(self):
        field = np.full((9, 11), 0.37)
        estimate, diagnostic = denoise_branch_posterior_transport_2d(
            field, angular_count=4, quantile_count=8, maximum_transports=4)
        np.testing.assert_allclose(estimate, field, atol=2e-15, rtol=0.0)
        self.assertFalse(diagnostic["branch_transport_ceiling_hit"])

    def test_causal_branch_action_reproduces_constant(self):
        field = np.full((9, 11), 0.37)
        estimate, diagnostic = denoise_causal_branch_action_2d(
            field, angular_count=4)
        np.testing.assert_allclose(estimate, field, atol=2e-15, rtol=0.0)
        self.assertEqual(diagnostic["maximum_target_self_coefficient"], 0.0)


if __name__ == "__main__":
    unittest.main()
