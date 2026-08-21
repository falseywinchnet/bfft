"""Invariants for factorized local continuous-scale edge families."""

import unittest

import numpy as np

from .continuous_scale_edge_family_transport_2d import (
    _contract_sparse_generator_box,
    continuous_scale_edge_family_transport_state_2d,
)


class ContinuousScaleEdgeFamilyTransport2DTests(unittest.TestCase):
    def test_local_family_recomposes_and_remains_feasible(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.4 + 0.11 * np.sin(0.7 * xx) + 0.04 * np.cos(0.5 * yy)
        state = continuous_scale_edge_family_transport_state_2d(image)
        self.assertLess(state["observation_recomposition_error"], 2e-15)
        self.assertLess(state["full_lineage_recomposition_error"], 3e-13)
        self.assertTrue(
            state["contraction"]["feasible_outer_component"])
        self.assertGreater(state["generator"].shape[1], image.size)
        self.assertEqual(
            set(state["branches"]),
            {"identity_lineage", "positive_push_lineage"},
        )

    def test_factorized_evolved_flux_reexpression_is_exact(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.35 + 0.12 * np.sin(0.8 * xx + 0.2 * yy)
        state = continuous_scale_edge_family_transport_state_2d(image)
        self.assertLess(
            state["evolved_edge_response_flux"][
                "reconstruction_maximum_error"],
            3e-13,
        )
        self.assertLess(
            state["evolved_zero_response_flux"][
                "reconstruction_maximum_error"],
            3e-13,
        )

    def test_transfer_enclosure_contains_sampled_local_members(self):
        rng = np.random.default_rng(772)
        yy, xx = np.mgrid[:8, :8]
        image = 0.45 + 0.09 * np.sin(0.6 * xx - 0.3 * yy)
        state = continuous_scale_edge_family_transport_state_2d(image)
        lower = state["coefficient_lower"]
        upper = state["coefficient_upper"]
        for _ in range(3):
            coefficient = lower + rng.random(lower.size) * (upper - lower)
            transfer = np.asarray(
                state["generator"] @ coefficient).reshape(image.shape)
            self.assertTrue(np.all(
                transfer >= state["transfer_enclosure_lower"] - 3e-15))
            self.assertTrue(np.all(
                transfer <= state["transfer_enclosure_upper"] + 3e-15))

    def test_sparse_contractor_handles_many_local_variables(self):
        from scipy import sparse

        # Three variables share two rows; the feasible box has positive width.
        generator = sparse.csc_matrix(np.asarray((
            (1.0, -0.5, 0.0),
            (0.0, 0.5, 1.0),
        )))
        lower, upper, diagnostic = _contract_sparse_generator_box(
            generator,
            np.asarray((-0.1, 0.2)),
            np.asarray((0.8, 1.1)),
        )
        self.assertTrue(diagnostic["feasible_outer_component"])
        self.assertTrue(np.all(lower >= 0.0))
        self.assertTrue(np.all(upper <= 1.0))
        self.assertTrue(np.all(lower <= upper))

    def test_constant_scene_is_stable(self):
        image = np.full((8, 8), 0.37)
        state = continuous_scale_edge_family_transport_state_2d(image)
        self.assertLess(state["observation_recomposition_error"], 2e-15)
        self.assertLess(state["full_lineage_recomposition_error"], 3e-13)
        self.assertTrue(
            state["contraction"]["feasible_outer_component"])


if __name__ == "__main__":
    unittest.main()
