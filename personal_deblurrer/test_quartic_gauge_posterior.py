from __future__ import annotations

import unittest

import numpy as np

from denoiser.run_2d_denoiser_battery import sources

from .full_quartic_transport import (
    _covariance_square_root,
    directional_quartic_dictionary,
    estimate_full_quartic_transport,
)
from .quartic_gauge_posterior import solve_quartic_gauge_posterior
from .spatial_transport import SpatialExposureField, SpatialReflectedExposureOperator


class QuarticGaugePosteriorTests(unittest.TestCase):
    def test_posterior_retains_both_gauges_and_avoids_unanchored_loss(
        self,
    ) -> None:
        truth = sources(48)["cameraman"]
        dictionary, _, _ = directional_quartic_dictionary(8)
        regimes = (
            (
                "strong_anchored",
                (((0, 1.0),),
                 ((0, 0.1), (1, 0.9)),
                 ((0, 0.1), (3, 0.9)),
                 ((0, 0.1), (10, 0.9))),
            ),
            (
                "opposed_unanchored",
                (((0, 0.25), (1, 0.75)),
                 ((0, 0.25), (5, 0.75)),
                 ((0, 0.25), (9, 0.75)),
                 ((0, 0.25), (13, 0.75))),
            ),
        )
        covariances = []
        for capture in range(4):
            angle = np.deg2rad(17.0 * capture)
            rotation = np.asarray((
                (np.cos(angle), -np.sin(angle)),
                (np.sin(angle), np.cos(angle)),
            ))
            covariances.append(
                rotation @ np.diag((1.2 + 0.2 * capture, 5.0 + capture))
                @ rotation.T)
        covariance_array = np.stack(covariances)
        for regime_name, specifications in regimes:
            observations = []
            for capture, specification in enumerate(specifications):
                factor = _covariance_square_root(covariance_array[capture])
                points = []
                weights = []
                for component, mass in specification:
                    standard_points, standard_weights = dictionary[component]
                    points.append(standard_points @ factor.T)
                    weights.append(mass * standard_weights)
                field = SpatialExposureField.from_barycentric_paths(
                    f"{regime_name}_{capture}",
                    np.zeros((*truth.shape, 2)),
                    np.concatenate(points),
                    np.concatenate(weights),
                    compact_global=True,
                )
                observations.append(
                    SpatialReflectedExposureOperator(field).forward(truth))
            estimate = estimate_full_quartic_transport(
                observations, covariance_array, maximum_frequency=0.14)
            solution = solve_quartic_gauge_posterior(
                observations,
                covariance_array,
                estimate=estimate,
                passes=8,
            )
            covariance_error = float(np.mean(
                (solution.covariance_solution.image - truth) ** 2))
            posterior_error = float(np.mean((solution.image - truth) ** 2))
            self.assertLessEqual(posterior_error, 1.001 * covariance_error)
            self.assertGreater(solution.quartic_posterior_mass, 0.0)
            self.assertLess(solution.quartic_posterior_mass, 1.0)
            self.assertGreaterEqual(float(np.min(solution.uncertainty)), 0.0)
            self.assertEqual(
                solution.diagnostics["selection_policy"],
                "both_gauges_reconstructed_and_retained_no_winner_branch",
            )


if __name__ == "__main__":
    unittest.main()
