"""Invariant tests for blur/noise uncertainty transport and workbench state."""

from __future__ import annotations

import unittest

import numpy as np

from .kernels import disk_kernel, gaussian_kernel, identity_kernel, line_kernel
from .synthetic import degrade
from .test_exposure_transport_deblur import _fixture
from .uncertainty import (
    deblur_pair_posterior,
    estimate_noise_discrepancy,
    estimate_pair_posterior,
    pseudo_huber,
)
from .workbench import BlurSpec, DeblurSession, load_v3_skimage_source


def _catalog() -> list:
    return [
        identity_kernel(), gaussian_kernel(2.0), disk_kernel(3.0),
        line_kernel(9.0, 0.0), line_kernel(9.0, 45.0),
        line_kernel(9.0, 90.0), line_kernel(9.0, 135.0),
    ]


class UncertaintyTransportTests(unittest.TestCase):
    def test_pseudo_huber_is_quadratic_then_linear(self) -> None:
        values = pseudo_huber(np.asarray((0.001, 10.0)), 0.1)
        self.assertAlmostEqual(values[0], 0.001 ** 2 / 0.2, places=8)
        self.assertAlmostEqual(values[1], 9.9005, places=3)

    def test_ideal_pair_posterior_collapses_to_true_transport(self) -> None:
        truth = _fixture(64)
        candidates = _catalog()
        first = candidates[3]
        second = candidates[5]
        posterior = estimate_pair_posterior(
            degrade(truth, first), degrade(truth, second), candidates)
        self.assertEqual(posterior.best.first.name, first.name)
        self.assertEqual(posterior.best.second.name, second.name)
        self.assertGreater(posterior.best.probability, 0.999)
        self.assertLess(posterior.effective_hypotheses, 1.01)

    def test_uncertainty_expands_when_measurements_are_ambiguous(self) -> None:
        truth = _fixture(64)
        candidates = _catalog()
        first = candidates[3]
        second = candidates[5]
        clean = estimate_pair_posterior(
            degrade(truth, first), degrade(truth, second), candidates)
        noisy = estimate_pair_posterior(
            degrade(truth, first, gaussian_sigma=0.015, seed=1),
            degrade(truth, second, gaussian_sigma=0.015, seed=2),
            candidates,
            noise_sigma=0.015,
        )
        self.assertGreater(noisy.effective_hypotheses, clean.effective_hypotheses)
        self.assertLess(noisy.best.probability, clean.best.probability)

    def test_posterior_transport_returns_finite_credible_images(self) -> None:
        truth = _fixture(64)
        candidates = _catalog()
        first = degrade(truth, candidates[3], gaussian_sigma=0.002, seed=3)
        second = degrade(truth, candidates[5], gaussian_sigma=0.002, seed=4)
        posterior = estimate_pair_posterior(
            first, second, candidates, noise_sigma=0.002)
        result = deblur_pair_posterior(
            first, second, posterior, noise_sigma=0.002,
            maximum_branches=3, passes=5)
        self.assertEqual(result.image.shape, truth.shape)
        self.assertTrue(np.all(np.isfinite(result.standard_deviation)))
        self.assertTrue(np.all(result.lower <= result.upper + 1e-14))
        self.assertGreater(result.retained_probability, 0.0)

    def test_noise_discrepancy_recovers_white_noise_scale(self) -> None:
        rng = np.random.default_rng(12)
        prediction = np.full((192, 192), 0.5)
        observation = prediction + rng.normal(0.0, 0.01, prediction.shape)
        estimate = estimate_noise_discrepancy(observation, prediction)
        self.assertAlmostEqual(estimate.read_sigma, 0.01, delta=0.0007)
        self.assertLess(estimate.outlier_fraction, 0.001)

    def test_all_workbench_blur_specs_are_positive_mass(self) -> None:
        for kind in ("None", "Gaussian", "Disk", "Line", "Curve", "Random path"):
            kernel = BlurSpec(kind=kind, seed=4).kernel()
            self.assertGreaterEqual(float(np.min(kernel.psf)), 0.0)
            self.assertAlmostEqual(kernel.mass, 1.0, places=13)

    def test_headless_workbench_synthesizes_and_deblurs(self) -> None:
        session = DeblurSession()
        index = session.add_array(_fixture(64), "fixture")
        session.synthesize(
            index, BlurSpec(kind="Gaussian", sigma=2.0),
            read_noise_sigma=0.002, seed=2)
        result = session.deblur_known(index, passes=4)
        self.assertEqual(result.image.shape, (64, 64))
        self.assertIsNotNone(session.sources[index].result)
        self.assertIn("noise_discrepancy", session.sources[index].diagnostics)

    def test_v3_skimage_portfolio_is_source_data_only(self) -> None:
        try:
            coffee = load_v3_skimage_source("coffee")
            checker = load_v3_skimage_source("checkerboard")
        except RuntimeError as error:
            self.skipTest(str(error))
        self.assertEqual(coffee.ndim, 3)
        self.assertEqual(checker.ndim, 2)
        self.assertGreaterEqual(float(np.min(coffee)), 0.0)
        self.assertLessEqual(float(np.max(coffee)), 1.0)

    def test_workbench_abstains_on_common_blur_pair(self) -> None:
        session = DeblurSession()
        first = session.add_array(_fixture(64), "first")
        second = session.add_array(_fixture(64), "second")
        spec = BlurSpec(kind="Gaussian", sigma=2.5)
        session.synthesize(first, spec, read_noise_sigma=0.001, seed=3)
        session.synthesize(second, spec, read_noise_sigma=0.001, seed=4)
        posterior, result = session.deblur_pair_uncertain(
            first, second, noise_sigma=0.001, passes=3)
        self.assertTrue(posterior.common_blur_unidentifiable)
        self.assertEqual(
            result.diagnostics["decision"], "abstain_common_blur_gauge")
        self.assertEqual(len(result.branch_images), 0)

    def test_workbench_runs_unified_multicapture_posterior(self) -> None:
        session = DeblurSession()
        truth = _fixture(32)
        indices = []
        for capture, angle in enumerate((0.0, 45.0, 90.0, 135.0)):
            index = session.add_array(truth, f"capture {capture}")
            session.synthesize(
                index,
                BlurSpec(
                    kind="Line",
                    length=5.0,
                    angle_degrees=angle,
                    shift_x=float(capture - 1.5),
                ),
                read_noise_sigma=0.002,
                seed=40 + capture,
            )
            indices.append(index)
        target, inverse, posterior = session.deblur_multicapture_posterior(
            indices, target_index=indices[1], passes=3)
        self.assertEqual(target, indices[1])
        self.assertEqual(posterior.image.shape, truth.shape)
        self.assertTrue(np.all(np.isfinite(posterior.image)))
        self.assertEqual(inverse.diagnostics["capture_count"], 4)
        self.assertEqual(
            session.sources[target].diagnostics["posterior"]["selection_policy"],
            "all_center_inverse_and_noise_measures_retained_no_winner_branch",
        )


if __name__ == "__main__":
    unittest.main()
