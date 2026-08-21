"""Invariant and positive-control tests for exposure transport deblurring."""

from __future__ import annotations

import unittest

import numpy as np

from .estimation import estimate_kernel_pair
from .kernels import (
    CircularTransportPlan,
    adjoint_circular,
    apply_circular,
    curved_path_kernel,
    disk_kernel,
    gaussian_kernel,
    identity_kernel,
    line_kernel,
)
from .solver import fuse_transport_observations, multi_wiener
from .synthetic import degrade


def _fixture(size: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:1:complex(size), 0:1:complex(size)]
    image = 0.18 + 0.16 * xx + 0.08 * yy
    image += 0.28 * (((xx - 0.28) ** 2 + (yy - 0.35) ** 2) < 0.13 ** 2)
    image += 0.22 * (xx + 0.55 * yy > 0.92)
    image += 0.08 * np.sin(2.0 * np.pi * (7.0 * xx + 3.0 * yy))
    image += 0.05 * np.sin(2.0 * np.pi * (2.0 * xx + 11.0 * yy))
    return np.clip(image, 0.0, 1.0)


class ExposureTransportTests(unittest.TestCase):
    def test_kernels_are_positive_unit_mass_measures(self) -> None:
        for kernel in (
            identity_kernel(), gaussian_kernel(1.7), disk_kernel(3.2),
            line_kernel(9.0, 37.0),
        ):
            self.assertGreaterEqual(float(np.min(kernel.psf)), 0.0)
            self.assertAlmostEqual(kernel.mass, 1.0, places=14)
            self.assertAlmostEqual(abs(kernel.otf((64, 64))[0, 0]), 1.0, places=13)

    def test_forward_and_adjoint_close(self) -> None:
        rng = np.random.default_rng(19)
        x = rng.normal(size=(48, 56))
        y = rng.normal(size=(48, 56))
        kernel = line_kernel(8.0, 31.0)
        left = float(np.vdot(apply_circular(x, kernel), y).real)
        right = float(np.vdot(x, adjoint_circular(y, kernel)).real)
        self.assertAlmostEqual(left, right, places=10)

    def test_circular_plan_matches_public_forward_and_adjoint(self) -> None:
        rng = np.random.default_rng(491)
        image = rng.random((31, 29, 3))
        kernel = curved_path_kernel(11.0, 37.0, 6.0)
        plan = CircularTransportPlan(kernel, image.shape[:2])
        np.testing.assert_array_equal(
            plan.forward(image), apply_circular(image, kernel))
        np.testing.assert_array_equal(
            plan.adjoint(image), adjoint_circular(image, kernel))

    def test_identity_is_an_exact_fixed_point(self) -> None:
        truth = _fixture(64)
        result = fuse_transport_observations(
            [truth], [identity_kernel()], tv_weight=0.0, passes=3)
        self.assertLess(float(np.max(np.abs(result.image - truth))), 1e-11)

    def test_complementary_lines_improve_over_each_capture(self) -> None:
        truth = _fixture()
        kernels = [line_kernel(11.0, 0.0), line_kernel(11.0, 90.0)]
        observations = [
            degrade(truth, kernel, gaussian_sigma=0.002, seed=50 + index)
            for index, kernel in enumerate(kernels)
        ]
        result = fuse_transport_observations(
            observations, kernels, tv_weight=0.0012,
            flux_penalty=0.035, passes=20)
        capture_mse = min(float(np.mean((value - truth) ** 2)) for value in observations)
        result_mse = float(np.mean((result.image - truth) ** 2))
        self.assertLess(result_mse, 0.70 * capture_mse)

    def test_cross_observation_closure_selects_true_kernel_pair(self) -> None:
        truth = _fixture()
        first = line_kernel(9.0, 0.0)
        second = line_kernel(9.0, 90.0)
        candidates = [
            identity_kernel(),
            line_kernel(7.0, 0.0), line_kernel(9.0, 0.0),
            line_kernel(11.0, 0.0), line_kernel(7.0, 90.0),
            line_kernel(9.0, 90.0), line_kernel(11.0, 90.0),
        ]
        estimate = estimate_kernel_pair(
            degrade(truth, first, gaussian_sigma=0.0005, seed=1),
            degrade(truth, second, gaussian_sigma=0.0005, seed=2),
            candidates,
        )
        self.assertEqual(estimate.first.name, first.name)
        self.assertEqual(estimate.second.name, second.name)

    def test_identical_blur_has_no_complementary_coverage_gain(self) -> None:
        shape = (64, 64)
        kernel = disk_kernel(3.0)
        one = multi_wiener([_fixture(64)], [kernel]).diagnostics["coverage"]
        two = multi_wiener(
            [_fixture(64), _fixture(64)], [kernel, kernel],
            precisions=np.asarray((0.5, 0.5)),
        ).diagnostics["coverage"]
        self.assertTrue(np.allclose(one["normalized"], two["normalized"]))
        self.assertEqual(one["dead_fraction"], two["dead_fraction"])

    def test_common_blur_is_reported_as_an_unidentifiable_gauge(self) -> None:
        truth = _fixture()
        kernel = disk_kernel(3.0)
        candidates = [
            identity_kernel(), gaussian_kernel(2.0), gaussian_kernel(3.0),
            disk_kernel(2.0), kernel, disk_kernel(4.0),
        ]
        estimate = estimate_kernel_pair(
            degrade(truth, kernel, gaussian_sigma=0.0005, seed=7),
            degrade(truth, kernel, gaussian_sigma=0.0005, seed=8),
            candidates,
        )
        self.assertTrue(estimate.common_blur_unidentifiable)
        self.assertLess(estimate.relative_transport_strength, 0.02)

    def test_unsupported_single_blur_uses_coverage_fallback(self) -> None:
        truth = _fixture()
        kernel = gaussian_kernel(2.0)
        observation = degrade(truth, kernel, gaussian_sigma=0.002, seed=22)
        result = fuse_transport_observations([observation], [kernel])
        self.assertEqual(
            result.diagnostics["method"], "coverage_gated_wiener_fallback")
        self.assertGreater(result.diagnostics["coverage"]["dead_fraction"], 0.30)


if __name__ == "__main__":
    unittest.main()
