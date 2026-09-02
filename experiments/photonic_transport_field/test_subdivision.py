from __future__ import annotations

import math
import unittest

import numpy as np

from .subdivision import (
    deliverable_energy_error_bound,
    display_subdivision_error_bound,
    lambertian_centroid_error_bound,
    lambertian_kernel,
)


class SubdivisionTheoremTests(unittest.TestCase):
    def test_random_kernel_variation_is_below_theorem_bound(self) -> None:
        rng = np.random.default_rng(9102026)
        source_center = np.array((0.0, 0.0, 0.0))
        receiver_center = np.array((4.0, 0.3, -0.2))
        source_normal = np.array((1.0, 0.0, 0.0))
        receiver_normal = np.array((-1.0, 0.0, 0.0))
        source_radius = 0.24
        receiver_radius = 0.31
        source_eta = 0.08
        receiver_eta = 0.11
        reference = lambertian_kernel(
            source_center,
            source_normal,
            receiver_center,
            receiver_normal,
        )
        bound = lambertian_centroid_error_bound(
            centroid_separation=float(np.linalg.norm(receiver_center)),
            source_radius=source_radius,
            receiver_radius=receiver_radius,
            source_normal_deviation=source_eta,
            receiver_normal_deviation=receiver_eta,
        )
        maximum_error = 0.0
        for _ in range(4000):
            source_offset = rng.normal(size=3)
            source_offset *= source_radius * rng.random() / np.linalg.norm(source_offset)
            receiver_offset = rng.normal(size=3)
            receiver_offset *= receiver_radius * rng.random() / np.linalg.norm(receiver_offset)
            source_delta = rng.normal(size=3)
            source_delta -= np.dot(source_delta, source_normal) * source_normal
            source_delta *= source_eta * rng.random() / np.linalg.norm(source_delta)
            varied_source_normal = source_normal + source_delta
            varied_source_normal /= np.linalg.norm(varied_source_normal)
            receiver_delta = rng.normal(size=3)
            receiver_delta -= np.dot(receiver_delta, receiver_normal) * receiver_normal
            receiver_delta *= receiver_eta * rng.random() / np.linalg.norm(receiver_delta)
            varied_receiver_normal = receiver_normal + receiver_delta
            varied_receiver_normal /= np.linalg.norm(varied_receiver_normal)
            value = lambertian_kernel(
                source_center + source_offset,
                varied_source_normal,
                receiver_center + receiver_offset,
                varied_receiver_normal,
            )
            maximum_error = max(maximum_error, abs(value - reference))
        self.assertLessEqual(maximum_error, bound)

    def test_dyadic_smooth_subdivision_reduces_bound(self) -> None:
        coarse = lambertian_centroid_error_bound(
            centroid_separation=5.0,
            source_radius=0.5,
            receiver_radius=0.4,
            source_normal_deviation=0.10,
            receiver_normal_deviation=0.08,
        )
        fine = lambertian_centroid_error_bound(
            centroid_separation=5.0,
            source_radius=0.25,
            receiver_radius=0.2,
            source_normal_deviation=0.05,
            receiver_normal_deviation=0.04,
        )
        self.assertLess(fine, 0.4 * coarse)

    def test_deliverable_bound_scales_with_energy_and_area(self) -> None:
        arguments = dict(
            source_energy=3.0,
            receiver_area=2.0,
            centroid_separation=6.0,
            source_radius=0.2,
            receiver_radius=0.3,
            source_normal_deviation=0.04,
            receiver_normal_deviation=0.05,
        )
        value = deliverable_energy_error_bound(**arguments)
        arguments["source_energy"] *= 2.0
        arguments["receiver_area"] *= 0.5
        self.assertTrue(math.isclose(
            deliverable_energy_error_bound(**arguments), value
        ))

    def test_display_bound_is_linear_in_cell_radius(self) -> None:
        coarse = display_subdivision_error_bound(
            surface_radius=0.5,
            radiance_lipschitz=2.0,
            display_derivative_bound=0.75,
        )
        fine = display_subdivision_error_bound(
            surface_radius=0.125,
            radiance_lipschitz=2.0,
            display_derivative_bound=0.75,
        )
        self.assertEqual(fine, 0.25 * coarse)


if __name__ == "__main__":
    unittest.main()
