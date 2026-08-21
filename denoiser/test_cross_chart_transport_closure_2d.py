"""Invariants for relative transport-chart closure."""

from __future__ import annotations

import unittest

import numpy as np

from .causal_information_lineage_2d import causal_information_lineage_law_2d
from .cross_chart_transport_closure_2d import (
    cross_chart_transport_closure_readout,
    denoise_cross_chart_transport_closure_2d,
)


def _synthetic_law(
    chart_signal: tuple[float, ...],
    chart_mass: tuple[float, ...] | None = None,
) -> dict[str, np.ndarray]:
    count = len(chart_signal)
    weights = (
        np.full(count, 1.0 / count)
        if chart_mass is None
        else np.asarray(chart_mass, dtype=np.float64)
    )
    return {
        "signal": np.broadcast_to(
            np.asarray(chart_signal, dtype=np.float64), (8, 8, count)).copy(),
        "hj_simplex_collision_mass": np.broadcast_to(
            weights, (8, 8, count)).copy(),
        "tangent": np.stack((
            np.cos(np.arange(count) * np.pi / count),
            np.sin(np.arange(count) * np.pi / count),
        ), axis=-1),
    }


class CrossChartTransportClosure2DTests(unittest.TestCase):
    def test_constant_is_exact_through_full_transport(self):
        image = np.full((8, 8), 0.37)
        estimate, diagnostic = denoise_cross_chart_transport_closure_2d(
            image, angular_count=4, quantile_count=8)
        np.testing.assert_allclose(estimate, image, atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            diagnostic["readouts"]["source_coverage_closure_barycenter"],
            image, atol=2e-15, rtol=0.0)
        self.assertEqual(diagnostic["closure"]["chart_count"], 4)

    def test_pairwise_closure_equals_centered_variance(self):
        image = np.full((8, 8), 0.5)
        law = _synthetic_law((0.2, 0.4, 0.9), (0.2, 0.3, 0.5))
        forms, _ = cross_chart_transport_closure_readout(image, law)
        z = forms["transport_chart_signal"]
        w = forms["transport_chart_ownership"]
        pairwise = np.zeros(image.shape)
        for first in range(3):
            for second in range(3):
                pairwise += 0.5 * w[..., first] * w[..., second] * (
                    z[..., first] - z[..., second]) ** 2
        np.testing.assert_allclose(
            forms["transport_chart_variance"], pairwise,
            atol=2e-15, rtol=0.0)

    def test_common_latent_shift_cancels_from_closure(self):
        image = np.full((8, 8), 0.65)
        law = _synthetic_law((0.2, 0.4, 0.8))
        forms, _ = cross_chart_transport_closure_readout(image, law)
        shift = 1.7
        shifted = {key: np.array(value, copy=True) for key, value in law.items()}
        shifted["signal"] += shift
        shifted_forms, _ = cross_chart_transport_closure_readout(
            image + shift, shifted)
        np.testing.assert_allclose(
            shifted_forms["transport_chart_variance"],
            forms["transport_chart_variance"], atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            shifted_forms["cross_chart_closure_barycenter"],
            forms["cross_chart_closure_barycenter"] + shift,
            atol=2e-15, rtol=0.0)

    def test_chart_permutation_and_projective_sign_are_invariant(self):
        image = np.full((8, 8), 0.6)
        law = _synthetic_law((0.1, 0.3, 0.7, 0.9))
        reference, _ = cross_chart_transport_closure_readout(image, law)
        permutation = np.array([2, 0, 3, 1])
        permuted = {
            "signal": law["signal"][..., permutation],
            "hj_simplex_collision_mass": law[
                "hj_simplex_collision_mass"][..., permutation],
            "tangent": -law["tangent"][permutation],
        }
        candidate, _ = cross_chart_transport_closure_readout(image, permuted)
        for name in (
            "cross_chart_closure_barycenter",
            "transport_chart_consensus",
            "transport_chart_variance",
            "transport_chart_authority",
        ):
            np.testing.assert_allclose(
                candidate[name], reference[name], atol=2e-15, rtol=0.0)

    def test_map_disagreement_reduces_consensus_authority(self):
        image = np.full((8, 8), 0.8)
        agreement, _ = cross_chart_transport_closure_readout(
            image, _synthetic_law((0.2, 0.2, 0.2, 0.2)))
        disagreement, _ = cross_chart_transport_closure_readout(
            image, _synthetic_law((0.0, 0.1, 0.3, 0.4)))
        self.assertGreater(
            float(np.mean(agreement["transport_chart_authority"])),
            float(np.mean(disagreement["transport_chart_authority"])))

    def test_real_law_exposes_conserved_chart_ownership(self):
        yy, xx = np.mgrid[:8, :8]
        image = 0.2 + 0.4 * xx / 7.0 + 0.1 * np.sin(yy)
        law, _ = causal_information_lineage_law_2d(
            image, angular_count=4, quantile_count=8)
        forms, diagnostic = cross_chart_transport_closure_readout(image, law)
        np.testing.assert_allclose(
            np.sum(forms["transport_chart_ownership"], axis=-1),
            1.0, atol=2e-15, rtol=0.0)
        self.assertLess(diagnostic["mass_maximum_error"], 2e-15)
        self.assertTrue(np.all(np.isfinite(
            forms["cross_chart_closure_barycenter"])))
        self.assertTrue(np.all(forms["source_chart_coverage"] >= 0.0))
        self.assertTrue(np.all(forms["source_common_variance"] >= 0.0))
        self.assertLessEqual(
            float(np.max(forms["source_coverage_authority"])), 1.0)


if __name__ == "__main__":
    unittest.main()
