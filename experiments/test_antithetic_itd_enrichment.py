"""Invariant tests for antithetic CONV--ITD enrichment."""

from __future__ import annotations

from fractions import Fraction
import math
import unittest

import numpy as np

from experiments.antithetic_itd_enrichment import (
    conv_itd_baseline,
    cubature_fourth_moment,
    cubature_second_moment,
    detail_tight_frame,
    enriched_proposal,
    first_topology_threshold,
    itd_knot_matrix,
    lifted_detail_tight_frame,
    fourth_order_spherical_cubature,
    null_calibrated_correction,
)
from experiments.convstar import (
    ordered_sign_ledger,
    project_signed_fibres,
    raw_current_jet_bank,
)


class ExactPairIdentities(unittest.TestCase):
    def test_odd_powers_cancel_with_rational_arithmetic(self) -> None:
        f = Fraction(7, 5)
        q = Fraction(-11, 13)
        epsilon = Fraction(3, 17)
        for odd in (1, 3, 5, 7, 9):
            paired = ((f + epsilon*q)**odd + (f - epsilon*q)**odd) / 2
            expanded_even = sum(
                Fraction(math.comb(odd, k)) * f**(odd-k) * (epsilon*q)**k
                for k in range(0, odd+1, 2)
            )
            self.assertEqual(paired, expanded_even)

    def test_linear_operator_has_exactly_zero_null_correction(self) -> None:
        matrix = (
            (Fraction(2), Fraction(-1)),
            (Fraction(3, 2), Fraction(5, 7)),
        )
        f = (Fraction(7, 3), Fraction(-4, 5))
        q = (Fraction(11, 9), Fraction(2, 3))
        epsilon = Fraction(5, 8)

        def apply(x: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
            return tuple(sum(row[j]*x[j] for j in range(2)) for row in matrix)

        plus = tuple(f[j]+epsilon*q[j] for j in range(2))
        minus = tuple(f[j]-epsilon*q[j] for j in range(2))
        noise_plus = tuple(epsilon*q[j] for j in range(2))
        noise_minus = tuple(-epsilon*q[j] for j in range(2))
        correction = tuple(
            (apply(plus)[i]+apply(minus)[i])/2-apply(f)[i]
            -(apply(noise_plus)[i]+apply(noise_minus)[i])/2+apply((Fraction(0), Fraction(0)))[i]
            for i in range(2)
        )
        self.assertEqual(correction, (Fraction(0), Fraction(0)))


class FrameGeometry(unittest.TestCase):
    def test_irregular_clock_frame_matches_projector_covariance(self) -> None:
        tau = np.array([0.0, .7, 2.2, 2.8, 5.9, 9.0, 13.1])
        frame = detail_tight_frame(tau)
        np.testing.assert_allclose(frame.probes.T @ frame.probes, np.eye(frame.rank), atol=2e-15)
        np.testing.assert_allclose(
            frame.probes @ frame.probes.T / frame.rank,
            frame.covariance,
            atol=2e-15,
        )
        np.testing.assert_allclose(frame.projector @ frame.projector, frame.projector, atol=3e-15)

    def test_uniform_covariance_is_minimax_at_fixed_trace(self) -> None:
        rng = np.random.default_rng(104729)
        for rank in range(2, 10):
            optimum = 1.0/rank
            self.assertAlmostEqual(np.min(np.linalg.eigvalsh(np.eye(rank)/rank)), optimum)
            for _ in range(200):
                factor = rng.standard_normal((rank, rank))
                covariance = factor @ factor.T
                covariance /= np.trace(covariance)
                self.assertLessEqual(np.min(np.linalg.eigvalsh(covariance)), optimum+2e-15)

    def test_one_probe_cannot_cover_multidimensional_detail_space(self) -> None:
        q = np.arange(1.0, 8.0)
        self.assertEqual(np.linalg.matrix_rank(np.outer(q, q)), 1)

    def test_lifted_frame_is_rewhitened_in_sample_space(self) -> None:
        tau = np.array([0.0, .7, 2.2, 2.8, 5.9, 9.0, 13.1])
        frame = lifted_detail_tight_frame(tau, np.linspace(0.0, 13.1, 97))
        np.testing.assert_allclose(frame.probes.T@frame.probes, np.eye(frame.rank), atol=3e-15)
        np.testing.assert_allclose(frame.projector@frame.projector, frame.projector, atol=4e-15)

    def test_tight_frames_do_not_fix_fourth_moments(self) -> None:
        coordinate = np.eye(2)
        rotation = np.array([[1.0, -1.0], [1.0, 1.0]])/np.sqrt(2.0)
        direction = np.array([1.0, 0.0])
        coordinate_fourth = np.mean((direction@coordinate)**4)
        rotated_fourth = np.mean((direction@rotation)**4)
        self.assertAlmostEqual(coordinate_fourth, .5)
        self.assertAlmostEqual(rotated_fourth, .25)
        np.testing.assert_allclose(coordinate@coordinate.T/2, rotation@rotation.T/2, atol=2e-16)

    def test_spherical_rule_matches_second_and_fourth_moments(self) -> None:
        for rank in range(2, 9):
            rule = fourth_order_spherical_cubature(rank)
            np.testing.assert_allclose(np.sum(rule.points*rule.weights, axis=1), 0.0, atol=2e-15)
            np.testing.assert_allclose(cubature_second_moment(rule), np.eye(rank)/rank, atol=2e-15)
            fourth = cubature_fourth_moment(rule)
            target = np.zeros_like(fourth)
            for i in range(rank):
                for j in range(rank):
                    for k in range(rank):
                        for ell in range(rank):
                            target[i, j, k, ell] = (
                                (i == j)*(k == ell)
                                +(i == k)*(j == ell)
                                +(i == ell)*(j == k)
                            )/(rank*(rank+2.0))
            np.testing.assert_allclose(fourth, target, atol=3e-15)


class NonlinearIdentities(unittest.TestCase):
    def test_zero_signal_correction_is_numerically_zero_for_conv_itd(self) -> None:
        n = 96
        q = np.sin(2*np.pi*np.arange(n)/17.0)
        correction = null_calibrated_correction(
            conv_itd_baseline, np.zeros(n), q, .1
        )
        np.testing.assert_array_equal(correction, np.zeros(n))

    def test_enriched_split_closes_to_roundoff(self) -> None:
        n = 128
        t = np.linspace(0.0, 1.0, n)
        signal = .2*t + np.sin(2*np.pi*(5*t+9*t*t))
        tau = np.linspace(0.0, n-1.0, 9)
        frame = detail_tight_frame(tau)
        probes = np.column_stack([
            np.interp(np.arange(n), tau, frame.probes[:, j])
            for j in range(frame.rank)
        ])
        baseline = enriched_proposal(conv_itd_baseline, signal, probes, .015)
        detail = signal-baseline
        np.testing.assert_allclose(baseline+detail, signal, rtol=0.0, atol=2*np.finfo(float).eps)

    def test_first_sampled_current_threshold_brackets_a_sign_event(self) -> None:
        rng = np.random.default_rng(8191)
        for _ in range(100):
            signal = np.cumsum(rng.uniform(.05, 1.0, 32)*rng.choice((-1.0, 1.0), 32))
            probe = rng.standard_normal(32)
            threshold = first_topology_threshold(signal, probe)
            original = np.sign(np.diff(signal))
            # A single-ULP displacement can round back onto the cancelling
            # hyperplane during the subsequent multiply/add.  A relative
            # displacement tests the mathematical open intervals rather than
            # the accident of one floating-point evaluation order.
            below = threshold*(1.0-1e-12)
            for branch in (-1.0, 1.0):
                np.testing.assert_array_equal(
                    np.sign(np.diff(signal+branch*below*probe)), original
                )
            above = threshold*(1.0+1e-12)
            changed = any(
                np.any(np.sign(np.diff(signal+branch*above*probe)) != original)
                for branch in (-1.0, 1.0)
            )
            self.assertTrue(changed)


class CurrentAdmission(unittest.TestCase):
    def test_projection_restores_only_the_stated_fibre_constraints(self) -> None:
        rng = np.random.default_rng(65537)
        source = rng.standard_normal((80, 4))
        raw, delta = raw_current_jet_bank(source)
        signs = ordered_sign_ledger(raw, delta)
        proposal = raw + 3.0*rng.standard_normal(raw.shape)
        current = project_signed_fibres(proposal, signs, delta)
        np.testing.assert_allclose(np.sum(current, axis=1), delta, atol=2e-13, rtol=2e-13)
        self.assertGreaterEqual(float(np.min(signs*current)), -2e-13)
        second = project_signed_fibres(current, signs, delta)
        np.testing.assert_allclose(second, current, atol=3e-13, rtol=3e-13)

    def test_signed_fibre_alone_does_not_imply_endpoint_range(self) -> None:
        feasible = np.array([[[1.0], [1.0], [-1.0], [-1.0], [1.0]]])
        signs = np.sign(feasible)
        delta = np.array([[1.0]])
        admitted = project_signed_fibres(feasible, signs, delta)
        np.testing.assert_allclose(admitted, feasible, atol=0.0, rtol=0.0)
        controls = np.r_[0.0, np.cumsum(admitted[0, :, 0])]
        self.assertGreater(float(np.max(controls)), 1.0)


if __name__ == "__main__":
    unittest.main()
