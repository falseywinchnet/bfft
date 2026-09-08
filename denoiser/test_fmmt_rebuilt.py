"""Structural invariants for the rebuilt-FMMT support donor."""

from __future__ import annotations

import unittest

import numpy as np

from .dcnt import target_excluded_conv_family
from .fmmt_rebuilt import (
    coherent_support_donation,
    denoise_rebuilt_fmmt,
    rebuild_fmmt_posterior,
    transport_supported_donation,
)
from .fmmt_certified import denoise_fmmt


class RebuiltFMMTTests(unittest.TestCase):
    def test_donation_stays_on_posterior_observation_segment(self):
        rng = np.random.default_rng(20260828)
        observation = rng.uniform(size=(18, 20))
        posterior = rng.uniform(size=observation.shape)
        family = target_excluded_conv_family(observation)
        estimate, diagnostic = coherent_support_donation(
            observation, posterior, family)
        lower = np.minimum(observation, posterior)
        upper = np.maximum(observation, posterior)
        self.assertTrue(np.all(estimate >= lower - 2e-15))
        self.assertTrue(np.all(estimate <= upper + 2e-15))
        self.assertLessEqual(diagnostic["maximum_segment_violation"], 2e-15)

    def test_unanimous_witnesses_donate_the_common_correction(self):
        posterior = np.full((12, 14), 0.3)
        observation = np.full((12, 14), 0.8)
        family = np.full((4, 12, 14), 0.6)
        family[0, 0::2, 0::2] = np.nan
        family[1, 0::2, 1::2] = np.nan
        family[2, 1::2, 0::2] = np.nan
        family[3, 1::2, 1::2] = np.nan
        estimate, _diagnostic = coherent_support_donation(
            observation, posterior, family)
        np.testing.assert_allclose(estimate, 0.6, atol=2e-15, rtol=0.0)

    def test_disagreement_contracts_without_sign_branch(self):
        posterior = np.full((12, 14), 0.5)
        observation = np.full((12, 14), 0.9)
        family = np.empty((4, 12, 14), dtype=np.float64)
        family[0] = 0.9
        family[1] = 0.5
        family[2] = 0.5
        family[3] = 0.5
        family[0, 0::2, 0::2] = np.nan
        family[1, 0::2, 1::2] = np.nan
        family[2, 1::2, 0::2] = np.nan
        family[3, 1::2, 1::2] = np.nan
        estimate, _diagnostic = coherent_support_donation(
            observation, posterior, family)
        self.assertTrue(np.all(estimate >= posterior))
        self.assertTrue(np.all(estimate < observation))

    def test_isolated_donation_does_not_reach_observer(self):
        posterior = np.full((18, 20), 0.4)
        observation = posterior.copy()
        observation[9, 10] = 0.9
        provisional = posterior.copy()
        provisional[9, 10] = 0.8
        estimate, diagnostic = transport_supported_donation(
            observation, posterior, provisional)
        self.assertAlmostEqual(float(estimate[9, 10]), 0.4, places=14)
        self.assertLess(
            diagnostic["transport_supported_action"], 1e-28)

    def test_precomputed_canon_reuse_is_exact(self):
        yy, xx = np.indices((12, 14), dtype=np.float64)
        observation = np.clip(
            0.3 + 0.02 * xx + 0.04 * np.sin(0.7 * yy), 0.0, 1.0)
        canonical, canonical_diagnostic = denoise_fmmt(observation)
        reused, _ = rebuild_fmmt_posterior(
            observation, canonical, canonical_diagnostic)
        direct, _ = denoise_rebuilt_fmmt(observation)
        np.testing.assert_array_equal(reused, direct)

if __name__ == "__main__":
    unittest.main()
