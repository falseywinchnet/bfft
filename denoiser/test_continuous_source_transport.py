"""Invariants for continuous conservative source-measure transport."""

from __future__ import annotations

import unittest

import numpy as np

from .continuous_source_transport import (
    ContinuousSourceResolution,
    _exclude_target_identity,
    denoise_continuous_source_transport,
    selling_decomposition,
    source_measure_operator,
)


class ContinuousSourceTransportTests(unittest.TestCase):
    def test_selling_flux_reconstructs_inverse_metric(self):
        yy, xx = np.mgrid[:11, :13]
        angle = 0.4 * np.sin(xx / 4.0) + 0.2 * np.cos(yy / 3.0)
        stretch = 1.0 + 0.8 * (xx + yy) / (xx.max() + yy.max())
        cosine = np.cos(angle)
        sine = np.sin(angle)
        mxx = stretch * cosine**2 + sine**2 / stretch
        mxy = (stretch - 1.0 / stretch) * cosine * sine
        myy = stretch * sine**2 + cosine**2 / stretch
        decomposition = selling_decomposition(mxx, mxy, myy)
        self.assertGreaterEqual(decomposition["minimum_coefficient"], 0.0)
        self.assertLess(decomposition["maximum_reconstruction_error"], 2e-14)

    def test_operator_is_reversible_constant_preserving_and_leave_one_out(self):
        shape = (12, 14)
        operator, diagnostic = source_measure_operator(
            np.ones(shape), np.zeros(shape), np.ones(shape))
        np.testing.assert_allclose(operator @ np.ones(np.prod(shape)), 1.0)
        np.testing.assert_allclose(operator.diagonal(), 0.0)
        self.assertLess(diagnostic["stationary_mass_maximum_error"], 2e-16)

    def test_conditioned_ancestry_conserves_mass_and_excludes_target(self):
        shape = (10, 12)
        operator, _diagnostic = source_measure_operator(
            np.ones(shape), np.zeros(shape), np.ones(shape))
        ancestry = _exclude_target_identity(operator @ operator.toarray())
        np.testing.assert_allclose(np.sum(ancestry, axis=1), 1.0)
        np.testing.assert_allclose(np.diag(ancestry), 0.0)

    def test_constant_field_is_exact(self):
        field = np.full((10, 12), 0.43)
        estimate, diagnostic = denoise_continuous_source_transport(field)
        np.testing.assert_allclose(estimate, field, atol=2e-15, rtol=0.0)
        self.assertFalse(diagnostic["continuation_ceiling_hit"])

    def test_every_accepted_continuation_descends_action_and_range(self):
        yy, xx = np.mgrid[:12, :14]
        field = 0.3 + 0.2 * np.sin(xx / 2.5) + 0.15 * (yy > 6)
        estimate, diagnostic = denoise_continuous_source_transport(
            field, ContinuousSourceResolution(maximum_continuations=8))
        for record in diagnostic["continuations"]:
            if record["accepted"]:
                self.assertLess(
                    record["residual_action_after"],
                    record["residual_action_before"])
        self.assertGreaterEqual(float(np.min(estimate)), float(np.min(field)))
        self.assertLessEqual(float(np.max(estimate)), float(np.max(field)))


if __name__ == "__main__":
    unittest.main()
