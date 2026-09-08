"""Structural checks for the Eikonal-state Meyer dual-flux experiment."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.meyer_bregman import rof_fgp
from experiments.meyer_eikonal_transport_split import (
    inverse_sqrt_metric,
    metric_rof,
)


class MeyerEikonalTransportSplitTests(unittest.TestCase):
    def test_identity_metric_reduces_to_ordinary_flux_iteration(self) -> None:
        rng = np.random.default_rng(8)
        source = rng.standard_normal((32, 32))
        identity = (np.ones_like(source), np.zeros_like(source), np.ones_like(source))
        # The first step is identical.  Later finite iterates differ because
        # the repository oracle uses FISTA momentum while the metric prototype
        # deliberately keeps monotone projected gradient.
        actual, _state, diagnostic = metric_rof(
            source, 0.17, identity, None, 1)
        expected, _ = rof_fgp(source, 0.17, 1)
        self.assertLess(np.max(np.abs(actual-expected)), 2e-14)
        self.assertLessEqual(diagnostic["maximum_whitened_flux_norm"], 1+1e-14)

    def test_inverse_square_root_reconstructs_inverse_metric(self) -> None:
        rng = np.random.default_rng(11)
        a = 1.0+rng.random((20, 21))*5.0
        c = 1.0+rng.random((20, 21))*4.0
        bound = 0.7*np.sqrt(a*c)
        b = (2*rng.random((20, 21))-1)*bound
        determinant = a*c-b*b
        metric = {"metric_xx": a, "metric_xy": b, "metric_yy": c}
        lxx, lxy, lyy = inverse_sqrt_metric(metric)
        inv_xx, inv_xy, inv_yy = c/determinant, -b/determinant, a/determinant
        self.assertLess(np.max(np.abs(lxx*lxx+lxy*lxy-inv_xx)), 2e-14)
        self.assertLess(np.max(np.abs(lxy*(lxx+lyy)-inv_xy)), 2e-14)
        self.assertLess(np.max(np.abs(lxy*lxy+lyy*lyy-inv_yy)), 2e-14)


if __name__ == "__main__":
    unittest.main()
