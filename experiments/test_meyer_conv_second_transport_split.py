"""Structural checks for the CONV two-jet Meyer transport experiment."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.meyer_conv_second_transport_split import (
    conv_hessian_adjoint,
    conv_second_order_rof,
    conv_symmetric_hessian,
    second_current_meyer,
)


class MeyerConvSecondTransportTests(unittest.TestCase):
    def test_twojet_and_adjoint_pair_exactly(self) -> None:
        rng = np.random.default_rng(912)
        source = rng.standard_normal((17, 19))
        tensor = tuple(rng.standard_normal(source.shape) for _ in range(3))
        jet = conv_symmetric_hessian(source)
        left = sum(float(np.vdot(a, b).real) for a, b in zip(jet, tensor))
        right = float(np.vdot(source, conv_hessian_adjoint(*tensor)).real)
        self.assertLess(abs(left - right), 2e-12 * max(abs(left), 1.0))

    def test_affine_field_is_in_twojet_nullspace(self) -> None:
        yy, xx = np.mgrid[:21, :23].astype(np.float64)
        jet = conv_symmetric_hessian(4.0 + 2.5 * xx - 1.75 * yy)
        self.assertLess(max(float(np.max(np.abs(part))) for part in jet), 2e-12)

    def test_dual_ball_and_recomposition(self) -> None:
        rng = np.random.default_rng(41)
        source = rng.standard_normal((24, 25))
        _smooth, state, diagnostic = conv_second_order_rof(
            source, fidelity=0.1, state=None, iterations=32)
        norm = np.sqrt(sum(component * component for component in state))
        self.assertLessEqual(float(np.max(norm)), 1.0 + 3e-15)
        cartoon, texture, split = second_current_meyer(
            source, mu2=20.0, outer_iterations=12,
            inner_iterations=4, conv_operator=True)
        self.assertLess(float(np.max(np.abs(source - cartoon - texture))), 2e-15)
        self.assertEqual(split["texture_operator"], "CONV two-jet")


if __name__ == "__main__":
    unittest.main()
