from __future__ import annotations

import unittest

import numpy as np

from experiments.conv_synthetic_evidence import _scene, _sign_changes
from experiments.convstar import (
    FUSED_CURRENT_KERNELS,
    _ordered_sign_ledger_reference,
    convstar_resize,
    ordered_sign_ledger,
    polyphase_tail_weights,
    project_signed_fibres,
    raw_current_fused_bank,
    raw_current_jet_bank,
    refine_lines,
)
from experiments.self_geometric_harmonic_interpolation import (
    _quintic_variation_lineage_profile,
    tensor_product_quintic_variation_resize,
)


class ConvStarTest(unittest.TestCase):
    def test_batched_ledger_is_exact_scalar_recurrence(self) -> None:
        rng = np.random.default_rng(20260828)
        for side, components in ((5, 1), (9, 7), (33, 29)):
            source = rng.normal(size=(side, components))
            source[:, ::3] = np.round(source[:, ::3], 1)
            raw, delta = raw_current_jet_bank(source)
            expected = _ordered_sign_ledger_reference(raw, delta)
            actual = ordered_sign_ledger(raw, delta)
            np.testing.assert_array_equal(actual, expected)

    def test_scalar_projection_satisfies_kkt_system(self) -> None:
        rng = np.random.default_rng(20260828)
        for side, components in ((5, 1), (11, 13), (35, 41)):
            source = rng.normal(size=(side, components))
            raw, delta = raw_current_jet_bank(source)
            signs = ordered_sign_ledger(raw, delta)
            current = project_signed_fibres(
                raw, signs, delta, chunk_size=97)
            np.testing.assert_allclose(
                np.sum(current, axis=1), delta, atol=2e-14, rtol=0.0)
            self.assertTrue(np.all(signs * current >= -2e-14))

            flat_raw = raw.transpose(0, 2, 1).reshape(-1, 5)
            flat_sign = signs.transpose(0, 2, 1).reshape(-1, 5)
            flat_current = current.transpose(0, 2, 1).reshape(-1, 5)
            for value, orientation, admitted in zip(
                    flat_raw, flat_sign, flat_current):
                if not np.any(orientation):
                    np.testing.assert_allclose(admitted, 0.0, atol=0.0)
                    continue
                active = np.abs(admitted) > 2e-13
                self.assertTrue(np.any(active))
                multipliers = value[active] - admitted[active]
                multiplier = float(np.mean(multipliers))
                np.testing.assert_allclose(
                    multipliers, multiplier, atol=3e-13, rtol=0.0)
                inactive = ~active
                self.assertTrue(np.all(
                    orientation[inactive]
                    * (value[inactive] - multiplier) <= 3e-13
                ))

    def test_declared_six_tap_bank_matches_jet_algebra(self) -> None:
        rng = np.random.default_rng(20260827)
        source = rng.normal(size=(19, 7))
        jet, delta = raw_current_jet_bank(source)
        fused, fused_delta = raw_current_fused_bank(source)
        np.testing.assert_allclose(fused_delta, delta, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(fused, jet, atol=9e-16, rtol=0.0)
        self.assertEqual(FUSED_CURRENT_KERNELS.shape, (5, 6))

    def test_tensorized_projection_matches_formal_current(self) -> None:
        rng = np.random.default_rng(7)
        for side in (5, 9, 17):
            source = rng.normal(size=(side, 11))
            raw, delta = raw_current_jet_bank(source)
            signs = ordered_sign_ledger(raw, delta)
            current = project_signed_fibres(raw, signs, delta, chunk_size=37)
            _, formal = _quintic_variation_lineage_profile(source)
            np.testing.assert_allclose(current, formal, atol=2e-14, rtol=0.0)

    def test_polyphase_synthesis_matches_formal_line(self) -> None:
        rng = np.random.default_rng(11)
        source = rng.normal(size=(17, 3))
        for scale in (2, 3, 4):
            compiled = refine_lines(source, scale)
            formal_source, formal_current = _quintic_variation_lineage_profile(source)
            # Evaluate the formal profile through direct quintic controls.
            expected = np.empty_like(compiled)
            expected[::scale] = formal_source
            for interval in range(source.shape[0] - 1):
                controls = np.concatenate((
                    formal_source[interval][None],
                    formal_source[interval][None]
                    + np.cumsum(formal_current[interval], axis=0),
                ))
                for phase in range(1, scale):
                    value = controls.copy()
                    u = phase / scale
                    for _ in range(5):
                        value = (1.0 - u) * value[:-1] + u * value[1:]
                    expected[interval * scale + phase] = value[0]
            np.testing.assert_allclose(compiled, expected, atol=3e-14, rtol=0.0)

    def test_cartesian_composition_matches_formal_conv(self) -> None:
        for kind in ("edge", "carrier", "curved", "crossing"):
            source = _scene(kind, 17, angle=37.5, phase=0.4, offset=0.002)
            compiled = convstar_resize(source, 2)
            formal = tensor_product_quintic_variation_resize(source, 2)
            np.testing.assert_allclose(compiled, formal, atol=8e-14, rtol=0.0)
            np.testing.assert_allclose(compiled[::2, ::2], source, atol=4e-14, rtol=0.0)

    def test_multichannel_and_topology(self) -> None:
        source = np.zeros((9, 3))
        source[4:, 0] = 1.0
        source[2:7, 1] = 1.0
        source[:, 2] = np.linspace(0.0, 1.0, 9)
        refined = refine_lines(source, 8)
        for channel in range(3):
            self.assertLessEqual(
                _sign_changes(np.diff(refined[:, channel])),
                _sign_changes(np.diff(source[:, channel])),
            )

    def test_two_x_tail_weights_are_exact_midpoint_weights(self) -> None:
        np.testing.assert_allclose(
            polyphase_tail_weights(2)[0],
            np.array((31, 26, 16, 6, 1), dtype=float) / 32.0,
            atol=0.0,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
