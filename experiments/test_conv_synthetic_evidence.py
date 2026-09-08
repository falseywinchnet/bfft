from __future__ import annotations

import unittest

import numpy as np

from experiments.conv_synthetic_evidence import (
    _conv_line,
    _lanczos_line,
    _linear_line,
    _raw_quintic_line,
    _raw_quintic_resize,
    _scene,
    _sign_changes,
)


class ConvSyntheticEvidenceTest(unittest.TestCase):
    def test_line_operators_are_cardinal(self) -> None:
        source = np.array((0.2, 0.8, -0.1, 0.4, 0.9, 0.7, 0.3))
        for operation in (
            _conv_line, _raw_quintic_line, _lanczos_line, _linear_line
        ):
            refined = operation(source, 2)
            np.testing.assert_allclose(refined[::2], source, atol=4e-15, rtol=0)

    def test_raw_quintic_is_nested_cardinal(self) -> None:
        source = np.arange(7 * 6 * 3, dtype=float).reshape(7, 6, 3) / 100.0
        refined = _raw_quintic_resize(source, 2)
        np.testing.assert_allclose(
            refined[::2, ::2], source, atol=4e-14, rtol=0
        )

    def test_conv_does_not_add_sign_pairs_on_step_or_box(self) -> None:
        for source in (
            np.array((0, 0, 0, 1, 1, 1, 1), dtype=float),
            np.array((0, 0, 1, 1, 1, 0, 0), dtype=float),
        ):
            refined = _conv_line(source, 8)
            self.assertLessEqual(
                _sign_changes(np.diff(refined)),
                _sign_changes(np.diff(source)),
            )

    def test_raw_quintic_is_the_active_projection_ablation(self) -> None:
        source = np.array((0, 0, 0, 1, 1, 1, 1), dtype=float)
        raw = _raw_quintic_line(source, 8)
        admitted = _conv_line(source, 8)
        self.assertGreater(
            _sign_changes(np.diff(raw)), _sign_changes(np.diff(source))
        )
        self.assertLessEqual(
            _sign_changes(np.diff(admitted)), _sign_changes(np.diff(source))
        )

    def test_analytic_scenes_have_declared_range(self) -> None:
        for kind in ("edge", "carrier", "curved", "crossing"):
            field = _scene(kind, 17, angle=37.5, phase=0.4, offset=0.001)
            self.assertTrue(np.all(np.isfinite(field)))
            self.assertGreaterEqual(float(np.min(field)), 0.0)
            self.assertLessEqual(float(np.max(field)), 1.0)


if __name__ == "__main__":
    unittest.main()
