from __future__ import annotations

import unittest

import numpy as np

from experiments.conv_distilled_core import (
    convstar_characteristic_transport_resize,
    distilled_conv_resize,
    convstar_bounded_resize,
    convstar_tangent_transport_resize,
    distilled_conv_synthesis,
    nodal_current_geometry,
    oriented_chord_conv_synthesis,
    symmetric_conv_synthesis,
)


class DistilledConvTests(unittest.TestCase):
    def test_tensor_weight_is_bounded_and_axis_covariant(self):
        rng = np.random.default_rng(20260828)
        source = rng.random((17, 19, 3), dtype=np.float32)
        beta = nodal_current_geometry(source)[3]
        transposed = nodal_current_geometry(np.swapaxes(source, 0, 1))[3]
        self.assertGreaterEqual(float(np.min(beta)), 0.0)
        self.assertLessEqual(float(np.max(beta)), 1.0)
        np.testing.assert_allclose(
            np.swapaxes(transposed, 0, 1), 1.0 - beta,
            atol=4.0e-15, rtol=0.0,
        )

    def test_direct_order_coordinate_equals_tensor_eigen_form(self):
        rng = np.random.default_rng(206)
        source = rng.normal(size=(17, 19, 4)).astype(np.float32)
        gxx, gxy, gyy, direct = nodal_current_geometry(source)
        trace = gxx + gyy
        gap = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4.0 * gxy * gxy, 0.0))
        chi = np.divide(gap, trace, out=np.zeros_like(gap), where=trace > 0.0)
        normal_y_squared = np.divide(
            0.5 * (gap - (gxx - gyy)),
            gap,
            out=np.full_like(gap, 0.5),
            where=gap > 0.0,
        )
        eigen_form = 0.5 * (1.0 - chi) + chi * normal_y_squared
        np.testing.assert_allclose(direct, eigen_form, atol=3.0e-16, rtol=0.0)

    def test_complete_synthesis_is_axis_covariant(self):
        rng = np.random.default_rng(91)
        source = rng.random((13, 15, 2), dtype=np.float32)
        direct = distilled_conv_synthesis(source, (25, 29))
        transposed = distilled_conv_synthesis(
            np.swapaxes(source, 0, 1), (29, 25)
        )
        np.testing.assert_allclose(
            direct, np.swapaxes(transposed, 0, 1), atol=2.0e-7, rtol=0.0
        )

    def test_fused_q1_write_matches_materialized_coordinate(self):
        from backend import conv_resize, linear_resize, q1_order_blend

        rng = np.random.default_rng(192)
        source = rng.random((17, 19, 3), dtype=np.float32)
        target = (33, 37)
        forward = conv_resize(source, target)
        reverse = np.swapaxes(
            conv_resize(np.swapaxes(source, 0, 1), (target[1], target[0])),
            0,
            1,
        )
        eta = nodal_current_geometry(source)[3].astype(np.float32)
        materialized = linear_resize(eta, target)[..., None]
        expected = forward + materialized * (reverse - forward)
        actual = q1_order_blend(eta, forward, reverse)
        np.testing.assert_allclose(actual, expected, atol=1.2e-7, rtol=0.0)

    def test_symmetric_baseline_is_axis_covariant(self):
        rng = np.random.default_rng(19)
        source = rng.random((13, 15, 2), dtype=np.float32)
        direct = symmetric_conv_synthesis(source, (25, 29))
        transposed = symmetric_conv_synthesis(
            np.swapaxes(source, 0, 1), (29, 25)
        )
        np.testing.assert_array_equal(direct, np.swapaxes(transposed, 0, 1))

    def test_constants_and_nested_sources_are_exact(self):
        constant = np.full((9, 11, 3), 0.375, dtype=np.float32)
        result = distilled_conv_synthesis(constant, (17, 21))
        np.testing.assert_array_equal(result, np.full((17, 21, 3), 0.375))
        rng = np.random.default_rng(7)
        source = rng.random((9, 11), dtype=np.float32)
        result = distilled_conv_synthesis(source, (17, 21))
        np.testing.assert_array_equal(result[::2, ::2], source)

    def test_reduction_remains_basin_analysis(self):
        rng = np.random.default_rng(8)
        source = rng.random((17, 21), dtype=np.float32)
        reduced = distilled_conv_resize(source, (9, 11))
        from backend import conv_basin_average
        np.testing.assert_array_equal(
            reduced, conv_basin_average(source, (9, 11))
        )

    def test_oriented_chord_is_cardinal_constant_and_axis_covariant(self):
        constant = np.full((9, 11, 3), 0.375, dtype=np.float32)
        result = oriented_chord_conv_synthesis(constant, (17, 21))
        np.testing.assert_array_equal(result, np.full((17, 21, 3), 0.375))
        rng = np.random.default_rng(81)
        source = rng.random((9, 11), dtype=np.float32)
        direct = oriented_chord_conv_synthesis(source, (17, 21))
        transposed = oriented_chord_conv_synthesis(
            source.T, (21, 17)
        ).T
        np.testing.assert_array_equal(direct[::2, ::2], source)
        np.testing.assert_allclose(direct, transposed, atol=3.0e-7, rtol=0.0)

    def test_native_oriented_chord_matches_reference(self):
        rng = np.random.default_rng(821)
        source = rng.random((9, 11, 3), dtype=np.float32)
        native = oriented_chord_conv_synthesis(
            source, (17, 21), jet_tensor=True, jet_chord=False
        )
        reference = oriented_chord_conv_synthesis(
            source, (17, 21), jet_tensor=True, jet_chord=False,
            native=False,
        )
        np.testing.assert_allclose(native, reference, atol=1.1e-6, rtol=0.0)

    def test_bounded_resize_uses_basin_analysis_on_reduction(self):
        rng = np.random.default_rng(82)
        source = rng.random((17, 21, 3), dtype=np.float32)
        result = convstar_bounded_resize(source, (9, 11))
        from backend import conv_basin_average
        np.testing.assert_array_equal(
            result, conv_basin_average(source, (9, 11))
        )

    def test_continuous_tensor_transport_is_cardinal_and_covariant(self):
        rng = np.random.default_rng(823)
        source = rng.random((9, 11, 3), dtype=np.float32)
        direct = convstar_tangent_transport_resize(source, (17, 21))
        transposed = convstar_tangent_transport_resize(
            np.swapaxes(source, 0, 1), (21, 17)
        )
        np.testing.assert_array_equal(direct[::2, ::2], source)
        np.testing.assert_allclose(
            direct, np.swapaxes(transposed, 0, 1), atol=4.0e-7, rtol=0.0
        )

    def test_curved_characteristic_is_cardinal_and_covariant(self):
        rng = np.random.default_rng(824)
        source = rng.random((9, 11), dtype=np.float32)
        direct = convstar_characteristic_transport_resize(source, (17, 21))
        transposed = convstar_characteristic_transport_resize(
            source.T, (21, 17)
        ).T
        np.testing.assert_array_equal(direct[::2, ::2], source)
        np.testing.assert_allclose(direct, transposed, atol=6.0e-7, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
