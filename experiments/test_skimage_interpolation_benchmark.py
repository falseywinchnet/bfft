from __future__ import annotations

import unittest

import numpy as np

from experiments.run_skimage_interpolation_benchmark import (
    _array_rasters,
    _patch_origins,
    _patch_side,
    evaluate_patch,
)
from experiments.skimage_interpolation_demo import compute_comparison


class SkimageInterpolationBenchmarkTest(unittest.TestCase):
    def test_stack_expansion_preserves_every_plane(self) -> None:
        stack = np.arange(5 * 9 * 11).reshape(5, 9, 11)
        rasters = list(_array_rasters("stack", stack))
        self.assertEqual(len(rasters), 5)
        np.testing.assert_array_equal(rasters[3][1], stack[3])

        color_sequence = np.zeros((4, 9, 11, 3), dtype=np.uint8)
        self.assertEqual(
            len(list(_array_rasters("color", color_sequence))), 4
        )

    def test_patch_geometry_is_native_nested_and_stratified(self) -> None:
        self.assertEqual(_patch_side((512, 300), 65, 2), 65)
        self.assertEqual(_patch_side((25, 25), 65, 2), 17)
        self.assertEqual(
            _patch_origins((100, 120), 65, 4),
            [(0, 0), (0, 55), (35, 0), (35, 55)],
        )

    def test_affine_native_patch_has_zero_bilinear_held_out_residual(self) -> None:
        y, x = np.mgrid[:17, :17].astype(np.float64)
        truth = 0.1 + 0.02 * x + 0.03 * y
        result = evaluate_patch(truth)
        exact = result["exact_sublattice"]
        self.assertLess(exact["bilinear"]["held_out_mse"], 2.0e-31)
        self.assertLess(
            exact["cosine_hermite_transport"]["held_out_mse"], 2.0e-29
        )
        for method in exact.values():
            self.assertLess(method["maximum_sample_error"], 2.0e-14)

    def test_demo_computes_the_same_native_protocol_on_demand(self) -> None:
        y, x = np.mgrid[:80, :90].astype(np.float64)
        image = 0.2 + 0.3 * x / 89.0 + 0.1 * y / 79.0
        result = compute_comparison(
            image,
            protocol="exact_sublattice",
            maximum_truth_side=65,
            origin_fraction_yx=(0.0, 1.0),
        )
        self.assertEqual(result["truth"].shape, (65, 65))
        self.assertEqual(result["source"].shape, (33, 33))
        self.assertEqual(result["origin_yx"], (0, 25))
        self.assertLess(
            result["metrics"]["Bilinear"]["held_out_mse"], 3.0e-31
        )
        self.assertIn("AMD FSR 1.0 EASU", result["reconstructions"])
        self.assertIsNone(
            result["metrics"]["AMD FSR 1.0 EASU"][
                "matched_family_round_trip_mse"
            ]
        )
        for name in (
            "Bilinear",
            "Lanczos-3 sinc",
            "Cosine-Hermite Riemannian transport",
        ):
            self.assertGreaterEqual(
                result["metrics"][name]["matched_family_round_trip_mse"],
                0.0,
            )

        lifting = compute_comparison(
            image,
            protocol="riemannian_lifting",
            maximum_truth_side=17,
            origin_fraction_yx=(0.0, 0.0),
        )
        self.assertEqual(lifting["source"].shape, (9, 9))
        self.assertLess(
            lifting["metrics"]["Cosine-Hermite Riemannian transport"][
                "matched_family_round_trip_mse"
            ],
            3.0e-29,
        )


if __name__ == "__main__":
    unittest.main()
