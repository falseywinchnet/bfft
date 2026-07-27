import unittest

import numpy as np

from error_spent_decomposition import (
    ErrorSpentConfig, ErrorSpentDecomposition,
)


class ErrorSpentDecompositionTests(unittest.TestCase):
    def test_spends_exact_budget_without_recursive_pixels(self):
        yy, xx = np.mgrid[0:48, 0:64]
        image = np.stack([
            xx / 63.0,
            yy / 47.0,
            0.25 + 0.25 * np.sin(xx / 3.0),
        ], axis=-1)
        cfg = ErrorSpentConfig(
            max_side=64, iterations=2, passes=2,
            total_cells=72, foundation_cells=12, allocation_batch=20)
        model = ErrorSpentDecomposition(image, cfg)
        self.assertEqual(len(model.seeds), 72)
        self.assertEqual(len(model.marks), 72)
        self.assertTrue(np.isfinite(model.reconstruction).all())
        self.assertGreater(model.reconstruction.std(), 0.08)
        self.assertLess(
            abs(float(model.reconstruction.mean()) - float(image.mean())),
            0.10)

    def test_more_error_spending_improves_fit(self):
        yy, xx = np.mgrid[0:56, 0:56]
        image = np.zeros((56, 56, 3), dtype=np.float64)
        image[..., 0] = (xx > 27)
        image[..., 1] = (yy > 27)
        image[..., 2] = ((xx // 4 + yy // 4) % 2)
        common = dict(
            max_side=56, iterations=2, passes=2,
            foundation_cells=12, allocation_batch=24)
        small = ErrorSpentDecomposition(
            image, ErrorSpentConfig(total_cells=48, **common))
        large = ErrorSpentDecomposition(
            image, ErrorSpentConfig(total_cells=120, **common))
        self.assertGreater(large.psnr, small.psnr)
        self.assertGreaterEqual(
            large.allocation_psnr[-1], large.allocation_psnr[0])

    def test_manual_round_adds_exactly_one_batch(self):
        yy, xx = np.mgrid[0:40, 0:52]
        image = np.stack([
            xx / 51.0, yy / 39.0, ((xx + yy) % 7) / 6.0,
        ], axis=-1)
        model = ErrorSpentDecomposition(
            image, ErrorSpentConfig(
                max_side=52, iterations=2, passes=2,
                foundation_cells=10, allocation_batch=13),
            initialize_only=True)
        before = model.psnr
        self.assertEqual(len(model.seeds), 10)
        self.assertEqual(model.step(), 13)
        self.assertEqual(len(model.seeds), 23)
        self.assertGreater(model.psnr, before)


if __name__ == "__main__":
    unittest.main()
