from __future__ import annotations

import math
import unittest

import numpy as np

from .native_retained import NativeRetainedPlan, native_available


@unittest.skipUnless(native_available(), "native retained-rank library is not built")
class NativeRetainedKernelTests(unittest.TestCase):
    def test_contiguous_and_indexed_blocks_match_numpy_outer_products(self) -> None:
        rng = np.random.default_rng(1234)
        nodes = 17
        modes = 7
        block_data = (
            (
                np.arange(0, 8, dtype=np.uint32),
                rng.random(8),
                np.arange(9, 17, dtype=np.uint32),
                rng.random(8) * 0.08,
                0.42,
                0.2,
                0.7,
                0.1,
            ),
            (
                np.array((0, 3, 5, 8), dtype=np.uint32),
                rng.random(4),
                np.array((10, 12, 15), dtype=np.uint32),
                rng.random(3) * 0.05,
                0.23,
                0.4,
                0.2,
                0.3,
            ),
        )
        power = rng.normal(size=(modes, nodes))
        power[:, 9:] = 0.0
        linear = rng.random(modes) * 0.8
        circular = rng.uniform(-0.5, 0.5, size=modes)
        active = np.any(power != 0.0, axis=0).astype(np.uint8)
        plan = NativeRetainedPlan(nodes, block_data)
        actual, stats = plan.apply(power, active, linear, circular)
        bound = plan.bind_modes(linear, circular)
        bound_actual, bound_stats = bound.apply(power, active)
        expected = np.zeros_like(power)
        for source, u, receiver, v, depth, extinction, linear_loss, circular_loss in block_data:
            gathered = np.sum(power[:, source] * u[None, :], axis=1)
            retention = np.exp(
                -depth
                * (extinction + linear_loss * linear + circular_loss * np.abs(circular))
            )
            expected[:, receiver] += (gathered * retention)[:, None] * v[None, :]
        np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)
        np.testing.assert_array_equal(bound_actual, actual)
        self.assertEqual(bound_stats, stats)
        self.assertEqual(stats.block_applications, 2)
        self.assertEqual(stats.contiguous_source_blocks, 1)
        self.assertEqual(stats.contiguous_receiver_blocks, 1)

    def test_signed_cancellation_and_inactive_blocks_are_exact(self) -> None:
        plan = NativeRetainedPlan(
            5,
            (
                (
                    np.array((0, 1), dtype=np.uint32),
                    np.array((0.5, 0.5)),
                    np.array((3,), dtype=np.uint32),
                    np.array((0.7,)),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                (
                    np.array((2,), dtype=np.uint32),
                    np.array((1.0,)),
                    np.array((4,), dtype=np.uint32),
                    np.array((0.8,)),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
            ),
        )
        power = np.array(((1.0, -1.0, 0.0, 0.0, 0.0),))
        output, stats = plan.apply(
            power,
            np.array((1, 1, 0, 0, 0), dtype=np.uint8),
            np.zeros(1),
            np.zeros(1),
        )
        self.assertTrue(np.array_equal(output, np.zeros_like(power)))
        self.assertEqual(stats.block_applications, 0)
        self.assertEqual(stats.gathered_source_coefficients, 2)


if __name__ == "__main__":
    unittest.main()
