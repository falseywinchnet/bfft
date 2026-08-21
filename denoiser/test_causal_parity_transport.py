"""Invariants for the shared-label pre-ownership parity experiment."""

from __future__ import annotations

import unittest

import numpy as np

from .causal_ancestry import shared_label_causal_forest
from .causal_parity_transport import (
    denoise_causal_parity_transport,
    parity_root_pixels,
)
from port_needed.continuous_eikonal_transport import prepare_continuous_metric


class CausalParityTransportTests(unittest.TestCase):
    def test_parity_roots_are_disjoint_and_complete(self):
        first = parity_root_pixels((9, 12), 0)
        second = parity_root_pixels((9, 12), 1)
        self.assertEqual(np.intersect1d(first, second).size, 0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate((first, second))), np.arange(9 * 12))

    def test_shared_label_front_has_nontrivial_ancestry(self):
        shape = (9, 11)
        prepared = prepare_continuous_metric(
            np.ones(shape), np.zeros(shape), np.ones(shape),
            consistency_limit=np.finfo(float).max)
        roots = parity_root_pixels(shape, 0)
        forest, ancestry = shared_label_causal_forest(roots, prepared)
        np.testing.assert_array_equal(forest["labels"], 0)
        self.assertGreater(float(np.max(ancestry.collision_population)), 1.0)

    def test_constant_field_is_exact_and_population_is_physical(self):
        field = np.full((12, 14), 0.43)
        estimate, diagnostic = denoise_causal_parity_transport(field)
        np.testing.assert_allclose(estimate, field, atol=1e-15, rtol=0.0)
        self.assertGreaterEqual(
            float(np.min(diagnostic["collision_population"])), 1.0)
        self.assertLessEqual(
            float(np.max(diagnostic["collision_population"])),
            max(field.shape) ** 2,
        )

    def test_memory_ceiling_is_only_a_visible_guard(self):
        with self.assertRaisesRegex(MemoryError, "memory ceiling"):
            denoise_causal_parity_transport(
                np.zeros((12, 14)), memory_ceiling_bytes=1)


if __name__ == "__main__":
    unittest.main()
