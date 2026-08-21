"""Invariant tests for exact eikonal ancestry transport."""

from __future__ import annotations

import unittest

import numpy as np

from .causal_ancestry import (
    shared_label_continuous_causal_forest,
    transport_causal_ancestry,
)
from port_needed.continuous_eikonal_transport import (
    continuous_first_partition,
    prepare_continuous_metric,
)


class CausalAncestryTests(unittest.TestCase):
    def test_chain_preserves_one_root_exactly(self):
        result = transport_causal_ancestry(
            np.array((-1, 0, 1, 2)),
            np.full(4, -1),
            np.zeros(4),
            np.arange(4),
            np.array((0, -1, -1, -1)),
        )
        np.testing.assert_array_equal(result.weights, np.ones((4, 1)))
        np.testing.assert_array_equal(result.collision_population, np.ones(4))

    def test_simplex_fraction_transports_source_law(self):
        result = transport_causal_ancestry(
            np.array((-1, -1, 0, 2)),
            np.array((-1, -1, 1, -1)),
            np.array((0.0, 0.0, 0.25, 0.0)),
            np.arange(4),
            np.array((0, 1, -1, -1)),
        )
        np.testing.assert_allclose(result.weights[2], (0.75, 0.25))
        np.testing.assert_allclose(result.weights[3], (0.75, 0.25))
        self.assertAlmostEqual(result.collision_population[2], 1.6)

    def test_overlapping_parents_are_not_counted_as_independent(self):
        # Node 3 mixes the already mixed node 2 with source 0. A scalar parent
        # count would miss their overlap; explicit ancestry gives (7/8, 1/8).
        result = transport_causal_ancestry(
            np.array((-1, -1, 0, 2)),
            np.array((-1, -1, 1, 0)),
            np.array((0.0, 0.0, 0.25, 0.5)),
            np.arange(4),
            np.array((0, 1, -1, -1)),
        )
        np.testing.assert_allclose(result.weights[3], (0.875, 0.125))
        self.assertAlmostEqual(
            result.collision_population[3], 1.0 / (0.875**2 + 0.125**2))

    def test_rejects_noncausal_order(self):
        with self.assertRaisesRegex(ValueError, "precede"):
            transport_causal_ancestry(
                np.array((-1, 0)),
                np.array((-1, -1)),
                np.zeros(2),
                np.array((1, 0)),
                np.array((0, -1)),
            )

    def test_consumes_the_actual_v3_eikonal_parent_stream(self):
        shape = (9, 11)
        metric_xx = np.ones(shape)
        metric_xy = np.zeros(shape)
        metric_yy = np.ones(shape)
        forest = continuous_first_partition(
            np.array(((0.2, 0.5), (0.8, 0.5))),
            metric_xx,
            metric_xy,
            metric_yy,
        )
        first = forest["parent_first"]
        labels = forest["labels"]
        roots = np.where(first < 0, labels, -1)
        result = transport_causal_ancestry(
            first,
            forest["parent_second"],
            forest["parent_fraction"],
            forest["acceptance_order"],
            roots,
        )
        # The production first-arrival solver only forms simplices between
        # equal owners, so this two-owner trace must remain exactly pure.
        np.testing.assert_array_equal(
            np.argmax(result.weights, axis=-1), labels)
        np.testing.assert_allclose(result.collision_population, 1.0)

    def test_continuous_shared_label_forest_conserves_distinct_germs(self):
        shape = (13, 19)
        ones = np.ones(shape, dtype=np.float64)
        zeros = np.zeros(shape, dtype=np.float64)
        prepared = prepare_continuous_metric(
            ones, zeros, ones, consistency_limit=np.finfo(float).max)
        centers = np.array([[0.24, 0.45], [0.76, 0.55]], dtype=np.float64)
        forest, ancestry = shared_label_continuous_causal_forest(
            centers, prepared)
        self.assertTrue(forest["continuous_germs"])
        self.assertEqual(ancestry.source_count, 2)
        np.testing.assert_allclose(
            np.sum(ancestry.weights, axis=-1), 1.0,
            atol=8.0 * np.finfo(float).eps,
            rtol=8.0 * np.finfo(float).eps,
        )
        self.assertGreaterEqual(
            float(np.max(ancestry.collision_population)), 1.0)
        represented = set(
            np.unique(forest["root_identity"])[1:].tolist())
        self.assertEqual(represented, {0, 1})


if __name__ == "__main__":
    unittest.main()
