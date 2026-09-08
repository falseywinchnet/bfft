from __future__ import annotations

import unittest

import numpy as np

from .oracle_word_geometry import (
    PronunciationCatalog,
    rank_pronunciation_catalog,
    target_catalog_rank,
)


class OracleWordGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = PronunciationCatalog(
            phones=(("A", "B"), ("A", "C"), ("D", "A")),
            words=(("ab",), ("ac",), ("da",)),
            columns=np.asarray(((0, 1), (0, 2), (3, 0))),
        )
        self.query = np.asarray(((1, 2, 4, 3), (3, 4, 2, 1)))

    def test_worst_then_sum_and_sum_then_worst_are_distinct(self) -> None:
        worst, ranks = rank_pronunciation_catalog(
            self.query, self.catalog, policy="worst_then_sum"
        )
        summed, _ = rank_pronunciation_catalog(
            self.query, self.catalog, policy="sum_then_worst"
        )
        np.testing.assert_array_equal(ranks, ((1, 4), (1, 2), (3, 3)))
        self.assertEqual(tuple(worst), (1, 2, 0))
        self.assertEqual(tuple(summed), (1, 0, 2))
        self.assertEqual(target_catalog_rank(worst, 0), 3)

    def test_learned_likelihood_can_value_nonmonotone_empirical_ranks(self) -> None:
        likelihood = np.asarray((0.0, 1.0, -2.0, 0.5, -1.0))
        order, _ = rank_pronunciation_catalog(
            self.query,
            self.catalog,
            policy="learned_rank_likelihood",
            rank_log_likelihood=likelihood,
        )
        self.assertEqual(tuple(order), (2, 0, 1))


if __name__ == "__main__":
    unittest.main()
