import unittest

import numpy as np

from .occurrence_statistics import (
    rank_occurrence_corrected_minimum,
    rank_occurrence_dominance,
    rank_occurrence_quantiles,
    rank_routed_occurrence_minimum,
    union_occurrence_proposals,
)


class OccurrenceStatisticsTests(unittest.TestCase):
    def test_zero_quantile_reproduces_per_label_minimum(self) -> None:
        ranking = rank_occurrence_quantiles(
            np.asarray(["A", "A", "B", "B"]),
            np.asarray(["a1", "a0", "b1", "b0"]),
            np.asarray([0.3, 0.1, 0.2, 0.4]),
            0.0,
        )
        self.assertEqual([row["phone"] for row in ranking], ["A", "B"])
        self.assertEqual([row["distance"] for row in ranking], [0.1, 0.2])
        self.assertEqual(ranking[0]["witness"], "a0")

    def test_median_removes_extra_minimum_lottery_ticket(self) -> None:
        ranking = rank_occurrence_quantiles(
            np.asarray(["COMMON", "COMMON", "COMMON", "RARE"]),
            np.asarray(["c0", "c1", "c2", "r0"]),
            np.asarray([0.01, 0.8, 0.9, 0.5]),
            0.5,
        )
        self.assertEqual([row["phone"] for row in ranking], ["RARE", "COMMON"])
        self.assertEqual(ranking[0]["occurrence_count"], 1)

    def test_rejects_invalid_quantile(self) -> None:
        with self.assertRaises(ValueError):
            rank_occurrence_quantiles(
                np.asarray(["A"]),
                np.asarray(["a"]),
                np.asarray([0.0]),
                1.1,
            )

    def test_union_preserves_primary_order_and_provenance(self) -> None:
        primary = [
            {"phone": "A", "rank": 1, "score": 0.1},
            {"phone": "B", "rank": 2, "score": 0.2},
            {"phone": "C", "rank": 3, "score": 0.3},
        ]
        balanced = [
            {"phone": "C", "rank": 1, "score": 0.05},
            {"phone": "A", "rank": 2, "score": 0.15},
            {"phone": "B", "rank": 3, "score": 0.25},
        ]
        union = union_occurrence_proposals(
            primary, balanced, minimum_depth=1, balanced_depth=1
        )
        self.assertEqual([row["phone"] for row in union], ["A", "C"])
        self.assertEqual(union[0]["geometry_proposal_channels"], [
            "geometry:occurrence_minimum"
        ])
        self.assertEqual(union[1]["geometry_proposal_channels"], [
            "geometry:balanced_tail"
        ])
        self.assertEqual(union[1]["balanced_tail_rank"], 1)

    def test_dominance_uses_complete_label_distribution(self) -> None:
        ranking = rank_occurrence_dominance(
            np.asarray(["A", "A", "B", "B", "C"]),
            np.asarray(["a0", "a1", "b0", "b1", "c0"]),
            np.asarray([0.1, 0.2, 0.4, 0.5, 0.9]),
        )
        self.assertEqual([row["phone"] for row in ranking], ["A", "B", "C"])
        self.assertAlmostEqual(ranking[0]["dominance_probability"], 1.0)
        self.assertAlmostEqual(ranking[-1]["dominance_probability"], 0.0)

    def test_corrected_minimum_penalizes_extra_null_chances(self) -> None:
        ranking = rank_occurrence_corrected_minimum(
            np.asarray(["A", "A", "B", "C", "D", "E"]),
            np.asarray(["a0", "a1", "b0", "c0", "d0", "e0"]),
            np.asarray([0.3, 0.9, 0.3, 0.4, 0.5, 0.6]),
        )
        by_phone = {row["phone"]: row for row in ranking}
        self.assertGreater(
            by_phone["A"]["corrected_minimum_probability"],
            by_phone["B"]["corrected_minimum_probability"],
        )

    def test_route_changes_eligibility_not_geometric_distance(self) -> None:
        labels = np.asarray(["A", "A", "B", "B"])
        witnesses = np.asarray(["a0", "a1", "b0", "b1"])
        distances = np.asarray([0.1, 0.8, 0.2, 0.7])
        routes = np.asarray([2.0, 0.0, 2.0, 0.0])
        ranking = rank_routed_occurrence_minimum(
            labels,
            witnesses,
            distances,
            routes,
            query_route_value=0.0,
            maximum_occurrences=1,
        )
        self.assertEqual([row["phone"] for row in ranking], ["B", "A"])
        self.assertEqual([row["distance"] for row in ranking], [0.7, 0.8])
        self.assertEqual(ranking[0]["routed_occurrence_count"], 1)


if __name__ == "__main__":
    unittest.main()
