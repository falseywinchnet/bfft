from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from .empirical_phone_compatibility import (
    EmpiricalPhoneCompatibility,
    combine_compatibility,
    load_empirical_phone_compatibility,
)


class EmpiricalPhoneCompatibilityTests(unittest.TestCase):
    def test_genuine_threshold_enrichment_recovers_phone(self) -> None:
        targets = np.asarray(("A", "A", "A", "B", "B", "B"))
        scores = np.asarray(
            (
                (0.05, 0.75),
                (0.10, 0.80),
                (0.15, 0.70),
                (0.70, 0.05),
                (0.80, 0.10),
                (0.75, 0.15),
            )
        )
        atlas = EmpiricalPhoneCompatibility(
            ("A", "B"), scores, targets, prior_strength=2.0
        )
        ranking = atlas.rank(np.asarray((0.12, 0.72)))
        self.assertEqual(ranking[0]["phone"], "A")
        self.assertGreater(ranking[0]["event_ratio"], 1.0)
        self.assertGreater(ranking[0]["compatibility"], ranking[1]["compatibility"])

    def test_bad_score_loses_conformal_support(self) -> None:
        targets = np.asarray(("A", "A", "B", "B"))
        scores = np.asarray(((0.1, 0.8), (0.2, 0.7), (0.7, 0.1), (0.8, 0.2)))
        atlas = EmpiricalPhoneCompatibility(("A", "B"), scores, targets)
        row = next(item for item in atlas.rank(np.asarray((0.95, 0.15))) if item["phone"] == "A")
        self.assertAlmostEqual(row["conformity"], 1.0 / 3.0)

    def test_round_trip_preserves_ranking(self) -> None:
        targets = np.asarray(("A", "A", "B", "B"))
        scores = np.asarray(((0.1, 0.8), (0.2, 0.7), (0.7, 0.1), (0.8, 0.2)))
        atlas = EmpiricalPhoneCompatibility(
            ("A", "B"), scores, targets, prior_strength=4.0, policy="minimum"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas.npz"
            atlas.save(path)
            loaded = load_empirical_phone_compatibility(path)
            self.assertEqual(
                [row["phone"] for row in atlas.rank(np.asarray((0.12, 0.75)))],
                [row["phone"] for row in loaded.rank(np.asarray((0.12, 0.75)))],
            )

    def test_vectorized_scores_match_detailed_rank(self) -> None:
        targets = np.asarray(("A", "A", "B", "B"))
        scores = np.asarray(((0.1, 0.8), (0.2, 0.7), (0.7, 0.1), (0.8, 0.2)))
        atlas = EmpiricalPhoneCompatibility(("A", "B"), scores, targets)
        query = np.asarray((0.12, 0.75))
        matrix = atlas.score_matrix(query[None, :])[0]
        detailed = {row["phone"]: row for row in atlas.rank(query)}
        np.testing.assert_allclose(
            matrix,
            [detailed[label]["compatibility"] for label in atlas.labels],
        )

    def test_combination_rejects_invalid_policy(self) -> None:
        with self.assertRaises(ValueError):
            combine_compatibility(0.5, 0.5, "invented")

    def test_unseen_phone_uses_pooled_support_without_false_certification(self) -> None:
        atlas = EmpiricalPhoneCompatibility(
            ("A", "B"),
            np.asarray(((0.1, 0.8), (0.2, 0.7))),
            np.asarray(("A", "A")),
        )
        row = next(item for item in atlas.rank(np.asarray((0.15, 0.4))) if item["phone"] == "B")
        self.assertEqual(row["positive_count"], 0)
        self.assertEqual(row["certified_event_ratio_lower_95"], 0.0)


if __name__ == "__main__":
    unittest.main()
