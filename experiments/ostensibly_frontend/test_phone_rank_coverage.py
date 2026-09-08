from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .phone_rank_coverage import (
    fit_phone_rank_coverage,
    load_phone_rank_coverage,
)


class PhoneRankCoverageTests(unittest.TestCase):
    def test_empirical_coverage_and_annotation_preserve_order(self) -> None:
        coverage = fit_phone_rank_coverage((1, 1, 2, 3), 3)
        self.assertEqual(coverage.coverage(2)["successes"], 3)
        rows = coverage.annotate(
            [
                {"phone": "A", "rank": 1},
                {"phone": "B", "rank": 2},
                {"phone": "C", "rank": 3},
            ]
        )
        self.assertEqual([row["phone"] for row in rows], ["A", "B", "C"])
        self.assertAlmostEqual(rows[1]["cumulative_target_coverage"], 0.75)
        self.assertAlmostEqual(rows[0]["empirical_target_frequency_at_rank"], 0.5)

    def test_minimum_rank_uses_conservative_lower_bound(self) -> None:
        coverage = fit_phone_rank_coverage((1,) * 90 + (2,) * 10, 2)
        self.assertEqual(coverage.minimum_rank_for_lower_coverage(0.8), 1)
        self.assertEqual(coverage.minimum_rank_for_lower_coverage(0.95), 2)

    def test_round_trip(self) -> None:
        coverage = fit_phone_rank_coverage((1, 2, 2, 3), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.npz"
            coverage.save(path)
            loaded = load_phone_rank_coverage(path)
            self.assertEqual(loaded.counts.tolist(), coverage.counts.tolist())


if __name__ == "__main__":
    unittest.main()
