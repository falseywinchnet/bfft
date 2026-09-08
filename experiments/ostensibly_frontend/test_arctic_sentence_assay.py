from __future__ import annotations

import unittest

from .run_arctic_sentence_assay import (
    normalized_distance_map,
    rank_summary,
    target_class_rank,
)
from .word_lattice import PronunciationProposalIndex, Pronunciation


class ArcticSentenceAssayTests(unittest.TestCase):
    def test_rank_summary_retains_missing_targets(self) -> None:
        self.assertEqual(
            rank_summary([1, 4, None]),
            {"count": 3, "finite": 2, "top1": 1, "top5": 2, "top20": 2, "median_rank": 2.5},
        )

    def test_target_rank_requires_word_and_pronunciation(self) -> None:
        index = PronunciationProposalIndex(
            (Pronunciation("one", ("W",)), Pronunciation("two", ("T",)))
        )
        ordinals = {item.phones: ordinal for ordinal, item in enumerate(index.classes)}
        rows = [
            (0, 0, ("T",), ("two",), ordinals[("T",)]),
            (0, 0, ("W",), ("one",), ordinals[("W",)]),
        ]
        self.assertEqual(target_class_rank(rows, index, "one", ("W",)), 2)

    def test_normalized_distance_map_preserves_relative_margin(self) -> None:
        values = normalized_distance_map(
            [
                {"phone": "A", "distance": 2.0},
                {"phone": "B", "distance": 3.0},
                {"phone": "C", "distance": 6.0},
            ]
        )
        self.assertEqual(values["A"], 0.0)
        self.assertEqual(values["B"], 1.0)
        self.assertEqual(values["C"], 4.0)


if __name__ == "__main__":
    unittest.main()
