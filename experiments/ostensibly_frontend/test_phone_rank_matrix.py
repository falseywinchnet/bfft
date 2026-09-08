from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .phone_rank_matrix import (
    PhoneRankMatrix,
    complete_rank_vector,
    load_phone_rank_matrix,
    save_phone_rank_matrix,
)


class PhoneRankMatrixTests(unittest.TestCase):
    def test_complete_rank_vector_uses_canonical_label_order(self) -> None:
        ranking = (
            {"phone": "B", "rank": 1},
            {"phone": "A", "rank": 2},
            {"phone": "C", "rank": 3},
        )
        np.testing.assert_array_equal(
            complete_rank_vector(ranking, ("A", "B", "C")), (2, 1, 3)
        )

    def test_candidate_sequence_and_roundtrip_are_exact(self) -> None:
        matrix = PhoneRankMatrix(
            labels=np.asarray(("A", "B", "C")),
            reference_speakers=np.asarray(("ref", "ref")),
            query_speakers=np.asarray(("query", "query")),
            utterances=np.asarray(("u", "u")),
            ordinals=np.asarray((0, 1)),
            target_phones=np.asarray(("A", "B")),
            ranks=np.asarray(((1, 2, 3), (3, 1, 2)), dtype=np.uint8),
            provenance={"test": True},
        )
        self.assertEqual(
            matrix.candidate_ranks("ref", "query", "u", 0, ("B", "C")),
            (2, 2),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranks.npz"
            save_phone_rank_matrix(path, matrix)
            loaded = load_phone_rank_matrix(path)
        np.testing.assert_array_equal(loaded.ranks, matrix.ranks)
        self.assertEqual(loaded.provenance, matrix.provenance)


if __name__ == "__main__":
    unittest.main()

