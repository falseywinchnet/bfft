from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from .generic_word_template_bank import (
    barycenter_word_signature,
    combine_generic_word_template_banks,
    compile_generic_word_template_bank,
    load_generic_word_template_bank,
    save_generic_word_template_bank,
    word_signature_aggregation_scores,
    word_signature_distance,
)


class GenericWordTemplateBankTests(unittest.TestCase):
    def test_complete_realizations_round_trip(self) -> None:
        frame = np.linspace(0.0, 1.0, 64)
        low = np.column_stack((0.2 + frame, frame, np.ones_like(frame)))
        high = np.column_stack((2.0 + frame, frame, np.ones_like(frame)))
        bank = compile_generic_word_template_bank(
            (("A", "a", low), ("B", "b", high)), 32
        )
        signatures = bank.word_signatures(("A", "B"), 2, 16, 8)
        self.assertEqual(len(signatures), 2)
        self.assertEqual(signatures[0][0].shape, (16, 8))
        barycenter = barycenter_word_signature(signatures)
        self.assertTrue(
            np.allclose(
                barycenter[0],
                (signatures[0][0] + signatures[1][0]) / 2.0,
            )
        )
        self.assertAlmostEqual(float(np.sum(barycenter[1])), 1.0)
        scores = word_signature_aggregation_scores(
            signatures[0], signatures, maximum_shift=2
        )
        self.assertEqual(set(scores), {"minimum", "mean", "median", "barycenter"})
        self.assertAlmostEqual(scores["minimum"], 0.0)
        self.assertAlmostEqual(
            word_signature_distance(
                signatures[0], signatures, 2, aggregation="mean"
            ),
            scores["mean"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.npz"
            save_generic_word_template_bank(path, bank)
            loaded = load_generic_word_template_bank(path)
        self.assertEqual(loaded.points_per_occurrence, 32)
        self.assertEqual(set(loaded.labels.tolist()), {"A", "B"})

    def test_combination_preserves_occurrences(self) -> None:
        frame = np.linspace(0.0, 1.0, 32)
        cloud = np.column_stack((frame, frame, np.ones_like(frame)))
        left = compile_generic_word_template_bank((("A", "a", cloud),), 16)
        right = compile_generic_word_template_bank((("B", "b", cloud),), 16)
        combined = combine_generic_word_template_banks((left, right))
        self.assertEqual(combined.labels.tolist(), ["A", "B"])

    def test_duration_weights_change_phone_lane_boundaries(self) -> None:
        frame = np.linspace(0.0, 1.0, 64)
        first = np.column_stack((frame, frame, np.ones_like(frame)))
        second = np.column_stack((1.0 + frame, frame, np.ones_like(frame)))
        bank = compile_generic_word_template_bank(
            (("A", "a", first), ("B", "b", second)), 32
        )
        equal = bank.word_signatures(("A", "B"), 1, 16, 8)
        weighted = bank.word_signatures(
            ("A", "B"),
            1,
            16,
            8,
            duration_weights=(3.0, 1.0),
        )
        self.assertFalse(np.allclose(equal[0][0], weighted[0][0]))

    def test_saliency_power_reaches_complete_word_signature(self) -> None:
        frame = np.linspace(0.0, 1.0, 64)
        height = np.linspace(0.01, 1.0, 64)
        cloud = np.column_stack((frame, frame, height))
        bank = compile_generic_word_template_bank((("A", "a", cloud),), 64)
        uniform = bank.word_signatures(("A",), 1, 16, 8)
        weighted = bank.word_signatures(
            ("A",), 1, 16, 8, saliency_power=0.5
        )
        self.assertFalse(np.allclose(uniform[0][0], weighted[0][0]))

    def test_contextual_pool_prefers_two_supported_neighbors(self) -> None:
        frame = np.linspace(0.0, 1.0, 32)
        exact = np.column_stack((frame, frame, np.ones_like(frame)))
        wrong = exact.copy(); wrong[:, 0] += 10.0
        bank = compile_generic_word_template_bank(
            (
                ("A", "exact-a", "<edge>", "B", exact),
                ("A", "wrong-a", "X", "Y", wrong),
                ("B", "exact-b", "A", "<edge>", exact),
                ("B", "wrong-b", "X", "Y", wrong),
            ),
            16,
        )
        pools = bank.pools(("A", "B"), context_conditioned=True)
        self.assertEqual(tuple(pool.shape[0] for pool in pools), (1, 1))
        self.assertEqual(bank.context_depths(("A", "B")), (1, 1))
        self.assertLess(float(np.max(pools[0][:, 0])), 2.0)
        mixed = bank.word_signatures(
            ("A", "B"), 4, 16, 8, context_policy="mixed"
        )
        self.assertEqual(len(mixed), 4)


if __name__ == "__main__":
    unittest.main()
