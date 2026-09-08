from __future__ import annotations

import unittest

import numpy as np

from .segmental_pronunciation_geometry import (
    PhoneSpanCost,
    best_contiguous_partition,
    debounced_state_landmarks,
    enclosing_speech_crop,
    score_to_dict,
    support_landmarks_within_crop,
)
from .cleanup_shark_vad import SpeechState


def _cost(distance: float, rank: int = 1) -> PhoneSpanCost:
    return PhoneSpanCost(distance, rank, "witness")


class SegmentalPronunciationGeometryTests(unittest.TestCase):
    def test_bottleneck_objective_recovers_the_only_uniformly_good_partition(self) -> None:
        phones = ("OW", "K", "EY")
        costs = {}
        for start in range(6):
            for stop in range(start + 1, 7):
                for phone in phones:
                    costs[(start, stop, phone)] = _cost(0.9, 20)
        costs[(0, 2, "OW")] = _cost(0.10, 2)
        costs[(2, 4, "K")] = _cost(0.20, 3)
        costs[(4, 6, "EY")] = _cost(0.15, 1)

        result = best_contiguous_partition(phones, 6, costs)

        self.assertEqual(result.cuts, (2, 4, 6))
        self.assertAlmostEqual(result.worst_distance, 0.20)
        self.assertAlmostEqual(result.mean_distance, 0.15)
        self.assertEqual(
            [row["region_span"] for row in score_to_dict(result)["assignments"]],
            [[0, 2], [2, 4], [4, 6]],
        )

    def test_worst_phone_prevents_one_excellent_phone_hiding_a_failure(self) -> None:
        phones = ("A", "B")
        costs = {
            (0, 1, "A"): _cost(0.01),
            (1, 3, "B"): _cost(0.60),
            (0, 2, "A"): _cost(0.31),
            (2, 3, "B"): _cost(0.31),
        }
        result = best_contiguous_partition(phones, 3, costs)
        self.assertEqual(result.cuts, (2, 3))
        self.assertAlmostEqual(result.worst_distance, 0.31)

    def test_within_span_rank_calibrates_incommensurate_distance_floors(self) -> None:
        phones = ("A", "B")
        costs = {
            (0, 1, "A"): _cost(0.50, 1),
            (1, 3, "B"): _cost(0.50, 1),
            (0, 2, "A"): _cost(0.10, 20),
            (2, 3, "B"): _cost(0.10, 20),
        }
        result = best_contiguous_partition(phones, 3, costs)
        self.assertEqual(result.cuts, (1, 3))
        self.assertEqual(result.worst_rank, 1)

    def test_rejects_more_candidate_phones_than_observed_regions(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer observed regions"):
            best_contiguous_partition(("A", "B", "C"), 2, {})

    def test_score_serializes_rank_coverage_without_changing_objective(self) -> None:
        covered = PhoneSpanCost(0.2, 3, "witness", 0.1, 0.7, 0.6)
        score = best_contiguous_partition(("A",), 1, {(0, 1, "A"): covered})
        assignment = score_to_dict(score)["assignments"][0]
        self.assertEqual(assignment["rank"], 3)
        self.assertAlmostEqual(assignment["cumulative_target_coverage_lower_95"], 0.6)

    def test_selects_crop_by_overlap_not_nearest_endpoint(self) -> None:
        mask = np.zeros(40, dtype=bool)
        mask[2:8] = True
        mask[14:31] = True
        self.assertEqual(enclosing_speech_crop(mask, 18, 25), (14, 31))

    def test_state_landmarks_remove_pitch_flicker_but_keep_closure(self) -> None:
        state = np.full(40, SpeechState.QUIET, dtype=np.int8)
        mask = np.zeros(40, dtype=bool)
        mask[5:34] = True
        state[5:12] = SpeechState.UNVOICED
        state[12:16] = SpeechState.VOICED
        state[16:20] = SpeechState.HANGOVER
        state[20:26] = SpeechState.UNVOICED
        state[22] = SpeechState.VOICED
        state[26:32] = SpeechState.VOICED
        state[29] = SpeechState.UNVOICED
        state[32:34] = SpeechState.HANGOVER

        landmarks, audit = debounced_state_landmarks(state, mask, 7, 31)

        self.assertEqual(landmarks, (5, 12, 16, 20, 26, 34))
        self.assertEqual(
            [row["state"] for row in audit],
            ["unvoiced", "voiced", "hangover", "unvoiced", "voiced"],
        )

    def test_support_landmarks_represent_both_edges_of_a_gap(self) -> None:
        landmarks = support_landmarks_within_crop(
            np.asarray([147, 151, 169]),
            np.asarray([151, 157, 173]),
            146,
            180,
        )
        self.assertEqual(landmarks, (146, 151, 157, 169, 173, 180))


if __name__ == "__main__":
    unittest.main()
