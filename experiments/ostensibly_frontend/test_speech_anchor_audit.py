from __future__ import annotations

import unittest

import numpy as np

from .speech_anchor_audit import (
    LabSegment,
    binary_counts,
    broad_phone_state,
    match_boundaries,
    summarize_counts,
    target_state_geometry,
    wilson_lower_bound,
)


class SpeechAnchorAuditTests(unittest.TestCase):
    def test_broad_phone_states_separate_silence_and_unvoiced_phones(self) -> None:
        self.assertEqual(broad_phone_state("pau"), "silence")
        self.assertEqual(broad_phone_state("TH"), "unvoiced")
        self.assertEqual(broad_phone_state("z"), "voiced")
        self.assertEqual(broad_phone_state("iy"), "voiced")

    def test_target_geometry_keeps_only_state_changing_boundaries(self) -> None:
        segments = (
            LabSegment(0.0, 0.1, "silence", "pau"),
            LabSegment(0.1, 0.2, "voiced", "aa"),
            LabSegment(0.2, 0.3, "voiced", "n"),
            LabSegment(0.3, 0.4, "unvoiced", "t"),
            LabSegment(0.4, 0.5, "silence", "pau"),
        )
        mask, speech, manner = target_state_geometry(
            segments, 6, sample_rate=1000, hop_length=100
        )
        np.testing.assert_array_equal(mask, (False, True, True, True, False, False))
        self.assertEqual(speech, (1, 4))
        self.assertEqual(manner, (3,))

    def test_ordered_boundary_matching_is_one_to_one(self) -> None:
        matched, errors = match_boundaries((9, 12, 31), (10, 30), tolerance_frames=2)
        self.assertEqual(matched, 2)
        self.assertEqual(errors, (1, 1))

    def test_binary_summary_is_additive_and_exact(self) -> None:
        counts = binary_counts(
            np.asarray((False, True, True, False)),
            np.asarray((True, True, False, False)),
        )
        self.assertEqual(counts, {
            "true_positive": 1,
            "false_positive": 1,
            "false_negative": 1,
            "true_negative": 1,
        })
        summary = summarize_counts(counts)
        self.assertEqual(summary["precision"], 0.5)
        self.assertEqual(summary["recall"], 0.5)
        self.assertEqual(summary["f1"], 0.5)

    def test_wilson_lower_bound_is_conservative(self) -> None:
        self.assertLess(wilson_lower_bound(99, 100), 0.99)
        self.assertGreater(wilson_lower_bound(999, 1000), 0.99)



if __name__ == "__main__":
    unittest.main()
