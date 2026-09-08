from __future__ import annotations

import unittest

import numpy as np

from .cleanup_shark_vad import SpeechState
from .state_phone_lattice import active_runs, bounded_lattice_spans, state_speech_crops


class StatePhoneLatticeTests(unittest.TestCase):
    def test_active_runs_are_half_open(self) -> None:
        mask = np.asarray((0, 1, 1, 0, 1, 0), dtype=bool)
        self.assertEqual(active_runs(mask), ((1, 3), (4, 5)))

    def test_state_crops_debounce_internal_landmarks(self) -> None:
        state = np.full(24, SpeechState.QUIET, dtype=np.int8)
        mask = np.zeros(24, dtype=bool)
        mask[3:20] = True
        state[3:8] = SpeechState.UNVOICED
        state[8:13] = SpeechState.VOICED
        state[13:20] = SpeechState.HANGOVER
        crops = state_speech_crops(state, mask)
        self.assertEqual(crops[0].landmarks, (3, 8, 13, 20))

    def test_bounded_spans_limit_phone_extent(self) -> None:
        spans = bounded_lattice_spans((10, 15, 20, 25, 30), maximum_compartments=2)
        self.assertEqual(
            spans,
            (
                (0, 1, 10, 15),
                (0, 2, 10, 20),
                (1, 2, 15, 20),
                (1, 3, 15, 25),
                (2, 3, 20, 25),
                (2, 4, 20, 30),
                (3, 4, 25, 30),
            ),
        )


if __name__ == "__main__":
    unittest.main()
