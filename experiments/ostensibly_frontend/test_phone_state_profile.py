from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from .cleanup_shark_vad import SpeechState
from .phone_state_profile import (
    compile_phone_state_profile_atlas,
    load_phone_state_profile_atlas,
    phone_state_features,
)


class PhoneStateProfileTests(unittest.TestCase):
    def test_voiced_and_unvoiced_frames_form_different_profiles(self) -> None:
        count = 8
        voiced = phone_state_features(
            np.full(count, 0.8),
            np.full(count, 12.0),
            np.full(count, 0.8),
            np.full(count, 1.2),
            np.full(count, 0.3),
            np.full(count, 10.0),
            np.full(count, SpeechState.VOICED),
        )
        unvoiced = phone_state_features(
            np.full(count, 0.1),
            np.full(count, 4.0),
            np.full(count, 0.4),
            np.full(count, 0.2),
            np.full(count, 0.8),
            np.full(count, 2.0),
            np.full(count, SpeechState.UNVOICED),
        )
        self.assertGreater(voiced[0], unvoiced[0])
        self.assertGreater(voiced[7], unvoiced[7])
        self.assertGreater(unvoiced[8], voiced[8])

    def test_atlas_recovers_profile_class_and_round_trips(self) -> None:
        base = np.linspace(0.1, 0.9, 11)
        atlas = compile_phone_state_profile_atlas(
            (
                ("A", base),
                ("A", base + 0.01),
                ("B", base[::-1]),
                ("B", base[::-1] - 0.01),
            )
        )
        self.assertEqual(atlas.rank(base)[0]["phone"], "A")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas.npz"
            atlas.save(path)
            loaded = load_phone_state_profile_atlas(path)
            self.assertEqual(loaded.rank(base)[0]["phone"], "A")


if __name__ == "__main__":
    unittest.main()
