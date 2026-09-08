from __future__ import annotations

import unittest

import numpy as np

from .articulatory_acoustics import (
    estimate_acoustic_trajectory,
    stable_vowel_span,
    waveguide_transfer_resonances,
)
from .articulatory_tract import tract_diameter_from_controls


class ArticulatoryAcousticsTests(unittest.TestCase):
    def test_static_waveguide_has_ordered_resonances(self) -> None:
        profile = tract_diameter_from_controls(17.7, 2.05, 0.82, 38.0, 1.45)
        formants = waveguide_transfer_resonances(profile[None, :], impulse_samples=2048)[0]
        self.assertTrue(np.all(np.isfinite(formants)))
        self.assertTrue(np.all(np.diff(formants) > 0.0))

    def test_stationary_tone_has_a_valid_stationary_span(self) -> None:
        sample_rate = 12_000
        time = np.arange(sample_rate) / sample_rate
        source = sum(
            amplitude * np.sin(2 * np.pi * frequency * time)
            for amplitude, frequency in ((1.0, 120.0), (0.5, 600.0), (0.3, 1200.0))
        )
        trajectory = estimate_acoustic_trajectory(
            source, sample_rate, hop_length=120, window_seconds=0.030
        )
        start, stop = stable_vowel_span(trajectory, window_frames=8)
        self.assertGreaterEqual(start, 0)
        self.assertEqual(stop - start, 8)


if __name__ == "__main__":
    unittest.main()
