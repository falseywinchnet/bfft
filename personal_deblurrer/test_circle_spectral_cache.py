from __future__ import annotations

import unittest

import numpy as np

from denoiser.run_2d_denoiser_battery import sources

from .circles import (
    phase_circle_translation,
    phase_circle_translation_from_spectra,
    prepare_phase_circle_spectrum,
)


class PhaseCircleSpectralCacheTests(unittest.TestCase):
    def test_prepared_spectrum_is_exactly_the_direct_estimator(self) -> None:
        first = sources(64)["cameraman"]
        second = np.roll(np.roll(first, 3, axis=0), -2, axis=1)
        direct_vector, direct_record = phase_circle_translation(first, second)
        cached_vector, cached_record = phase_circle_translation_from_spectra(
            prepare_phase_circle_spectrum(first),
            prepare_phase_circle_spectrum(second),
        )
        np.testing.assert_array_equal(cached_vector, direct_vector)
        self.assertEqual(cached_record, direct_record)


if __name__ == "__main__":
    unittest.main()
