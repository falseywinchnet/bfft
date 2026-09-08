from __future__ import annotations

import unittest

import numpy as np

from .cleanup_shark_vad import SpeechState
from .spectral_gain_transport import transported_spectral_gain


class SpectralGainTransportTests(unittest.TestCase):
    def test_voiced_only_preserves_every_nonvoiced_coefficient(self) -> None:
        gain = np.asarray(((0.0, 0.5), (0.2, 0.6), (0.4, 0.8)))
        state = np.asarray(
            (SpeechState.UNVOICED, SpeechState.VOICED, SpeechState.HANGOVER)
        )
        result = transported_spectral_gain(
            gain, state, gain_floor=0.5, mode="voiced-only"
        )
        np.testing.assert_array_equal(result[0], 1.0)
        np.testing.assert_allclose(result[1], 0.5 + 0.5 * gain[1])
        np.testing.assert_array_equal(result[2], 1.0)

    def test_global_mode_is_the_declared_affine_floor(self) -> None:
        gain = np.asarray(((0.0, 0.5), (0.2, 1.0)))
        state = np.asarray((SpeechState.QUIET, SpeechState.VOICED))
        np.testing.assert_allclose(
            transported_spectral_gain(gain, state, gain_floor=0.25, mode="global"),
            0.25 + 0.75 * gain,
        )


if __name__ == "__main__":
    unittest.main()
