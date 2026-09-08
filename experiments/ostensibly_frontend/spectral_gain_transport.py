"""State-aware transport laws for Cleanup frequency gains."""

from __future__ import annotations

import numpy as np

from .cleanup_shark_vad import SpeechState


def transported_spectral_gain(
    spectral_gain: np.ndarray,
    state: np.ndarray,
    *,
    gain_floor: float,
    mode: str,
) -> np.ndarray:
    """Apply a bounded gain globally or only on measured voiced frames."""

    gain = np.asarray(spectral_gain, dtype=np.float64)
    states = np.asarray(state, dtype=np.int8)
    if (
        gain.ndim != 2
        or gain.shape[0] != states.size
        or not gain.size
        or not np.all(np.isfinite(gain))
        or np.any(gain < 0.0)
        or not 0.0 <= gain_floor <= 1.0
        or mode not in {"global", "voiced-only"}
    ):
        raise ValueError("spectral gain transport configuration is invalid")
    bounded = gain_floor + (1.0 - gain_floor) * gain
    if mode == "global":
        return bounded
    output = np.ones_like(bounded)
    voiced = states == SpeechState.VOICED
    output[voiced] = bounded[voiced]
    return output
