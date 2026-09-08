"""Deterministic acoustic probes for English consonant candidate classes."""

from __future__ import annotations

import numpy as np
from scipy import signal

from experiments.ostensibly_frontend.synthetic_vowels import (
    VoiceProbe,
    _resonator,
    synthesize_vowel,
)


CONSONANTS = (
    "P", "B", "T", "D", "K", "G",
    "CH", "JH",
    "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH",
    "M", "N", "NG",
    "L", "R", "W", "Y",
)


FRICATIVE_BANDS = {
    "F": (450.0, 2600.0),
    "V": (450.0, 2600.0),
    "TH": (700.0, 3000.0),
    "DH": (700.0, 3000.0),
    "S": (2200.0, 3900.0),
    "Z": (2200.0, 3900.0),
    "SH": (1500.0, 3200.0),
    "ZH": (1500.0, 3200.0),
    "HH": (300.0, 3300.0),
}

VOICED_FRICATIVES = frozenset(("V", "DH", "Z", "ZH"))
STOP_BANDS = {
    "P": (450.0, 1400.0),
    "B": (450.0, 1400.0),
    "T": (2200.0, 3900.0),
    "D": (2200.0, 3900.0),
    "K": (1200.0, 2600.0),
    "G": (1200.0, 2600.0),
}
VOICED_STOPS = frozenset(("B", "D", "G"))
SONORANT_FORMANTS = {
    "M": (260.0, 900.0, 2200.0),
    "N": (280.0, 1450.0, 2400.0),
    "NG": (300.0, 1900.0, 2550.0),
    "L": (360.0, 1100.0, 2600.0),
    "R": (460.0, 1300.0, 1700.0),
    "W": (300.0, 650.0, 2200.0),
    "Y": (280.0, 2150.0, 2950.0),
}


def _band_noise(
    count: int,
    sample_rate: int,
    band: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    high = min(band[1], 0.47 * sample_rate)
    low = min(band[0], high * 0.8)
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    output = signal.sosfilt(sos, rng.normal(size=count))
    return output / max(float(np.max(np.abs(output))), 1e-30)


def _voiced_resonance(
    formants: tuple[float, float, float],
    voice: VoiceProbe,
    *,
    count: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    impulses = np.zeros(count, dtype=np.float64)
    position = 0.0
    while int(position) < count:
        impulses[int(position)] = 1.0
        position += sample_rate / max(
            voice.f0 * (1.0 + voice.jitter * rng.normal()), 1.0
        )
    output = impulses + voice.breath * rng.normal(size=count)
    for frequency, bandwidth in zip(formants, (65.0, 100.0, 145.0)):
        numerator, denominator = _resonator(
            frequency * voice.tract_scale,
            bandwidth,
            sample_rate,
        )
        output = signal.lfilter(numerator, denominator, output)
    output = signal.lfilter([1.0, -0.94], [1.0], output)
    return output / max(float(np.max(np.abs(output))), 1e-30)


def _smooth_envelope(count: int, sample_rate: int, edge_seconds: float = 0.02) -> np.ndarray:
    edge = min(int(round(edge_seconds * sample_rate)), count // 3)
    envelope = np.ones(count, dtype=np.float64)
    if edge:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, edge)) ** 2
        envelope[:edge] = ramp
        envelope[-edge:] = ramp[::-1]
    return envelope


def synthesize_consonant(
    label: str,
    voice: VoiceProbe,
    *,
    sample_rate: int = 48_000,
    duration: float = 0.22,
    seed: int = 0,
) -> np.ndarray:
    """Generate one exact-label consonant probe from declared mechanics."""
    if label not in CONSONANTS:
        raise KeyError(label)
    rng = np.random.default_rng(seed)
    count = int(round(duration * sample_rate))

    if label in FRICATIVE_BANDS:
        output = _band_noise(count, sample_rate, FRICATIVE_BANDS[label], rng)
        if label == "HH":
            vowel = synthesize_vowel(
                "AH", voice, sample_rate=sample_rate, duration=duration, seed=seed + 7
            )
            output *= 0.45 + 0.55 * np.abs(vowel)
        elif label in VOICED_FRICATIVES:
            voice_bar = _voiced_resonance(
                (220.0, 850.0, 2100.0),
                voice,
                count=count,
                sample_rate=sample_rate,
                rng=rng,
            )
            output = 0.68 * output + 0.32 * voice_bar
        output *= _smooth_envelope(count, sample_rate)

    elif label in STOP_BANDS:
        closure = min(
            int(round(0.055 * sample_rate)),
            max(1, int(round(0.40 * count))),
        )
        burst = min(
            int(round(0.025 * sample_rate)),
            max(1, int(round(0.20 * count))),
            count - closure,
        )
        output = np.zeros(count, dtype=np.float64)
        if label in VOICED_STOPS:
            output[:closure] = 0.10 * _voiced_resonance(
                (180.0, 700.0, 1800.0),
                voice,
                count=closure,
                sample_rate=sample_rate,
                rng=rng,
            )
        output[closure : closure + burst] += _band_noise(
            burst, sample_rate, STOP_BANDS[label], rng
        )
        release_count = count - closure - burst
        if release_count:
            release = synthesize_vowel(
                "AH",
                voice,
                sample_rate=sample_rate,
                duration=release_count / sample_rate,
                seed=seed + 11,
            )
            output[closure + burst :] += 0.55 * release

    elif label in ("CH", "JH"):
        closure = min(
            int(round(0.04 * sample_rate)),
            max(1, int(round(0.35 * count))),
        )
        output = np.zeros(count, dtype=np.float64)
        output[closure:] = _band_noise(
            count - closure, sample_rate, FRICATIVE_BANDS["SH"], rng
        )
        if label == "JH":
            output += 0.28 * _voiced_resonance(
                (230.0, 900.0, 2100.0),
                voice,
                count=count,
                sample_rate=sample_rate,
                rng=rng,
            )
        output *= _smooth_envelope(count, sample_rate)

    else:
        output = _voiced_resonance(
            SONORANT_FORMANTS[label],
            voice,
            count=count,
            sample_rate=sample_rate,
            rng=rng,
        )
        if label in ("M", "N", "NG"):
            notch_frequency = {"M": 1250.0, "N": 1800.0, "NG": 2300.0}[label]
            b, a = signal.iirnotch(
                notch_frequency * voice.tract_scale,
                5.0,
                fs=sample_rate,
            )
            output = signal.lfilter(b, a, output)
        output *= _smooth_envelope(count, sample_rate)

    return output / max(float(np.max(np.abs(output))), 1e-30)
