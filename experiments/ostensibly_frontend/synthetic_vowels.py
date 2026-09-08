"""Deterministic source/resonator vowel probes for a non-trained inventory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


# Approximate adult-English monophthong centers in hertz.  These are probe
# coordinates, not learned statistics and not claims about one speaker.
FORMANTS = {
    "IY": (270.0, 2290.0, 3010.0),
    "IH": (390.0, 1990.0, 2550.0),
    "EH": (530.0, 1840.0, 2480.0),
    "AE": (660.0, 1720.0, 2410.0),
    "AA": (730.0, 1090.0, 2440.0),
    "AO": (570.0, 840.0, 2410.0),
    "UH": (440.0, 1020.0, 2240.0),
    "UW": (300.0, 870.0, 2240.0),
    "AH": (640.0, 1190.0, 2390.0),
    "ER": (490.0, 1350.0, 1690.0),
}

DIPHTHONGS = {
    "EY": ("EH", "IY"),
    "AY": ("AA", "IY"),
    "OW": ("AO", "UW"),
    "AW": ("AA", "UW"),
    "OY": ("AO", "IY"),
}


@dataclass(frozen=True)
class VoiceProbe:
    name: str
    f0: float
    tract_scale: float
    breath: float
    jitter: float


VOICE_PROBES = (
    VoiceProbe("low_long", 92.0, 0.88, 0.015, 0.0020),
    VoiceProbe("low_mid", 112.0, 0.95, 0.012, 0.0015),
    VoiceProbe("mid", 135.0, 1.00, 0.010, 0.0010),
    VoiceProbe("high_mid", 165.0, 1.07, 0.014, 0.0015),
    VoiceProbe("high_short", 205.0, 1.14, 0.018, 0.0020),
)


def _resonator(frequency: float, bandwidth: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    radius = np.exp(-np.pi * bandwidth / sample_rate)
    angle = 2.0 * np.pi * frequency / sample_rate
    denominator = np.array(
        [1.0, -2.0 * radius * np.cos(angle), radius * radius],
        dtype=np.float64,
    )
    numerator = np.array([1.0 - radius], dtype=np.float64)
    return numerator, denominator


def synthesize_vowel(
    label: str,
    voice: VoiceProbe,
    *,
    sample_rate: int = 48_000,
    duration: float = 0.28,
    seed: int = 0,
) -> np.ndarray:
    """Generate one controlled vowel without a fitted or neural model."""
    if label not in FORMANTS:
        raise KeyError(label)
    rng = np.random.default_rng(seed)
    count = int(round(duration * sample_rate))
    impulses = np.zeros(count, dtype=np.float64)
    position = 0.0
    while int(position) < count:
        impulses[int(position)] = 1.0
        perturbation = 1.0 + voice.jitter * rng.normal()
        position += sample_rate / max(voice.f0 * perturbation, 1.0)
    excitation = impulses + voice.breath * rng.normal(size=count)
    output = excitation
    bandwidths = (70.0, 95.0, 130.0)
    for frequency, bandwidth in zip(FORMANTS[label], bandwidths):
        shifted = min(frequency * voice.tract_scale, 0.46 * sample_rate)
        numerator, denominator = _resonator(
            shifted,
            bandwidth * np.sqrt(voice.tract_scale),
            sample_rate,
        )
        output = signal.lfilter(numerator, denominator, output)
    # Lip-radiation difference and a smooth phone-length envelope.
    output = signal.lfilter([1.0, -0.96], [1.0], output)
    edge = min(int(0.035 * sample_rate), count // 3)
    envelope = np.ones(count, dtype=np.float64)
    if edge:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, edge)) ** 2
        envelope[:edge] = ramp
        envelope[-edge:] = ramp[::-1]
    output *= envelope
    peak = max(float(np.max(np.abs(output))), 1e-30)
    return output / peak


def synthesize_english_vowel(
    label: str,
    voice: VoiceProbe,
    *,
    sample_rate: int = 48_000,
    duration: float = 0.28,
    seed: int = 0,
) -> np.ndarray:
    """Generate a monophthong or a two-target diphthong probe."""
    if label in FORMANTS:
        return synthesize_vowel(
            label,
            voice,
            sample_rate=sample_rate,
            duration=duration,
            seed=seed,
        )
    if label not in DIPHTHONGS:
        raise KeyError(label)
    first, second = DIPHTHONGS[label]
    left = synthesize_vowel(
        first,
        voice,
        sample_rate=sample_rate,
        duration=duration,
        seed=seed,
    )
    right = synthesize_vowel(
        second,
        voice,
        sample_rate=sample_rate,
        duration=duration,
        seed=seed + 1,
    )
    mix = np.linspace(0.0, 1.0, left.size, dtype=np.float64)
    output = (1.0 - mix) * left + mix * right
    return output / max(float(np.max(np.abs(output))), 1e-30)


def apply_ssb_probe_channel(
    samples: np.ndarray,
    *,
    sample_rate: int = 48_000,
    low_hz: float = 250.0,
    high_hz: float = 3_000.0,
    noise_db: float = 28.0,
    seed: int = 0,
) -> np.ndarray:
    """Apply only the declared radio passband and reproducible channel noise."""
    sos = signal.butter(
        6,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, np.asarray(samples, dtype=np.float64))
    rng = np.random.default_rng(seed)
    noise = signal.sosfiltfilt(sos, rng.normal(size=filtered.size))
    signal_power = max(float(np.mean(filtered * filtered)), 1e-30)
    noise_power = max(float(np.mean(noise * noise)), 1e-30)
    noise *= np.sqrt(signal_power / (10.0 ** (noise_db / 10.0) * noise_power))
    return filtered + noise
