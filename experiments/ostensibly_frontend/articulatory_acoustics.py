"""Analytic acoustic coordinates for articulatory trajectory inversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, resample_poly

from .cleanup_shark_vad import normalized_pitch_periodicity


@dataclass(frozen=True)
class AcousticTrajectory:
    times_seconds: np.ndarray
    periodicity: np.ndarray
    f0_hz: np.ndarray
    formants_hz: np.ndarray
    register_semitones: np.ndarray


def _centered_frames(samples: np.ndarray, centers: np.ndarray, length: int) -> np.ndarray:
    left = length // 2
    right = length - left
    padded = np.pad(np.asarray(samples, dtype=np.float64), (left, right))
    return np.lib.stride_tricks.sliding_window_view(padded, length)[centers]


def _lpc_formants(frame: np.ndarray, sample_rate: int, order: int) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float64)
    values = values - np.mean(values)
    values = np.r_[values[0], values[1:] - 0.97 * values[:-1]] * np.hamming(values.size)
    correlation = np.correlate(values, values, mode="full")[values.size - 1 : values.size + order]
    if correlation[0] <= 1e-12:
        return np.full(3, np.nan)
    try:
        coefficients = solve_toeplitz(
            (correlation[:order], correlation[:order]), -correlation[1 : order + 1]
        )
    except np.linalg.LinAlgError:
        return np.full(3, np.nan)
    roots = np.roots(np.r_[1.0, coefficients])
    roots = roots[np.imag(roots) >= 0.0]
    frequency = np.angle(roots) * sample_rate / (2.0 * np.pi)
    bandwidth = -np.log(np.maximum(np.abs(roots), 1e-12)) * sample_rate / np.pi
    selected = np.sort(frequency[(frequency >= 120.0) & (frequency <= 4500.0) & (bandwidth < 700.0)])
    result = np.full(3, np.nan)
    result[: min(3, selected.size)] = selected[:3]
    return result


def estimate_acoustic_trajectory(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hop_length: int = 512,
    window_seconds: float = 0.030,
) -> AcousticTrajectory:
    """Estimate F0 and F1--F3 as relative-register acoustic coordinates."""

    waveform = np.asarray(samples, dtype=np.float64)
    frame_count = max(1, 1 + waveform.size // hop_length)
    centers = np.minimum(
        np.arange(frame_count, dtype=np.int64) * hop_length,
        max(waveform.size - 1, 0),
    )
    periodicity, f0 = normalized_pitch_periodicity(
        waveform,
        centers,
        sample_rate=sample_rate,
        window_seconds=0.025,
    )
    # LPC roots become dominated by irrelevant high-frequency structure at
    # 48 kHz.  Estimate the first three oral resonances on a 12 kHz analysis
    # copy while retaining pitch on the original waveform.
    formant_rate = min(sample_rate, 12_000)
    if formant_rate != sample_rate:
        divisor = sample_rate // np.gcd(sample_rate, formant_rate)
        multiplier = formant_rate // np.gcd(sample_rate, formant_rate)
        formant_waveform = resample_poly(waveform, multiplier, divisor)
        formant_centers = np.minimum(
            np.rint(centers * formant_rate / sample_rate).astype(np.int64),
            formant_waveform.size - 1,
        )
    else:
        formant_waveform = waveform
        formant_centers = centers
    length = max(256, int(round(window_seconds * formant_rate)))
    frames = _centered_frames(formant_waveform, formant_centers, length)
    order = max(10, 2 + formant_rate // 4000)
    formants = np.stack(
        [_lpc_formants(frame, formant_rate, order) for frame in frames]
    )
    for column in range(3):
        finite = np.isfinite(formants[:, column])
        if np.sum(finite) >= 2:
            formants[:, column] = np.interp(
                np.arange(frame_count), np.flatnonzero(finite), formants[finite, column]
            )
            formants[:, column] = median_filter(formants[:, column], size=3, mode="nearest")
    register = 12.0 * np.log2(
        np.maximum(formants, 1e-12) / np.maximum(f0[:, None], 1e-12)
    )
    register[periodicity < 0.40] = np.nan
    return AcousticTrajectory(
        times_seconds=centers / sample_rate,
        periodicity=periodicity,
        f0_hz=f0,
        formants_hz=formants,
        register_semitones=register,
    )


def stable_vowel_span(
    trajectory: AcousticTrajectory,
    *,
    window_frames: int = 10,
) -> tuple[int, int]:
    """Return the most stationary sustained voiced span in register space."""

    register = np.asarray(trajectory.register_semitones, dtype=np.float64)
    count = register.shape[0]
    width = min(max(3, window_frames), count)
    best = (np.inf, 0, width)
    for start in range(0, count - width + 1):
        stop = start + width
        local = register[start:stop]
        if (
            np.mean(trajectory.periodicity[start:stop] >= 0.55) < 0.8
            or np.mean(np.isfinite(local)) < 0.9
        ):
            continue
        center = np.nanmedian(local, axis=0)
        scatter = float(np.nanmean((local - center) ** 2))
        slope = float(np.nanmean(np.diff(local, axis=0) ** 2))
        score = scatter + 2.0 * slope
        if score < best[0]:
            best = (score, start, stop)
    if not np.isfinite(best[0]):
        kernel = np.ones(width, dtype=np.float64)
        support = np.convolve(trajectory.periodicity, kernel, mode="valid")
        start = int(np.argmax(support))
        return start, start + width
    return int(best[1]), int(best[2])


def waveguide_transfer_resonances(
    profiles: np.ndarray,
    *,
    sample_rate: int = 48_000,
    impulse_samples: int = 4096,
) -> np.ndarray:
    """Measure the first three resonances for a batch of static tract shapes."""

    diameter = np.asarray(profiles, dtype=np.float64)
    if diameter.ndim != 2 or diameter.shape[1] < 30:
        raise ValueError("static tract profiles need batch-by-section geometry")
    batch, sections = diameter.shape
    area = diameter * diameter
    reflection = (area[:, :-1] - area[:, 1:]) / np.maximum(
        area[:, :-1] + area[:, 1:], 1e-12
    )
    right = np.zeros((batch, sections))
    left = np.zeros((batch, sections))
    junction_right = np.zeros((batch, sections + 1))
    junction_left = np.zeros((batch, sections + 1))
    output = np.zeros((batch, impulse_samples))
    for sample_index in range(impulse_samples):
        excitation = 1.0 if sample_index == 0 else 0.0
        value = np.zeros(batch)
        for _ in range(2):
            junction_right[:, 0] = 0.75 * left[:, 0] + excitation
            junction_left[:, -1] = -0.85 * right[:, -1]
            wave = reflection * (right[:, :-1] + left[:, 1:])
            junction_right[:, 1:-1] = right[:, :-1] - wave
            junction_left[:, 1:-1] = left[:, 1:] + wave
            right[:] = 0.999 * junction_right[:, :-1]
            left[:] = 0.999 * junction_left[:, 1:]
            value += right[:, -1]
        output[:, sample_index] = value
    magnitude = np.abs(np.fft.rfft(output * np.hanning(impulse_samples), axis=1))
    frequencies = np.fft.rfftfreq(impulse_samples, 1.0 / sample_rate)
    result = np.full((batch, 3), np.nan)
    minimum_distance = max(1, int(round(140.0 * impulse_samples / sample_rate)))
    for index, row in enumerate(magnitude):
        db = 20.0 * np.log10(np.maximum(row, 1e-12))
        peaks, _ = find_peaks(db, distance=minimum_distance, prominence=2.0)
        candidates = frequencies[peaks]
        candidates = candidates[(candidates >= 120.0) & (candidates <= 4500.0)]
        result[index, : min(3, candidates.size)] = candidates[:3]
    return result
