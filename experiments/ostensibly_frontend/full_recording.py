"""Full-recording Ostensibly texture, Meyer cartoon, and chunk proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

try:
    import librosa
except ModuleNotFoundError:  # The corrected SciPy/BFFT path does not need it.
    librosa = None
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

import bfft
from experiments.ostensibly_frontend.trace_geometry import ridge_score


@dataclass(frozen=True)
class TimeInterval:
    frame0: int
    frame1: int
    seconds0: float
    seconds1: float
    score: float


def exact_low_rows_from_stft(
    spectrum: np.ndarray,
    *,
    n_fft: int,
    crop_rows: int,
    column_block: int = 256,
) -> np.ndarray:
    """Blockwise exact double-IRFFT, retaining only requested low rows."""
    output = np.empty((crop_rows, spectrum.shape[1]), dtype=np.float64)
    for start in range(0, spectrum.shape[1], column_block):
        stop = min(start + column_block, spectrum.shape[1])
        recovered = np.fft.irfft(spectrum[:, start:stop], axis=0)
        if recovered.shape[0] != n_fft:
            raise RuntimeError("first inverse does not match n_fft")
        second = np.fft.irfft(recovered, axis=0)
        if second.shape[0] != 2 * (n_fft - 1):
            raise RuntimeError("unexpected default second-irfft length")
        output[:, start:stop] = np.abs(second[:crop_rows])
    return output


def two_population_threshold(values: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Deterministic two-centroid threshold for a one-dimensional score."""
    x = np.asarray(values, dtype=np.float64)
    centers = np.array(np.percentile(x, [25.0, 75.0]), dtype=np.float64)
    for _ in range(32):
        assignment = np.abs(x[:, None] - centers[None, :]).argmin(axis=1)
        updated = centers.copy()
        for index in range(2):
            selected = x[assignment == index]
            if selected.size:
                updated[index] = float(np.mean(selected))
        if np.allclose(updated, centers, rtol=0.0, atol=1e-12):
            break
        centers = updated
    centers.sort()
    return float(np.mean(centers)), (float(centers[0]), float(centers[1]))


def activity_intervals(
    cartoon: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    minimum_seconds: float = 0.08,
    close_gap_seconds: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, tuple[TimeInterval, ...], dict]:
    """Return a telecom-style persistent ridge activity mask and intervals."""
    evidence = ridge_score(cartoon)
    # A few high-valued rows carry a voiced ridge; the median is deliberately
    # ignored because most of this sparse field is background.
    top_count = max(2, evidence.shape[0] // 32)
    top = np.partition(evidence, -top_count, axis=0)[-top_count:]
    raw_score = np.mean(top, axis=0)
    smooth_score = ndi.gaussian_filter1d(raw_score, sigma=1.25, mode="nearest")
    log_score = np.log1p(smooth_score)
    threshold, centers = two_population_threshold(log_score)
    active = log_score >= threshold
    close_frames = max(1, int(round(close_gap_seconds * sample_rate / hop_length)))
    minimum_frames = max(1, int(round(minimum_seconds * sample_rate / hop_length)))
    active = ndi.binary_closing(active, structure=np.ones(close_frames, dtype=bool))
    active = ndi.binary_opening(active, structure=np.ones(minimum_frames, dtype=bool))
    labels, count = ndi.label(active)
    intervals: list[TimeInterval] = []
    for label in range(1, count + 1):
        indices = np.flatnonzero(labels == label)
        if indices.size < minimum_frames:
            continue
        frame0 = int(indices[0])
        frame1 = int(indices[-1] + 1)
        intervals.append(
            TimeInterval(
                frame0=frame0,
                frame1=frame1,
                seconds0=frame0 * hop_length / sample_rate,
                seconds1=frame1 * hop_length / sample_rate,
                score=float(np.mean(log_score[frame0:frame1])),
            )
        )
    diagnostics = {
        "threshold": threshold,
        "population_centers": list(centers),
        "active_fraction": float(np.mean(active)),
    }
    return log_score, active, tuple(intervals), diagnostics


def _intervals_from_activity(
    active: np.ndarray,
    score: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    minimum_frames: int,
) -> tuple[TimeInterval, ...]:
    labels, count = ndi.label(active)
    intervals: list[TimeInterval] = []
    for label in range(1, count + 1):
        indices = np.flatnonzero(labels == label)
        if indices.size < minimum_frames:
            continue
        frame0 = int(indices[0])
        frame1 = int(indices[-1] + 1)
        intervals.append(TimeInterval(
            frame0=frame0,
            frame1=frame1,
            seconds0=frame0 * hop_length / sample_rate,
            seconds1=frame1 * hop_length / sample_rate,
            score=float(np.mean(score[frame0:frame1])),
        ))
    return tuple(intervals)


def waveform_rms_score(
    samples: np.ndarray,
    *,
    frame_count: int,
    hop_length: int,
    window_length: int = 2048,
) -> np.ndarray:
    """Return centered frame log-RMS without constructing a frame matrix."""

    waveform = np.asarray(samples, dtype=np.float64)
    if waveform.ndim != 1:
        raise ValueError("waveform RMS expects mono samples")
    if frame_count < 1 or hop_length < 1 or window_length < 1:
        raise ValueError("frame geometry must be positive")
    cumulative = np.concatenate(([0.0], np.cumsum(waveform * waveform)))
    centers = np.arange(frame_count, dtype=np.int64) * hop_length
    starts = np.maximum(0, centers - window_length // 2)
    stops = np.minimum(waveform.size, centers + window_length // 2)
    counts = np.maximum(stops - starts, 1)
    energy = cumulative[stops] - cumulative[starts]
    rms = np.sqrt(np.maximum(energy / counts, 0.0))
    return np.log(np.maximum(rms, 1e-12))


def supported_activity_intervals(
    trace_field: np.ndarray,
    samples: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    minimum_seconds: float = 0.08,
    close_gap_seconds: float = 0.10,
    padding_seconds: float = 0.0,
    rms_window_length: int = 2048,
) -> tuple[np.ndarray, np.ndarray, tuple[TimeInterval, ...], dict]:
    """Fuse harmonic existence with an independent waveform-energy witness."""

    harmonic_score, harmonic_active, _, harmonic_diagnostics = activity_intervals(
        trace_field,
        sample_rate=sample_rate,
        hop_length=hop_length,
        minimum_seconds=minimum_seconds,
        close_gap_seconds=close_gap_seconds,
    )
    energy_score = waveform_rms_score(
        samples,
        frame_count=trace_field.shape[1],
        hop_length=hop_length,
        window_length=rms_window_length,
    )
    energy_threshold, energy_centers = two_population_threshold(energy_score)
    energy_active = energy_score >= energy_threshold
    close_frames = max(1, int(round(close_gap_seconds * sample_rate / hop_length)))
    minimum_frames = max(1, int(round(minimum_seconds * sample_rate / hop_length)))
    active = harmonic_active | energy_active
    active = ndi.binary_closing(active, structure=np.ones(close_frames, dtype=bool))
    active = ndi.binary_opening(active, structure=np.ones(minimum_frames, dtype=bool))
    padding_frames = max(
        0, int(round(padding_seconds * sample_rate / hop_length))
    )
    if padding_frames:
        active = ndi.binary_dilation(
            active,
            structure=np.ones(2 * padding_frames + 1, dtype=bool),
        )
    # The two centered scores are only interval-quality witnesses. They are
    # never substituted for the promoted 2-D comparison field.
    harmonic_centered = harmonic_score - harmonic_diagnostics["threshold"]
    energy_centered = energy_score - energy_threshold
    activity_score = np.maximum(harmonic_centered, energy_centered)
    intervals = _intervals_from_activity(
        active,
        activity_score,
        sample_rate=sample_rate,
        hop_length=hop_length,
        minimum_frames=minimum_frames,
    )
    diagnostics = {
        "activity_method": "harmonic_ridge_or_centered_waveform_log_rms",
        "harmonic_threshold": harmonic_diagnostics["threshold"],
        "harmonic_population_centers": harmonic_diagnostics["population_centers"],
        "harmonic_active_fraction": harmonic_diagnostics["active_fraction"],
        "energy_threshold": energy_threshold,
        "energy_population_centers": list(energy_centers),
        "energy_active_fraction_before_morphology": float(np.mean(energy_active)),
        "active_fraction": float(np.mean(active)),
        "minimum_seconds": minimum_seconds,
        "close_gap_seconds": close_gap_seconds,
        "padding_seconds": padding_seconds,
    }
    return activity_score, active, intervals, diagnostics


def phone_boundary_proposals(
    cartoon: np.ndarray,
    speech: tuple[TimeInterval, ...],
    *,
    sample_rate: int,
    hop_length: int,
    minimum_seconds: float = 0.045,
    maximum_seconds: float = 0.240,
) -> tuple[TimeInterval, ...]:
    """Split speech intervals at persistent changes of normalized geometry."""
    positive = np.maximum(np.asarray(cartoon, dtype=np.float64), 0.0)
    norms = np.linalg.norm(positive, axis=0)
    unit = positive / np.maximum(norms[None, :], 1e-30)
    novelty = np.zeros(unit.shape[1], dtype=np.float64)
    novelty[1:] = 1.0 - np.sum(unit[:, 1:] * unit[:, :-1], axis=0)
    novelty = ndi.gaussian_filter1d(novelty, sigma=1.0, mode="nearest")
    minimum_frames = max(1, int(round(minimum_seconds * sample_rate / hop_length)))
    maximum_frames = max(minimum_frames + 1, int(round(maximum_seconds * sample_rate / hop_length)))
    phones: list[TimeInterval] = []
    for span in speech:
        local = novelty[span.frame0 : span.frame1]
        if local.size <= minimum_frames:
            boundaries = [span.frame0, span.frame1]
        else:
            prominence = max(float(np.std(local)) * 0.35, 1e-6)
            peaks, _ = find_peaks(
                local,
                distance=minimum_frames,
                prominence=prominence,
            )
            boundaries = [span.frame0]
            boundaries.extend(int(span.frame0 + peak) for peak in peaks)
            boundaries.append(span.frame1)
            boundaries = sorted(set(boundaries))
        dense = [boundaries[0]]
        for boundary in boundaries[1:]:
            while boundary - dense[-1] > maximum_frames:
                search0 = dense[-1] + minimum_frames
                search1 = min(dense[-1] + maximum_frames + 1, boundary)
                if search1 <= search0:
                    candidate = dense[-1] + maximum_frames
                else:
                    candidate = search0 + int(np.argmax(
                        novelty[search0:search1]
                    ))
                dense.append(candidate)
            if boundary - dense[-1] >= minimum_frames or boundary == span.frame1:
                dense.append(boundary)
        for frame0, frame1 in zip(dense[:-1], dense[1:]):
            if frame1 <= frame0:
                continue
            phones.append(
                TimeInterval(
                    frame0=frame0,
                    frame1=frame1,
                    seconds0=frame0 * hop_length / sample_rate,
                    seconds1=frame1 * hop_length / sample_rate,
                    score=float(np.max(novelty[frame0:frame1], initial=0.0)),
                )
            )
    return tuple(phones)


def build_full_recording(
    wav_path: Path,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    crop_rows: int = 256,
) -> dict:
    if librosa is None:
        raise RuntimeError(
            "the retained failed-control runner requires librosa; "
            "use corrected_full_recording for the promoted path")
    started = time.perf_counter()
    samples, sample_rate = librosa.load(wav_path, sr=None, mono=True)
    spectrum = librosa.stft(
        samples,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=True,
    )
    texture = exact_low_rows_from_stft(
        spectrum,
        n_fft=n_fft,
        crop_rows=crop_rows,
    )
    peak = max(float(np.max(texture)), 1e-30)
    meyer_input = np.ascontiguousarray(texture * (255.0 / peak))
    meyer_started = time.perf_counter()
    cartoon, meyer_texture = bfft.meyer_split(meyer_input, threads=4)
    meyer_ms = 1000.0 * (time.perf_counter() - meyer_started)
    activity_score, active, speech, activity_diagnostics = activity_intervals(
        cartoon,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    phones = phone_boundary_proposals(
        cartoon,
        speech,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    return {
        "samples": samples,
        "sample_rate": sample_rate,
        "spectrum_shape": spectrum.shape,
        "texture": texture,
        "meyer_input": meyer_input,
        "cartoon": cartoon,
        "meyer_texture": meyer_texture,
        "activity_score": activity_score,
        "active": active,
        "speech": speech,
        "phones": phones,
        "diagnostics": {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "crop_rows": crop_rows,
            "duration_seconds": samples.size / sample_rate,
            "frames": int(spectrum.shape[1]),
            "meyer_ms": meyer_ms,
            "total_ms": 1000.0 * (time.perf_counter() - started),
            "speech_intervals": len(speech),
            "phone_intervals": len(phones),
            **activity_diagnostics,
        },
    }


def serializable_intervals(values: tuple[TimeInterval, ...]) -> list[dict]:
    return [asdict(value) for value in values]
