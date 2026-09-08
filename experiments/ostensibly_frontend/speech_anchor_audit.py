"""Label-held-out measurements for Cleanup/SHARK temporal anchors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .state_phone_lattice import state_speech_crops


SILENCE_LABELS = frozenset({"pau", "ssil", "sil"})
UNVOICED_LABELS = frozenset({"ch", "f", "hh", "k", "p", "s", "sh", "t", "th"})


@dataclass(frozen=True)
class LabSegment:
    seconds0: float
    seconds1: float
    state: str
    label: str


def broad_phone_state(label: str) -> str:
    """Map an audit label to silence, unvoiced, or voiced speech."""

    normalized = label.strip().lower()
    if normalized in SILENCE_LABELS:
        return "silence"
    if normalized in UNVOICED_LABELS:
        return "unvoiced"
    return "voiced"


def load_lab_segments(path: Path) -> tuple[LabSegment, ...]:
    """Read consecutive Festival-style endpoint labels without dropping silence."""

    start = 0.0
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[0].startswith("#"):
            continue
        stop = float(fields[0])
        if not np.isfinite(stop) or stop <= start:
            raise ValueError(f"invalid label endpoint in {path}")
        label = fields[2].lower()
        output.append(LabSegment(start, stop, broad_phone_state(label), label))
        start = stop
    if not output:
        raise ValueError(f"no timed labels in {path}")
    return tuple(output)


def target_state_geometry(
    segments: tuple[LabSegment, ...],
    frame_count: int,
    *,
    sample_rate: int,
    hop_length: int,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Return speech mask, speech edges, and voiced/unvoiced audit boundaries."""

    if frame_count < 1 or sample_rate < 1 or hop_length < 1 or not segments:
        raise ValueError("invalid target state geometry")
    centers = np.arange(frame_count, dtype=np.float64) * hop_length / sample_rate
    stops = np.asarray([segment.seconds1 for segment in segments])
    index = np.searchsorted(stops, centers, side="right")
    states = np.full(frame_count, "silence", dtype="<U8")
    valid = index < len(segments)
    states[valid] = np.asarray([segment.state for segment in segments])[index[valid]]
    speech = states != "silence"

    speech_edges = []
    manner_edges = []
    for left, right in zip(segments[:-1], segments[1:], strict=True):
        if left.state == right.state:
            continue
        frame = int(round(left.seconds1 * sample_rate / hop_length))
        if (left.state == "silence") != (right.state == "silence"):
            speech_edges.append(frame)
        elif {left.state, right.state} == {"voiced", "unvoiced"}:
            manner_edges.append(frame)
    return speech, tuple(speech_edges), tuple(manner_edges)


def predicted_state_geometry(
    state: np.ndarray,
    speech_mask: np.ndarray,
    *,
    minimum_run_frames: int = 3,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Extract crop edges and stable voiced/nonvoiced transitions."""

    crop_edges = []
    manner_edges = []
    for crop in state_speech_crops(
        state,
        speech_mask,
        minimum_run_frames=minimum_run_frames,
    ):
        crop_edges.extend((crop.frame0, crop.frame1))
        for left, right in zip(crop.state_audit[:-1], crop.state_audit[1:], strict=True):
            left_voiced = left["state"] == "voiced"
            right_voiced = right["state"] == "voiced"
            if left_voiced != right_voiced:
                manner_edges.append(int(left["frame1"]))
    return tuple(crop_edges), tuple(manner_edges)


def match_boundaries(
    predicted: tuple[int, ...],
    target: tuple[int, ...],
    *,
    tolerance_frames: int,
) -> tuple[int, tuple[int, ...]]:
    """Maximum-cardinality ordered matching within a frame tolerance."""

    if tolerance_frames < 0:
        raise ValueError("boundary tolerance must be nonnegative")
    left = sorted(int(value) for value in predicted)
    right = sorted(int(value) for value in target)
    i = 0
    j = 0
    errors = []
    while i < len(left) and j < len(right):
        delta = left[i] - right[j]
        if abs(delta) <= tolerance_frames:
            errors.append(abs(delta))
            i += 1
            j += 1
        elif left[i] < right[j] - tolerance_frames:
            i += 1
        else:
            j += 1
    return len(errors), tuple(errors)


def binary_counts(predicted: np.ndarray, target: np.ndarray) -> dict[str, int]:
    """Return additive confusion counts for two frame masks."""

    guess = np.asarray(predicted, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if guess.shape != truth.shape or guess.ndim != 1:
        raise ValueError("binary masks have incompatible shapes")
    return {
        "true_positive": int(np.sum(guess & truth)),
        "false_positive": int(np.sum(guess & ~truth)),
        "false_negative": int(np.sum(~guess & truth)),
        "true_negative": int(np.sum(~guess & ~truth)),
    }


def summarize_counts(counts: dict[str, int]) -> dict[str, float | int]:
    """Add precision, recall, and F1 to accumulated confusion counts."""

    tp = int(counts["true_positive"])
    fp = int(counts["false_positive"])
    fn = int(counts["false_negative"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-30),
    }


def wilson_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """One-sided Wilson lower confidence bound for a binomial rate."""

    if not 0 <= successes <= trials or trials < 1 or not 0.5 < confidence < 1.0:
        raise ValueError("invalid Wilson-bound configuration")
    z = NormalDist().inv_cdf(confidence)
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = rate + z * z / (2.0 * trials)
    radius = z * np.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
    return float((center - radius) / denominator)
