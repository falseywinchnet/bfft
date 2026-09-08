"""Candidate-conditioned contiguous phone geometry for fragmented speech."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np

from .cleanup_shark_vad import SpeechState


@dataclass(frozen=True)
class PhoneSpanCost:
    """Atlas evidence for assigning one phone to one contiguous region span."""

    distance: float
    rank: int
    witness: str
    empirical_target_frequency_at_rank: float | None = None
    cumulative_target_coverage: float | None = None
    cumulative_target_coverage_lower_95: float | None = None


@dataclass(frozen=True)
class SegmentalPronunciationScore:
    """The best bottleneck-first partition for one pronunciation."""

    phones: tuple[str, ...]
    cuts: tuple[int, ...]
    phone_costs: tuple[PhoneSpanCost, ...]

    @property
    def worst_distance(self) -> float:
        return max(cost.distance for cost in self.phone_costs)

    @property
    def mean_distance(self) -> float:
        return sum(cost.distance for cost in self.phone_costs) / len(
            self.phone_costs
        )

    @property
    def worst_rank(self) -> int:
        return max(cost.rank for cost in self.phone_costs)

    @property
    def mean_rank(self) -> float:
        return sum(cost.rank for cost in self.phone_costs) / len(
            self.phone_costs
        )

    @property
    def spans(self) -> tuple[tuple[int, int], ...]:
        boundaries = (0, *self.cuts)
        return tuple(zip(boundaries, self.cuts, strict=True))

    @property
    def objective(self) -> tuple[int, float, float, float, tuple[int, ...]]:
        """Calibrated identity score: every phone must rank plausibly.

        Raw distances from different observed spans are not commensurate: a
        short, simple patch can have a much lower distance floor than a longer
        patch.  Atlas rank is the empirical within-span gauge.  Raw distance
        remains only as a late tie-break between equally ranked explanations.
        """

        return (
            self.worst_rank,
            self.mean_rank,
            self.worst_distance,
            self.mean_distance,
            self.cuts,
        )


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    """Return half-open constant runs of one integer sequence."""

    sequence = np.asarray(values)
    if sequence.ndim != 1:
        raise ValueError("state sequence must be one-dimensional")
    if not sequence.size:
        return []
    boundaries = np.flatnonzero(np.r_[True, sequence[1:] != sequence[:-1], True])
    return [
        (int(start), int(stop), int(sequence[start]))
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def enclosing_speech_crop(
    speech_mask: np.ndarray,
    query_frame0: int,
    query_frame1: int,
) -> tuple[int, int]:
    """Select the measured crop with greatest overlap with a queried word."""

    mask = np.asarray(speech_mask, dtype=bool)
    if mask.ndim != 1 or not (0 <= query_frame0 < query_frame1 <= mask.size):
        raise ValueError("invalid queried frame span")
    candidates = [
        (start, stop)
        for start, stop, active in _runs(mask.astype(np.int8))
        if active
    ]
    if not candidates:
        raise ValueError("speech mask contains no measured crops")
    scored = [
        (
            max(0, min(stop, query_frame1) - max(start, query_frame0)),
            -(stop - start),
            start,
            stop,
        )
        for start, stop in candidates
    ]
    overlap, _, start, stop = max(scored)
    if overlap <= 0:
        raise ValueError("no measured speech crop overlaps the queried word")
    return start, stop


def debounced_state_landmarks(
    state: np.ndarray,
    speech_mask: np.ndarray,
    query_frame0: int,
    query_frame1: int,
    *,
    minimum_run_frames: int = 3,
) -> tuple[tuple[int, ...], tuple[dict[str, int | str], ...]]:
    """Turn stable voiced/unvoiced phases into label-free cut coordinates.

    Single-frame pitch toggles are acquisition jitter, not plausible phone
    compartments.  Short categorical runs are iteratively absorbed into the
    longer neighbor (or the common neighbor), after which every stable state
    transition becomes a possible segmental boundary.  The crop endpoints are
    supplied by the delayed-commit speech tracker rather than the old ridge
    islands.
    """

    values = np.asarray(state, dtype=np.int8)
    mask = np.asarray(speech_mask, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape or minimum_run_frames < 1:
        raise ValueError("invalid speech-state geometry")
    crop0, crop1 = enclosing_speech_crop(mask, query_frame0, query_frame1)
    coarse = np.full(crop1 - crop0, 0, dtype=np.int8)
    local = values[crop0:crop1]
    coarse[local == SpeechState.VOICED] = 1
    coarse[local == SpeechState.UNVOICED] = 2
    coarse[
        (local == SpeechState.HANGOVER) | (local == SpeechState.CANDIDATE)
    ] = 3

    while True:
        runs = _runs(coarse)
        short = [
            index
            for index, (start, stop, _) in enumerate(runs)
            if stop - start < minimum_run_frames
        ]
        # First close a short island whose neighbors agree.  Consuming a
        # neighboring fragment first can incorrectly move a genuine boundary
        # toward the island instead of removing the island itself.
        bridged = [
            index
            for index in short
            if 0 < index < len(runs) - 1
            and runs[index - 1][2] == runs[index + 1][2]
        ]
        short_index = bridged[0] if bridged else (short[0] if short else None)
        if short_index is None or len(runs) == 1:
            break
        start, stop, _ = runs[short_index]
        previous = runs[short_index - 1] if short_index > 0 else None
        following = runs[short_index + 1] if short_index + 1 < len(runs) else None
        if previous is not None and following is not None and previous[2] == following[2]:
            replacement = previous[2]
        elif previous is None:
            assert following is not None
            replacement = following[2]
        elif following is None:
            replacement = previous[2]
        else:
            previous_length = previous[1] - previous[0]
            following_length = following[1] - following[0]
            replacement = (
                following[2] if following_length > previous_length else previous[2]
            )
        coarse[start:stop] = replacement

    stable_runs = _runs(coarse)
    landmarks = (crop0, *(crop0 + stop for _, stop, _ in stable_runs[:-1]), crop1)
    names = {0: "quiet", 1: "voiced", 2: "unvoiced", 3: "hangover"}
    audit = tuple(
        {
            "frame0": crop0 + start,
            "frame1": crop0 + stop,
            "frames": stop - start,
            "state": names[label],
        }
        for start, stop, label in stable_runs
    )
    return tuple(int(value) for value in landmarks), audit


def support_landmarks_within_crop(
    region_frame0: np.ndarray,
    region_frame1: np.ndarray,
    crop_frame0: int,
    crop_frame1: int,
    *,
    minimum_interval_frames: int = 2,
) -> tuple[int, ...]:
    """Expose both edges of harmonic support inside a measured speech crop.

    Unlike the old gap-right/left policy, a silence or closure contributes two
    observable coordinates.  The candidate-conditioned partition may then
    assign that compartment geometrically instead of having its ownership
    decided in advance.
    """

    starts = np.asarray(region_frame0, dtype=np.int64)
    stops = np.asarray(region_frame1, dtype=np.int64)
    if (
        starts.ndim != 1
        or stops.shape != starts.shape
        or crop_frame0 >= crop_frame1
        or minimum_interval_frames < 1
        or np.any(stops <= starts)
    ):
        raise ValueError("invalid support/crop geometry")
    interior = sorted(
        {
            int(edge)
            for edge in np.concatenate((starts, stops))
            if crop_frame0 + minimum_interval_frames
            <= edge
            <= crop_frame1 - minimum_interval_frames
        }
    )
    landmarks = [int(crop_frame0)]
    for edge in interior:
        if edge - landmarks[-1] >= minimum_interval_frames:
            landmarks.append(edge)
    if crop_frame1 - landmarks[-1] < minimum_interval_frames:
        landmarks.pop()
    landmarks.append(int(crop_frame1))
    return tuple(landmarks)


def best_contiguous_partition(
    phones: tuple[str, ...],
    region_count: int,
    costs: Mapping[tuple[int, int, str], PhoneSpanCost],
) -> SegmentalPronunciationScore:
    """Find the lexicographically best rank-calibrated bottleneck partition.

    ``costs[(start, stop, phone)]`` describes a half-open span in coordinates
    local to the queried word.  Each phone receives at least one observed
    region.  Atlas rank calibrates the same geometric distance independently
    inside every queried span, making different boundary hypotheses comparable.
    """

    if not phones:
        raise ValueError("segmental pronunciation requires phones")
    if region_count < len(phones):
        raise ValueError("fewer observed regions than candidate phones")

    best: SegmentalPronunciationScore | None = None
    for interior in combinations(range(1, region_count), len(phones) - 1):
        boundaries = (0, *interior, region_count)
        selected = tuple(
            costs[(start, stop, phone)]
            for start, stop, phone in zip(
                boundaries[:-1], boundaries[1:], phones, strict=True
            )
        )
        candidate = SegmentalPronunciationScore(
            phones=phones,
            cuts=tuple(boundaries[1:]),
            phone_costs=selected,
        )
        if best is None or candidate.objective < best.objective:
            best = candidate
    assert best is not None
    return best


def score_to_dict(score: SegmentalPronunciationScore) -> dict[str, object]:
    boundaries = (0, *score.cuts)
    return {
        "phones": list(score.phones),
        "cuts": list(score.cuts[:-1]),
        "worst_distance": score.worst_distance,
        "mean_distance": score.mean_distance,
        "worst_rank": score.worst_rank,
        "mean_rank": score.mean_rank,
        "assignments": [
            {
                "phone": phone,
                "region_span": [start, stop],
                "distance": cost.distance,
                "rank": cost.rank,
                "witness": cost.witness,
                **(
                    {
                        "empirical_target_frequency_at_rank": cost.empirical_target_frequency_at_rank,
                        "cumulative_target_coverage": cost.cumulative_target_coverage,
                        "cumulative_target_coverage_lower_95": cost.cumulative_target_coverage_lower_95,
                    }
                    if cost.empirical_target_frequency_at_rank is not None
                    else {}
                ),
            }
            for phone, start, stop, cost in zip(
                score.phones,
                boundaries[:-1],
                boundaries[1:],
                score.phone_costs,
                strict=True,
            )
        ],
    }
