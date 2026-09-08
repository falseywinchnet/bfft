"""Reusable raw-phone occurrence bank for coherent generic word surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata

from .conditional_ridge_atlas import (
    conditional_ridge_distance,
    conditional_ridge_signature,
)


FORMAT = "ostensibly_generic_word_template_bank_v2"
LEGACY_FORMAT = "ostensibly_generic_word_template_bank_v1"
UNCONDITIONED_CONTEXT = "*"
EDGE_CONTEXT = "<edge>"


def barycenter_word_signature(
    signatures: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse realization quantiles into their equal-weight 1-D W2 barycenter."""

    if not signatures:
        raise ValueError("word signature barycenter requires realizations")
    surfaces = np.stack(
        [np.asarray(signature[0], dtype=np.float64) for signature in signatures]
    )
    masses = np.stack(
        [np.asarray(signature[1], dtype=np.float64) for signature in signatures]
    )
    if surfaces.ndim != 3 or masses.shape != surfaces.shape[:2]:
        raise ValueError("word signature realizations have incompatible shapes")
    return np.mean(surfaces, axis=0), np.mean(masses, axis=0)


def word_signature_aggregation_scores(
    query_signature: tuple[np.ndarray, np.ndarray],
    reference_signatures: tuple[tuple[np.ndarray, np.ndarray], ...],
    maximum_shift: int,
) -> dict[str, float]:
    """Measure independent-realization agreement and literal surface fusion."""

    if not reference_signatures:
        raise ValueError("word signature scoring requires realizations")
    distances = np.asarray(
        [
            conditional_ridge_distance(
                query_signature[0],
                query_signature[1],
                reference[0],
                reference[1],
                maximum_shift=maximum_shift,
            )
            for reference in reference_signatures
        ],
        dtype=np.float64,
    )
    barycenter = barycenter_word_signature(reference_signatures)
    return {
        "minimum": float(np.min(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "barycenter": conditional_ridge_distance(
            query_signature[0],
            query_signature[1],
            barycenter[0],
            barycenter[1],
            maximum_shift=maximum_shift,
        ),
    }


def word_signature_distance(
    query_signature: tuple[np.ndarray, np.ndarray],
    reference_signatures: tuple[tuple[np.ndarray, np.ndarray], ...],
    maximum_shift: int,
    aggregation: str = "mean",
) -> float:
    """Score one requested realization rule without computing unused rules."""

    if aggregation == "barycenter":
        reference = barycenter_word_signature(reference_signatures)
        return conditional_ridge_distance(
            query_signature[0],
            query_signature[1],
            reference[0],
            reference[1],
            maximum_shift=maximum_shift,
        )
    if aggregation not in ("minimum", "mean", "median"):
        raise ValueError("unknown word signature aggregation")
    distances = [
        conditional_ridge_distance(
            query_signature[0],
            query_signature[1],
            reference[0],
            reference[1],
            maximum_shift=maximum_shift,
        )
        for reference in reference_signatures
    ]
    reducer = {
        "minimum": np.min,
        "mean": np.mean,
        "median": np.median,
    }[aggregation]
    return float(reducer(distances))


def _subset(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] <= count:
        return points
    return points[np.linspace(0, points.shape[0] - 1, count, dtype=np.int64)]


def _joint_word_cloud(
    chunks: tuple[np.ndarray, ...],
    lane_weights: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Place complete phone occurrences before the signature's single gauge."""

    if not chunks:
        raise ValueError("generic word cloud requires phone chunks")
    weights = np.ones(len(chunks), dtype=np.float64) if lane_weights is None else np.asarray(lane_weights, dtype=np.float64)
    if (
        weights.shape != (len(chunks),)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("generic word lane weights must be finite and positive")
    boundaries = np.concatenate(([0.0], np.cumsum(weights)))
    boundaries /= boundaries[-1]
    values = [np.asarray(chunk, dtype=np.float64).copy() for chunk in chunks]
    for slot, chunk in enumerate(values):
        chunk[:, 1] = boundaries[slot] + chunk[:, 1] * (
            boundaries[slot + 1] - boundaries[slot]
        )
    return np.concatenate(values)


@dataclass(frozen=True)
class GenericWordTemplateBank:
    labels: np.ndarray
    witnesses: np.ndarray
    clouds: np.ndarray
    left_labels: np.ndarray
    right_labels: np.ndarray

    def __post_init__(self) -> None:
        count = self.labels.size
        if (
            self.labels.shape != (count,)
            or self.witnesses.shape != (count,)
            or self.left_labels.shape != (count,)
            or self.right_labels.shape != (count,)
            or self.clouds.ndim != 3
            or self.clouds.shape[0] != count
            or self.clouds.shape[2] != 3
            or count < 1
            or not np.all(np.isfinite(self.clouds))
        ):
            raise ValueError("generic word template bank has invalid occurrences")

    @property
    def points_per_occurrence(self) -> int:
        return int(self.clouds.shape[1])

    def pools(
        self, phones: tuple[str, ...], context_conditioned: bool = False
    ) -> tuple[np.ndarray, ...]:
        output = []
        labels = self.labels.astype(str)
        left_labels = self.left_labels.astype(str)
        right_labels = self.right_labels.astype(str)
        for slot, phone in enumerate(phones):
            indices = np.flatnonzero(labels == phone)
            if context_conditioned and indices.size:
                scores = np.zeros(indices.size, dtype=np.int8)
                if slot:
                    scores += (
                        left_labels[indices] == phones[slot - 1]
                    ).astype(np.int8)
                if slot + 1 < len(phones):
                    scores += (
                        right_labels[indices] == phones[slot + 1]
                    ).astype(np.int8)
                indices = indices[scores == np.max(scores)]
            pool = self.clouds[indices]
            if not pool.size:
                raise KeyError(f"generic word bank has no {phone!r} occurrences")
            output.append(pool)
        return tuple(output)

    def context_depths(self, phones: tuple[str, ...]) -> tuple[int, ...]:
        """Report how many requested neighbors support each selected pool."""

        labels = self.labels.astype(str)
        left_labels = self.left_labels.astype(str)
        right_labels = self.right_labels.astype(str)
        output = []
        for slot, phone in enumerate(phones):
            indices = np.flatnonzero(labels == phone)
            if not indices.size:
                raise KeyError(f"generic word bank has no {phone!r} occurrences")
            scores = np.zeros(indices.size, dtype=np.int8)
            if slot:
                scores += (
                    left_labels[indices] == phones[slot - 1]
                ).astype(np.int8)
            if slot + 1 < len(phones):
                scores += (
                    right_labels[indices] == phones[slot + 1]
                ).astype(np.int8)
            output.append(int(np.max(scores)))
        return tuple(output)

    def word_signatures(
        self,
        phones: tuple[str, ...],
        realization_count: int = 4,
        time_bins: int = 64,
        row_quantiles: int = 24,
        context_conditioned: bool = False,
        context_policy: str | None = None,
        duration_weights: tuple[float, ...] | None = None,
        saliency_power: float = 0.0,
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Build complete deterministic realizations, then gauge each as a word."""

        if realization_count < 1:
            raise ValueError("generic word signatures require realizations")
        if duration_weights is not None and len(duration_weights) != len(phones):
            raise ValueError("generic word duration count disagrees with phones")
        policy = context_policy or (
            "maximum" if context_conditioned else "unconditional"
        )
        if policy not in ("unconditional", "maximum", "mixed"):
            raise ValueError("unknown generic word context policy")
        labels = self.labels.astype(str)
        left_labels = self.left_labels.astype(str)
        right_labels = self.right_labels.astype(str)
        pools = []
        schedules = []
        for slot, phone in enumerate(phones):
            indices = np.flatnonzero(labels == phone)
            if not indices.size:
                raise KeyError(f"generic word bank has no {phone!r} occurrences")
            scores = np.zeros(indices.size, dtype=np.int8)
            if slot:
                scores += (
                    left_labels[indices] == phones[slot - 1]
                ).astype(np.int8)
            if slot + 1 < len(phones):
                scores += (
                    right_labels[indices] == phones[slot + 1]
                ).astype(np.int8)
            maximum = int(np.max(scores))
            if policy == "unconditional":
                selected_indices = indices
                tier_schedule = np.zeros(realization_count, dtype=np.int8)
            elif policy == "maximum":
                selected_indices = indices[scores == maximum]
                tier_schedule = np.full(
                    realization_count, maximum, dtype=np.int8
                )
            else:
                # Preserve the backoff hierarchy instead of replacing it.
                # Four samples become 2/1/1 exact/one-sided/unconditional for
                # an interior triphone, or 3/1 one-sided/unconditional at an
                # edge. Quantile placement generalizes the ratio to other R.
                quantiles = (
                    np.arange(realization_count, dtype=np.float64) + 0.5
                ) / realization_count
                if maximum >= 2:
                    tier_schedule = np.where(
                        quantiles < 0.25,
                        0,
                        np.where(quantiles < 0.5, 1, 2),
                    ).astype(np.int8)
                elif maximum == 1:
                    tier_schedule = np.where(
                        quantiles < 0.25, 0, 1
                    ).astype(np.int8)
                else:
                    tier_schedule = np.zeros(
                        realization_count, dtype=np.int8
                    )
                available = set(int(value) for value in scores)
                tier_schedule = np.asarray(
                    [
                        (
                            max(tier for tier in available if tier <= target)
                            if any(tier <= target for tier in available)
                            else min(available)
                        )
                        for target in tier_schedule
                    ],
                    dtype=np.int8,
                )
                selected_indices = indices
            tier_schedule = np.roll(tier_schedule, slot)
            schedule = np.empty(realization_count, dtype=np.int64)
            for tier in np.unique(tier_schedule):
                positions = np.flatnonzero(tier_schedule == tier)
                tier_indices = (
                    selected_indices
                    if policy != "mixed"
                    else indices[scores == tier]
                )
                choices = np.linspace(
                    0, tier_indices.size - 1, positions.size, dtype=np.int64
                )
                schedule[positions] = tier_indices[choices]
            pools.append(self.clouds)
            schedules.append(schedule)
        return tuple(
            conditional_ridge_signature(
                _joint_word_cloud(
                    tuple(
                        pool[int(schedules[slot][realization])]
                        for slot, pool in enumerate(pools)
                    ),
                    duration_weights,
                ),
                time_bins,
                row_quantiles,
                saliency_power,
            )
            for realization in range(realization_count)
        )


def compile_generic_word_template_bank(
    occurrences: Iterable[
        tuple[str, str, np.ndarray]
        | tuple[str, str, str, str, np.ndarray]
    ],
    points_per_occurrence: int = 512,
) -> GenericWordTemplateBank:
    if points_per_occurrence < 16:
        raise ValueError("generic word bank resolution is too small")
    labels = []
    witnesses = []
    left_labels = []
    right_labels = []
    clouds = []
    for occurrence in occurrences:
        if len(occurrence) == 3:
            label, witness, points = occurrence
            left_label = right_label = UNCONDITIONED_CONTEXT
        elif len(occurrence) == 5:
            label, witness, left_label, right_label, points = occurrence
        else:
            raise ValueError("generic word occurrence has an invalid schema")
        cloud = _subset(np.asarray(points, dtype=np.float64), points_per_occurrence).copy()
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not cloud.size:
            raise ValueError("generic word bank occurrence is not an N x 3 cloud")
        if cloud.shape[0] < points_per_occurrence:
            indices = np.arange(points_per_occurrence) % cloud.shape[0]
            cloud = cloud[indices]
        count = cloud.shape[0]
        cloud[:, 1] = (rankdata(cloud[:, 1], method="average") - 0.5) / count
        labels.append(str(label))
        witnesses.append(str(witness))
        left_labels.append(str(left_label))
        right_labels.append(str(right_label))
        clouds.append(cloud.astype(np.float32))
    if not clouds:
        raise ValueError("generic word bank requires occurrences")
    return GenericWordTemplateBank(
        labels=np.asarray(labels),
        witnesses=np.asarray(witnesses),
        clouds=np.stack(clouds),
        left_labels=np.asarray(left_labels),
        right_labels=np.asarray(right_labels),
    )


def combine_generic_word_template_banks(
    banks: Iterable[GenericWordTemplateBank],
) -> GenericWordTemplateBank:
    values = tuple(banks)
    if not values:
        raise ValueError("generic word bank combination requires inputs")
    points = values[0].points_per_occurrence
    if any(bank.points_per_occurrence != points for bank in values[1:]):
        raise ValueError("generic word banks use incompatible resolutions")
    return GenericWordTemplateBank(
        labels=np.concatenate([bank.labels for bank in values]),
        witnesses=np.concatenate([bank.witnesses for bank in values]),
        clouds=np.concatenate([bank.clouds for bank in values]),
        left_labels=np.concatenate([bank.left_labels for bank in values]),
        right_labels=np.concatenate([bank.right_labels for bank in values]),
    )


def save_generic_word_template_bank(
    path: Path, bank: GenericWordTemplateBank
) -> None:
    metadata = json.dumps(
        {
            "format": FORMAT,
            "points_per_occurrence": bank.points_per_occurrence,
        },
        sort_keys=True,
    )
    np.savez_compressed(
        path,
        metadata=np.asarray(metadata),
        labels=bank.labels,
        witnesses=bank.witnesses,
        clouds=bank.clouds,
        left_labels=bank.left_labels,
        right_labels=bank.right_labels,
    )


def load_generic_word_template_bank(path: Path) -> GenericWordTemplateBank:
    with np.load(path, allow_pickle=False) as document:
        metadata = json.loads(str(document["metadata"]))
        if metadata.get("format") not in (FORMAT, LEGACY_FORMAT):
            raise ValueError("unsupported generic word template bank format")
        count = np.asarray(document["labels"]).size
        bank = GenericWordTemplateBank(
            labels=np.asarray(document["labels"]),
            witnesses=np.asarray(document["witnesses"]),
            clouds=np.asarray(document["clouds"], dtype=np.float32),
            left_labels=(
                np.asarray(document["left_labels"])
                if "left_labels" in document
                else np.full(count, UNCONDITIONED_CONTEXT)
            ),
            right_labels=(
                np.asarray(document["right_labels"])
                if "right_labels" in document
                else np.full(count, UNCONDITIONED_CONTEXT)
            ),
        )
    if bank.points_per_occurrence != int(metadata["points_per_occurrence"]):
        raise ValueError("generic word bank metadata resolution disagrees")
    return bank
