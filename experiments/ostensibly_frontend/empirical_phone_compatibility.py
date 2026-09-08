"""Finite-sample phone compatibility from cross-speaker rank evidence.

This module deliberately does not invent a point likelihood.  For each phone
candidate it reports a conformal survival probability among genuine examples
and the empirical enrichment of the threshold event ``score <= observed``
among genuine versus impostor examples.  A small hierarchical count prior
borrows only the pooled cross-speaker event rate for rare phones.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


FORMAT = "ostensibly_empirical_phone_compatibility_v1"
POLICIES = ("conformity", "event", "geometric", "minimum", "harmonic")


def _wilson_interval(successes: int, count: int, z: float = 1.96) -> tuple[float, float]:
    if count < 1 or not 0 <= successes <= count:
        raise ValueError("Wilson interval requires valid finite counts")
    estimate = successes / count
    z2 = z * z
    denominator = 1.0 + z2 / count
    center = (estimate + z2 / (2.0 * count)) / denominator
    radius = (
        z
        * np.sqrt(
            estimate * (1.0 - estimate) / count + z2 / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def combine_compatibility(
    conformity: float,
    event_probability: float,
    policy: str,
) -> float:
    if policy not in POLICIES:
        raise ValueError("unknown phone compatibility policy")
    if not 0.0 <= conformity <= 1.0 or not 0.0 <= event_probability <= 1.0:
        raise ValueError("compatibility coordinates must be probabilities")
    if policy == "conformity":
        return conformity
    if policy == "event":
        return event_probability
    if policy == "geometric":
        return float(np.sqrt(conformity * event_probability))
    if policy == "minimum":
        return min(conformity, event_probability)
    denominator = conformity + event_probability
    return 0.0 if denominator <= 0.0 else 2.0 * conformity * event_probability / denominator


@dataclass(frozen=True)
class EmpiricalPhoneCompatibility:
    labels: tuple[str, ...]
    calibration_scores: np.ndarray
    calibration_targets: np.ndarray
    prior_strength: float = 8.0
    policy: str = "geometric"
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        scores = np.asarray(self.calibration_scores, dtype=np.float64)
        targets = np.asarray(self.calibration_targets).astype(str)
        label_count = len(self.labels)
        if (
            label_count < 2
            or len(set(self.labels)) != label_count
            or scores.ndim != 2
            or scores.shape[1] != label_count
            or targets.shape != (scores.shape[0],)
            or scores.shape[0] < 2
            or not np.all(np.isfinite(scores))
            or np.any(scores <= 0.0)
            or np.any(scores > 1.0)
            or not set(targets).issubset(self.labels)
            or not np.isfinite(self.prior_strength)
            or self.prior_strength < 0.0
            or self.policy not in POLICIES
        ):
            raise ValueError("empirical phone compatibility has invalid calibration")
        object.__setattr__(self, "calibration_scores", scores)
        object.__setattr__(self, "calibration_targets", targets)

    def _validate_queries(self, query_scores: np.ndarray) -> np.ndarray:
        values = np.asarray(query_scores, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.labels)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError("query phone scores have invalid shape or range")
        return values

    def score_matrix(self, query_scores: np.ndarray) -> np.ndarray:
        """Vectorized compatibility scores for query-by-phone geometry scores."""

        values = self._validate_queries(query_scores)
        label_array = np.asarray(self.labels)
        positive_mask = self.calibration_targets[:, None] == label_array[None, :]
        pooled_positive = self.calibration_scores[positive_mask]
        pooled_negative = self.calibration_scores[~positive_mask]
        sorted_pooled_positive = np.sort(pooled_positive)
        sorted_pooled_negative = np.sort(pooled_negative)
        output = np.empty_like(values)
        for index, label in enumerate(self.labels):
            observed = values[:, index]
            mask = self.calibration_targets == label
            positive = np.sort(self.calibration_scores[mask, index])
            negative = np.sort(self.calibration_scores[~mask, index])
            positive_successes = np.searchsorted(positive, observed, side="right")
            negative_successes = np.searchsorted(negative, observed, side="right")
            pooled_positive_rate = (
                np.searchsorted(sorted_pooled_positive, observed, side="right") + 0.5
            ) / (sorted_pooled_positive.size + 1.0)
            pooled_negative_rate = (
                np.searchsorted(sorted_pooled_negative, observed, side="right") + 0.5
            ) / (sorted_pooled_negative.size + 1.0)
            positive_rate = (
                positive_successes
                + 0.5
                + self.prior_strength * pooled_positive_rate
            ) / (positive.size + 1.0 + self.prior_strength)
            negative_rate = (
                negative_successes
                + 0.5
                + self.prior_strength * pooled_negative_rate
            ) / (negative.size + 1.0 + self.prior_strength)
            event_ratio = positive_rate / np.maximum(negative_rate, 1e-12)
            event_probability = event_ratio / (1.0 + event_ratio)
            if positive.size:
                conformity = (
                    1.0
                    + positive.size
                    - np.searchsorted(positive, observed, side="left")
                ) / (positive.size + 1.0)
            else:
                conformity = (
                    1.0
                    + sorted_pooled_positive.size
                    - np.searchsorted(
                        sorted_pooled_positive, observed, side="left"
                    )
                ) / (sorted_pooled_positive.size + 1.0)
            if self.policy == "conformity":
                output[:, index] = conformity
            elif self.policy == "event":
                output[:, index] = event_probability
            elif self.policy == "geometric":
                output[:, index] = np.sqrt(conformity * event_probability)
            elif self.policy == "minimum":
                output[:, index] = np.minimum(conformity, event_probability)
            else:
                denominator = conformity + event_probability
                output[:, index] = np.divide(
                    2.0 * conformity * event_probability,
                    denominator,
                    out=np.zeros_like(denominator),
                    where=denominator > 0.0,
                )
        return output

    def rank(self, query_scores: np.ndarray) -> list[dict[str, object]]:
        values = self._validate_queries(np.asarray(query_scores)[None, :])[0]
        compatibility_values = self.score_matrix(values[None, :])[0]
        label_array = np.asarray(self.labels)
        positive_mask = self.calibration_targets[:, None] == label_array[None, :]
        pooled_positive = self.calibration_scores[positive_mask]
        pooled_negative = self.calibration_scores[~positive_mask]
        rows = []
        for index, (label, observed, compatibility) in enumerate(
            zip(self.labels, values, compatibility_values, strict=True)
        ):
            mask = self.calibration_targets == label
            positive = self.calibration_scores[mask, index]
            negative = self.calibration_scores[~mask, index]
            positive_successes = int(np.sum(positive <= observed))
            negative_successes = int(np.sum(negative <= observed))
            pooled_positive_rate = (
                float(np.sum(pooled_positive <= observed)) + 0.5
            ) / (pooled_positive.size + 1.0)
            pooled_negative_rate = (
                float(np.sum(pooled_negative <= observed)) + 0.5
            ) / (pooled_negative.size + 1.0)
            positive_rate = (
                positive_successes
                + 0.5
                + self.prior_strength * pooled_positive_rate
            ) / (positive.size + 1.0 + self.prior_strength)
            negative_rate = (
                negative_successes
                + 0.5
                + self.prior_strength * pooled_negative_rate
            ) / (negative.size + 1.0 + self.prior_strength)
            event_ratio = positive_rate / max(negative_rate, 1e-12)
            event_probability = event_ratio / (1.0 + event_ratio)
            conformity_population = positive if positive.size else pooled_positive
            conformity = (
                1.0 + float(np.sum(conformity_population >= observed))
            ) / (conformity_population.size + 1.0)
            positive_interval = (
                _wilson_interval(positive_successes, positive.size)
                if positive.size
                else (0.0, 1.0)
            )
            negative_interval = (
                _wilson_interval(negative_successes, negative.size)
                if negative.size
                else (0.0, 1.0)
            )
            certified_ratio_lower = positive_interval[0] / max(
                negative_interval[1], 1e-12
            )
            rows.append(
                {
                    "phone": label,
                    "score": float(1.0 - compatibility),
                    "compatibility": float(compatibility),
                    "conformity": float(conformity),
                    "event_probability_equal_prior": float(event_probability),
                    "event_ratio": float(event_ratio),
                    "certified_event_ratio_lower_95": float(certified_ratio_lower),
                    "observed_geometry_score": float(observed),
                    "positive_event_count": positive_successes,
                    "positive_count": int(positive.size),
                    "negative_event_count": negative_successes,
                    "negative_count": int(negative.size),
                }
            )
        rows.sort(key=lambda row: (-float(row["compatibility"]), str(row["phone"])))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            row["distance"] = row["score"]
            row["witness"] = "cross-speaker empirical compatibility"
        return rows

    def save(self, path: Path) -> None:
        metadata = {
            "format": FORMAT,
            "labels": list(self.labels),
            "prior_strength": self.prior_strength,
            "policy": self.policy,
            "provenance": dict(self.provenance or {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            calibration_scores=self.calibration_scores.astype(np.float32),
            calibration_targets=self.calibration_targets,
        )


def load_empirical_phone_compatibility(path: Path) -> EmpiricalPhoneCompatibility:
    with np.load(path, allow_pickle=False) as document:
        metadata = json.loads(str(document["metadata"]))
        if metadata.get("format") != FORMAT:
            raise ValueError("unsupported empirical phone compatibility format")
        return EmpiricalPhoneCompatibility(
            labels=tuple(str(value) for value in metadata["labels"]),
            calibration_scores=np.asarray(
                document["calibration_scores"], dtype=np.float64
            ),
            calibration_targets=np.asarray(document["calibration_targets"]).astype(str),
            prior_strength=float(metadata["prior_strength"]),
            policy=str(metadata["policy"]),
            provenance=metadata.get("provenance", {}),
        )
