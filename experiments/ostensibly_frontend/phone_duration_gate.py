"""Robust phone-duration proposals that gate, but never score, geometry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FORMAT = "ostensibly_phone_duration_gate_v1"


@dataclass(frozen=True)
class PhoneDurationGate:
    """An occurrence inventory used to propose duration-compatible labels."""

    labels: np.ndarray
    log_durations: np.ndarray
    minimum_scale: float = 0.12

    def __post_init__(self) -> None:
        if (
            self.labels.ndim != 1
            or self.log_durations.shape != self.labels.shape
            or not self.labels.size
            or self.minimum_scale <= 0.0
            or not np.all(np.isfinite(self.log_durations))
        ):
            raise ValueError("phone duration gate has invalid occurrences")

    def label_statistics(self) -> dict[str, tuple[float, float]]:
        """Return robust log-location and log-MAD for each phone label."""

        statistics = {}
        for label in sorted(set(str(value) for value in self.labels)):
            values = self.log_durations[self.labels.astype(str) == label]
            location = float(np.median(values))
            scale = max(
                float(np.median(np.abs(values - location))), self.minimum_scale
            )
            statistics[label] = (location, scale)
        return statistics

    def allowed_labels(self, duration: float, label_count: int) -> tuple[str, ...]:
        """Return the nearest robust duration envelopes without scoring identity."""

        statistics = self.label_statistics()
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("phone duration must be finite and positive")
        if not 1 <= label_count <= len(statistics):
            raise ValueError("duration proposal count is outside the label inventory")
        return self.rank_labels(duration)[:label_count]

    def rank_labels(self, duration: float) -> tuple[str, ...]:
        """Order duration envelopes for routing, never for identity scoring."""

        statistics = self.label_statistics()
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("phone duration must be finite and positive")
        value = float(np.log(duration))
        ordered = sorted(
            statistics,
            key=lambda label: (
                abs(value - statistics[label][0]) / statistics[label][1],
                label,
            ),
        )
        return tuple(ordered)

    def filter_ranking(
        self,
        duration: float,
        ranking: Sequence[dict[str, object]],
        label_count: int,
    ) -> list[dict[str, object]]:
        """Keep geometric order among labels admitted by duration evidence."""

        allowed = frozenset(self.allowed_labels(duration, label_count))
        return [dict(item) for item in ranking if str(item["phone"]) in allowed]


def compile_phone_duration_gate(
    occurrences: Iterable[tuple[str, float]], minimum_scale: float = 0.12
) -> PhoneDurationGate:
    labels = []
    values = []
    for label, duration in occurrences:
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("phone duration occurrences must be finite and positive")
        labels.append(str(label))
        values.append(float(np.log(duration)))
    if not labels:
        raise ValueError("phone duration gate requires occurrences")
    return PhoneDurationGate(
        labels=np.asarray(labels),
        log_durations=np.asarray(values, dtype=np.float64),
        minimum_scale=minimum_scale,
    )


def combine_phone_duration_gates(
    gates: Iterable[PhoneDurationGate],
) -> PhoneDurationGate:
    values = tuple(gates)
    if not values:
        raise ValueError("phone duration gate combination requires inputs")
    minimum_scale = values[0].minimum_scale
    if any(gate.minimum_scale != minimum_scale for gate in values[1:]):
        raise ValueError("phone duration gates use incompatible robust scales")
    return PhoneDurationGate(
        labels=np.concatenate([gate.labels for gate in values]),
        log_durations=np.concatenate([gate.log_durations for gate in values]),
        minimum_scale=minimum_scale,
    )


def save_phone_duration_gate(path: Path, gate: PhoneDurationGate) -> None:
    metadata = json.dumps(
        {"format": FORMAT, "minimum_scale": gate.minimum_scale}, sort_keys=True
    )
    np.savez_compressed(
        path,
        metadata=np.asarray(metadata),
        labels=gate.labels,
        log_durations=gate.log_durations,
    )


def load_phone_duration_gate(path: Path) -> PhoneDurationGate:
    with np.load(path, allow_pickle=False) as document:
        metadata = json.loads(str(document["metadata"]))
        if metadata.get("format") != FORMAT:
            raise ValueError("unsupported phone duration gate format")
        return PhoneDurationGate(
            labels=np.asarray(document["labels"]),
            log_durations=np.asarray(document["log_durations"], dtype=np.float64),
            minimum_scale=float(metadata["minimum_scale"]),
        )
