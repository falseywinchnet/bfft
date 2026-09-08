"""Exchangeable top-k coverage law for a fixed phone ranking channel."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


FORMAT = "ostensibly_phone_rank_coverage_v1"


def wilson_interval(successes: int, count: int, z: float = 1.96) -> tuple[float, float]:
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


@dataclass(frozen=True)
class PhoneRankCoverage:
    counts: np.ndarray
    provenance: dict[str, object] | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.counts, dtype=np.int64)
        if values.ndim != 1 or values.size < 2 or np.any(values < 0) or np.sum(values) < 1:
            raise ValueError("phone rank coverage has invalid counts")
        object.__setattr__(self, "counts", values)

    @property
    def label_count(self) -> int:
        return int(self.counts.size)

    @property
    def query_count(self) -> int:
        return int(np.sum(self.counts))

    def coverage(self, rank: int) -> dict[str, float | int]:
        if not 1 <= rank <= self.label_count:
            raise ValueError("coverage rank is outside the phone inventory")
        successes = int(np.sum(self.counts[:rank]))
        lower, upper = wilson_interval(successes, self.query_count)
        return {
            "rank": rank,
            "successes": successes,
            "count": self.query_count,
            "estimate": successes / self.query_count,
            "lower_95": lower,
            "upper_95": upper,
        }

    def minimum_rank_for_lower_coverage(self, target: float) -> int:
        if not 0.0 < target < 1.0:
            raise ValueError("target coverage must lie strictly within (0, 1)")
        for rank in range(1, self.label_count + 1):
            if float(self.coverage(rank)["lower_95"]) >= target:
                return rank
        return self.label_count

    def annotate(
        self, ranking: Iterable[dict[str, object]]
    ) -> list[dict[str, object]]:
        rows = [dict(row) for row in ranking]
        ranks = sorted(int(row["rank"]) for row in rows)
        if ranks != list(range(1, self.label_count + 1)):
            raise ValueError("ranking does not match coverage inventory")
        for row in rows:
            rank = int(row["rank"])
            coverage = self.coverage(rank)
            row["empirical_target_frequency_at_rank"] = float(
                self.counts[rank - 1] / self.query_count
            )
            row["cumulative_target_coverage"] = float(coverage["estimate"])
            row["cumulative_target_coverage_lower_95"] = float(
                coverage["lower_95"]
            )
            row["cumulative_target_coverage_upper_95"] = float(
                coverage["upper_95"]
            )
        return rows

    def save(self, path: Path) -> None:
        metadata = {
            "format": FORMAT,
            "provenance": dict(self.provenance or {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            counts=self.counts,
        )


def fit_phone_rank_coverage(
    ranks: Iterable[int], label_count: int, *, provenance: dict[str, object] | None = None
) -> PhoneRankCoverage:
    values = np.asarray(tuple(ranks), dtype=np.int64)
    if (
        label_count < 2
        or values.ndim != 1
        or not values.size
        or np.any(values < 1)
        or np.any(values > label_count)
    ):
        raise ValueError("target ranks do not match phone inventory")
    return PhoneRankCoverage(
        np.bincount(values - 1, minlength=label_count), provenance=provenance
    )


def load_phone_rank_coverage(path: Path) -> PhoneRankCoverage:
    with np.load(path, allow_pickle=False) as document:
        metadata = json.loads(str(document["metadata"]))
        if metadata.get("format") != FORMAT:
            raise ValueError("unsupported phone rank coverage format")
        return PhoneRankCoverage(
            np.asarray(document["counts"], dtype=np.int64),
            provenance=metadata.get("provenance", {}),
        )
