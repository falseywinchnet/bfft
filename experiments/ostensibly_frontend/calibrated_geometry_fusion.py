"""Label-blind calibration of complementary phone-geometry ranks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def rank_percentiles(
    ranking: Iterable[Mapping[str, object]],
) -> dict[str, float]:
    rows = tuple(ranking)
    if not rows:
        raise ValueError("ranking must not be empty")
    count = len(rows)
    result = {
        str(row["phone"]): (int(row["rank"]) - 0.5) / count for row in rows
    }
    if len(result) != count:
        raise ValueError("ranking contains duplicate phone labels")
    return result


def fuse_rank_channels(
    topology: Iterable[Mapping[str, object]],
    physical: Iterable[Mapping[str, object]],
    physical_weight: float,
    policy: str = "arithmetic",
) -> list[dict[str, object]]:
    """Fuse empirical rank coordinates, never incomparable raw distances."""

    if not np.isfinite(physical_weight) or physical_weight < 0.0:
        raise ValueError("physical weight must be finite and nonnegative")
    if policy not in ("arithmetic", "geometric", "minimum", "maximum"):
        raise ValueError("unknown geometry fusion policy")
    top = rank_percentiles(topology)
    spacing = rank_percentiles(physical)
    if top.keys() != spacing.keys():
        raise ValueError("geometry channels have different phone inventories")
    normalizer = 1.0 + physical_weight
    def score(phone: str) -> float:
        if physical_weight == 0.0:
            return top[phone]
        if policy == "arithmetic":
            return (top[phone] + physical_weight * spacing[phone]) / normalizer
        if policy == "geometric":
            return float(
                np.exp(
                    (np.log(top[phone]) + physical_weight * np.log(spacing[phone]))
                    / normalizer
                )
            )
        if policy == "minimum":
            return min(top[phone], spacing[phone])
        return max(top[phone], spacing[phone])
    rows = [
        {
            "phone": phone,
            "score": score(phone),
            "topology_percentile": top[phone],
            "physical_percentile": spacing[phone],
        }
        for phone in top
    ]
    rows.sort(key=lambda row: (float(row["score"]), str(row["phone"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def target_rank(
    topology: Iterable[Mapping[str, object]],
    physical: Iterable[Mapping[str, object]],
    target: str,
    physical_weight: float,
    policy: str = "arithmetic",
) -> int:
    return next(
        int(row["rank"])
        for row in fuse_rank_channels(
            topology, physical, physical_weight, policy=policy
        )
        if row["phone"] == target
    )


def rank_summary(ranks: Iterable[int]) -> dict[str, float | int]:
    values = np.asarray(tuple(ranks), dtype=np.int64)
    if values.ndim != 1 or not values.size or np.any(values < 1):
        raise ValueError("ranks must be positive and nonempty")
    return {
        "count": int(values.size),
        "top1": int(np.sum(values == 1)),
        "top5": int(np.sum(values <= 5)),
        "top10": int(np.sum(values <= 10)),
        "mean_reciprocal_rank": float(np.mean(1.0 / values)),
        "median_rank": float(np.median(values)),
        "mean_rank": float(np.mean(values)),
    }


def calibration_objective(summary: Mapping[str, float | int]) -> tuple[float, ...]:
    """Higher retrieval first, then lower median/mean rank."""

    return (
        -float(summary["top1"]),
        -float(summary["top5"]),
        -float(summary["top10"]),
        -float(summary["mean_reciprocal_rank"]),
        float(summary["median_rank"]),
        float(summary["mean_rank"]),
    )
