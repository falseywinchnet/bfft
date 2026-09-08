"""Label-balanced statistics over occurrence-level geometric distances."""

from __future__ import annotations

import numpy as np


def rank_occurrence_quantiles(
    labels: np.ndarray,
    witnesses: np.ndarray,
    distances: np.ndarray,
    quantile: float = 0.0,
) -> list[dict[str, object]]:
    """Rank labels by the same within-label distance quantile.

    A minimum gives labels with many witnesses more chances to realize an
    accidental extreme. Applying one declared quantile independently to each
    label makes the order statistic comparable despite unequal inventory
    counts. Quantile zero intentionally reproduces the historical reducer.
    """

    label_values = np.asarray(labels).astype(str)
    witness_values = np.asarray(witnesses).astype(str)
    distance_values = np.asarray(distances, dtype=np.float64)
    if (
        label_values.ndim != 1
        or witness_values.shape != label_values.shape
        or distance_values.shape != label_values.shape
        or not label_values.size
        or not np.all(np.isfinite(distance_values))
        or not np.isfinite(quantile)
        or not 0.0 <= quantile <= 1.0
    ):
        raise ValueError("occurrence quantile inputs are invalid")

    rows = []
    for label in sorted(set(label_values)):
        selected = np.flatnonzero(label_values == label)
        value = float(np.quantile(distance_values[selected], quantile))
        representative = min(
            selected,
            key=lambda index: (
                abs(float(distance_values[index]) - value),
                witness_values[index],
            ),
        )
        rows.append({
            "phone": label,
            "distance": value,
            "witness": witness_values[representative],
            "occurrence_count": int(selected.size),
            "occurrence_quantile": float(quantile),
        })
    rows.sort(key=lambda item: (item["distance"], item["phone"]))
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank
    return rows


def rank_occurrence_dominance(
    labels: np.ndarray,
    witnesses: np.ndarray,
    distances: np.ndarray,
) -> list[dict[str, object]]:
    """Rank labels by exact occurrence-vs-background stochastic dominance.

    The score is ``P(D_label < D_other) + 0.5 P(D_label = D_other)``. It uses
    every observed occurrence while giving each label one expectation rather
    than one extreme-value lottery. Higher dominance is better.
    """

    label_values = np.asarray(labels).astype(str)
    witness_values = np.asarray(witnesses).astype(str)
    distance_values = np.asarray(distances, dtype=np.float64)
    if (
        label_values.ndim != 1
        or witness_values.shape != label_values.shape
        or distance_values.shape != label_values.shape
        or not label_values.size
        or not np.all(np.isfinite(distance_values))
    ):
        raise ValueError("occurrence dominance inputs are invalid")

    rows = []
    for label in sorted(set(label_values)):
        selected = label_values == label
        within = distance_values[selected]
        background = np.sort(distance_values[~selected])
        if not background.size:
            raise ValueError("occurrence dominance needs a non-label background")
        below_or_equal = np.searchsorted(background, within, side="right")
        strictly_below = np.searchsorted(background, within, side="left")
        greater = background.size - below_or_equal
        equal = below_or_equal - strictly_below
        dominance = float(np.mean((greater + 0.5 * equal) / background.size))
        representative_indices = np.flatnonzero(selected)
        representative = min(
            representative_indices,
            key=lambda index: (distance_values[index], witness_values[index]),
        )
        rows.append({
            "phone": label,
            "distance": 1.0 - dominance,
            "dominance_probability": dominance,
            "witness": witness_values[representative],
            "occurrence_count": int(within.size),
        })
    rows.sort(key=lambda item: (item["distance"], item["phone"]))
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank
    return rows


def rank_occurrence_corrected_minimum(
    labels: np.ndarray,
    witnesses: np.ndarray,
    distances: np.ndarray,
) -> list[dict[str, object]]:
    """Rank closest witnesses by an exact unequal-count null correction.

    For each label, the observed minimum is located in the empirical non-label
    distance background. If that one-draw CDF value is ``F`` and the label had
    ``n`` witnesses, ``1 - (1 - F)**n`` is the chance that at least one of ``n``
    exchangeable null draws would be as close. Lower corrected chance is better.
    """

    label_values = np.asarray(labels).astype(str)
    witness_values = np.asarray(witnesses).astype(str)
    distance_values = np.asarray(distances, dtype=np.float64)
    if (
        label_values.ndim != 1
        or witness_values.shape != label_values.shape
        or distance_values.shape != label_values.shape
        or not label_values.size
        or not np.all(np.isfinite(distance_values))
    ):
        raise ValueError("corrected minimum inputs are invalid")

    rows = []
    for label in sorted(set(label_values)):
        selected = label_values == label
        indices = np.flatnonzero(selected)
        representative = min(
            indices,
            key=lambda index: (distance_values[index], witness_values[index]),
        )
        minimum = float(distance_values[representative])
        background = np.sort(distance_values[~selected])
        if not background.size:
            raise ValueError("corrected minimum needs a non-label background")
        left = int(np.searchsorted(background, minimum, side="left"))
        right = int(np.searchsorted(background, minimum, side="right"))
        one_draw_cdf = (left + 0.5 * (right - left)) / background.size
        count = int(indices.size)
        corrected = (
            1.0
            if one_draw_cdf >= 1.0
            else float(-np.expm1(count * np.log1p(-one_draw_cdf)))
        )
        rows.append({
            "phone": label,
            "distance": corrected,
            "corrected_minimum_probability": corrected,
            "minimum_distance": minimum,
            "one_draw_background_cdf": float(one_draw_cdf),
            "witness": witness_values[representative],
            "occurrence_count": count,
        })
    rows.sort(key=lambda item: (item["distance"], item["minimum_distance"], item["phone"]))
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank
    return rows


def rank_routed_occurrence_minimum(
    labels: np.ndarray,
    witnesses: np.ndarray,
    distances: np.ndarray,
    route_values: np.ndarray,
    query_route_value: float,
    maximum_occurrences: int | None,
) -> list[dict[str, object]]:
    """Route within each label by independent evidence, then rank geometry.

    ``route_values`` decide only which occurrences may be compared. The output
    distance is the unchanged minimum geometric distance over that subset.
    """

    label_values = np.asarray(labels).astype(str)
    witness_values = np.asarray(witnesses).astype(str)
    distance_values = np.asarray(distances, dtype=np.float64)
    route = np.asarray(route_values, dtype=np.float64)
    if (
        label_values.ndim != 1
        or witness_values.shape != label_values.shape
        or distance_values.shape != label_values.shape
        or route.shape != label_values.shape
        or not label_values.size
        or not np.all(np.isfinite(distance_values))
        or not np.all(np.isfinite(route))
        or not np.isfinite(query_route_value)
        or (maximum_occurrences is not None and maximum_occurrences < 1)
    ):
        raise ValueError("routed occurrence inputs are invalid")

    rows = []
    for label in sorted(set(label_values)):
        indices = np.flatnonzero(label_values == label)
        ordered = sorted(
            indices,
            key=lambda index: (
                abs(float(route[index]) - query_route_value),
                witness_values[index],
            ),
        )
        retained = np.asarray(
            ordered[:maximum_occurrences] if maximum_occurrences else ordered,
            dtype=np.int64,
        )
        representative = min(
            retained,
            key=lambda index: (distance_values[index], witness_values[index]),
        )
        rows.append({
            "phone": label,
            "distance": float(distance_values[representative]),
            "witness": witness_values[representative],
            "occurrence_count": int(indices.size),
            "routed_occurrence_count": int(retained.size),
            "route_residual": abs(
                float(route[representative]) - query_route_value
            ),
        })
    rows.sort(key=lambda item: (item["distance"], item["phone"]))
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank
    return rows


def union_occurrence_proposals(
    minimum_ranking: list[dict[str, object]],
    balanced_ranking: list[dict[str, object]],
    *,
    minimum_depth: int,
    balanced_depth: int,
) -> list[dict[str, object]]:
    """Retain either ranking's top-k while preserving separate provenance."""

    if minimum_depth < 1 or balanced_depth < 1:
        raise ValueError("occurrence proposal depths must be positive")
    minimum = {str(row["phone"]): dict(row) for row in minimum_ranking}
    balanced = {str(row["phone"]): dict(row) for row in balanced_ranking}
    if minimum.keys() != balanced.keys() or not minimum:
        raise ValueError("occurrence proposal rankings disagree on labels")
    label_count = len(minimum)
    for ranking in (minimum, balanced):
        if sorted(int(row["rank"]) for row in ranking.values()) != list(
            range(1, label_count + 1)
        ):
            raise ValueError("occurrence proposal ranking is not a permutation")

    output = []
    for phone, row in minimum.items():
        minimum_rank = int(row["rank"])
        balanced_row = balanced[phone]
        balanced_rank = int(balanced_row["rank"])
        channels = []
        if minimum_rank <= minimum_depth:
            channels.append("geometry:occurrence_minimum")
        if balanced_rank <= balanced_depth:
            channels.append("geometry:balanced_tail")
        if not channels:
            continue
        row["balanced_tail_rank"] = balanced_rank
        if "score" in balanced_row:
            row["balanced_tail_score"] = float(balanced_row["score"])
        for name in (
            "empirical_target_frequency_at_rank",
            "cumulative_target_coverage",
            "cumulative_target_coverage_lower_95",
            "cumulative_target_coverage_upper_95",
        ):
            if name in balanced_row:
                row[f"balanced_tail_{name}"] = balanced_row[name]
        row["geometry_proposal_channels"] = channels
        output.append(row)
    output.sort(key=lambda row: (int(row["rank"]), str(row["phone"])))
    return output
