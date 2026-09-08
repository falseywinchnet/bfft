"""Unlabeled population alignment for physical ridge-interval geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def positive_adjacent_intervals(surfaces: np.ndarray) -> np.ndarray:
    values = np.asarray(surfaces, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] < 2:
        raise ValueError("interval surfaces must be count x time x quantile")
    intervals = np.diff(values, axis=2).reshape(-1)
    return intervals[np.isfinite(intervals) & (intervals > 1e-9)]


@dataclass(frozen=True)
class PopulationIntervalAlignment:
    row_multiplier: float
    reference_count: int
    query_count: int
    log_median_reference: float
    log_median_query: float
    fitted_log_shift: float
    quantile_rms_before: float
    quantile_rms_after: float


def estimate_population_row_multiplier(
    reference_surfaces: np.ndarray,
    query_surfaces: np.ndarray,
    *,
    quantile_count: int = 257,
) -> PopulationIntervalAlignment:
    """Project the query distribution onto the reference in log-Wasserstein.

    The estimator consumes no phone identities.  A global row translation has
    already disappeared under adjacent differences; the only fitted degree of
    freedom is one multiplicative row-unit conversion shared by every query.
    """

    if quantile_count < 17:
        raise ValueError("population alignment needs at least 17 quantiles")
    reference = positive_adjacent_intervals(reference_surfaces)
    query = positive_adjacent_intervals(query_surfaces)
    if min(reference.size, query.size) < quantile_count:
        raise ValueError("insufficient interval population")
    log_reference = np.log(reference)
    log_query = np.log(query)
    reference_center = float(np.median(log_reference))
    query_center = float(np.median(log_query))
    probabilities = np.linspace(0.01, 0.99, quantile_count)
    reference_quantiles = np.quantile(log_reference, probabilities)
    query_quantiles = np.quantile(log_query, probabilities)
    # For a fixed quantile coupling, the least-squares translation has the
    # closed form mean(reference_quantile - query_quantile).  This uses the
    # whole trimmed distribution; matching only the median can increase the
    # actual population discrepancy when quantized tails differ.
    fitted_log_shift = float(np.mean(reference_quantiles - query_quantiles))
    multiplier = float(np.exp(fitted_log_shift))
    before = float(
        np.sqrt(np.mean((query_quantiles - reference_quantiles) ** 2))
    )
    after = float(
        np.sqrt(
            np.mean(
                (
                    query_quantiles
                    + np.log(multiplier)
                    - reference_quantiles
                )
                ** 2
            )
        )
    )
    return PopulationIntervalAlignment(
        row_multiplier=multiplier,
        reference_count=int(reference.size),
        query_count=int(query.size),
        log_median_reference=reference_center,
        log_median_query=query_center,
        fitted_log_shift=fitted_log_shift,
        quantile_rms_before=before,
        quantile_rms_after=after,
    )
