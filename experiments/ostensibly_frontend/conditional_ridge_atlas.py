"""Ordered conditional ridge-trajectory signatures for phone clouds."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata

from .occurrence_statistics import rank_occurrence_quantiles


FORMAT = "ostensibly_conditional_ridge_atlas_v1"


def conditional_ridge_signature(
    points: np.ndarray,
    time_bins: int = 32,
    row_quantiles: int = 16,
    saliency_power: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure row occupation conditionally in ordered physical-time slices.

    The cloud's empirical multiplicity already represents square-root field
    mass.  A positive ``saliency_power`` additionally weights each sample by
    its stored height.  In particular, ``0.5`` approximately restores the
    original occupation-field measure without having to reconstruct the
    raster from the finite cloud.
    """

    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not cloud.size:
        raise ValueError("conditional ridge signature requires an N x 3 cloud")
    if time_bins < 4 or row_quantiles < 4:
        raise ValueError("conditional ridge signature resolution is too small")
    if not np.isfinite(saliency_power) or saliency_power < 0.0:
        raise ValueError("saliency power must be finite and nonnegative")
    count = cloud.shape[0]
    if saliency_power == 0.0:
        weights = np.ones(count, dtype=np.float64)
        rows = (rankdata(cloud[:, 0], method="average") - 0.5) / count
    else:
        weights = np.power(np.maximum(cloud[:, 2], 0.0), saliency_power)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            raise ValueError("saliency-weighted signature has no positive mass")
        order = np.argsort(cloud[:, 0], kind="stable")
        sorted_rows = cloud[order, 0]
        sorted_weights = weights[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_rows) != 0.0) + 1]
        stops = np.r_[starts[1:], count]
        group_weights = np.add.reduceat(sorted_weights, starts)
        group_midpoints = (
            np.cumsum(group_weights) - 0.5 * group_weights
        ) / total_weight
        sorted_ranks = np.empty(count, dtype=np.float64)
        for start, stop, midpoint in zip(
            starts, stops, group_midpoints, strict=True
        ):
            sorted_ranks[start:stop] = midpoint
        rows = np.empty(count, dtype=np.float64)
        rows[order] = sorted_ranks
    frame0 = float(np.min(cloud[:, 1]))
    frame_scale = float(np.max(cloud[:, 1]) - frame0)
    frames = (cloud[:, 1] - frame0) / max(frame_scale, 1e-12)
    bins = np.minimum((frames * time_bins).astype(np.int64), time_bins - 1)
    quantiles = (np.arange(row_quantiles, dtype=np.float64) + 0.5) / row_quantiles
    surface = np.full((time_bins, row_quantiles), np.nan, dtype=np.float64)
    mass = np.bincount(
        bins, weights=weights, minlength=time_bins
    ).astype(np.float64)
    mass /= max(float(np.sum(mass)), 1.0)
    for index in range(time_bins):
        selected = bins == index
        values = rows[selected]
        if values.size:
            if saliency_power == 0.0:
                surface[index] = np.quantile(values, quantiles)
            else:
                local_weights = weights[selected]
                order = np.argsort(values, kind="stable")
                ordered_values = values[order]
                ordered_weights = local_weights[order]
                locations = (
                    np.cumsum(ordered_weights) - 0.5 * ordered_weights
                ) / np.sum(ordered_weights)
                surface[index] = np.interp(
                    quantiles,
                    locations,
                    ordered_values,
                    left=ordered_values[0],
                    right=ordered_values[-1],
                )
    valid = np.flatnonzero(np.isfinite(surface[:, 0]))
    if not valid.size:
        raise ValueError("conditional ridge surface contains no occupied slices")
    for column in range(row_quantiles):
        surface[:, column] = np.interp(
            np.arange(time_bins), valid, surface[valid, column]
        )
    return surface, mass


def conditional_ridge_distance(
    query_surface: np.ndarray,
    query_mass: np.ndarray,
    reference_surface: np.ndarray,
    reference_mass: np.ndarray,
    maximum_shift: int = 2,
    derivative_weight: float = 0.5,
    mass_weight: float = 0.25,
) -> float:
    """Compare two ordered signatures under the atlas's bounded time shift."""

    query = np.asarray(query_surface, dtype=np.float64)
    reference = np.asarray(reference_surface, dtype=np.float64)
    query_weight = np.asarray(query_mass, dtype=np.float64)
    reference_weight = np.asarray(reference_mass, dtype=np.float64)
    if (
        query.ndim != 2
        or reference.shape != query.shape
        or query_weight.shape != (query.shape[0],)
        or reference_weight.shape != query_weight.shape
        or maximum_shift < 0
        or min(derivative_weight, mass_weight) < 0.0
    ):
        raise ValueError("conditional ridge distance has incompatible signatures")
    time_bins = query.shape[0]
    distance = np.inf
    for shift in range(-maximum_shift, maximum_shift + 1):
        if shift < 0:
            query_slice = slice(-shift, time_bins)
            reference_slice = slice(0, time_bins + shift)
        elif shift > 0:
            query_slice = slice(0, time_bins - shift)
            reference_slice = slice(shift, time_bins)
        else:
            query_slice = reference_slice = slice(None)
        left = query[query_slice]
        right = reference[reference_slice]
        base = float(np.sqrt(np.mean((right - left) ** 2)))
        derivative = float(
            np.sqrt(np.mean((np.diff(right, axis=0) - np.diff(left, axis=0)) ** 2))
        )
        mass = float(
            np.sqrt(
                np.mean(
                    (
                        reference_weight[reference_slice]
                        - query_weight[query_slice]
                    )
                    ** 2
                )
            )
        )
        distance = min(
            distance,
            base + derivative_weight * derivative + mass_weight * mass,
        )
    return float(distance)


@dataclass(frozen=True)
class ConditionalRidgeAtlas:
    labels: np.ndarray
    witnesses: np.ndarray
    surfaces: np.ndarray
    masses: np.ndarray
    maximum_shift: int
    derivative_weight: float
    mass_weight: float

    def __post_init__(self) -> None:
        count, time_bins, _ = self.surfaces.shape
        if (
            count < 1
            or self.masses.shape != (count, time_bins)
            or self.labels.shape != (count,)
            or self.witnesses.shape != (count,)
            or self.maximum_shift < 0
            or min(self.derivative_weight, self.mass_weight) < 0.0
        ):
            raise ValueError("conditional ridge atlas has invalid shape")

    @property
    def time_bins(self) -> int:
        return int(self.surfaces.shape[1])

    @property
    def row_quantiles(self) -> int:
        return int(self.surfaces.shape[2])

    def occurrence_distances(self, cloud: np.ndarray) -> np.ndarray:
        """Measure the query against every retained reference occurrence."""

        surface, mass = conditional_ridge_signature(
            cloud, self.time_bins, self.row_quantiles
        )
        distances = np.full(self.surfaces.shape[0], np.inf, dtype=np.float64)
        for shift in range(-self.maximum_shift, self.maximum_shift + 1):
            if shift < 0:
                query_slice = slice(-shift, self.time_bins)
                reference_slice = slice(0, self.time_bins + shift)
            elif shift > 0:
                query_slice = slice(0, self.time_bins - shift)
                reference_slice = slice(shift, self.time_bins)
            else:
                query_slice = reference_slice = slice(None)
            query = surface[query_slice]
            reference = self.surfaces[:, reference_slice]
            base = np.sqrt(np.mean((reference - query[None, :, :]) ** 2, axis=(1, 2)))
            query_delta = np.diff(query, axis=0)
            reference_delta = np.diff(reference, axis=1)
            derivative = np.sqrt(
                np.mean((reference_delta - query_delta[None, :, :]) ** 2, axis=(1, 2))
            )
            mass_distance = np.sqrt(
                np.mean(
                    (self.masses[:, reference_slice] - mass[query_slice][None, :]) ** 2,
                    axis=1,
                )
            )
            candidate = base + self.derivative_weight * derivative + self.mass_weight * mass_distance
            distances = np.minimum(distances, candidate)
        return distances

    def rank(
        self, cloud: np.ndarray, occurrence_quantile: float = 0.0
    ) -> list[dict[str, object]]:
        return rank_occurrence_quantiles(
            self.labels,
            self.witnesses,
            self.occurrence_distances(cloud),
            occurrence_quantile,
        )


def compile_conditional_ridge_atlas(
    occurrences: Iterable[tuple[str, str, np.ndarray]],
    time_bins: int = 32,
    row_quantiles: int = 16,
    maximum_shift: int = 2,
    derivative_weight: float = 0.5,
    mass_weight: float = 0.25,
) -> ConditionalRidgeAtlas:
    labels = []
    witnesses = []
    surfaces = []
    masses = []
    for label, witness, cloud in occurrences:
        surface, mass = conditional_ridge_signature(cloud, time_bins, row_quantiles)
        labels.append(str(label)); witnesses.append(str(witness)); surfaces.append(surface); masses.append(mass)
    if not surfaces:
        raise ValueError("conditional ridge atlas requires occurrences")
    return ConditionalRidgeAtlas(
        labels=np.asarray(labels), witnesses=np.asarray(witnesses),
        surfaces=np.stack(surfaces).astype(np.float32), masses=np.stack(masses).astype(np.float32),
        maximum_shift=maximum_shift, derivative_weight=derivative_weight, mass_weight=mass_weight,
    )


def combine_conditional_ridge_atlases(
    atlases: Iterable[ConditionalRidgeAtlas],
) -> ConditionalRidgeAtlas:
    values = tuple(atlases)
    if not values:
        raise ValueError("conditional ridge atlas combination requires inputs")
    first = values[0]
    if any(
        atlas.surfaces.shape[1:] != first.surfaces.shape[1:]
        or atlas.maximum_shift != first.maximum_shift
        or atlas.derivative_weight != first.derivative_weight
        or atlas.mass_weight != first.mass_weight
        for atlas in values[1:]
    ):
        raise ValueError("conditional ridge atlases are incompatible")
    return ConditionalRidgeAtlas(
        labels=np.concatenate([atlas.labels for atlas in values]),
        witnesses=np.concatenate([atlas.witnesses for atlas in values]),
        surfaces=np.concatenate([atlas.surfaces for atlas in values]),
        masses=np.concatenate([atlas.masses for atlas in values]),
        maximum_shift=first.maximum_shift,
        derivative_weight=first.derivative_weight,
        mass_weight=first.mass_weight,
    )


def save_conditional_ridge_atlas(path: Path, atlas: ConditionalRidgeAtlas) -> None:
    metadata = {"format": FORMAT, "maximum_shift": atlas.maximum_shift, "derivative_weight": atlas.derivative_weight, "mass_weight": atlas.mass_weight}
    np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), labels=atlas.labels, witnesses=atlas.witnesses, surfaces=atlas.surfaces, masses=atlas.masses)


def load_conditional_ridge_atlas(path: Path) -> ConditionalRidgeAtlas:
    with np.load(path, allow_pickle=False) as document:
        metadata=json.loads(str(document["metadata"]))
        if metadata.get("format") != FORMAT: raise ValueError("unsupported conditional ridge atlas format")
        return ConditionalRidgeAtlas(
            labels=np.asarray(document["labels"]), witnesses=np.asarray(document["witnesses"]),
            surfaces=np.asarray(document["surfaces"], dtype=np.float32), masses=np.asarray(document["masses"], dtype=np.float32),
            maximum_shift=int(metadata["maximum_shift"]), derivative_weight=float(metadata["derivative_weight"]), mass_weight=float(metadata["mass_weight"]),
        )
