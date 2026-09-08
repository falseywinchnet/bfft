"""Shift-invariant physical ridge-interval geometry.

The conditional ridge signature deliberately rank-gauges the frequency axis.
That is useful for topology but erases the physical spacing between occupied
bands.  This module preserves those spacings while remaining invariant to a
global pitch translation.  It is an experimental complementary coordinate,
not a replacement for conditional topology.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .occurrence_statistics import rank_occurrence_quantiles


def physical_interval_signature(
    points: np.ndarray,
    time_bins: int = 32,
    row_quantiles: int = 16,
    row_scale: float = 128.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered physical-row quantiles and normalized time occupation."""

    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not cloud.size:
        raise ValueError("physical interval signature requires an N x 3 cloud")
    if time_bins < 4 or row_quantiles < 4 or row_scale <= 0.0:
        raise ValueError("physical interval signature configuration is invalid")
    rows = cloud[:, 0] / row_scale
    frame0 = float(np.min(cloud[:, 1]))
    frame_scale = float(np.max(cloud[:, 1]) - frame0)
    frames = (cloud[:, 1] - frame0) / max(frame_scale, 1e-12)
    bins = np.minimum((frames * time_bins).astype(np.int64), time_bins - 1)
    quantiles = (
        np.arange(row_quantiles, dtype=np.float64) + 0.5
    ) / row_quantiles
    surface = np.full((time_bins, row_quantiles), np.nan, dtype=np.float64)
    mass = np.bincount(bins, minlength=time_bins).astype(np.float64)
    mass /= float(np.sum(mass))
    for index in range(time_bins):
        values = rows[bins == index]
        if values.size:
            surface[index] = np.quantile(values, quantiles)
    valid = np.flatnonzero(np.isfinite(surface[:, 0]))
    if not valid.size:
        raise ValueError("physical interval surface contains no occupied slices")
    for column in range(row_quantiles):
        surface[:, column] = np.interp(
            np.arange(time_bins), valid, surface[valid, column]
        )
    return surface, mass


def physical_interval_distance(
    query_surface: np.ndarray,
    query_mass: np.ndarray,
    reference_surface: np.ndarray,
    reference_mass: np.ndarray,
    maximum_shift: int = 2,
    trajectory_weight: float = 0.5,
    mass_weight: float = 0.25,
    interval_weight: float = 1.0,
) -> float:
    """Compare physical band intervals under bounded temporal registration.

    Adjacent frequency-quantile differences preserve band spacing while
    cancelling any global row translation.  The derivative of the median row
    retains pitch motion without reintroducing absolute pitch.
    """

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
        or min(interval_weight, trajectory_weight, mass_weight) < 0.0
        or interval_weight + trajectory_weight + mass_weight <= 0.0
    ):
        raise ValueError("physical interval distance has incompatible signatures")
    time_bins = query.shape[0]
    best = np.inf
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
        intervals = float(
            np.sqrt(
                np.mean(
                    (np.diff(left, axis=1) - np.diff(right, axis=1)) ** 2
                )
            )
        )
        left_median = np.median(left, axis=1)
        right_median = np.median(right, axis=1)
        trajectory = float(
            np.sqrt(
                np.mean(
                    (np.diff(left_median) - np.diff(right_median)) ** 2
                )
            )
        )
        mass = float(
            np.sqrt(
                np.mean(
                    (
                        query_weight[query_slice]
                        - reference_weight[reference_slice]
                    )
                    ** 2
                )
            )
        )
        best = min(
            best,
            interval_weight * intervals
            + trajectory_weight * trajectory
            + mass_weight * mass,
        )
    return float(best)


@dataclass(frozen=True)
class PhysicalIntervalAtlas:
    """Vectorized witness inventory for shift-invariant physical spacing."""

    labels: np.ndarray
    witnesses: np.ndarray
    surfaces: np.ndarray
    masses: np.ndarray
    maximum_shift: int = 2
    trajectory_weight: float = 0.5
    mass_weight: float = 0.25
    row_scale: float = 128.0

    def __post_init__(self) -> None:
        count, time_bins, _ = self.surfaces.shape
        if (
            count < 1
            or self.labels.shape != (count,)
            or self.witnesses.shape != (count,)
            or self.masses.shape != (count, time_bins)
            or self.maximum_shift < 0
            or min(self.trajectory_weight, self.mass_weight) < 0.0
            or self.row_scale <= 0.0
        ):
            raise ValueError("physical interval atlas has invalid geometry")

    @property
    def time_bins(self) -> int:
        return int(self.surfaces.shape[1])

    @property
    def row_quantiles(self) -> int:
        return int(self.surfaces.shape[2])

    def occurrence_distances_with_weights(
        self,
        cloud: np.ndarray,
        *,
        interval_weight: float,
        trajectory_weight: float,
        mass_weight: float,
    ) -> np.ndarray:
        """Measure every occurrence with explicit evidence weights.

        This exposes the three measured coordinates for controlled ablations;
        it does not alter the compiled witness inventory.
        """

        weights = (interval_weight, trajectory_weight, mass_weight)
        if (
            any(not np.isfinite(value) or value < 0.0 for value in weights)
            or sum(weights) <= 0.0
        ):
            raise ValueError("physical evidence weights must be finite and nonzero")
        query, mass = physical_interval_signature(
            cloud,
            self.time_bins,
            self.row_quantiles,
            self.row_scale,
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
            left = query[query_slice]
            right = self.surfaces[:, reference_slice]
            intervals = np.sqrt(
                np.mean(
                    (np.diff(right, axis=2) - np.diff(left, axis=1)[None]) ** 2,
                    axis=(1, 2),
                )
            )
            left_median = np.median(left, axis=1)
            right_median = np.median(right, axis=2)
            trajectory = np.sqrt(
                np.mean(
                    (
                        np.diff(right_median, axis=1)
                        - np.diff(left_median)[None]
                    )
                    ** 2,
                    axis=1,
                )
            )
            mass_distance = np.sqrt(
                np.mean(
                    (self.masses[:, reference_slice] - mass[query_slice][None])
                    ** 2,
                    axis=1,
                )
            )
            candidate = (
                interval_weight * intervals
                + trajectory_weight * trajectory
                + mass_weight * mass_distance
            )
            distances = np.minimum(distances, candidate)
        return distances

    def rank_with_weights(
        self,
        cloud: np.ndarray,
        *,
        interval_weight: float,
        trajectory_weight: float,
        mass_weight: float,
        occurrence_quantile: float = 0.0,
    ) -> list[dict[str, object]]:
        distances = self.occurrence_distances_with_weights(
            cloud,
            interval_weight=interval_weight,
            trajectory_weight=trajectory_weight,
            mass_weight=mass_weight,
        )
        return rank_occurrence_quantiles(
            self.labels,
            self.witnesses,
            distances,
            occurrence_quantile,
        )

    def rank(self, cloud: np.ndarray) -> list[dict[str, object]]:
        return self.rank_with_weights(
            cloud,
            interval_weight=1.0,
            trajectory_weight=self.trajectory_weight,
            mass_weight=self.mass_weight,
        )


def compile_physical_interval_atlas(
    occurrences: Iterable[tuple[str, str, np.ndarray]],
    *,
    time_bins: int = 32,
    row_quantiles: int = 16,
    maximum_shift: int = 2,
    trajectory_weight: float = 0.5,
    mass_weight: float = 0.25,
    row_scale: float = 128.0,
) -> PhysicalIntervalAtlas:
    labels: list[str] = []
    witnesses: list[str] = []
    surfaces: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    for label, witness, cloud in occurrences:
        surface, mass = physical_interval_signature(
            cloud, time_bins, row_quantiles, row_scale
        )
        labels.append(str(label))
        witnesses.append(str(witness))
        surfaces.append(surface)
        masses.append(mass)
    if not surfaces:
        raise ValueError("physical interval atlas requires occurrences")
    return PhysicalIntervalAtlas(
        labels=np.asarray(labels),
        witnesses=np.asarray(witnesses),
        surfaces=np.stack(surfaces).astype(np.float32),
        masses=np.stack(masses).astype(np.float32),
        maximum_shift=maximum_shift,
        trajectory_weight=trajectory_weight,
        mass_weight=mass_weight,
        row_scale=row_scale,
    )


def combine_physical_interval_atlases(
    atlases: Iterable[PhysicalIntervalAtlas],
) -> PhysicalIntervalAtlas:
    values = tuple(atlases)
    if not values:
        raise ValueError("physical interval atlas combination requires inputs")
    first = values[0]
    if any(
        atlas.surfaces.shape[1:] != first.surfaces.shape[1:]
        or atlas.maximum_shift != first.maximum_shift
        or atlas.trajectory_weight != first.trajectory_weight
        or atlas.mass_weight != first.mass_weight
        or atlas.row_scale != first.row_scale
        for atlas in values[1:]
    ):
        raise ValueError("physical interval atlases are incompatible")
    return PhysicalIntervalAtlas(
        labels=np.concatenate([atlas.labels for atlas in values]),
        witnesses=np.concatenate([atlas.witnesses for atlas in values]),
        surfaces=np.concatenate([atlas.surfaces for atlas in values]),
        masses=np.concatenate([atlas.masses for atlas in values]),
        maximum_shift=first.maximum_shift,
        trajectory_weight=first.trajectory_weight,
        mass_weight=first.mass_weight,
        row_scale=first.row_scale,
    )


def load_physical_interval_atlas(path: Path) -> PhysicalIntervalAtlas:
    with np.load(path, allow_pickle=False) as document:
        metadata = json.loads(str(document["metadata"]))
        return PhysicalIntervalAtlas(
            labels=np.asarray(document["labels"]),
            witnesses=np.asarray(document["witnesses"]),
            surfaces=np.asarray(document["surfaces"], dtype=np.float32),
            masses=np.asarray(document["masses"], dtype=np.float32),
            maximum_shift=int(metadata["maximum_shift"]),
            trajectory_weight=float(metadata["trajectory_weight"]),
            mass_weight=float(metadata["mass_weight"]),
            row_scale=float(metadata["row_scale"]),
        )
