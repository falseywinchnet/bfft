"""Compiled density-free local ridge support for phone occurrences."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi


FORMAT = "ostensibly_support_phone_atlas_v1"


def support_grid(points: np.ndarray, bins: int, minimum_cell_count: int) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not cloud.size:
        raise ValueError("support atlas requires a nonempty N x 3 cloud")
    if bins < 16 or minimum_cell_count < 1:
        raise ValueError("support atlas resolution is invalid")
    coordinates = np.clip(
        cloud[:, :2], 0.0, np.nextafter(1.0, 0.0)
    )
    cells = np.floor(coordinates * bins).astype(np.int64)
    counts = np.zeros((bins, bins), dtype=np.int32)
    np.add.at(counts, (cells[:, 0], cells[:, 1]), 1)
    grid = counts >= minimum_cell_count
    if not np.any(grid):
        raise ValueError("no support cells survive the requested depth")
    return grid


@dataclass(frozen=True)
class CompiledSupportPhoneAtlas:
    labels: np.ndarray
    witnesses: np.ndarray
    supports: np.ndarray
    distance_fields: np.ndarray
    minimum_cell_count: int
    sigma_cells: float

    def __post_init__(self) -> None:
        count, bins0, bins1 = self.supports.shape
        if (
            count < 1
            or bins0 != bins1
            or self.distance_fields.shape != self.supports.shape
            or self.labels.shape != (count,)
            or self.witnesses.shape != (count,)
            or self.sigma_cells <= 0.0
        ):
            raise ValueError("compiled support atlas has invalid shape")

    @property
    def bins(self) -> int:
        return int(self.supports.shape[1])

    def rank(self, cloud: np.ndarray) -> list[dict[str, object]]:
        query = support_grid(cloud, self.bins, self.minimum_cell_count)
        query_distance = ndi.distance_transform_edt(~query)
        sigma = self.sigma_cells
        query_miss = 1.0 - np.exp(-0.5 * (query_distance / sigma) ** 2)
        reference_miss = 1.0 - np.exp(
            -0.5 * (self.distance_fields[:, query] / sigma) ** 2
        )
        query_to_reference = np.mean(reference_miss, axis=1)
        reference_counts = np.sum(self.supports, axis=(1, 2))
        reference_to_query = np.sum(
            self.supports * query_miss[None, :, :], axis=(1, 2)
        ) / reference_counts
        distances = 0.5 * (query_to_reference + reference_to_query)
        best = {}
        for label, witness, distance in zip(
            self.labels, self.witnesses, distances, strict=True
        ):
            item = {
                "phone": str(label),
                "distance": float(distance),
                "witness": str(witness),
            }
            current = best.get(item["phone"])
            if current is None or (item["distance"], item["witness"]) < (
                current["distance"], current["witness"]
            ):
                best[item["phone"]] = item
        ranking = sorted(best.values(), key=lambda item: (item["distance"], item["phone"]))
        for rank, item in enumerate(ranking, start=1):
            item["rank"] = rank
        return ranking


def compile_support_phone_atlas(
    occurrences: Iterable[tuple[str, str, np.ndarray]],
    bins: int = 64,
    minimum_cell_count: int = 2,
    sigma_cells: float = 1.5,
) -> CompiledSupportPhoneAtlas:
    labels = []
    witnesses = []
    supports = []
    for label, witness, cloud in occurrences:
        labels.append(str(label))
        witnesses.append(str(witness))
        supports.append(support_grid(cloud, bins, minimum_cell_count))
    if not supports:
        raise ValueError("support phone atlas requires occurrences")
    grids = np.stack(supports)
    distances = np.stack(
        [ndi.distance_transform_edt(~grid) for grid in grids]
    ).astype(np.float32)
    return CompiledSupportPhoneAtlas(
        labels=np.asarray(labels),
        witnesses=np.asarray(witnesses),
        supports=grids,
        distance_fields=distances,
        minimum_cell_count=minimum_cell_count,
        sigma_cells=sigma_cells,
    )


def combine_support_phone_atlases(
    atlases: Iterable[CompiledSupportPhoneAtlas],
) -> CompiledSupportPhoneAtlas:
    values = tuple(atlases)
    if not values:
        raise ValueError("support atlas combination requires inputs")
    first = values[0]
    if any(
        atlas.bins != first.bins
        or atlas.minimum_cell_count != first.minimum_cell_count
        or atlas.sigma_cells != first.sigma_cells
        for atlas in values[1:]
    ):
        raise ValueError("support phone atlases use incompatible grids")
    return CompiledSupportPhoneAtlas(
        labels=np.concatenate([atlas.labels for atlas in values]),
        witnesses=np.concatenate([atlas.witnesses for atlas in values]),
        supports=np.concatenate([atlas.supports for atlas in values]),
        distance_fields=np.concatenate([atlas.distance_fields for atlas in values]),
        minimum_cell_count=first.minimum_cell_count,
        sigma_cells=first.sigma_cells,
    )


def save_support_phone_atlas(path: Path, atlas: CompiledSupportPhoneAtlas) -> None:
    metadata = {
        "format": FORMAT,
        "minimum_cell_count": atlas.minimum_cell_count,
        "sigma_cells": atlas.sigma_cells,
    }
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        labels=atlas.labels,
        witnesses=atlas.witnesses,
        supports=atlas.supports,
        distance_fields=atlas.distance_fields.astype(np.float16),
    )


def load_support_phone_atlas(path: Path) -> CompiledSupportPhoneAtlas:
    with np.load(path, allow_pickle=False) as document:
        metadata = json.loads(str(document["metadata"]))
        if metadata.get("format") != FORMAT:
            raise ValueError("unsupported support phone atlas format")
        return CompiledSupportPhoneAtlas(
            labels=np.asarray(document["labels"]),
            witnesses=np.asarray(document["witnesses"]),
            supports=np.asarray(document["supports"], dtype=bool),
            distance_fields=np.asarray(document["distance_fields"], dtype=np.float32),
            minimum_cell_count=int(metadata["minimum_cell_count"]),
            sigma_cells=float(metadata["sigma_cells"]),
        )
