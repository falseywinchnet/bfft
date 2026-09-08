"""Reusable fixed-quantile SWD projections for reference diphone occurrences."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .occupation_point_cloud import (
    CloudFitConfig,
    resample_sliced_wasserstein_projection,
    sliced_wasserstein_projection,
)


ATLAS_FORMAT = "ostensibly_boundary_swd_atlas_v1"


@dataclass(frozen=True)
class CompiledBoundaryAtlas:
    phones: np.ndarray
    witnesses: np.ndarray
    projections: np.ndarray
    quantile_count: int
    projection_count: int
    projection_seed: int
    metric_scales: tuple[float, float, float]
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        occurrence_count = self.projections.shape[0]
        if (
            self.projections.ndim != 3
            or self.projections.shape[1] != self.quantile_count
            or self.projections.shape[2] != self.projection_count + 3
        ):
            raise ValueError("compiled atlas projection tensor has invalid shape")
        if self.phones.shape != (occurrence_count, 2):
            raise ValueError("compiled atlas phone table has invalid shape")
        if self.witnesses.shape != (occurrence_count,):
            raise ValueError("compiled atlas witness table has invalid shape")
        if occurrence_count < 1:
            raise ValueError("compiled atlas cannot be empty")

    def distance_config(self) -> CloudFitConfig:
        return CloudFitConfig(
            distance_mode="sliced_wasserstein",
            row_metric_scale=self.metric_scales[0],
            frame_metric_scale=self.metric_scales[1],
            height_metric_scale=self.metric_scales[2],
            sliced_projection_count=self.projection_count,
            sliced_projection_seed=self.projection_seed,
        )

    def query_projection(self, cloud: np.ndarray) -> np.ndarray:
        return resample_sliced_wasserstein_projection(
            sliced_wasserstein_projection(cloud, self.distance_config()),
            self.quantile_count,
        )

    def rank(
        self, query_cloud: np.ndarray, chunk_size: int = 64
    ) -> list[dict[str, object]]:
        """Rank pair types by their nearest compiled occurrence."""

        if chunk_size < 1:
            raise ValueError("atlas query chunk size must be positive")
        query = self.query_projection(query_cloud).astype(np.float32, copy=False)
        distances = np.empty(self.projections.shape[0], dtype=np.float64)
        for start in range(0, self.projections.shape[0], chunk_size):
            stop = min(start + chunk_size, self.projections.shape[0])
            difference = self.projections[start:stop] - query[None, :, :]
            distances[start:stop] = np.sqrt(
                np.mean(difference * difference, axis=(1, 2), dtype=np.float64)
            )
        best_by_pair: dict[tuple[str, str], dict[str, object]] = {}
        for labels, witness, distance in zip(
            self.phones, self.witnesses, distances, strict=True
        ):
            pair = (str(labels[0]), str(labels[1]))
            candidate = {
                "phones": list(pair),
                "distance": float(distance),
                "witness": str(witness),
            }
            current = best_by_pair.get(pair)
            if current is None or (candidate["distance"], candidate["witness"]) < (
                current["distance"],
                current["witness"],
            ):
                best_by_pair[pair] = candidate
        ranking = sorted(
            best_by_pair.values(),
            key=lambda item: (item["distance"], item["phones"]),
        )
        for rank, item in enumerate(ranking, start=1):
            item["rank"] = rank
        return ranking

    def rank_occurrences(
        self, query_cloud: np.ndarray, chunk_size: int = 64
    ) -> list[dict[str, object]]:
        """Rank every stored occurrence without collapsing equal pair labels."""

        if chunk_size < 1:
            raise ValueError("atlas query chunk size must be positive")
        query = self.query_projection(query_cloud).astype(np.float32, copy=False)
        distances = np.empty(self.projections.shape[0], dtype=np.float64)
        for start in range(0, self.projections.shape[0], chunk_size):
            stop = min(start + chunk_size, self.projections.shape[0])
            difference = self.projections[start:stop] - query[None, :, :]
            distances[start:stop] = np.sqrt(
                np.mean(difference * difference, axis=(1, 2), dtype=np.float64)
            )
        output = [
            {
                "phones": [str(labels[0]), str(labels[1])],
                "distance": float(distance),
                "witness": str(witness),
            }
            for labels, witness, distance in zip(
                self.phones, self.witnesses, distances, strict=True
            )
        ]
        output.sort(
            key=lambda item: (item["distance"], item["phones"], item["witness"])
        )
        for rank, item in enumerate(output, start=1):
            item["occurrence_rank"] = rank
        return output


def compile_boundary_atlas(
    occurrences: Iterable[tuple[tuple[str, str], str, np.ndarray]],
    config: CloudFitConfig,
    quantile_count: int = 256,
) -> CompiledBoundaryAtlas:
    """Compile labeled diphone clouds into float32 fixed-quantile projections."""

    phones = []
    witnesses = []
    projections = []
    for labels, witness, cloud in occurrences:
        if len(labels) != 2:
            raise ValueError("compiled boundary occurrence must have two phones")
        phones.append(tuple(str(phone) for phone in labels))
        witnesses.append(str(witness))
        projections.append(
            resample_sliced_wasserstein_projection(
                sliced_wasserstein_projection(cloud, config), quantile_count
            ).astype(np.float32)
        )
    if not projections:
        raise ValueError("compiled boundary atlas requires occurrences")
    return CompiledBoundaryAtlas(
        phones=np.asarray(phones),
        witnesses=np.asarray(witnesses),
        projections=np.stack(projections),
        quantile_count=quantile_count,
        projection_count=config.sliced_projection_count,
        projection_seed=config.sliced_projection_seed,
        metric_scales=(
            config.row_metric_scale,
            config.frame_metric_scale,
            config.height_metric_scale,
        ),
    )


def save_boundary_atlas(path: Path, atlas: CompiledBoundaryAtlas) -> None:
    metadata = {
        "format": ATLAS_FORMAT,
        "quantile_count": atlas.quantile_count,
        "projection_count": atlas.projection_count,
        "projection_seed": atlas.projection_seed,
        "metric_scales": list(atlas.metric_scales),
        "provenance": dict(atlas.provenance or {}),
    }
    np.savez(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        phones=atlas.phones,
        witnesses=atlas.witnesses,
        projections=atlas.projections,
    )


def load_boundary_atlas(path: Path) -> CompiledBoundaryAtlas:
    with np.load(path, allow_pickle=False) as document:
        metadata = json.loads(str(document["metadata"]))
        if metadata.get("format") != ATLAS_FORMAT:
            raise ValueError("unsupported compiled boundary atlas format")
        return CompiledBoundaryAtlas(
            phones=np.asarray(document["phones"]),
            witnesses=np.asarray(document["witnesses"]),
            projections=np.asarray(document["projections"], dtype=np.float32),
            quantile_count=int(metadata["quantile_count"]),
            projection_count=int(metadata["projection_count"]),
            projection_seed=int(metadata["projection_seed"]),
            metric_scales=tuple(float(x) for x in metadata["metric_scales"]),
            provenance=metadata.get("provenance", {}),
        )
