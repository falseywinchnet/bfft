"""Reusable fixed-quantile SWD projections for labeled phone occurrences."""

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


ATLAS_FORMAT = "ostensibly_phone_swd_atlas_v1"


@dataclass(frozen=True)
class CompiledPhoneAtlas:
    labels: np.ndarray
    witnesses: np.ndarray
    projections: np.ndarray
    quantile_count: int
    projection_count: int
    projection_seed: int
    metric_scales: tuple[float, float, float]
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        count = self.projections.shape[0]
        if (
            self.projections.ndim != 3
            or self.projections.shape[1] != self.quantile_count
            or self.projections.shape[2] != self.projection_count + 3
            or self.labels.shape != (count,)
            or self.witnesses.shape != (count,)
            or count < 1
        ):
            raise ValueError("compiled phone atlas has invalid shape")

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

    def rank(self, cloud: np.ndarray, chunk_size: int = 64) -> list[dict[str, object]]:
        if chunk_size < 1:
            raise ValueError("phone atlas query chunk size must be positive")
        query = self.query_projection(cloud).astype(np.float32, copy=False)
        distances = np.empty(self.projections.shape[0], dtype=np.float64)
        for start in range(0, self.projections.shape[0], chunk_size):
            stop = min(start + chunk_size, self.projections.shape[0])
            difference = self.projections[start:stop] - query[None, :, :]
            distances[start:stop] = np.sqrt(
                np.mean(difference * difference, axis=(1, 2), dtype=np.float64)
            )
        best_by_label: dict[str, dict[str, object]] = {}
        for label, witness, distance in zip(
            self.labels, self.witnesses, distances, strict=True
        ):
            key = str(label)
            candidate = {
                "phone": key,
                "distance": float(distance),
                "witness": str(witness),
            }
            current = best_by_label.get(key)
            if current is None or (candidate["distance"], candidate["witness"]) < (
                current["distance"], current["witness"]
            ):
                best_by_label[key] = candidate
        ranking = sorted(
            best_by_label.values(), key=lambda item: (item["distance"], item["phone"])
        )
        for rank, item in enumerate(ranking, start=1):
            item["rank"] = rank
        return ranking


def compile_phone_atlas(
    occurrences: Iterable[tuple[str, str, np.ndarray]],
    config: CloudFitConfig,
    quantile_count: int = 256,
    provenance: Mapping[str, object] | None = None,
) -> CompiledPhoneAtlas:
    labels = []
    witnesses = []
    projections = []
    for label, witness, cloud in occurrences:
        labels.append(str(label))
        witnesses.append(str(witness))
        projections.append(
            resample_sliced_wasserstein_projection(
                sliced_wasserstein_projection(cloud, config), quantile_count
            ).astype(np.float32)
        )
    if not projections:
        raise ValueError("compiled phone atlas requires occurrences")
    return CompiledPhoneAtlas(
        labels=np.asarray(labels),
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
        provenance=provenance,
    )


def save_phone_atlas(path: Path, atlas: CompiledPhoneAtlas) -> None:
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
        labels=atlas.labels,
        witnesses=atlas.witnesses,
        projections=atlas.projections,
    )


def load_phone_atlas(path: Path) -> CompiledPhoneAtlas:
    with np.load(path, allow_pickle=False) as document:
        metadata = json.loads(str(document["metadata"]))
        if metadata.get("format") != ATLAS_FORMAT:
            raise ValueError("unsupported compiled phone atlas format")
        return CompiledPhoneAtlas(
            labels=np.asarray(document["labels"]),
            witnesses=np.asarray(document["witnesses"]),
            projections=np.asarray(document["projections"], dtype=np.float32),
            quantile_count=int(metadata["quantile_count"]),
            projection_count=int(metadata["projection_count"]),
            projection_seed=int(metadata["projection_seed"]),
            metric_scales=tuple(float(x) for x in metadata["metric_scales"]),
            provenance=metadata.get("provenance", {}),
        )
