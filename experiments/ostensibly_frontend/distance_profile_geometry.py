"""Cross-speaker centroid geometry over complete phone-distance profiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DistanceProfileAtlas:
    channels: tuple[str, ...]
    labels: tuple[str, ...]
    feature_centers: np.ndarray
    feature_scales: np.ndarray
    class_centroids: np.ndarray
    class_counts: np.ndarray

    def __post_init__(self) -> None:
        channel_count = len(self.channels)
        label_count = len(self.labels)
        if (
            self.feature_centers.shape != (channel_count, label_count)
            or self.feature_scales.shape != (channel_count, label_count)
            or self.class_centroids.shape
            != (channel_count, label_count, label_count)
            or self.class_counts.shape != (channel_count, label_count)
            or np.any(self.feature_scales <= 0.0)
            or np.any(self.class_counts < 1)
        ):
            raise ValueError("distance-profile atlas has invalid shape")

    def rank(self, profiles: dict[str, np.ndarray]) -> list[dict[str, object]]:
        scores = np.zeros(len(self.labels), dtype=np.float64)
        used = 0
        for channel_index, channel in enumerate(self.channels):
            if channel not in profiles:
                continue
            profile = np.asarray(profiles[channel], dtype=np.float64)
            if profile.shape != (len(self.labels),):
                raise ValueError("query distance profile has invalid shape")
            standardized = (
                profile - self.feature_centers[channel_index]
            ) / self.feature_scales[channel_index]
            difference = self.class_centroids[channel_index] - standardized[None, :]
            channel_scores = np.sqrt(np.mean(difference * difference, axis=1))
            floor = float(np.min(channel_scores))
            scale = float(np.quantile(channel_scores, 0.75) - floor)
            scores += (channel_scores - floor) / max(scale, 1e-12)
            used += 1
        if used < 1:
            raise ValueError("query contains no recognized profile channels")
        scores /= used
        order = np.argsort(scores, kind="stable")
        return [
            {
                "rank": rank,
                "phone": self.labels[int(index)],
                "distance": float(scores[int(index)]),
            }
            for rank, index in enumerate(order, start=1)
        ]


def ranking_profile(
    ranking: list[dict[str, object]], labels: tuple[str, ...]
) -> np.ndarray:
    distances = {str(item["phone"]): float(item["distance"]) for item in ranking}
    if set(distances) != set(labels):
        raise ValueError("phone ranking does not cover the profile label catalog")
    return np.asarray([distances[label] for label in labels], dtype=np.float64)


def fit_distance_profile_atlas(
    profiles: np.ndarray,
    targets: np.ndarray,
    profile_channels: np.ndarray,
    channels: tuple[str, ...],
    labels: tuple[str, ...],
) -> DistanceProfileAtlas:
    """Fit robust channel charts and labeled centroids without user speech."""

    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("calibration profile matrix has invalid shape")
    target_values = np.asarray(targets).astype(str)
    channel_values = np.asarray(profile_channels).astype(str)
    centers = []
    scales = []
    centroids = []
    counts = []
    for channel in channels:
        mask = channel_values == channel
        subset = values[mask]
        subset_targets = target_values[mask]
        if not subset.size:
            raise ValueError(f"profile channel {channel!r} has no examples")
        center = np.median(subset, axis=0)
        scale = np.quantile(subset, 0.75, axis=0) - np.quantile(
            subset, 0.25, axis=0
        )
        fallback = np.std(subset, axis=0)
        scale = np.where(scale > 1e-12, scale, np.maximum(fallback, 1e-12))
        standardized = (subset - center[None, :]) / scale[None, :]
        channel_centroids = []
        channel_counts = []
        for label in labels:
            class_values = standardized[subset_targets == label]
            if not class_values.size:
                raise ValueError(f"profile channel {channel!r} has no {label!r}")
            channel_centroids.append(np.mean(class_values, axis=0))
            channel_counts.append(class_values.shape[0])
        centers.append(center)
        scales.append(scale)
        centroids.append(np.stack(channel_centroids))
        counts.append(np.asarray(channel_counts, dtype=np.int64))
    return DistanceProfileAtlas(
        channels=channels,
        labels=labels,
        feature_centers=np.stack(centers),
        feature_scales=np.stack(scales),
        class_centroids=np.stack(centroids),
        class_counts=np.stack(counts),
    )


def load_distance_profile_atlas(path: Path) -> DistanceProfileAtlas:
    with np.load(path, allow_pickle=False) as document:
        metadata = json.loads(str(document["metadata"]))
        if metadata.get("format") != "ostensibly_distance_profile_atlas_v1":
            raise ValueError("unsupported distance-profile atlas format")
        return DistanceProfileAtlas(
            channels=tuple(str(value) for value in metadata["channels"]),
            labels=tuple(str(value) for value in metadata["labels"]),
            feature_centers=np.asarray(document["feature_centers"], dtype=np.float64),
            feature_scales=np.asarray(document["feature_scales"], dtype=np.float64),
            class_centroids=np.asarray(document["class_centroids"], dtype=np.float64),
            class_counts=np.asarray(document["class_counts"], dtype=np.int64),
        )
