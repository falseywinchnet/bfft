"""Cleanup/SHARK state-and-manner profiles for phone proposal routing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import expit

from .cleanup_shark_vad import SpeechState


FORMAT = "ostensibly_phone_state_profile_atlas_v1"
FEATURE_NAMES = (
    "pitch_mean",
    "pitch_q75",
    "pitch_above_threshold_fraction",
    "prominence_probability_mean",
    "cleanup_probability_mean",
    "voice_gate_probability_mean",
    "unvoiced_score_mean",
    "state_voiced_fraction",
    "state_unvoiced_fraction",
    "state_hangover_fraction",
    "energy_snr_probability_mean",
)


def phone_state_features(
    pitch_periodicity: np.ndarray,
    spectral_prominence_db: np.ndarray,
    cleanup_probability: np.ndarray,
    fused_score: np.ndarray,
    unvoiced_score: np.ndarray,
    energy_snr_db: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            pitch_periodicity,
            spectral_prominence_db,
            cleanup_probability,
            fused_score,
            unvoiced_score,
            energy_snr_db,
        )
    )
    states = np.asarray(state, dtype=np.int8)
    if (
        not arrays[0].size
        or any(value.ndim != 1 or value.shape != arrays[0].shape for value in arrays)
        or states.shape != arrays[0].shape
        or not all(np.all(np.isfinite(value)) for value in arrays)
    ):
        raise ValueError("phone state profile received incompatible frame evidence")
    pitch = np.clip(arrays[0], 0.0, 1.0)
    prominence_probability = expit((arrays[1] - 6.5) / 3.0)
    voice_probability = expit((arrays[3] - 0.95) / 0.20)
    snr_probability = expit((arrays[5] - 3.0) / 3.0)
    return np.asarray(
        (
            np.mean(pitch),
            np.quantile(pitch, 0.75),
            np.mean(pitch >= 0.4),
            np.mean(prominence_probability),
            np.mean(np.clip(arrays[2], 0.0, 1.0)),
            np.mean(voice_probability),
            np.mean(np.clip(arrays[4], 0.0, 1.0)),
            np.mean(states == SpeechState.VOICED),
            np.mean(states == SpeechState.UNVOICED),
            np.mean(states == SpeechState.HANGOVER),
            np.mean(snr_probability),
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PhoneStateProfileAtlas:
    labels: tuple[str, ...]
    feature_center: np.ndarray
    feature_scale: np.ndarray
    class_centroids: np.ndarray
    class_counts: np.ndarray
    occurrences: np.ndarray
    occurrence_labels: np.ndarray

    def __post_init__(self) -> None:
        label_count = len(self.labels)
        feature_count = len(FEATURE_NAMES)
        if (
            label_count < 2
            or len(set(self.labels)) != label_count
            or self.feature_center.shape != (feature_count,)
            or self.feature_scale.shape != (feature_count,)
            or self.class_centroids.shape != (label_count, feature_count)
            or self.class_counts.shape != (label_count,)
            or self.occurrences.ndim != 2
            or self.occurrences.shape[1] != feature_count
            or self.occurrence_labels.shape != (self.occurrences.shape[0],)
            or np.any(self.feature_scale <= 0.0)
            or np.any(self.class_counts < 1)
            or not np.all(np.isfinite(self.class_centroids))
        ):
            raise ValueError("phone state-profile atlas has invalid geometry")

    def rank(self, features: np.ndarray) -> list[dict[str, object]]:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != self.feature_center.shape or not np.all(np.isfinite(values)):
            raise ValueError("query state profile has invalid shape")
        standardized = (values - self.feature_center) / self.feature_scale
        distances = np.sqrt(
            np.mean((self.class_centroids - standardized[None, :]) ** 2, axis=1)
        )
        order = np.lexsort((np.asarray(self.labels), distances))
        return [
            {
                "phone": self.labels[int(index)],
                "rank": rank,
                "distance": float(distances[int(index)]),
                "witness_count": int(self.class_counts[int(index)]),
            }
            for rank, index in enumerate(order, start=1)
        ]

    def save(self, path: Path) -> None:
        metadata = {
            "format": FORMAT,
            "labels": list(self.labels),
            "feature_names": list(FEATURE_NAMES),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            feature_center=self.feature_center.astype(np.float32),
            feature_scale=self.feature_scale.astype(np.float32),
            class_centroids=self.class_centroids.astype(np.float32),
            class_counts=self.class_counts,
            occurrences=self.occurrences.astype(np.float32),
            occurrence_labels=self.occurrence_labels,
        )


def compile_phone_state_profile_atlas(
    occurrences: Iterable[tuple[str, np.ndarray]],
    *,
    minimum_scale: float = 0.04,
) -> PhoneStateProfileAtlas:
    rows = tuple((str(label), np.asarray(features, dtype=np.float64)) for label, features in occurrences)
    if not rows or minimum_scale <= 0.0:
        raise ValueError("state-profile compilation requires occurrences")
    values = np.stack([features for _, features in rows])
    if values.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(values)):
        raise ValueError("state-profile occurrences have invalid features")
    occurrence_labels = np.asarray([label for label, _ in rows])
    labels = tuple(sorted(set(occurrence_labels)))
    center = np.median(values, axis=0)
    scale = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    scale = np.maximum(scale, minimum_scale)
    standardized = (values - center[None, :]) / scale[None, :]
    centroids = []
    counts = []
    for label in labels:
        selected = standardized[occurrence_labels == label]
        centroids.append(np.median(selected, axis=0))
        counts.append(selected.shape[0])
    return PhoneStateProfileAtlas(
        labels=labels,
        feature_center=center,
        feature_scale=scale,
        class_centroids=np.stack(centroids),
        class_counts=np.asarray(counts, dtype=np.int64),
        occurrences=values,
        occurrence_labels=occurrence_labels,
    )


def load_phone_state_profile_atlas(path: Path) -> PhoneStateProfileAtlas:
    with np.load(path, allow_pickle=False) as document:
        metadata = json.loads(str(document["metadata"]))
        if (
            metadata.get("format") != FORMAT
            or tuple(metadata.get("feature_names", ())) != FEATURE_NAMES
        ):
            raise ValueError("unsupported phone state-profile atlas format")
        return PhoneStateProfileAtlas(
            labels=tuple(str(value) for value in metadata["labels"]),
            feature_center=np.asarray(document["feature_center"], dtype=np.float64),
            feature_scale=np.asarray(document["feature_scale"], dtype=np.float64),
            class_centroids=np.asarray(document["class_centroids"], dtype=np.float64),
            class_counts=np.asarray(document["class_counts"], dtype=np.int64),
            occurrences=np.asarray(document["occurrences"], dtype=np.float64),
            occurrence_labels=np.asarray(document["occurrence_labels"]).astype(str),
        )
