"""Label-free phone-lattice topology from Cleanup/SHARK speech states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segmental_pronunciation_geometry import debounced_state_landmarks


@dataclass(frozen=True)
class StateSpeechCrop:
    crop_index: int
    frame0: int
    frame1: int
    landmarks: tuple[int, ...]
    state_audit: tuple[dict[str, int | str], ...]


def active_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("speech mask must be one-dimensional")
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple(
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
    )


def state_speech_crops(
    state: np.ndarray,
    speech_mask: np.ndarray,
    *,
    minimum_run_frames: int = 3,
) -> tuple[StateSpeechCrop, ...]:
    values = np.asarray(state, dtype=np.int8)
    mask = np.asarray(speech_mask, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("speech state and mask have incompatible shapes")
    crops = []
    for crop_index, (frame0, frame1) in enumerate(active_runs(mask)):
        landmarks, audit = debounced_state_landmarks(
            values,
            mask,
            frame0,
            frame1,
            minimum_run_frames=minimum_run_frames,
        )
        crops.append(
            StateSpeechCrop(
                crop_index=crop_index,
                frame0=frame0,
                frame1=frame1,
                landmarks=landmarks,
                state_audit=audit,
            )
        )
    return tuple(crops)


def bounded_lattice_spans(
    landmarks: tuple[int, ...],
    *,
    maximum_compartments: int = 3,
) -> tuple[tuple[int, int, int, int], ...]:
    if (
        len(landmarks) < 2
        or maximum_compartments < 1
        or any(right <= left for left, right in zip(landmarks[:-1], landmarks[1:]))
    ):
        raise ValueError("state landmarks cannot form a phone lattice")
    compartment_count = len(landmarks) - 1
    return tuple(
        (start, stop, landmarks[start], landmarks[stop])
        for start in range(compartment_count)
        for stop in range(
            start + 1,
            min(compartment_count, start + maximum_compartments) + 1,
        )
    )
