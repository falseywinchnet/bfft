"""Cross-observation estimation of positive exposure transport.

For registered captures of one latent image,

    Y_i(k) = H_i(k) X(k) + noise.

The closure ``Y_0 H_1 - Y_1 H_0`` cancels the unknown scene without dividing
by it.  The residual is pooled on Fourier circles so a few texture-rich
directions cannot silently become the whole verdict.  This identifies
*relative* blur.  A common blur factor is a real gauge and is reported as
ambiguity rather than declared recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circles import circle_pool
from .kernels import TransportKernel


@dataclass(frozen=True)
class PairEstimate:
    first: TransportKernel
    second: TransportKernel
    score: float
    runner_up_score: float
    ambiguity_ratio: float
    relative_transport_strength: float
    common_blur_unidentifiable: bool
    ring_residual: np.ndarray
    ring_support: np.ndarray
    ranked: tuple[tuple[float, str, str], ...]


def _luminance(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        return value
    if value.ndim == 3 and value.shape[2] >= 3:
        return (
            0.2126 * value[..., 0]
            + 0.7152 * value[..., 1]
            + 0.0722 * value[..., 2]
        )
    raise ValueError("observations must be HxW or RGB")


def _closure_record(
    first_fft: np.ndarray,
    second_fft: np.ndarray,
    first_otf: np.ndarray,
    second_otf: np.ndarray,
    *,
    ring_width: float,
    minimum_radius: float,
    maximum_radius_fraction: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    left = first_fft * second_otf
    right = second_fft * first_otf
    energy = np.abs(left) ** 2 + np.abs(right) ** 2
    robust_floor = max(
        float(np.quantile(energy, 0.10)),
        float(np.max(energy)) * 1e-14,
        np.finfo(float).tiny,
    )
    coefficient_residual = np.abs(left - right) ** 2 / (energy + robust_floor)
    # Energy is support, but sqrt compression prevents a few bright edges
    # from deciding every ring.
    coefficient_support = np.sqrt(energy + robust_floor)
    ring_residual, ring_support = circle_pool(
        coefficient_residual,
        weights=coefficient_support,
        width=ring_width,
    )
    radius = (np.arange(len(ring_residual), dtype=np.float64) + 0.5) * ring_width
    maximum = maximum_radius_fraction * 0.5 * min(first_fft.shape)
    valid = (
        (radius >= minimum_radius)
        & (radius <= maximum)
        & (ring_support > np.quantile(ring_support[ring_support > 0.0], 0.10))
    )
    selected = ring_residual[valid]
    if not len(selected):
        return float("inf"), ring_residual, ring_support
    # A robust circle aggregate: most radii must close, while isolated OTF
    # zeros or scene-poor rings cannot dominate.
    score = float(0.75 * np.median(selected) + 0.25 * np.mean(
        np.sort(selected)[: max(1, int(np.ceil(0.8 * len(selected))))]
    ))
    return score, ring_residual, ring_support


def estimate_kernel_pair(
    first: np.ndarray,
    second: np.ndarray,
    candidates: list[TransportKernel],
    *,
    ring_width: float = 1.5,
    minimum_radius: float = 2.0,
    maximum_radius_fraction: float = 0.92,
    common_blur_gauge_penalty: float = 0.04,
    complexity_tiebreak: float = 1e-8,
    ranking_limit: int = 12,
) -> PairEstimate:
    """Estimate two relative kernels from registered observations.

    Cross-closure cannot identify a blur common to both captures. The
    ``common_blur_gauge_penalty`` selects the maximally covered representative
    of that equivalence class; it is a declared gauge choice, not evidence
    that common blur was measured. ``complexity_tiebreak`` only resolves exact
    remaining ties.
    """
    a = _luminance(first)
    b = _luminance(second)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("registered observations must share one 2-D shape")
    if not candidates:
        raise ValueError("kernel estimation needs a non-empty candidate family")
    first_fft = np.fft.fft2(a - np.mean(a))
    second_fft = np.fft.fft2(b - np.mean(b))
    transfers = [candidate.otf(a.shape) for candidate in candidates]
    complexities = [float(np.trace(candidate.covariance)) for candidate in candidates]
    records: list[tuple[float, float, int, int, np.ndarray, np.ndarray]] = []
    for i, first_otf in enumerate(transfers):
        for j, second_otf in enumerate(transfers):
            closure, ring_residual, ring_support = _closure_record(
                first_fft,
                second_fft,
                first_otf,
                second_otf,
                ring_width=ring_width,
                minimum_radius=minimum_radius,
                maximum_radius_fraction=maximum_radius_fraction,
            )
            joint_transfer = 0.5 * (
                np.abs(first_otf) ** 2 + np.abs(second_otf) ** 2)
            common_attenuation = float(np.mean(-np.log(np.maximum(
                joint_transfer, 1e-8))))
            ranked_score = (
                closure
                + max(float(common_blur_gauge_penalty), 0.0)
                * common_attenuation
                + complexity_tiebreak * (complexities[i] + complexities[j])
            )
            records.append((
                ranked_score, closure, i, j, ring_residual, ring_support))
    records.sort(key=lambda item: item[0])
    winner = records[0]
    runner = records[1] if len(records) > 1 else records[0]
    score = float(winner[1])
    runner_score = float(runner[1])
    ratio = runner_score / max(score, np.finfo(float).tiny)
    winner_first_otf = transfers[winner[2]]
    winner_second_otf = transfers[winner[3]]
    relative_strength = float(np.sqrt(np.mean(
        np.abs(winner_first_otf - winner_second_otf) ** 2
    )))
    ranked = tuple(
        (float(record[1]), candidates[record[2]].name, candidates[record[3]].name)
        for record in records[: max(int(ranking_limit), 1)]
    )
    return PairEstimate(
        first=candidates[winner[2]],
        second=candidates[winner[3]],
        score=score,
        runner_up_score=runner_score,
        ambiguity_ratio=float(ratio),
        relative_transport_strength=relative_strength,
        common_blur_unidentifiable=bool(relative_strength < 0.02),
        ring_residual=winner[4],
        ring_support=winner[5],
        ranked=ranked,
    )
