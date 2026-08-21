"""Deterministic, locally optimal binary palette construction in OKLCH space."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from tlvector.core import _srgb_to_oklab


@dataclass(frozen=True)
class BifurcationResult:
    palette_lab_alpha: np.ndarray
    parent_lab_alpha: np.ndarray
    leaf_population: np.ndarray
    total_gain: float
    splits: int


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert OKLab to unclipped sRGB."""
    value = np.asarray(lab, dtype=np.float64)
    lightness, a_axis, b_axis = np.moveaxis(value, -1, 0)
    l_root = lightness + 0.3963377774 * a_axis + 0.2158037573 * b_axis
    m_root = lightness - 0.1055613458 * a_axis - 0.0638541728 * b_axis
    s_root = lightness - 0.0894841775 * a_axis - 1.2914855480 * b_axis
    lms = np.stack((l_root**3, m_root**3, s_root**3), axis=-1)
    linear = lms @ np.array([
        [4.0767416621, -1.2684380046, -0.0041960863],
        [-3.3077115913, 2.6097574011, -0.7034186147],
        [0.2309699292, -0.3413193965, 1.7076147010],
    ])
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )


def gamut_map_oklch(lab: np.ndarray, *, iterations: int = 18) -> np.ndarray:
    """Fit colors into sRGB by reducing OKLCH chroma while preserving L and h."""
    source = np.asarray(lab, dtype=np.float64)
    result = source.copy()
    flat = result.reshape(-1, 3)
    for row in flat:
        direct = oklab_to_srgb(row)
        if np.all((direct >= 0.0) & (direct <= 1.0)):
            continue
        chroma = float(np.hypot(row[1], row[2]))
        if chroma <= 1e-12:
            row[0] = np.clip(row[0], 0.0, 1.0)
            continue
        direction = row[1:3] / chroma
        row[0] = np.clip(row[0], 0.0, 1.0)
        low, high = 0.0, chroma
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            candidate = np.array([row[0], *(direction * middle)])
            rgb = oklab_to_srgb(candidate)
            if np.all((rgb >= 0.0) & (rgb <= 1.0)):
                low = middle
            else:
                high = middle
        row[1:3] = direction * low
    return result


def lab_alpha_to_rgba(nodes: np.ndarray) -> np.ndarray:
    mapped = gamut_map_oklch(np.asarray(nodes, dtype=np.float64)[..., :3])
    rgb = np.clip(np.round(oklab_to_srgb(mapped) * 255.0), 0, 255).astype(np.uint8)
    alpha = np.clip(np.round(np.asarray(nodes)[..., 3] * 255.0), 0, 255).astype(np.uint8)
    return np.concatenate((rgb, alpha[..., None]), axis=-1)


def oklch_distance2(
    samples: np.ndarray,
    centers: np.ndarray,
    *,
    lightness_weight: float = 1.0,
    chroma_weight: float = 1.0,
    hue_weight: float = 1.0,
    alpha_weight: float = 0.7,
) -> np.ndarray:
    """Squared cylindrical distance; unit weights equal OKLab distance."""
    values = np.asarray(samples, dtype=np.float64).reshape(-1, 4)
    nodes = np.asarray(centers, dtype=np.float64).reshape(-1, 4)
    sample_c = np.hypot(values[:, 1], values[:, 2])
    center_c = np.hypot(nodes[:, 1], nodes[:, 2])
    lightness2 = lightness_weight**2
    chroma2 = chroma_weight**2
    hue2 = hue_weight**2
    alpha2 = alpha_weight**2
    sample_norm = (
        lightness2 * values[:, 0] ** 2
        + chroma2 * sample_c**2
        + alpha2 * values[:, 3] ** 2
    )
    center_norm = (
        lightness2 * nodes[:, 0] ** 2
        + chroma2 * center_c**2
        + alpha2 * nodes[:, 3] ** 2
    )
    sample_features = np.stack((
        lightness2 * values[:, 0],
        hue2 * values[:, 1],
        hue2 * values[:, 2],
        alpha2 * values[:, 3],
        -(hue2 - chroma2) * sample_c,
    ), axis=1)
    center_features = np.stack((
        nodes[:, 0], nodes[:, 1], nodes[:, 2], nodes[:, 3], center_c
    ), axis=1)
    distance = (
        sample_norm[:, None]
        + center_norm[None, :]
        - 2.0 * (sample_features @ center_features.T)
    )
    return np.maximum(distance, 0.0)


def oklch_pair_distance2(
    first: np.ndarray,
    second: np.ndarray,
    *,
    lightness_weight: float = 1.0,
    chroma_weight: float = 1.0,
    hue_weight: float = 1.0,
    alpha_weight: float = 0.7,
) -> np.ndarray:
    """Distance for aligned color pairs without constructing a cross matrix."""
    left = np.asarray(first, dtype=np.float64).reshape(-1, 4)
    right = np.asarray(second, dtype=np.float64).reshape(-1, 4)
    if len(left) != len(right):
        raise ValueError("paired color arrays must have equal length")
    left_c = np.hypot(left[:, 1], left[:, 2])
    right_c = np.hypot(right[:, 1], right[:, 2])
    chroma_term = (
        chroma_weight**2 * (left_c**2 + right_c**2)
        + 2.0 * (hue_weight**2 - chroma_weight**2) * left_c * right_c
        - 2.0
        * hue_weight**2
        * (left[:, 1] * right[:, 1] + left[:, 2] * right[:, 2])
    )
    return (
        (lightness_weight * (left[:, 0] - right[:, 0])) ** 2
        + np.maximum(chroma_term, 0.0)
        + (alpha_weight * (left[:, 3] - right[:, 3])) ** 2
    )


def _tangent_coordinates(
    values: np.ndarray,
    weights: tuple[float, float, float, float],
    sample_weights: np.ndarray,
) -> np.ndarray:
    lightness_weight, chroma_weight, hue_weight, alpha_weight = weights
    chroma = np.hypot(values[:, 1], values[:, 2])
    hue = np.arctan2(values[:, 2], values[:, 1])
    vector = np.sum(sample_weights * chroma * np.exp(1j * hue))
    center_hue = float(np.angle(vector)) if abs(vector) > 1e-12 else 0.0
    delta_hue = np.angle(np.exp(1j * (hue - center_hue)))
    mean_chroma = float(np.average(chroma, weights=sample_weights))
    hue_radius = np.sqrt(np.maximum(chroma * mean_chroma, 1e-8))
    return np.stack((
        lightness_weight * values[:, 0],
        chroma_weight * chroma,
        hue_weight * hue_radius * delta_hue,
        alpha_weight * values[:, 3],
    ), axis=1)


def _two_means_proposal(
    values: np.ndarray,
    weights: tuple[float, float, float, float],
    sample_weights: np.ndarray,
    minimum_leaf: int,
) -> tuple[float, np.ndarray] | None:
    if len(values) < 2 * minimum_leaf:
        return None
    mass = np.asarray(sample_weights, dtype=np.float64)
    safe_mass = max(float(np.sum(mass)), 1e-15)
    coordinates = _tangent_coordinates(values, weights, mass)
    center = np.sum(coordinates * mass[:, None], axis=0) / safe_mass
    centered = coordinates - center
    old_sse = float(np.sum(mass[:, None] * centered * centered))
    if old_sse <= 1e-14:
        return None
    covariance = centered.T @ (centered * mass[:, None])
    _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    projections = [centered @ eigenvectors[:, -1]]
    projections.extend(centered[:, axis] for axis in range(centered.shape[1]))
    best_gain = 0.0
    best_side: np.ndarray | None = None
    seen_splits: set[bytes] = set()
    for projection in projections:
        order = np.argsort(projection, kind="stable")
        ordered_mass = mass[order]
        ordered = coordinates[order]
        prefix_mass = np.cumsum(ordered_mass)
        prefix_sum = np.cumsum(ordered * ordered_mass[:, None], axis=0)
        prefix_norm = np.cumsum(
            ordered_mass * np.einsum("ij,ij->i", ordered, ordered)
        )
        cuts = np.arange(minimum_leaf - 1, len(values) - minimum_leaf)
        left_mass = prefix_mass[cuts]
        right_mass = safe_mass - left_mass
        left_sum = prefix_sum[cuts]
        right_sum = prefix_sum[-1] - left_sum
        left_sse = prefix_norm[cuts] - np.einsum(
            "ij,ij->i", left_sum, left_sum
        ) / left_mass
        right_sse = (prefix_norm[-1] - prefix_norm[cuts]) - np.einsum(
            "ij,ij->i", right_sum, right_sum
        ) / right_mass
        split = int(cuts[int(np.argmin(left_sse + right_sse))]) + 1
        side = np.zeros(len(values), dtype=bool)
        side[order[split:]] = True
        signature = np.packbits(side).tobytes()
        if signature in seen_splits:
            continue
        seen_splits.add(signature)

        centers = np.stack((
            np.sum(coordinates[~side] * mass[~side, None], axis=0)
            / np.sum(mass[~side]),
            np.sum(coordinates[side] * mass[side, None], axis=0)
            / np.sum(mass[side]),
        ))
        for _ in range(12):
            # Comparing squared distances only needs the separating hyperplane;
            # avoid constructing an N x 2 x D temporary on every Lloyd step.
            threshold = 0.5 * (
                np.dot(centers[1], centers[1])
                - np.dot(centers[0], centers[0])
            )
            updated = coordinates @ (centers[1] - centers[0]) > threshold
            left_count = int(np.count_nonzero(~updated))
            right_count = len(values) - left_count
            if left_count < minimum_leaf or right_count < minimum_leaf:
                break
            if np.array_equal(updated, side):
                break
            side = updated
            next_centers = np.stack((
                np.sum(coordinates[~side] * mass[~side, None], axis=0)
                / np.sum(mass[~side]),
                np.sum(coordinates[side] * mass[side, None], axis=0)
                / np.sum(mass[side]),
            ))
            centers = next_centers
        residual = coordinates - centers[side.astype(np.int8)]
        new_sse = float(np.sum(mass * np.einsum("ij,ij->i", residual, residual)))
        gain = old_sse - new_sse
        if gain > best_gain + 1e-14:
            best_gain = gain
            best_side = side.copy()
    if best_side is None:
        return None
    return best_gain, best_side


def _partition_gain(
    values: np.ndarray,
    weights: tuple[float, float, float, float],
    sample_weights: np.ndarray,
    side: np.ndarray,
) -> float:
    """Measure a fixed partition in one metric's distortion units."""
    mass = np.asarray(sample_weights, dtype=np.float64)
    coordinates = _tangent_coordinates(values, weights, mass)
    center = np.average(coordinates, axis=0, weights=mass)
    old_sse = float(np.sum(mass[:, None] * (coordinates - center) ** 2))
    new_sse = 0.0
    for selection in (~side, side):
        child = np.average(
            coordinates[selection], axis=0, weights=mass[selection]
        )
        residual = coordinates[selection] - child
        new_sse += float(np.sum(mass[selection, None] * residual * residual))
    return max(0.0, old_sse - new_sse)


def bifurcate_palette(
    samples_lab_alpha: np.ndarray,
    colors: int,
    *,
    sample_weights: np.ndarray | None = None,
    lightness_weight: float = 1.0,
    chroma_weight: float = 1.0,
    hue_weight: float = 1.0,
    alpha_weight: float = 0.7,
    minimum_leaf: int = 8,
    family_priority: float = 0.0,
) -> BifurcationResult:
    """Split the globally most profitable occupied perceptual node.

    When ``family_priority`` is positive and at least four colors are
    requested, the root split temporarily emphasizes chroma and hue.  Later
    splits return to the caller's perceptual metric.  This prevents a small
    palette from spending every early branch on lightness before distinct
    color families have a leaf of their own.
    """
    samples = np.asarray(samples_lab_alpha, dtype=np.float64).reshape(-1, 4)
    if not len(samples):
        raise ValueError("cannot bifurcate an empty color population")
    requested = max(1, min(int(colors), len(samples)))
    if sample_weights is None:
        importance = np.ones(len(samples), dtype=np.float64)
    else:
        importance = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if len(importance) != len(samples):
            raise ValueError("sample_weights must match the sample population")
        if np.any(~np.isfinite(importance)) or np.any(importance <= 0.0):
            raise ValueError("sample_weights must be finite and positive")
    weights = (
        max(0.0, float(lightness_weight)),
        max(0.0, float(chroma_weight)),
        max(0.0, float(hue_weight)),
        max(0.0, float(alpha_weight)),
    )
    leaves: dict[int, np.ndarray] = {0: np.arange(len(samples), dtype=np.int64)}
    centers: dict[int, np.ndarray] = {
        0: np.average(samples, axis=0, weights=importance)
    }
    parents: dict[int, np.ndarray] = {0: centers[0].copy()}
    heap: list[tuple[float, int, int, np.ndarray]] = []
    serial = 0

    def queue(label: int) -> None:
        nonlocal serial
        proposal = _two_means_proposal(
            samples[leaves[label]],
            weights,
            importance[leaves[label]],
            max(1, int(minimum_leaf)),
        )
        if proposal is None:
            return
        gain, side = proposal
        heapq.heappush(heap, (-gain, serial, label, side))
        serial += 1

    total_gain = 0.0
    next_label = 1
    bootstrap = max(0.0, float(family_priority))
    if bootstrap > 0.0 and requested >= 4:
        family_weights = (
            weights[0] / (1.0 + 1.85 * bootstrap),
            weights[1] * (1.0 + 0.4 * bootstrap),
            weights[2] * (1.0 + 1.2 * bootstrap),
            weights[3],
        )
        proposal = _two_means_proposal(
            samples,
            family_weights,
            importance,
            max(1, int(minimum_leaf)),
        )
        if proposal is not None:
            _family_gain, side = proposal
            indices = leaves.pop(0)
            parent_center = centers.pop(0)
            parents.pop(0)
            for child_indices in (indices[~side], indices[side]):
                child_label = next_label
                next_label += 1
                leaves[child_label] = child_indices
                centers[child_label] = np.average(
                    samples[child_indices],
                    axis=0,
                    weights=importance[child_indices],
                )
                parents[child_label] = parent_center
                queue(child_label)
            total_gain += _partition_gain(
                samples, weights, importance, side
            )
        else:
            queue(0)
    else:
        queue(0)

    while len(leaves) < requested and heap:
        negative_gain, _serial, label, side = heapq.heappop(heap)
        if label not in leaves:
            continue
        indices = leaves.pop(label)
        parent_center = centers.pop(label)
        parents.pop(label)
        left = indices[~side]
        right = indices[side]
        first_label = next_label
        second_label = next_label + 1
        next_label += 2
        for child_label, child_indices in (
            (first_label, left), (second_label, right)
        ):
            leaves[child_label] = child_indices
            centers[child_label] = np.average(
                samples[child_indices],
                axis=0,
                weights=importance[child_indices],
            )
            parents[child_label] = parent_center
            queue(child_label)
        total_gain += -float(negative_gain)

    order = sorted(leaves)
    return BifurcationResult(
        palette_lab_alpha=np.stack([centers[label] for label in order]),
        parent_lab_alpha=np.stack([parents[label] for label in order]),
        leaf_population=np.asarray([len(leaves[label]) for label in order]),
        total_gain=total_gain,
        splits=len(order) - 1,
    )


def separate_nodes(result: BifurcationResult, factor: float) -> np.ndarray:
    """Move leaves along their final bifurcation vectors for stylization."""
    amount = max(0.0, float(factor))
    shifted = result.parent_lab_alpha + amount * (
        result.palette_lab_alpha - result.parent_lab_alpha
    )
    shifted[:, 0] = np.clip(shifted[:, 0], 0.0, 1.0)
    shifted[:, 3] = np.clip(shifted[:, 3], 0.0, 1.0)
    shifted[:, :3] = gamut_map_oklch(shifted[:, :3])
    return shifted
