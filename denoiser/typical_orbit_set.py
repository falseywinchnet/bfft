"""First post-FMMT checkpoint: a typical-orbit feasible-set denoiser.

The estimator never builds a smoothed guide.  A target pixel is described only
by raw, target-excluded rings.  Dihedral-invariant ring moments retrieve other
image locations carrying similar local structure.  Their centre values become
distinct clean-patch hypotheses after an offset gauge transfer.

The readout is set-valued before it is scalar:

* retain a fixed numerical budget of nearest structural-orbit components;
* find the narrowest interval containing a strict majority of their values;
* keep the observation exactly when it is inside that interval;
* otherwise select the actual component medoid, never the component mean.

The finite candidate count and ring catalogue are representation resolutions,
not learned noise settings.  This module is a falsifiable first checkpoint,
not a promoted denoiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TypicalOrbitResolution:
    """Numerical resolution of the finite typical-orbit representation."""

    ring_radii: tuple[int, ...] = (1, 2, 3)
    candidate_count: int = 9
    query_multiplier: int = 6

    def __post_init__(self) -> None:
        radii = tuple(int(radius) for radius in self.ring_radii)
        if not radii or any(radius < 1 for radius in radii):
            raise ValueError("ring radii must be positive")
        if tuple(sorted(set(radii))) != radii:
            raise ValueError("ring radii must be unique and increasing")
        if int(self.candidate_count) < 3 or int(self.candidate_count) % 2 == 0:
            raise ValueError("candidate count must be odd and at least three")
        if int(self.query_multiplier) < 2:
            raise ValueError("query multiplier must be at least two")


@dataclass
class TypicalOrbitState:
    """Scalar projection and the feasible interval that produced it."""

    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    ambiguity: np.ndarray
    local_scale: np.ndarray
    selected_distance: np.ndarray
    retained_observation: np.ndarray
    observation_support_count: np.ndarray | None = None
    component_lower: np.ndarray | None = None
    component_upper: np.ndarray | None = None
    orbit_coherence: np.ndarray | None = None


@dataclass(frozen=True)
class LocalOrbitResolution:
    """Quadrature of target-excluded one-sided affine orbit charts."""

    radii: tuple[int, ...] = (1, 2, 3)

    def __post_init__(self) -> None:
        radii = tuple(int(radius) for radius in self.radii)
        if not radii or any(radius < 1 for radius in radii):
            raise ValueError("local orbit radii must be positive")
        if tuple(sorted(set(radii))) != radii:
            raise ValueError("local orbit radii must be unique and increasing")


def _ring_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """Clockwise Chebyshev ring; D4 actions are shifts and reversals."""
    r = int(radius)
    offsets: list[tuple[int, int]] = []
    for x in range(-r, r):
        offsets.append((-r, x))
    for y in range(-r, r):
        offsets.append((y, r))
    for x in range(r, -r, -1):
        offsets.append((r, x))
    for y in range(r, -r, -1):
        offsets.append((y, -r))
    return tuple(offsets)


def _shift_reflect(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    pad_y = abs(int(dy))
    pad_x = abs(int(dx))
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    y0 = pad_y + int(dy)
    x0 = pad_x + int(dx)
    return padded[y0:y0 + image.shape[0], x0:x0 + image.shape[1]]


def _ring_stack(image: np.ndarray, radius: int) -> np.ndarray:
    return np.stack(
        [_shift_reflect(image, dy, dx) for dy, dx in _ring_offsets(radius)],
        axis=0,
    )


def _invariant_descriptor(
    image: np.ndarray,
    radii: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D4- and offset-invariant raw-ring descriptors with no target sample."""
    features: list[np.ndarray] = []
    locations: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    epsilon = np.finfo(np.float64).eps
    for radius in radii:
        ring = _ring_stack(image, int(radius))
        location = np.median(ring, axis=0)
        absolute = np.abs(ring - location[None, ...])
        scale = np.median(absolute, axis=0)
        positive = scale[scale > epsilon]
        scale_floor = (
            float(np.median(positive)) * np.sqrt(epsilon)
            if positive.size
            else epsilon
        )
        normalized = (ring - location[None, ...]) / np.maximum(
            scale[None, ...], scale_floor)
        normalized = normalized / np.sqrt(1.0 + normalized * normalized)

        quantiles = np.quantile(
            normalized, (0.125, 0.25, 0.75, 0.875), axis=0)
        features.extend(quantiles[index] for index in range(quantiles.shape[0]))

        count = normalized.shape[0]
        lags = tuple(sorted(set((1, max(1, count // 8), count // 4, count // 2))))
        for lag in lags:
            features.append(np.mean(
                normalized * np.roll(normalized, lag, axis=0), axis=0))
        # Reflection-symmetrized third-order orbit moments.  These are the
        # first phase-sensitive terms in the finite descriptor.
        for lag_a, lag_b in ((1, count // 4), (count // 8, count // 2)):
            forward = np.mean(
                normalized
                * np.roll(normalized, lag_a, axis=0)
                * np.roll(normalized, lag_b, axis=0),
                axis=0,
            )
            reverse = np.mean(
                normalized
                * np.roll(normalized, -lag_a, axis=0)
                * np.roll(normalized, -lag_b, axis=0),
                axis=0,
            )
            features.append(0.5 * (forward + reverse))
        first_difference = normalized - np.roll(normalized, 1, axis=0)
        second_difference = (
            np.roll(normalized, -1, axis=0)
            - 2.0 * normalized
            + np.roll(normalized, 1, axis=0)
        )
        features.append(np.mean(np.abs(first_difference), axis=0))
        features.append(np.mean(np.abs(second_difference), axis=0))
        locations.append(location)
        scales.append(scale)

    descriptor = np.stack(features, axis=-1)
    # The photometric chart is separate from the invariant descriptor.
    location = np.median(np.stack(locations, axis=0), axis=0)
    local_scale = np.median(np.stack(scales, axis=0), axis=0)
    return descriptor, location, local_scale


def _reference_indices(
    descriptor: np.ndarray,
    candidate_count: int,
    exclusion_radius: int,
    query_multiplier: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width, feature_count = descriptor.shape
    points = descriptor.reshape(-1, feature_count)
    tree = cKDTree(points)
    requested = min(
        points.shape[0],
        max(candidate_count + 1, candidate_count * query_multiplier),
    )
    distances, indices = tree.query(points, k=requested, workers=-1)
    if requested == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    row = np.arange(points.shape[0]) // width
    column = np.arange(points.shape[0]) % width
    selected_indices = np.empty((points.shape[0], candidate_count), np.int64)
    selected_distances = np.empty((points.shape[0], candidate_count), np.float64)
    for target in range(points.shape[0]):
        accepted: list[tuple[int, float]] = []
        fallback: list[tuple[int, float]] = []
        for distance, reference in zip(distances[target], indices[target]):
            reference = int(reference)
            if reference == target:
                continue
            item = (reference, float(distance))
            fallback.append(item)
            dy = abs(int(row[reference]) - int(row[target]))
            dx = abs(int(column[reference]) - int(column[target]))
            if max(dy, dx) > exclusion_radius:
                accepted.append(item)
            if len(accepted) >= candidate_count:
                break
        if len(accepted) < candidate_count:
            present = {reference for reference, _ in accepted}
            for item in fallback:
                if item[0] not in present:
                    accepted.append(item)
                    present.add(item[0])
                if len(accepted) >= candidate_count:
                    break
        if len(accepted) < candidate_count:
            raise RuntimeError("insufficient distinct typical-orbit references")
        selected_indices[target] = [item[0] for item in accepted[:candidate_count]]
        selected_distances[target] = [item[1] for item in accepted[:candidate_count]]
    return selected_indices, selected_distances


def _majority_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Narrowest strict-majority interval and its actual component medoid."""
    ordered = np.sort(values, axis=1)
    count = ordered.shape[1]
    majority = count // 2 + 1
    widths = ordered[:, majority - 1:] - ordered[:, :count - majority + 1]
    start = np.argmin(widths, axis=1)
    rows = np.arange(ordered.shape[0])
    lower = ordered[rows, start]
    upper = ordered[rows, start + majority - 1]
    medoid = ordered[rows, start + majority // 2]
    return lower, upper, medoid


def infer_typical_orbit_set(
    image: np.ndarray,
    resolution: TypicalOrbitResolution = TypicalOrbitResolution(),
) -> TypicalOrbitState:
    """Construct and project the finite typical-orbit feasible set."""
    observed = np.asarray(image, dtype=np.float64)
    if observed.ndim != 2:
        raise ValueError("the first typical-orbit checkpoint is grayscale 2-D")
    if min(observed.shape) <= 2 * max(resolution.ring_radii) + 1:
        raise ValueError("image is too small for the requested raw-ring resolution")
    if not np.all(np.isfinite(observed)):
        raise ValueError("image contains non-finite values")
    observed = np.clip(observed, 0.0, 1.0)

    descriptor, location, local_scale = _invariant_descriptor(
        observed, resolution.ring_radii)
    references, distances = _reference_indices(
        descriptor,
        int(resolution.candidate_count),
        2 * max(resolution.ring_radii),
        int(resolution.query_multiplier),
    )
    flat = observed.ravel()
    flat_location = location.ravel()
    # Offset-gauge transfer from each raw reference patch into the target chart.
    candidate_values = (
        flat[references]
        - flat_location[references]
        + flat_location[:, None]
    )
    candidate_values = np.clip(candidate_values, 0.0, 1.0)
    lower, upper, medoid = _majority_interval(candidate_values)
    retained = (flat >= lower) & (flat <= upper)
    estimate = np.where(retained, flat, medoid)
    selected_distance = np.median(distances, axis=1)

    shape = observed.shape
    return TypicalOrbitState(
        estimate=estimate.reshape(shape),
        lower=lower.reshape(shape),
        upper=upper.reshape(shape),
        ambiguity=(upper - lower).reshape(shape),
        local_scale=local_scale,
        selected_distance=selected_distance.reshape(shape),
        retained_observation=retained.reshape(shape),
    )


def denoise_typical_orbit_set(
    image: np.ndarray,
    resolution: TypicalOrbitResolution = TypicalOrbitResolution(),
) -> tuple[np.ndarray, dict[str, float | int | list[int]]]:
    state = infer_typical_orbit_set(image, resolution)
    diagnostics: dict[str, float | int | list[int]] = {
        "ring_radii": list(resolution.ring_radii),
        "candidate_count": int(resolution.candidate_count),
        "retained_observation_fraction": float(np.mean(state.retained_observation)),
        "changed_fraction": float(np.mean(~state.retained_observation)),
        "median_ambiguity": float(np.median(state.ambiguity)),
        "maximum_ambiguity": float(np.max(state.ambiguity)),
        "median_selected_descriptor_distance": float(
            np.median(state.selected_distance)),
        "readout": "strict-majority component medoid; no hypothesis mean",
    }
    return state.estimate.copy(), diagnostics


def fmmt_feasible_afterpass(
    image: np.ndarray,
    resolution: TypicalOrbitResolution = TypicalOrbitResolution(),
) -> tuple[np.ndarray, dict[str, object]]:
    """Allow FMMT only inside a narrow, already-selected feasible component."""
    try:
        from .fmmt_certified import denoise_fmmt
    except ImportError:
        from fmmt_certified import denoise_fmmt

    state = infer_typical_orbit_set(image, resolution)
    fmmt, fmmt_diagnostics = denoise_fmmt(state.estimate)
    median_ambiguity = float(np.median(state.ambiguity))
    admitted = (
        (state.ambiguity <= median_ambiguity)
        & (fmmt >= state.lower)
        & (fmmt <= state.upper)
    )
    output = state.estimate.copy()
    output[admitted] = fmmt[admitted]
    diagnostics: dict[str, object] = {
        "ring_radii": list(resolution.ring_radii),
        "candidate_count": int(resolution.candidate_count),
        "fmmt_admitted_fraction": float(np.mean(admitted)),
        "median_ambiguity": median_ambiguity,
        "gate": "FMMT inside selected majority interval and no wider than median ambiguity",
        "fmmt": fmmt_diagnostics,
    }
    return output, diagnostics


def infer_local_orbit_survival(
    image: np.ndarray,
    resolution: LocalOrbitResolution = LocalOrbitResolution(),
) -> TypicalOrbitState:
    """Retain a sample unless every target-excluded line chart falsifies it.

    Each direction/scale chart predicts the target independently from its two
    sides.  The interval between those predictions is the complete bounded
    component for that chart.  A step crossing is therefore ambiguous rather
    than averaged; a locally isolated replacement is outside every component.
    """
    observed = np.asarray(image, dtype=np.float64)
    if observed.ndim != 2:
        raise ValueError("local orbit survival is grayscale 2-D")
    maximum_radius = max(resolution.radii)
    if min(observed.shape) <= 4 * maximum_radius + 1:
        raise ValueError("image is too small for the local orbit resolution")
    if not np.all(np.isfinite(observed)):
        raise ValueError("image contains non-finite values")
    observed = np.clip(observed, 0.0, 1.0)

    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    left_predictions: list[np.ndarray] = []
    right_predictions: list[np.ndarray] = []
    for radius in resolution.radii:
        for dy, dx in directions:
            near_left = _shift_reflect(observed, -radius * dy, -radius * dx)
            far_left = _shift_reflect(observed, -2 * radius * dy, -2 * radius * dx)
            near_right = _shift_reflect(observed, radius * dy, radius * dx)
            far_right = _shift_reflect(observed, 2 * radius * dy, 2 * radius * dx)
            left_predictions.append(2.0 * near_left - far_left)
            right_predictions.append(2.0 * near_right - far_right)

    left = np.stack(left_predictions, axis=0)
    right = np.stack(right_predictions, axis=0)
    lower = np.minimum(left, right)
    upper = np.maximum(left, right)
    support = (observed[None, ...] >= lower) & (observed[None, ...] <= upper)
    retained = np.any(support, axis=0)

    endpoints = np.concatenate((left, right), axis=0)
    # Every endpoint is an actual one-sided affine hypothesis.  Count how many
    # bounded components contain it, then take a medoid only to resolve equal
    # coverage.  No incompatible components are averaged.
    coverage = np.sum(
        (endpoints[:, None, ...] >= lower[None, ...])
        & (endpoints[:, None, ...] <= upper[None, ...]),
        axis=1,
    )
    maximum_coverage = np.max(coverage, axis=0)
    endpoint_median = np.median(endpoints, axis=0)
    tie_cost = np.abs(endpoints - endpoint_median[None, ...])
    tie_cost = np.where(
        coverage == maximum_coverage[None, ...], tie_cost, np.inf)
    selected_index = np.argmin(tie_cost, axis=0)
    selected = np.take_along_axis(
        endpoints, selected_index[None, ...], axis=0)[0]

    selected_membership = (
        (selected[None, ...] >= lower)
        & (selected[None, ...] <= upper)
    )
    selected_lower = np.max(
        np.where(selected_membership, lower, -np.inf), axis=0)
    selected_upper = np.min(
        np.where(selected_membership, upper, np.inf), axis=0)
    # An endpoint always belongs to at least its originating interval.
    if not np.all(np.isfinite(selected_lower) & np.isfinite(selected_upper)):
        raise RuntimeError("selected local component has no feasible interval")

    estimate = np.where(retained, observed, np.clip(selected, 0.0, 1.0))
    support_count = np.sum(support, axis=0)
    width_by_scale = (upper - lower).reshape(
        len(resolution.radii), len(directions), *observed.shape)
    support_by_scale = support.reshape(
        len(resolution.radii), len(directions), *observed.shape)
    feasible_width = np.where(support_by_scale, width_by_scale, np.inf)
    winning_direction = np.argmin(feasible_width, axis=1)
    ordered_width = np.sort(feasible_width, axis=1)
    best_width = ordered_width[:, 0]
    second_width = ordered_width[:, 1]
    finite_pair = np.isfinite(best_width) & np.isfinite(second_width)
    direction_weight = np.zeros_like(best_width)
    direction_weight[finite_pair] = (
        (second_width[finite_pair] - best_width[finite_pair])
        / np.maximum(
            second_width[finite_pair] + best_width[finite_pair],
            np.finfo(np.float64).eps,
        )
    )
    # Projective D4 tangent phases: theta and theta+pi are one direction.
    phase_real = np.array((1.0, -1.0, 0.0, 0.0))
    phase_imag = np.array((0.0, 0.0, 1.0, -1.0))
    real = np.sum(
        direction_weight
        * phase_real[winning_direction],
        axis=0,
    )
    imag = np.sum(
        direction_weight
        * phase_imag[winning_direction],
        axis=0,
    )
    total_weight = np.sum(direction_weight, axis=0)
    active_scales = np.sum(
        direction_weight > np.sqrt(np.finfo(np.float64).eps), axis=0)
    resultant = np.hypot(real, imag) / np.maximum(
        total_weight, np.finfo(np.float64).eps)
    # The acceptance cone is exactly half one D4 angular cell.
    orbit_coherence = (
        (active_scales == len(resolution.radii))
        & (resultant >= np.cos(np.pi / 4.0))
    )
    return TypicalOrbitState(
        estimate=estimate,
        lower=np.clip(selected_lower, 0.0, 1.0),
        upper=np.clip(selected_upper, 0.0, 1.0),
        ambiguity=np.maximum(selected_upper - selected_lower, 0.0),
        local_scale=np.median(upper - lower, axis=0),
        selected_distance=(len(left_predictions) - support_count).astype(np.float64),
        retained_observation=retained,
        observation_support_count=support_count,
        component_lower=lower,
        component_upper=upper,
        orbit_coherence=orbit_coherence,
    )


def denoise_local_orbit_survival(
    image: np.ndarray,
    resolution: LocalOrbitResolution = LocalOrbitResolution(),
) -> tuple[np.ndarray, dict[str, object]]:
    state = infer_local_orbit_survival(image, resolution)
    diagnostics: dict[str, object] = {
        "radii": list(resolution.radii),
        "direction_count": 4,
        "component_count": 4 * len(resolution.radii),
        "retained_observation_fraction": float(np.mean(state.retained_observation)),
        "conclusively_falsified_fraction": float(np.mean(~state.retained_observation)),
        "median_selected_component_width": float(np.median(state.ambiguity)),
        "readout": "observation unless all charts falsify; otherwise actual affine endpoint",
    }
    return state.estimate.copy(), diagnostics


def fmmt_local_feasible_afterpass(
    image: np.ndarray,
    resolution: LocalOrbitResolution = LocalOrbitResolution(),
) -> tuple[np.ndarray, dict[str, object]]:
    """Run FMMT after structural falsification, constrained to that component."""
    try:
        from .fmmt_certified import denoise_fmmt
    except ImportError:
        from fmmt_certified import denoise_fmmt

    state = infer_local_orbit_survival(image, resolution)
    fmmt, fmmt_diagnostics = denoise_fmmt(state.estimate)
    # FMMT has authority only where the structural stage made an affirmative
    # replacement decision, and only inside the selected feasible component.
    admitted = (
        (~state.retained_observation)
        & (fmmt >= state.lower)
        & (fmmt <= state.upper)
    )
    output = state.estimate.copy()
    output[admitted] = fmmt[admitted]
    return output, {
        "radii": list(resolution.radii),
        "fmmt_admitted_fraction": float(np.mean(admitted)),
        "structurally_falsified_fraction": float(
            np.mean(~state.retained_observation)),
        "gate": "only structurally replaced samples; FMMT must remain in selected component",
        "fmmt": fmmt_diagnostics,
    }


def fmmt_local_redundancy_afterpass(
    image: np.ndarray,
    resolution: LocalOrbitResolution = LocalOrbitResolution(),
) -> tuple[np.ndarray, dict[str, object]]:
    """Let FMMT act only inside a strict-majority local structural component."""
    try:
        from .fmmt_certified import denoise_fmmt
    except ImportError:
        from fmmt_certified import denoise_fmmt

    observed = np.clip(np.asarray(image, np.float64), 0.0, 1.0)
    state = infer_local_orbit_survival(observed, resolution)
    fmmt, fmmt_diagnostics = denoise_fmmt(observed)
    if (
        state.observation_support_count is None
        or state.component_lower is None
        or state.component_upper is None
    ):
        raise RuntimeError("local component ledger is unavailable")
    component_count = state.component_lower.shape[0]
    strict_majority = component_count // 2 + 1
    observed_membership = (
        (observed[None, ...] >= state.component_lower)
        & (observed[None, ...] <= state.component_upper)
    )
    fmmt_membership = (
        (fmmt[None, ...] >= state.component_lower)
        & (fmmt[None, ...] <= state.component_upper)
    )
    shared_component = np.any(observed_membership & fmmt_membership, axis=0)
    redundant = state.observation_support_count >= strict_majority
    retained_admission = state.retained_observation & redundant & shared_component
    replaced_admission = (
        (~state.retained_observation)
        & (fmmt >= state.lower)
        & (fmmt <= state.upper)
    )
    admitted = retained_admission | replaced_admission
    output = state.estimate.copy()
    output[admitted] = fmmt[admitted]
    return output, {
        "radii": list(resolution.radii),
        "component_count": int(component_count),
        "strict_majority": int(strict_majority),
        "redundant_fraction": float(np.mean(redundant)),
        "fmmt_admitted_fraction": float(np.mean(admitted)),
        "retained_admission_fraction": float(np.mean(retained_admission)),
        "replaced_admission_fraction": float(np.mean(replaced_admission)),
        "gate": "strict chart majority and one shared bounded component",
        "fmmt": fmmt_diagnostics,
    }


def fmmt_local_coherence_veto(
    image: np.ndarray,
    resolution: LocalOrbitResolution = LocalOrbitResolution(),
) -> tuple[np.ndarray, dict[str, object]]:
    """FMMT cleanup with a cross-scale local-orbit structural veto."""
    try:
        from .fmmt_certified import denoise_fmmt
    except ImportError:
        from fmmt_certified import denoise_fmmt

    observed = np.clip(np.asarray(image, np.float64), 0.0, 1.0)
    state = infer_local_orbit_survival(observed, resolution)
    fmmt, fmmt_diagnostics = denoise_fmmt(observed)
    if state.orbit_coherence is None:
        raise RuntimeError("cross-scale orbit coherence is unavailable")
    protected = state.orbit_coherence & state.retained_observation
    falsified = ~state.retained_observation
    output = fmmt.copy()
    output[protected] = observed[protected]
    output[falsified] = state.estimate[falsified]
    return output, {
        "radii": list(resolution.radii),
        "protected_cross_scale_orbit_fraction": float(np.mean(protected)),
        "locally_falsified_fraction": float(np.mean(falsified)),
        "gate": "projective tangent resultant within half one D4 angular cell",
        "fmmt": fmmt_diagnostics,
    }
