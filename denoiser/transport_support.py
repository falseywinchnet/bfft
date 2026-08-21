"""Continuous, budget-stopped support transport for the FMMT experiment.

This module is intentionally a research layer, not a replacement claim.  It
replaces the checkpoint's fixed support scales, threshold ramps, and sweep
count with three state-derived objects:

* a scale-space support density measured by agreement between disjoint
  observation lanes;
* an observation authority obtained from empirical entropy and residual
  participation; and
* a conservative flux whose horizon is the observation's unsupported
  residual-action budget.

The same equations work on a line or an image.  Grid steps and scale-space
quadrature remain numerical representations of a continuum law; they are
reported separately from model state in every diagnostic record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class TransportResolution:
    """Numerical resolution controls, never fitted against image quality."""

    scale_samples: int = 7
    histogram_bins: int = 64
    maximum_steps: int = 4096
    courant_fraction: float = 0.90


def _validate_field(value: np.ndarray) -> np.ndarray:
    field = np.asarray(value, dtype=np.float64)
    if field.ndim not in (1, 2):
        raise ValueError("support transport expects a 1-D or 2-D scalar field")
    if min(field.shape) < 8:
        raise ValueError("every transported axis must contain at least 8 samples")
    if not np.all(np.isfinite(field)):
        raise ValueError("support transport requires finite samples")
    return field


def continuum_scales(shape: tuple[int, ...], samples: int) -> np.ndarray:
    """Log-quadrature nodes from one grid cell to one natural domain period."""
    if samples < 2:
        raise ValueError("scale quadrature needs at least two nodes")
    upper = max(1.0, min(shape) / (2.0 * math.pi))
    return np.geomspace(1.0, upper, int(samples), dtype=np.float64)


@lru_cache(maxsize=16)
def _residue_lanes(shape: tuple[int, ...]) -> np.ndarray:
    coordinates = np.indices(shape)
    parity = np.mod(np.sum(coordinates, axis=0), 2)
    return np.stack((parity == 0, parity == 1)).astype(np.float64)


def _lane_scale_space(field: np.ndarray, sigma: float) -> np.ndarray:
    lanes = _residue_lanes(field.shape)
    # The two checkerboard lanes are exact complements. Gaussian filtering is
    # linear and preserves constants under reflection, so one lane plus the
    # full field determines the other. This removes one mask filter and one
    # masked-field filter per physical scale.
    denominator_first = ndimage.gaussian_filter(
        lanes[0], sigma, mode="reflect")
    numerator_first = ndimage.gaussian_filter(
        field * lanes[0], sigma, mode="reflect")
    full = ndimage.gaussian_filter(field, sigma, mode="reflect")
    denominator_second = 1.0 - denominator_first
    numerator_second = full - numerator_first
    tiny = np.finfo(float).tiny
    return np.stack((
        numerator_first / np.maximum(denominator_first, tiny),
        numerator_second / np.maximum(denominator_second, tiny),
    ))


def support_density(
    observation: np.ndarray,
    resolution: TransportResolution = TransportResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a smooth support density integrated over physical scale.

    At each log-scale, ``q`` is reproducible curvature divided by lane
    disagreement.  ``q²/(1+q²)`` is a continuous evidence map with no
    classification threshold.  Logarithmic averaging makes the result
    invariant to the number of quadrature nodes in the continuum limit.
    """
    field = _validate_field(observation)
    scales = continuum_scales(field.shape, resolution.scale_samples)
    log_survival = np.zeros_like(field)
    evidence_means = []
    for sigma in scales:
        lane_estimates = _lane_scale_space(field, float(sigma))
        lane_curvature = np.stack([
            -(sigma * sigma) * ndimage.gaussian_laplace(
                estimate, sigma, mode="reflect")
            for estimate in lane_estimates
        ])
        center = np.mean(lane_curvature, axis=0)
        disagreement = np.sqrt(np.mean(
            (lane_curvature - center[None, ...]) ** 2, axis=0))
        reference = np.sqrt(np.mean(center * center))
        denominator = np.hypot(disagreement, np.finfo(float).eps * max(reference, 1.0))
        ratio = np.abs(center) / denominator
        reproducibility = ratio * ratio / (1.0 + ratio * ratio)
        # Agreement is insufficient by itself: at broad scales two lanes can
        # agree on an arbitrarily weak random curvature.  Compare the shared
        # curvature with its own scale energy.  This is a homogeneous,
        # state-derived normalization, not an amplitude threshold.
        scale_energy = float(np.mean(center * center))
        amplitude = center * center / np.maximum(
            center * center + scale_energy, np.finfo(float).tiny)
        curvature_evidence = reproducibility * amplitude

        lane_gradients = np.stack([
            np.stack(np.gradient(estimate), axis=0) * sigma
            for estimate in lane_estimates
        ])
        gradient_center = np.mean(lane_gradients, axis=0)
        gradient_disagreement = np.sqrt(np.mean(np.sum(
            (lane_gradients - gradient_center[None, ...]) ** 2,
            axis=1,
        ), axis=0))
        gradient_energy = np.sum(gradient_center * gradient_center, axis=0)
        gradient_reference = float(np.mean(gradient_energy))
        gradient_ratio = np.sqrt(gradient_energy) / np.hypot(
            gradient_disagreement,
            np.finfo(float).eps * max(math.sqrt(gradient_reference), 1.0),
        )
        gradient_reproducibility = (
            gradient_ratio * gradient_ratio
            / (1.0 + gradient_ratio * gradient_ratio)
        )
        gradient_amplitude = gradient_energy / np.maximum(
            gradient_energy + gradient_reference, np.finfo(float).tiny)
        edge_evidence = gradient_reproducibility * gradient_amplitude
        evidence = 1.0 - (1.0 - curvature_evidence) * (1.0 - edge_evidence)
        log_survival += np.log1p(-np.minimum(evidence, 1.0 - np.finfo(float).eps))
        evidence_means.append(float(np.mean(evidence)))
    density = 1.0 - np.exp(log_survival / len(scales))
    density = ndimage.gaussian_filter(density, scales[0], mode="reflect")
    return density, {
        "scale_quadrature": [float(value) for value in scales],
        "scale_evidence_mean": evidence_means,
        "mean_support_density": float(np.mean(density)),
    }


def observation_authority(
    observation: np.ndarray,
    provisional: np.ndarray,
    resolution: TransportResolution = TransportResolution(),
) -> tuple[float, dict[str, float]]:
    """Measure whether the unchanged observation may rewrite its bootstrap.

    Histogram entropy measures how much of the available intensity state is
    occupied.  Residual participation is one only when residual energy is
    distributed, and tends smoothly to zero when a few remote atoms own it.
    Neither term contains a hand-selected transition threshold.
    """
    y = _validate_field(observation)
    x0 = _validate_field(provisional)
    if y.shape != x0.shape:
        raise ValueError("observation and provisional state must align")
    bins = int(resolution.histogram_bins)
    counts, _ = np.histogram(np.clip(y, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    probability = counts.astype(np.float64) / max(float(np.sum(counts)), 1.0)
    nonzero = probability > 0.0
    entropy = -float(np.sum(probability[nonzero] * np.log(probability[nonzero])))
    entropy /= math.log(bins)

    residual = np.abs(y - x0).ravel()
    l1 = float(np.mean(residual))
    l2_squared = float(np.mean(residual * residual))
    participation = l1 * l1 / max(l2_squared, np.finfo(float).tiny)
    authority = math.sqrt(max(entropy * participation, 0.0))
    return authority, {
        "observation_entropy": entropy,
        "residual_participation": participation,
        "support_witness_authority": authority,
    }


def _edge_conductances(conductance: np.ndarray) -> list[np.ndarray]:
    result = []
    for axis in range(conductance.ndim):
        left = [slice(None)] * conductance.ndim
        right = [slice(None)] * conductance.ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        result.append(0.5 * (conductance[tuple(left)] + conductance[tuple(right)]))
    return result


def _flux_step(
    state: np.ndarray,
    edges: list[np.ndarray],
    dt: float,
) -> tuple[np.ndarray, float, float]:
    update = np.zeros_like(state)
    action = 0.0
    flux_squared = 0.0
    tiny = np.finfo(float).tiny
    for axis, edge in enumerate(edges):
        left = [slice(None)] * state.ndim
        right = [slice(None)] * state.ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        gradient = state[tuple(right)] - state[tuple(left)]
        flux = dt * edge * gradient
        update[tuple(left)] += flux
        update[tuple(right)] -= flux
        action += float(np.sum(flux * flux / np.maximum(dt * edge, tiny)))
        flux_squared += float(np.sum((edge * gradient) ** 2))
    return state + update, action, flux_squared


def transport_support_birth(
    observation: np.ndarray,
    provisional: np.ndarray,
    resolution: TransportResolution = TransportResolution(),
    *,
    action_budget_multiplier: float = 1.0,
    support_field: np.ndarray | None = None,
    support_diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Evolve unsupported state until its measured action budget is spent.

    The flow is ``dc/dt = div((1-S)^2 a grad(c))``.  Its action budget is
    the unsupported provisional residual energy available in the unchanged
    observation.  The last numerical step is shortened to spend that budget
    exactly, so ``maximum_steps`` is only a failure guard.
    """
    y = _validate_field(observation)
    state = _validate_field(provisional).copy()
    budget_multiplier = float(action_budget_multiplier)
    if not np.isfinite(budget_multiplier) or budget_multiplier < 0.0:
        raise ValueError("action_budget_multiplier must be finite and nonnegative")
    if y.shape != state.shape:
        raise ValueError("observation and provisional state must align")
    if support_field is None:
        support, support_diag = support_density(y, resolution)
    else:
        support = _validate_field(support_field)
        if support.shape != y.shape:
            raise ValueError("precomputed support must align with observation")
        if np.any((support < 0.0) | (support > 1.0)):
            raise ValueError("precomputed support must be a density in [0, 1]")
        support_diag = dict(support_diagnostics or {})
        support_diag.setdefault("mean_support_density", float(np.mean(support)))
    authority, authority_diag = observation_authority(y, state, resolution)
    conductance = authority * (1.0 - support) ** 2
    edges = _edge_conductances(conductance)
    maximum_edge = max((float(np.max(edge)) for edge in edges), default=0.0)
    residual = y - state
    scales = continuum_scales(y.shape, resolution.scale_samples)
    transportable_residual = np.mean(np.stack([
        ndimage.gaussian_filter(residual, float(sigma), mode="reflect")
        for sigma in scales
    ]), axis=0)
    measured_budget = 0.5 * authority * float(np.sum(
        (1.0 - support) * transportable_residual * transportable_residual))
    budget = budget_multiplier * measured_budget
    initial_mass = float(np.sum(state))
    initial_min = float(np.min(state))
    initial_max = float(np.max(state))
    action = 0.0
    steps = 0
    stopped_by = "zero_transport"
    flux_power = 0.0

    if budget > 0.0 and maximum_edge > 0.0:
        dt = resolution.courant_fraction / (2.0 * state.ndim * maximum_edge)
        stopped_by = "numerical_guard"
        for step in range(int(resolution.maximum_steps)):
            candidate, increment, flux_power = _flux_step(state, edges, dt)
            if increment <= np.finfo(float).eps * max(budget, 1.0):
                stopped_by = "stationary_flow"
                break
            remaining = budget - action
            if increment >= remaining:
                fraction = remaining / increment
                state, spent, flux_power = _flux_step(state, edges, dt * fraction)
                action += spent
                steps = step + 1
                stopped_by = "action_budget"
                break
            state = candidate
            action += increment
            steps = step + 1

    barrier_gate = 1.0 - authority * (1.0 - support)
    diagnostics: dict[str, Any] = {
        "support_law": "continuous scale agreement + conservative action budget",
        "transport_dimension": int(state.ndim),
        "transport_action_budget": budget,
        "measured_transport_action_budget": measured_budget,
        "action_budget_multiplier": budget_multiplier,
        "transport_action_spent": action,
        "transport_steps": steps,
        "transport_stop": stopped_by,
        "terminal_flux_power": flux_power,
        "mass_conservation_error": float(np.sum(state)) - initial_mass,
        "maximum_principle_undershoot": max(initial_min - float(np.min(state)), 0.0),
        "maximum_principle_overshoot": max(float(np.max(state)) - initial_max, 0.0),
        "mean_bootstrap_conductance": float(np.mean(conductance)),
        "mean_eikonal_barrier_gate": float(np.mean(barrier_gate)),
        "numerical_resolution": asdict(resolution),
        **support_diag,
        **authority_diag,
    }
    return state, barrier_gate, diagnostics


def denoise_1d(
    observation: np.ndarray,
    resolution: TransportResolution = TransportResolution(),
    *,
    provisional_sigma: float = 1.0,
    action_budget_multiplier: float = 1.0,
    continuation_rounds: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One-dimensional test form of the support law.

    A Gaussian semigroup supplies a provisional chart. Its scale, the measured
    action budget multiple, and the number of discrepancy continuations are
    explicit laboratory controls. Every continuation recomputes a fresh
    observation-derived budget; the conservative equation remains the same
    support transport used by the 2-D FMMT path.
    """
    y = _validate_field(observation)
    if y.ndim != 1:
        raise ValueError("denoise_1d expects a vector")
    sigma = float(provisional_sigma)
    rounds = int(continuation_rounds)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("provisional_sigma must be finite and nonnegative")
    if rounds < 1:
        raise ValueError("continuation_rounds must be positive")
    state = (
        ndimage.gaussian_filter1d(y, sigma, mode="reflect")
        if sigma > 0.0 else y.copy()
    )
    records = []
    for _round in range(rounds):
        state, _barrier, record = transport_support_birth(
            y,
            state,
            resolution,
            action_budget_multiplier=action_budget_multiplier,
        )
        records.append(record)
    diagnostics = dict(records[-1])
    diagnostics.update({
        "provisional_operator": "Gaussian semigroup",
        "provisional_sigma": sigma,
        "continuation_rounds": rounds,
        "total_transport_steps": int(sum(
            record["transport_steps"] for record in records)),
        "total_transport_action_spent": float(sum(
            record["transport_action_spent"] for record in records)),
        "round_diagnostics": records,
    })
    return state, diagnostics


def denoise_2d_fmmt(
    observation: np.ndarray,
    *,
    resolution: TransportResolution = TransportResolution(),
    precomputed_support: tuple[np.ndarray, dict[str, Any]] | None = None,
    **fmmt_options: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the unchanged FMMT measure posterior after transport support birth."""
    try:
        from .fmmt_certified import denoise_fmmt
    except ImportError:  # Direct script import used by the GUI and CLI.
        from fmmt_certified import denoise_fmmt

    def operator(y: np.ndarray, provisional: np.ndarray):
        if precomputed_support is None:
            return transport_support_birth(y, provisional, resolution)
        support, support_diagnostic = precomputed_support
        return transport_support_birth(
            y,
            provisional,
            resolution,
            support_field=support,
            support_diagnostics=support_diagnostic,
        )

    output, diagnostics = denoise_fmmt(
        observation,
        support_operator=operator,
        certify_support=False,
        **fmmt_options,
    )
    diagnostics["certified_support_birth"] = "continuous_transport"
    return output, diagnostics
