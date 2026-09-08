"""Transport-admissible Chambolle denoising.

Chambolle's projection algorithm can be separated into four pieces:

1. a Euclidean observation fidelity, whose readout is ``y + D* p``;
2. a fixed Cartesian gradient/divergence pair ``D, D*``;
3. one isotropic, spatially constant dual flux ball;
4. projected descent (optionally accelerated) on the dual flux.

This module retains the convex dual-flux mechanism and replaces piece 3 by a
transport-derived field of oriented zonotopes.  Four parity-shifted CONV*
resize charts provide three target-excluded witnesses at every pixel.  Their
distribution determines both how much of the observation is unsupported and
which gradient direction persists across observers.  No noise class, band,
edge threshold, or user denoising weight enters the operator.

The resulting fixed-observer problem is still a single convex problem::

    min_{p(x) in K_x}  1/2 ||r_noise + D* p||^2,

where ``r_noise`` is the signed incoherent part of resize resistance and
``K_x`` is the local transport flux zonotope.  Thus the flux is asked to
explain measured noise, not given an unsigned license to reduce arbitrary
image variation.  The final readout is ``u = y + D* p``.  FISTA momentum lives
only on the flux, following the useful lesson of the Meyer-Bregman
experiments; an objective restart keeps the contraction monotone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .dcnt import (
        DCNTResolution,
        _meyer_rof,
        _transport_uncertainty_batch,
        _validate_image,
        minimal_supported_noise,
        transport_uncertainty,
    )
except ImportError:  # pragma: no cover - direct research-script execution
    from dcnt import (
        DCNTResolution,
        _meyer_rof,
        _transport_uncertainty_batch,
        _validate_image,
        minimal_supported_noise,
        transport_uncertainty,
    )


Array = np.ndarray
_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class TransportChambolleResolution:
    """Numerical resolution, not denoising strength."""

    maximum_iterations: int = 200
    relative_tolerance: float = 2.0e-6
    minimum_iterations: int = 8
    maximum_observer_cycles: int = 8


@dataclass(frozen=True)
class TransportFluxBody:
    """A pixelwise oriented rectangular (zonotopic) dual flux body."""

    normal_x: Array
    normal_y: Array
    normal_capacity: Array
    tangent_capacity: Array
    unsupported_amplitude: Array
    coherent_resistance: Array
    removable_amplitude: Array
    predictive_scale: Array
    coherence: Array
    centre: Array
    chart_family: Array


def _forward_gradient(field: Array) -> tuple[Array, Array]:
    """Cartesian forward gradient with Chambolle's free-end closure."""

    value = np.asarray(field, dtype=np.float64)
    gy = np.zeros_like(value)
    gx = np.zeros_like(value)
    gy[:-1] = value[1:] - value[:-1]
    gx[:, :-1] = value[:, 1:] - value[:, :-1]
    return gy, gx


def _flux_readout(flux_y: Array, flux_x: Array) -> Array:
    """Return the adjoint readout used by scikit's Chambolle recurrence."""

    out = -flux_y - flux_x
    out = out.copy()
    out[1:] += flux_y[:-1]
    out[:, 1:] += flux_x[:, :-1]
    return out


def _positive_excess_structure(
    mean_y: Array,
    mean_x: Array,
    variance_y: Array,
    variance_x: Array,
) -> tuple[Array, Array, Array]:
    """Separate chart-persistent gradient energy from chart disagreement."""

    persistent = mean_y * mean_y + mean_x * mean_x
    disagreement = variance_y + variance_x
    excess = np.maximum(persistent - disagreement, 0.0)
    numerical = 64.0 * _EPS * np.maximum(persistent + disagreement, 1.0)
    coherence = np.divide(
        excess,
        persistent + disagreement + numerical,
        out=np.zeros_like(excess),
        where=(persistent + disagreement + numerical) > 0.0,
    )
    magnitude = np.hypot(mean_y, mean_x)
    normal_y = np.divide(
        mean_y, magnitude, out=np.zeros_like(mean_y), where=magnitude > numerical)
    normal_x = np.divide(
        mean_x, magnitude, out=np.ones_like(mean_x), where=magnitude > numerical)
    return normal_y, normal_x, np.clip(coherence, 0.0, 1.0)


def _transport_flux_body_from_law(
    image: Array,
    law,
) -> TransportFluxBody:
    """Complete one flux body from an already transported uncertainty law.

    The predictive scale is the RMS radius of the three finite witnesses
    about their median. ``unsupported_amplitude`` is the distance from the
    wider zonotope interval: uncertainty about transport itself is therefore
    retained before resistance is decomposed.

    The chart-mean gradient supplies the candidate structure normal.  Only
    gradient energy above between-chart disagreement narrows flux in that
    direction.  The tangent remains available for contour-wise cleanup.
    """

    finite = np.isfinite(law.family)
    count = np.sum(finite, axis=0)
    if not np.all(count == 3):
        raise RuntimeError("transport flux body requires three held-out witnesses")

    predictive_scale = np.sqrt(law.second_moment / count)
    # The entire generator sum is uncertainty about resize transport itself.
    # Anything outside it resists every admissible chart, but resistance alone
    # is not evidence of noise: a fine coherent line resists coarse transport
    # too.  The Meyer cartoon proximal keeps that coherent resistance; only
    # the complementary oscillation may generate denoising flux.
    unsupported, _admitted = minimal_supported_noise(
        image, law.centre, law.zonotope_radius)
    coherent_resistance, _cartoon_diagnostic = _meyer_rof(
        unsupported, DCNTResolution())
    removable = unsupported - coherent_resistance

    # At each chart's own anchor, substitute the target-excluded median only
    # for differentiating that chart.  The observation is never inserted.
    completed = np.where(finite, law.family, law.centre[None, ...])
    gradients_y = np.empty_like(completed)
    gradients_x = np.empty_like(completed)
    for chart in range(completed.shape[0]):
        gradients_y[chart], gradients_x[chart] = _forward_gradient(
            completed[chart])
    mean_y = np.mean(gradients_y, axis=0)
    mean_x = np.mean(gradients_x, axis=0)
    variance_y = np.mean((gradients_y - mean_y) ** 2, axis=0)
    variance_x = np.mean((gradients_x - mean_x) ** 2, axis=0)
    normal_y, normal_x, chart_coherence = _positive_excess_structure(
        mean_y, mean_x, variance_y, variance_x)

    point_capacity = np.abs(removable)
    # Chambolle stores both outgoing edge fluxes at a lattice site.  A point
    # residual must authorize every edge incident to that point, including the
    # coordinates stored at its upper and left neighbours.  The local maximum
    # is the exact union of endpoint permissions; using only the source site
    # produces a one-sided discrete contraction.
    capacity = point_capacity.copy()
    capacity[:-1] = np.maximum(capacity[:-1], point_capacity[1:])
    capacity[1:] = np.maximum(capacity[1:], point_capacity[:-1])
    capacity[:, :-1] = np.maximum(capacity[:, :-1], point_capacity[:, 1:])
    capacity[:, 1:] = np.maximum(capacity[:, 1:], point_capacity[:, :-1])
    # A supported underlying edge narrows cross-contour flux only while its
    # predictive action dominates the removable residual.  At an impulsive
    # corruption on a real edge, noise action relaxes that protection
    # continuously instead of forcing the edge and noise into one class.
    predictive_action = predictive_scale * predictive_scale
    removable_action = removable * removable
    attribution = np.divide(
        predictive_action,
        predictive_action + removable_action + 64.0 * _EPS,
        out=np.zeros_like(predictive_action),
        where=(predictive_action + removable_action) > 0.0,
    )
    coherence = chart_coherence * attribution
    # The supported structure fraction removes cross-contour authority.  This
    # is a continuous quotient of measured actions, not an edge threshold.
    normal_capacity = capacity * (1.0 - coherence)
    tangent_capacity = capacity
    return TransportFluxBody(
        normal_x=np.ascontiguousarray(normal_x),
        normal_y=np.ascontiguousarray(normal_y),
        normal_capacity=np.ascontiguousarray(normal_capacity),
        tangent_capacity=np.ascontiguousarray(tangent_capacity),
        unsupported_amplitude=np.ascontiguousarray(unsupported),
        coherent_resistance=np.ascontiguousarray(coherent_resistance),
        removable_amplitude=np.ascontiguousarray(removable),
        predictive_scale=np.ascontiguousarray(predictive_scale),
        coherence=np.ascontiguousarray(coherence),
        centre=np.ascontiguousarray(law.centre),
        chart_family=np.ascontiguousarray(law.family),
    )


def transport_flux_body(observation: Array) -> TransportFluxBody:
    """Measure one local dual flux body from target-excluded resize charts."""

    image = _validate_image(observation)
    return _transport_flux_body_from_law(image, transport_uncertainty(image))


def transport_flux_bodies(observations: Array) -> list[TransportFluxBody]:
    """Measure independent flux bodies through one component-batched CONV*.

    Only the resize transport is shared.  Meyer resistance, chart statistics,
    and every returned body remain separate, so this is a representation
    optimization rather than a population coupling.
    """

    images = np.asarray(observations, dtype=np.float64)
    if images.ndim != 3 or min(images.shape[1:]) < 10:
        raise ValueError("flux-body population must have shape BxHxW")
    if not np.all(np.isfinite(images)):
        raise ValueError("transport flux bodies require finite samples")
    law = _transport_uncertainty_batch(images)
    bodies: list[TransportFluxBody] = []
    for index, image in enumerate(images):
        scalar_law = type(law)(
            centre=law.centre[index],
            second_moment=law.second_moment[index],
            convex_radius=law.convex_radius[index],
            zonotope_radius=law.zonotope_radius[index],
            family=law.family[index],
        )
        bodies.append(_transport_flux_body_from_law(image, scalar_law))
    return bodies


def _project_flux_zonotope(
    flux_y: Array,
    flux_x: Array,
    body: TransportFluxBody,
) -> tuple[Array, Array]:
    """Euclidean projection onto each oriented rectangular flux body."""

    normal_component = flux_y * body.normal_y + flux_x * body.normal_x
    tangent_component = -flux_y * body.normal_x + flux_x * body.normal_y
    normal_component = np.clip(
        normal_component, -body.normal_capacity, body.normal_capacity)
    tangent_component = np.clip(
        tangent_component, -body.tangent_capacity, body.tangent_capacity)
    projected_y = (
        normal_component * body.normal_y
        - tangent_component * body.normal_x
    )
    projected_x = (
        normal_component * body.normal_x
        + tangent_component * body.normal_y
    )
    return projected_y, projected_x


def _contraction_objective(
    removable: Array,
    flux_y: Array,
    flux_x: Array,
) -> float:
    unexplained = removable + _flux_readout(flux_y, flux_x)
    return float(0.5 * np.sum(unexplained * unexplained))


def _contract_transport_chambolle_once(
    observation: Array,
    resolution: TransportChambolleResolution,
    *,
    body: TransportFluxBody | None = None,
    return_diagnostics: bool = True,
) -> tuple[Array, dict[str, Any]] | Array:
    """Resolve one frozen-observer transport flux contraction."""

    image = _validate_image(observation)
    maximum = int(resolution.maximum_iterations)
    minimum = int(resolution.minimum_iterations)
    tolerance = float(resolution.relative_tolerance)
    if maximum < 1 or minimum < 0 or tolerance < 0.0:
        raise ValueError("invalid transport Chambolle numerical resolution")

    body = transport_flux_body(image) if body is None else body
    if not np.any(body.tangent_capacity > 0.0):
        diagnostic = {
            "method": "transport-admissible Chambolle",
            "status": "transport identity",
            "iterations": 0,
            "restarts": 0,
            "mean_predictive_scale": float(np.mean(body.predictive_scale)),
            "mean_unsupported_amplitude": 0.0,
            "mean_removable_amplitude": 0.0,
            "mean_structure_coherence": float(np.mean(body.coherence)),
            "contraction_objective_trace": [0.0],
            "unexplained_noise_action": 0.0,
            "recomposition_error": 0.0,
        }
        return (image.copy(), diagnostic) if return_diagnostics else image.copy()

    flux_y = np.zeros_like(image)
    flux_x = np.zeros_like(image)
    previous_y = flux_y.copy()
    previous_x = flux_x.copy()
    momentum_clock = 1.0
    step = 1.0 / (2.0 * image.ndim)
    removable = body.removable_amplitude
    current_objective = float(0.5 * np.sum(removable * removable))
    objective_trace = [current_objective]
    restarts = 0
    converged = False
    capacity_scale = max(float(np.sqrt(np.mean(
        body.tangent_capacity * body.tangent_capacity))), _EPS)

    for iteration in range(maximum):
        new_clock = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum_clock**2))
        momentum = (momentum_clock - 1.0) / new_clock
        extrapolated_y = flux_y + momentum * (flux_y - previous_y)
        extrapolated_x = flux_x + momentum * (flux_x - previous_x)
        unexplained = removable + _flux_readout(
            extrapolated_y, extrapolated_x)
        gradient_y, gradient_x = _forward_gradient(unexplained)
        candidate_y, candidate_x = _project_flux_zonotope(
            extrapolated_y - step * gradient_y,
            extrapolated_x - step * gradient_x,
            body,
        )
        candidate_objective = _contraction_objective(
            removable, candidate_y, candidate_x)

        # Adaptive restart is a numerical safeguard on the convex dual, not a
        # data-dependent regularization decision.
        numerical = 128.0 * _EPS * max(abs(current_objective), image.size, 1.0)
        if candidate_objective > current_objective + numerical:
            restarts += 1
            new_clock = 1.0
            momentum = 0.0
            unexplained = removable + _flux_readout(flux_y, flux_x)
            gradient_y, gradient_x = _forward_gradient(unexplained)
            candidate_y, candidate_x = _project_flux_zonotope(
                flux_y - step * gradient_y,
                flux_x - step * gradient_x,
                body,
            )
            candidate_objective = _contraction_objective(
                removable, candidate_y, candidate_x)

        delta = np.sqrt(np.mean(
            (candidate_y - flux_y) ** 2 + (candidate_x - flux_x) ** 2))
        previous_y, previous_x = flux_y, flux_x
        flux_y, flux_x = candidate_y, candidate_x
        momentum_clock = new_clock
        objective_change = abs(candidate_objective - current_objective)
        current_objective = candidate_objective
        objective_trace.append(current_objective)
        objective_scale = max(abs(current_objective), float(image.size) * _EPS)
        if (
            iteration + 1 >= minimum
            and delta <= tolerance * capacity_scale
            and objective_change <= tolerance * objective_scale
        ):
            converged = True
            break

    estimate = image + _flux_readout(flux_y, flux_x)
    residual = image - estimate
    recomposition_error = float(np.max(np.abs(estimate + residual - image)))
    diagnostic = {
        "method": "transport-admissible Chambolle",
        "status": "dual equilibrium" if converged else "iteration ceiling",
        "iterations": int(iteration + 1),
        "restarts": int(restarts),
        "step": step,
        "mean_predictive_scale": float(np.mean(body.predictive_scale)),
        "mean_unsupported_amplitude": float(np.mean(
            np.abs(body.unsupported_amplitude))),
        "mean_coherent_resistance": float(np.mean(
            np.abs(body.coherent_resistance))),
        "mean_removable_amplitude": float(np.mean(
            np.abs(body.removable_amplitude))),
        "unsupported_fraction": float(np.mean(body.tangent_capacity > 0.0)),
        "mean_structure_coherence": float(np.mean(body.coherence)),
        "mean_normal_capacity": float(np.mean(body.normal_capacity)),
        "mean_tangent_capacity": float(np.mean(body.tangent_capacity)),
        "contraction_objective_trace": objective_trace,
        "unexplained_noise_action": _contraction_objective(
            removable, flux_y, flux_x),
        "recomposition_error": recomposition_error,
        "observer": "three target-excluded CONV* resize witnesses",
        "resistance_contractor": "Meyer ROF coherent-resistance proximal",
        "flux_body": "oriented transport zonotope",
        "convex_action": "closest divergence-compatible removable residual",
    }
    return (estimate, diagnostic) if return_diagnostics else estimate


def denoise_transport_chambolle(
    observation: Array,
    resolution: TransportChambolleResolution = TransportChambolleResolution(),
    *,
    return_diagnostics: bool = True,
) -> tuple[Array, dict[str, Any]] | Array:
    """Re-observe and contract until noise and structure actions balance.

    Each inner problem is a static convex projected-flux contraction.  Its
    output is then passed through the target-excluded resize observer again.
    The outer evolution stops when the next removable action no longer exceeds
    the next chart-supported structure action.  ``maximum_observer_cycles`` is
    only a numerical guard on a failed phase crossing.
    """

    image = _validate_image(observation)
    ceiling = int(resolution.maximum_observer_cycles)
    if ceiling < 1:
        raise ValueError("maximum_observer_cycles must be positive")
    current = image.copy()
    body = transport_flux_body(current)
    history: list[dict[str, Any]] = []
    phase_balanced = False
    last: dict[str, Any] = {}
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)), 1.0)
    numerical = 256.0 * _EPS * magnitude * magnitude
    for cycle in range(ceiling):
        candidate, last = _contract_transport_chambolle_once(
            current, resolution, body=body, return_diagnostics=True)
        update_rms = float(np.sqrt(np.mean((candidate - current) ** 2)))
        next_body = transport_flux_body(candidate)
        removable_action = float(np.mean(
            next_body.removable_amplitude * next_body.removable_amplitude))
        supported_structure_action = float(np.mean(
            next_body.coherence * next_body.predictive_scale**2))
        history.append({
            "cycle": cycle + 1,
            "update_rms": update_rms,
            "next_removable_action": removable_action,
            "next_supported_structure_action": supported_structure_action,
            "inner": last,
        })
        current = candidate
        if removable_action <= supported_structure_action + numerical:
            phase_balanced = True
            break
        body = next_body

    residual = image - current
    diagnostic = {
        **last,
        "method": "transport-admissible Chambolle",
        "status": (
            "transport identity"
            if last.get("status") == "transport identity"
            else (
                "transport noise/structure phase balance"
                if phase_balanced else "observer-cycle numerical ceiling"
            )
        ),
        "observer_cycles": len(history),
        "observer_history": history,
        "maximum_observer_cycles": ceiling,
        "residual": residual,
        "recomposition_error": float(np.max(np.abs(
            current + residual - image))),
        "outer_stop": (
            "next removable action <= next chart-supported structure action"
        ),
    }
    return (current, diagnostic) if return_diagnostics else current


__all__ = [
    "TransportChambolleResolution",
    "TransportFluxBody",
    "denoise_transport_chambolle",
    "transport_flux_body",
    "transport_flux_bodies",
]
