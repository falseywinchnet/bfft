"""Hodge-lifted transport-uncertainty hypothesis for Chambolle denoising.

Pointwise disagreement is not uncertainty about *transport*.  This experiment
lifts the observation departure and every target-excluded CONV* generator
through one positive Selling-graph Hodge factorization.  The witness fluxes
generate a zonotope in the same nonlocal coordinate used to explain the
observation.  Flux outside that set is the minimal unsupported transport.

For graph incidence ``B`` and positive Selling conductance ``C``, the Hodge
lift of a zero-mean field ``f`` is

    L phi = f,       L = B^T C B,
    p_f = C B phi,   B^T p_f = f.

If ``p_a`` are the target-excluded witness-generator lifts, then

    H = sum_a |p_a|,
    p_N = p_y - clip(p_y, -H, H),
    r_N = B^T p_N.

The final Meyer proximal keeps coherent content of ``r_N`` and removes only
its oscillatory complement.  No support threshold, image scale, noise class,
or denoising weight is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized

try:
    from .continual_eikonal_noise_transport_2d import continual_transport_metric
    from .dcnt import DCNTResolution, _meyer_rof, _validate_image, transport_uncertainty
    from .selling_chambolle import _selling_conductance_graph
except ImportError:  # pragma: no cover
    from continual_eikonal_noise_transport_2d import continual_transport_metric
    from dcnt import DCNTResolution, _meyer_rof, _validate_image, transport_uncertainty
    from selling_chambolle import _selling_conductance_graph


Array = np.ndarray
_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class HodgeTransportState:
    centre: Array
    predictive_scale: Array
    first: Array
    second: Array
    conductance: Array
    departure_flux: Array
    generator_flux: Array
    flux_radius: Array
    excess_flux: Array
    unsupported: Array
    coherent_resistance: Array
    removable: Array
    component_count: int
    hodge_reconstruction_error: float
    selling_reconstruction_error: float


def _laplacian(
    pixels: int,
    first: Array,
    second: Array,
    conductance: Array,
) -> sparse.csc_matrix:
    diagonal = np.bincount(
        np.concatenate((first, second)),
        weights=np.concatenate((conductance, conductance)),
        minlength=pixels,
    )
    return sparse.coo_matrix((
        np.concatenate((diagonal, -conductance, -conductance)),
        (
            np.concatenate((np.arange(pixels), first, second)),
            np.concatenate((np.arange(pixels), second, first)),
        ),
    ), shape=(pixels, pixels)).tocsc()


def _hodge_lifts(
    fields: Array,
    first: Array,
    second: Array,
    conductance: Array,
) -> tuple[Array, float, int]:
    """Lift aligned scalar fields through one pinned graph factorization."""

    value = np.asarray(fields, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError("Hodge fields must have shape (lanes,H,W)")
    lanes, height, width = value.shape
    pixels = height * width
    laplacian = _laplacian(pixels, first, second, conductance)
    component_count, labels = connected_components(
        laplacian, directed=False, return_labels=True)
    component_nodes = [
        np.flatnonzero(labels == component)
        for component in range(component_count)
    ]
    solvers = [
        factorized(laplacian[nodes[1:], :][:, nodes[1:]])
        if nodes.size > 1 else None
        for nodes in component_nodes
    ]
    fluxes = np.empty((lanes, first.size), dtype=np.float64)
    maximum_error = 0.0
    for lane in range(lanes):
        flat = value[lane].ravel()
        potential = np.zeros(pixels, dtype=np.float64)
        rhs = np.zeros(pixels, dtype=np.float64)
        for nodes, solve in zip(component_nodes, solvers):
            rhs[nodes] = flat[nodes] - np.mean(flat[nodes])
            if solve is not None:
                potential[nodes[1:]] = solve(rhs[nodes[1:]])
        flux = conductance * (potential[second] - potential[first])
        fluxes[lane] = flux
        returned = (
            np.bincount(second, weights=flux, minlength=pixels)
            - np.bincount(first, weights=flux, minlength=pixels)
        )
        maximum_error = max(
            maximum_error, float(np.max(np.abs(returned - rhs))))
    return fluxes, maximum_error, int(component_count)


def hodge_transport_state(observation: Array) -> HodgeTransportState:
    """Construct the flux-space uncertainty zonotope and scalar posterior."""

    image = _validate_image(observation)
    law = transport_uncertainty(image)
    finite = np.isfinite(law.family)
    count = np.sum(finite, axis=0)
    predictive_scale = np.sqrt(law.second_moment / count)
    completed = np.where(finite, law.family, law.centre[None, ...])
    generators = completed - law.centre[None, ...]
    departure = image - law.centre

    metric = continual_transport_metric(
        law.centre, predictive_scale * predictive_scale)
    first, second, conductance, selling = _selling_conductance_graph(
        np.asarray(metric["metric_xx"]),
        np.asarray(metric["metric_xy"]),
        np.asarray(metric["metric_yy"]),
    )
    lifted, hodge_error, component_count = _hodge_lifts(
        np.concatenate((departure[None, ...], generators), axis=0),
        first, second, conductance,
    )
    departure_flux = lifted[0]
    generator_flux = lifted[1:]
    flux_radius = np.sum(np.abs(generator_flux), axis=0)
    admitted_flux = np.clip(departure_flux, -flux_radius, flux_radius)
    excess_flux = departure_flux - admitted_flux
    flat_unsupported = (
        np.bincount(second, weights=excess_flux, minlength=image.size)
        - np.bincount(first, weights=excess_flux, minlength=image.size)
    )
    unsupported = flat_unsupported.reshape(image.shape)
    coherent, _diagnostic = _meyer_rof(unsupported, DCNTResolution())
    removable = unsupported - coherent
    return HodgeTransportState(
        centre=np.ascontiguousarray(law.centre),
        predictive_scale=np.ascontiguousarray(predictive_scale),
        first=np.ascontiguousarray(first),
        second=np.ascontiguousarray(second),
        conductance=np.ascontiguousarray(conductance),
        departure_flux=np.ascontiguousarray(departure_flux),
        generator_flux=np.ascontiguousarray(generator_flux),
        flux_radius=np.ascontiguousarray(flux_radius),
        excess_flux=np.ascontiguousarray(excess_flux),
        unsupported=np.ascontiguousarray(unsupported),
        coherent_resistance=np.ascontiguousarray(coherent),
        removable=np.ascontiguousarray(removable),
        component_count=component_count,
        hodge_reconstruction_error=hodge_error,
        selling_reconstruction_error=float(
            selling["selling_reconstruction_error"]),
    )


def denoise_hodge_zonotope_chambolle(
    observation: Array,
    *,
    maximum_observer_cycles: int = 8,
    return_diagnostics: bool = True,
) -> tuple[Array, dict[str, Any]] | Array:
    """Re-observe until flux-excess action falls below witness-flux action."""

    image = _validate_image(observation)
    if int(maximum_observer_cycles) < 1:
        raise ValueError("maximum_observer_cycles must be positive")
    current = image.copy()
    history: list[dict[str, Any]] = []
    balanced = False
    last: HodgeTransportState | None = None
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)), 1.0)
    numerical = 256.0 * _EPS * magnitude * magnitude
    for cycle in range(int(maximum_observer_cycles)):
        state = hodge_transport_state(current)
        candidate = current - state.removable
        next_state = hodge_transport_state(candidate)
        next_excess_action = float(np.mean(next_state.excess_flux**2))
        next_witness_action = float(np.mean(next_state.generator_flux**2))
        history.append({
            "cycle": cycle + 1,
            "update_rms": float(np.sqrt(np.mean((candidate - current) ** 2))),
            "unsupported_action": float(np.mean(state.unsupported**2)),
            "coherent_resistance_action": float(np.mean(
                state.coherent_resistance**2)),
            "removed_action": float(np.mean(state.removable**2)),
            "next_excess_flux_action": next_excess_action,
            "next_mean_witness_flux_action": next_witness_action,
            "hodge_reconstruction_error": state.hodge_reconstruction_error,
            "hodge_component_count": state.component_count,
            "selling_reconstruction_error": state.selling_reconstruction_error,
        })
        current = candidate
        last = state
        if next_excess_action <= next_witness_action + numerical:
            balanced = True
            break
    assert last is not None
    residual = image - current
    diagnostic = {
        "method": "Hodge-zonotope transport Chambolle",
        "status": (
            "flux uncertainty phase balance"
            if balanced else "observer-cycle numerical ceiling"
        ),
        "observer_cycles": len(history),
        "observer_history": history,
        "selling_edge_count": int(last.first.size),
        "witness_flux_generators": int(last.generator_flux.shape[0]),
        "mean_flux_radius": float(np.mean(last.flux_radius)),
        "mean_excess_flux": float(np.mean(np.abs(last.excess_flux))),
        "residual": residual,
        "recomposition_error": float(np.max(np.abs(
            current + residual - image))),
        "coordinate": "minimum-action Selling Hodge flux zonotope",
    }
    return (current, diagnostic) if return_diagnostics else current


__all__ = [
    "HodgeTransportState",
    "denoise_hodge_zonotope_chambolle",
    "hodge_transport_state",
]
