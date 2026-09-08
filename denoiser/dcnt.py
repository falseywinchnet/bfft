"""DCNT: descent-contracted nodal transport denoising.

DCNT is deliberately independent of FMMT and of the earlier endpoint-action
experiments.  Its observation operator is CONV*: four parity-shifted coarse
lattices are refined back to the image lattice.  At a pixel, the chart for
which that pixel was an anchor is discarded.  The remaining three values are
therefore target-excluded transport predictions.

Their median is the transported centre and their centred deviations are
generators of a one-dimensional zonotope.  The distance of the observation
from that interval is the *minimal* noise required by transport uncertainty.
The Meyer Split-Bregman/Hodge ROF descent then separates coherent unsupported
detail from the incoherent portion that may actually be removed.

The module also contains a literal NumPy reconstruction of scikit-image's
Chambolle TV iteration.  It is a comparison oracle, not DCNT's optimizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from experiments.convstar import convstar_resize


Array = np.ndarray
_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class DCNTResolution:
    """Numerical ceilings for the research realization.

    ``maximum_cycles`` is not a denoising strength: every cycle recomputes
    the target-excluded feasible set and the recurrence stops at binary64
    stationarity.  ``rof_sweeps`` and ``hodge_after`` resolve the fixed Meyer
    proximal problem.
    """

    maximum_cycles: int = 3
    rof_sweeps: int = 8
    hodge_after: int = 4


@dataclass(frozen=True)
class TransportUncertainty:
    centre: Array
    second_moment: Array
    convex_radius: Array
    zonotope_radius: Array
    family: Array


def _validate_image(image: Array) -> Array:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 10:
        raise ValueError("DCNT expects a finite HxW image with both sides >= 10")
    if not np.all(np.isfinite(value)):
        raise ValueError("DCNT requires finite image samples")
    return value


def tv_chambolle_reference(
    image: Array,
    weight: float = 0.1,
    eps: float = 2.0e-4,
    maximum_iterations: int = 200,
    *,
    return_diagnostics: bool = False,
) -> Array | tuple[Array, dict[str, Any]]:
    """Reproduce scikit-image's n-D Chambolle TV recurrence.

    The array equations, boundary closure, energy normalization, and stopping
    test match ``skimage.restoration._denoise_tv_chambolle_nd``.  Keeping the
    baseline here lets the DCNT probes compare algorithms without treating a
    library call as an unexplained primitive.
    """

    value = np.asarray(image)
    if not np.issubdtype(value.dtype, np.floating):
        value = value.astype(np.float64)
    if value.ndim < 1 or not np.all(np.isfinite(value)):
        raise ValueError("TV input must be a finite floating array")
    if float(weight) <= 0.0:
        raise ValueError("TV weight must be positive")
    ndim = value.ndim
    p = np.zeros((ndim,) + value.shape, dtype=value.dtype)
    gradient = np.zeros_like(p)
    divergence = np.zeros_like(value)
    iteration = 0
    energy_trace: list[float] = []
    output = value
    while iteration < int(maximum_iterations):
        if iteration > 0:
            divergence = -p.sum(0)
            slices_d = [slice(None)] * ndim
            slices_p = [slice(None)] * (ndim + 1)
            for axis in range(ndim):
                slices_d[axis] = slice(1, None)
                slices_p[axis + 1] = slice(0, -1)
                slices_p[0] = axis
                divergence[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[axis] = slice(None)
                slices_p[axis + 1] = slice(None)
            output = value + divergence
        else:
            output = value
        energy = float(np.sum(divergence * divergence))
        slices_g = [slice(None)] * (ndim + 1)
        for axis in range(ndim):
            slices_g[axis + 1] = slice(0, -1)
            slices_g[0] = axis
            gradient[tuple(slices_g)] = np.diff(output, axis=axis)
            slices_g[axis + 1] = slice(None)
        norm = np.sqrt(np.sum(gradient * gradient, axis=0))[None, ...]
        energy += float(weight) * float(np.sum(norm))
        tau = 1.0 / (2.0 * ndim)
        norm *= tau / float(weight)
        norm += 1.0
        p -= tau * gradient
        p /= norm
        energy /= float(value.size)
        energy_trace.append(energy)
        if iteration == 0:
            initial_energy = energy
            previous_energy = energy
        elif abs(previous_energy - energy) < float(eps) * initial_energy:
            break
        else:
            previous_energy = energy
        iteration += 1
    if not return_diagnostics:
        return output
    return output, {
        "method": "literal scikit Chambolle TV recurrence",
        "iterations": int(iteration + 1),
        "weight": float(weight),
        "energy_trace": energy_trace,
    }


def _target_excluded_conv_family_batch(images: Array) -> Array:
    """Batch independent images through the four exact CONV* charts.

    CONV* already treats trailing coordinates as independent components.
    Moving the image population there therefore shares each ordered spatial
    scan while preserving the scalar chart equations.  The result has shape
    ``Bx4xHxW``; only last-bit contraction order may differ.
    """

    value = np.asarray(images, dtype=np.float64)
    if value.ndim != 3 or min(value.shape[1:]) < 10:
        raise ValueError("batched DCNT input must have shape BxHxW")
    if not np.all(np.isfinite(value)):
        raise ValueError("DCNT requires finite image samples")
    padding = 4
    padded = np.pad(
        value, ((0, 0), (padding, padding), (padding, padding)),
        mode="reflect")
    component_field = np.moveaxis(padded, 0, -1)
    height, width = component_field.shape[:2]
    charts = []
    for offset_y in (0, 1):
        for offset_x in (0, 1):
            coarse = component_field[offset_y::2, offset_x::2]
            refined = convstar_resize(coarse, 2)
            chart = np.full(
                (height, width, value.shape[0]), np.nan, dtype=np.float64)
            y_stop = offset_y + refined.shape[0]
            x_stop = offset_x + refined.shape[1]
            chart[offset_y:y_stop, offset_x:x_stop] = refined
            chart[offset_y:y_stop:2, offset_x:x_stop:2] = np.nan
            charts.append(np.moveaxis(chart[
                padding:padding + value.shape[1],
                padding:padding + value.shape[2],
            ], -1, 0))
    family = np.stack(charts, axis=1)
    count = np.sum(np.isfinite(family), axis=1)
    if not np.all(count == 3):
        raise RuntimeError("parity closure failed to provide three witnesses")
    return family


def target_excluded_conv_family(image: Array) -> Array:
    """Return four parity-chart CONV* predictions, NaN at own anchors.

    Four reflected cells of padding are dictated by the compact two-jet
    closure and the twofold parity chart. Whole-sample reflection is required:
    unlike half-sample symmetry it does not duplicate a boundary target into
    the closure. After cropping, every target has exactly three finite
    predictions, none of which sampled that target.
    """

    value = _validate_image(image)
    return _target_excluded_conv_family_batch(value[None, ...])[0]


def _transport_uncertainty_batch(images: Array) -> TransportUncertainty:
    """Construct independent uncertainty laws in one component transport."""

    family = _target_excluded_conv_family_batch(images)
    # Every target has exactly three finite witnesses and one own-anchor NaN.
    # Replacing that NaN by +inf makes the second order statistic the exact
    # median of the three witnesses, without nanmedian's masked-array copies.
    centre = np.partition(
        np.where(np.isfinite(family), family, np.inf), 1, axis=1)[:, 1]
    generator = np.where(
        np.isfinite(family), family - centre[:, None, ...], 0.0)
    second_moment = np.sum(generator * generator, axis=1)
    return TransportUncertainty(
        centre=centre,
        second_moment=second_moment,
        convex_radius=np.max(np.abs(generator), axis=1),
        zonotope_radius=np.sum(np.abs(generator), axis=1),
        family=family,
    )


def transport_uncertainty(image: Array) -> TransportUncertainty:
    """Construct the target-excluded transport distribution and zonotope."""

    value = _validate_image(image)
    law = _transport_uncertainty_batch(value[None, ...])
    return TransportUncertainty(
        centre=law.centre[0],
        second_moment=law.second_moment[0],
        convex_radius=law.convex_radius[0],
        zonotope_radius=law.zonotope_radius[0],
        family=law.family[0],
    )


def minimal_supported_noise(
    observation: Array,
    centre: Array,
    radius: Array,
) -> tuple[Array, Array]:
    """Project onto a bounded transport law and return its minimal residual."""

    y = np.asarray(observation, dtype=np.float64)
    m = np.asarray(centre, dtype=np.float64)
    h = np.maximum(np.asarray(radius, dtype=np.float64), 0.0)
    if m.shape != y.shape or h.shape != y.shape:
        raise ValueError("transport law must align with the observation")
    admitted = np.clip(y, m - h, m + h)
    return y - admitted, admitted


def _meyer_rof(
    field: Array,
    resolution: DCNTResolution,
    *,
    scale: float | None = None,
) -> tuple[Array, dict[str, Any]]:
    """Run the optimized Meyer ROF descent with a portable reference fallback."""

    value = np.asarray(field, dtype=np.float64)
    rms_scale = (
        float(np.sqrt(np.mean(value * value)))
        if scale is None else max(float(scale), 0.0)
    )
    if rms_scale <= 64.0 * _EPS * max(float(np.max(np.abs(value))), 1.0):
        return value.copy(), {
            "backend": "zero fixed point",
            "rms_scale": rms_scale,
            "fidelity": None,
        }
    # TV(s z) + c/2 ||s(z-w)||^2 is dimensionless when c*s = 1.
    fidelity = 1.0 / rms_scale
    try:
        import bfft

        hodge_after = min(
            max(int(resolution.hodge_after), 0), int(resolution.rof_sweeps))
        output = bfft.rof(
            value,
            c=fidelity,
            eta=10.0 * fidelity,
            sweeps=int(resolution.rof_sweeps),
            tol=0.0,
            hodge_after=hodge_after,
        )
        backend = "native Meyer Split-Bregman/Hodge"
    except (ImportError, OSError, RuntimeError):
        from experiments.meyer_bregman import rof_sb

        output, _state = rof_sb(
            value,
            fidelity,
            eta=10.0 * fidelity,
            sweeps=int(resolution.rof_sweeps),
        )
        hodge_after = 0
        backend = "NumPy Meyer Split-Bregman fallback"
    return output, {
        "backend": backend,
        "rms_scale": rms_scale,
        "fidelity": fidelity,
        "sweeps": int(resolution.rof_sweeps),
        "hodge_after": int(hodge_after),
    }


def denoise_dcnt(
    observation: Array,
    *,
    mode: str = "uncertainty",
    resolution: DCNTResolution | None = None,
) -> tuple[Array, dict[str, Any]]:
    """Run the first DCNT recurrence.

    ``transport_descent`` implements the first proposed optional step: smooth
    the component admitted by transport, reintegrate the resistant component,
    and form fresh observer charts.

    ``uncertainty`` implements the second: the full zonotope is uncertainty
    about transport itself; its interval distance is the minimal forced noise.
    Meyer ROF retains the coherent part of that resistance and only its
    oscillatory complement is removed.
    """

    y = _validate_image(observation)
    if mode not in ("transport_descent", "uncertainty"):
        raise ValueError("DCNT mode must be 'transport_descent' or 'uncertainty'")
    config = resolution or DCNTResolution()
    if config.maximum_cycles < 1 or config.rof_sweeps < 1:
        raise ValueError("DCNT numerical ceilings must be positive")
    current = y.copy()
    history: list[dict[str, Any]] = []
    residual = np.zeros_like(y)
    data_scale = max(float(np.ptp(y)), float(np.max(np.abs(y))), 1.0)
    stationarity = 128.0 * _EPS * data_scale
    for cycle in range(int(config.maximum_cycles)):
        law = transport_uncertainty(current)
        radius = (
            law.zonotope_radius
            if mode == "uncertainty"
            else law.convex_radius
        )
        resistant, admitted = minimal_supported_noise(y, law.centre, radius)
        if mode == "transport_descent":
            resistant_scale = float(np.sqrt(np.mean(resistant * resistant)))
            smoothed, rof_diagnostic = _meyer_rof(
                admitted, config, scale=resistant_scale)
            candidate = smoothed + resistant
            removed = y - candidate
            coherent_resistance = resistant
        else:
            coherent_resistance, rof_diagnostic = _meyer_rof(
                resistant, config)
            removed = resistant - coherent_resistance
            candidate = y - removed
        update = candidate - current
        residual = y - candidate
        record = {
            "cycle": cycle + 1,
            "transport_second_moment": float(np.mean(law.second_moment)),
            "convex_radius_mean": float(np.mean(law.convex_radius)),
            "zonotope_radius_mean": float(np.mean(law.zonotope_radius)),
            "resistant_energy": float(np.mean(resistant * resistant)),
            "coherent_resistance_energy": float(np.mean(
                coherent_resistance * coherent_resistance)),
            "removed_energy": float(np.mean(removed * removed)),
            "update_rms": float(np.sqrt(np.mean(update * update))),
            "rof": rof_diagnostic,
        }
        history.append(record)
        current = candidate
        if record["update_rms"] <= stationarity:
            break
    diagnostic = {
        "method": "DCNT",
        "mode": mode,
        "resolution": asdict(config),
        "cycles": len(history),
        "history": history,
        "residual": residual,
        "recomposition_error": float(np.max(np.abs(current + residual - y))),
        "target_excluded_predictions_per_pixel": 3,
        "persistent_state_coordinates_per_pixel": 2,
        "transient_transport_witnesses_per_pixel": 3,
        "status": "foundational experiment; not a promoted denoiser",
    }
    return current, diagnostic
