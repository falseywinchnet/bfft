"""Invariant geometry kernels for the post-FMMT denoising theory.

This is not a complete denoiser.  It implements the part of the fused system
that can already be stated without a support threshold, smoothing pilot, named
noise law, or tuned metric gain:

* the observation is represented by one joint signal/residual measure on the
  exact graph ``y = z + r``;
* support precision can be pulled back from a *transported predictive* signal
  measure, not a structure tensor of raw noisy samples;
* the determinant of that precision is the population volume form; and
* its determinant-normalized tensor is the eikonal metric.

The predictive qualification is essential.  Applying this geometry to raw
one-hot observations makes independent noise look like information.  A future
solver must first transport the joint measure to causal predictive equilibrium,
then call these kernels inside the fixed-point loop.
"""

from __future__ import annotations

from typing import Any
import math

import numpy as np


def _probability_field(probability: np.ndarray) -> np.ndarray:
    field = np.asarray(probability, dtype=np.float64)
    if field.ndim != 3 or min(field.shape[:2]) < 2 or field.shape[2] < 2:
        raise ValueError("predictive measure must have shape HxWxK with K >= 2")
    if not np.all(np.isfinite(field)) or np.any(field < 0.0):
        raise ValueError("predictive measure must be finite and nonnegative")
    mass = np.sum(field, axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("every spatial point needs positive predictive mass")
    return field / mass


def domain_unit_precision(shape: tuple[int, int]) -> tuple[float, float]:
    """Return the canonical topology floor and physical pixel area.

    Pixel coordinates use the longest image side as one physical unit.  The
    constant isotropic precision is normalized so its Riemannian volume over
    the whole rectangular domain is exactly one support unit.  It is therefore
    a topology normalization, not a tuned minimum scale.
    """
    height, width = (int(shape[0]), int(shape[1]))
    if height < 2 or width < 2:
        raise ValueError("the spatial domain must be at least 2x2")
    longest = float(max(height, width))
    pixel_area = 1.0 / (longest * longest)
    domain_area = height * width * pixel_area
    return math.pi / domain_area, pixel_area


def _geometry_from_pullback(
    pullback_xx: np.ndarray,
    pullback_xy: np.ndarray,
    pullback_yy: np.ndarray,
    *,
    geometry_name: str,
) -> dict[str, Any]:
    """Add the canonical domain floor and split volume from anisotropy."""
    fxx = np.asarray(pullback_xx, dtype=np.float64)
    fxy = np.asarray(pullback_xy, dtype=np.float64)
    fyy = np.asarray(pullback_yy, dtype=np.float64)
    if fxx.ndim != 2 or fxy.shape != fxx.shape or fyy.shape != fxx.shape:
        raise ValueError("pullback tensor components must be aligned 2-D fields")
    if min(fxx.shape) < 2:
        raise ValueError("the spatial domain must be at least 2x2")
    if not all(np.all(np.isfinite(field)) for field in (fxx, fxy, fyy)):
        raise ValueError("pullback tensor must be finite")

    base, pixel_area = domain_unit_precision(fxx.shape)
    qxx = base + fxx
    qxy = fxy
    qyy = base + fyy
    determinant = np.maximum(qxx * qyy - qxy * qxy, np.finfo(float).tiny)
    volume = np.sqrt(determinant)
    raw_measure = volume * pixel_area / math.pi
    implied_support = float(np.sum(raw_measure))
    measure = raw_measure / implied_support

    # The determinant-normalized metric carries anisotropy only.  Absolute
    # information volume already owns population and cannot be counted twice.
    mxx = qxx / volume
    mxy = qxy / volume
    myy = qyy / volume
    metric_determinant = mxx * myy - mxy * mxy
    return {
        "measure": np.ascontiguousarray(measure, dtype=np.float64),
        "implied_support": implied_support,
        "precision_xx": np.ascontiguousarray(qxx, dtype=np.float64),
        "precision_xy": np.ascontiguousarray(qxy, dtype=np.float64),
        "precision_yy": np.ascontiguousarray(qyy, dtype=np.float64),
        "metric_xx": np.ascontiguousarray(mxx, dtype=np.float64),
        "metric_xy": np.ascontiguousarray(mxy, dtype=np.float64),
        "metric_yy": np.ascontiguousarray(myy, dtype=np.float64),
        "metric_determinant": np.ascontiguousarray(
            metric_determinant, dtype=np.float64),
        "domain_base_precision": float(base),
        "pixel_area": float(pixel_area),
        "information_trace_mean": float(np.mean(fxx + fyy)),
        "pullback_geometry": geometry_name,
        "theory_status": "geometry kernel; predictive fixed-point solver pending",
    }


def combine_information_geometries(
    *geometries: dict[str, Any],
) -> dict[str, Any]:
    """Add predictive pullbacks while retaining one canonical domain metric."""
    if not geometries:
        raise ValueError("at least one information geometry is required")
    shape = np.asarray(geometries[0]["precision_xx"]).shape
    base = float(geometries[0]["domain_base_precision"])
    pullback_xx = np.zeros(shape, dtype=np.float64)
    pullback_xy = np.zeros(shape, dtype=np.float64)
    pullback_yy = np.zeros(shape, dtype=np.float64)
    names = []
    for geometry in geometries:
        if np.asarray(geometry["precision_xx"]).shape != shape:
            raise ValueError("information geometries must share one domain")
        local_base = float(geometry["domain_base_precision"])
        if local_base != base:
            raise ValueError("information geometries must share one domain base")
        pullback_xx += np.asarray(geometry["precision_xx"]) - base
        pullback_xy += np.asarray(geometry["precision_xy"])
        pullback_yy += np.asarray(geometry["precision_yy"]) - base
        names.append(str(geometry["pullback_geometry"]))
    result = _geometry_from_pullback(
        pullback_xx,
        pullback_xy,
        pullback_yy,
        geometry_name=" + ".join(names),
    )
    result["component_geometries"] = names
    return result


def observation_graph_measure(
    signal_probability: np.ndarray,
    observation: np.ndarray,
    signal_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Lift a signal marginal to the exact joint graph ``y = z + r``.

    No residual histogram or likelihood floor is introduced.  The residual
    coordinate is the deterministic push-forward ``r = y-z`` of each signal
    atom, and the same probability mass occupies both coordinates.
    """
    probability = _probability_field(signal_probability)
    y = np.asarray(observation, dtype=np.float64)
    values = np.asarray(signal_values, dtype=np.float64)
    if y.shape != probability.shape[:2]:
        raise ValueError("observation and predictive field must align")
    if values.ndim != 1 or values.size != probability.shape[2]:
        raise ValueError("signal_values must index the predictive atoms")
    signal = np.broadcast_to(values, probability.shape)
    residual = y[..., None] - signal
    return {
        "mass": probability,
        "signal": signal,
        "residual": residual,
    }


def predictive_information_geometry(
    signal_probability: np.ndarray,
) -> dict[str, Any]:
    """Extract population and eikonal geometry from a predictive measure.

    For the normalized signal marginal ``p_x(z)``, the information pullback is

        F_ab(x) = 4 integral d_a sqrt(p_x) d_b sqrt(p_x) dz.

    ``Q = G_domain + F`` is positive definite without an empirical scale floor.
    In two dimensions, ``M = Q / sqrt(det Q)`` has determinant one.  Therefore
    ``det Q`` decides *how much support exists*, while ``M`` decides *how it
    travels*; no metric-strength setting mixes those roles.
    """
    probability = _probability_field(signal_probability)
    height, width, _atoms = probability.shape
    longest = float(max(height, width))
    root = np.sqrt(probability)
    derivative_y, derivative_x = np.gradient(
        root,
        1.0 / longest,
        1.0 / longest,
        axis=(0, 1),
        edge_order=1,
    )
    fisher_xx = 4.0 * np.sum(derivative_x * derivative_x, axis=-1)
    fisher_xy = 4.0 * np.sum(derivative_x * derivative_y, axis=-1)
    fisher_yy = 4.0 * np.sum(derivative_y * derivative_y, axis=-1)

    return _geometry_from_pullback(
        fisher_xx,
        fisher_xy,
        fisher_yy,
        geometry_name="Fisher--Rao density pullback",
    )


def predictive_wasserstein_geometry(
    signal_particles: np.ndarray,
) -> dict[str, Any]:
    """Pull back scalar predictive laws without bins or bandwidths.

    ``signal_particles[y, x]`` is an equal-mass empirical measure on the
    scalar signal coordinate.  In one value dimension the exact quadratic
    Wasserstein coupling is the common quantile coordinate, so

        W_ab(x) = integral_0^1 d_a q_x(u) d_b q_x(u) du.

    Sorting implements that integral exactly for equal particles.  The result
    is invariant to particle order and to uniform replication of every atom;
    unlike a histogram Fisher tensor, it therefore has a meaningful atomic
    refinement limit.

    This kernel deliberately uses ordinary spatial derivatives.  It is a
    diagnostic and a building block, not the finished denoising support law.
    The fused solver must replace them by horizontal derivatives after value
    and jet particles have been parallel-transported to a common base point.
    Without that connection, an affine ramp and finite-sample noise both create
    false support.
    """
    particles = np.asarray(signal_particles, dtype=np.float64)
    if particles.ndim != 3 or min(particles.shape[:2]) < 2:
        raise ValueError("signal particles must have shape HxWxK")
    if particles.shape[2] < 2:
        raise ValueError("at least two signal particles are required")
    if not np.all(np.isfinite(particles)):
        raise ValueError("signal particles must be finite")

    quantiles = np.sort(particles, axis=-1)
    height, width, _atoms = quantiles.shape
    longest = float(max(height, width))
    derivative_y, derivative_x = np.gradient(
        quantiles,
        1.0 / longest,
        1.0 / longest,
        axis=(0, 1),
        edge_order=1,
    )
    wasserstein_xx = np.mean(derivative_x * derivative_x, axis=-1)
    wasserstein_xy = np.mean(derivative_x * derivative_y, axis=-1)
    wasserstein_yy = np.mean(derivative_y * derivative_y, axis=-1)
    return _geometry_from_pullback(
        wasserstein_xx,
        wasserstein_xy,
        wasserstein_yy,
        geometry_name="quadratic Wasserstein quantile pullback",
    )


def predictive_horizontal_wasserstein_geometry(
    signal_particles: np.ndarray,
) -> dict[str, Any]:
    """Quotient scalar predictive laws by their transported translation.

    In the scalar signal fibre, translation is the first jet connection. For
    quantile law ``q_x(u)``, its barycentric jet transports the common location
    ``m_x = integral q_x(u) du``. The horizontal quantile is

        q_x^H(u) = q_x(u) - m_x,

    and the pullback uses ``D_a q = d_a q - d_a m``. Consequently every
    spatially translating law—including an affine signal with a stationary
    residual shape—has exactly one domain support unit. Shape, spread, and
    higher-jet changes remain measurable.

    This is the exact translation quotient, not yet the full Sasaki connection
    for direction and curvature particles.
    """
    particles = np.asarray(signal_particles, dtype=np.float64)
    if particles.ndim != 3 or min(particles.shape[:2]) < 2:
        raise ValueError("signal particles must have shape HxWxK")
    if particles.shape[2] < 2 or not np.all(np.isfinite(particles)):
        raise ValueError("at least two finite signal particles are required")
    quantiles = np.sort(particles, axis=-1)
    barycenter = np.mean(quantiles, axis=-1, keepdims=True)
    horizontal = quantiles - barycenter
    height, width, _atoms = horizontal.shape
    longest = float(max(height, width))
    derivative_y, derivative_x = np.gradient(
        horizontal,
        1.0 / longest,
        1.0 / longest,
        axis=(0, 1),
        edge_order=1,
    )
    pullback_xx = np.mean(derivative_x * derivative_x, axis=-1)
    pullback_xy = np.mean(derivative_x * derivative_y, axis=-1)
    pullback_yy = np.mean(derivative_y * derivative_y, axis=-1)
    result = _geometry_from_pullback(
        pullback_xx,
        pullback_xy,
        pullback_yy,
        geometry_name="horizontal quadratic Wasserstein translation quotient",
    )
    mean_gradient_y, mean_gradient_x = np.gradient(
        barycenter[..., 0],
        1.0 / longest,
        1.0 / longest,
        edge_order=1,
    )
    result.update({
        "connection_gradient_x": np.ascontiguousarray(mean_gradient_x),
        "connection_gradient_y": np.ascontiguousarray(mean_gradient_y),
        "horizontal_quantile_mean_max": float(np.max(np.abs(
            np.mean(horizontal, axis=-1)))),
        "connection_status": (
            "exact scalar translation quotient; direction/curvature Sasaki "
            "transport pending"
        ),
    })
    return result


def weighted_empirical_quantiles(
    values: np.ndarray,
    mass: np.ndarray,
    quantile_count: int,
) -> np.ndarray:
    """Resolve spatially varying weighted atomic laws on common quantiles.

    The midpoint quantile grid is numerical measure resolution. No value bins,
    density kernel, or signal bandwidth enter the construction.
    """
    atoms = np.asarray(values, dtype=np.float64)
    probability = np.asarray(mass, dtype=np.float64)
    count = int(quantile_count)
    if atoms.ndim != 3 or probability.shape != atoms.shape:
        raise ValueError("values and mass must be aligned HxWxK fields")
    if min(atoms.shape[:2]) < 2 or atoms.shape[-1] < 2 or count < 2:
        raise ValueError("spatial laws and quantile resolution need size >= 2")
    if (
        not np.all(np.isfinite(atoms))
        or not np.all(np.isfinite(probability))
        or np.any(probability < 0.0)
    ):
        raise ValueError("weighted empirical laws must be finite and nonnegative")
    total = np.sum(probability, axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("every empirical law needs positive mass")
    probability = probability / total
    order = np.argsort(atoms, axis=-1, kind="stable")
    ordered_values = np.take_along_axis(atoms, order, axis=-1)
    ordered_mass = np.take_along_axis(probability, order, axis=-1)
    cumulative = np.cumsum(ordered_mass, axis=-1)
    cumulative[..., -1] = 1.0
    quantiles = np.empty(atoms.shape[:2] + (count,), dtype=np.float64)
    for index in range(count):
        level = (index + 0.5) / count
        selected = np.argmax(cumulative >= level, axis=-1)
        quantiles[..., index] = np.take_along_axis(
            ordered_values, selected[..., None], axis=-1)[..., 0]
    return quantiles


def weighted_support_quantiles(
    weights: np.ndarray,
    support_values: np.ndarray,
    quantile_count: int,
) -> np.ndarray:
    """Resolve spatial source-lineage laws on a shared scalar support."""
    mass = np.asarray(weights, dtype=np.float64)
    support = np.asarray(support_values, dtype=np.float64).reshape(-1)
    count = int(quantile_count)
    if mass.ndim != 3 or mass.shape[-1] != support.size:
        raise ValueError("weights must be HxWxK aligned with K support values")
    if min(mass.shape[:2]) < 2 or support.size < 1 or count < 2:
        raise ValueError(
            "source laws need support and quantile resolution must be >= 2")
    if (
        not np.all(np.isfinite(mass))
        or not np.all(np.isfinite(support))
        or np.any(mass < 0.0)
    ):
        raise ValueError("source-lineage laws must be finite and nonnegative")
    total = np.sum(mass, axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("every source-lineage law needs positive mass")
    mass = mass / total
    order = np.argsort(support, kind="stable")
    ordered_values = support[order]
    cumulative = np.cumsum(mass[..., order], axis=-1)
    cumulative[..., -1] = 1.0
    quantiles = np.empty(mass.shape[:2] + (count,), dtype=np.float64)
    for index in range(count):
        level = (index + 0.5) / count
        selected = np.argmax(cumulative >= level, axis=-1)
        quantiles[..., index] = ordered_values[selected]
    return quantiles


def predictive_lineage_jet_geometry(
    source_lineage: np.ndarray,
    source_gradient_x: np.ndarray,
    source_gradient_y: np.ndarray,
    *,
    quantile_count: int = 32,
) -> dict[str, Any]:
    """Measure vertical jet change after transport through source lineage.

    Each target receives a probability law over *source identities* before
    source jets are compared. Repeated characteristic views of one source are
    therefore quotiented before uncertainty or geometry is formed. This is a
    precursor to the same construction on a Hopf--Lax parent DAG.
    """
    lineage = np.asarray(source_lineage, dtype=np.float64)
    gradient_x = np.asarray(source_gradient_x, dtype=np.float64).reshape(-1)
    gradient_y = np.asarray(source_gradient_y, dtype=np.float64).reshape(-1)
    if lineage.ndim != 3 or min(lineage.shape[:2]) < 2:
        raise ValueError("source lineage must have shape HxWxK")
    height, width, sources = lineage.shape
    if gradient_x.size != sources or gradient_y.size != sources:
        raise ValueError("source jet components must index lineage sources")
    longest = float(max(height, width))
    jet_x = weighted_support_quantiles(
        lineage, longest * gradient_x, quantile_count)
    jet_y = weighted_support_quantiles(
        lineage, longest * gradient_y, quantile_count)
    dxx_y, dxx_x = np.gradient(
        jet_x, 1.0 / longest, 1.0 / longest,
        axis=(0, 1), edge_order=1)
    dyy_y, dyy_x = np.gradient(
        jet_y, 1.0 / longest, 1.0 / longest,
        axis=(0, 1), edge_order=1)
    pullback_xx = np.mean(dxx_x * dxx_x + dyy_x * dyy_x, axis=-1)
    pullback_xy = np.mean(dxx_x * dxx_y + dyy_x * dyy_y, axis=-1)
    pullback_yy = np.mean(dxx_y * dxx_y + dyy_y * dyy_y, axis=-1)
    result = _geometry_from_pullback(
        pullback_xx,
        pullback_xy,
        pullback_yy,
        geometry_name="source-lineage transported vertical jet pullback",
    )
    result.update({
        "quantile_count": int(quantile_count),
        "maximum_lineage_row_mass_error": float(np.max(np.abs(
            np.sum(lineage, axis=-1) - 1.0))),
        "connection_status": (
            "characteristic source lineage; Hopf--Lax parent transport pending"
        ),
    })
    return result


def predictive_lineage_prolongation_geometry(
    source_lineage: np.ndarray,
    source_signal: np.ndarray,
) -> dict[str, Any]:
    """Differentiate the scalar section only after identity transport.

    ``z(x) = integral z_i dA_x(i)`` is the barycentric section induced by the
    transported source law.  Its first prolongation has vertical differential
    equal to the physical Hessian of ``z``.  The resulting pullback is

        V_ab = sum_c (partial_a partial_c z)(partial_b partial_c z).

    Constant and affine sections therefore contribute no vertical volume,
    while curvature that survives source transport remains.  This is the
    order dual to :func:`predictive_lineage_jet_geometry`, which differentiates
    source charts first and transports their jets afterward.
    """
    lineage = np.asarray(source_lineage, dtype=np.float64)
    signal = np.asarray(source_signal, dtype=np.float64).reshape(-1)
    if lineage.ndim != 3 or min(lineage.shape[:2]) < 3:
        raise ValueError("source lineage must have shape HxWxK with H,W >= 3")
    height, width, sources = lineage.shape
    if signal.size != sources:
        raise ValueError("source signal must index lineage sources")
    if (
        not np.all(np.isfinite(lineage))
        or not np.all(np.isfinite(signal))
        or np.any(lineage < 0.0)
    ):
        raise ValueError("transported section must be finite and nonnegative")
    row_mass = np.sum(lineage, axis=-1, keepdims=True)
    if np.any(row_mass <= 0.0):
        raise ValueError("every transported section point needs source mass")
    normalized = lineage / row_mass
    section = normalized @ signal
    longest = float(max(height, width))
    spacing = 1.0 / longest
    gradient_y, gradient_x = np.gradient(
        section, spacing, spacing, edge_order=2)
    hessian_xy, hessian_xx = np.gradient(
        gradient_x, spacing, spacing, edge_order=2)
    hessian_yy, hessian_yx = np.gradient(
        gradient_y, spacing, spacing, edge_order=2)
    hessian_cross = 0.5 * (hessian_xy + hessian_yx)
    pullback_xx = hessian_xx * hessian_xx + hessian_cross * hessian_cross
    pullback_xy = hessian_cross * (hessian_xx + hessian_yy)
    pullback_yy = hessian_cross * hessian_cross + hessian_yy * hessian_yy
    result = _geometry_from_pullback(
        pullback_xx,
        pullback_xy,
        pullback_yy,
        geometry_name="post-lineage first-jet prolongation pullback",
    )
    result.update({
        "transported_section": np.ascontiguousarray(section),
        "transported_gradient_x": np.ascontiguousarray(gradient_x),
        "transported_gradient_y": np.ascontiguousarray(gradient_y),
        "transported_hessian_xx": np.ascontiguousarray(hessian_xx),
        "transported_hessian_xy": np.ascontiguousarray(hessian_cross),
        "transported_hessian_yy": np.ascontiguousarray(hessian_yy),
        "maximum_lineage_row_mass_error": float(np.max(np.abs(
            row_mass[..., 0] - 1.0))),
        "connection_status": "source identity transported before differentiation",
    })
    return result


def predictive_jet_horizontal_wasserstein_geometry(
    signal_particles: np.ndarray,
    signal_mass: np.ndarray,
    connection_gradient_x: np.ndarray,
    connection_gradient_y: np.ndarray,
    *,
    quantile_count: int = 32,
) -> dict[str, Any]:
    """Pull back a weighted signal law through its transported first jet.

    For weighted quantile ``q_x(u)`` and posterior first jet ``j(x)``, this
    evaluates ``D_x q = partial_x q - j_x`` and its y analogue. A law moving
    exactly as its jet predicts has no information volume beyond the canonical
    one-unit topology. Shape change, curvature, jet inconsistency, and
    interfaces remain visible.
    """
    quantiles = weighted_empirical_quantiles(
        signal_particles, signal_mass, quantile_count)
    gradient_x = np.asarray(connection_gradient_x, dtype=np.float64)
    gradient_y = np.asarray(connection_gradient_y, dtype=np.float64)
    if gradient_x.shape != quantiles.shape[:2] or gradient_y.shape != gradient_x.shape:
        raise ValueError("connection jet must align with the spatial law")
    if not np.all(np.isfinite(gradient_x)) or not np.all(np.isfinite(gradient_y)):
        raise ValueError("connection jet must be finite")
    height, width, _atoms = quantiles.shape
    longest = float(max(height, width))
    derivative_y, derivative_x = np.gradient(
        quantiles,
        1.0 / longest,
        1.0 / longest,
        axis=(0, 1),
        edge_order=1,
    )
    horizontal_x = derivative_x - longest * gradient_x[..., None]
    horizontal_y = derivative_y - longest * gradient_y[..., None]
    pullback_xx = np.mean(horizontal_x * horizontal_x, axis=-1)
    pullback_xy = np.mean(horizontal_x * horizontal_y, axis=-1)
    pullback_yy = np.mean(horizontal_y * horizontal_y, axis=-1)
    result = _geometry_from_pullback(
        pullback_xx,
        pullback_xy,
        pullback_yy,
        geometry_name="jet-horizontal quadratic Wasserstein pullback",
    )
    result.update({
        "connection_gradient_x": np.ascontiguousarray(gradient_x),
        "connection_gradient_y": np.ascontiguousarray(gradient_y),
        "quantile_count": int(quantile_count),
        "connection_status": (
            "posterior first jet removes predicted signal translation; "
            "curvature/Sasaki jet transport remains in the pullback"
        ),
    })
    return result


def predictive_directional_jet_sasaki_geometry(
    signal_particles: np.ndarray,
    signal_mass: np.ndarray,
    directional_derivative: np.ndarray,
    tangent: np.ndarray,
    *,
    quantile_count: int = 32,
) -> dict[str, Any]:
    """Combine horizontal signal shape with vertical directional-jet change.

    The scalar signal law is quotiented by its Wasserstein translation. For
    each projective tangent, the associated directional-jet law is then
    resolved on the same numerical quantile grid and differentiated as a
    cotangent-fibre coordinate. Integrating those vertical pullbacks over the
    projective circle is the first executable Sasaki decomposition: affine
    translation vanishes, while curvature and changing continuation survive.
    """
    signal = np.asarray(signal_particles, dtype=np.float64)
    probability = np.asarray(signal_mass, dtype=np.float64)
    derivative = np.asarray(directional_derivative, dtype=np.float64)
    directions = np.asarray(tangent, dtype=np.float64)
    if signal.ndim != 3 or probability.shape != signal.shape:
        raise ValueError("signal and mass must be aligned HxWxK fields")
    if derivative.shape != signal.shape or directions.shape != (signal.shape[-1], 2):
        raise ValueError("directional jet state must align with signal particles")
    if not np.all(np.isfinite(derivative)) or not np.all(np.isfinite(directions)):
        raise ValueError("directional jet state must be finite")
    length = np.linalg.norm(directions, axis=1)
    if not np.allclose(length, 1.0, atol=8e-15, rtol=0.0):
        raise ValueError("tangent directions must have unit length")

    signal_quantiles = weighted_empirical_quantiles(
        signal, probability, quantile_count)
    centered_signal = signal_quantiles - np.mean(
        signal_quantiles, axis=-1, keepdims=True)
    height, width, _atoms = signal.shape
    longest = float(max(height, width))
    signal_y, signal_x = np.gradient(
        centered_signal,
        1.0 / longest,
        1.0 / longest,
        axis=(0, 1),
        edge_order=1,
    )
    pullback_xx = np.mean(signal_x * signal_x, axis=-1)
    pullback_xy = np.mean(signal_x * signal_y, axis=-1)
    pullback_yy = np.mean(signal_y * signal_y, axis=-1)

    unique, inverse = np.unique(directions, axis=0, return_inverse=True)
    vertical_xx = np.zeros((height, width), dtype=np.float64)
    vertical_xy = np.zeros((height, width), dtype=np.float64)
    vertical_yy = np.zeros((height, width), dtype=np.float64)
    represented_directions = np.zeros((height, width), dtype=np.float64)

    def masked_gradient(
        field: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Differentiate only across present conditional directional laws."""
        out_y = np.zeros_like(field)
        out_x = np.zeros_like(field)
        for row in range(height):
            for column in range(width):
                if not valid[row, column]:
                    continue
                if (
                    column > 0 and column + 1 < width
                    and valid[row, column - 1] and valid[row, column + 1]
                ):
                    out_x[row, column] = 0.5 * longest * (
                        field[row, column + 1] - field[row, column - 1])
                elif column + 1 < width and valid[row, column + 1]:
                    out_x[row, column] = longest * (
                        field[row, column + 1] - field[row, column])
                elif column > 0 and valid[row, column - 1]:
                    out_x[row, column] = longest * (
                        field[row, column] - field[row, column - 1])
                if (
                    row > 0 and row + 1 < height
                    and valid[row - 1, column] and valid[row + 1, column]
                ):
                    out_y[row, column] = 0.5 * longest * (
                        field[row + 1, column] - field[row - 1, column])
                elif row + 1 < height and valid[row + 1, column]:
                    out_y[row, column] = longest * (
                        field[row + 1, column] - field[row, column])
                elif row > 0 and valid[row - 1, column]:
                    out_y[row, column] = longest * (
                        field[row, column] - field[row - 1, column])
        return out_y, out_x

    for direction_index in range(len(unique)):
        member = inverse == direction_index
        directional_mass = probability[..., member]
        represented = np.sum(directional_mass, axis=-1) > 0.0
        safe_mass = directional_mass.copy()
        safe_mass[~represented] = 1.0
        jet_quantiles = weighted_empirical_quantiles(
            longest * derivative[..., member],
            safe_mass,
            quantile_count,
        )
        jet_y, jet_x = masked_gradient(jet_quantiles, represented)
        vertical_xx += np.mean(jet_x * jet_x, axis=-1)
        vertical_xy += np.mean(jet_x * jet_y, axis=-1)
        vertical_yy += np.mean(jet_y * jet_y, axis=-1)
        represented_directions += represented
    if np.any(represented_directions <= 0.0):
        raise RuntimeError("directional jet law left a spatial point unsupported")
    vertical_xx /= represented_directions
    vertical_xy /= represented_directions
    vertical_yy /= represented_directions
    result = _geometry_from_pullback(
        pullback_xx + vertical_xx,
        pullback_xy + vertical_xy,
        pullback_yy + vertical_yy,
        geometry_name=(
            "horizontal signal plus vertical directional-jet Sasaki pullback"
        ),
    )
    result.update({
        "quantile_count": int(quantile_count),
        "projective_tangent_count": int(len(unique)),
        "minimum_represented_tangent_count": int(np.min(
            represented_directions)),
        "horizontal_signal_trace_mean": float(np.mean(
            pullback_xx + pullback_yy)),
        "vertical_jet_trace_mean": float(np.mean(
            vertical_xx + vertical_yy)),
        "connection_status": (
            "Wasserstein translation quotient with projective directional-jet "
            "vertical transport; full causal curvature connection pending"
        ),
    })
    return result


def extract_causal_support(
    signal_probability: np.ndarray,
    *,
    memory_ceiling: int,
) -> dict[str, Any]:
    """Quantize smooth-density Fisher volume and solve V3 first arrival.

    This retained adapter is the executable bridge from the smooth-density
    diagnostic to the segmenter's intrinsic transport core.  The final fused
    solver must feed the analogous horizontal Wasserstein geometry after joint
    particle transport; this function does not manufacture that predictive
    state.
    ``memory_ceiling`` is an allocation guard only: when it is not hit, no
    image-content setting remains between the predictive law and its support.
    The imports are lazy so the geometry invariants remain usable without a
    compiled BFFT vision library.
    """
    ceiling = int(memory_ceiling)
    if ceiling < 1:
        raise ValueError("memory_ceiling must be positive")
    from port_needed.continuous_eikonal_transport import (
        continuous_first_partition_prepared,
        prepare_continuous_metric,
    )
    from port_needed.density_population import emit_density_population

    information = predictive_information_geometry(signal_probability)
    population_geometry = {
        "measure": information["measure"],
        "implied_cells": information["implied_support"],
    }
    centers, population = emit_density_population(
        population_geometry, safety_cells=ceiling)
    prepared = prepare_continuous_metric(
        information["metric_xx"],
        information["metric_xy"],
        information["metric_yy"],
    )
    forest = continuous_first_partition_prepared(
        centers, prepared, source_gradients=True)
    return {
        "information_geometry": information,
        "centers": centers,
        "population": population,
        "prepared_metric": prepared,
        "forest": forest,
    }
