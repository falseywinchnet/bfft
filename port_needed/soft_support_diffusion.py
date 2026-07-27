"""Owner-free soft support as transport-gated anisotropic heat flow.

The hard first-arrival labels are an initial condition, not permanent
ownership.  If ``1_Ci`` is the indicator of hard cell i, define

    w_i(t) = exp(t L_g) 1_Ci.

The edge-weighted diffusion generator ``L_g`` preserves constants, so
``sum_i w_i(t) = 1`` without constructing a dense pixels-by-sites array.
Any site colour or fitted field can therefore be diffused directly and is
exactly the corresponding partition-of-unity blend.

Conductance is the product of two already measured permissions:

* inverse BFFT transport action along the edge;
* unchanged-target colour agreement across the edge.

A real image boundary blocks the flow.  An unsupported cell boundary loses
identity continuously rather than being selected for deletion.
"""

from __future__ import annotations

import numpy as np

from bfft.effects import srgb_to_lab
from bfft.vision import soft_support_diffuse_native

from .wide_stencil_transport import _metric_fields


def _edge_conductance(
    target_lab: np.ndarray,
    metric: tuple[np.ndarray, np.ndarray, np.ndarray],
    source_slice: tuple[slice, slice],
    target_slice: tuple[slice, slice],
    dx: int,
    dy: int,
    colour_scale_squared: float,
) -> np.ndarray:
    mxx, mxy, myy = metric
    a = 0.5 * (mxx[source_slice] + mxx[target_slice])
    b = 0.5 * (mxy[source_slice] + mxy[target_slice])
    c = 0.5 * (myy[source_slice] + myy[target_slice])
    action_squared = np.maximum(
        dx * dx * a + 2.0 * dx * dy * b + dy * dy * c,
        1e-30,
    )
    difference = (
        target_lab[target_slice] - target_lab[source_slice])
    colour_energy = np.sum(difference * difference, axis=2)
    return (
        np.exp(-colour_energy / max(colour_scale_squared, 1e-30))
        / action_squared
    )


def build_soft_support_conductance(
    geometry: dict,
    target_rgb: np.ndarray,
    *,
    metric_strength: float = 1.5,
    colour_percentile: float = 60.0,
) -> dict:
    """Build four undirected edge families for a rotation-aware soft cover."""
    target_lab = np.asarray(srgb_to_lab(target_rgb), dtype=np.float64)
    horizontal_difference = (
        target_lab[:, 1:] - target_lab[:, :-1])
    vertical_difference = (
        target_lab[1:] - target_lab[:-1])
    horizontal_energy = np.sum(
        horizontal_difference * horizontal_difference, axis=2)
    vertical_energy = np.sum(
        vertical_difference * vertical_difference, axis=2)
    sample = np.concatenate((
        horizontal_energy.ravel(),
        vertical_energy.ravel(),
    ))
    colour_scale_squared = max(float(np.percentile(
        sample,
        np.clip(float(colour_percentile), 0.0, 100.0),
    )), 1e-8)
    metric = _metric_fields(geometry, metric_strength)
    return {
        "horizontal": _edge_conductance(
            target_lab, metric,
            (slice(None), slice(0, -1)),
            (slice(None), slice(1, None)),
            1, 0, colour_scale_squared,
        ),
        "vertical": _edge_conductance(
            target_lab, metric,
            (slice(0, -1), slice(None)),
            (slice(1, None), slice(None)),
            0, 1, colour_scale_squared,
        ),
        "diagonal_down_right": _edge_conductance(
            target_lab, metric,
            (slice(0, -1), slice(0, -1)),
            (slice(1, None), slice(1, None)),
            1, 1, colour_scale_squared,
        ),
        "diagonal_down_left": _edge_conductance(
            target_lab, metric,
            (slice(0, -1), slice(1, None)),
            (slice(1, None), slice(0, -1)),
            -1, 1, colour_scale_squared,
        ),
        "colour_scale_squared": colour_scale_squared,
        "colour_percentile": float(colour_percentile),
        "metric_strength": float(metric_strength),
    }


def diffuse_soft_support(
    field: np.ndarray,
    conductance: dict,
    *,
    passes: int,
    coupling: float = 0.8,
) -> np.ndarray:
    """Apply stable simultaneous heat steps to one or more fields.

    Each step is a convex local average, hence it preserves the convex hull,
    a constant field, and the sum-to-one invariant of implicit site weights.
    """
    native = soft_support_diffuse_native(
        field, conductance, passes, coupling)
    if native is not None:
        return native

    value = np.asarray(field, dtype=np.float64)
    scalar = value.ndim == 2
    if scalar:
        value = value[..., None]
    if value.ndim != 3:
        raise ValueError("soft-support field must have shape HxW or HxWxC")
    value = value.copy()
    amount = max(float(coupling), 0.0)
    if amount == 0.0:
        return value[..., 0] if scalar else value

    edge_families = (
        (
            conductance["horizontal"],
            (slice(None), slice(0, -1)),
            (slice(None), slice(1, None)),
        ),
        (
            conductance["vertical"],
            (slice(0, -1), slice(None)),
            (slice(1, None), slice(None)),
        ),
        (
            conductance["diagonal_down_right"],
            (slice(0, -1), slice(0, -1)),
            (slice(1, None), slice(1, None)),
        ),
        (
            conductance["diagonal_down_left"],
            (slice(0, -1), slice(1, None)),
            (slice(1, None), slice(0, -1)),
        ),
    )
    for _ in range(max(int(passes), 0)):
        numerator = value.copy()
        denominator = np.ones(value.shape[:2], dtype=np.float64)
        for weight, first, second in edge_families:
            exchange = amount * np.asarray(weight, dtype=np.float64)
            numerator[first] += exchange[..., None] * value[second]
            numerator[second] += exchange[..., None] * value[first]
            denominator[first] += exchange
            denominator[second] += exchange
        value = numerator / denominator[..., None]
    return value[..., 0] if scalar else value


def conductance_field(conductance: dict) -> np.ndarray:
    """Reduce edge permission to one per-pixel diagnostic."""
    first = np.zeros(
        (
            conductance["horizontal"].shape[0],
            conductance["horizontal"].shape[1] + 1,
        ),
        dtype=np.float64,
    )
    count = np.zeros_like(first)
    for weight, source, target in (
        (
            conductance["horizontal"],
            (slice(None), slice(0, -1)),
            (slice(None), slice(1, None)),
        ),
        (
            conductance["vertical"],
            (slice(0, -1), slice(None)),
            (slice(1, None), slice(None)),
        ),
        (
            conductance["diagonal_down_right"],
            (slice(0, -1), slice(0, -1)),
            (slice(1, None), slice(1, None)),
        ),
        (
            conductance["diagonal_down_left"],
            (slice(0, -1), slice(1, None)),
            (slice(1, None), slice(0, -1)),
        ),
    ):
        first[source] += weight
        first[target] += weight
        count[source] += 1.0
        count[target] += 1.0
    return first / np.maximum(count, 1.0)
