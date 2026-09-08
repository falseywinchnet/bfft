"""Reference implementation of the reduced two-dimensional CONV equation.

This is an evidence oracle, not the fast path.  It evaluates one anisotropic
Eikonal action per source site, forms the paper's inverse-square partition,
and blends the two Cartesian CONV factor orders exactly as stated.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from backend import conv_resize  # noqa: E402
from experiments.conv_distilled_core import (  # noqa: E402
    nodal_current_geometry,
    reverse_conv_resize,
)
from port_needed.continuous_eikonal_transport import (  # noqa: E402
    continuous_first_partition_prepared,
    prepare_continuous_metric,
)


Array = np.ndarray


def _target_action(distance: Array, target: tuple[int, int]) -> Array:
    """Bilinearly evaluate one nodal action field on an endpoint lattice."""

    source_height, source_width = distance.shape
    y = np.linspace(0.0, source_height - 1.0, target[0])
    x = np.linspace(0.0, source_width - 1.0, target[1])
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    fy = y - y0
    fx = x - x0
    return (
        distance[y0[:, None], x0[None, :]]
        * (1.0 - fy)[:, None] * (1.0 - fx)[None, :]
        + distance[y0[:, None], x1[None, :]]
        * (1.0 - fy)[:, None] * fx[None, :]
        + distance[y1[:, None], x0[None, :]]
        * fy[:, None] * (1.0 - fx)[None, :]
        + distance[y1[:, None], x1[None, :]]
        * fy[:, None] * fx[None, :]
    )


def paper_metric_and_admission(values: Array) -> tuple[dict[str, Array], Array]:
    """Return the prepared determinant-one metric and nodal order coordinates."""

    field = np.asarray(values, dtype=np.float64)
    if field.ndim == 2:
        field = field[..., None]
    gxx, gxy, gyy, order_coordinate = nodal_current_geometry(field)
    trace = gxx + gyy
    inv_trace = np.divide(
        1.0, trace, out=np.zeros_like(trace), where=trace > 0.0
    )
    qxx = 1.0 + gxx * inv_trace
    qxy = gxy * inv_trace
    qyy = 1.0 + gyy * inv_trace
    determinant = qxx * qyy - qxy * qxy
    root = np.sqrt(determinant)
    mxx = qxx / root
    mxy = qxy / root
    myy = qyy / root

    return (
        prepare_continuous_metric(mxx, mxy, myy),
        np.clip(order_coordinate, 0.0, 1.0),
    )


def paper_action_admission(
    values: Array,
    target_shape: tuple[int, int],
) -> Array:
    """Evaluate beta=sum_a w_a eta_a from all source actions."""

    source = np.asarray(values, dtype=np.float64)
    height, width = source.shape[:2]
    target = tuple(map(int, target_shape))
    prepared, eta = paper_metric_and_admission(source)
    numerator = np.zeros(target, dtype=np.float64)
    denominator = np.zeros(target, dtype=np.float64)
    exact_value = np.zeros(target, dtype=np.float64)
    exact = np.zeros(target, dtype=bool)
    epsilon = 64.0 * np.finfo(np.float64).eps
    tiny = np.finfo(np.float64).tiny

    for source_y in range(height):
        for source_x in range(width):
            center = np.array([[
                (source_x + 0.5) / width,
                (source_y + 0.5) / height,
            ]], dtype=np.float64)
            arrival = continuous_first_partition_prepared(
                center, prepared, compact=True, source_gradients=False
            )
            action = _target_action(
                np.asarray(arrival["distance"], dtype=np.float64), target
            )
            zero = action <= epsilon
            new_exact = zero & ~exact
            exact_value[new_exact] = eta[source_y, source_x]
            exact |= zero
            inverse = 1.0 / np.maximum(action * action, tiny)
            denominator += np.where(zero, 0.0, inverse)
            numerator += np.where(
                zero, 0.0, inverse * eta[source_y, source_x]
            )
    beta = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0
    )
    beta[exact] = exact_value[exact]
    return np.clip(beta, 0.0, 1.0)


def eikonal_factor_blend_synthesis(
    values: Array,
    target_shape: tuple[int, int],
) -> Array:
    """Evaluate the paper's reduced Eikonal two-order reconstruction."""

    source = np.asarray(values, dtype=np.float32)
    target = tuple(map(int, target_shape))
    forward = np.asarray(conv_resize(source, target), dtype=np.float64)
    reverse = np.asarray(reverse_conv_resize(source, target), dtype=np.float64)
    beta = paper_action_admission(source, target)
    if source.ndim == 3:
        beta = beta[..., None]
    return np.asarray((1.0 - beta) * forward + beta * reverse, dtype=np.float32)
