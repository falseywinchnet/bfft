"""Exact ancestry transport on a continuous-eikonal parent simplex.

The V3 first-arrival solver records one accepted parent or a barycentric pair
for every accepted point. This module transports source identity through that
directed acyclic graph. It deliberately does not infer a population from
conductance magnitudes: the participation field is computed from the complete
transported ancestry measure.

This is an executable geometry kernel for the next fused experiment, not a
denoising rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CausalAncestry:
    """Transported source law and its exact collision participation."""

    weights: np.ndarray
    collision_population: np.ndarray
    entropy_population: np.ndarray
    source_count: int


def shared_label_causal_forest(
    root_pixel: np.ndarray,
    prepared_metric: dict[str, np.ndarray],
    *,
    memory_ceiling_bytes: int | None = None,
) -> tuple[dict[str, Any], CausalAncestry]:
    """March distinct roots under one label, then transport their identities.

    Hard source ownership is deliberately absent: every root carries transport
    label zero, so the continuous Hopf--Lax simplex may combine fronts from
    distinct roots. Root identity remains a separate coordinate and is pushed
    through the resulting causal DAG by :func:`transport_causal_ancestry`.

    Roots are exact raster points with zero arrival value. The optional memory
    ceiling only guards the dense experimental ancestry representation; it
    does not alter a run that fits.
    """
    roots = np.asarray(root_pixel, dtype=np.int64).reshape(-1)
    mxx = np.asarray(prepared_metric["mxx"], dtype=np.float64)
    if mxx.ndim != 2:
        raise ValueError("prepared metric must describe a 2-D domain")
    height, width = mxx.shape
    pixels = height * width
    if roots.size == 0 or np.any(roots < 0) or np.any(roots >= pixels):
        raise ValueError("causal roots must be nonempty in-domain pixels")
    if np.unique(roots).size != roots.size:
        raise ValueError("causal root pixels must be distinct")
    required_bytes = pixels * roots.size * np.dtype(np.float64).itemsize
    if (
        memory_ceiling_bytes is not None
        and required_bytes > int(memory_ceiling_bytes)
    ):
        raise MemoryError(
            "dense causal ancestry needs "
            f"{required_bytes} bytes, above the numerical memory ceiling"
        )

    from bfft.vision import fast_march_first_label_native
    from port_needed.continuous_eikonal_transport import _fast_march_first_label

    seed_pixel = np.ascontiguousarray(roots, dtype=np.int32)
    seed_value = np.zeros(roots.size, dtype=np.float64)
    shared_label = np.zeros(roots.size, dtype=np.int32)
    seed_gradient = np.zeros(roots.size, dtype=np.float64)
    marched = fast_march_first_label_native(
        seed_pixel,
        seed_value,
        shared_label,
        seed_gradient,
        seed_gradient,
        prepared_metric,
        source_gradients=False,
    )
    if marched is None:
        marched = _fast_march_first_label(
            seed_pixel,
            seed_value,
            shared_label,
            seed_gradient,
            seed_gradient,
            prepared_metric["directions"],
            prepared_metric["direction_costs"],
            prepared_metric["direction_valid"],
            prepared_metric["cardinal_costs"],
            prepared_metric["inverse_offset"],
            prepared_metric["inverse_receiver"],
            prepared_metric["mxx"],
            prepared_metric["mxy"],
            prepared_metric["myy"],
        )
    (
        owner,
        distance,
        gradient_x,
        gradient_y,
        _source_gradient_x,
        _source_gradient_y,
        parent_first,
        parent_second,
        parent_fraction,
        acceptance_order,
        push_count,
        maximum_heap_size,
    ) = marched
    root_identity = np.full(pixels, -1, dtype=np.int64)
    root_identity[roots] = np.arange(roots.size, dtype=np.int64)
    ancestry = transport_causal_ancestry(
        parent_first.reshape(height, width),
        parent_second.reshape(height, width),
        parent_fraction.reshape(height, width),
        acceptance_order,
        root_identity.reshape(height, width),
    )
    forest = {
        "labels": owner.reshape(height, width),
        "distance": distance.reshape(height, width),
        "gradient_x": gradient_x.reshape(height, width),
        "gradient_y": gradient_y.reshape(height, width),
        "parent_first": parent_first.reshape(height, width),
        "parent_second": parent_second.reshape(height, width),
        "parent_fraction": parent_fraction.reshape(height, width),
        "acceptance_order": acceptance_order,
        "front_pushes": int(push_count),
        "front_maximum_heap": int(maximum_heap_size),
        "root_pixel": roots,
        "shared_transport_label": True,
        "dense_ancestry_bytes": int(required_bytes),
    }
    return forest, ancestry


def shared_label_continuous_causal_forest(
    centers: np.ndarray,
    prepared_metric: dict[str, np.ndarray],
    *,
    memory_ceiling_bytes: int | None = None,
) -> tuple[dict[str, Any], CausalAncestry]:
    """March subpixel germs under one label while conserving germ identity."""
    points = np.asarray(centers, dtype=np.float64)
    mxx = np.asarray(prepared_metric["mxx"], dtype=np.float64)
    if (
        points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1
        or not np.all(np.isfinite(points))
        or np.any(points < 0.0) or np.any(points > 1.0)
    ):
        raise ValueError("continuous centers must be a nonempty Nx2 unit-square array")
    if mxx.ndim != 2:
        raise ValueError("prepared metric must describe a 2-D domain")
    height, width = mxx.shape
    pixels = height * width
    required_bytes = pixels * len(points) * np.dtype(np.float64).itemsize
    if (
        memory_ceiling_bytes is not None
        and required_bytes > int(memory_ceiling_bytes)
    ):
        raise MemoryError(
            "dense continuous causal ancestry needs "
            f"{required_bytes} bytes, above the numerical memory ceiling"
        )

    center_x = np.clip(points[:, 0] * width - 0.5, 0.0, width - 1.0)
    center_y = np.clip(points[:, 1] * height - 0.5, 0.0, height - 1.0)
    x0 = np.floor(center_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y0 = np.floor(center_y).astype(np.int32)
    y1 = np.minimum(y0 + 1, height - 1)
    seed_x = np.column_stack((x0, x1, x0, x1)).ravel()
    seed_y = np.column_stack((y0, y0, y1, y1)).ravel()
    seed_identity = np.repeat(
        np.arange(len(points), dtype=np.int32), 4)
    dx = seed_x.astype(np.float64) - center_x[seed_identity]
    dy = seed_y.astype(np.float64) - center_y[seed_identity]
    a = np.asarray(prepared_metric["mxx"])[seed_y, seed_x]
    b = np.asarray(prepared_metric["mxy"])[seed_y, seed_x]
    c = np.asarray(prepared_metric["myy"])[seed_y, seed_x]
    seed_value = np.sqrt(np.maximum(
        a * dx * dx + 2.0 * b * dx * dy + c * dy * dy,
        0.0,
    ))
    nonzero = seed_value > 1.0e-15
    seed_gradient_x = np.zeros_like(seed_value)
    seed_gradient_y = np.zeros_like(seed_value)
    seed_gradient_x[nonzero] = (
        a[nonzero] * dx[nonzero] + b[nonzero] * dy[nonzero]
    ) / seed_value[nonzero]
    seed_gradient_y[nonzero] = (
        b[nonzero] * dx[nonzero] + c[nonzero] * dy[nonzero]
    ) / seed_value[nonzero]
    seed_pixel = seed_y * width + seed_x

    from bfft.vision import fast_march_first_label_native
    from port_needed.continuous_eikonal_transport import _fast_march_first_label

    shared_label = np.zeros(seed_pixel.size, dtype=np.int32)
    marched = fast_march_first_label_native(
        np.ascontiguousarray(seed_pixel, dtype=np.int32),
        np.ascontiguousarray(seed_value, dtype=np.float64),
        shared_label,
        np.ascontiguousarray(seed_gradient_x, dtype=np.float64),
        np.ascontiguousarray(seed_gradient_y, dtype=np.float64),
        prepared_metric,
        source_gradients=False,
    )
    if marched is None:
        marched = _fast_march_first_label(
            np.ascontiguousarray(seed_pixel, dtype=np.int32),
            np.ascontiguousarray(seed_value, dtype=np.float64),
            shared_label,
            np.ascontiguousarray(seed_gradient_x, dtype=np.float64),
            np.ascontiguousarray(seed_gradient_y, dtype=np.float64),
            prepared_metric["directions"],
            prepared_metric["direction_costs"],
            prepared_metric["direction_valid"],
            prepared_metric["cardinal_costs"],
            prepared_metric["inverse_offset"],
            prepared_metric["inverse_receiver"],
            prepared_metric["mxx"],
            prepared_metric["mxy"],
            prepared_metric["myy"],
        )
    (
        owner,
        distance,
        gradient_x,
        gradient_y,
        _source_gradient_x,
        _source_gradient_y,
        parent_first,
        parent_second,
        parent_fraction,
        acceptance_order,
        push_count,
        maximum_heap_size,
    ) = marched

    winning_value = np.full(pixels, np.inf, dtype=np.float64)
    winning_identity = np.full(pixels, -1, dtype=np.int64)
    for index in range(seed_pixel.size):
        pixel = int(seed_pixel[index])
        value = float(seed_value[index])
        identity = int(seed_identity[index])
        if (
            value < winning_value[pixel]
            or (
                value == winning_value[pixel]
                and (winning_identity[pixel] < 0 or identity < winning_identity[pixel])
            )
        ):
            winning_value[pixel] = value
            winning_identity[pixel] = identity
    root = np.flatnonzero(parent_first < 0)
    if np.any(winning_identity[root] < 0):
        raise RuntimeError("continuous march produced a parentless non-seed point")
    root_identity = np.full(pixels, -1, dtype=np.int64)
    root_identity[root] = winning_identity[root]
    represented = np.unique(root_identity[root])
    if represented.size != len(points):
        missing = np.setdiff1d(np.arange(len(points)), represented)
        raise RuntimeError(
            "continuous causal population lost germ identities "
            f"{missing.tolist()} during seed competition"
        )
    ancestry = transport_causal_ancestry(
        parent_first.reshape(height, width),
        parent_second.reshape(height, width),
        parent_fraction.reshape(height, width),
        acceptance_order,
        root_identity.reshape(height, width),
    )
    forest = {
        "labels": owner.reshape(height, width),
        "distance": distance.reshape(height, width),
        "gradient_x": gradient_x.reshape(height, width),
        "gradient_y": gradient_y.reshape(height, width),
        "parent_first": parent_first.reshape(height, width),
        "parent_second": parent_second.reshape(height, width),
        "parent_fraction": parent_fraction.reshape(height, width),
        "acceptance_order": acceptance_order,
        "front_pushes": int(push_count),
        "front_maximum_heap": int(maximum_heap_size),
        "root_pixel": root,
        "root_identity": root_identity.reshape(height, width),
        "centers": points,
        "shared_transport_label": True,
        "continuous_germs": True,
        "dense_ancestry_bytes": int(required_bytes),
    }
    return forest, ancestry


def transport_causal_ancestry(
    parent_first: np.ndarray,
    parent_second: np.ndarray,
    parent_fraction: np.ndarray,
    acceptance_order: np.ndarray,
    root_identity: np.ndarray,
) -> CausalAncestry:
    """Push root identities through accepted eikonal parent fractions.

    ``root_identity[p]`` is a nonnegative source identity for accepted roots
    and ``-1`` elsewhere. Multiple raster seeds belonging to one continuous
    source must share an identity; this prevents pixelization from creating
    fictitious population.

    For a simplex child ``x`` with parents ``p`` and ``q`` and stored fraction
    ``t``, the transported law is exactly

    ``A[x] = (1-t) A[p] + t A[q]``.

    The returned collision population ``1 / sum(A**2)`` is therefore computed
    from explicit source overlap. It is not the invalid effective-sample proxy
    obtained from normalized local conductances.
    """
    first_shape = np.asarray(parent_first).shape
    first = np.asarray(parent_first, dtype=np.int64).reshape(-1)
    second = np.asarray(parent_second, dtype=np.int64).reshape(-1)
    fraction = np.asarray(parent_fraction, dtype=np.float64).reshape(-1)
    roots = np.asarray(root_identity, dtype=np.int64).reshape(-1)
    order = np.asarray(acceptance_order, dtype=np.int64).reshape(-1)
    pixels = first.size
    if second.size != pixels or fraction.size != pixels or roots.size != pixels:
        raise ValueError("parent and root fields must have identical size")
    if order.size != pixels or np.unique(order).size != pixels:
        raise ValueError("acceptance order must contain every point exactly once")
    if np.any(order < 0) or np.any(order >= pixels):
        raise ValueError("acceptance order contains an out-of-domain point")
    if not np.all(np.isfinite(fraction)):
        raise ValueError("parent fractions must be finite")
    if np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ValueError("parent fractions must lie in [0,1]")
    if np.any(roots < -1):
        raise ValueError("root identities must be -1 or nonnegative")
    source_count = int(roots.max(initial=-1)) + 1
    if source_count == 0:
        raise ValueError("at least one causal root identity is required")

    ancestry = np.zeros((pixels, source_count), dtype=np.float64)
    accepted = np.zeros(pixels, dtype=bool)
    for child in order:
        parent_a = int(first[child])
        parent_b = int(second[child])
        identity = int(roots[child])
        if parent_a < 0:
            if parent_b >= 0 or identity < 0:
                raise ValueError("each parentless accepted point must be a root")
            ancestry[child, identity] = 1.0
        else:
            if identity >= 0:
                raise ValueError("a non-root point cannot inject new ancestry")
            if parent_a >= pixels or not accepted[parent_a]:
                raise ValueError("first parent must precede its child")
            if parent_b < 0:
                ancestry[child] = ancestry[parent_a]
            else:
                if parent_b >= pixels or not accepted[parent_b]:
                    raise ValueError("second parent must precede its child")
                t = float(fraction[child])
                ancestry[child] = (
                    (1.0 - t) * ancestry[parent_a]
                    + t * ancestry[parent_b]
                )
        accepted[child] = True

    mass = ancestry.sum(axis=1)
    if not np.allclose(mass, 1.0, rtol=8.0 * np.finfo(float).eps,
                       atol=8.0 * np.finfo(float).eps):
        raise RuntimeError("causal ancestry transport failed to conserve mass")
    collision = 1.0 / np.sum(ancestry * ancestry, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = np.exp(-np.sum(
            np.where(ancestry > 0.0, ancestry * np.log(ancestry), 0.0),
            axis=1,
        ))
    return CausalAncestry(
        weights=ancestry.reshape(first_shape + (source_count,)),
        collision_population=collision.reshape(first_shape),
        entropy_population=entropy.reshape(first_shape),
        source_count=source_count,
    )
