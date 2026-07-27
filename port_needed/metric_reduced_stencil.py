"""Local lattice reduction for a continuum anisotropic eikonal solve.

This is not a larger menu of hand-authored directions.  Every pixel derives
its own three-vector obtuse superbase from the local SPD transport metric.
The six signed vectors form a causal triangulation adapted to that metric.

The construction is the two-dimensional Lagrange/Gauss reduction used by
FM-LBR (Mirebeau, SIAM J. Numer. Anal. 2014).  A subsequent Hopf--Lax
simplex update may choose any barycentric point between adjacent vectors, so
these vectors bound continuous cones rather than quantizing path direction.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _identity(function):  # pragma: no cover
    return function


_compile = njit(cache=True) if njit is not None else _identity


@_compile
def _reduce_metric_field(
    mxx: np.ndarray,
    mxy: np.ndarray,
    myy: np.ndarray,
) -> np.ndarray:
    height, width = mxx.shape
    superbase = np.empty((height, width, 3, 2), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            a = mxx[y, x]
            b = mxy[y, x]
            c = myy[y, x]
            ux, uy = 1, 0
            vx, vy = 0, 1
            for _ in range(64):
                norm_u = (
                    a * ux * ux + 2.0 * b * ux * uy + c * uy * uy)
                norm_v = (
                    a * vx * vx + 2.0 * b * vx * vy + c * vy * vy)
                if norm_v + 1e-14 < norm_u:
                    temporary = ux
                    ux = vx
                    vx = temporary
                    temporary = uy
                    uy = vy
                    vy = temporary
                    norm_u, norm_v = norm_v, norm_u
                inner = (
                    a * ux * vx
                    + b * (ux * vy + uy * vx)
                    + c * uy * vy
                )
                quotient = int(np.rint(inner / max(norm_u, 1e-30)))
                if quotient == 0:
                    break
                vx -= quotient * ux
                vy -= quotient * uy

            inner = (
                a * ux * vx
                + b * (ux * vy + uy * vx)
                + c * uy * vy
            )
            if inner > 0.0:
                vx = -vx
                vy = -vy
            superbase[y, x, 0, 0] = -ux - vx
            superbase[y, x, 0, 1] = -uy - vy
            superbase[y, x, 1, 0] = ux
            superbase[y, x, 1, 1] = uy
            superbase[y, x, 2, 0] = vx
            superbase[y, x, 2, 1] = vy
    return superbase


def metric_reduced_superbase(
    mxx: np.ndarray,
    mxy: np.ndarray,
    myy: np.ndarray,
) -> np.ndarray:
    """Return ``int32[height,width,3,(dx,dy)]`` local superbases."""
    mxx = np.ascontiguousarray(mxx, dtype=np.float64)
    mxy = np.ascontiguousarray(mxy, dtype=np.float64)
    myy = np.ascontiguousarray(myy, dtype=np.float64)
    if mxx.shape != mxy.shape or mxx.shape != myy.shape:
        raise ValueError("metric fields must have identical shapes")
    determinant = mxx * myy - mxy * mxy
    if np.any(mxx <= 0.0) or np.any(determinant <= 0.0):
        raise ValueError("metric must be symmetric positive definite")
    return _reduce_metric_field(mxx, mxy, myy)


def validate_obtuse_superbase(
    superbase: np.ndarray,
    mxx: np.ndarray,
    mxy: np.ndarray,
    myy: np.ndarray,
) -> dict[str, float]:
    """Measure the defining lattice and metric conditions."""
    vectors = np.asarray(superbase, dtype=np.float64)
    total = np.sum(vectors, axis=2)
    determinant = (
        vectors[..., 0, 0] * vectors[..., 1, 1]
        - vectors[..., 0, 1] * vectors[..., 1, 0]
    )
    largest_inner = -np.inf
    for first, second in ((0, 1), (1, 2), (2, 0)):
        u = vectors[..., first, :]
        v = vectors[..., second, :]
        inner = (
            np.asarray(mxx) * u[..., 0] * v[..., 0]
            + np.asarray(mxy) * (
                u[..., 0] * v[..., 1] + u[..., 1] * v[..., 0])
            + np.asarray(myy) * u[..., 1] * v[..., 1]
        )
        largest_inner = max(largest_inner, float(np.max(inner)))
    reach = np.max(np.abs(vectors), axis=(2, 3))
    return {
        "maximum_sum_error": float(np.max(np.abs(total))),
        "maximum_unimodular_error": float(
            np.max(np.abs(np.abs(determinant) - 1.0))),
        "maximum_pair_inner_product": largest_inner,
        "reach_p50": float(np.percentile(reach, 50.0)),
        "reach_p90": float(np.percentile(reach, 90.0)),
        "reach_p99": float(np.percentile(reach, 99.0)),
        "reach_max": float(np.max(reach)),
    }
