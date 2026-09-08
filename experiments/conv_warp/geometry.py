"""Coordinate geometry for direct inverse image warps.

Every matrix in this module maps a target coordinate directly to the source
coordinate at which reconstruction is evaluated.  Consequently, no affine or
projective map is decomposed into shears and no intermediate raster exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ProjectiveMap:
    """A nonsingular target-to-source homography on normalized coordinates."""

    matrix: Array

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("a projective map requires a 3-by-3 matrix")
        if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-14:
            raise ValueError("projective matrix must be finite and nonsingular")
        object.__setattr__(self, "matrix", matrix / matrix[2, 2])

    @property
    def is_affine(self) -> bool:
        return bool(np.array_equal(self.matrix[2], np.array([0.0, 0.0, 1.0])))

    def map(self, x: Array, y: Array) -> tuple[Array, Array]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        h = self.matrix
        denominator = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        if np.any(np.abs(denominator) <= 64.0 * np.finfo(np.float64).eps):
            raise ValueError("the requested domain intersects the homography pole")
        return (
            (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / denominator,
            (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / denominator,
        )

    def jacobian(self, x: Array, y: Array) -> Array:
        """Return the exact target-to-source Jacobian at each query."""

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        px, py = self.map(x, y)
        h = self.matrix
        denominator = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        output = np.empty(x.shape + (2, 2), dtype=np.float64)
        output[..., 0, 0] = (h[0, 0] - px * h[2, 0]) / denominator
        output[..., 0, 1] = (h[0, 1] - px * h[2, 1]) / denominator
        output[..., 1, 0] = (h[1, 0] - py * h[2, 0]) / denominator
        output[..., 1, 1] = (h[1, 1] - py * h[2, 1]) / denominator
        return output

    def affine_parts(self) -> tuple[Array, Array]:
        if not self.is_affine:
            raise ValueError("the map is projective, not affine")
        return self.matrix[:2, :2].copy(), self.matrix[:2, 2].copy()


def affine_about_center(
    *,
    angle_degrees: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    shear_x: float = 0.0,
    shear_y: float = 0.0,
    translation: tuple[float, float] = (0.0, 0.0),
) -> ProjectiveMap:
    """Construct one direct affine inverse map about the unit-square centre."""

    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError("affine scales must be positive")
    angle = math.radians(float(angle_degrees))
    rotation = np.array(
        ((math.cos(angle), -math.sin(angle)),
         (math.sin(angle), math.cos(angle))),
        dtype=np.float64,
    )
    shear = np.array(((1.0, shear_x), (shear_y, 1.0)), dtype=np.float64)
    scale = np.diag((float(scale_x), float(scale_y)))
    linear = rotation @ shear @ scale
    centre = np.array((0.5, 0.5), dtype=np.float64)
    offset = centre + np.asarray(translation, dtype=np.float64) - linear @ centre
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = linear
    matrix[:2, 2] = offset
    return ProjectiveMap(matrix)


def perspective_about_center(
    *,
    perspective_x: float,
    perspective_y: float,
    angle_degrees: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    shear_x: float = 0.0,
    shear_y: float = 0.0,
) -> ProjectiveMap:
    """Construct ``p = 1/2 + A(q-1/2)/(1+g.(q-1/2))`` exactly."""

    affine = affine_about_center(
        angle_degrees=angle_degrees,
        scale_x=scale_x,
        scale_y=scale_y,
        shear_x=shear_x,
        shear_y=shear_y,
    )
    linear, _ = affine.affine_parts()
    g, h = float(perspective_x), float(perspective_y)
    denominator_constant = 1.0 - 0.5 * (g + h)
    matrix = np.array((
        (linear[0, 0] + 0.5*g, linear[0, 1] + 0.5*h,
         0.5 - 0.5*(linear[0, 0] + linear[0, 1]) - 0.25*(g+h)),
        (linear[1, 0] + 0.5*g, linear[1, 1] + 0.5*h,
         0.5 - 0.5*(linear[1, 0] + linear[1, 1]) - 0.25*(g+h)),
        (g, h, denominator_constant),
    ), dtype=np.float64)
    return ProjectiveMap(matrix)


def pullback_spd(metric: Array, jacobian: Array) -> tuple[Array, Array]:
    """Pull back an SPD source metric and split scale from determinant-one shape.

    The first result is ``J^T M J``.  The second divides it by the positive
    square root of its determinant, and consequently has determinant one in
    two dimensions.  This combines anisotropy and warp geometry before any
    square root or directional support is constructed.
    """

    m = np.asarray(metric, dtype=np.float64)
    j = np.asarray(jacobian, dtype=np.float64)
    if m.shape[-2:] != (2, 2) or j.shape[-2:] != (2, 2):
        raise ValueError("metric and Jacobian must end in 2-by-2 axes")
    pulled = np.einsum("...ji,...jk,...kl->...il", j, m, j, optimize=True)
    determinant = np.linalg.det(pulled)
    if np.any(determinant <= 0.0):
        raise ValueError("pullback metric is not positive definite")
    shape = pulled / np.sqrt(determinant)[..., None, None]
    return pulled, shape
