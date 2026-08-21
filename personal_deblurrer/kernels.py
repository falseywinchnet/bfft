"""Positive, mass-preserving exposure-path kernels.

The normative state is a non-negative measure on image displacement.  A
kernel is therefore never an unconstrained signed deconvolution stencil: it
has unit mass, an explicit centroid, and a measured covariance.  Its Fourier
transform is the characteristic function of the exposure displacement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TransportKernel:
    """A finite positive exposure-transport measure."""

    name: str
    psf: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.psf, dtype=np.float64)
        if value.ndim != 2 or min(value.shape) < 1:
            raise ValueError("a transport PSF must be a non-empty 2-D array")
        if np.any(~np.isfinite(value)) or np.any(value < -1e-15):
            raise ValueError("a transport PSF must be finite and non-negative")
        total = float(np.sum(value))
        if total <= 0.0:
            raise ValueError("a transport PSF must carry positive mass")
        normalized = np.maximum(value, 0.0) / total
        object.__setattr__(self, "psf", np.ascontiguousarray(normalized))

    @property
    def mass(self) -> float:
        return float(np.sum(self.psf))

    @property
    def centroid(self) -> np.ndarray:
        yy, xx = np.mgrid[: self.psf.shape[0], : self.psf.shape[1]]
        center = 0.5 * (np.asarray(self.psf.shape, dtype=np.float64) - 1.0)
        return np.asarray((
            np.sum(self.psf * (xx - center[1])),
            np.sum(self.psf * (yy - center[0])),
        ))

    @property
    def covariance(self) -> np.ndarray:
        yy, xx = np.mgrid[: self.psf.shape[0], : self.psf.shape[1]]
        center = 0.5 * (np.asarray(self.psf.shape, dtype=np.float64) - 1.0)
        dx = xx - center[1] - self.centroid[0]
        dy = yy - center[0] - self.centroid[1]
        return np.asarray((
            (np.sum(self.psf * dx * dx), np.sum(self.psf * dx * dy)),
            (np.sum(self.psf * dx * dy), np.sum(self.psf * dy * dy)),
        ))

    def otf(self, shape: tuple[int, int]) -> np.ndarray:
        """Return the circular-convolution transfer function for ``shape``."""
        height, width = map(int, shape)
        kh, kw = self.psf.shape
        if kh > height or kw > width:
            raise ValueError("the PSF support must fit inside the image")
        padded = np.zeros((height, width), dtype=np.float64)
        padded[:kh, :kw] = self.psf
        padded = np.roll(padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
        return np.fft.fft2(padded)


def identity_kernel() -> TransportKernel:
    return TransportKernel("identity", np.ones((1, 1), dtype=np.float64))


def gaussian_kernel(sigma: float, truncate: float = 3.5) -> TransportKernel:
    sigma = float(sigma)
    if sigma <= 0.0:
        return identity_kernel()
    radius = max(1, int(math.ceil(truncate * sigma)))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    value = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return TransportKernel(f"gaussian_sigma_{sigma:g}", value)


def disk_kernel(radius: float, supersample: int = 8) -> TransportKernel:
    """Area-sampled circular aperture PSF."""
    radius = float(radius)
    if radius <= 0.0:
        return identity_kernel()
    half = max(1, int(math.ceil(radius + 0.5)))
    sub = max(int(supersample), 1)
    offsets = (np.arange(sub, dtype=np.float64) + 0.5) / sub - 0.5
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    value = np.zeros_like(xx, dtype=np.float64)
    for oy in offsets:
        for ox in offsets:
            value += ((xx + ox) ** 2 + (yy + oy) ** 2 <= radius * radius)
    return TransportKernel(f"disk_radius_{radius:g}", value)


def _deposit_bilinear(
    points_xy: np.ndarray,
    weights: np.ndarray,
    radius: int,
) -> np.ndarray:
    side = 2 * radius + 1
    out = np.zeros((side, side), dtype=np.float64)
    x = np.asarray(points_xy[:, 0], dtype=np.float64) + radius
    y = np.asarray(points_xy[:, 1], dtype=np.float64) + radius
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    for dx, dy, factor in (
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (1, 0, fx * (1.0 - fy)),
        (0, 1, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ):
        ix = x0 + dx
        iy = y0 + dy
        valid = (ix >= 0) & (ix < side) & (iy >= 0) & (iy < side)
        np.add.at(out, (iy[valid], ix[valid]), weights[valid] * factor[valid])
    return out


def path_kernel(
    points_xy: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    name: str = "path",
    recenter: bool = True,
) -> TransportKernel:
    """Rasterize a continuous exposure path as a positive measure.

    Points are displacements in ``(x, y)`` pixel coordinates. Bilinear
    deposition preserves their positive quadrature mass. Recentering removes
    the unobservable global registration gauge while retaining asymmetric
    path phase.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("an exposure path must be an Nx2 point array")
    if weights is None:
        mass = np.ones(len(points), dtype=np.float64)
    else:
        mass = np.asarray(weights, dtype=np.float64)
        if mass.shape != (len(points),) or np.any(mass < 0.0):
            raise ValueError("path weights must be one non-negative value per point")
    if float(np.sum(mass)) <= 0.0:
        raise ValueError("path weights must carry positive mass")
    mass = mass / np.sum(mass)
    if recenter:
        points = points - np.sum(points * mass[:, None], axis=0, keepdims=True)
    radius = max(1, int(math.ceil(float(np.max(np.abs(points))) + 1.0)))
    return TransportKernel(name, _deposit_bilinear(points, mass, radius))


def line_kernel(length: float, angle_degrees: float, samples: int = 257) -> TransportKernel:
    """Uniform exposure along a centered continuous line segment."""
    length = float(length)
    if length <= 1e-12:
        return identity_kernel()
    angle = math.radians(float(angle_degrees))
    t = np.linspace(-0.5, 0.5, max(int(samples), 2), dtype=np.float64)
    points = length * t[:, None] * np.asarray((math.cos(angle), math.sin(angle)))
    return path_kernel(
        points,
        name=f"line_length_{length:g}_angle_{float(angle_degrees) % 180.0:g}",
    )


def curved_path_kernel(
    length: float,
    angle_degrees: float,
    bend: float,
    samples: int = 257,
) -> TransportKernel:
    """Quadratic camera-shake path with a signed transverse bend."""
    angle = math.radians(float(angle_degrees))
    tangent = np.asarray((math.cos(angle), math.sin(angle)))
    normal = np.asarray((-math.sin(angle), math.cos(angle)))
    t = np.linspace(-0.5, 0.5, max(int(samples), 3), dtype=np.float64)
    points = (
        float(length) * t[:, None] * tangent
        + float(bend) * (t * t - np.mean(t * t))[:, None] * normal
    )
    return path_kernel(
        points,
        name=(f"curve_length_{float(length):g}_angle_"
              f"{float(angle_degrees) % 180.0:g}_bend_{float(bend):g}"),
    )


def translated_kernel(
    kernel: TransportKernel,
    shift_xy: tuple[float, float] | np.ndarray,
    *,
    name: str | None = None,
) -> TransportKernel:
    """Translate a positive exposure measure without changing its mixing law.

    This is the discrete form of the deterministic/mixing factorization.  The
    requested translation becomes the kernel centroid; its centered residual
    retains the original positive displacement distribution.
    """
    shift = np.asarray(shift_xy, dtype=np.float64)
    if shift.shape != (2,) or np.any(~np.isfinite(shift)):
        raise ValueError("shift_xy must contain finite x and y translations")
    yy, xx = np.mgrid[: kernel.psf.shape[0], : kernel.psf.shape[1]]
    center = 0.5 * (np.asarray(kernel.psf.shape, dtype=np.float64) - 1.0)
    points = np.column_stack((
        (xx - center[1]).ravel(),
        (yy - center[0]).ravel(),
    ))
    points += shift[None, :]
    return path_kernel(
        points,
        weights=kernel.psf.ravel(),
        name=(name or f"{kernel.name}_shift_{shift[0]:g}_{shift[1]:g}"),
        recenter=False,
    )


class CircularTransportPlan:
    """One immutable OTF shared by repeated circular forward/adjoint calls."""

    backend = "numpy_rfft2_batched_hwc"

    def __init__(
        self,
        kernel: TransportKernel,
        shape: tuple[int, int],
    ) -> None:
        self.kernel = kernel
        self.shape = (int(shape[0]), int(shape[1]))
        self.transfer = np.ascontiguousarray(kernel.otf(self.shape))
        half_width = self.shape[1] // 2 + 1
        self.half_transfer = np.ascontiguousarray(
            self.transfer[:, :half_width])
        self.adjoint_half_transfer = np.ascontiguousarray(
            np.conj(self.half_transfer))

    def _validate(self, image: np.ndarray) -> np.ndarray:
        value = np.asarray(image, dtype=np.float64)
        if value.ndim not in (2, 3) or value.shape[:2] != self.shape:
            raise ValueError("circular transport image shape does not match its plan")
        return value

    def _apply(self, image: np.ndarray, transfer: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        multiplier = transfer if value.ndim == 2 else transfer[..., None]
        return np.fft.irfft2(
            np.fft.rfft2(value, axes=(0, 1)) * multiplier,
            s=self.shape,
            axes=(0, 1),
        )

    def forward(self, image: np.ndarray) -> np.ndarray:
        return self._apply(image, self.half_transfer)

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        return self._apply(image, self.adjoint_half_transfer)


def apply_circular(image: np.ndarray, kernel: TransportKernel) -> np.ndarray:
    """Apply the kernel with periodic boundaries, channel by channel."""
    value = np.asarray(image, dtype=np.float64)
    if value.ndim not in (2, 3):
        raise ValueError("an image must be HxW or HxWxC")
    return CircularTransportPlan(kernel, value.shape[:2]).forward(value)


def adjoint_circular(image: np.ndarray, kernel: TransportKernel) -> np.ndarray:
    """Apply the exact adjoint exposure transport."""
    value = np.asarray(image, dtype=np.float64)
    if value.ndim not in (2, 3):
        raise ValueError("an image must be HxW or HxWxC")
    return CircularTransportPlan(kernel, value.shape[:2]).adjoint(value)
