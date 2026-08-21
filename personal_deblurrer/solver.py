"""Warm one-sweep descent for multi-observation exposure transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circles import coverage_report
from .kernels import TransportKernel, apply_circular


@dataclass(frozen=True)
class DeblurResult:
    image: np.ndarray
    diagnostics: dict[str, object]


def _gradient(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.roll(value, -1, axis=1) - value,
        np.roll(value, -1, axis=0) - value,
    )


def _gradient_adjoint(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return (
        np.roll(px, 1, axis=1) - px
        + np.roll(py, 1, axis=0) - py
    )


def _positive_laplacian_symbol(shape: tuple[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    return (
        4.0 * np.sin(np.pi * fy[:, None]) ** 2
        + 4.0 * np.sin(np.pi * fx[None, :]) ** 2
    )


def _validate_inputs(
    observations: list[np.ndarray],
    kernels: list[TransportKernel],
    precisions: np.ndarray | None,
) -> tuple[list[np.ndarray], np.ndarray]:
    if not observations or len(observations) != len(kernels):
        raise ValueError("one transport kernel is required per observation")
    values = [np.asarray(value, dtype=np.float64) for value in observations]
    if values[0].ndim not in (2, 3):
        raise ValueError("observations must be HxW or HxWxC")
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("all observations must be registered on one raster")
    weight = (
        np.ones(len(values), dtype=np.float64)
        if precisions is None
        else np.asarray(precisions, dtype=np.float64)
    )
    if weight.shape != (len(values),) or np.any(weight <= 0.0):
        raise ValueError("one positive precision is required per observation")
    return values, weight


def multi_wiener(
    observations: list[np.ndarray],
    kernels: list[TransportKernel],
    *,
    precisions: np.ndarray | None = None,
    regularization: float = 1e-3,
) -> DeblurResult:
    """Closed-form multi-observation Tikhonov baseline."""
    values, weight = _validate_inputs(observations, kernels, precisions)
    shape = values[0].shape[:2]
    transfers = [kernel.otf(shape) for kernel in kernels]
    channels = 1 if values[0].ndim == 2 else values[0].shape[2]
    denominator = np.full(shape, max(float(regularization), 0.0), dtype=np.float64)
    for precision, transfer in zip(weight, transfers):
        denominator += precision * np.abs(transfer) ** 2
    output = []
    for channel in range(channels):
        numerator = np.zeros(shape, dtype=np.complex128)
        for precision, transfer, observation in zip(weight, transfers, values):
            plane = observation if channels == 1 else observation[..., channel]
            numerator += precision * np.conj(transfer) * np.fft.fft2(plane)
        output.append(np.fft.ifft2(numerator / np.maximum(denominator, 1e-15)).real)
    image = output[0] if channels == 1 else np.stack(output, axis=2)
    return DeblurResult(
        image=np.clip(image, 0.0, 1.0),
        diagnostics={
            "method": "multi_wiener",
            "passes": 1,
            "regularization": float(regularization),
            "coverage": coverage_report(transfers, weight),
        },
    )


def fuse_transport_observations(
    observations: list[np.ndarray],
    kernels: list[TransportKernel],
    *,
    precisions: np.ndarray | None = None,
    tv_weight: float = 0.003,
    flux_penalty: float = 0.04,
    passes: int = 24,
    clip: tuple[float, float] | None = (0.0, 1.0),
    maximum_dead_fraction: float | None = 0.30,
) -> DeblurResult:
    """Fuse blur families by interleaved exact solve and flux projection.

    Each pass performs one exact Fourier solve for the latent image, one
    isotropic shrink of its persistent gradient flux, and one Bregman update.
    There are no nested tolerances or inner convergence loops.
    """
    values, weight = _validate_inputs(observations, kernels, precisions)
    count = max(int(passes), 0)
    rho = float(flux_penalty)
    if rho <= 0.0:
        raise ValueError("flux_penalty must be positive")
    lam = max(float(tv_weight), 0.0)
    shape = values[0].shape[:2]
    transfers = [kernel.otf(shape) for kernel in kernels]
    coverage_diagnostics = coverage_report(transfers, weight)
    if (
        maximum_dead_fraction is not None
        and float(coverage_diagnostics["dead_fraction"])
        > float(maximum_dead_fraction)
    ):
        fallback = multi_wiener(
            values,
            kernels,
            precisions=weight,
            regularization=max(1e-3, 0.5 * lam),
        )
        return DeblurResult(
            image=fallback.image,
            diagnostics={
                **fallback.diagnostics,
                "method": "coverage_gated_wiener_fallback",
                "requested_method": "warm_split_bregman_exposure_transport",
                "requested_passes": count,
                "maximum_dead_fraction": float(maximum_dead_fraction),
                "coverage": coverage_diagnostics,
            },
        )
    coverage = np.zeros(shape, dtype=np.float64)
    for precision, transfer in zip(weight, transfers):
        coverage += precision * np.abs(transfer) ** 2
    denominator = coverage + rho * _positive_laplacian_symbol(shape)
    denominator[0, 0] = max(denominator[0, 0], np.finfo(float).tiny)
    channels = 1 if values[0].ndim == 2 else values[0].shape[2]
    output = []
    channel_records = []
    for channel in range(channels):
        data_rhs = np.zeros(shape, dtype=np.complex128)
        for precision, transfer, observation in zip(weight, transfers, values):
            plane = observation if channels == 1 else observation[..., channel]
            data_rhs += precision * np.conj(transfer) * np.fft.fft2(plane)
        # The multi-observation least-squares image is the physical basin.
        supported = coverage > np.max(coverage) * 1e-12
        initial_spectrum = np.zeros(shape, dtype=np.complex128)
        initial_spectrum[supported] = data_rhs[supported] / coverage[supported]
        latent = np.fft.ifft2(initial_spectrum).real
        dx, dy = _gradient(latent)
        bx = np.zeros(shape, dtype=np.float64)
        by = np.zeros(shape, dtype=np.float64)
        primal_trace: list[float] = []
        for _ in range(count):
            rhs = data_rhs + rho * np.fft.fft2(
                _gradient_adjoint(dx - bx, dy - by))
            latent = np.fft.ifft2(rhs / denominator).real
            if clip is not None:
                latent = np.clip(latent, float(clip[0]), float(clip[1]))
            gx, gy = _gradient(latent)
            vx = gx + bx
            vy = gy + by
            magnitude = np.hypot(vx, vy)
            factor = np.maximum(1.0 - lam / np.maximum(rho * magnitude, 1e-30), 0.0)
            dx = factor * vx
            dy = factor * vy
            bx += gx - dx
            by += gy - dy
            primal_trace.append(float(np.sqrt(np.mean((gx - dx) ** 2 + (gy - dy) ** 2))))
        output.append(latent)
        forward_mse = []
        for kernel, observation in zip(kernels, values):
            plane = observation if channels == 1 else observation[..., channel]
            residual = apply_circular(latent, kernel) - plane
            forward_mse.append(float(np.mean(residual * residual)))
        channel_records.append({
            "forward_mse": forward_mse,
            "split_residual_trace": primal_trace,
            "flux_peak": float(np.max(np.hypot(dx, dy))),
        })
    image = output[0] if channels == 1 else np.stack(output, axis=2)
    return DeblurResult(
        image=image,
        diagnostics={
            "method": "warm_split_bregman_exposure_transport",
            "passes": count,
            "tv_weight": lam,
            "flux_penalty": rho,
            "kernel_mass_error": [abs(kernel.mass - 1.0) for kernel in kernels],
            "coverage": coverage_diagnostics,
            "channels": channel_records,
        },
    )
