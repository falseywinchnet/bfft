"""Analytic geometry of one positive blur-transport measure.

There is no blur catalogue in this module.  Gaussian, aperture, line, curve,
camera path, and finite optimized filters are all non-negative probability
measures on displacement.  Their direction is the flow of the logarithmic
characteristic attenuation; their low-frequency basis is the exact cumulant
jet of that same measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import TransportKernel, identity_kernel, path_kernel


@dataclass(frozen=True)
class AnalyticTransportSupport:
    """Finite cumulant/eikonal basis extracted without a family decision."""

    centroid_xy: np.ndarray
    covariance: np.ndarray
    covariance_eigenvalues: np.ndarray
    covariance_directions_xy: np.ndarray
    numerical_dimension: int
    third_central_moment: np.ndarray
    fourth_cumulant: np.ndarray
    principal_direction_xy: np.ndarray
    transverse_direction_xy: np.ndarray
    signed_bend_coupling: float
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class FourierEikonalField:
    """Exact characteristic attenuation and its frequency-space flow."""

    frequencies_xy: np.ndarray
    characteristic: np.ndarray
    attenuation_action: np.ndarray
    flow_xy: np.ndarray
    supported: np.ndarray


def _atoms(
    kernel: TransportKernel,
    *,
    centered: bool,
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[: kernel.psf.shape[0], : kernel.psf.shape[1]]
    center = 0.5 * (np.asarray(kernel.psf.shape, dtype=np.float64) - 1.0)
    points = np.column_stack((
        (xx - center[1]).ravel(),
        (yy - center[0]).ravel(),
    ))
    weights = kernel.psf.ravel()
    positive = weights > 0.0
    points = points[positive]
    weights = weights[positive]
    weights = weights / np.sum(weights)
    if centered:
        points = points - np.sum(points * weights[:, None], axis=0)
    return points, weights


def _canonical_axis(direction: np.ndarray) -> np.ndarray:
    axis = np.asarray(direction, dtype=np.float64).copy()
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis *= -1.0
    return axis


def analyze_transport_support(
    kernel: TransportKernel,
) -> AnalyticTransportSupport:
    """Return the exact finite cumulant jet of ``kernel``.

    Numerical dimension is matrix rank at floating-point resolution, not a
    tuned anisotropy threshold.  The returned directions are coordinates of
    one measure; they are never mapped to blur-family labels.
    """
    raw_points, weights = _atoms(kernel, centered=False)
    centroid = np.sum(raw_points * weights[:, None], axis=0)
    points = raw_points - centroid[None, :]
    covariance = np.einsum("n,ni,nj->ij", weights, points, points)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    directions = eigenvectors[:, order].T
    directions = np.stack([_canonical_axis(axis) for axis in directions])
    largest = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    rank_tolerance = (
        np.finfo(float).eps
        * max(len(points), covariance.shape[0])
        * max(largest, 1.0)
    )
    dimension = int(np.sum(eigenvalues > rank_tolerance))
    third = np.einsum("n,ni,nj,nk->ijk", weights, points, points, points)
    fourth_moment = np.einsum(
        "n,ni,nj,nk,nl->ijkl", weights, points, points, points, points)
    fourth = fourth_moment.copy()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    fourth[i, j, k, l] -= (
                        covariance[i, j] * covariance[k, l]
                        + covariance[i, k] * covariance[j, l]
                        + covariance[i, l] * covariance[j, k]
                    )
    principal = (
        directions[0]
        if largest > rank_tolerance else np.asarray((1.0, 0.0)))
    transverse = np.asarray((-principal[1], principal[0]))
    longitudinal = points @ principal
    normal = points @ transverse
    mixed_third = float(np.sum(weights * longitudinal * longitudinal * normal))
    bend_scale = max(largest ** 1.5, np.finfo(float).tiny)
    bend = mixed_third / bend_scale if largest > rank_tolerance else 0.0
    return AnalyticTransportSupport(
        centroid_xy=centroid,
        covariance=covariance,
        covariance_eigenvalues=eigenvalues,
        covariance_directions_xy=directions,
        numerical_dimension=dimension,
        third_central_moment=third,
        fourth_cumulant=fourth,
        principal_direction_xy=principal,
        transverse_direction_xy=transverse,
        signed_bend_coupling=float(bend),
        diagnostics={
            "basis": "one_positive_displacement_measure",
            "family_classification": False,
            "atom_count": int(len(points)),
            "centroid_xy": centroid.tolist(),
            "covariance": covariance.tolist(),
            "covariance_eigenvalues": eigenvalues.tolist(),
            "covariance_directions_xy": directions.tolist(),
            "numerical_dimension": dimension,
            "rank_tolerance": float(rank_tolerance),
            "principal_direction_xy": principal.tolist(),
            "transverse_direction_xy": transverse.tolist(),
            "signed_bend_coupling": float(bend),
            "direction_origin": (
                "eigenframe_of_exact_second_cumulant_with_third_cumulant_bend"
            ),
        },
    )


def covariance_transport_kernel(
    covariance: np.ndarray,
    *,
    name: str = "analytic_covariance_measure",
) -> TransportKernel:
    """Return the unique tensor Lobatto cubature used for one covariance.

    Every positive eigenaxis contributes the same three-node moment rule
    ``(-sqrt(3 lambda), 0, +sqrt(3 lambda))`` with weights
    ``(1/6, 2/3, 1/6)``.  Tensoring the active axes yields one positive
    continuous measure with exactly the requested covariance. Bilinear
    rasterization may add its honest subpixel footprint covariance. No
    anisotropy threshold or shape label is involved.
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (2, 2) or np.any(~np.isfinite(matrix)):
        raise ValueError("covariance must be one finite 2x2 matrix")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    tolerance = (
        np.finfo(float).eps
        * matrix.shape[0]
        * max(float(np.max(eigenvalues)), 1.0)
    )
    points = np.zeros((1, 2), dtype=np.float64)
    weights = np.ones(1, dtype=np.float64)
    for value, direction in zip(eigenvalues, eigenvectors.T):
        if value <= tolerance:
            continue
        extent = np.sqrt(3.0 * float(value)) * direction
        axis_points = np.stack((-extent, np.zeros(2), extent), axis=0)
        axis_weights = np.asarray((1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0))
        points = (points[:, None, :] + axis_points[None, :, :]).reshape(-1, 2)
        weights = (weights[:, None] * axis_weights[None, :]).reshape(-1)
    if len(points) == 1:
        return identity_kernel()
    return path_kernel(points, weights=weights, name=name)


def fourier_eikonal_field(
    kernel: TransportKernel,
    frequencies_xy: np.ndarray,
) -> FourierEikonalField:
    """Evaluate ``-log |phi|^2`` and its exact analytic gradient.

    ``phi`` is the characteristic function of the centered positive measure.
    The flow is evaluated from its finite atoms, not by finite differences,
    optimization, family fitting, or directional templates.  At a true
    characteristic zero the direction is honestly unsupported.
    """
    frequencies = np.asarray(frequencies_xy, dtype=np.float64)
    if frequencies.shape[-1:] != (2,):
        raise ValueError("frequencies_xy must end in an x,y coordinate")
    points, weights = _atoms(kernel, centered=True)
    flat = frequencies.reshape(-1, 2)
    phase = np.exp(-2j * np.pi * (flat @ points.T))
    characteristic = phase @ weights
    derivative = -2j * np.pi * np.einsum(
        "fa,aj,a->fj", phase, points, weights)
    magnitude_squared = np.abs(characteristic) ** 2
    support_floor = (
        np.finfo(float).eps * max(len(points), 1)
    ) ** 2
    supported = magnitude_squared > support_floor
    action = np.full(len(flat), np.inf, dtype=np.float64)
    flow = np.zeros((len(flat), 2), dtype=np.float64)
    action[supported] = -np.log(magnitude_squared[supported])
    flow[supported] = -2.0 * np.real(
        derivative[supported] / characteristic[supported, None])
    output_shape = frequencies.shape[:-1]
    return FourierEikonalField(
        frequencies_xy=frequencies.copy(),
        characteristic=characteristic.reshape(output_shape),
        attenuation_action=action.reshape(output_shape),
        flow_xy=flow.reshape(output_shape + (2,)),
        supported=supported.reshape(output_shape),
    )
