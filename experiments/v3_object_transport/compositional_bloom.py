"""Parameter-free compositional controls for participation kernels.

These operators test whether generic transitivity is sufficient.  The
spectral exponential sums every path order after normalizing by the measured
spectral radius.  The typed order-two kernel retains every ordered pair of
coordinate types with unit multiplicity.  Both are PSD and seed-free; neither
contains an object-specific transition rule.
"""

from __future__ import annotations

import numpy as np


def _normalize_diagonal(kernel: np.ndarray) -> np.ndarray:
    value = 0.5 * (
        np.asarray(kernel, dtype=np.float64)
        + np.asarray(kernel, dtype=np.float64).T
    )
    diagonal = np.maximum(np.diag(value), 0.0)
    denominator = np.sqrt(diagonal[:, None] * diagonal[None, :])
    return np.divide(
        value, denominator, out=np.zeros_like(value),
        where=denominator > 1e-30)


def spectral_exponential_bloom(kernel: np.ndarray) -> np.ndarray:
    """Analytically sum all path orders at one measured spectral unit."""
    value = 0.5 * (
        np.asarray(kernel, dtype=np.float64)
        + np.asarray(kernel, dtype=np.float64).T
    )
    eigenvalue, eigenvector = np.linalg.eigh(value)
    scale = max(float(eigenvalue[-1]), 1e-30)
    gain = np.expm1(np.clip(eigenvalue / scale, 0.0, None))
    bloomed = (eigenvector * gain) @ eigenvector.T
    return _normalize_diagonal(bloomed)


def typed_order_two_bloom(
    kernels: list[np.ndarray] | tuple[np.ndarray, ...],
) -> np.ndarray:
    """Direct-sum feature kernel of every coordinate word of length 1 or 2."""
    matrices = [np.asarray(kernel, dtype=np.float64) for kernel in kernels]
    if not matrices:
        raise ValueError("at least one coordinate kernel is required")
    if any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("coordinate kernels must have equal shape")
    one_port = sum(matrix @ matrix for matrix in matrices)
    result = one_port.copy()
    # If F_ij = K_i K_j is the feature matrix of the typed word ij,
    # F_ij F_ij^T = K_i K_j^2 K_i.  Summing every i,j gives this factorization.
    for matrix in matrices:
        result += matrix @ one_port @ matrix
    return _normalize_diagonal(result)
