"""Complete tensor algebra of independent participation coordinates.

Role, one-sided contour, and bounded enclosure answer different questions.
Collapsing them by a weighted sum would reintroduce a hand-authored object
rule.  Their complete non-empty tensor algebra contains every coordinate and
every conjunction with unit algebraic multiplicity:

    K = (1 + K_role) (1 + K_contour) (1 + K_enclosure) - 1.

The products are Schur products of positive kernels, hence remain positive.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def normalized_linear_kernel(embedding: np.ndarray) -> np.ndarray:
    value = np.asarray(embedding, dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("embedding must be a finite matrix")
    norm = np.linalg.norm(value, axis=1)
    denominator = norm[:, None] * norm[None, :]
    kernel = np.divide(
        value @ value.T,
        denominator,
        out=np.zeros((len(value), len(value)), dtype=np.float64),
        where=denominator > 1e-30,
    )
    return np.clip(0.5 * (kernel + kernel.T), -1.0, 1.0)


def complete_kernel_algebra(
    kernels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return every nonempty Schur monomial and their unit-multiplicity sum."""
    if not kernels:
        raise ValueError("at least one participation kernel is required")
    names = list(kernels)
    matrices = [np.asarray(kernels[name], dtype=np.float64) for name in names]
    if any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("participation kernels must share one square shape")
    if matrices[0].ndim != 2 or matrices[0].shape[0] != matrices[0].shape[1]:
        raise ValueError("participation kernels must be square")
    identity_kernel = np.ones_like(matrices[0])
    raw_complete = identity_kernel.copy()
    for matrix in matrices:
        raw_complete *= identity_kernel + matrix
    raw_complete -= identity_kernel
    raw_complete = 0.5 * (raw_complete + raw_complete.T)
    diagonal = np.maximum(np.diag(raw_complete), 0.0)
    denominator = np.sqrt(diagonal[:, None] * diagonal[None, :])
    normalized = np.divide(
        raw_complete,
        denominator,
        out=np.zeros_like(raw_complete),
        where=denominator > 1e-30,
    )
    result = {}
    for size in range(1, len(names) + 1):
        for subset in combinations(range(len(names)), size):
            key = "_".join(names[index] for index in subset)
            value = np.ones_like(matrices[0])
            for index in subset:
                value *= matrices[index]
            result[key] = value
    result["complete"] = np.ascontiguousarray(normalized)
    return result


def complete_participation_kernel(
    role: np.ndarray,
    contour: np.ndarray,
    enclosure: np.ndarray,
) -> dict[str, np.ndarray]:
    """Backward-compatible three-coordinate participation algebra."""
    return complete_kernel_algebra({
        "role": role,
        "contour": contour,
        "enclosure": enclosure,
    })
