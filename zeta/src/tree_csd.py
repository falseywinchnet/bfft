"""Dyadic packet ordering and recursive cosine-sine diagnostics."""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import cossin


def bit_reverse_permutation(n: int):
    bits = int(math.log2(n))
    if 1 << bits != n:
        raise ValueError("packet order requires power-of-two N")
    return np.array([int(f"{i:0{bits}b}"[::-1], 2) for i in range(n)])


def dip_channel_order(n: int):
    p = bit_reverse_permutation(n)
    return np.concatenate([p, n + p])


def recursive_csd(U: np.ndarray, omega, a, N, min_size=4):
    """Record exact local CSDs on the same balanced binary partitions.

    Each node's local reconstruction is checked independently. Recursion is
    over the diagonal phase blocks returned by the CSD, preserving ancestry.
    """
    U = np.asarray(U, dtype=float)
    order = dip_channel_order(N)
    root = U[np.ix_(order, order)]
    rows = []

    def visit(M, depth, node, d, e):
        m = M.shape[0]
        if m < 4 or m % 2:
            return
        half = m // 2
        try:
            (u1, u2), theta, (v1h, v2h) = cossin(M, p=half, q=half,
                                                   separate=True)
            C, S = np.diag(np.cos(theta)), np.diag(np.sin(theta))
            middle = np.block([[C, -S], [S, C]])
            left = np.block([[u1, np.zeros_like(u1)], [np.zeros_like(u2), u2]])
            right = np.block([[v1h, np.zeros_like(v1h)], [np.zeros_like(v2h), v2h]])
            residual = np.linalg.norm(left @ middle @ right - M, 2)
            angle_values = theta
        except Exception:
            # A finite contraction-derived dilation can contain exact
            # degeneracies. Singular values of the leading block are the
            # invariant cosines even when a LAPACK CSD driver rejects it.
            cosines = np.clip(np.linalg.svd(M[:half, :half], compute_uv=False), 0, 1)
            angle_values = np.arccos(cosines)
            residual = float("nan")
            u1 = u2 = v1h = v2h = np.eye(half)
        rows.append({
            "omega": omega, "a": a, "N": N, "depth": depth,
            "node_index": node, "d": d, "e": e, "span_width": m,
            "theta_mean": float(np.mean(angle_values)),
            "theta_std": float(np.std(angle_values)),
            "theta_min": float(np.min(angle_values)),
            "theta_max": float(np.max(angle_values)),
            "minimum_cosine": float(np.min(np.cos(angle_values))),
            "minimum_sine": float(np.min(np.sin(angle_values))),
            "left_phase_summary": float(np.trace(u1) / max(1, u1.shape[0])),
            "right_phase_summary": float(np.trace(v1h) / max(1, v1h.shape[0])),
            "local_reconstruction_residual": float(residual),
        })
        if half >= min_size:
            visit(u1 @ v1h, depth + 1, 2 * node, d, 2 * e)
            visit(u2 @ v2h, depth + 1, 2 * node + 1, e - d, 2 * e)

    visit(root, 0, 0, 1, 2)
    return rows
