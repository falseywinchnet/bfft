"""Halmos lossless dilation of a real symmetric contraction."""
from __future__ import annotations
import numpy as np


def defect_operator(K: np.ndarray, clip_tolerance=1e-12):
    lam, Q = np.linalg.eigh((K + K.T) / 2)
    if np.max(np.abs(lam)) > 1 + clip_tolerance:
        raise ValueError("K is not a contraction")
    d = np.sqrt(np.maximum(0.0, 1.0 - lam * lam))
    return (Q * d) @ Q.T


def halmos_dilation(K: np.ndarray):
    K = (np.asarray(K) + np.asarray(K).T) / 2
    D = defect_operator(K)
    U = np.block([[K, D], [D, -K]])
    residual = np.linalg.norm(U.T @ U - np.eye(U.shape[0]), 2)
    symmetry = np.linalg.norm(U - U.T, 2)
    return U, D, {"unitarity_residual": float(residual),
                  "symmetry_residual": float(symmetry)}
