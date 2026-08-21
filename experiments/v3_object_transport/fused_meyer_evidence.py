"""Deterministic fused Meyer evidence aligned to an unchanged V3 raster.

The final V3 region labels remain authoritative.  This adapter runs the later
warm-interleaved Gilles--Osher/Bregman decomposition on the same OKLab
luminance raster and returns cartoon, texture, and unresolved residual as
separate evidence coordinates.  It does not resegment or merge anything.
"""

from __future__ import annotations

import numpy as np

from experiments.meyer_bregman import a2bc_fused


def build_fused_meyer_evidence(
    target_lab: np.ndarray,
    *,
    passes: int = 400,
    lam: float = 0.05,
    mu: float = 40.0,
) -> dict:
    """Return a fixed-pass fused split in normalized OKLab luminance units.

    The published experiment uses image values in ``[0, 255]``.  Fixed pass
    count replaces its wall-clock stopping form so control artifacts are
    reproducible across hosts.  The exact residual is retained instead of
    being hidden inside either named component.
    """
    lab = np.asarray(target_lab, dtype=np.float64)
    if lab.ndim != 3 or lab.shape[2] < 1 or not np.all(np.isfinite(lab)):
        raise ValueError("target_lab must be a finite HxWxC array")
    if int(passes) < 1 or float(lam) <= 0.0 or float(mu) <= 0.0:
        raise ValueError("passes, lambda, and mu must be positive")
    target = 255.0 * lab[..., 0]
    cartoon, texture, completed, _ = a2bc_fused(
        target,
        float(lam),
        float(mu),
        beta=0.0,
        seed_sweeps=0,
        sweeps_per_step=1,
        max_outer=int(passes),
        # A fixed numerical experiment must not stop at image-dependent
        # iteration counts.  A negative tolerance makes the loop exact.
        outer_tol=-1.0,
        checkpoints=None,
    )
    residual = target - cartoon - texture
    scale = 1.0 / 255.0
    normalized_target = target * scale
    normalized_cartoon = cartoon * scale
    normalized_texture = texture * scale
    normalized_residual = residual * scale
    recomposed = (
        normalized_cartoon + normalized_texture + normalized_residual)
    return {
        "target": np.ascontiguousarray(normalized_target),
        "cartoon": np.ascontiguousarray(normalized_cartoon),
        "texture": np.ascontiguousarray(normalized_texture),
        "residual": np.ascontiguousarray(normalized_residual),
        "outer_passes": int(completed),
        "lambda": float(lam),
        "mu": float(mu),
        "recomposition_max_abs": float(np.max(np.abs(
            recomposed - normalized_target))),
        "residual_rms": float(np.sqrt(np.mean(
            normalized_residual * normalized_residual))),
    }
