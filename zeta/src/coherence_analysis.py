"""Refinement, effective-rank, description-length, and control analyses."""
from __future__ import annotations
import numpy as np

OBSERVABLES = ("theta_mean", "theta_std", "minimum_cosine", "minimum_sine")


def refinement_metrics(coarse, fine):
    fine_map = {(r["depth"], r["node_index"]): r for r in fine}
    drifts = []
    for r in coarse:
        # Same physical packet is represented by the same ancestry label.
        q = fine_map.get((r["depth"], r["node_index"]))
        if q:
            drifts.append(max(abs(float(r[k]) - float(q[k])) for k in OBSERVABLES))
    a = np.asarray(drifts or [np.nan])
    return {"matched_nodes": len(drifts), "median_drift": float(np.nanmedian(a)),
            "p95_drift": float(np.nanpercentile(a, 95)), "max_drift": float(np.nanmax(a))}


def observable_matrix(rows, key="theta_mean"):
    depths = sorted({int(r["depth"]) for r in rows})
    width = max(2 ** d for d in depths)
    M = np.full((len(depths), width), np.nan)
    for r in rows:
        d = depths.index(int(r["depth"]))
        j = int(r["node_index"])
        stride = width // (2 ** int(r["depth"]))
        M[d, j * stride:(j + 1) * stride] = float(r[key])
    means = np.nanmean(M, axis=1)
    inds = np.where(np.isnan(M)); M[inds] = means[inds[0]]
    return M


def effective_rank(rows, key="theta_mean", thresholds=(.99, .999, .99999)):
    M = observable_matrix(rows, key)
    s = np.linalg.svd(M - M.mean(), compute_uv=False)
    energy = np.cumsum(s * s) / max(np.sum(s * s), np.finfo(float).tiny)
    result = {"observable": key, "rows": M.shape[0], "columns": M.shape[1]}
    for t in thresholds:
        result[f"rank_{100*t:g}pct"] = int(np.searchsorted(energy, t) + 1)
    return result


def description_length(rows, key="theta_mean"):
    M = observable_matrix(rows, key)
    centered = M - M.mean()
    norm = max(np.linalg.norm(centered), np.finfo(float).tiny)
    out = []
    # depth-only and diagonal-only additive reconstructions.
    models = [("constant", np.full_like(M, M.mean()), 1),
              ("depth_only", np.repeat(M.mean(1)[:, None], M.shape[1], 1), M.shape[0]),
              ("diagonal_only", np.repeat(M.mean(0)[None, :], M.shape[0], 0), M.shape[1])]
    for name, fit, p in models:
        out.append({"model": name, "parameters": p,
                    "relative_error": float(np.linalg.norm(M-fit)/norm)})
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    for rank in range(1, min(M.shape) + 1):
        fit = M.mean() + (u[:, :rank] * s[:rank]) @ vh[:rank]
        out.append({"model": f"rank_{rank}",
                    "parameters": rank * (M.shape[0] + M.shape[1] + 1) + 1,
                    "relative_error": float(np.linalg.norm(M-fit)/norm)})
    out.append({"model": "unrestricted", "parameters": M.size, "relative_error": 0.0})
    return out


def control_contractions(n: int, seed=19):
    """Four symmetric Hankel controls normalized to a common contraction."""
    j = np.arange(2*n-1, dtype=float)
    rng = np.random.default_rng(seed)
    phases = {
        "gaussian_fourier_phase": np.exp(-0.025*(j-n/2)**2) * np.cos(.31*j),
        "finite_blaschke": (0.72**j) * np.cos(.47*j),
        "random_rational_allpass": sum(rng.uniform(.3,.9)**j*np.cos(rng.uniform(0,np.pi)*j)
                                        for _ in range(5)),
        "generic_smooth_unit_phase": np.cos(.17*j + .8*np.sin(.037*j)),
    }
    idx = np.add.outer(np.arange(n), np.arange(n))
    out = {}
    for name, h in phases.items():
        K = h[idx]
        K *= .8 / max(np.linalg.norm(K, 2), np.finfo(float).tiny)
        out[name] = K
    return out
