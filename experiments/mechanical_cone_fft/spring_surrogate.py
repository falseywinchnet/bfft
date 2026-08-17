
#!/usr/bin/env python3
"""
Computational flexure surrogate for an N=8 normalized Bruun/DIF Fourier walk.

Models:
1. One normalized Bruun 4x4 rotation cell as a reciprocal positive-spring graph.
2. A direct N=8 cone-lifted Fourier map balanced into a doubly stochastic
   bipartite spring graph.
3. A local three-stage N=8 Bruun-style factorization realized as cascaded
   spring-average/lever cells, exposing reciprocal back-loading.
4. Monte Carlo spring-ratio tolerance.

This is a lumped linear-elastic model, not continuum beam FEA.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def real_fourier_matrix(n: int) -> np.ndarray:
    t = np.arange(n)
    rows = [np.ones(n) / np.sqrt(n)]
    for k in range(1, n // 2):
        rows.append(np.sqrt(2 / n) * np.cos(2 * np.pi * k * t / n))
        rows.append(-np.sqrt(2 / n) * np.sin(2 * np.pi * k * t / n))
    rows.append(((-1.0) ** t) / np.sqrt(n))
    return np.vstack(rows)


def cone_lift(m: np.ndarray) -> np.ndarray:
    pos = np.maximum(m, 0.0)
    neg = np.maximum(-m, 0.0)
    return np.block([[pos, neg], [neg, pos]])


def bruun_norm_cell(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[1, 0, c, -s],
         [0, 1, s,  c],
         [1, 0, -c, s],
         [0, -1, s, c]], dtype=float
    ) / np.sqrt(2)


def top_half_split(n: int) -> np.ndarray:
    h = n // 2
    out = np.zeros((n, n))
    for j in range(h):
        out[j, j] = out[j, j + h] = 1 / np.sqrt(2)
        out[h + j, j] = 1 / np.sqrt(2)
        out[h + j, j + h] = -1 / np.sqrt(2)
    return out


def bruun8_factorization() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return three active stages, a free output permutation, and target F8."""
    f8 = real_fourier_matrix(8)
    s0 = top_half_split(8)

    # Odd half-bin branch: [B0,B1,B2,B3] -> [cos1,-sin1,cos3,-sin3].
    pbin = np.eye(4)[[0, 2, 1, 3], :]
    odd = np.diag([1, -1, 1, -1]) @ bruun_norm_cell(np.pi / 4) @ pbin

    # Even branch: a 4-point real Fourier walk split over two stages.
    e0 = top_half_split(4)
    e1 = np.array([
        [1 / np.sqrt(2),  1 / np.sqrt(2), 0,  0],
        [0,               0,              1,  0],
        [0,               0,              0, -1],
        [1 / np.sqrt(2), -1 / np.sqrt(2), 0,  0],
    ])

    s1 = linalg.block_diag(e0, odd)
    s2 = linalg.block_diag(e1, np.eye(4))

    # Natural output is [DC,cos2,-sin2,Nyq,cos1,-sin1,cos3,-sin3].
    perm = np.eye(8)[[0, 4, 5, 1, 2, 6, 7, 3], :]
    product = perm @ s2 @ s1 @ s0
    assert np.linalg.norm(product - f8) < 2e-14
    return [s0, s1, s2], perm, f8


def backward_gauges(stages: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    gauges: list[np.ndarray] = [None] * (len(stages) + 1)  # type: ignore[list-item]
    gauges[-1] = np.ones(stages[-1].shape[0])
    for s in range(len(stages) - 1, -1, -1):
        gauges[s] = np.abs(stages[s]).T @ gauges[s + 1]
    gauged = [
        (gauges[s + 1][:, None] * m) / gauges[s][None, :]
        for s, m in enumerate(stages)
    ]
    return gauges, gauged


def best_scalar_error(m: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    scale = float(np.sum(m * target) / np.sum(m * m))
    err = float(np.linalg.norm(scale * m - target) / np.linalg.norm(target))
    return err, scale


def sinkhorn_abs(m: np.ndarray, tol: float = 1e-14) -> tuple[np.ndarray, np.ndarray]:
    a = np.abs(m)
    r = np.ones(a.shape[0])
    c = np.ones(a.shape[1])
    for _ in range(10000):
        r /= r * (a @ c)
        c /= c * (a.T @ r)
        b = (r[:, None] * a) * c[None, :]
        if max(np.max(np.abs(b.sum(0) - 1)), np.max(np.abs(b.sum(1) - 1))) < tol:
            return r, c
    raise RuntimeError("Sinkhorn balancing did not converge")


def lever_spring_chain(
    weights: list[np.ndarray],
    input_gauge: np.ndarray,
    stiffness: np.ndarray,
    nominal_levers: list[np.ndarray],
) -> np.ndarray:
    """
    Each edge W[j,i] is a positive spring. Output j is read through lever l[j]:
        E_s = k_s/2 sum_ji W_ji (y_j/l_j - x_i)^2.
    Isolated, nominal W and l=row_sum(W) give y=W x.
    """
    stages = len(weights)
    m = weights[0].shape[0]
    d = m // 2
    h = np.zeros((stages * m, stages * m))
    rhs = np.zeros((stages * m, d))
    inject = np.vstack([np.diag(input_gauge), np.zeros((d, d))])

    for s, w in enumerate(weights):
        k = float(stiffness[s])
        lever = nominal_levers[s]
        row_sum = w.sum(axis=1)
        cur = s * m
        h[cur:cur+m, cur:cur+m] += k * np.diag(row_sum / lever**2)
        cross = -k * np.diag(1 / lever) @ w

        if s == 0:
            rhs[cur:cur+m] += k * np.diag(1 / lever) @ w @ inject
        else:
            prev = (s - 1) * m
            h[prev:prev+m, prev:prev+m] += k * np.diag(w.sum(axis=0))
            h[cur:cur+m, prev:prev+m] += cross
            h[prev:prev+m, cur:cur+m] += cross.T

    state = np.linalg.solve(h, rhs)
    project = np.hstack([np.eye(d), -np.eye(d)])
    return project @ state[-m:]


def optimize_stiffness(
    cone_stages: list[np.ndarray],
    input_gauge: np.ndarray,
    output_perm: np.ndarray,
    target: np.ndarray,
    total_span: float,
) -> tuple[np.ndarray, float]:
    levers = [a.sum(axis=1) for a in cone_stages]
    log_span = np.log(total_span)

    def objective(v: np.ndarray) -> float:
        a, b = v
        k = np.exp([a + b, b, 0.0])
        m = output_perm @ lever_spring_chain(cone_stages, input_gauge, k, levers)
        return best_scalar_error(m, target)[0]

    constraints = [
        {"type": "ineq", "fun": lambda v: v[0]},
        {"type": "ineq", "fun": lambda v: v[1]},
        {"type": "ineq", "fun": lambda v: log_span - np.sum(v)},
    ]
    result = minimize(
        objective, np.full(2, log_span / 2),
        method="SLSQP", constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    a, b = result.x
    k = np.exp([a + b, b, 0.0])
    return k, float(result.fun)


def perturbed_cone_stage(a: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    b = a.copy()
    mask = np.abs(b) > 1e-14
    sigma_ln = np.sqrt(np.log1p(sigma**2))
    b[mask] *= np.exp(rng.normal(-0.5 * sigma_ln**2, sigma_ln, mask.sum()))
    return cone_lift(b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("build/mechanical-cone-fft/springs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260812)

    # 1. Isolated normalized Bruun cell.
    cell = bruun_norm_cell(np.pi / 8)
    g_cell = np.abs(cell).T @ np.ones(4)
    cell_g = cell / g_cell[None, :]
    cell_cone = cone_lift(cell_g)
    cell_edges = cell_cone[cell_cone > 1e-15]
    cell_error = np.linalg.norm(
        np.hstack([np.eye(4), -np.eye(4)])
        @ cell_cone
        @ np.vstack([np.diag(g_cell), np.zeros((4, 4))])
        - cell
    ) / np.linalg.norm(cell)

    # 2. Direct exact N=8 reciprocal spring network.
    f8 = real_fourier_matrix(8)
    r, c = sinkhorn_abs(f8)
    balanced = (r[:, None] * f8) * c[None, :]
    direct_cone = cone_lift(balanced)
    direct_k = np.block([
        [np.diag(direct_cone.sum(0)), -direct_cone.T],
        [-direct_cone, np.diag(direct_cone.sum(1))],
    ])
    direct_error = np.linalg.norm(np.diag(1/r) @ balanced @ np.diag(1/c) - f8) / np.linalg.norm(f8)
    direct_eigs = np.linalg.eigvalsh(direct_k)

    # 3. Three active stages of the N=8 Bruun-style walk.
    stages, perm, target = bruun8_factorization()
    gauges, gauged = backward_gauges(stages)
    cone_stages = [cone_lift(a) for a in gauged]
    levers = [a.sum(axis=1) for a in cone_stages]

    span_rows = []
    for span in [1, 10, 100, 1000, 10000, 1_000_000]:
        if span == 1:
            stiffness = np.ones(3)
            m = perm @ lever_spring_chain(cone_stages, gauges[0], stiffness, levers)
            err, scale = best_scalar_error(m, target)
        else:
            stiffness, err = optimize_stiffness(cone_stages, gauges[0], perm, target, span)
            m = perm @ lever_spring_chain(cone_stages, gauges[0], stiffness, levers)
            err, scale = best_scalar_error(m, target)
        span_rows.append({
            "total_stiffness_span": span,
            "k_stage_0": stiffness[0],
            "k_stage_1": stiffness[1],
            "k_stage_2": stiffness[2],
            "shape_error_after_global_gain": err,
            "global_gain": scale,
        })

    # Use the practical-ish 1000:1 total stiffness design for tolerance tests.
    k_1000, baseline = optimize_stiffness(cone_stages, gauges[0], perm, target, 1000)
    tolerance_rows = []
    for sigma in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]:
        errors = []
        for _ in range(args.trials):
            perturbed = [perturbed_cone_stage(a, sigma, rng) for a in gauged]
            m = perm @ lever_spring_chain(perturbed, gauges[0], k_1000, levers)
            errors.append(best_scalar_error(m, target)[0])
        tolerance_rows.append({
            "spring_sigma": sigma,
            "median_shape_error": float(np.median(errors)),
            "p95_shape_error": float(np.quantile(errors, 0.95)),
        })

    summary = [
        ("bruun_cell_relative_error", cell_error),
        ("bruun_cell_spring_count", int(cell_edges.size)),
        ("bruun_cell_spring_ratio", float(cell_edges.max() / cell_edges.min())),
        ("bruun_cell_beam_thickness_ratio_if_k_proportional_t_cubed",
         float((cell_edges.max() / cell_edges.min()) ** (1/3))),
        ("direct_f8_relative_error", direct_error),
        ("direct_f8_spring_count", int(np.count_nonzero(direct_cone > 1e-15))),
        ("direct_f8_min_nonrigid_stiffness_eigenvalue", float(direct_eigs[1])),
        ("direct_f8_input_lever_ratio", float(c.max() / c.min())),
        ("direct_f8_output_lever_ratio", float(r.max() / r.min())),
        ("cascade_1000_baseline_shape_error", baseline),
    ]

    with (args.out / "bfft_flexure_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary)

    with (args.out / "bfft_flexure_stiffness.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=span_rows[0].keys())
        writer.writeheader()
        writer.writerows(span_rows)

    with (args.out / "bfft_flexure_tolerance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tolerance_rows[0].keys())
        writer.writeheader()
        writer.writerows(tolerance_rows)

    x = np.array([row["total_stiffness_span"] for row in span_rows], float)
    y = np.array([row["shape_error_after_global_gain"] for row in span_rows], float)
    plt.figure(figsize=(7, 4.5))
    plt.loglog(x, y, marker="o")
    plt.xlabel("Total stiffness span")
    plt.ylabel("Relative transform shape error")
    plt.title("N=8 staged Bruun flexure: reciprocal back-loading")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(args.out / "bfft_flexure_stiffness.png", dpi=180)
    plt.close()

    x = np.array([row["spring_sigma"] for row in tolerance_rows], float)
    med = np.array([row["median_shape_error"] for row in tolerance_rows], float)
    p95 = np.array([row["p95_shape_error"] for row in tolerance_rows], float)
    plt.figure(figsize=(7, 4.5))
    plt.loglog(x, med, marker="o", label="median")
    plt.loglog(x, p95, marker="o", label="95th percentile")
    plt.xlabel("Per-spring relative standard deviation")
    plt.ylabel("Relative transform shape error")
    plt.title("N=8 staged Bruun flexure: fabrication tolerance")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "bfft_flexure_tolerance.png", dpi=180)
    plt.close()

    print("Bruun cell error:", f"{cell_error:.3e}")
    print("Direct N=8 spring-network error:", f"{direct_error:.3e}")
    print("Direct N=8 springs:", int(np.count_nonzero(direct_cone > 1e-15)))
    print("Staged 1000:1 baseline shape error:", f"{baseline:.4%}")
    print("Wrote results to", args.out)


if __name__ == "__main__":
    main()
