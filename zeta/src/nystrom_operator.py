"""Dense finite Hankel operators and Fredholm determinant quotients."""
from __future__ import annotations

import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from suzuki_kernel import k_omega


def composite_gauss_legendre(A: float, n: int, panel_order: int = 16):
    panels = max(1, int(math.ceil(n / panel_order)))
    order = max(2, int(math.ceil(n / panels)))
    base_x, base_w = leggauss(order)
    edges = np.linspace(-A, A, panels + 1)
    xs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        xs.append((lo + hi) / 2 + (hi - lo) / 2 * base_x)
        ws.append((hi - lo) / 2 * base_w)
    return np.concatenate(xs), np.concatenate(ws)


def nystrom_matrix(omega: float, a: float, n: int, panel_order: int = 16):
    A = math.log(a)
    u, q = composite_gauss_legendre(A, n, panel_order)
    values = k_omega(omega, (u[:, None] + u[None, :]).ravel()).reshape(u.size, u.size)
    K = np.sqrt(q[:, None] * q[None, :]) * values
    return u, q, (K + K.T) / 2


def uniform_hankel(omega: float, a: float, n: int, cell_average: bool = True):
    A = math.log(a)
    du = 2 * A / n
    u = -A + (np.arange(n) + 0.5) * du
    r = -2 * A + (np.arange(2 * n - 1) + 1.0) * du
    if cell_average:
        # Piecewise-constant Galerkin coefficient. The triangular average
        # resolves every r=log(integer) crossing instead of point-sampling a
        # cusp/singularity. K_ij remains exactly Hankel and symmetric.
        samples = np.empty_like(r)
        max_integer = int(math.ceil(a * a))
        breakpoints = [math.log(j) for j in range(1, max_integer + 1)]
        for p, center in enumerate(r):
            lo, hi = center - du, center + du
            pts = [center] + [q for q in breakpoints if lo < q < hi]
            def integrand(t):
                return (du - abs(t - center)) * k_omega(omega, t) / (du * du)
            samples[p] = quad(integrand, lo, hi, points=sorted(set(pts)),
                              epsabs=2e-12, epsrel=2e-12, limit=120)[0]
    else:
        samples = k_omega(omega, r)
    idx = np.add.outer(np.arange(n), np.arange(n))
    return u, du, samples, du * samples[idx]


def fredholm_summary(K: np.ndarray):
    K = np.asarray(K, dtype=float)
    sym = np.linalg.norm(K - K.T, 2) / max(np.linalg.norm(K, 2), np.finfo(float).tiny)
    eig = np.linalg.eigvalsh((K + K.T) / 2)
    rho = float(np.max(np.abs(eig)))
    plus = 1 + eig
    minus = 1 - eig
    valid = bool(np.all(plus > 0) and np.all(minus > 0))
    logdet_plus = float(np.log(plus).sum()) if valid else float("nan")
    logdet_minus = float(np.log(minus).sum()) if valid else float("nan")
    log_m = logdet_plus - logdet_minus
    return {
        "N": K.shape[0], "symmetry_residual": float(sym),
        "lambda_min": float(eig[0]), "lambda_max": float(eig[-1]),
        "spectral_radius": rho, "defect_margin": 1 - rho * rho,
        "determinant_arguments_positive": valid,
        "logdet_plus": logdet_plus, "logdet_minus": logdet_minus,
        "log_m": log_m, "m": float(np.exp(log_m)) if valid and log_m < 700 else float("inf"),
    }


def solve_fredholm(K: np.ndarray, b: np.ndarray):
    eye = np.eye(K.shape[0])
    return np.linalg.solve(eye + K, b), np.linalg.solve(eye - K, b)
