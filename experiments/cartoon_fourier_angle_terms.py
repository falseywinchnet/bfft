#!/usr/bin/env python3
"""Can finitely many active-angle terms improve the one-shot Hodge drop?

The one-shot starts with a flux ``p0`` whose divergence is the current primal
request ``q = c*(u-g)``.  Pointwise disk projection makes that flux feasible
but changes its divergence.  The nonlinear coupling left between those two
constraints can be written as alternating exact projections:

    p_(k+1) = P_disk(P_div=q(p_k)).

Each composition refreshes every active angle exactly; it is stronger than
adding another local sine/cosine Taylor coefficient.  We evaluate a fixed
number of these terms without a convergence tolerance and accept every primal
proposal against the original isotropic objective.

Two oracles diagnose why the terms saturate:

1. a convex projected-gradient solve measures the minimum possible
   ``||div(p)-q||`` over the unit disk and certifies its KKT fixed point;
2. at small size, the exact fixed-normal Schur inverse measures the tangent
   excursion.  If ``|t| >= 1``, the square-root active-angle series
   ``sqrt(1-t^2)`` is outside its real convergence domain.

The oracles are diagnostic only.  They are deliberately not candidate
production solvers.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cartoon_fourier_active_coupling as active_coupling  # noqa: E402
import cartoon_fourier_transport as fourier  # noqa: E402
import meyer_bregman as meyer  # noqa: E402


@dataclass
class AngleTerm:
    term: int
    alpha: float
    objective: float
    equivalent_pass: str
    divergence_residual: float
    preprojection_overload: float
    preprojection_maximum: float


@dataclass
class FeasibilityOracle:
    relative_residual: float
    maximum_residual: float
    fixed_point_error: float
    saturated_fraction: float


@dataclass
class AngleRadius:
    active: int
    rank: int
    normal_residual: float
    tangent_outside_radius: float
    tangent_median: float
    tangent_p95: float
    tangent_maximum: float


@dataclass
class CurvatureTerm:
    order: int
    coefficient_norm: float
    active_capacity_rms: float
    global_overload: float
    maximum_norm: float


def project_disk(
    px: np.ndarray,
    py: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(1.0, np.hypot(px, py))
    return px / scale, py / scale


def _relative_divergence_residual(
    px: np.ndarray,
    py: np.ndarray,
    requested: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(meyer.div(px, py) - requested)
        / max(np.linalg.norm(requested), 1e-30)
    )


def refreshed_angle_terms(
    current: np.ndarray,
    current_flux: tuple[np.ndarray, np.ndarray],
    g: np.ndarray,
    c: float,
    objectives: np.ndarray,
    terms: int,
) -> list[AngleTerm]:
    """Return a fixed number of exact angle-refresh compositions.

    There is no tolerance or adaptive stopping rule.  Term one is precisely
    the existing Hodge proposal.  Later terms add one longitudinal Poisson
    closure and one exact pointwise disk projection apiece.
    """
    requested = c * (current - g)
    baseline_objective = fourier.objective(current, g, c)
    route_x, route_y = current_flux
    result = []
    for term in range(1, terms + 1):
        pre_x, pre_y = fourier.routed_preflux(
            current, (route_x, route_y), g, c
        )
        pre_norm = np.hypot(pre_x, pre_y)
        route_x, route_y = project_disk(pre_x, pre_y)
        raw = g + meyer.div(route_x, route_y) / c
        accepted, alpha = fourier._segment_taylor_drop(
            current, raw, g, c
        )
        value = fourier.objective(accepted, g, c)
        # The exact objective gate guarantees this, but retaining the
        # baseline on a rejected proposal makes the contract explicit.
        value = min(value, baseline_objective)
        result.append(
            AngleTerm(
                term=term,
                alpha=alpha,
                objective=value,
                equivalent_pass=fourier.equivalent_pass(value, objectives),
                divergence_residual=_relative_divergence_residual(
                    route_x, route_y, requested
                ),
                preprojection_overload=float(np.mean(pre_norm > 1.0)),
                preprojection_maximum=float(np.max(pre_norm)),
            )
        )
    return result


def closest_disk_divergence(
    requested: np.ndarray,
    *,
    iterations: int = 5000,
    polish: int = 256,
    step: float = 0.124,
) -> tuple[np.ndarray, np.ndarray, FeasibilityOracle]:
    """Convex oracle for ``min_|p|<=1 0.5*||div(p)-requested||^2``.

    The divergence operator has squared norm at most eight, hence
    ``step < 1/8``.  FISTA obtains the solution quickly and ordinary
    projected-gradient polishing makes the final KKT fixed-point diagnostic
    easy to interpret.
    """
    px = np.zeros_like(requested)
    py = np.zeros_like(requested)
    extrapolated_x = px.copy()
    extrapolated_y = py.copy()
    momentum = 1.0
    for _ in range(iterations):
        residual = meyer.div(extrapolated_x, extrapolated_y) - requested
        gx, gy = meyer.grad(residual)
        next_x, next_y = project_disk(
            extrapolated_x + step * gx,
            extrapolated_y + step * gy,
        )
        next_momentum = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)
        )
        factor = (momentum - 1.0) / next_momentum
        extrapolated_x = next_x + factor * (next_x - px)
        extrapolated_y = next_y + factor * (next_y - py)
        px, py = next_x, next_y
        momentum = next_momentum

    for _ in range(polish):
        residual = meyer.div(px, py) - requested
        gx, gy = meyer.grad(residual)
        px, py = project_disk(px + step * gx, py + step * gy)

    residual = meyer.div(px, py) - requested
    gx, gy = meyer.grad(residual)
    fixed_x, fixed_y = project_disk(px + step * gx, py + step * gy)
    scale = max(float(np.linalg.norm(px) + np.linalg.norm(py)), 1.0)
    fixed_point_error = float(
        (np.linalg.norm(fixed_x - px) + np.linalg.norm(fixed_y - py))
        / scale
    )
    return (
        px,
        py,
        FeasibilityOracle(
            relative_residual=float(
                np.linalg.norm(residual)
                / max(np.linalg.norm(requested), 1e-30)
            ),
            maximum_residual=float(np.max(np.abs(residual))),
            fixed_point_error=fixed_point_error,
            saturated_fraction=float(
                np.mean(np.hypot(px, py) >= 1.0 - 1e-10)
            ),
        ),
    )


def active_response_matrices(
    active: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normal and tangent response to fixed-normal transverse sources."""
    h, w = active.shape
    y, x = np.nonzero(active)
    normal_x = nx[active]
    normal_y = ny[active]
    tangent_x = -normal_y
    tangent_y = normal_x
    kxx, kxy, kyx, kyy = active_coupling.transverse_kernel(active.shape)
    dy = (y[:, None] - y[None, :]) % h
    dx = (x[:, None] - x[None, :]) % w

    def response(left_x, left_y):
        return (
            left_x[:, None] * normal_x[None, :] * kxx[dy, dx]
            + left_x[:, None] * normal_y[None, :] * kxy[dy, dx]
            + left_y[:, None] * normal_x[None, :] * kyx[dy, dx]
            + left_y[:, None] * normal_y[None, :] * kyy[dy, dx]
        )

    normal = response(normal_x, normal_y)
    normal = 0.5 * (normal + normal.T)
    tangent = response(tangent_x, tangent_y)
    return normal, tangent


def fixed_normal_angle_radius(
    p0x: np.ndarray,
    p0y: np.ndarray,
    *,
    relative_eigenvalue_cutoff: float = 1e-8,
) -> AngleRadius:
    """Measure whether higher square-root angle terms can even converge."""
    magnitude = np.hypot(p0x, p0y)
    active = magnitude > 1.0 + 1e-10
    nx = p0x / np.maximum(magnitude, 1e-30)
    ny = p0y / np.maximum(magnitude, 1e-30)
    normal, tangent = active_response_matrices(active, nx, ny)
    eigenvalues, eigenvectors = np.linalg.eigh(normal)
    threshold = max(float(eigenvalues[-1]), 1e-30) * \
        relative_eigenvalue_cutoff
    retained = eigenvalues > threshold
    rhs = 1.0 - magnitude[active]
    multiplier = eigenvectors[:, retained] @ (
        (eigenvectors[:, retained].T @ rhs) / eigenvalues[retained]
    )
    tangent_response = tangent @ multiplier
    return AngleRadius(
        active=rhs.size,
        rank=int(np.count_nonzero(retained)),
        normal_residual=float(
            np.linalg.norm(normal @ multiplier - rhs)
            / max(np.linalg.norm(rhs), 1e-30)
        ),
        tangent_outside_radius=float(np.mean(np.abs(tangent_response) >= 1.0)),
        tangent_median=float(np.median(np.abs(tangent_response))),
        tangent_p95=float(np.percentile(np.abs(tangent_response), 95.0)),
        tangent_maximum=float(np.max(np.abs(tangent_response), initial=0.0)),
    )


def fixed_normal_curvature_terms(
    p0x: np.ndarray,
    p0y: np.ndarray,
    *,
    order: int = 4,
    relative_eigenvalue_cutoff: float = 1e-8,
) -> list[CurvatureTerm]:
    """Evaluate the first four explicit active-angle curvature terms.

    For a fixed initial active mask and fixed normal sources, let

        r = S*lambda,  t = R*lambda,
        m + r = sqrt(1-t^2).

    Scaling the initial overload by a formal epsilon and writing
    ``lambda = sum epsilon^k lambda_k`` gives

        S lambda_1 = 1-m
        S lambda_2 = -t_1^2/2
        S lambda_3 = -t_1*t_2
        S lambda_4 = -t_1*t_3 - t_2^2/2 - t_1^4/8.

    These are genuine additional algebraic terms, not repeated nonlinear
    projection steps.  They can only converge while ``|t| < 1`` and while
    the initial active mask remains valid.
    """
    if order < 1 or order > 4:
        raise ValueError("the explicit curvature expansion supports order 1..4")
    magnitude = np.hypot(p0x, p0y)
    active = magnitude > 1.0 + 1e-10
    nx = p0x / np.maximum(magnitude, 1e-30)
    ny = p0y / np.maximum(magnitude, 1e-30)
    normal, tangent = active_response_matrices(active, nx, ny)
    eigenvalues, eigenvectors = np.linalg.eigh(normal)
    threshold = max(float(eigenvalues[-1]), 1e-30) * \
        relative_eigenvalue_cutoff
    retained = eigenvalues > threshold

    def inverse(rhs):
        return eigenvectors[:, retained] @ (
            (eigenvectors[:, retained].T @ rhs) / eigenvalues[retained]
        )

    coefficients: list[np.ndarray] = []
    tangent_coefficients: list[np.ndarray] = []
    coefficients.append(inverse(1.0 - magnitude[active]))
    tangent_coefficients.append(tangent @ coefficients[0])
    if order >= 2:
        coefficients.append(inverse(-0.5 * tangent_coefficients[0] ** 2))
        tangent_coefficients.append(tangent @ coefficients[1])
    if order >= 3:
        coefficients.append(
            inverse(-tangent_coefficients[0] * tangent_coefficients[1])
        )
        tangent_coefficients.append(tangent @ coefficients[2])
    if order >= 4:
        coefficients.append(
            inverse(
                -tangent_coefficients[0] * tangent_coefficients[2]
                - 0.5 * tangent_coefficients[1] ** 2
                - 0.125 * tangent_coefficients[0] ** 4
            )
        )

    total = np.zeros_like(coefficients[0])
    result = []
    for index, coefficient in enumerate(coefficients, 1):
        total += coefficient
        source_x = np.zeros_like(p0x)
        source_y = np.zeros_like(p0y)
        source_x[active] = nx[active] * total
        source_y[active] = ny[active] * total
        correction_x, correction_y = active_coupling.transverse_project(
            source_x, source_y
        )
        corrected_norm = np.hypot(
            p0x + correction_x, p0y + correction_y
        )
        result.append(
            CurvatureTerm(
                order=index,
                coefficient_norm=float(np.linalg.norm(coefficient)),
                active_capacity_rms=float(
                    np.sqrt(np.mean((corrected_norm[active] - 1.0) ** 2))
                ),
                global_overload=float(np.mean(corrected_norm > 1.0 + 1e-10)),
                maximum_norm=float(np.max(corrected_norm)),
            )
        )
    return result


def run_case(
    name: str,
    size: int,
    pass_count: int,
    c: float,
    eta: float,
    terms: int,
    oracle_iterations: int,
) -> None:
    g = fourier.tree_experiment._load_image(name, size)
    trace, fluxes, objectives = fourier.bregman_trace(
        g, c, eta, max(32, pass_count + 16)
    )
    current = trace[pass_count]
    requested = c * (current - g)
    sequence = refreshed_angle_terms(
        current, fluxes[pass_count], g, c, objectives, terms
    )
    _, _, oracle = closest_disk_divergence(
        requested, iterations=oracle_iterations
    )

    print(f"\n{name} {size}x{size}, pass {pass_count}")
    print(
        "term alpha objective_gain equiv div_residual pre_overload pre_max"
    )
    baseline = fourier.objective(current, g, c)
    for item in sequence:
        print(
            f"{item.term:4d} {item.alpha:5.3f} "
            f"{baseline-item.objective:14.6g} "
            f"{item.equivalent_pass:>5s} "
            f"{item.divergence_residual:12.4f} "
            f"{item.preprojection_overload:12.1%} "
            f"{item.preprojection_maximum:7.3f}"
        )
    print(
        f"convex disk/divergence oracle: residual "
        f"{oracle.relative_residual:.4f}, max {oracle.maximum_residual:.4g}, "
        f"KKT fixed-point {oracle.fixed_point_error:.2e}, "
        f"saturated {oracle.saturated_fraction:.1%}"
    )

    if size <= 32:
        p0x, p0y = fourier.routed_preflux(
            current, fluxes[pass_count], g, c
        )
        full = fixed_normal_angle_radius(p0x, p0y)
        stable = fixed_normal_angle_radius(
            p0x, p0y, relative_eigenvalue_cutoff=0.1
        )
        print(
            f"full fixed-normal inverse: rank {full.rank}/{full.active}, "
            f"normal residual {full.normal_residual:.2e}, "
            f"|t|>=1 {full.tangent_outside_radius:.1%}, "
            f"|t| median/p95/max {full.tangent_median:.3g}/"
            f"{full.tangent_p95:.3g}/{full.tangent_maximum:.3g}"
        )
        print(
            f"radius-safe truncation: rank {stable.rank}/{stable.active}, "
            f"normal residual {stable.normal_residual:.3f}, "
            f"|t|>=1 {stable.tangent_outside_radius:.1%}, "
            f"max {stable.tangent_maximum:.3g}"
        )
        curvature = fixed_normal_curvature_terms(p0x, p0y)
        print(
            "curvature order: coefficient_norm active_rms "
            "global_overload maximum"
        )
        for item in curvature:
            print(
                f"{item.order:15d} {item.coefficient_norm:16.4g} "
                f"{item.active_capacity_rms:10.4g} "
                f"{item.global_overload:15.1%} "
                f"{item.maximum_norm:9.4g}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--pass-count", type=int, default=4)
    parser.add_argument("--terms", type=int, default=8)
    parser.add_argument("--oracle-iterations", type=int, default=5000)
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=0.10)
    parser.add_argument(
        "--images", nargs="+", default=["cameraman", "synthetic"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for size in args.sizes:
        for name in args.images:
            run_case(
                name,
                size,
                args.pass_count,
                args.c,
                args.eta,
                args.terms,
                args.oracle_iterations,
            )


if __name__ == "__main__":
    main()
