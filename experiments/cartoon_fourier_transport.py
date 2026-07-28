#!/usr/bin/env python3
"""Can the fixed-geometry Fourier transport tail be taken in one drop?

For periodic isotropic ROF Split Bregman, freeze the spatial projection
branch.  The remaining longitudinal transport is diagonal in Fourier space.
If ``D_k = FFT(u_k-u_{k-1})``, its saturated-branch recurrence multiplier is

    r(omega) = eta*L(omega) / (c + eta*L(omega)),

where L is the positive symbol of -Laplacian.  The whole remaining geometric
tail is therefore ``D_k*r/(1-r)``.  This experiment tests that exact frozen
symbol and a phase-coherent shell estimate of r.  Each is a finite proposal,
not an iterative solver, and is accepted only through the original isotropic
ROF objective.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cartoon_characteristic_tree as tree_experiment  # noqa: E402
import meyer_bregman as meyer  # noqa: E402


@dataclass
class Drop:
    u: np.ndarray
    alpha: float
    raw_objective: float
    accepted_objective: float
    coherent_power: float
    median_ratio: float


@dataclass
class HodgeDiagnostic:
    transverse_energy: float
    longitudinal_overload: float
    longitudinal_maximum: float
    transverse_reference_cosine: float


def objective(u: np.ndarray, g: np.ndarray, c: float) -> float:
    gx, gy = meyer.grad(u)
    return float(np.sum(np.hypot(gx, gy)) + 0.5 * c * np.sum((u - g) ** 2))


def _segment_taylor_drop(
    current: np.ndarray,
    proposal: np.ndarray,
    g: np.ndarray,
    c: float,
) -> tuple[np.ndarray, float]:
    """One one-sided second-order drop along a Fourier proposal."""
    direction = proposal - current
    if float(np.vdot(direction, direction).real) <= 1e-28:
        return current, 0.0
    gx, gy = meyer.grad(current)
    dx, dy = meyer.grad(direction)
    magnitude = np.hypot(gx, gy)
    tolerance = 1e-10 * max(float(np.ptp(current)), 1.0)
    moving = magnitude > tolerance
    dot = gx * dx + gy * dy
    first = c * float(np.vdot(current - g, direction).real)
    first += float(np.sum(dot[moving] / magnitude[moving]))
    # Exact right derivative of |grad u + alpha grad d| at zero grad u.
    first += float(np.sum(np.hypot(dx[~moving], dy[~moving])))
    cross = gx * dy - gy * dx
    second = c * float(np.vdot(direction, direction).real)
    second += float(np.sum(cross[moving] ** 2 / magnitude[moving] ** 3))
    if first >= 0.0:
        return current, 0.0
    alpha = float(np.clip(-first / max(second, 1e-30), 0.0, 1.0))
    candidate = current + alpha * direction
    baseline_objective = float(
        np.sum(magnitude) + 0.5 * c * np.sum((current - g) ** 2)
    )
    candidate_objective = float(
        np.sum(np.hypot(gx + alpha * dx, gy + alpha * dy))
        + 0.5 * c * np.sum((current + alpha * direction - g) ** 2)
    )
    if candidate_objective < baseline_objective:
        return candidate, alpha
    return current, 0.0


def frozen_symbol_proposal(
    previous: np.ndarray,
    current: np.ndarray,
    c: float,
    eta: float,
) -> np.ndarray:
    """Sum the exact geometric tail for a frozen projection branch."""
    increment = np.fft.rfft2(current - previous)
    laplacian = -meyer.lap_hat(current.shape)
    # r/(1-r) simplifies exactly to eta*L/c; no pole or tolerance.
    remaining = increment * (eta * laplacian / c)
    return np.fft.irfft2(np.fft.rfft2(current) + remaining, s=current.shape)


def shell_limit_proposal(
    older: np.ndarray,
    previous: np.ndarray,
    current: np.ndarray,
    *,
    shells: int = 64,
    coherence_floor: float = 0.85,
    ratio_ceiling: float = 0.98,
) -> tuple[np.ndarray, float, float]:
    """Estimate a real recurrence ratio on equal-symbol Fourier shells.

    Only phase-coherent shells are transported.  The current increment keeps
    its complex phase; the estimate changes magnitude only.
    """
    d0 = np.fft.rfft2(previous - older)
    d1 = np.fft.rfft2(current - previous)
    laplacian = -meyer.lap_hat(current.shape)
    shell_index = np.minimum(
        (shells * laplacian / max(float(np.max(laplacian)), 1e-30)).astype(
            np.int64
        ),
        shells - 1,
    )
    weights = np.full(d0.shape, 2.0)
    weights[:, 0] = 1.0
    if current.shape[1] % 2 == 0:
        weights[:, -1] = 1.0

    ratio_by_shell = np.zeros(shells)
    coherent = np.zeros(shells, dtype=bool)
    shell_power = np.zeros(shells)
    for shell in range(shells):
        selected = shell_index == shell
        if not np.any(selected):
            continue
        a = d0[selected]
        b = d1[selected]
        w = weights[selected]
        power0 = float(np.sum(w * np.abs(a) ** 2))
        power1 = float(np.sum(w * np.abs(b) ** 2))
        cross = float(np.sum(w * np.real(b * np.conj(a))))
        shell_power[shell] = power1
        coherence = cross / max(np.sqrt(power0 * power1), 1e-30)
        ratio = cross / max(power0, 1e-30)
        if coherence >= coherence_floor and 0.0 < ratio < ratio_ceiling:
            ratio_by_shell[shell] = ratio
            coherent[shell] = True

    ratio = ratio_by_shell[shell_index]
    future_factor = ratio / np.maximum(1.0 - ratio, 1e-30)
    proposal_hat = np.fft.rfft2(current) + d1 * future_factor
    proposal = np.fft.irfft2(proposal_hat, s=current.shape)
    total_power = float(np.sum(shell_power))
    coherent_power = float(np.sum(shell_power[coherent])) / max(
        total_power, 1e-30
    )
    used_ratios = ratio_by_shell[coherent]
    median_ratio = float(np.median(used_ratios)) if used_ratios.size else 0.0
    return proposal, coherent_power, median_ratio


def make_drop(
    raw: np.ndarray,
    current: np.ndarray,
    g: np.ndarray,
    c: float,
    coherent_power: float = 1.0,
    median_ratio: float = 0.0,
) -> Drop:
    accepted, alpha = _segment_taylor_drop(current, raw, g, c)
    return Drop(
        u=accepted,
        alpha=alpha,
        raw_objective=objective(raw, g, c),
        accepted_objective=objective(accepted, g, c),
        coherent_power=coherent_power,
        median_ratio=median_ratio,
    )


def bregman_trace(
    g: np.ndarray,
    c: float,
    eta: float,
    passes: int,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    state = meyer.RofState(g.shape)
    images = [g.copy()]
    fluxes = [(np.zeros_like(g), np.zeros_like(g))]
    objectives = []
    for _ in range(passes):
        u, state = meyer.rof_sb(g, c, eta=eta, state=state, sweeps=1)
        images.append(u.copy())
        fluxes.append((eta * state.bx.copy(), eta * state.by.copy()))
        objectives.append(objective(u, g, c))
    return images, fluxes, np.asarray(objectives)


def hodge_split(
    px: np.ndarray,
    py: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact periodic longitudinal/transverse split of a vector field."""
    h, w = px.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(w)
    ky = 2.0 * np.pi * np.fft.fftfreq(h)
    # div symbol for p-roll(p,1).
    bx = 1.0 - np.exp(-1j * kx)[None, :]
    by = 1.0 - np.exp(-1j * ky)[:, None]
    px_hat = np.fft.fft2(px)
    py_hat = np.fft.fft2(py)
    divergence_hat = bx * px_hat + by * py_hat
    denominator = np.abs(bx) ** 2 + np.abs(by) ** 2
    safe = np.where(denominator > 0.0, denominator, 1.0)
    longitudinal_x_hat = np.conj(bx) * divergence_hat / safe
    longitudinal_y_hat = np.conj(by) * divergence_hat / safe
    longitudinal_x_hat[0, 0] = 0.0
    longitudinal_y_hat[0, 0] = 0.0
    lx = np.fft.ifft2(longitudinal_x_hat).real
    ly = np.fft.ifft2(longitudinal_y_hat).real
    return lx, ly, px - lx, py - ly


def longitudinal_flux(divergence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-L2 periodic flux having the requested zero-mean divergence."""
    h, w = divergence.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(w)
    ky = 2.0 * np.pi * np.fft.fftfreq(h)
    bx = 1.0 - np.exp(-1j * kx)[None, :]
    by = 1.0 - np.exp(-1j * ky)[:, None]
    denominator = np.abs(bx) ** 2 + np.abs(by) ** 2
    safe = np.where(denominator > 0.0, denominator, 1.0)
    divergence_hat = np.fft.fft2(divergence - np.mean(divergence))
    px_hat = np.conj(bx) * divergence_hat / safe
    py_hat = np.conj(by) * divergence_hat / safe
    px_hat[0, 0] = 0.0
    py_hat[0, 0] = 0.0
    return np.fft.ifft2(px_hat).real, np.fft.ifft2(py_hat).real


def routed_preflux(
    desired_u: np.ndarray,
    current_flux: tuple[np.ndarray, np.ndarray],
    g: np.ndarray,
    c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Add the minimum-energy longitudinal divergence correction to a route."""
    desired_divergence = c * (desired_u - g)
    mismatch = desired_divergence - meyer.div(*current_flux)
    mismatch -= np.mean(mismatch)
    positive_laplacian = -meyer.lap_hat(g.shape)
    safe = np.where(positive_laplacian > 0.0, positive_laplacian, 1.0)
    potential_hat = -np.fft.rfft2(mismatch) / safe
    potential_hat[0, 0] = 0.0
    potential = np.fft.irfft2(potential_hat, s=g.shape)
    correction_x, correction_y = meyer.grad(potential)
    px = current_flux[0] + correction_x
    py = current_flux[1] + correction_y
    return px, py


def routed_flux_proposal(
    desired_u: np.ndarray,
    current_flux: tuple[np.ndarray, np.ndarray],
    g: np.ndarray,
    c: float,
) -> np.ndarray:
    """Fourier longitudinal drop + retained transverse route + one disk hit."""
    px, py = routed_preflux(desired_u, current_flux, g, c)
    magnitude = np.maximum(1.0, np.hypot(px, py))
    px /= magnitude
    py /= magnitude
    return g + meyer.div(px, py) / c


def hodge_diagnostic(
    flux: tuple[np.ndarray, np.ndarray],
    reference_flux: tuple[np.ndarray, np.ndarray],
) -> HodgeDiagnostic:
    px, py = flux
    lx, ly, tx, ty = hodge_split(px, py)
    _, _, reference_tx, reference_ty = hodge_split(*reference_flux)
    full_energy = float(np.sum(px * px + py * py))
    transverse_energy = float(np.sum(tx * tx + ty * ty)) / max(
        full_energy, 1e-30
    )
    longitudinal_norm = np.hypot(lx, ly)
    longitudinal_overload = float(np.mean(longitudinal_norm > 1.0))
    numerator = float(np.sum(tx * reference_tx + ty * reference_ty))
    denominator = np.sqrt(
        float(np.sum(tx * tx + ty * ty))
        * float(np.sum(reference_tx**2 + reference_ty**2))
    )
    return HodgeDiagnostic(
        transverse_energy=transverse_energy,
        longitudinal_overload=longitudinal_overload,
        longitudinal_maximum=float(np.max(longitudinal_norm)),
        transverse_reference_cosine=numerator / max(denominator, 1e-30),
    )


def equivalent_pass(value: float, objectives: np.ndarray) -> str:
    indices = np.flatnonzero(objectives <= value)
    return str(int(indices[0] + 1)) if indices.size else f">{objectives.size}"


def run_case(
    name: str,
    size: int,
    c: float,
    eta: float,
    checkpoints: list[int],
    trace_passes: int,
    reference_iterations: int,
) -> None:
    g = tree_experiment._load_image(name, size)
    trace, fluxes, objectives = bregman_trace(g, c, eta, trace_passes)
    reference, reference_state = meyer.rof_fgp(g, c, reference_iterations)
    reference_objective = objective(reference, g, c)
    print(f"\n{name} {size}x{size}, c={c:g}, eta={eta:g}")
    print(
        "pass method    alpha  coherent  median_r  excess       "
        "base_excess  equiv  accepted"
    )
    for pass_count in checkpoints:
        current = trace[pass_count]
        base_objective = objectives[pass_count - 1]

        hodge_raw = routed_flux_proposal(
            current, fluxes[pass_count], g, c
        )
        hodge = make_drop(hodge_raw, current, g, c)
        print(
            f"{pass_count:4d} hodge   {hodge.alpha:6.3f}  "
            f"{1.0:8.3f}  {0.0:8.3f}  "
            f"{hodge.accepted_objective-reference_objective:10.4g}  "
            f"{base_objective-reference_objective:11.4g}  "
            f"{equivalent_pass(hodge.accepted_objective, objectives):>5s}  "
            f"{'yes' if hodge.alpha > 0 else 'no'}"
        )

        symbol_raw = frozen_symbol_proposal(
            trace[pass_count - 1], current, c, eta
        )
        symbol = make_drop(symbol_raw, current, g, c)
        print(
            f"{pass_count:4d} symbol  {symbol.alpha:6.3f}  "
            f"{symbol.coherent_power:8.3f}  {symbol.median_ratio:8.3f}  "
            f"{symbol.accepted_objective-reference_objective:10.4g}  "
            f"{base_objective-reference_objective:11.4g}  "
            f"{equivalent_pass(symbol.accepted_objective, objectives):>5s}  "
            f"{'yes' if symbol.alpha > 0 else 'no'}"
        )
        routed_symbol_raw = routed_flux_proposal(
            symbol_raw, fluxes[pass_count], g, c
        )
        routed_symbol = make_drop(routed_symbol_raw, current, g, c)
        print(
            f"{pass_count:4d} routeS  {routed_symbol.alpha:6.3f}  "
            f"{1.0:8.3f}  {0.0:8.3f}  "
            f"{routed_symbol.accepted_objective-reference_objective:10.4g}  "
            f"{base_objective-reference_objective:11.4g}  "
            f"{equivalent_pass(routed_symbol.accepted_objective, objectives):>5s}  "
            f"{'yes' if routed_symbol.alpha > 0 else 'no'}"
        )

        if pass_count >= 2:
            shell_raw, coherent_power, median_ratio = shell_limit_proposal(
                trace[pass_count - 2],
                trace[pass_count - 1],
                current,
            )
            shell = make_drop(
                shell_raw,
                current,
                g,
                c,
                coherent_power,
                median_ratio,
            )
            print(
                f"{pass_count:4d} shell   {shell.alpha:6.3f}  "
                f"{shell.coherent_power:8.3f}  {shell.median_ratio:8.3f}  "
                f"{shell.accepted_objective-reference_objective:10.4g}  "
                f"{base_objective-reference_objective:11.4g}  "
                f"{equivalent_pass(shell.accepted_objective, objectives):>5s}  "
                f"{'yes' if shell.alpha > 0 else 'no'}"
            )
            routed_shell_raw = routed_flux_proposal(
                shell_raw, fluxes[pass_count], g, c
            )
            routed_shell = make_drop(routed_shell_raw, current, g, c)
            print(
                f"{pass_count:4d} routeE  {routed_shell.alpha:6.3f}  "
                f"{coherent_power:8.3f}  {median_ratio:8.3f}  "
                f"{routed_shell.accepted_objective-reference_objective:10.4g}  "
                f"{base_objective-reference_objective:11.4g}  "
                f"{equivalent_pass(routed_shell.accepted_objective, objectives):>5s}  "
                f"{'yes' if routed_shell.alpha > 0 else 'no'}"
            )

    print(
        "pass  transverse_E  long>|1|  max|long|  "
        "transverse-reference cosine"
    )
    reference_flux = (reference_state.px, reference_state.py)
    for pass_count in checkpoints:
        diagnostic = hodge_diagnostic(fluxes[pass_count], reference_flux)
        print(
            f"{pass_count:4d}  {diagnostic.transverse_energy:12.3f}  "
            f"{diagnostic.longitudinal_overload:8.3f}  "
            f"{diagnostic.longitudinal_maximum:9.3f}  "
            f"{diagnostic.transverse_reference_cosine:27.3f}"
        )
    reference_diagnostic = hodge_diagnostic(reference_flux, reference_flux)
    print(
        f"{'ref':>4s}  {reference_diagnostic.transverse_energy:12.3f}  "
        f"{reference_diagnostic.longitudinal_overload:8.3f}  "
        f"{reference_diagnostic.longitudinal_maximum:9.3f}  "
        f"{reference_diagnostic.transverse_reference_cosine:27.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=0.10)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--trace-passes", type=int, default=128)
    parser.add_argument("--reference-iterations", type=int, default=8000)
    parser.add_argument("--images", nargs="+", default=["cameraman", "synthetic"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.images:
        run_case(
            name,
            args.size,
            args.c,
            args.eta,
            args.checkpoints,
            args.trace_passes,
            args.reference_iterations,
        )


if __name__ == "__main__":
    main()
