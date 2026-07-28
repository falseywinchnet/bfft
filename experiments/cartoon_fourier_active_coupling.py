#!/usr/bin/env python3
"""Fixed-capacity-mask Schur experiment for Fourier/Hodge ROF closure.

The Hodge closure first produces a flux p0 with the requested divergence.
Where |p0| > 1, let N sample the outward normal component and let P_T be the
periodic divergence-free projector.  The exact fixed-normal coupling is

    S = N P_T N*,
    delta_p = P_T N* S^+ (1 - |p0|).

S is built explicitly from the inverse-FFT Green tensor of P_T.  The dense
solve is an oracle only: the point is to measure locality/rank and determine
whether an analytical block decomposition exists.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import label

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cartoon_fourier_transport as fourier  # noqa: E402
import meyer_bregman as meyer  # noqa: E402


@dataclass
class SpectrumSummary:
    active: int
    active_fraction: float
    components: int
    largest_component: int
    off_component_fraction: float
    rank: int
    condition: float
    rank90: int
    rank99: int
    local_fractions: dict[int, float]


def transverse_kernel(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    h, w = shape
    kx = 2.0 * np.pi * np.fft.fftfreq(w)
    ky = 2.0 * np.pi * np.fft.fftfreq(h)
    bx = 1.0 - np.exp(-1j * kx)[None, :]
    by = 1.0 - np.exp(-1j * ky)[:, None]
    denominator = np.abs(bx) ** 2 + np.abs(by) ** 2
    safe = np.where(denominator > 0.0, denominator, 1.0)
    pxx = 1.0 - np.abs(bx) ** 2 / safe
    pxy = -np.conj(bx) * by / safe
    pyx = -np.conj(by) * bx / safe
    pyy = 1.0 - np.abs(by) ** 2 / safe
    # The zero-frequency vector field is harmonic and divergence-free.
    pxx[0, 0], pxy[0, 0] = 1.0, 0.0
    pyx[0, 0], pyy[0, 0] = 0.0, 1.0
    return tuple(np.fft.ifft2(symbol).real for symbol in (pxx, pxy, pyx, pyy))


def active_schur(
    active: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = active.shape
    y, x = np.nonzero(active)
    ax = nx[active]
    ay = ny[active]
    kxx, kxy, kyx, kyy = transverse_kernel(active.shape)
    dy = (y[:, None] - y[None, :]) % h
    dx = (x[:, None] - x[None, :]) % w
    schur = (
        ax[:, None] * ax[None, :] * kxx[dy, dx]
        + ax[:, None] * ay[None, :] * kxy[dy, dx]
        + ay[:, None] * ax[None, :] * kyx[dy, dx]
        + ay[:, None] * ay[None, :] * kyy[dy, dx]
    )
    schur = 0.5 * (schur + schur.T)
    return schur, y, x


def active_vector_schur(
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = active.shape
    y, x = np.nonzero(active)
    kxx, kxy, kyx, kyy = transverse_kernel(active.shape)
    dy = (y[:, None] - y[None, :]) % h
    dx = (x[:, None] - x[None, :]) % w
    schur = np.block(
        [
            [kxx[dy, dx], kxy[dy, dx]],
            [kyx[dy, dx], kyy[dy, dx]],
        ]
    )
    return 0.5 * (schur + schur.T), y, x


def spectral_inverse(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    threshold = max(float(eigenvalues[-1]), 1e-30) * 1e-8
    coefficients = eigenvectors.T @ rhs
    coefficients = np.where(
        eigenvalues > threshold,
        coefficients / np.maximum(eigenvalues, threshold),
        0.0,
    )
    return eigenvectors @ coefficients, eigenvalues, eigenvectors


def summarize_schur(
    schur: np.ndarray,
    active: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> tuple[SpectrumSummary, np.ndarray, np.ndarray]:
    structure = np.ones((3, 3), dtype=np.int8)
    component_map, component_count = label(active, structure=structure)
    component_ids = component_map[y, x]
    component_sizes = np.bincount(component_ids)[1:]
    same_component = component_ids[:, None] == component_ids[None, :]
    total_squared = float(np.sum(schur * schur))
    off_component_fraction = np.sqrt(
        float(np.sum(schur[~same_component] ** 2)) / max(total_squared, 1e-30)
    )

    eigenvalues, eigenvectors = np.linalg.eigh(schur)
    maximum = max(float(eigenvalues[-1]), 1e-30)
    retained = eigenvalues > maximum * 1e-8
    positive = eigenvalues[retained]
    condition = maximum / max(float(positive[0]), 1e-30)
    energy = np.maximum(eigenvalues, 0.0) ** 2
    descending = np.sort(energy)[::-1]
    cumulative = np.cumsum(descending) / max(float(np.sum(descending)), 1e-30)
    rank90 = int(np.searchsorted(cumulative, 0.90) + 1)
    rank99 = int(np.searchsorted(cumulative, 0.99) + 1)

    h, w = active.shape
    ddy = np.minimum((y[:, None] - y[None, :]) % h, (y[None, :] - y[:, None]) % h)
    ddx = np.minimum((x[:, None] - x[None, :]) % w, (x[None, :] - x[:, None]) % w)
    distance = np.maximum(ddy, ddx)
    local_fractions = {}
    for radius in (1, 2, 4, 8):
        local_fractions[radius] = np.sqrt(
            float(np.sum(schur[distance <= radius] ** 2))
            / max(total_squared, 1e-30)
        )

    return (
        SpectrumSummary(
            active=schur.shape[0],
            active_fraction=schur.shape[0] / active.size,
            components=component_count,
            largest_component=int(np.max(component_sizes, initial=0)),
            off_component_fraction=off_component_fraction,
            rank=int(np.count_nonzero(retained)),
            condition=condition,
            rank90=rank90,
            rank99=rank99,
            local_fractions=local_fractions,
        ),
        eigenvalues,
        eigenvectors,
    )


def transverse_project(
    px: np.ndarray,
    py: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, tx, ty = fourier.hodge_split(px, py)
    return tx, ty


def active_tensor_spectrum(
    active: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
) -> tuple[dict[int, float], dict[int, float]]:
    fields = (
        active * nx * nx,
        active * nx * ny,
        active * ny * ny,
    )
    energy = sum(
        np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2 for field in fields
    )
    total = max(float(np.sum(energy)), 1e-30)
    h, w = active.shape
    cy, cx = h // 2, w // 2
    low = {}
    for radius in (1, 2, 4, 8, 16, 32):
        if radius > min(h, w) // 2:
            continue
        low[radius] = float(
            np.sum(
                energy[
                    cy - radius : cy + radius + 1,
                    cx - radius : cx + radius + 1,
                ]
            )
            / total
        )
    descending = np.sort(energy.ravel())[::-1]
    cumulative = np.cumsum(descending) / total
    top = {
        count: float(cumulative[count - 1])
        for count in (16, 64, 256, 1024)
        if count <= cumulative.size
    }
    return low, top


def fixed_mask_oracle(
    p0x: np.ndarray,
    p0y: np.ndarray,
    active: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    schur: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.hypot(p0x, p0y)
    rhs = 1.0 - magnitude[active]
    threshold = max(float(eigenvalues[-1]), 1e-30) * 1e-8
    coefficients = eigenvectors.T @ rhs
    coefficients = np.where(
        eigenvalues > threshold, coefficients / np.maximum(eigenvalues, threshold), 0.0
    )
    multiplier = eigenvectors @ coefficients
    source_x = np.zeros_like(p0x)
    source_y = np.zeros_like(p0y)
    source_x[active] = nx[active] * multiplier
    source_y[active] = ny[active] * multiplier
    correction_x, correction_y = transverse_project(source_x, source_y)
    return p0x + correction_x, p0y + correction_y


def fixed_direction_oracle(
    p0x: np.ndarray,
    p0y: np.ndarray,
    active: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    schur: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(np.count_nonzero(active))
    rhs = np.concatenate(
        [nx[active] - p0x[active], ny[active] - p0y[active]]
    )
    multiplier, eigenvalues, _ = spectral_inverse(schur, rhs)
    source_x = np.zeros_like(p0x)
    source_y = np.zeros_like(p0y)
    source_x[active] = multiplier[:count]
    source_y[active] = multiplier[count:]
    correction_x, correction_y = transverse_project(source_x, source_y)
    return p0x + correction_x, p0y + correction_y, eigenvalues


def diagonal_capacity_drop(
    p0x: np.ndarray,
    p0y: np.ndarray,
    active: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    schur_diagonal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One local Schur inverse plus an analytical global Rayleigh gain."""
    magnitude = np.hypot(p0x, p0y)
    rhs = 1.0 - magnitude[active]
    multiplier = rhs / np.maximum(schur_diagonal, 1e-12)
    source_x = np.zeros_like(p0x)
    source_y = np.zeros_like(p0y)
    source_x[active] = nx[active] * multiplier
    source_y[active] = ny[active] * multiplier
    correction_x, correction_y = transverse_project(source_x, source_y)
    response = (
        nx[active] * correction_x[active]
        + ny[active] * correction_y[active]
    )
    gain = float(
        np.clip(
            np.vdot(response, rhs).real
            / max(float(np.vdot(response, response).real), 1e-30),
            0.0,
            1.0,
        )
    )
    return p0x + gain * correction_x, p0y + gain * correction_y, gain


def causal_capacity_events(
    p0x: np.ndarray,
    p0y: np.ndarray,
    *,
    radius: int | None,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Strongest-first capacity events; each initially overloaded pixel once."""
    h, w = p0x.shape
    kxx, kxy, kyx, kyy = transverse_kernel(p0x.shape)
    px = p0x.copy()
    py = p0y.copy()
    initial = np.hypot(px, py)
    order = np.argsort(initial.ravel())[::-1]
    order = order[initial.ravel()[order] > 1.0 + 1e-10]
    accepted: list[int] = []
    for flat in order:
        y, x = divmod(int(flat), w)
        magnitude = math.hypot(float(px[y, x]), float(py[y, x]))
        if magnitude <= 1.0 + 1e-10:
            continue
        nx = px[y, x] / magnitude
        ny = py[y, x] / magnitude
        self_response = (
            nx * nx * kxx[0, 0]
            + nx * ny * (kxy[0, 0] + kyx[0, 0])
            + ny * ny * kyy[0, 0]
        )
        multiplier = (1.0 - magnitude) / max(self_response, 1e-12)
        if radius is None:
            px += multiplier * (
                nx * np.roll(kxx, (y, x), axis=(0, 1))
                + ny * np.roll(kxy, (y, x), axis=(0, 1))
            )
            py += multiplier * (
                nx * np.roll(kyx, (y, x), axis=(0, 1))
                + ny * np.roll(kyy, (y, x), axis=(0, 1))
            )
        else:
            for dy in range(-radius, radius + 1):
                target_y = (y + dy) % h
                kernel_y = dy % h
                for dx in range(-radius, radius + 1):
                    target_x = (x + dx) % w
                    kernel_x = dx % w
                    px[target_y, target_x] += multiplier * (
                        nx * kxx[kernel_y, kernel_x]
                        + ny * kxy[kernel_y, kernel_x]
                    )
                    py[target_y, target_x] += multiplier * (
                        nx * kyx[kernel_y, kernel_x]
                        + ny * kyy[kernel_y, kernel_x]
                    )
        accepted.append(int(flat))

    final_norm = np.hypot(px, py)
    reopened = (
        float(np.mean(final_norm.ravel()[accepted] > 1.0 + 1e-10))
        if accepted
        else 0.0
    )
    divergence_change = float(
        np.linalg.norm(meyer.div(px - p0x, py - p0y))
    )
    return px, py, len(accepted), reopened, divergence_change


def disk_primal(
    px: np.ndarray,
    py: np.ndarray,
    g: np.ndarray,
    c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.maximum(1.0, np.hypot(px, py))
    feasible_x = px / scale
    feasible_y = py / scale
    return g + meyer.div(feasible_x, feasible_y) / c, feasible_x, feasible_y


def equivalent_pass(value: float, objectives: np.ndarray) -> str:
    indices = np.flatnonzero(objectives <= value)
    return str(int(indices[0] + 1)) if indices.size else f">{objectives.size}"


def run_case(name: str, size: int, pass_count: int, c: float, eta: float) -> None:
    g = fourier.tree_experiment._load_image(name, size)
    trace, fluxes, objectives = fourier.bregman_trace(
        g, c, eta, max(32, pass_count + 16)
    )
    current = trace[pass_count]
    p0x, p0y = fourier.routed_preflux(current, fluxes[pass_count], g, c)
    magnitude = np.hypot(p0x, p0y)
    active = magnitude > 1.0 + 1e-10
    nx = p0x / np.maximum(magnitude, 1e-30)
    ny = p0y / np.maximum(magnitude, 1e-30)
    schur, y, x = active_schur(active, nx, ny)
    summary, eigenvalues, eigenvectors = summarize_schur(schur, active, y, x)
    low_spectrum, top_spectrum = active_tensor_spectrum(active, nx, ny)

    baseline_raw, _, _ = disk_primal(p0x, p0y, g, c)
    baseline, baseline_alpha = fourier._segment_taylor_drop(
        current, baseline_raw, g, c
    )
    oracle_x, oracle_y = fixed_mask_oracle(
        p0x,
        p0y,
        active,
        nx,
        ny,
        schur,
        eigenvalues,
        eigenvectors,
    )
    oracle_pre_norm = np.hypot(oracle_x, oracle_y)
    oracle_raw, feasible_x, feasible_y = disk_primal(oracle_x, oracle_y, g, c)
    oracle, oracle_alpha = fourier._segment_taylor_drop(
        current, oracle_raw, g, c
    )
    divergence_change = np.linalg.norm(
        meyer.div(oracle_x - p0x, oracle_y - p0y)
    )
    normal_residual = np.max(
        np.abs(
            nx[active] * oracle_x[active]
            + ny[active] * oracle_y[active]
            - 1.0
        ),
        initial=0.0,
    )

    print(f"\n{name} {size}x{size}, pass {pass_count}")
    print(
        f"active {summary.active} ({summary.active_fraction:.1%}), "
        f"components {summary.components}, largest {summary.largest_component}"
    )
    print(
        f"S rank {summary.rank}/{summary.active}, condition {summary.condition:.2e}, "
        f"rank90 {summary.rank90}, rank99 {summary.rank99}"
    )
    print(
        f"off-component Frobenius {summary.off_component_fraction:.3f}; "
        + ", ".join(
            f"r<={radius}: {fraction:.3f}"
            for radius, fraction in summary.local_fractions.items()
        )
    )
    print(
        "active-tensor spectral energy: "
        + ", ".join(f"low r<={r}: {v:.3f}" for r, v in low_spectrum.items())
        + "; "
        + ", ".join(f"top {k}: {v:.3f}" for k, v in top_spectrum.items())
    )
    print(
        f"fixed-mask certificate: div change {divergence_change:.2e}, "
        f"normal residual {normal_residual:.2e}"
    )
    print(
        f"after correction: overload {np.mean(oracle_pre_norm > 1.0 + 1e-10):.1%}, "
        f"max norm {np.max(oracle_pre_norm):.3f}, "
        f"post-project changed {np.mean(np.hypot(feasible_x, feasible_y) >= 1-1e-12):.1%}"
    )
    baseline_objective = fourier.objective(baseline, g, c)
    oracle_objective = fourier.objective(oracle, g, c)
    print(
        f"Hodge alpha {baseline_alpha:.3f}, equiv {equivalent_pass(baseline_objective, objectives)}; "
        f"Schur alpha {oracle_alpha:.3f}, equiv {equivalent_pass(oracle_objective, objectives)}; "
        f"objective gain {baseline_objective-oracle_objective:.6g}"
    )

    vector_schur, _, _ = active_vector_schur(active)
    vector_x, vector_y, vector_eigenvalues = fixed_direction_oracle(
        p0x, p0y, active, nx, ny, vector_schur
    )
    vector_norm = np.hypot(vector_x, vector_y)
    vector_raw, _, _ = disk_primal(vector_x, vector_y, g, c)
    vector, vector_alpha = fourier._segment_taylor_drop(
        current, vector_raw, g, c
    )
    vector_objective = fourier.objective(vector, g, c)
    vector_maximum = max(float(vector_eigenvalues[-1]), 1e-30)
    vector_positive = vector_eigenvalues[
        vector_eigenvalues > vector_maximum * 1e-8
    ]
    direction_residual = max(
        np.max(np.abs(vector_x[active] - nx[active]), initial=0.0),
        np.max(np.abs(vector_y[active] - ny[active]), initial=0.0),
    )
    print(
        f"vector S rank {vector_positive.size}/{vector_schur.shape[0]}, "
        f"condition {vector_maximum/max(float(vector_positive[0]),1e-30):.2e}; "
        f"direction residual {direction_residual:.2e}"
    )
    print(
        f"fixed-direction overload {np.mean(vector_norm > 1.0 + 1e-10):.1%}, "
        f"max norm {np.max(vector_norm):.3f}; alpha {vector_alpha:.3f}, "
        f"equiv {equivalent_pass(vector_objective, objectives)}, "
        f"gain vs Hodge {baseline_objective-vector_objective:.6g}"
    )

    diagonal_x, diagonal_y, diagonal_gain = diagonal_capacity_drop(
        p0x, p0y, active, nx, ny, np.diag(schur)
    )
    diagonal_norm = np.hypot(diagonal_x, diagonal_y)
    diagonal_raw, _, _ = disk_primal(diagonal_x, diagonal_y, g, c)
    diagonal, diagonal_alpha = fourier._segment_taylor_drop(
        current, diagonal_raw, g, c
    )
    diagonal_objective = fourier.objective(diagonal, g, c)
    print(
        f"diagonal Schur gain {diagonal_gain:.3f}, "
        f"overload {np.mean(diagonal_norm > 1.0 + 1e-10):.1%}, "
        f"max norm {np.max(diagonal_norm):.3f}; alpha {diagonal_alpha:.3f}, "
        f"equiv {equivalent_pass(diagonal_objective, objectives)}, "
        f"gain vs Hodge {baseline_objective-diagonal_objective:.6g}"
    )

    for radius, event_name in ((2, "event-r2"), (None, "event-full")):
        event_x, event_y, event_count, reopened, event_divergence = (
            causal_capacity_events(p0x, p0y, radius=radius)
        )
        event_norm = np.hypot(event_x, event_y)
        event_raw, _, _ = disk_primal(event_x, event_y, g, c)
        event, event_alpha = fourier._segment_taylor_drop(
            current, event_raw, g, c
        )
        event_objective = fourier.objective(event, g, c)
        print(
            f"{event_name}: accepted {event_count}, reopened {reopened:.1%}, "
            f"overload {np.mean(event_norm > 1.0 + 1e-10):.1%}, "
            f"div change {event_divergence:.2e}; alpha {event_alpha:.3f}, "
            f"equiv {equivalent_pass(event_objective, objectives)}, "
            f"gain vs Hodge {baseline_objective-event_objective:.6g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--pass-count", type=int, default=4)
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=0.10)
    parser.add_argument("--images", nargs="+", default=["cameraman", "synthetic"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.images:
        run_case(name, args.size, args.pass_count, args.c, args.eta)


if __name__ == "__main__":
    main()
