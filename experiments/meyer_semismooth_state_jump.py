#!/usr/bin/env python3
"""Semismooth deep jump on the actual fused Meyer state.

This is a falsification probe, not a public method.  It keeps the exact
reduced Split-Bregman state used by the optimized C++ spectral kernel,

    z = (u, w, t_u^x, t_u^y, t_w^x, t_w^y),

runs a short ordinary prefix so the two feasible dual routes are observed,
and applies fixed finite-horizon polynomials of the semismooth tangent map.
The only branch derivative is that of the Euclidean disk projection already
present in Split Bregman.  No image-content statistic or ownership rule is
introduced.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402
from experiments.meyer_deep_jump_causal_audit import (  # noqa: E402
    canonical_scene,
    pure_carrier_scene,
    pure_edge_scene,
)
from experiments.meyer_first_pass_conditioning import (  # noqa: E402
    checker_support_scene,
)
from experiments.meyer_preconditioning_research import (  # noqa: E402
    junction_texture_scene,
)
from experiments.meyer_tsv_validation import (  # noqa: E402
    multiscale_crossing_scene,
    symmetric_support_scene,
)


OUT = ROOT / "experiments" / "out" / "meyer_semismooth_state_jump"


def grad(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.roll(value, -1, axis=1) - value, np.roll(value, -1, axis=0) - value


def div(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return px - np.roll(px, 1, axis=1) + py - np.roll(py, 1, axis=0)


def laplacian_symbol(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    ky = 2.0 * np.pi * np.arange(height) / height
    kx = 2.0 * np.pi * np.arange(width) / width
    return (
        4.0 - 2.0 * np.cos(ky)[:, None] - 2.0 * np.cos(kx)[None, :]
    )


def solve_screened(
    rhs: np.ndarray, c: float, eta: float, symbol: np.ndarray
) -> np.ndarray:
    return np.fft.ifft2(np.fft.fft2(rhs) / (c + eta * symbol)).real


def project_disk(
    tx: np.ndarray, ty: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.hypot(tx, ty)
    scale = np.minimum(1.0, radius / np.maximum(magnitude, 1e-30))
    return scale * tx, scale * ty


def reflected(
    tx: np.ndarray, ty: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    px, py = project_disk(tx, ty, radius)
    return tx - 2.0 * px, ty - 2.0 * py


def project_disk_derivative(
    tx: np.ndarray,
    ty: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One Clarke derivative of the Euclidean disk projection."""

    magnitude = np.hypot(tx, ty)
    outside = magnitude > radius
    result_x = hx.copy()
    result_y = hy.copy()
    if np.any(outside):
        inverse = 1.0 / magnitude[outside]
        nx = tx[outside] * inverse
        ny = ty[outside] * inverse
        tangent = -ny * hx[outside] + nx * hy[outside]
        factor = radius * inverse
        result_x[outside] = factor * (-ny) * tangent
        result_y[outside] = factor * nx * tangent
    return result_x, result_y


@dataclass
class State:
    u: np.ndarray
    w: np.ndarray
    tux: np.ndarray
    tuy: np.ndarray
    twx: np.ndarray
    twy: np.ndarray

    def fields(self) -> tuple[np.ndarray, ...]:
        return self.u, self.w, self.tux, self.tuy, self.twx, self.twy


class ReducedMeyerMap:
    def __init__(self, image: np.ndarray, lam: float, mu: float):
        self.image = np.asarray(image, dtype=np.float64)
        self.shape = self.image.shape
        self.count = self.image.size
        self.lam = float(lam)
        self.mu = float(mu)
        self.cu = self.lam
        self.etau = 2.0 * self.lam
        self.cw = 1.0 / self.mu
        self.etaw = 10.0 / self.mu
        self.ru = 1.0 / self.etau
        self.rw = 1.0 / self.etaw
        self.symbol = laplacian_symbol(self.shape)

    def initial(self) -> State:
        u = solve_screened(
            self.cu * self.image, self.cu, self.etau, self.symbol
        )
        w = solve_screened(
            self.cw * (self.image - u), self.cw, self.etaw, self.symbol
        )
        tux, tuy = grad(u)
        twx, twy = grad(w)
        return State(u, w, tux, tuy, twx, twy)

    def step(self, state: State) -> State:
        rux, ruy = reflected(state.tux, state.tuy, self.ru)
        u = solve_screened(
            self.cu * (state.u + state.w) - self.etau * div(rux, ruy),
            self.cu,
            self.etau,
            self.symbol,
        )
        rwx, rwy = reflected(state.twx, state.twy, self.rw)
        w = solve_screened(
            self.cw * (self.image - u) - self.etaw * div(rwx, rwy),
            self.cw,
            self.etaw,
            self.symbol,
        )
        bux, buy = project_disk(state.tux, state.tuy, self.ru)
        bwx, bwy = project_disk(state.twx, state.twy, self.rw)
        gux, guy = grad(u)
        gwx, gwy = grad(w)
        return State(u, w, gux + bux, guy + buy, gwx + bwx, gwy + bwy)

    def tangent(self, state: State, direction: State) -> State:
        pux, puy = project_disk_derivative(
            state.tux, state.tuy, direction.tux, direction.tuy, self.ru
        )
        rux = direction.tux - 2.0 * pux
        ruy = direction.tuy - 2.0 * puy
        du = solve_screened(
            self.cu * (direction.u + direction.w)
            - self.etau * div(rux, ruy),
            self.cu,
            self.etau,
            self.symbol,
        )
        pwx, pwy = project_disk_derivative(
            state.twx, state.twy, direction.twx, direction.twy, self.rw
        )
        rwx = direction.twx - 2.0 * pwx
        rwy = direction.twy - 2.0 * pwy
        dw = solve_screened(
            -self.cw * du - self.etaw * div(rwx, rwy),
            self.cw,
            self.etaw,
            self.symbol,
        )
        gux, guy = grad(du)
        gwx, gwy = grad(dw)
        return State(du, dw, gux + pux, guy + puy, gwx + pwx, gwy + pwy)

    def pack(self, state: State) -> np.ndarray:
        return np.concatenate([field.ravel() for field in state.fields()])

    def unpack(self, vector: np.ndarray) -> State:
        fields = np.asarray(vector, dtype=np.float64).reshape(6, *self.shape)
        return State(*(field.copy() for field in fields))

    def subtract(self, left: State, right: State) -> State:
        return State(*(a - b for a, b in zip(left.fields(), right.fields())))

    def add_scaled(self, state: State, direction: State, alpha: float) -> State:
        return State(*(
            value + float(alpha) * delta
            for value, delta in zip(state.fields(), direction.fields())
        ))

    def retract_dual_graph(self, state: State) -> tuple[State, float]:
        """Project only the post-jump Bregman residuals back to feasibility.

        An exact pass ends with ``t = D primal + b`` and ``|b| <= radius``.
        A tangent polynomial preserves this only to first order.  This
        pointwise graph retraction restores it without a screened solve.
        """

        gux, guy = grad(state.u)
        gwx, gwy = grad(state.w)
        bux, buy = project_disk(
            state.tux - gux, state.tuy - guy, self.ru
        )
        bwx, bwy = project_disk(
            state.twx - gwx, state.twy - gwy, self.rw
        )
        retracted = State(
            state.u, state.w,
            gux + bux, guy + buy,
            gwx + bwx, gwy + bwy,
        )
        correction = rms(np.concatenate([
            (retracted.tux - state.tux).ravel(),
            (retracted.tuy - state.tuy).ravel(),
            (retracted.twx - state.twx).ravel(),
            (retracted.twy - state.twy).ravel(),
        ]))
        return retracted, correction

    def residual(self, state: State) -> State:
        return self.subtract(self.step(state), state)

    def residual_norm(self, state: State) -> float:
        residual = self.pack(self.residual(state))
        return float(np.linalg.norm(residual) / np.sqrt(residual.size))

    def newton_jump(
        self, state: State, krylov_steps: int, fixed_alpha: float | None = None
    ) -> tuple[State, dict[str, object]]:
        rhs_state = self.residual(state)
        rhs = self.pack(rhs_state)
        size = rhs.size

        def apply(vector: np.ndarray) -> np.ndarray:
            direction = self.unpack(vector)
            tangent = self.tangent(state, direction)
            return vector - self.pack(tangent)

        operator = LinearOperator((size, size), matvec=apply, dtype=np.float64)
        iteration_residuals: list[float] = []
        delta, info = gmres(
            operator,
            rhs,
            restart=int(krylov_steps),
            maxiter=1,
            rtol=0.0,
            atol=0.0,
            callback=lambda value: iteration_residuals.append(float(value)),
            callback_type="pr_norm",
        )
        direction = self.unpack(delta)
        before = self.residual_norm(state)
        accepted = state
        accepted_alpha = 0.0
        accepted_norm = before
        trials = []
        alpha_grid = (
            (float(fixed_alpha),)
            if fixed_alpha is not None
            else (1.0, 0.5, 0.25, 0.125, 0.0625)
        )
        for alpha in alpha_grid:
            candidate = self.add_scaled(state, direction, alpha)
            candidate_norm = self.residual_norm(candidate)
            trials.append({"alpha": alpha, "residual": candidate_norm})
            if fixed_alpha is not None or candidate_norm < accepted_norm:
                accepted = candidate
                accepted_alpha = alpha
                accepted_norm = candidate_norm
                break
        return accepted, {
            "gmres_info": int(info),
            "krylov_steps": len(iteration_residuals),
            "linear_relative_residuals": iteration_residuals,
            "alpha": accepted_alpha,
            "residual_before": before,
            "residual_after": accepted_norm,
            "trials": trials,
        }

    def finite_horizon_jump(
        self,
        state: State,
        krylov_steps: int,
        horizon: int,
        seed_state: State | None = None,
    ) -> tuple[State, dict[str, object]]:
        """Arnoldi compression of ``sum_{j=0}^{horizon-1} J^j r``.

        For the frozen local branch, the displacement after ``horizon``
        ordinary nonlinear passes is exactly this geometric action, where
        ``r=T(z)-z`` and ``J`` is the semismooth tangent of ``T`` at ``z``.
        Arnoldi applies that polynomial without solving for the stationary
        state and therefore does not overshoot finite fused time by design.
        """

        residual = self.pack(
            self.residual(state) if seed_state is None else seed_state
        )
        first_power = 0 if seed_state is None else 1
        beta = float(np.linalg.norm(residual))
        if not (beta > 0.0):
            return state, {
                "krylov_steps": 0,
                "horizon": int(horizon),
                "breakdown": True,
                "residual_before": 0.0,
                "residual_after": 0.0,
            }
        basis: list[np.ndarray] = [residual / beta]
        hessenberg = np.zeros((krylov_steps + 1, krylov_steps))
        breakdown = False
        for column in range(krylov_steps):
            tangent = self.pack(self.tangent(state, self.unpack(basis[column])))
            for row in range(column + 1):
                coefficient = float(basis[row] @ tangent)
                hessenberg[row, column] = coefficient
                tangent -= coefficient * basis[row]
            # A second modified Gram-Schmidt pass controls loss of
            # orthogonality for the strongly nonnormal coupled map.
            for row in range(column + 1):
                correction = float(basis[row] @ tangent)
                hessenberg[row, column] += correction
                tangent -= correction * basis[row]
            norm = float(np.linalg.norm(tangent))
            hessenberg[column + 1, column] = norm
            if column + 1 == krylov_steps:
                continue
            if norm <= 1e-13 * beta:
                breakdown = True
                hessenberg = hessenberg[: column + 2, : column + 1]
                break
            basis.append(tangent / norm)

        dimension = hessenberg.shape[1]
        square = hessenberg[:dimension, :dimension]
        unit = np.zeros(dimension)
        unit[0] = 1.0
        geometric = np.zeros(dimension)
        power = unit.copy()
        for exponent in range(first_power, first_power + int(horizon)):
            if exponent > 0:
                power = square @ power
            geometric += power
        displacement = beta * sum(
            geometric[index] * basis[index] for index in range(dimension)
        )
        candidate = self.add_scaled(state, self.unpack(displacement), 1.0)
        return candidate, {
            "krylov_steps": int(dimension),
            "horizon": int(horizon),
            "seed": (
                "exact_current_residual"
                if seed_state is None else "last_observed_increment"
            ),
            "breakdown": bool(breakdown),
            "projected_spectral_radius": float(
                np.max(np.abs(np.linalg.eigvals(square)))
            ),
            "residual_before": self.residual_norm(state),
            "residual_after": self.residual_norm(candidate),
        }


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2)))


def run_scene(
    scene: dict[str, object],
    *,
    lam: float,
    mu: float,
    prefix: int,
    krylov_steps: int,
    tail: int,
    threads: int,
    alpha: float | None,
    method: str,
    horizon: int,
    jumps: int,
    seed_mode: str,
    retract_dual: bool,
    array_sink: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    image = np.asarray(scene["source"], dtype=np.float64)
    model = ReducedMeyerMap(image, lam, mu)
    state = model.initial()
    previous_state = State(*(
        np.zeros_like(field) for field in state.fields()
    ))
    for _ in range(1, prefix):
        previous_state = state
        state = model.step(state)
    last_increment = model.subtract(state, previous_state)
    prefix_state = state
    diagnostics = []
    nonlinear_trial_cost = (
        1 if method == "newton" or seed_mode in ("residual", "transported")
        else 0
    )
    for _ in range(jumps):
        if method == "newton":
            state, diagnostic = model.newton_jump(
                state, krylov_steps, fixed_alpha=alpha
            )
        else:
            if seed_mode == "transported":
                jump_seed = model.tangent(state, last_increment)
            elif seed_mode == "last":
                jump_seed = last_increment
            else:
                jump_seed = None
            state, diagnostic = model.finite_horizon_jump(
                state,
                krylov_steps,
                horizon,
                seed_state=jump_seed,
            )
        if retract_dual:
            state, retraction = model.retract_dual_graph(state)
            diagnostic["dual_graph_retraction_rms"] = retraction
            diagnostic["residual_after_retraction"] = model.residual_norm(state)
        diagnostics.append(diagnostic)
        for _ in range(tail):
            previous_state = state
            state = model.step(state)
            last_increment = model.subtract(state, previous_state)

    fused64 = bfft.MeyerPlan(
        image.shape, lam=lam, mu=mu, passes=64, threads=threads
    ).split_legacy(image)
    hard = bfft.MeyerPlan(
        image.shape, lam=lam, mu=mu, passes=1, threads=threads
    ).split_jump_measure(image, virtual_passes=12)
    baseline_passes = prefix + jumps * (
        krylov_steps + nonlinear_trial_cost + tail
    )
    baseline = bfft.MeyerPlan(
        image.shape, lam=lam, mu=mu, passes=baseline_passes, threads=threads
    ).split_legacy(image)
    texture = image - state.u - state.w
    prefix_texture = image - prefix_state.u - prefix_state.w
    target_texture = fused64[1]
    if array_sink is not None:
        array_sink.update({
            "source": image,
            "fused_cartoon": image - target_texture,
            "fused_texture": target_texture,
            "hard_cartoon": hard[0],
            "hard_texture": hard[1],
            "jump_cartoon": image - texture,
            "jump_texture": texture,
            "baseline_cartoon": baseline[0],
            "baseline_texture": baseline[1],
            "jump_error": texture - target_texture,
            "hard_error": hard[1] - target_texture,
        })
    return {
        "name": str(scene["name"]),
        "cost_model": {
            "prefix_passes": prefix,
            "krylov_tangent_steps": krylov_steps,
            "nonlinear_residual_trials_charged_as_passes": nonlinear_trial_cost,
            "tail_passes": tail,
            "jump_count": jumps,
            "baseline_passes": baseline_passes,
        },
        "jumps": diagnostics,
        "texture_error_rms_to_fused64": {
            "prefix": rms(prefix_texture - target_texture),
            "equal_cost_baseline": rms(baseline[1] - target_texture),
            "semismooth_jump": rms(texture - target_texture),
            "hard_jump": rms(hard[1] - target_texture),
        },
        "jump_vs_equal_cost_ratio": rms(texture - target_texture)
        / max(rms(baseline[1] - target_texture), 1e-30),
        "jump_recomposition_linf": float(np.max(np.abs(
            image - (image - texture) - texture
        ))),
    }


def render_atlas(
    rows: list[tuple[str, dict[str, np.ndarray]]], path: Path
) -> None:
    titles = (
        "source", "fused-64 cartoon", "hard cartoon", "flow-jump cartoon",
        "fused-64 texture", "hard texture", "flow-jump texture",
        "hard minus fused", "flow jump minus fused",
    )
    figure, axes = plt.subplots(len(rows), len(titles), figsize=(14.2, 9.0))
    for row, (name, arrays) in enumerate(rows):
        signed = (
            arrays["fused_texture"], arrays["hard_texture"],
            arrays["jump_texture"], arrays["hard_error"],
            arrays["jump_error"],
        )
        limit = max(
            1.0,
            max(float(np.percentile(np.abs(value), 99.5)) for value in signed),
        )
        panels = (
            (arrays["source"], "gray", 0.0, 255.0),
            (arrays["fused_cartoon"], "gray", 0.0, 255.0),
            (arrays["hard_cartoon"], "gray", 0.0, 255.0),
            (arrays["jump_cartoon"], "gray", 0.0, 255.0),
            *((value, "coolwarm", -limit, limit) for value in signed),
        )
        for column, (value, cmap, low, high) in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(value, cmap=cmap, vmin=low, vmax=high,
                        interpolation="nearest")
            if row == 0:
                axis.set_title(titles[column], fontsize=8.2)
            if column == 0:
                axis.set_ylabel(name.replace("_", " "), fontsize=8.0)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout(pad=0.35, w_pad=0.15, h_pad=0.2)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=40.0)
    parser.add_argument("--prefix", type=int, default=4)
    parser.add_argument("--krylov-steps", type=int, default=4)
    parser.add_argument("--tail", type=int, default=0)
    parser.add_argument(
        "--method", choices=("horizon", "newton"), default="horizon"
    )
    parser.add_argument(
        "--horizon", type=int, default=60,
        help="remaining fused passes represented by the frozen tangent flow",
    )
    parser.add_argument("--jumps", type=int, default=1)
    parser.add_argument(
        "--seed-mode",
        choices=("last", "transported", "residual"),
        default="last",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="fixed finite-horizon fraction of the stationary correction",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--retract-dual", action="store_true",
        help="pointwise post-jump retraction t=D(primal)+Pi(t-D(primal))",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.prefix < 1 or args.krylov_steps < 1 or args.tail < 0:
        raise ValueError("prefix and Krylov steps must be positive; tail nonnegative")
    if args.horizon < 1 or args.jumps < 1:
        raise ValueError("horizon and jump count must be positive")
    if args.alpha is not None and not (0.0 < args.alpha <= 1.0):
        raise ValueError("alpha must lie in (0,1]")
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = (
        pure_edge_scene(args.size),
        pure_carrier_scene(args.size),
        canonical_scene(symmetric_support_scene(args.size)),
        canonical_scene(checker_support_scene(args.size)),
        canonical_scene(multiscale_crossing_scene(args.size)),
        canonical_scene(junction_texture_scene(args.size)),
    )
    reports = []
    atlas_rows = []
    archive: dict[str, np.ndarray] = {}
    for scene in scenes:
        arrays: dict[str, np.ndarray] = {}
        item = run_scene(
            scene,
            lam=args.lam,
            mu=args.mu,
            prefix=args.prefix,
            krylov_steps=args.krylov_steps,
            tail=args.tail,
            threads=args.threads,
            alpha=args.alpha,
            method=args.method,
            horizon=args.horizon,
            jumps=args.jumps,
            seed_mode=args.seed_mode,
            retract_dual=args.retract_dual,
            array_sink=arrays,
        )
        reports.append(item)
        atlas_rows.append((str(scene["name"]), arrays))
        for key, value in arrays.items():
            archive[f"{scene['name']}_{key}"] = np.asarray(
                value, dtype=np.float32
            )
    report = {
        "claim_under_test": (
            "Short fused prefixes followed by finite-horizon semismooth "
            "state jumps can replace more ordinary passes at equal operator "
            "cost without a content law."
        ),
        "parameters": vars(args) | {"out": str(args.out)},
        "scenes": reports,
    }
    path = args.out / "results.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.out / "arrays.npz", **archive)
    render_atlas(atlas_rows, args.out / "atlas.png")
    for item in reports:
        errors = item["texture_error_rms_to_fused64"]
        last_jump = item["jumps"][-1]
        step_label = (
            f"alpha={last_jump['alpha']:.3f}"
            if "alpha" in last_jump
            else f"rho={last_jump['projected_spectral_radius']:.3f}"
        )
        print(
            f"{item['name']:20s} {step_label} "
            f"base={errors['equal_cost_baseline']:.5f} "
            f"jump={errors['semismooth_jump']:.5f} "
            f"ratio={item['jump_vs_equal_cost_ratio']:.4f} "
            f"hard={errors['hard_jump']:.5f}"
        )


if __name__ == "__main__":
    main()
