#!/usr/bin/env python3
"""Synthetic-truth research into finite Meyer state preconditioning.

The first Gilles--Osher pass has no nonlinear state.  In a texture interior
its cartoon side is therefore the screened multiplier

    H(omega) = lambda / (lambda + 2 lambda L(omega)).

Repeated early passes largely discover that the texture should be removed
from the cartoon.  ``virtual_power_split`` takes that *linear interior tail*
in one spectral multiplication, blocks it at a high-certainty symmetric-
variation gate, and performs one conditioned cartoon solve.  It is a finite
preconditioner: there is no runtime candidate scan and ``virtual_passes`` is
an exponent, not a loop.

Every quality source below is analytic and authored in this file or in
``meyer_tsv_validation.py``.  Photographs and inherited gallery images are
deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import bfft
from experiments.meyer_first_pass_conditioning import (
    checker_support_scene,
    lap_hat,
    normalize_gate,
    predicted_reflection,
    screened,
    tsv_one_forward,
)
from experiments.meyer_tsv_validation import (
    _dilate,
    _gradient_magnitude,
    multiscale_crossing_scene,
    score_split,
    symmetric_support_scene,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out" / "meyer_preconditioning"


def junction_texture_scene(size: int = 256) -> dict:
    """Thin authentic geometry crossing two known oscillatory supports."""
    y, x = np.mgrid[:size, :size].astype(np.float64)
    xn, yn = x / size, y / size
    smooth_cartoon = 101.0 + 9.0 * xn - 5.0 * yn

    # A thin bent object and a sharp diamond are deliberately dangerous:
    # both have relatively cheap divergence representations and can be
    # mistaken for G-texture even by a converged Meyer split.
    diagonal = np.abs(yn - (0.18 + 0.62 * xn)) < 0.012
    upright = (np.abs(xn - 0.72) < 0.012) & (yn > 0.26) & (yn < 0.82)
    diamond = np.abs(xn - 0.28) + 1.15 * np.abs(yn - 0.68) < 0.13
    authored_jump = np.zeros_like(smooth_cartoon)
    authored_jump[diagonal | upright] += 73.0
    authored_jump[diamond] -= 58.0
    hard_composition = smooth_cartoon + authored_jump

    r0 = np.sqrt(((xn - 0.66) / 0.27) ** 2 + ((yn - 0.30) / 0.20) ** 2)
    support0 = np.clip((1.0 - r0) / 0.16, 0.0, 1.0)
    distance = np.minimum.reduce((
        xn - 0.10, 0.53 - xn, yn - 0.43, 0.91 - yn,
    ))
    support1 = np.clip(distance / 0.045, 0.0, 1.0)
    material_texture = 14.0 * support0 * np.cos(
        2.0 * np.pi * (x - 1.4 * y) / 6.0 + 0.37
    )
    material_texture += 18.0 * support1 * np.cos(
        2.0 * np.pi * (x + 0.31 * y) / 15.0 - 0.62
    )
    jump_mean = float(np.mean(authored_jump))
    jump_potential = authored_jump - jump_mean
    symbol = lap_hat(smooth_cartoon.shape)
    first_cartoon_resolvent = 1.0 / (1.0 - 2.0 * symbol)
    smooth_jump = np.fft.ifft2(
        np.fft.fft2(jump_potential) * first_cartoon_resolvent
    ).real
    boundary_texture = jump_potential - smooth_jump
    cartoon = smooth_cartoon + jump_mean + smooth_jump
    texture = boundary_texture + material_texture
    source = cartoon + texture
    contour = _dilate(_gradient_magnitude(hard_composition) > 4.0, 3)
    interior = ((support0 > 0.995) | (support1 > 0.995))
    interior &= ~_dilate(contour, 5)
    return {
        "name": "junction_texture",
        "source": source,
        "cartoon": cartoon,
        "smooth_cartoon": smooth_cartoon,
        "hard_composition": hard_composition,
        "jump_potential": jump_potential,
        "smooth_jump": smooth_jump,
        "boundary_texture": boundary_texture,
        "texture": texture,
        "material_texture": material_texture,
        "fine_support": support0,
        "contour": contour,
        "texture_interior": interior,
    }


def structural_gate(source: np.ndarray) -> np.ndarray:
    """Frozen high-certainty structural tail used by both preconditioners."""
    return normalize_gate(tsv_one_forward(source)) ** 6


def conditioned_cartoon(
    source: np.ndarray,
    gate: np.ndarray,
    *,
    lam: float,
    strength: float,
) -> np.ndarray:
    eta = 2.0 * lam
    rx, ry = predicted_reflection(source, eta)
    return screened(
        source,
        lam,
        eta,
        (strength * gate * rx, strength * gate * ry),
    )


def virtual_power_split(
    source: np.ndarray,
    *,
    lam: float = 0.05,
    virtual_passes: int = 8,
    gate_power: float = 8.0,
    strength: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jump the linear texture-interior tail, then solve geometry once.

    ``H**virtual_passes`` is evaluated directly.  The spatial gate makes the
    seed nonstationary, so the final conditioned solve is essential: it
    restores PDE consistency instead of alpha-blending two finished images.
    """
    source = np.asarray(source, dtype=np.float64)
    gate = structural_gate(source)
    eta = 2.0 * lam
    transfer = lam / (lam - eta * lap_hat(source.shape))
    virtual_cartoon = np.fft.ifft2(
        np.fft.fft2(source) * transfer ** int(virtual_passes)
    ).real
    texture_confidence = (1.0 - gate) ** float(gate_power)
    seed = texture_confidence * (source - virtual_cartoon)
    target = source - seed
    cartoon = conditioned_cartoon(
        target, gate, lam=lam, strength=strength
    )
    # Once the seeded residual lies in the intended G-ball, retaining the
    # first linear texture-side survivor only re-attenuates it.  The exact
    # residual is the finite projected-state readout under test.
    return cartoon, source - cartoon, gate


def hodge_disk_projection(
    value: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, dict]:
    """One-shot feasible G-ball readout through a longitudinal Hodge lift.

    If ``value = div(p)`` and ``|p(x)| <= radius``, then ``value`` is in the
    discrete Meyer G-ball of that radius.  The Fourier lift below is the
    minimum-L2 longitudinal field with the requested divergence.  Radial
    disk projection may change its divergence, but the returned divergence
    is immediately and constructively feasible--there is no optimizer or
    convergence assumption in that statement.
    """
    value = np.asarray(value, dtype=np.float64)
    h, w = value.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(w)[None, :]
    ky = 2.0 * np.pi * np.fft.fftfreq(h)[:, None]
    dx = 1.0 - np.exp(-1j * kx)
    dy = 1.0 - np.exp(-1j * ky)
    denominator = np.abs(dx) ** 2 + np.abs(dy) ** 2
    safe = np.where(denominator > 0.0, denominator, 1.0)
    spectrum = np.fft.fft2(value - np.mean(value))
    px_hat = np.conj(dx) * spectrum / safe
    py_hat = np.conj(dy) * spectrum / safe
    px_hat[0, 0] = 0.0
    py_hat[0, 0] = 0.0
    px = np.fft.ifft2(px_hat).real
    py = np.fft.ifft2(py_hat).real
    before = np.hypot(px, py)
    scale = np.minimum(1.0, float(radius) / np.maximum(before, 1e-30))
    px *= scale
    py *= scale
    projected = (
        px - np.roll(px, 1, axis=1)
        + py - np.roll(py, 1, axis=0)
    )
    return projected, {
        "radius": float(radius),
        "preprojection_overload_fraction": float(np.mean(before > radius)),
        "preprojection_maximum": float(np.max(before)),
        "postprojection_maximum": float(np.max(np.hypot(px, py))),
        "constructively_g_feasible": bool(
            np.max(np.hypot(px, py)) <= radius * (1.0 + 1e-12)
        ),
    }


def virtual_hodge_split(
    source: np.ndarray,
    *,
    lam: float = 0.05,
    mu: float = 40.0,
    virtual_passes: int = 8,
    gate_power: float = 8.0,
    strength: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Selected finite preconditioner with a constructive G constraint."""
    _cartoon, proposed_texture, _gate = virtual_power_split(
        source,
        lam=lam,
        virtual_passes=virtual_passes,
        gate_power=gate_power,
        strength=strength,
    )
    texture, diagnostic = hodge_disk_projection(proposed_texture, mu)
    return np.asarray(source, dtype=np.float64) - texture, texture, diagnostic


def balanced_band_seed(source: np.ndarray) -> np.ndarray:
    """Falsification control: fixed Littlewood--Paley-like local bands.

    This is intentionally retained beside the stronger virtual-state method:
    it shows that simply finding locally sign-balanced bands recovers texture
    but also mistakes edge halos for oscillation.
    """
    source = np.asarray(source, dtype=np.float64)
    h, w = source.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(h)[:, None]
    kx = 2.0 * np.pi * np.fft.rfftfreq(w)[None, :]
    radius2 = kx * kx + ky * ky
    source_hat = np.fft.rfft2(source)

    def smooth(value: np.ndarray, sigma: float) -> np.ndarray:
        return np.fft.irfft2(
            np.fft.rfft2(value) * np.exp(-0.5 * sigma * sigma * radius2),
            s=source.shape,
        )

    previous = source
    seed = np.zeros_like(source)
    for sigma in (1.0, 2.0, 4.0, 8.0, 16.0):
        low = np.fft.irfft2(
            source_hat * np.exp(-0.5 * sigma * sigma * radius2),
            s=source.shape,
        )
        band = previous - low
        previous = low
        positive = np.maximum(smooth(np.maximum(band, 0.0), 2.0 * sigma), 0.0)
        negative = np.maximum(smooth(np.maximum(-band, 0.0), 2.0 * sigma), 0.0)
        balance = 2.0 * np.minimum(positive, negative) / np.maximum(
            positive + negative, 1e-12
        )
        seed += band * balance * balance
    return seed


def band_seed_split(
    source: np.ndarray,
    *,
    lam: float = 0.05,
    gate_power: float = 4.0,
    strength: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    gate = structural_gate(source)
    seed = balanced_band_seed(source) * (1.0 - gate) ** gate_power
    cartoon = conditioned_cartoon(
        source - seed, gate, lam=lam, strength=strength
    )
    return cartoon, source - cartoon


def native_splits(
    scene: dict,
    passes: tuple[int, ...],
    *,
    lam: float,
    mu: float,
    threads: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    source = np.asarray(scene["source"], dtype=np.float64)
    wanted = set(passes)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    plan = bfft.MeyerPlan(
        source.shape,
        lam=lam,
        mu=mu,
        passes=max(wanted),
        rung_sweeps=1,
        rung_tol=0.0,
        threads=threads,
    )

    def visit(number, cartoon, texture):
        if number in wanted:
            result[f"pass{number}"] = (cartoon.copy(), texture.copy())

    plan.visit(source, visit)
    result["conditioned1"] = plan.split_conditioned_first(source)
    return result


def oracle_ablation(scene: dict, lam: float, mu: float) -> dict:
    """Identify which exact hidden states are needed by one outer pass."""
    source = scene["source"]
    truth_texture = scene["texture"]
    eta_u = 2.0 * lam
    c_v, eta_v = 1.0 / mu, 10.0 / mu

    def linear_tail(cartoon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        remainder = source - cartoon
        survivor = screened(remainder, c_v, eta_v)
        return cartoon, remainder - survivor

    cold = linear_tail(screened(source, lam, eta_u))
    # Texture seed is exact, but the cartoon and texture-side dual capacity
    # fields are still cold.
    truth_seed = linear_tail(screened(source - truth_texture, lam, eta_u))
    # Cartoon state is exact, but the cold texture-side projection still
    # attenuates the known feasible texture.
    truth_cartoon = linear_tail(scene["cartoon"].copy())
    exact_states = (scene["cartoon"].copy(), truth_texture.copy())
    return {
        name: score_split(*split, scene)
        for name, split in (
            ("cold", cold),
            ("exact_texture_seed_only", truth_seed),
            ("exact_cartoon_state_only", truth_cartoon),
            ("both_projector_states_exact", exact_states),
        )
    }


def response_atlas(
    size: int,
    *,
    lam: float,
    mu: float,
    threads: int,
) -> list[dict]:
    """Measure the nonlinear descent against exact pure-carrier truth."""
    y, x = np.mgrid[:size, :size].astype(np.float64)
    rows = []
    for period in (4, 6, 8, 12, 16, 24, 32, 48):
        for amplitude in (4.0, 16.0, 64.0):
            truth = amplitude * np.cos(
                2.0 * np.pi * (x + 0.375 * y) / period + 0.23
            )
            source = 128.0 + truth
            gains = {}
            plan = bfft.MeyerPlan(
                source.shape,
                lam=lam,
                mu=mu,
                passes=8,
                rung_sweeps=1,
                rung_tol=0.0,
                threads=threads,
            )
            denominator = float(np.sum(truth * truth))

            def visit(number, _cartoon, texture):
                if number in (1, 2, 4, 8):
                    gains[str(number)] = float(
                        np.sum(texture * truth) / denominator
                    )

            plan.visit(source, visit)
            rows.append({
                "period": period,
                "amplitude": amplitude,
                "texture_gain": gains,
            })
    return rows


def render(scene: dict, splits: dict, path: Path) -> None:
    arrays = [("source", scene["source"])]
    for name in ("pass1", "conditioned1", "virtual8_hodge", "pass64"):
        cartoon, texture = splits[name]
        arrays.extend(((f"{name} cartoon", cartoon), (f"{name} texture", texture)))
    panels = []
    for name, value in arrays:
        shown = np.clip(127.5 + 2.0 * value, 0.0, 255.0) \
            if "texture" in name else np.clip(value, 0.0, 255.0)
        image = Image.fromarray(shown.astype(np.uint8), mode="L").convert("RGB")
        panel = Image.new("RGB", (image.width, image.height + 22), "white")
        panel.paste(image, (0, 22))
        ImageDraw.Draw(panel).text((4, 4), name, fill="black")
        panels.append(panel)
    output = Image.new(
        "RGB", (3 * panels[0].width, 3 * panels[0].height), "white"
    )
    for index, panel in enumerate(panels):
        output.paste(
            panel,
            ((index % 3) * panel.width, (index // 3) * panel.height),
        )
    output.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=40.0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    scenes = (
        symmetric_support_scene(args.size),
        multiscale_crossing_scene(args.size),
        checker_support_scene(args.size),
        junction_texture_scene(args.size),
    )
    report = {
        "quality_sources": "authored analytic truth only",
        "selected_preconditioner": {
            "virtual_passes": 8,
            "gate_power": 8.0,
            "runtime_scan": False,
        },
        "scenes": {},
        "response_atlas": response_atlas(
            min(args.size, 128),
            lam=args.lam,
            mu=args.mu,
            threads=args.threads,
        ),
    }
    for scene in scenes:
        source = scene["source"]
        t0 = time.perf_counter()
        splits = native_splits(
            scene, (1, 4, 64),
            lam=args.lam,
            mu=args.mu,
            threads=args.threads,
        )
        native_ms = 1000.0 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        splits["band_seed"] = band_seed_split(source, lam=args.lam)
        band_ms = 1000.0 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        splits["virtual8_gate8"] = virtual_power_split(
            source, lam=args.lam, virtual_passes=8, gate_power=8.0
        )[:2]
        virtual_ms = 1000.0 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        splits["virtual12_gate8"] = virtual_power_split(
            source, lam=args.lam, virtual_passes=12, gate_power=8.0
        )[:2]
        virtual12_ms = 1000.0 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        virtual_hodge = virtual_hodge_split(
            source,
            lam=args.lam,
            mu=args.mu,
            virtual_passes=8,
            gate_power=8.0,
        )
        splits["virtual8_hodge"] = virtual_hodge[:2]
        virtual_hodge_ms = 1000.0 * (time.perf_counter() - t0)

        scores = {
            name: score_split(*split, scene)
            for name, split in splits.items()
        }
        report["scenes"][scene["name"]] = {
            "scores": scores,
            "oracle_ablation": oracle_ablation(scene, args.lam, args.mu),
            "virtual8_hodge_diagnostic": virtual_hodge[2],
            "timing_milliseconds": {
                "native_trace_1_to_64": native_ms,
                "python_band_seed": band_ms,
                "python_virtual8_gate8": virtual_ms,
                "python_virtual12_gate8": virtual12_ms,
                "python_virtual8_hodge": virtual_hodge_ms,
            },
        }
        print(f"\n{scene['name']}")
        print("method               tex_gain  tex_error  contour_leak")
        for name in (
            "pass1", "conditioned1", "band_seed", "virtual8_gate8",
            "virtual8_hodge", "virtual12_gate8", "pass4", "pass64",
        ):
            score = scores[name]
            print(
                f"{name:20s} "
                f"{score['interior_texture_gain']:8.3f}  "
                f"{score['interior_texture_relative_rms_error']:9.3f}  "
                f"{score['contour_excess_texture_rms']:12.3f}"
            )
        render(scene, splits, OUT / f"{scene['name']}.png")

    path = OUT / "results.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
