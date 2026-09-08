"""Transported zonotopic-mixture enrichment for the CONV--ITD synthetic.

This is an isolated experiment.  The initial auxiliary law is approximated by
a symmetric Gauss--Hermite mixture of nested bounded zonotopes.  A fixed
deterministic bank of orthogonalized Rademacher generators supplies
in-component cubature.  At
each ITD level the noise profiles are themselves split by the same CONV--ITD
baseline/rotation operator; the transported rotations enrich the signal.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.antithetic_itd_enrichment import conv_itd_baseline
from experiments.conv_itd_extrema_ablation import conv_current_extrema
from experiments.conv_itd_multicomponent import make_signal
from experiments.conv_itd_vs_canonical import representation_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments" / "out" / "conv_itd_zonotopic_enrichment"


def gaussian_radial_modes(order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Absolute Gauss--Hermite nodes with paired standard-normal weights."""

    node, weight = np.polynomial.hermite_e.hermegauss(order)
    weight = weight/np.sqrt(2.0*np.pi)
    magnitude: list[float] = []
    mass: list[float] = []
    for value in sorted(set(np.round(np.abs(node), 14))):
        selected = np.isclose(np.abs(node), value, rtol=0.0, atol=1e-13)
        magnitude.append(float(value))
        mass.append(float(np.sum(weight[selected])))
    return np.asarray(magnitude), np.asarray(mass)


def rademacher_generators(n: int, count: int, seed: int) -> np.ndarray:
    """Deterministic orthogonalized Rademacher generators with unit RMS."""

    rng = np.random.default_rng(seed)
    profiles = rng.choice((-1.0, 1.0), size=(n, count))
    # Remove the DC component because ITD should not spend an enrichment
    # direction translating the entire record.  QR prevents duplicate or
    # nearly duplicate directions in this small deterministic cubature bank.
    profiles -= np.mean(profiles, axis=0, keepdims=True)
    q, _ = np.linalg.qr(profiles, mode="reduced")
    return q*np.sqrt(n)


def plain_decompose(signal: np.ndarray, max_levels: int) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, float]]]:
    state = np.asarray(signal, dtype=np.float64).copy()
    rotations: list[np.ndarray] = []
    records: list[dict[str, float]] = []
    for level in range(max_levels):
        count = int(conv_current_extrema(state)[0].size-2)
        if count < 2:
            break
        next_state = conv_itd_baseline(state)
        rotations.append(state-next_state)
        records.append({
            "level": level+1,
            "signal_input_extrema": count,
            "signal_output_extrema": int(conv_current_extrema(next_state)[0].size-2),
            "noise_rotation_rms": 0.0,
        })
        state = next_state
    return rotations, state, records


def enriched_decompose(
    signal: np.ndarray,
    *,
    alpha: float,
    probe_count: int,
    max_levels: int,
    transported: bool,
    normalize_rotation: bool,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, float]]]:
    """Decompose with static or recursively transported auxiliary measure."""

    state = np.asarray(signal, dtype=np.float64).copy()
    original_probe = rademacher_generators(state.size, probe_count, seed)
    coarse_probe = original_probe.copy()
    magnitude, mass = gaussian_radial_modes(5)
    rotations: list[np.ndarray] = []
    records: list[dict[str, float]] = []

    for level in range(max_levels):
        signal_extrema = int(conv_current_extrema(state)[0].size-2)
        if signal_extrema < 2:
            break

        source_probe = coarse_probe if transported else original_probe
        noise_rotation = np.empty_like(source_probe)
        next_coarse = np.empty_like(source_probe)
        for j in range(probe_count):
            baseline_probe = conv_itd_baseline(source_probe[:, j])
            noise_rotation[:, j] = source_probe[:, j]-baseline_probe
            next_coarse[:, j] = baseline_probe

        rms = np.sqrt(np.mean(noise_rotation*noise_rotation, axis=0))
        usable = rms > 64*np.finfo(float).eps
        if not np.any(usable):
            break
        profile = noise_rotation[:, usable].copy()
        if normalize_rotation:
            profile /= rms[usable][None, :]

        scale = alpha*np.std(state)
        accumulated = np.zeros_like(state)
        total_weight = 0.0
        # The zero Gauss--Hermite node contributes the unperturbed baseline.
        for radius, radial_weight in zip(magnitude, mass):
            if radius == 0.0:
                accumulated += radial_weight*conv_itd_baseline(state)
                total_weight += radial_weight
                continue
            direction_weight = radial_weight/profile.shape[1]
            for j in range(profile.shape[1]):
                displacement = scale*radius*profile[:, j]
                accumulated += 0.5*direction_weight*(
                    conv_itd_baseline(state+displacement)
                    + conv_itd_baseline(state-displacement)
                )
                total_weight += direction_weight
        next_state = accumulated/total_weight
        rotations.append(state-next_state)
        records.append({
            "level": level+1,
            "signal_input_extrema": signal_extrema,
            "signal_output_extrema": int(conv_current_extrema(next_state)[0].size-2),
            "noise_rotation_rms": float(np.sqrt(np.mean(noise_rotation*noise_rotation))),
            "usable_noise_profiles": int(np.sum(usable)),
            "injected_rms_at_unit_radius": float(scale*np.sqrt(np.mean(profile*profile))),
            "mixture_weight_sum": total_weight,
        })
        state = next_state
        if transported:
            coarse_probe = next_coarse
    return rotations, state, records


def evaluate(
    rotations: list[np.ndarray], baseline: np.ndarray, truth: dict[str, np.ndarray]
) -> tuple[dict[str, object], np.ndarray]:
    metrics, correlation = representation_metrics(rotations, baseline, truth)
    reconstruction = baseline+np.sum(rotations, axis=0)
    metrics["rotation_count"] = len(rotations)
    metrics["closure_linf"] = float(np.max(np.abs(reconstruction-(baseline+np.sum(rotations, axis=0)))))
    return metrics, correlation


def run(
    *, n: int,
    probe_count: int,
    max_levels: int,
    alphas: tuple[float, ...],
    seed: int,
    out: Path,
    render: bool = True,
) -> dict[str, object]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/conv_itd_mpl")
    if render:
        import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    t, signal, truth = make_signal(n)
    methods: dict[str, tuple[list[np.ndarray], np.ndarray, list[dict[str, float]], float]] = {}

    start = time.perf_counter()
    rotations, baseline, levels = plain_decompose(signal, max_levels)
    methods["plain CONV-ITD"] = (rotations, baseline, levels, time.perf_counter()-start)

    for alpha in alphas:
        variants = (
            ("static", False, True),
            ("transported", True, False),
            ("transported normalized", True, True),
        )
        for label, transported, normalized in variants:
            start = time.perf_counter()
            rotations, baseline, levels = enriched_decompose(
                signal,
                alpha=alpha,
                probe_count=probe_count,
                max_levels=max_levels,
                transported=transported,
                normalize_rotation=normalized,
                seed=seed,
            )
            methods[f"{label} a={alpha:g}"] = (
                rotations, baseline, levels, time.perf_counter()-start
            )

    report: dict[str, object] = {
        "configuration": {
            "sample_count": n,
            "probe_count": probe_count,
            "max_levels": max_levels,
            "alphas": list(alphas),
            "seed": seed,
            "radial_modes": gaussian_radial_modes(5)[0].tolist(),
            "radial_weights": gaussian_radial_modes(5)[1].tolist(),
        },
        "methods": {},
    }
    correlations: dict[str, np.ndarray] = {}
    for name, (rotations, baseline, levels, elapsed) in methods.items():
        metrics, correlation = evaluate(rotations, baseline, truth)
        # Closure against the original signal, retained separately from the
        # tautological local expression used in ``evaluate``.
        reconstruction = baseline+np.sum(rotations, axis=0)
        metrics["closure_linf"] = float(np.max(np.abs(reconstruction-signal)))
        metrics["elapsed_seconds"] = elapsed
        metrics["levels"] = levels
        report["methods"][name] = metrics
        correlations[name] = correlation

    (out/"metrics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    np.savez_compressed(
        out/"decompositions.npz",
        t=t,
        signal=signal,
        **{
            f"{name.replace(' ', '_').replace('=', '_').replace('.', '_')}_{kind}": value
            for name, (rotation, baseline, _, _) in methods.items()
            for kind, value in (("rotations", np.stack(rotation)), ("baseline", baseline))
        },
    )

    if render:
        method_names = list(methods)
        fig, axes = plt.subplots(len(method_names), 1, figsize=(15, 2.0*len(method_names)), sharex=True, constrained_layout=True)
        if len(method_names) == 1:
            axes = [axes]
        for ax, name in zip(axes, method_names):
            rotations, baseline, _, _ = methods[name]
            ax.plot(t, signal, color=".84", lw=.45)
            ax.plot(t, baseline, color="#166534", lw=1.0, label="final baseline")
            ax.plot(t, truth["trend"], "k--", lw=.8, label="truth trend")
            ax.set_title(name, loc="left", fontsize=9)
        axes[0].legend(ncol=3, fontsize=7)
        axes[-1].set_xlabel("normalized time")
        fig.savefig(out/"baseline_comparison.png", dpi=180)
        plt.close(fig)

        truth_names = next(iter(report["methods"].values()))["component_names"]
        fig, axes = plt.subplots(1, len(method_names), figsize=(3.2*len(method_names), 5.2), constrained_layout=True)
        if len(method_names) == 1:
            axes = [axes]
        for ax, name in zip(axes, method_names):
            image = ax.imshow(correlations[name], vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
            ax.set_title(name, fontsize=8)
            ax.set_xticks(range(len(truth_names)), truth_names, rotation=70, ha="right", fontsize=6)
            ax.set_yticks(range(correlations[name].shape[0]), [f"R{i+1}" for i in range(correlations[name].shape[0])], fontsize=6)
        fig.colorbar(image, ax=axes, label="Pearson correlation", shrink=.7)
        fig.savefig(out/"component_correlations.png", dpi=180)
        plt.close(fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--alphas", default="0.05,0.1")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run(
        n=args.n,
        probe_count=args.probes,
        max_levels=args.levels,
        alphas=tuple(float(value) for value in args.alphas.split(",")),
        seed=args.seed,
        out=args.out,
        render=not args.no_plots,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
