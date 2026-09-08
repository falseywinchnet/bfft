#!/usr/bin/env python3
"""Exact defect decomposition of the fixed hard jump against fused Meyer.

This is a diagnosis, not a replacement operator.  Authored truth is used only
to name the error terms after both native methods have run.

Let ``R = H**K`` be the virtual cartoon resolvent and ``P = I-R``.  Take the
64-pass fused texture ``v*`` as the target and fold its small ROF survivor
into the two-product cartoon ``u* = f-v*``.  If ``s0`` and ``s1`` are the two
measured hard-jump potentials, then the implementation obeys

    observation_2 = u* + R v* + P(s0-u*),
    raw_texture   = v* - R v* + P(u*-s1),
    raw_error     = P(u*-s1) - R v*.

These are algebraic identities, not a truth convention.  ``R v*`` is the
fused texture retained in hard-jump cartoon; ``P(u*-s1)`` is the signed
structural halo/mismatch emitted by the scalar complement.  The same
``R v*`` is already present in the second jump observation, proving that the
two visible defects share the finite virtual-resolvent remainder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    lap_hat,
)
from experiments.meyer_preconditioning_research import (  # noqa: E402
    junction_texture_scene,
)
from experiments.meyer_transverse_route_research import (  # noqa: E402
    jump_texture_components,
    native_structural_gate,
)
from experiments.meyer_tsv_validation import (  # noqa: E402
    multiscale_crossing_scene,
    symmetric_support_scene,
)


OUT = ROOT / "experiments" / "out" / "meyer_hard_jump_defect_proof"


def rms(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value)))


def gain(value: np.ndarray, reference: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()
    return float(value @ reference / max(float(reference @ reference), 1e-30))


def apply(multiplier: np.ndarray, value: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(multiplier * np.fft.fft2(value)).real


def analyze(
    scene: dict[str, object], *, lam: float, mu: float, depth: int, threads: int
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    source = np.asarray(scene["source"], dtype=np.float64)
    resolvent = lam / (lam - 2.0 * lam * lap_hat(source.shape))
    retained = resolvent ** int(depth)
    complement = 1.0 - retained

    states: dict[str, np.ndarray] = {}
    raw_texture, final_jump, _ = jump_texture_components(
        source,
        native_structural_gate(source),
        lam=lam,
        virtual_passes=depth,
        state_sink=states,
    )
    initial_jump = states["initial_jump_potential"]
    second_input = states["second_observation_input"]

    hard_plan = bfft.MeyerPlan(
        source.shape, lam=lam, mu=mu, passes=1, threads=threads
    )
    hard_cartoon, hard_texture = hard_plan.split_jump_measure(
        source, virtual_passes=depth
    )
    fused_plan = bfft.MeyerPlan(
        source.shape, lam=lam, mu=mu, passes=64, threads=threads
    )
    fused_model_cartoon, fused_texture = fused_plan.split_legacy(source)
    # Gilles' two ROF maps leave the texture-side survivor w.  For a direct
    # two-product comparison with the exactly recomposing hard jump, fold w
    # into cartoon: u* = f-v* = u+w.  This changes neither fused texture nor
    # any visible target allocation.
    fused_cartoon = source - fused_texture
    fused_survivor = fused_cartoon - fused_model_cartoon

    retained_texture = apply(retained, fused_texture)
    initial_structural_mismatch = apply(
        complement, initial_jump - fused_cartoon
    )
    structural_halo = apply(complement, fused_cartoon - final_jump)
    predicted_second_input = (
        fused_cartoon + retained_texture + initial_structural_mismatch
    )
    predicted_raw_texture = (
        fused_texture - retained_texture + structural_halo
    )
    predicted_raw_error = structural_halo - retained_texture
    capacity_correction = hard_texture - raw_texture
    predicted_final_error = predicted_raw_error + capacity_correction

    report = {
        "identity_residuals": {
            "second_observation_linf": float(np.max(np.abs(
                second_input - predicted_second_input
            ))),
            "raw_texture_linf": float(np.max(np.abs(
                raw_texture - predicted_raw_texture
            ))),
            "raw_error_linf": float(np.max(np.abs(
                (raw_texture - fused_texture) - predicted_raw_error
            ))),
            "final_error_linf": float(np.max(np.abs(
                (hard_texture - fused_texture) - predicted_final_error
            ))),
        },
        "defect_terms": {
            "retained_texture_rms": rms(retained_texture),
            "retained_texture_gain": gain(retained_texture, fused_texture),
            "structural_halo_rms": rms(structural_halo),
            "initial_structural_mismatch_rms": rms(
                initial_structural_mismatch
            ),
            "raw_texture_error_to_fused64_rms": rms(
                raw_texture - fused_texture
            ),
            "capacity_route_change_rms": rms(capacity_correction),
        },
        "native_outputs": {
            "hard_texture_error_to_fused64_rms": rms(
                hard_texture - fused_texture
            ),
            "hard_cartoon_error_to_fused64_rms": rms(
                hard_cartoon - fused_cartoon
            ),
            "fused64_survivor_rms": rms(fused_survivor),
            "hard_recomposition_linf": float(np.max(np.abs(
                hard_cartoon + hard_texture - source
            ))),
            "fused64_two_product_recomposition_linf": float(np.max(np.abs(
                fused_cartoon + fused_texture - source
            ))),
        },
    }
    arrays = {
        "source": source,
        "fused64_cartoon": fused_cartoon,
        "fused64_texture": fused_texture,
        "hard_cartoon": hard_cartoon,
        "hard_texture": hard_texture,
        "retained_texture": retained_texture,
        "structural_halo": structural_halo,
        "initial_structural_mismatch": initial_structural_mismatch,
        "capacity_correction": capacity_correction,
        "predicted_raw_error": predicted_raw_error,
        "predicted_final_error": predicted_final_error,
    }
    return report, arrays


def render(rows: list[tuple[str, dict[str, np.ndarray]]], path: Path) -> None:
    titles = (
        "source", "fused 64 cartoon", "hard-jump cartoon",
        "fused 64 texture", "hard-jump texture",
        r"retained $R_Kv^*$", r"structural $P_K(u^*-s_1)$",
        "capacity correction", "proved hard-minus-fused",
    )
    fig, axes = plt.subplots(len(rows), len(titles), figsize=(12.8, 8.4))
    for row_index, (name, values) in enumerate(rows):
        signed = [
            values["fused64_texture"], values["hard_texture"],
            values["retained_texture"], values["structural_halo"],
            values["capacity_correction"], values["predicted_final_error"],
        ]
        limit = max(
            max(float(np.percentile(np.abs(value), 99.5)) for value in signed),
            1.0,
        )
        panels = (
            (values["source"], "gray", 0.0, 255.0),
            (values["fused64_cartoon"], "gray", 0.0, 255.0),
            (values["hard_cartoon"], "gray", 0.0, 255.0),
            *[(value, "coolwarm", -limit, limit) for value in signed],
        )
        for column, (value, cmap, low, high) in enumerate(panels):
            axis = axes[row_index, column]
            axis.imshow(value, cmap=cmap, vmin=low, vmax=high,
                        interpolation="nearest")
            if row_index == 0:
                axis.set_title(titles[column], fontsize=8.1)
            if column == 0:
                axis.set_ylabel(name.replace("_", " "), fontsize=8.0)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout(pad=0.35, w_pad=0.15, h_pad=0.2)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=40.0)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = (
        pure_edge_scene(args.size),
        pure_carrier_scene(args.size),
        canonical_scene(symmetric_support_scene(args.size)),
        canonical_scene(checker_support_scene(args.size)),
        canonical_scene(multiscale_crossing_scene(args.size)),
        canonical_scene(junction_texture_scene(args.size)),
    )
    report: dict[str, object] = {
        "claim": (
            "The hard-jump halo and carrier residue are the two exact defect "
            "terms of one finite virtual-resolvent approximation."
        ),
        "parameters": {
            "size": args.size,
            "lambda": args.lam,
            "mu": args.mu,
            "virtual_depth": args.depth,
            "threads": args.threads,
        },
        "scenes": {},
    }
    rows: list[tuple[str, dict[str, np.ndarray]]] = []
    archive: dict[str, np.ndarray] = {}
    for scene in scenes:
        name = str(scene["name"])
        scene_report, arrays = analyze(
            scene, lam=args.lam, mu=args.mu, depth=args.depth,
            threads=args.threads,
        )
        report["scenes"][name] = scene_report
        rows.append((name, arrays))
        for key, value in arrays.items():
            archive[f"{name}_{key}"] = np.asarray(value, dtype=np.float32)
        terms = scene_report["defect_terms"]
        identities = scene_report["identity_residuals"]
        print(
            f"{name:20s} retained={terms['retained_texture_rms']:.4f} "
            f"structural={terms['structural_halo_rms']:.4f} "
            f"route={terms['capacity_route_change_rms']:.4f} "
            f"identity={max(identities.values()):.3e}"
        )

    (args.out / "proof.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.out / "proof_arrays.npz", **archive)
    render(rows, args.out / "defect_atlas.png")


if __name__ == "__main__":
    main()
