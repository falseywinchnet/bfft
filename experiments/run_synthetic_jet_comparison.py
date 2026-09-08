"""Synthetic reconstruction and matched restriction/prolongation study.

Seven analytic scenes isolate affine precision, oblique transport, carrier
phase, curved and crossing geometry, above-Nyquist content, and maximal
checkerboard aliasing.  Reconstruction arms share a common exact source;
restriction arms are scored only through their own matched cycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.easu_reference import easu_nested_2x
from experiments.self_geometric_harmonic_interpolation import (
    _batched_retracted_profile_fields,
    _cosine_spectral_jet,
    _natural_cubic_variational_jet,
    _sbp42_jet,
    analytic_scene,
    batched_cosine_hermite_characteristic_resize,
    bilinear_resample_to_shape,
    compact_spline_hermite_characteristic_resize,
    cosine_hermite_characteristic_resize,
    harmonic_resize,
    lanczos_resize,
    lanczos_resample_to_shape,
    positive_restrict,
    directional_variation_measure_resize,
    riemannian_harmonic_adjoint_restrict,
    riemannian_lifting_restrict,
    riemannian_transport_restrict,
    riemannian_product_variation_resize,
    sbp42_hermite_characteristic_resize,
    tensor_product_quintic_variation_resize,
)


Array = np.ndarray
RESOLVABLE_SCENES = (
    "oblique_edge",
    "oblique_carrier",
    "curved_edge",
    "crossing",
)
ALIAS_STRESS_SCENES = (
    "super_nyquist_carrier",
    "fine_checker",
)
SCENES = ("affine",) + RESOLVABLE_SCENES + ALIAS_STRESS_SCENES
METHODS = {
    "bilinear": lambda source, scale: harmonic_resize(
        source, scale, adaptive=False
    ),
    "active_dct": cosine_hermite_characteristic_resize,
    "batched_dct": batched_cosine_hermite_characteristic_resize,
    "compact_spline": compact_spline_hermite_characteristic_resize,
    "sbp42_local": sbp42_hermite_characteristic_resize,
    "quintic_variation_product": tensor_product_quintic_variation_resize,
    "riemannian_variation_product": riemannian_product_variation_resize,
    "directional_variation_measure": directional_variation_measure_resize,
    "lanczos3": lambda source, scale: lanczos_resize(source, scale, radius=3),
    "easu": lambda source, scale: easu_nested_2x(source),
}
PROPOSALS = {
    "batched_dct": _cosine_spectral_jet,
    "compact_spline": _natural_cubic_variational_jet,
    "sbp42_local": _sbp42_jet,
}


def _mse(value: Array, truth: Array, mask: Array | None = None) -> float:
    residual = np.asarray(value) - np.asarray(truth)
    if mask is not None:
        residual = residual[mask]
    return float(np.mean(residual * residual))


def _global_overshoot(value: Array, source: Array) -> float:
    low = float(np.min(source))
    high = float(np.max(source))
    return max(
        float(np.max(low - value, initial=0.0)),
        float(np.max(value - high, initial=0.0)),
    )


def _timed(operation: object, source: Array, scale: int, repeats: int) -> tuple[Array, float]:
    result = operation(source, scale)
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation(source, scale)
        elapsed.append(time.perf_counter() - started)
    return result, float(np.median(elapsed))


def _matched_family_cycle(
    method: str,
    truth: Array,
    source_side: int,
    scale: int,
) -> Array | None:
    """Apply each family-defined restriction followed by its prolongation."""

    if method in {"active_dct", "batched_dct"}:
        source = riemannian_lifting_restrict(truth, scale)
        return METHODS[method](source, scale)
    if method in {
        "compact_spline",
        "sbp42_local",
        "quintic_variation_product",
        "riemannian_variation_product",
        "directional_variation_measure",
    }:
        return None
    if method == "bilinear":
        source = bilinear_resample_to_shape(
            truth, (source_side, source_side)
        )
        return bilinear_resample_to_shape(source, truth.shape[:2])
    if method == "lanczos3":
        source = lanczos_resample_to_shape(
            truth, (source_side, source_side), radius=3
        )
        return lanczos_resample_to_shape(source, truth.shape[:2], radius=3)
    if method == "easu":
        return None
    raise KeyError(method)


def _jet_diagnostics(source: Array, repeats: int) -> dict[str, object]:
    field = source[..., None]
    derivatives: dict[str, tuple[Array, Array]] = {}
    runtimes: dict[str, float] = {}
    for name, proposal in PROPOSALS.items():
        derivatives[name] = _batched_retracted_profile_fields(field, proposal)
        elapsed = []
        for _ in range(repeats):
            started = time.perf_counter()
            _batched_retracted_profile_fields(field, proposal)
            elapsed.append(time.perf_counter() - started)
        runtimes[name] = float(np.median(elapsed))
    reference_row, reference_column = derivatives["batched_dct"]
    return {
        name: {
            "median_seconds": runtimes[name],
            "maximum_admitted_row_jet_delta_vs_dct": float(np.max(np.abs(
                derivative[0] - reference_row
            ))),
            "maximum_admitted_column_jet_delta_vs_dct": float(np.max(np.abs(
                derivative[1] - reference_column
            ))),
            "rms_admitted_jet_delta_vs_dct": float(np.sqrt(0.5 * (
                np.mean((derivative[0] - reference_row) ** 2)
                + np.mean((derivative[1] - reference_column) ** 2)
            ))),
        }
        for name, derivative in derivatives.items()
    }


def run_comparison(
    *, source_side: int = 33, scale: int = 2, repeats: int = 3
) -> dict[str, object]:
    if scale != 2:
        raise ValueError("the EASU comparison arm is defined for scale=2")
    truth_side = (source_side - 1) * scale + 1
    held_out = np.ones((truth_side, truth_side), dtype=bool)
    held_out[::scale, ::scale] = False
    crop = 2 * scale
    interior = np.zeros_like(held_out)
    interior[crop:-crop, crop:-crop] = True
    interior &= held_out
    records: dict[str, object] = {}
    for scene in SCENES:
        truth = analytic_scene(scene, truth_side)
        source = truth[::scale, ::scale]
        reconstructions: dict[str, Array] = {}
        methods: dict[str, object] = {}
        for name, operation in METHODS.items():
            reconstruction, runtime = _timed(operation, source, scale, repeats)
            reconstructions[name] = reconstruction
            returned = positive_restrict(reconstruction, scale)
            matched_cycle = _matched_family_cycle(
                name, truth, source_side, scale
            )
            methods[name] = {
                "held_out_mse": _mse(reconstruction, truth, held_out),
                "interior_held_out_mse": _mse(reconstruction, truth, interior),
                "maximum_sample_error": float(np.max(np.abs(
                    reconstruction[::scale, ::scale] - source
                ))),
                "global_source_range_overshoot": _global_overshoot(
                    reconstruction, source
                ),
                "common_positive_return_mse_interior": _mse(
                    returned[1:-1, 1:-1], source[1:-1, 1:-1]
                ),
                "matched_family_round_trip_mse_interior": (
                    None if matched_cycle is None else _mse(
                        matched_cycle[crop:-crop, crop:-crop],
                        truth[crop:-crop, crop:-crop],
                    )
                ),
                "median_runtime_seconds": runtime,
            }
        active = reconstructions["active_dct"]
        for name, reconstruction in reconstructions.items():
            methods[name]["maximum_output_delta_vs_active_dct"] = float(
                np.max(np.abs(reconstruction - active))
            )
            methods[name]["held_out_mse_delta_vs_active_dct"] = (
                methods[name]["held_out_mse"]
                - methods["active_dct"]["held_out_mse"]
            )
        records[scene] = {
            "methods": methods,
            "jet_diagnostics": _jet_diagnostics(source, repeats),
            "restriction_study": {},
        }
        restrictions = {
            "exact_nested_selection": truth[::scale, ::scale],
            "positive_triangle": positive_restrict(truth, scale),
            "riemannian_harmonic_adjoint": (
                riemannian_harmonic_adjoint_restrict(truth, scale)
            ),
            "riemannian_transport_composed": (
                riemannian_transport_restrict(truth, scale)
            ),
            "riemannian_one_stage_lifting": (
                riemannian_lifting_restrict(truth, scale)
            ),
        }
        for restriction_name, restricted in restrictions.items():
            cycle = cosine_hermite_characteristic_resize(restricted, scale)
            records[scene]["restriction_study"][restriction_name] = {
                "coarse_global_range_overshoot": _global_overshoot(
                    restricted, truth
                ),
                "cycle_mse": _mse(cycle, truth),
                "cycle_mse_interior": _mse(
                    cycle[crop:-crop, crop:-crop],
                    truth[crop:-crop, crop:-crop],
                ),
            }
        lanczos_cycle = _matched_family_cycle(
            "lanczos3", truth, source_side, scale
        )
        records[scene]["restriction_study"]["lanczos3_family"] = {
            "coarse_global_range_overshoot": _global_overshoot(
                lanczos_resample_to_shape(
                    truth, (source_side, source_side), radius=3
                ),
                truth,
            ),
            "cycle_mse": _mse(lanczos_cycle, truth),
            "cycle_mse_interior": _mse(
                lanczos_cycle[crop:-crop, crop:-crop],
                truth[crop:-crop, crop:-crop],
            ),
        }

    nonaffine = [records[name] for name in SCENES if name != "affine"]
    summary: dict[str, object] = {}
    for method in METHODS:
        deltas = np.array([
            record["methods"][method]["held_out_mse_delta_vs_active_dct"]
            for record in nonaffine
        ])
        summary[method] = {
            "nonaffine_wins_ties_losses_vs_active_dct": [
                int(np.sum(deltas < 0.0)),
                int(np.sum(deltas == 0.0)),
                int(np.sum(deltas > 0.0)),
            ],
            "mean_nonaffine_mse_delta_vs_active_dct": float(np.mean(deltas)),
            "mean_nonaffine_held_out_mse": float(np.mean([
                record["methods"][method]["held_out_mse"]
                for record in nonaffine
            ])),
            "maximum_global_source_range_overshoot": float(max(
                record["methods"][method]["global_source_range_overshoot"]
                for record in records.values()
            )),
            "maximum_sample_error": float(max(
                record["methods"][method]["maximum_sample_error"]
                for record in records.values()
            )),
        }
    restriction_names = tuple(
        records[SCENES[0]]["restriction_study"].keys()
    )
    restriction_summary: dict[str, object] = {}
    for restriction_name in restriction_names:
        groups: dict[str, object] = {}
        for group_name, scenes in (
            ("resolvable", RESOLVABLE_SCENES),
            ("alias_stress", ALIAS_STRESS_SCENES),
        ):
            values = np.array([
                records[scene]["restriction_study"][restriction_name][
                    "cycle_mse"
                ]
                for scene in scenes
            ])
            lanczos_values = np.array([
                records[scene]["restriction_study"]["lanczos3_family"][
                    "cycle_mse"
                ]
                for scene in scenes
            ])
            groups[group_name] = {
                "wins_ties_losses_vs_lanczos3_family": [
                    int(np.sum(values < lanczos_values)),
                    int(np.sum(values == lanczos_values)),
                    int(np.sum(values > lanczos_values)),
                ],
                "mean_cycle_mse": float(np.mean(values)),
                "mean_per_scene_mse_ratio_vs_lanczos3_family": float(
                    np.mean(values / lanczos_values)
                ),
            }
        restriction_summary[restriction_name] = groups
    return {
        "benchmark": "analytic source-line jet representation comparison",
        "source_side": source_side,
        "scale": scale,
        "truth_side": truth_side,
        "runtime_repeats": repeats,
        "metric_definitions": {
            "common_source_mse": (
                "Every method reconstructs the same exact decimation of an "
                "analytic scene and is scored against that analytic scene."
            ),
            "common_positive_return_mse": (
                "A diagnostic that applies the same positive restriction to "
                "every reconstruction; it is not a method-matched round trip."
            ),
            "matched_family_round_trip_mse": (
                "Fine samples are restricted and prolonged by the same "
                "operator family. The active and algebraically equivalent "
                "batched operators use the local-support-projected one-stage "
                "Riemannian lifting restriction; Lanczos-3 uses bandwidth-"
                "scaled Lanczos-3 in both directions. Derivative ablations "
                "and EASU are null because no matched restriction is declared."
            ),
            "restriction_study": (
                "The active upsampler is held fixed while exact nested "
                "selection, triangular full weighting, and the normalized "
                "adjoint of positive Riemannian harmonic coordinates supply "
                "the coarse field. The composed arm blends that adjoint with "
                "cardinal characteristic pullback using the upsampler's same "
                "kappa(M)-1 coefficient. The lifting arm predicts from the "
                "cardinal pullback and returns its residual once through the "
                "fine-geometry harmonic adjoint. Lanczos uses its own "
                "down/up family."
            ),
        },
        "scenes": list(SCENES),
        "records": records,
        "summary": summary,
        "restriction_summary": restriction_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-side", type=int, default=33)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/synthetic_jet_comparison.json"),
    )
    args = parser.parse_args()
    result = run_comparison(
        source_side=args.source_side, scale=args.scale, repeats=args.repeats
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "reconstruction_summary": result["summary"],
        "restriction_summary": result["restriction_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
