"""Broad gate for continuous action-contracted connection transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_predictive_transport import (
    action_contracting_connection_readout_forms,
)
from .run_1d_cross_predictive_battery import CONDITIONS, PRESET_NAMES, metrics
from .sample_series import PRESETS, compose_series, corrupt


METHODS = (
    "fused_collision_mean",
    "action_contracted_mean",
    "action_contracted_collision_mean",
    "action_contracted_path_collision_mean",
)


def evaluate(
    value: np.ndarray,
    parallel_transport_connection: bool,
    require_population_phase_collision: bool,
    fuse_population_phase_odds: bool,
    fuse_connection_acceleration_odds: bool,
    fuse_connection_jerk_odds: bool,
    fuse_connection_tangent_odds: bool,
    fuse_connection_spherical_phase_odds: bool,
    fuse_connection_spherical_phase_union: bool,
    suppress_connection_on_spherical_phase: bool,
    fuse_phase_defect_spherical_odds: bool,
    newton_optimize_connection: bool,
    marginalize_connection_action: bool,
    marginalize_gaussian_connection: bool,
    phase_coherent_connection_posterior: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    forms, diagnostic = action_contracting_connection_readout_forms(
        value,
        parallel_transport_connection=parallel_transport_connection,
        require_population_phase_collision=(
            require_population_phase_collision),
        fuse_population_phase_odds=fuse_population_phase_odds,
        fuse_connection_acceleration_odds=(
            fuse_connection_acceleration_odds),
        fuse_connection_jerk_odds=fuse_connection_jerk_odds,
        fuse_connection_tangent_odds=fuse_connection_tangent_odds,
        fuse_connection_spherical_phase_odds=(
            fuse_connection_spherical_phase_odds),
        fuse_connection_spherical_phase_union=(
            fuse_connection_spherical_phase_union),
        suppress_connection_on_spherical_phase=(
            suppress_connection_on_spherical_phase),
        fuse_phase_defect_spherical_odds=(
            fuse_phase_defect_spherical_odds),
        newton_optimize_connection=newton_optimize_connection,
        marginalize_connection_action=marginalize_connection_action,
        marginalize_gaussian_connection=marginalize_gaussian_connection,
        phase_coherent_connection_posterior=(
            phase_coherent_connection_posterior),
    )
    return {
        "fused_collision_mean": forms["baseline_collision_mean"],
        "action_contracted_mean": forms["mean"],
        "action_contracted_collision_mean": forms["collision_mean"],
        "action_contracted_path_collision_mean": forms[
            "path_collision_mean"],
    }, diagnostic


def run(
    size: int,
    seeds: int,
    parallel_transport_connection: bool,
    require_population_phase_collision: bool,
    fuse_population_phase_odds: bool,
    fuse_connection_acceleration_odds: bool,
    fuse_connection_jerk_odds: bool,
    fuse_connection_tangent_odds: bool,
    fuse_connection_spherical_phase_odds: bool,
    fuse_connection_spherical_phase_union: bool,
    suppress_connection_on_spherical_phase: bool,
    fuse_phase_defect_spherical_odds: bool,
    newton_optimize_connection: bool,
    marginalize_connection_action: bool,
    marginalize_gaussian_connection: bool,
    phase_coherent_connection_posterior: bool,
) -> dict:
    clean_rows: list[dict] = []
    noisy_rows: list[dict] = []

    def row(
        preset: str,
        truth: np.ndarray,
        value: np.ndarray,
        *,
        condition: str | None = None,
        seed: int | None = None,
    ) -> dict:
        forms, diagnostic = evaluate(
            value,
            parallel_transport_connection,
            require_population_phase_collision,
            fuse_population_phase_odds,
            fuse_connection_acceleration_odds,
            fuse_connection_jerk_odds,
            fuse_connection_tangent_odds,
            fuse_connection_spherical_phase_odds,
            fuse_connection_spherical_phase_union,
            suppress_connection_on_spherical_phase,
            fuse_phase_defect_spherical_odds,
            newton_optimize_connection,
            marginalize_connection_action,
            marginalize_gaussian_connection,
            phase_coherent_connection_posterior,
        )
        spherical_phase = (
            diagnostic.get("connection_spherical_phase")
            or diagnostic.get("phase_connection")
            or {}
        )
        result = {
            "preset": preset,
            **{name: metrics(section, truth) for name, section in forms.items()},
            "mean_branch_connection_contrast": diagnostic[
                "mean_branch_connection_contrast"],
            "maximum_harmonic_action_violation": diagnostic[
                "maximum_harmonic_action_violation"],
            "mean_lineage_population": diagnostic[
                "mean_lineage_population"],
            "mean_connection_action_posterior": diagnostic[
                "mean_connection_action_posterior"],
            "mean_connection_action_posterior_variance": diagnostic[
                "mean_connection_action_posterior_variance"],
            "mean_spherical_phase_resultant": float(
                spherical_phase.get("mean_phase_resultant", 0.0)),
        }
        if condition is not None:
            result["condition"] = condition
        if seed is not None:
            result["seed"] = seed
        return result

    for preset in PRESET_NAMES:
        truth = compose_series(size, PRESETS[preset])[1]
        clean_rows.append(row(preset, truth, truth))
        for condition, kind, amount, density in CONDITIONS:
            for seed in range(seeds):
                observation = corrupt(
                    truth,
                    kind,
                    amount=amount,
                    density=density,
                    seed=20100 + seed,
                )
                noisy_rows.append(row(
                    preset,
                    truth,
                    observation,
                    condition=condition,
                    seed=seed,
                ))

    def summarize(rows: list[dict]) -> dict:
        return {
            method: {
                key: float(np.mean([entry[method][key] for entry in rows]))
                for key in rows[0][method]
            }
            for method in METHODS
        }

    def state_summary(rows: list[dict]) -> dict:
        return {
            key: float(np.mean([entry[key] for entry in rows]))
            for key in (
                "mean_branch_connection_contrast",
                "maximum_harmonic_action_violation",
                "mean_lineage_population",
                "mean_connection_action_posterior",
                "mean_connection_action_posterior_variance",
                "mean_spherical_phase_resultant",
            )
        }

    return {
        "purpose": (
            "falsify branch-resolved harmonic connection-action contraction"
        ),
        "size": int(size),
        "seeds": int(seeds),
        "parallel_transport_connection": bool(
            parallel_transport_connection),
        "population_phase_collision": bool(
            require_population_phase_collision),
        "population_phase_odds_fusion": bool(fuse_population_phase_odds),
        "connection_acceleration_odds_fusion": bool(
            fuse_connection_acceleration_odds),
        "connection_jerk_odds_fusion": bool(fuse_connection_jerk_odds),
        "connection_tangent_odds_fusion": bool(
            fuse_connection_tangent_odds),
        "connection_spherical_phase_odds_fusion": bool(
            fuse_connection_spherical_phase_odds),
        "connection_spherical_phase_union_fusion": bool(
            fuse_connection_spherical_phase_union),
        "connection_spherical_phase_suppression": bool(
            suppress_connection_on_spherical_phase),
        "phase_defect_spherical_odds_fusion": bool(
            fuse_phase_defect_spherical_odds),
        "newton_optimized_connection": bool(newton_optimize_connection),
        "marginalized_connection_action": bool(
            marginalize_connection_action),
        "marginalized_gaussian_connection": bool(
            marginalize_gaussian_connection),
        "phase_coherent_connection_posterior": bool(
            phase_coherent_connection_posterior),
        "clean_summary": summarize(clean_rows),
        "noisy_summary": summarize(noisy_rows),
        "clean_state": state_summary(clean_rows),
        "noisy_state": state_summary(noisy_rows),
        "by_condition": {
            condition: summarize([
                entry for entry in noisy_rows
                if entry["condition"] == condition
            ])
            for condition, *_rest in CONDITIONS
        },
        "clean_rows": clean_rows,
        "rows": noisy_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument(
        "--parallel-transport-connection",
        action="store_true",
    )
    parser.add_argument(
        "--population-phase-collision",
        action="store_true",
    )
    parser.add_argument(
        "--population-phase-odds",
        action="store_true",
    )
    parser.add_argument(
        "--connection-acceleration-odds",
        action="store_true",
    )
    parser.add_argument(
        "--connection-jerk-odds",
        action="store_true",
    )
    parser.add_argument(
        "--connection-tangent-odds",
        action="store_true",
    )
    parser.add_argument(
        "--connection-spherical-phase-odds",
        action="store_true",
    )
    parser.add_argument(
        "--connection-spherical-phase-union",
        action="store_true",
    )
    parser.add_argument(
        "--connection-spherical-phase-veto",
        action="store_true",
    )
    parser.add_argument(
        "--phase-defect-spherical-odds",
        action="store_true",
    )
    parser.add_argument(
        "--newton-connection",
        action="store_true",
    )
    parser.add_argument(
        "--connection-action-posterior",
        action="store_true",
    )
    parser.add_argument(
        "--gaussian-connection-posterior",
        action="store_true",
    )
    parser.add_argument(
        "--phase-coherent-connection-posterior",
        action="store_true",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        args.seeds,
        args.parallel_transport_connection,
        args.population_phase_collision,
        args.population_phase_odds,
        args.connection_acceleration_odds,
        args.connection_jerk_odds,
        args.connection_tangent_odds,
        args.connection_spherical_phase_odds,
        args.connection_spherical_phase_union,
        args.connection_spherical_phase_veto,
        args.phase_defect_spherical_odds,
        args.newton_connection,
        args.connection_action_posterior,
        args.gaussian_connection_posterior,
        args.phase_coherent_connection_posterior,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "clean": result["clean_summary"],
        "noisy": result["noisy_summary"],
        "state": {
            "clean": result["clean_state"],
            "noisy": result["noisy_state"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
