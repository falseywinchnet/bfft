"""Benchmark the fixed-dimensional lift against the expanded contractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from .causal_scale_transport_2d import causal_scale_transport_observation_2d
from .joint_value_jet_zonotope_contractor_2d import (
    contract_joint_value_jet_scale_edge_state_2d,
)
from .lifted_scale_moment_transport_2d import (
    lifted_scale_moment_transport_state_2d,
)
from .run_2d_denoiser_battery import sources
from .sample_series import corrupt


def _sparse_bytes(matrix: sparse.spmatrix) -> int:
    compressed = matrix.tocsr()
    return int(
        compressed.data.nbytes
        + compressed.indices.nbytes
        + compressed.indptr.nbytes
    )


def _expanded_core_bytes(state: dict[str, Any]) -> int:
    constrained = state["constrained_zonotope"]
    arrays = (
        state["edge_response"],
        state["pushed_zero_response"],
        state["coefficient_lower"],
        state["coefficient_upper"],
    )
    return int(
        sum(value.nbytes for value in arrays)
        + _sparse_bytes(state["generator"])
        + _sparse_bytes(constrained["constraint_matrix"])
        + _sparse_bytes(constrained["coordinate_operator"])
    )


def _lifted_core_bytes(state: dict[str, Any]) -> int:
    arrays = (
        state["posterior_after_erosion"],
        state["vertex_lift"],
        state["pushed_vertex_lift"],
        state["transport_uncertainty"],
        state["metric"]["metric_xx"],
        state["metric"]["metric_xy"],
        state["metric"]["metric_yy"],
    )
    return int(sum(value.nbytes for value in arrays))


def run(size: int, selected: tuple[str, ...]) -> dict[str, Any]:
    catalogue = sources(size)
    rows: list[dict[str, Any]] = []
    for source in selected:
        truth = catalogue[source]
        cases = (
            ("clean", truth),
            (
                "mixed replacement + uniform 0.25",
                corrupt(
                    truth,
                    "mixed replacement + uniform",
                    amount=0.10,
                    density=0.25,
                    seed=9100,
                ),
            ),
        )
        for condition, observation in cases:
            _base, _residual, base_diagnostic = (
                causal_scale_transport_observation_2d(observation))
            initial = np.asarray(
                base_diagnostic["readouts"]["phase_susceptibility"])
            lifted_states = []
            refinement_rows = []
            for refinement in (0, 1):
                started = perf_counter()
                lifted = lifted_scale_moment_transport_state_2d(
                    observation,
                    initial_posterior=initial,
                    trace_refinement=refinement,
                )
                elapsed = perf_counter() - started
                lifted_states.append(lifted["vertex_lift"].copy())
                audit = lifted["joint_normal_audit"]
                refinement_rows.append({
                    "trace_refinement": refinement,
                    "elapsed_seconds": elapsed,
                    "persistent_dimension": lifted["persistent_dimension"],
                    "lineage_count_used_to_form_moments": lifted[
                        "lineage_count_used_to_form_moments"],
                    "core_bytes": _lifted_core_bytes(lifted),
                    "dense_push_to_lifted_push_scalar_ratio": lifted[
                        "dense_push_to_lifted_push_scalar_ratio"],
                    "observation_recomposition_error": lifted[
                        "observation_recomposition_error"],
                    "lifted_residual_recomposition_error": lifted[
                        "lifted_residual_recomposition_error"],
                    "pushed_signed_commutation_error": lifted[
                        "pushed_signed_commutation_error"],
                    "mean_transport_uncertainty": float(np.mean(
                        lifted["transport_uncertainty"])),
                    "mean_inherited_scale_variance": float(np.mean(
                        lifted["inherited_readout"]["scale_variance"])),
                    "mean_inherited_sign_coherence": float(np.mean(
                        lifted["inherited_readout"]["sign_coherence"])),
                    "full_action_constraint_violation_fraction": audit[
                        "full_action_constraint_violation_fraction"],
                    "full_action_joint_edge_compatibility": audit[
                        "full_action_joint_edge_compatibility"],
                })
                del lifted

            started = perf_counter()
            expanded = contract_joint_value_jet_scale_edge_state_2d(
                observation,
                initial_posterior=initial,
                trace_refinement=0,
            )
            expanded_elapsed = perf_counter() - started
            expanded_violation = expanded["constrained_zonotope"][
                "full_transfer_violation_fraction"]
            expanded_bytes = _expanded_core_bytes(expanded)
            lift0 = refinement_rows[0]
            signed_refinement_error = float(np.max(np.abs(
                lifted_states[0][[0, 5]] - lifted_states[1][[0, 5]]
            )))
            scale_moment_refinement_change = float(np.mean(np.abs(
                lifted_states[0][[1, 2, 3, 4, 6, 7, 8, 9]]
                - lifted_states[1][[1, 2, 3, 4, 6, 7, 8, 9]]
            )))
            rows.append({
                "source": source,
                "condition": condition,
                "lifted_refinements": refinement_rows,
                "signed_refinement_maximum_error": signed_refinement_error,
                "scale_moment_refinement_mean_change": (
                    scale_moment_refinement_change),
                "expanded_refinement_zero": {
                    "elapsed_seconds": expanded_elapsed,
                    "core_bytes": expanded_bytes,
                    "variable_count": int(expanded["generator"].shape[1]),
                    "constraint_count": int(expanded[
                        "constrained_zonotope"]["constraint_matrix"].shape[0]),
                    "full_action_constraint_violation_fraction": (
                        expanded_violation),
                },
                "expanded_to_lifted_runtime_ratio": float(
                    expanded_elapsed / lift0["elapsed_seconds"]),
                "expanded_to_lifted_core_memory_ratio": float(
                    expanded_bytes / lift0["core_bytes"]),
                "full_action_violation_agreement_error": float(abs(
                    expanded_violation
                    - lift0["full_action_constraint_violation_fraction"]
                )),
            })
            del expanded
    return {
        "purpose": (
            "compare a ten-channel raw scale-moment lift with the expanded "
            "lineage-edge joint contractor in conservation, local normal "
            "audit, refinement behavior, runtime, and core representation"
        ),
        "size": int(size),
        "sources": list(selected),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--sources", default="cameraman,tapered hair,woven chirps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        base = row["lifted_refinements"][0]
        print(
            row["source"], "|", row["condition"],
            "runtime ratio", round(row["expanded_to_lifted_runtime_ratio"], 2),
            "memory ratio", round(
                row["expanded_to_lifted_core_memory_ratio"], 2),
            "lineages->dims", (
                base["lineage_count_used_to_form_moments"],
                base["persistent_dimension"],
            ),
            "normal agreement error", row[
                "full_action_violation_agreement_error"],
            "signed refinement error", row[
                "signed_refinement_maximum_error"],
        )


if __name__ == "__main__":
    main()
