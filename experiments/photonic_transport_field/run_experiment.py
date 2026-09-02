"""Run the first compression, occlusion, and transport controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .scenes import facing_patch_grids
from .budgeted import BudgetedTransportGeometry, solve_budgeted_transport
from .transport import (
    compile_dense_centroid_field,
    compile_hierarchical_field,
    dense_fixed_point,
    solve_transport,
)


def _compile_record(side: int, occluder_radius: float | None) -> dict:
    patches, blockers = facing_patch_grids(
        side=side,
        separation=8.0,
        extent=2.0,
        occluder_radius=occluder_radius,
    )
    started = time.perf_counter()
    dense = compile_dense_centroid_field(patches, occluders=blockers)
    dense_seconds = time.perf_counter() - started
    started = time.perf_counter()
    field = compile_hierarchical_field(
        patches,
        occluders=blockers,
        admissibility=0.10,
    )
    hierarchy_seconds = time.perf_counter() - started
    dense_matrix = dense.dense_matrix()
    field_matrix = field.dense_matrix()
    relative_operator_error = float(
        np.linalg.norm(field_matrix - dense_matrix)
        / max(np.linalg.norm(dense_matrix), np.finfo(float).tiny)
    )
    emission = np.stack([patch.emission for patch in patches])
    albedo = np.stack([patch.albedo for patch in patches])
    started = time.perf_counter()
    solution = solve_transport(field, emission, albedo)
    solve_seconds = time.perf_counter() - started
    reference = dense_fixed_point(field, emission, albedo)
    requested_error = 1.0e-5 * float(np.sum(emission))
    geometry = BudgetedTransportGeometry(
        patches,
        occluders=blockers,
        admissibility=0.10,
    )
    started = time.perf_counter()
    budgeted = solve_budgeted_transport(
        geometry,
        emission,
        albedo,
        outgoing_error_budget=requested_error,
    )
    budgeted_seconds = time.perf_counter() - started
    depth_work = [{
        "depth": int(record["depth"]),
        "frontier_energy": float(record["frontier_energy"]),
        "cluster_pair_visits": int(record["cluster_pair_visits"]),
        "blocks": int(record["block_count"]),
        "visibility_tests": int(record["exact_visibility_tests"]),
        "discarded_incident_bound": float(record["energy_pruned_bound"]),
        "next_frontier_energy": float(record["next_frontier_energy"]),
    } for record in budgeted.depth_diagnostics]
    return {
        "nodes": len(patches),
        "occluder_radius": occluder_radius,
        "dense_compile_seconds": dense_seconds,
        "hierarchical_compile_seconds": hierarchy_seconds,
        "solve_seconds": solve_seconds,
        "relative_operator_error": relative_operator_error,
        "relative_fixed_point_error": float(
            np.linalg.norm(solution.outgoing - reference)
            / max(np.linalg.norm(reference), np.finfo(float).tiny)
        ),
        "maximum_conservation_error": float(np.max(np.abs(
            solution.conservation_error
        ))),
        "bounce_depths": solution.depth_count,
        "energy_gated": {
            "requested_outgoing_error_budget": requested_error,
            "solve_seconds": budgeted_seconds,
            "depths": budgeted.depth_count,
            "actual_l1_error": float(np.sum(np.abs(
                budgeted.outgoing - reference
            ))),
            "certified_l1_error_bound": budgeted.outgoing_error_bound,
            "depth_work": depth_work,
        },
        "dense_stored_coefficients": dense.stored_coefficients,
        "hierarchical_stored_coefficients": field.stored_coefficients,
        "compression_ratio": (
            dense.stored_coefficients / max(field.stored_coefficients, 1)
        ),
        "compiler": field.diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {
        "experiment": "energy_gated_adaptive_diffuse_transport_field_v2",
        "open": _compile_record(args.side, None),
        "occluded": _compile_record(args.side, 0.4),
        "interpretation": (
            "transport geometry is discovered per residual frontier; coherent "
            "far exchange is low-rank, uncertain beam hulls refine, and complete "
            "receiver subtrees vanish when their deliverable energy fits inside "
            "the shared remaining error budget"
        ),
    }
    payload = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
