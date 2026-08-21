"""Corruption gate for CRPS-stopped continuous-source Sasaki geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .continuous_tangent_source_geometry_2d import (
    continuous_tangent_source_geometry_2d,
)
from .probe_continuous_tangent_convergence import CONDITIONS
from .run_2d_denoiser_battery import sources
from .sample_series import corrupt


def _summary(geometry: dict) -> dict[str, float]:
    xx = np.asarray(geometry["metric_xx"], dtype=np.float64)
    xy = np.asarray(geometry["metric_xy"], dtype=np.float64)
    yy = np.asarray(geometry["metric_yy"], dtype=np.float64)
    coherence = np.hypot(xx - yy, 2.0 * xy) / np.maximum(
        xx + yy, np.finfo(float).tiny)
    return {
        "implied_support": float(geometry["implied_support"]),
        "information_trace_mean": float(geometry["information_trace_mean"]),
        "metric_coherence_mean": float(np.mean(coherence)),
        "metric_coherence_p90": float(np.quantile(coherence, 0.90)),
    }


def run(
    size: int,
    selected_sources: tuple[str, ...],
    angular_count: int,
    quantile_count: int,
    ceiling: int,
    remetricize: bool,
    selected_conditions: tuple[str, ...] | None = None,
    joint_jet_action: bool = False,
    joint_jet_law_action: bool = False,
    joint_bundle_action: bool = False,
    strict_joint_bundle_action: bool = False,
    line_search: bool = False,
    local_joint_transport: bool = False,
    causal_support_joint_transport: bool = False,
    strict_joint_graph_gradient_flow: bool = False,
) -> dict:
    catalogue = sources(size)
    rows = []
    methods = (
        "horizontal", "initial_vertical", "vertical",
        "initial_fused", "fused",
        "initial_prolongation", "prolongation",
        "initial_prolongation_fused", "prolongation_fused",
    )
    for source in selected_sources:
        truth = catalogue[source]
        for condition, kind, amount, density in CONDITIONS:
            if selected_conditions is not None and condition not in selected_conditions:
                continue
            observation = (
                truth if kind is None else corrupt(
                    truth, kind, amount=amount, density=density, seed=31013)
            )
            geometry, diagnostic = continuous_tangent_source_geometry_2d(
                observation,
                angular_count=angular_count,
                quantile_count=quantile_count,
                maximum_source_transports=ceiling,
                remetricize=remetricize,
                joint_jet_action=joint_jet_action,
                joint_jet_law_action=joint_jet_law_action,
                joint_bundle_action=joint_bundle_action,
                strict_joint_bundle_action=strict_joint_bundle_action,
                line_search=line_search,
                local_joint_transport=local_joint_transport,
                causal_support_joint_transport=causal_support_joint_transport,
                strict_joint_graph_gradient_flow=(
                    strict_joint_graph_gradient_flow),
            )
            rows.append({
                "source": source,
                "condition": condition,
                "geometry": {
                    method: _summary(geometry[method]) for method in methods
                },
                "accepted_source_transports": diagnostic[
                    "accepted_source_transports"],
                "source_transport_ceiling_hit": diagnostic[
                    "source_transport_ceiling_hit"],
                "terminal_held_out_residual_crps": diagnostic[
                    "terminal_held_out_residual_crps"],
                "terminal_projective_jet_crps": diagnostic[
                    "terminal_projective_jet_crps"],
                "terminal_transport_action": diagnostic[
                    "terminal_transport_action"],
                "first_source_transport": (
                    {
                        key: (
                            value.item()
                            if isinstance(value, np.generic) else value
                        )
                        for key, value in diagnostic["source_transports"][0].items()
                    }
                    if diagnostic["source_transports"] else None
                ),
                "maximum_target_self_lineage": diagnostic[
                    "maximum_target_self_lineage"],
                "lineage_row_mass_maximum_error": diagnostic[
                    "lineage_row_mass_maximum_error"],
            })
    clean = {
        (row["source"], method): row["geometry"][method]["implied_support"]
        for row in rows if row["condition"] == "clean"
        for method in methods
    }
    condition_summary = {}
    for condition, _kind, _amount, _density in CONDITIONS:
        if selected_conditions is not None and condition not in selected_conditions:
            continue
        selected = [row for row in rows if row["condition"] == condition]
        condition_summary[condition] = {
            method: {
                "mean_implied_support": float(np.mean([
                    row["geometry"][method]["implied_support"]
                    for row in selected
                ])),
                "mean_ratio_to_source_clean": (
                    float(np.mean([
                        row["geometry"][method]["implied_support"]
                        / clean[(row["source"], method)]
                        for row in selected
                        if (row["source"], method) in clean
                    ]))
                    if any(
                        (row["source"], method) in clean for row in selected
                    ) else None
                ),
            }
            for method in methods
        }
        condition_summary[condition].update({
            "mean_accepted_source_transports": float(np.mean([
                row["accepted_source_transports"] for row in selected
            ])),
            "maximum_accepted_source_transports": int(np.max([
                row["accepted_source_transports"] for row in selected
            ])),
        })
    return {
        "purpose": (
            "test the fused ordering: held-out horizontal geometry, CRPS-stopped "
            "continuous source transport, then vertical jet information"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "source_transport_ceiling": int(ceiling),
        "remetricize": bool(remetricize),
        "joint_jet_action": bool(joint_jet_action),
        "joint_jet_law_action": bool(joint_jet_law_action),
        "joint_bundle_action": bool(joint_bundle_action),
        "strict_joint_bundle_action": bool(strict_joint_bundle_action),
        "line_search": bool(line_search),
        "local_joint_transport": bool(local_joint_transport),
        "causal_support_joint_transport": bool(
            causal_support_joint_transport),
        "strict_joint_graph_gradient_flow": bool(
            strict_joint_graph_gradient_flow),
        "condition_summary": condition_summary,
        "ceiling_hits": int(sum(
            row["source_transport_ceiling_hit"] for row in rows)),
        "maximum_target_self_lineage": float(max(
            row["maximum_target_self_lineage"] for row in rows)),
        "maximum_lineage_row_mass_error": float(max(
            row["lineage_row_mass_maximum_error"] for row in rows)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--sources",
        default="tapered hair,geometric interfaces,woven chirps",
    )
    parser.add_argument("--angular-count", type=int, default=16)
    parser.add_argument("--quantile-count", type=int, default=32)
    parser.add_argument("--source-transports", type=int, default=32)
    parser.add_argument("--remetricize", action="store_true")
    parser.add_argument("--joint-jet-action", action="store_true")
    parser.add_argument("--joint-jet-law-action", action="store_true")
    parser.add_argument("--joint-bundle-action", action="store_true")
    parser.add_argument("--strict-joint-bundle-action", action="store_true")
    parser.add_argument("--line-search", action="store_true")
    parser.add_argument("--local-joint-transport", action="store_true")
    parser.add_argument(
        "--causal-support-joint-transport", action="store_true")
    parser.add_argument(
        "--strict-joint-graph-gradient-flow", action="store_true")
    parser.add_argument(
        "--conditions",
        default="",
        help="optional comma-separated subset of condition labels",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(
        value.strip() for value in args.sources.split(",") if value.strip())
    result = run(
        args.size,
        selected,
        args.angular_count,
        args.quantile_count,
        args.source_transports,
        args.remetricize,
        tuple(
            value.strip() for value in args.conditions.split(",")
            if value.strip()
        ) or None,
        args.joint_jet_action,
        args.joint_jet_law_action,
        args.joint_bundle_action,
        args.strict_joint_bundle_action,
        args.line_search,
        args.local_joint_transport,
        args.causal_support_joint_transport,
        args.strict_joint_graph_gradient_flow,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "condition_summary": result["condition_summary"],
        "ceiling_hits": result["ceiling_hits"],
        "maximum_target_self_lineage": result[
            "maximum_target_self_lineage"],
    }, indent=2))


if __name__ == "__main__":
    main()
