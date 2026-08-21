"""Matched tangent-quadrature convergence study for the joint image law."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fmmt_certified import denoise_fmmt
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt
from .witnessed_characteristic_transport_2d import joint_characteristic_measure_2d


CONDITIONS = (
    ("clean", None, 0.0, 0.0),
    ("uniform 0.10", "uniform additive", 0.10, 0.25),
    ("replacement 0.25", "random-value replacement", 0.10, 0.25),
    ("mixed 0.25", "mixed replacement + uniform", 0.10, 0.25),
)


def run(
    size: int,
    selected_sources: tuple[str, ...],
    orders: tuple[int, ...],
) -> dict:
    catalogue = sources(size)
    unknown = sorted(set(selected_sources) - set(catalogue))
    if unknown:
        raise ValueError(f"unknown sources: {unknown}")
    if len(orders) < 2 or tuple(sorted(set(orders))) != orders:
        raise ValueError("orders must be distinct and strictly increasing")
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        for condition, kind, amount, density in CONDITIONS:
            observation = (
                truth if kind is None else corrupt(
                    truth, kind, amount=amount, density=density, seed=19000)
            )
            fields = {}
            order_metrics = {}
            diagnostics = {}
            for order in orders:
                field, diagnostic = joint_characteristic_measure_2d(
                    observation,
                    barycenter="median",
                    angular_order=order,
                )
                fields[order] = field
                order_metrics[str(order)] = metrics(field, truth)
                diagnostics[str(order)] = {
                    "direction_count": diagnostic["signal_law"]["proposal"][
                        "direction_count"],
                    "proposal_count": diagnostic["signal_law"]["proposal"][
                        "proposal_count"],
                }
            differences = {}
            for previous, current in zip(orders, orders[1:]):
                delta = fields[current] - fields[previous]
                differences[f"{previous}->{current}"] = {
                    "rms": float(np.sqrt(np.mean(delta * delta))),
                    "maximum": float(np.max(np.abs(delta))),
                }
            rows.append({
                "source": source,
                "condition": condition,
                "orders": order_metrics,
                "quadrature": diagnostics,
                "successive_field_difference": differences,
                "integrated_fmmt": metrics(denoise_fmmt(observation)[0], truth),
            })
    metric_names = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention",
    )
    summary = {
        str(order): {
            metric: float(np.mean([
                row["orders"][str(order)][metric] for row in rows
            ]))
            for metric in metric_names
        }
        for order in orders
    }
    summary["integrated_fmmt"] = {
        metric: float(np.mean([row["integrated_fmmt"][metric] for row in rows]))
        for metric in metric_names
    }
    difference_summary = {
        pair: {
            "mean_rms": float(np.mean([
                row["successive_field_difference"][pair]["rms"]
                for row in rows
            ])),
            "maximum": float(np.max([
                row["successive_field_difference"][pair]["maximum"]
                for row in rows
            ])),
        }
        for pair in rows[0]["successive_field_difference"]
    }
    return {
        "purpose": (
            "measure numerical tangent-sphere convergence without selecting an order"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "orders": list(orders),
        "summary": summary,
        "successive_field_difference": difference_summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument(
        "--sources",
        default="cameraman,tapered hair,geometric interfaces,woven chirps",
    )
    parser.add_argument("--orders", default="1,2,3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(value.strip() for value in args.sources.split(",") if value.strip())
    orders = tuple(int(value) for value in args.orders.split(",") if value.strip())
    result = run(args.size, selected, orders)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "field_difference": result["successive_field_difference"],
    }, indent=2))


if __name__ == "__main__":
    main()
