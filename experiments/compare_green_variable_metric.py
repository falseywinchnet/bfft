"""Variable-metric Green-admission destination reference.

This probe builds a monotone Selling-stencil discretization of
rho - Delta_M, solves its two right-hand sides by one direct sparse
factorization, and compares the raw Green ratio and its exactly cardinal
weighted-harmonic form with local Q1 admission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from experiments.compare_eikonal_convstar import _cases, _scene
from experiments.conv_distilled_core import (
    nodal_current_geometry,
    reverse_conv_resize,
)
from experiments.eikonal_markov_spectral_probe import selling_stencil
from standalone_conv_resize_demo.backend import (
    conv_basin_average,
    conv_resize,
    linear_resize,
)


Array = np.ndarray


def _target_metric(values: Array, target: tuple[int, int]) -> tuple[Array, Array, Array]:
    field = np.asarray(values, dtype=np.float64)
    gxx, gxy, gyy, _ = nodal_current_geometry(field)
    trace = gxx + gyy
    inverse_trace = np.divide(
        1.0, trace, out=np.zeros_like(trace), where=trace > 0.0
    )
    qxx = 1.0 + gxx * inverse_trace
    qxy = gxy * inverse_trace
    qyy = 1.0 + gyy * inverse_trace
    root = np.sqrt(qxx * qyy - qxy * qxy)
    mxx = np.asarray(linear_resize((qxx / root).astype(np.float32), target), float)
    mxy = np.asarray(linear_resize((qxy / root).astype(np.float32), target), float)
    myy = np.asarray(linear_resize((qyy / root).astype(np.float32), target), float)
    interpolated_root = np.sqrt(mxx * myy - mxy * mxy)
    return mxx / interpolated_root, mxy / interpolated_root, myy / interpolated_root


def _metric_graph(
    metric: tuple[Array, Array, Array],
    spacing: tuple[float, float],
) -> sparse.csr_matrix:
    mxx, mxy, myy = metric
    height, width = mxx.shape
    hmat = np.diag(spacing)
    edges: dict[tuple[int, int], float] = {}

    def node(y: int, x: int) -> int:
        return y * width + x

    for y in range(height):
        for x in range(width):
            matrix = np.array(
                ((mxx[y, x], mxy[y, x]), (mxy[y, x], myy[y, x])),
                dtype=float,
            )
            pulled_metric = hmat.T @ matrix @ hmat
            directions, coefficients = selling_stencil(
                np.linalg.inv(pulled_metric)
            )
            first = node(y, x)
            for direction, coefficient in zip(directions, coefficients):
                if coefficient <= 0.0:
                    continue
                dx, dy = int(direction[0]), int(direction[1])
                for sign in (-1, 1):
                    xx, yy = x + sign * dx, y + sign * dy
                    if not (0 <= xx < width and 0 <= yy < height):
                        continue
                    second = node(yy, xx)
                    edge = (min(first, second), max(first, second))
                    edges[edge] = edges.get(edge, 0.0) + 0.5 * float(coefficient)

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for (first, second), weight in edges.items():
        rows.extend((first, second))
        columns.extend((second, first))
        data.extend((weight, weight))
    count = height * width
    adjacency = sparse.coo_matrix(
        (data, (rows, columns)), shape=(count, count)
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return sparse.diags(degree) - adjacency


def _source_nodes(
    source_shape: tuple[int, int], target: tuple[int, int]
) -> Array:
    y = np.rint(np.linspace(0, target[0] - 1, source_shape[0])).astype(int)
    x = np.rint(np.linspace(0, target[1] - 1, source_shape[1])).astype(int)
    if not (
        np.allclose(y, np.linspace(0, target[0] - 1, source_shape[0]))
        and np.allclose(x, np.linspace(0, target[1] - 1, source_shape[1]))
    ):
        raise ValueError("reference requires a nested target lattice")
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return (yy * target[1] + xx).reshape(-1)


def _green_coordinates(
    graph: sparse.csr_matrix,
    eta: Array,
    target: tuple[int, int],
) -> tuple[Array, Array, dict[str, float]]:
    count = target[0] * target[1]
    source = _source_nodes(eta.shape, target)
    labels = np.asarray(eta, dtype=float).reshape(-1)
    density = 1.0
    operator = (graph + density * sparse.eye(count, format="csr")).tocsc()
    factor = splu(operator)
    rhs_u = np.zeros(count, dtype=float)
    rhs_v = np.zeros(count, dtype=float)
    rhs_u[source] = 1.0
    rhs_v[source] = labels
    u = factor.solve(rhs_u)
    v = factor.solve(rhs_v)
    ratio = v / u

    # Exact discrete counterpart of div(U^2 A grad beta)=0.
    coo = graph.tocoo()
    mask = coo.row < coo.col
    first = coo.row[mask]
    second = coo.col[mask]
    base = -coo.data[mask]
    positive = base > 0.0
    first, second, base = first[positive], second[positive], base[positive]
    conductance = base * 0.5 * (u[first] ** 2 + u[second] ** 2)
    rows = np.concatenate((first, second))
    columns = np.concatenate((second, first))
    data = np.concatenate((conductance, conductance))
    adjacency = sparse.coo_matrix(
        (data, (rows, columns)), shape=(count, count)
    ).tocsr()
    weighted = sparse.diags(
        np.asarray(adjacency.sum(axis=1)).reshape(-1)
    ) - adjacency
    fixed = np.zeros(count, dtype=bool)
    fixed[source] = True
    free = np.flatnonzero(~fixed)
    cardinal = np.empty(count, dtype=float)
    cardinal[source] = labels
    cardinal[free] = splu(weighted[free][:, free].tocsc()).solve(
        -weighted[free][:, source] @ labels
    )
    tolerance = 2.0e-10
    if float(np.min(cardinal)) < float(np.min(labels)) - tolerance:
        raise RuntimeError("weighted Green coordinate violated its minimum principle")
    if float(np.max(cardinal)) > float(np.max(labels)) + tolerance:
        raise RuntimeError("weighted Green coordinate violated its maximum principle")
    return (
        ratio.reshape(target),
        cardinal.reshape(target),
        {
            "raw_ratio_source_linf": float(np.max(np.abs(
                ratio[source] - labels
            ))),
            "cardinal_source_linf": float(np.max(np.abs(
                cardinal[source] - labels
            ))),
            "minimum_u": float(np.min(u)),
            "maximum_u": float(np.max(u)),
        },
    )


def _mse(first: Array, second: Array) -> float:
    residual = np.asarray(first, float) - np.asarray(second, float)
    return float(np.mean(residual * residual))


def _geomean(values: list[float]) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(values, 1.0e-300)))))


def run(source_sides: tuple[int, ...], target_side: int) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for source_side in source_sides:
        if (target_side - 1) % (source_side - 1):
            raise ValueError("target side must nest every source side")
        for case in _cases():
            truth = _scene(
                str(case["kind"]),
                target_side,
                angle=float(case["angle"]),
                phase=float(case["phase"]),
                offset=float(case["offset"]),
            ).astype(np.float32)
            coarse = conv_basin_average(
                truth, (source_side, source_side)
            ).astype(np.float32)
            forward = np.asarray(conv_resize(coarse, truth.shape), float)
            reverse = np.asarray(reverse_conv_resize(coarse, truth.shape), float)
            eta = nodal_current_geometry(coarse)[3]
            local_beta = np.asarray(
                linear_resize(eta.astype(np.float32), truth.shape), float
            )
            metric = _target_metric(coarse, truth.shape)
            spacing = (
                (source_side - 1) / (target_side - 1),
                (source_side - 1) / (target_side - 1),
            )
            graph = _metric_graph(metric, spacing)
            ratio_beta, cardinal_beta, audit = _green_coordinates(
                graph, eta, truth.shape
            )
            outputs = {
                "local": (1.0 - local_beta) * forward + local_beta * reverse,
                "green_ratio": (
                    (1.0 - ratio_beta) * forward + ratio_beta * reverse
                ),
                "green_cardinal": (
                    (1.0 - cardinal_beta) * forward + cardinal_beta * reverse
                ),
            }
            records.append({
                "source_side": source_side,
                "target_side": target_side,
                "case": case,
                "audit": audit,
                "truth_mse": {
                    name: _mse(output, truth) for name, output in outputs.items()
                },
                "beta_rms_from_local": {
                    "green_ratio": float(np.sqrt(_mse(
                        ratio_beta, local_beta
                    ))),
                    "green_cardinal": float(np.sqrt(_mse(
                        cardinal_beta, local_beta
                    ))),
                },
                "output_rms_from_local": {
                    name: float(np.sqrt(_mse(output, outputs["local"])))
                    for name, output in outputs.items() if name != "local"
                },
            })

    summary: dict[str, object] = {}
    for source_side in source_sides:
        group = [row for row in records if row["source_side"] == source_side]
        methods = ("local", "green_ratio", "green_cardinal")
        summary[str(source_side)] = {
            "case_count": len(group),
            "truth_geomean_mse": {
                method: _geomean([
                    float(row["truth_mse"][method]) for row in group
                ])
                for method in methods
            },
            "wins_against_local": {
                method: sum(
                    float(row["truth_mse"][method])
                    < float(row["truth_mse"]["local"])
                    for row in group
                )
                for method in methods[1:]
            },
            "beta_rms_from_local": {
                method: float(np.sqrt(np.mean([
                    float(row["beta_rms_from_local"][method]) ** 2
                    for row in group
                ])))
                for method in methods[1:]
            },
            "output_rms_from_local": {
                method: float(np.sqrt(np.mean([
                    float(row["output_rms_from_local"][method]) ** 2
                    for row in group
                ])))
                for method in methods[1:]
            },
            "raw_ratio_maximum_source_linf": max(
                float(row["audit"]["raw_ratio_source_linf"]) for row in group
            ),
            "cardinal_maximum_source_linf": max(
                float(row["audit"]["cardinal_source_linf"]) for row in group
            ),
        }
    return {
        "definition": {
            "operator": "rho - discrete monotone Selling Laplace-Beltrami",
            "rho": 1.0,
            "boundary": "missing graph edges, equivalent to no flux",
            "solver": "one direct sparse LU for U,V; one direct sparse LU for cardinal weighted harmonic beta",
            "role": "destination reference, not a production implementation",
        },
        "summary": summary,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sides", default="9,17")
    parser.add_argument("--target-side", type=int, default=33)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "output/support_geometry/green_variable_metric_reference.json"
        ),
    )
    args = parser.parse_args()
    result = run(
        tuple(int(value) for value in args.source_sides.split(",")),
        args.target_side,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
