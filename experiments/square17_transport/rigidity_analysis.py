#!/usr/bin/env python3
"""Extract the connection-lifted local jamming stress for 17 squares."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geometry import PAIR_I, PAIR_J, capacity_state
from reference_chart import REFERENCE_SIDE, reference_chart


@dataclass
class ContactLinearization:
    operator: np.ndarray
    side_response: np.ndarray
    labels: list[dict[str, int | str]]


def coherent_contact_linearization(tolerance: float = 1.0e-9) -> ContactLinearization:
    """Build the active transport with one phase per coherent angle packet.

    The physical chart has 34 independent translations, one common angle for
    the six-square diagonal packet, and one angle for the remaining rotated
    square.  Axis-aligned phases stay at the cusp of their legal chart.
    """

    poses = reference_chart()
    state = capacity_state(poses, REFERENCE_SIDE)
    full_rows = []
    side_response = []
    labels: list[dict[str, int | str]] = []
    cosine = np.cos(poses[:, 2])
    sine = np.sin(poses[:, 2])
    half_width_prime = 0.5 * (
        -np.sign(cosine) * sine + np.sign(sine) * cosine
    )
    boundary_gradients = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    for square in range(17):
        for face, gradient in enumerate(boundary_gradients):
            if state.boundary_clearance[square, face] > tolerance:
                continue
            row = np.zeros(51, dtype=np.float64)
            row[3 * square:3 * square + 2] = gradient
            row[3 * square + 2] = -half_width_prime[square]
            full_rows.append(row)
            side_response.append(1.0 if face in (1, 3) else 0.0)
            labels.append({"kind": "boundary", "square": square, "face": face})
    for pair, (first, second) in enumerate(zip(PAIR_I, PAIR_J)):
        if state.pair_clearance[pair] > tolerance:
            continue
        row = np.zeros(51, dtype=np.float64)
        row[3 * first:3 * first + 2] = state.pair_center_i_gradient[pair]
        row[3 * first + 2] = state.pair_theta_i_gradient[pair]
        row[3 * second:3 * second + 2] = state.pair_center_j_gradient[pair]
        row[3 * second + 2] = state.pair_theta_j_gradient[pair]
        full_rows.append(row)
        side_response.append(0.0)
        labels.append({"kind": "pair", "first": int(first), "second": int(second)})

    full = np.asarray(full_rows)
    operator = np.zeros((len(full), 36), dtype=np.float64)
    for square in range(17):
        operator[:, 2 * square:2 * square + 2] = full[:, 3 * square:3 * square + 2]
    operator[:, 34] = sum(full[:, 3 * square + 2] for square in range(8, 14))
    operator[:, 35] = full[:, 3 * 15 + 2]
    return ContactLinearization(operator, np.asarray(side_response), labels)


def nonnegative_shrink_stress(
    linearization: ContactLinearization,
    *,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Find y>=0, A.T@y=0, and side_response.T@y=1.

    The left nullspace is three-dimensional for this chart.  After the unit
    work normalization, feasibility is a two-dimensional half-plane problem;
    enumerating its boundary intersections is deterministic and complete.
    """

    operator = linearization.operator
    side = linearization.side_response
    u, singular, _ = np.linalg.svd(operator, full_matrices=True)
    rank = int(np.sum(singular > 1.0e-9))
    null = u[:, rank:]
    work = null.T @ side
    base = work / float(np.dot(work, work))
    _, _, vh = np.linalg.svd(work.reshape(1, -1))
    free = vh[1:].T
    halfplanes = null @ free
    offset = null @ base
    best: np.ndarray | None = None
    for first in range(len(offset)):
        for second in range(first + 1, len(offset)):
            equations = np.stack((halfplanes[first], halfplanes[second]))
            determinant = float(np.linalg.det(equations))
            if abs(determinant) <= 1.0e-13:
                continue
            coordinate = np.linalg.solve(
                equations, -np.asarray([offset[first], offset[second]])
            )
            candidate = null @ (base + free @ coordinate)
            if float(np.min(candidate)) < -tolerance:
                continue
            candidate[np.abs(candidate) <= tolerance] = 0.0
            if best is None or np.linalg.norm(candidate) < np.linalg.norm(best):
                best = candidate
    if best is None:
        raise RuntimeError("active chart has no nonnegative shrink stress")
    return best


def analyze() -> dict[str, object]:
    chart = coherent_contact_linearization()
    stress = nonnegative_shrink_stress(chart)
    singular = np.linalg.svd(chart.operator, compute_uv=False)
    rank = int(np.sum(singular > 1.0e-9))
    equilibrium = chart.operator.T @ stress
    weighted_contacts = []
    for label, weight in zip(chart.labels, stress):
        record = dict(label)
        record["weight"] = float(weight)
        weighted_contacts.append(record)
    return {
        "status": "local_first_order_certificate_not_global_optimality_proof",
        "side": REFERENCE_SIDE,
        "active_contacts": len(chart.operator),
        "chart_dimension": chart.operator.shape[1],
        "operator_rank": rank,
        "left_stress_dimension": len(chart.operator) - rank,
        "smallest_singular_value": float(singular[-1]),
        "stress_positive_count": int(np.sum(stress > 1.0e-9)),
        "stress_zero_count": int(np.sum(np.abs(stress) <= 1.0e-9)),
        "stress_minimum": float(np.min(stress)),
        "stress_maximum": float(np.max(stress)),
        "equilibrium_residual": float(np.linalg.norm(equilibrium)),
        "shrink_work": float(np.dot(stress, chart.side_response)),
        "contacts": weighted_contacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze()
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
