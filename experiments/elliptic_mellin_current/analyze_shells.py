"""Extract the three-shell cancellation law from a saved current probe."""

from __future__ import annotations

import json
from math import exp, pi, sqrt
from pathlib import Path

import numpy as np


def linear_root(x0: float, y0: float, x1: float, y1: float) -> float:
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def main() -> None:
    source = Path(__file__).with_name("elliptic_mellin_current.json")
    report = json.loads(source.read_text())["record"]
    t = np.asarray(report["t"], dtype=float)
    current = np.asarray(report["current"], dtype=float)
    roots = []
    for index in range(1, len(current)):
        if t[index] <= 0.02:
            continue
        if current[index - 1] * current[index] < 0:
            roots.append(
                linear_root(
                    t[index - 1],
                    current[index - 1],
                    t[index],
                    current[index],
                )
            )
    # Insert the linearly located zeroes so the three shell quadratures form
    # an exact partition of the sampled interval rather than dropping the two
    # grid cells crossed by a root.
    t = np.concatenate((t, np.asarray(roots)))
    current = np.concatenate((current, np.zeros(len(roots))))
    order = np.argsort(t)
    t = t[order]
    current = current[order]
    bounds = [0.0, *roots, float(t[-1])]
    alpha = 2.0 * pi / sqrt(float(report["conductor"]))
    output = {
        "roots": roots,
        "dimensionless_s_roots": [alpha * exp(root) for root in roots],
        "alpha": alpha,
        "shell_moments": {},
    }
    for degree in (1, 3, 5):
        pieces = []
        for left, right in zip(bounds, bounds[1:]):
            selected = (t >= left) & (t <= right)
            pieces.append(
                float(
                    np.trapezoid(
                        2.0 * current[selected] * t[selected] ** degree,
                        t[selected],
                    )
                )
            )
        output["shell_moments"][str(degree)] = {
            "pieces": pieces,
            "sum": sum(pieces),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
