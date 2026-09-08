"""Resolve the record current by total prime-factor degree Omega(n)."""

from __future__ import annotations

import json
from math import exp, pi, sqrt
from pathlib import Path

import numpy as np

from mellin_current import dirichlet_coefficients, prime_sieve
from run_packet_probe import simpson
from run_probe import CONTROL, RECORD


def omega_table(limit: int) -> np.ndarray:
    _, spf = prime_sieve(limit)
    omega = np.zeros(limit + 1, dtype=np.int16)
    for n in range(2, limit + 1):
        omega[n] = omega[n // spf[n]] + 1
    return omega


def degree_report(curve, limit: int) -> dict[str, object]:
    coefficients = dirichlet_coefficients(curve, limit)
    omega = omega_table(limit)
    maximum_degree = int(np.max(omega))
    n = np.arange(1, limit + 1, dtype=np.float64)
    a = coefficients[1:].astype(np.float64)
    degrees = omega[1:]
    t = np.linspace(0.0, 11.0, 2201)
    currents = np.zeros((maximum_degree + 1, len(t)), dtype=np.float64)
    alpha = 2.0 * pi / sqrt(curve.conductor)
    for ti, value in enumerate(t):
        et = exp(float(value))
        terms = et * a * np.exp(-alpha * et * n)
        currents[:, ti] = np.bincount(
            degrees,
            weights=terms,
            minlength=maximum_degree + 1,
        )
    spacing = float(t[1] - t[0])
    groups = {}
    for degree in range(maximum_degree + 1):
        groups[str(degree)] = {
            "coefficient_count": int(np.sum(degrees == degree)),
            "moments": {
                str(moment): float(
                    2.0 * simpson(currents[degree] * t**moment, spacing)
                )
                for moment in (1, 3, 5)
            },
        }
    return {
        "curve": curve.name,
        "conductor": curve.conductor,
        "limit": limit,
        "groups": groups,
        "moment_sums": {
            str(moment): sum(
                group["moments"][str(moment)] for group in groups.values()
            )
            for moment in (1, 3, 5)
        },
    }


def main() -> None:
    output = {
        "record": degree_report(RECORD, 30_000),
        "control": degree_report(CONTROL, 2_000),
    }
    target = Path("/tmp/elliptic_euler_degree.json")
    target.write_text(json.dumps(output, indent=2))
    for name, report in output.items():
        print(name)
        for degree, group in report["groups"].items():
            moments = group["moments"]
            if max(abs(value) for value in moments.values()) > 1e-7:
                print("  Omega", degree, moments)
        print("  sums", report["moment_sums"])
    print("wrote", target)


if __name__ == "__main__":
    main()

