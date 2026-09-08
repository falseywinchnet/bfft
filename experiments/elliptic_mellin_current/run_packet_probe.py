"""Resolve the record's Mellin moments by dyadic Hecke coefficient packets."""

from __future__ import annotations

import json
from math import exp, pi, sqrt
from pathlib import Path

import numpy as np

from mellin_current import dirichlet_coefficients
from run_probe import RECORD


def simpson(values: np.ndarray, spacing: float) -> float:
    return spacing / 3.0 * (
        values[0]
        + values[-1]
        + 4.0 * np.sum(values[1:-1:2])
        + 2.0 * np.sum(values[2:-1:2])
    )


def main() -> None:
    limit = 30_000
    coefficients = dirichlet_coefficients(RECORD, limit)
    n = np.arange(1, limit + 1, dtype=np.float64)
    a = coefficients[1:].astype(np.float64)
    edges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, limit + 1]
    labels = [f"{left}:{right - 1}" for left, right in zip(edges, edges[1:])]
    t = np.linspace(0.0, 11.0, 2201)
    packet_current = np.zeros((len(labels), len(t)), dtype=np.float64)
    alpha = 2.0 * pi / sqrt(RECORD.conductor)
    for ti, value in enumerate(t):
        et = exp(float(value))
        terms = et * a * np.exp(-alpha * et * n)
        for packet, (left, right) in enumerate(zip(edges, edges[1:])):
            packet_current[packet, ti] = np.sum(terms[left - 1 : right - 1])

    spacing = float(t[1] - t[0])
    output = {"packets": {}}
    for packet, label in enumerate(labels):
        output["packets"][label] = {
            str(degree): float(
                2.0 * simpson(packet_current[packet] * t**degree, spacing)
            )
            for degree in (1, 3, 5)
        }
    output["sums"] = {
        str(degree): sum(packet[str(degree)] for packet in output["packets"].values())
        for degree in (1, 3, 5)
    }
    target = Path("/tmp/elliptic_mellin_packets.json")
    target.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print("wrote", target)


if __name__ == "__main__":
    main()

