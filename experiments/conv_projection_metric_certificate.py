"""Exact rational Gram matrices for CONV's quintic current tails."""

from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path


def add(first: list[Fraction], second: list[Fraction]) -> list[Fraction]:
    n = max(len(first), len(second))
    return [
        (first[i] if i < len(first) else Fraction(0))
        + (second[i] if i < len(second) else Fraction(0))
        for i in range(n)
    ]


def multiply(first: list[Fraction], second: list[Fraction]) -> list[Fraction]:
    result = [Fraction(0)] * (len(first) + len(second) - 1)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            result[i + j] += a * b
    return result


def derivative(value: list[Fraction]) -> list[Fraction]:
    return [Fraction(i) * value[i] for i in range(1, len(value))]


def integral_unit(value: list[Fraction]) -> Fraction:
    return sum((coefficient / Fraction(i + 1) for i, coefficient in enumerate(value)), Fraction(0))


def bernstein(degree: int, index: int) -> list[Fraction]:
    # C(n,k) u^k (1-u)^(n-k)
    result = [Fraction(0)] * (degree + 1)
    for j in range(degree - index + 1):
        result[index + j] = Fraction(
            math.comb(degree, index) * math.comb(degree - index, j) * ((-1) ** j)
        )
    return result


def matrices() -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    basis = [bernstein(5, index) for index in range(6)]
    tails = []
    for k in range(5):
        tail = [Fraction(0)]
        for index in range(k + 1, 6):
            tail = add(tail, basis[index])
        tails.append(tail)
    w0 = [[integral_unit(multiply(a, b)) for b in tails] for a in tails]
    slopes = [derivative(tail) for tail in tails]
    w1 = [[integral_unit(multiply(a, b)) for b in slopes] for a in slopes]
    return w0, w1


def matmul(first: list[list[Fraction]], second: list[list[Fraction]]) -> list[list[Fraction]]:
    return [[
        sum((first[i][k] * second[k][j] for k in range(len(second))), Fraction(0))
        for j in range(len(second[0]))
    ] for i in range(len(first))]


def transpose(value: list[list[Fraction]]) -> list[list[Fraction]]:
    return [list(row) for row in zip(*value)]


def encode(matrix: list[list[Fraction]]) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def main() -> None:
    w0, w1 = matrices()
    reversal_w0 = all(w0[i][j] == w0[4 - i][4 - j] for i in range(5) for j in range(5))
    reversal_w1 = all(w1[i][j] == w1[4 - i][4 - j] for i in range(5) for j in range(5))
    reversal = [[Fraction(int(i + j == 4)) for j in range(5)] for i in range(5)]
    zero_sum = [
        [Fraction(int(i == j)) for j in range(4)]
        if i < 4 else [Fraction(-1)] * 4
        for i in range(5)
    ]
    reversed_w0 = matmul(matmul(reversal, w0), reversal)
    difference = [[reversed_w0[i][j] - w0[i][j] for j in range(5)] for i in range(5)]
    restricted = matmul(matmul(transpose(zero_sum), difference), zero_sum)
    constrained_reversal_w0 = all(value == 0 for row in restricted for value in row)
    result = {
        "W0": encode(w0),
        "W1": encode(w1),
        "reversal_symmetric": {
            "W0_full_space": reversal_w0,
            "W0_on_zero_sum_current_errors": constrained_reversal_w0,
            "W1_full_space": reversal_w1,
        },
    }
    target = Path("output/support_geometry/conv_projection_metric_certificate.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
