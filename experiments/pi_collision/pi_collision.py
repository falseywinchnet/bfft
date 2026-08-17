"""Geometric tangent/Richardson approximations to pi.

This module keeps a deliberately clear mpmath implementation next to the C++
benchmark.  The important routine is ``pi_collision_fused``: it streams the
closed-form q-binomial weights alongside the tangent multiple-angle transport,
so it uses O(1) live multiprecision values instead of an O(N**2) triangle.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import mpmath as mp


def mul(z: tuple[mp.mpf, mp.mpf], w: tuple[mp.mpf, mp.mpf]):
    """Multiply complex pairs using real arithmetic only."""
    a, b = z
    c, d = w
    return a * c - b * d, a * d + b * c


def tangent_multiple(t: mp.mpf, m: int) -> mp.mpf:
    """Return tan(m*atan(t)) using an addition chain for (1+i*t)**m."""
    if m < 1:
        raise ValueError("m must be positive")
    if m == 1:
        return t
    if m == 2:
        return (2 * t) / (1 - t * t)
    if m == 3:
        t2 = t * t
        return t * (3 - t2) / (1 - 3 * t2)

    out = (mp.mpf(1), mp.mpf(0))
    base = (mp.mpf(1), t)
    exponent = m
    while exponent:
        if exponent & 1:
            out = mul(out, base)
        exponent >>= 1
        if exponent:
            base = mul(base, base)
    return out[1] / out[0]


def tangent_tower(t: mp.mpf, m: int, depth: int) -> mp.mpf:
    """Return D_{m**depth}(t) by composing D_m, avoiding a huge integer."""
    for _ in range(depth):
        t = tangent_multiple(t, m)
    return t


def smallest_root(m: int, depth: int, dps: int) -> mp.mpf:
    """Solve D_{m**depth}(t)=1 on the first branch.

    Newton is staged at geometrically increasing precision.  That avoids an
    otherwise unnecessary log(precision) multiplier on the expensive tower.
    """
    if m < 2 or depth < 0:
        raise ValueError("require m >= 2 and depth >= 0")

    target_dps = dps + 20
    stages: list[int] = []
    precision = min(50, target_dps)
    while precision < target_dps:
        stages.append(precision)
        precision = min(target_dps, 2 * precision)
    stages.append(target_dps)

    M = mp.mpf(m) ** depth
    x = None
    for stage_index, precision in enumerate(stages):
        mp.mp.dps = precision
        M = mp.mpf(m) ** depth
        if x is None:
            x = mp.mpf(7) / (8 * M)
            iterations = 7
        else:
            x = +x
            iterations = 2

        for _ in range(iterations):
            y = tangent_tower(x, m, depth)
            correction = (y - 1) * (1 + x * x) / (M * (1 + y * y))
            x -= correction

    mp.mp.dps = dps
    return +x


def closed_weights(m: int, depth: int) -> list[mp.mpf]:
    """Return interpolation weights for nodes r**j, r=m**-2, j=0..N."""
    r = mp.mpf(1) / (m * m)
    qpoch = [mp.mpf(1)]
    power = r
    for _ in range(depth):
        qpoch.append(qpoch[-1] * (1 - power))
        power *= r

    weights = []
    for j in range(depth + 1):
        s = depth - j
        weights.append(
            (-1) ** s * r ** (s * (s + 1) // 2) / (qpoch[j] * qpoch[s])
        )
    return weights


def pi_collision_triangle(m: int, depth: int, dps: int) -> mp.mpf:
    """Original O(N**2) Richardson triangle, retained as an oracle."""
    mp.mp.dps = dps
    M = mp.mpf(m) ** depth
    x = smallest_root(m, depth, dps)
    approximations = [None] * (depth + 1)
    for j in range(depth, -1, -1):
        approximations[j] = 4 * mp.mpf(m) ** j * x
        if j:
            x = tangent_multiple(x, m)

    for k in range(1, depth + 1):
        q = mp.mpf(m) ** (2 * k)
        approximations = [
            (q * approximations[j + 1] - approximations[j]) / (q - 1)
            for j in range(len(approximations) - 1)
        ]
    return approximations[0]


def pi_collision_weighted(m: int, depth: int, dps: int) -> mp.mpf:
    """O(N) closed weights, materialized for validation."""
    mp.mp.dps = dps
    x = smallest_root(m, depth, dps)
    values = [None] * (depth + 1)
    for j in range(depth, -1, -1):
        values[j] = 4 * mp.mpf(m) ** j * x
        if j:
            x = tangent_multiple(x, m)
    return mp.fsum(w * value for w, value in zip(closed_weights(m, depth), values))


def pi_collision_fused(m: int, depth: int, dps: int) -> mp.mpf:
    """Stream q-binomial weights and D_m transports in O(N) time/O(1) space."""
    mp.mp.dps = dps + 20
    r = mp.mpf(1) / (m * m)
    M = mp.mpf(m) ** depth
    x = smallest_root(m, depth, dps + 20)

    # w_0 here is lambda_N: s=0 corresponds to the deepest node j=N.
    qpoch = mp.mpf(1)
    r_power = r
    for _ in range(depth):
        qpoch *= 1 - r_power
        r_power *= r

    weight = 1 / qpoch
    scale = 4 * M
    head_power = r
    tail_power = r**depth
    total = mp.mpf(0)

    for s in range(depth + 1):
        total += weight * scale * x
        if s == depth:
            break
        x = tangent_multiple(x, m)
        weight *= -head_power * (1 - tail_power) / (1 - head_power)
        scale /= m
        head_power *= r
        tail_power /= r

    mp.mp.dps = dps
    return +total


def recommended_depth(bits: int, m: int, guard: int = 3) -> int:
    """Conservative depth from the leading exp(-N**2 log(m)) error."""
    if bits < 2 or m < 2:
        raise ValueError("require bits >= 2 and m >= 2")
    return math.ceil(math.sqrt(bits / math.log2(m))) + guard


def _correct_bits(value: mp.mpf) -> float:
    # Recompute pi above value's storage precision; comparing at the same
    # precision can make two identically rounded values look exactly equal.
    with mp.workdps(mp.mp.dps + 30):
        error = abs(value - mp.pi)
        return math.inf if not error else float(-mp.log(error, 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--digits", type=int, default=1000)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--validate-triangle", action="store_true")
    args = parser.parse_args()

    bits = math.ceil(args.digits * math.log2(10))
    depth = args.depth or recommended_depth(bits, args.m)
    started = time.perf_counter()
    value = pi_collision_fused(args.m, depth, args.digits + 20)
    elapsed = time.perf_counter() - started
    result = {
        "m": args.m,
        "depth": depth,
        "digits": args.digits,
        "correct_bits": _correct_bits(value),
        "seconds": elapsed,
    }
    if args.validate_triangle:
        triangle = pi_collision_triangle(args.m, depth, args.digits + 20)
        weighted = pi_collision_weighted(args.m, depth, args.digits + 20)
        result["fused_vs_triangle"] = mp.nstr(abs(value - triangle), 8)
        result["weighted_vs_triangle"] = mp.nstr(abs(weighted - triangle), 8)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
