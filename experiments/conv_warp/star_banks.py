"""Exact-rational state-conditioned FIR and 2-D phase-bank compilation.

A proof implementation: admission discovers the state; the returned bank is
then one linear contraction. Uses the existing exact CONV admission oracle.
"""
from fractions import Fraction as F
from functools import lru_cache
from experiments.conv_exact_fusion.polynomial_proof import (
    poly, raw_bank, scalar_profile, tails, evaluate,
)


@lru_cache(None)
def linear_stencils(length, cell):
    raw, delta = [], []
    for node in range(length):
        a, d = raw_bank([poly([int(k == node)]) for k in range(length)])
        raw.append(tuple(v[0] for v in a[cell]))
        delta.append(d[cell][0])
    return tuple(raw), tuple(delta)


def phase_stencil(signal, cell, phase):
    signal = list(map(F, signal))
    active = [k for k, p in enumerate(scalar_profile(signal)[cell]) if p[0]]
    weights = [F(k == cell) for k in range(len(signal))]
    if not active:
        return weights
    tau = [evaluate(t, phase) for t in tails()]
    mean = sum(tau[k] for k in active) / len(active)
    raw, delta = linear_stencils(len(signal), cell)
    for node in range(len(signal)):
        weights[node] += mean*delta[node] + sum((tau[k]-mean)*raw[node][k] for k in active)
    return weights


def contract(weights, values):
    return sum(a*b for a, b in zip(weights, values))


def compile_xy(rows, i, j, u, v):
    first = [phase_stencil(row, i, u) for row in rows]
    intermediate = [contract(w, row) for w, row in zip(first, rows)]
    second = phase_stencil(intermediate, j, v)
    return [[second[y]*weight for weight in first[y]] for y in range(len(rows))]


def compile_blend(rows, i, j, u, v, beta):
    xy = compile_xy(rows, i, j, u, v)
    yx = list(zip(*compile_xy(list(zip(*rows)), j, i, v, u)))
    return [[(1-beta)*a+beta*b for a, b in zip(arow, brow)] for arow, brow in zip(xy, yx)]


def contract_2d(bank, rows):
    return sum(contract(w, row) for w, row in zip(bank, rows))
