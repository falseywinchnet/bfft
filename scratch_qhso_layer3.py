"""Layer-3 QHSO recursive quotient-state reference.

This file is the first concrete replacement for the dense completion solve.
It computes the correction state directly from the input samples, not from the
unknown target FFT outputs.

It is still a reference, not the final SIMD kernel. The expensive part is the
state update quotient_adjoint_state_stream(), which is written as a direct
stream over input samples and active carrier columns. Its kernel is the exact
rank-one-displacement/Dirichlet state. Production replaces that update with a
blocked Chebyshev/displacement recurrence. Everything after that update is
already in power-of-two FFT form plus tiny local row solves.

Forward convention:
    X[k] = sum_m x[m] exp(-2*pi*i*k*m/T)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def prev_pow2(n: int) -> int:
    return 1 << (n.bit_length() - 1)


@dataclass(frozen=True)
class Branch:
    name: str
    M: int
    base: int
    stride: int
    C: int

    def columns(self) -> np.ndarray:
        return self.base + self.stride * np.arange(self.C)


@dataclass(frozen=True)
class QHSOStatePlan:
    T: int
    P: int
    M: int
    s: int
    h: int
    branches: tuple[Branch, ...]

    def columns(self) -> np.ndarray:
        return np.concatenate([b.columns() for b in self.branches])


def plan_layer3(T: int, force_two_halves: bool = False) -> QHSOStatePlan | None:
    if T < 1:
        raise ValueError("T must be positive")
    if is_pow2(T):
        return None
    P = prev_pow2(T)
    M = 2 * P
    s = T - P
    h = P // 2
    branches = [Branch("even", M, 0, 2, P), Branch("odd_A", M, 1, 4, h)]
    if force_two_halves or s > h:
        branches.append(Branch("odd_B", M, 3, 4, h))
    return QHSOStatePlan(T=T, P=P, M=M, s=s, h=h, branches=tuple(branches))


def _geometric_sum(r, n: int):
    r_arr = np.asarray(r, dtype=np.complex128)
    out = np.empty_like(r_arr)
    near = np.abs(r_arr - 1.0) < 1e-12
    out[near] = n
    out[~near] = (1.0 - r_arr[~near] ** n) / (1.0 - r_arr[~near])
    if np.isscalar(r):
        return complex(out.reshape(()))
    return out


def quotient_adjoint_state_stream(x: np.ndarray, plan: QHSOStatePlan) -> list[np.ndarray]:
    x = np.asarray(x, dtype=np.complex128)
    T, M = plan.T, plan.M
    tau = np.exp(-2j * np.pi / T)
    omega = np.exp(-2j * np.pi / M)
    states: list[np.ndarray] = []
    for br in plan.branches:
        qcols = br.columns()
        qroots = omega ** qcols
        b = np.zeros(br.C, dtype=np.complex128)
        for m, xm in enumerate(x):
            rt = (tau ** m) / qroots
            rc = (omega ** m) / qroots
            b += xm * (_geometric_sum(rt, T) - _geometric_sum(rc, T))
        states.append(b)
    return states


def synthesize_rows_from_state(states: list[np.ndarray], plan: QHSOStatePlan) -> np.ndarray:
    T, M = plan.T, plan.M
    omega = np.exp(-2j * np.pi / M)
    rows = np.arange(T)
    w = np.zeros(T, dtype=np.complex128)
    for br, b in zip(plan.branches, states):
        fb = np.fft.fft(b)
        w += (omega ** (br.base * rows)) * fb[rows % br.C]
    return w


def apply_row_gram_inverse(w: np.ndarray, plan: QHSOStatePlan) -> np.ndarray:
    w = np.asarray(w, dtype=np.complex128)
    T, P, M, s, h = plan.T, plan.P, plan.M, plan.s, plan.h
    if len(plan.branches) == 3:
        return w / M
    y = np.zeros_like(w)
    G2 = h * np.array([[3.0, 1.0j], [-1.0j, 3.0]], dtype=np.complex128)
    G3 = h * np.array(
        [[3.0, 1.0j, 1.0], [-1.0j, 3.0, 1.0j], [1.0, -1.0j, 3.0]],
        dtype=np.complex128,
    )
    inv2 = np.linalg.inv(G2)
    inv3 = np.linalg.inv(G3)
    for r in range(h):
        idx = [r, r + h]
        if r < s:
            idx.append(r + P)
            y[idx] = inv3 @ w[idx]
        else:
            y[idx] = inv2 @ w[idx]
    return y


def visible_carrier_rows(x: np.ndarray, M: int, T: int) -> np.ndarray:
    z = np.zeros(M, dtype=np.complex128)
    z[:T] = x
    return np.fft.fft(z)[:T]


def qhso_layer3_fft(x: np.ndarray, force_two_halves: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    T = x.size
    if is_pow2(T):
        return np.fft.fft(x)
    plan = plan_layer3(T, force_two_halves=force_two_halves)
    assert plan is not None
    visible = visible_carrier_rows(x, plan.M, plan.T)
    state = quotient_adjoint_state_stream(x, plan)
    gram_d = synthesize_rows_from_state(state, plan)
    correction = apply_row_gram_inverse(gram_d, plan)
    return visible + correction


def check(T: int, seed: int = 0, force_two_halves: bool = False):
    rng = np.random.default_rng(seed + 1009 * T)
    x = rng.standard_normal(T) + 1j * rng.standard_normal(T)
    y = qhso_layer3_fft(x, force_two_halves=force_two_halves)
    ref = np.fft.fft(x)
    rel = np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1.0)
    mx = np.max(np.abs(y - ref)) / max(np.max(np.abs(ref)), 1.0)
    plan = plan_layer3(T, force_two_halves=force_two_halves)
    if plan is None:
        return {"T": T, "relerr": float(rel), "maxerr": float(mx), "branches": 0, "mode": "pow2"}
    return {
        "T": T, "P": plan.P, "M": plan.M, "s": plan.s,
        "branches": len(plan.branches),
        "cols": int(sum(b.C for b in plan.branches)),
        "relerr": float(rel), "maxerr": float(mx),
        "mode": "two_halves" if force_two_halves or len(plan.branches) == 3 else "sparse_one_half",
    }


def sweep() -> None:
    tests = list(range(2, 65)) + [97, 127, 128, 129, 191, 193, 251, 257]
    worst = None
    for T in tests:
        row = check(T, seed=3)
        if worst is None or row["relerr"] > worst["relerr"]:
            worst = row
        print(row)
    print("worst", worst)


if __name__ == "__main__":
    sweep()
