"""QHSO complete half-coset sidecar reference.

This is not the fast kernel. It is the small executable foundation an implementer
can use to verify the math before wiring the sidecar into an existing power-of-two
Bruun FFT.

It demonstrates four things:
  1. how to pick the complete half-coset sidecar branches for any non-pow2 T;
  2. how a coset branch is evaluated by phase-folding plus a power-of-two FFT;
  3. why the complete half-coset frame is metrologically safe;
  4. an exact dense oracle for the sidecar correction, used only for testing.

Production replaces dense_sidecar_oracle() with the recursive quotient-state
sidecar described in qhso_bruun_extension_brief.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def prev_pow2(n: int) -> int:
    return 1 << (n.bit_length() - 1)


def dft_matrix(n: int) -> np.ndarray:
    k, m = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.exp(-2j * np.pi * k * m / n)


@dataclass(frozen=True)
class CosetNode:
    M: int
    base: int
    stride: int
    C: int

    def columns(self) -> np.ndarray:
        return self.base + self.stride * np.arange(self.C)


@dataclass(frozen=True)
class QHSOPlan:
    T: int
    P: int
    M: int
    s: int
    ordinary: CosetNode
    sidecars: tuple[CosetNode, ...]


def plan_qhso(T: int) -> QHSOPlan | None:
    if T < 1:
        raise ValueError("T must be positive")
    if is_pow2(T):
        return None
    P = prev_pow2(T)
    M = 2 * P
    s = T - P
    h = P // 2
    ordinary = CosetNode(M=M, base=0, stride=2, C=P)
    sidecars = [CosetNode(M=M, base=1, stride=4, C=h)]
    if s > h:
        sidecars.append(CosetNode(M=M, base=3, stride=4, C=h))
    return QHSOPlan(T=T, P=P, M=M, s=s, ordinary=ordinary, sidecars=tuple(sidecars))


def coset_eval_by_pow2_fft(x: np.ndarray, node: CosetNode) -> np.ndarray:
    """Evaluate p(z)=sum x[m]z^m on z=omega_M^(base+stride*q).

    This is the implementation hook for an existing power-of-two FFT:
    phase-fold the coefficients into length C, then call a C-point pow2 FFT.
    """
    x = np.asarray(x, dtype=np.complex128)
    omega = np.exp(-2j * np.pi / node.M)
    folded = np.zeros(node.C, dtype=np.complex128)
    for m, xm in enumerate(x):
        folded[m % node.C] += xm * omega ** (node.base * m)
    return np.fft.fft(folded)


def direct_coset_eval(x: np.ndarray, node: CosetNode) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    omega = np.exp(-2j * np.pi / node.M)
    q = node.columns()
    m = np.arange(x.size)
    return (omega ** (q[:, None] * m[None, :])) @ x



def selected_completion_nodes(plan: QHSOPlan) -> tuple[CosetNode, ...]:
    """Completion frame uses ordinary branch plus complete active half-cosets."""
    return (plan.ordinary,) + plan.sidecars


def selected_completion_columns(plan: QHSOPlan) -> np.ndarray:
    return np.concatenate([n.columns() for n in selected_completion_nodes(plan)])


def completion_frame(plan: QHSOPlan) -> np.ndarray:
    """First T rows of selected carrier columns.

    The frame is deliberately overcomplete when only one half-coset is needed.
    This is the safe metrology object. It contains the ordinary P coset and the
    active complete odd half-cosets.
    """
    FM = dft_matrix(plan.M)
    return FM[:plan.T, selected_completion_columns(plan)]


def dense_completion_oracle(x: np.ndarray, plan: QHSOPlan) -> np.ndarray:
    """Exact correction by dense solve. Test oracle only.

    Computes c such that
        F_T x = F_M[:T,:T] x + completion_frame @ c.

    Production replaces this solve with a recursive quotient-state evaluator.
    """
    x = np.asarray(x, dtype=np.complex128)
    T, M = plan.T, plan.M
    FT = dft_matrix(T)
    FM = dft_matrix(M)
    visible = FM[:T, :T] @ x
    target = FT @ x
    A = completion_frame(plan)
    coeff, *_ = np.linalg.lstsq(A, target - visible, rcond=None)
    return A @ coeff


def qhso_dense_oracle_fft(x: np.ndarray) -> np.ndarray:
    """Arbitrary-T FFT oracle using the complete half-coset completion frame."""
    x = np.asarray(x, dtype=np.complex128)
    T = x.size
    if is_pow2(T):
        return np.fft.fft(x)
    plan = plan_qhso(T)
    assert plan is not None
    FM = dft_matrix(plan.M)
    visible = FM[:T, :T] @ x
    return visible + dense_completion_oracle(x, plan)


def half_coset_row_isometry_error(plan: QHSOPlan) -> float:
    """Check the active half-coset sidecar frame itself.

    This excludes the ordinary branch and checks the metrology property of the
    complete sidecar branch: after normalization, row drops remain contractions.
    """
    rows = np.arange(plan.s)[:, None]
    cols = np.concatenate([n.columns() for n in plan.sidecars])[None, :]
    scale = math.sqrt(plan.P // 2) if len(plan.sidecars) == 1 else math.sqrt(plan.P)
    A = np.exp(-2j * np.pi * rows * cols / plan.M) / scale
    return float(np.linalg.norm(A @ A.conj().T - np.eye(plan.s), 2))


def check(T: int, seed: int = 0) -> dict[str, float | int]:
    rng = np.random.default_rng(seed + T)
    x = rng.standard_normal(T) + 1j * rng.standard_normal(T)
    y = qhso_dense_oracle_fft(x)
    ref = np.fft.fft(x)
    rel = np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1.0)
    if is_pow2(T):
        return {"T": T, "relerr": float(rel), "branches": 0, "row_iso_err": 0.0}
    plan = plan_qhso(T)
    assert plan is not None
    coset_err = 0.0
    for node in selected_completion_nodes(plan):
        fast = coset_eval_by_pow2_fft(x, node)
        direct = direct_coset_eval(x, node)
        coset_err = max(coset_err, np.linalg.norm(fast - direct) / max(np.linalg.norm(direct), 1.0))
    A = completion_frame(plan)
    return {
        "T": T,
        "P": plan.P,
        "M": plan.M,
        "s": plan.s,
        "branches": len(plan.sidecars),
        "relerr": float(rel),
        "completion_cond": float(np.linalg.cond(A)),
        "sidecar_row_iso_err": half_coset_row_isometry_error(plan),
        "coset_eval_err": float(coset_err),
    }


if __name__ == "__main__":
    tests = list(range(2, 34)) + [47, 63, 64, 65, 97, 127]
    worst = None
    for T in tests:
        row = check(T, seed=1)
        if worst is None or row["relerr"] > worst["relerr"]:
            worst = row
        print(row)
    print("worst", worst)
