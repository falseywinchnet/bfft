"""Layer-3R real-only coset-evaluation reference.

This is not the full QHSO arbitrary-N algorithm. It is the routing primitive that
replaces complex branch FFTs. It proves the required hook:

    complex coset evaluation = two real folded planes + two real rfft passes

The implementation target is a fused two-plane Bruun rfft, carrying residues as
real coefficients and combining only at the public rfft output boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _rfft_full(y: np.ndarray) -> np.ndarray:
    """Full complex FFT reconstructed from a real rfft result."""
    y = np.asarray(y, dtype=np.float64)
    C = y.size
    half = np.fft.rfft(y)
    out = np.empty(C, dtype=np.complex128)
    out[: C // 2 + 1] = half
    if C > 1:
        out[C // 2 + 1 :] = np.conj(half[1 : C // 2][::-1])
    return out


@dataclass(frozen=True)
class RealPairedBranch:
    M: int
    base: int
    stride: int
    C: int

    def columns(self) -> np.ndarray:
        return (self.base + self.stride * np.arange(self.C)) % self.M

    def conjugate_columns(self) -> np.ndarray:
        return (-self.columns()) % self.M


def real_plane_fold(x: np.ndarray, br: RealPairedBranch) -> tuple[np.ndarray, np.ndarray]:
    """Fold real input onto two real modulation planes for one paired coset."""
    x = np.asarray(x, dtype=np.float64)
    u = np.zeros(br.C, dtype=np.float64)
    v = np.zeros(br.C, dtype=np.float64)
    theta = -2.0 * np.pi * br.base / br.M
    for m, xm in enumerate(x):
        r = m % br.C
        u[r] += xm * np.cos(theta * m)
        v[r] += xm * np.sin(theta * m)
    return u, v


def coset_eval_real_only(x: np.ndarray, br: RealPairedBranch) -> np.ndarray:
    """Evaluate p at omega^(base + stride*j) using only real branch inputs."""
    u, v = real_plane_fold(x, br)
    Fu = _rfft_full(u)
    Fv = _rfft_full(v)
    return Fu + 1j * Fv


def coset_eval_direct(x: np.ndarray, br: RealPairedBranch) -> np.ndarray:
    """Direct dense reference for the same coset."""
    x = np.asarray(x, dtype=np.float64)
    omega = np.exp(-2j * np.pi / br.M)
    cols = br.columns()
    m = np.arange(x.size)
    return np.array([np.sum(x * (omega ** (q * m))) for q in cols], dtype=np.complex128)


def conjugate_coset_from_real_pair(vals: np.ndarray, br: RealPairedBranch) -> np.ndarray:
    """For real input, derive the conjugate coset values without another branch."""
    # vals[j] is at q_j = base + stride*j.  The conjugate roots appear in the
    # reversed order determined by -q_j mod M.  Values are conjugates.
    return np.conj(vals)


def check_branch(T: int, M: int, base: int, stride: int, seed: int = 0) -> dict[str, float | int]:
    rng = np.random.default_rng(seed + 1009 * T + 17 * M + base)
    x = rng.standard_normal(T)
    C = M // stride
    br = RealPairedBranch(M=M, base=base, stride=stride, C=C)
    got = coset_eval_real_only(x, br)
    ref = coset_eval_direct(x, br)
    rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1.0)
    mx = np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1.0)
    return {"T": T, "M": M, "base": base, "stride": stride, "C": C, "relerr": float(rel), "maxerr": float(mx)}


def sweep() -> None:
    rows = []
    for M in [16, 32, 64, 128, 512]:
        P = M // 2
        C = P // 2
        # Odd paired half-coset A/B: A is base 1 stride 4, B is its conjugate.
        for T in [P + 1, P + C // 2, P + C, M - 1]:
            if 1 <= T < M:
                rows.append(check_branch(T, M, base=1, stride=4, seed=3))
        # A generic real paired coset as an additional sanity check.
        rows.append(check_branch(P + 1, M, base=3, stride=8 if M >= 32 else 4, seed=4))
    worst = max(rows, key=lambda r: r["relerr"])
    for r in rows:
        print(r)
    print("worst", worst)


if __name__ == "__main__":
    sweep()
