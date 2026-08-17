#!/usr/bin/env python3
"""DIP2 research reference: a phase-axis Bruun remainder walk.

This is deliberately not an optimized version of ``bruun_dip_kernel.hpp``.
The old DIP packet is a dyadic Zak/comb state.  DIP2 starts from a different
invariant: Bruun's real quadratic factor is linear on the phase coordinate

    t = z + z^-1 = 2 cos(theta).

For a real length-N record, pair samples n and N-n and write

    P(z) = A(t) + (z - z^-1) B(t),

where A is a Chebyshev-T series and B a Chebyshev-U series.  At the kth RFFT
root z_k = exp(-2*pi*i*k/N),

    X[k] = A(t_k) - 2*i*sin(theta_k)*B(t_k).

Thus the transform is two entirely real multipoint evaluations on the ordered
nodes t_k.  A balanced product/remainder tree over consecutive t_k intervals
is a literal decimation of the phase axis: every subtree owns a contiguous
frequency band.  It is not a DIF/DIT splice and it does not carry complex
packets internally.

The stable evaluator below establishes the invariant.  The explicit remainder
tree establishes the new walk and its ordering for small N.  Its monomial
basis is intentionally kept as a diagnostic: it becomes ill-conditioned and
its reductions are dense.  DIP2 is therefore a genuine walk candidate, not
yet an O(N log N) production FFT.  The remaining research problem is a stable
phase-local basis in which those interval reductions have fast structure.
"""

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


def _power_of_two(n: int) -> bool:
    return n >= 4 and (n & (n - 1)) == 0


def phase_series(x):
    """Return the real Chebyshev T/U coefficient packets (A, B).

    ``A`` is in T_n(t/2); ``B`` is in U_n(t/2).  No complex values are
    created.  These two arrays are the DIP2 input boundary representation.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if not _power_of_two(n):
        raise ValueError("length must be a power of two >= 4")
    m = n // 2

    a = np.empty(m + 1, dtype=np.float64)
    a[0] = x[0]
    a[m] = x[m]
    for j in range(1, m):
        a[j] = x[j] + x[n - j]

    b = np.empty(max(m - 1, 0), dtype=np.float64)
    for j in range(1, m):
        b[j - 1] = 0.5 * (x[j] - x[n - j])
    return a, b


def _u_series_eval(coeff, y):
    """Evaluate sum coeff[j] U_j(y), stably by backward recurrence."""
    coeff = np.asarray(coeff, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if coeff.size == 0:
        return np.zeros_like(y)
    # Clenshaw for U_j: U_{j+1}=2*y*U_j-U_{j-1}.
    q1 = np.zeros_like(y)
    q2 = np.zeros_like(y)
    for c in coeff[::-1]:
        q0 = c + 2.0 * y * q1 - q2
        q2, q1 = q1, q0
    return q1


def dip2_rfft(x):
    """Stable strict-real DIP2 invariant evaluator.

    This currently evaluates the two phase series directly, so its runtime is
    quadratic.  It is the numerical oracle for the phase remainder walk.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    a, b = phase_series(x)
    k = np.arange(n // 2 + 1, dtype=np.float64)
    theta = 2.0 * np.pi * k / n
    y = np.cos(theta)
    av = np.polynomial.chebyshev.chebval(y, a)
    bv = _u_series_eval(b, y)
    # Complex is only the public RFFT container.  Every interior quantity
    # above is real, and (av, bv) are Bruun residue coordinates.
    return av - 2.0j * np.sin(theta) * bv


def _poly_add_scaled(a, b, scale):
    out = np.zeros(max(len(a), len(b)), dtype=np.float64)
    out[:len(a)] += a
    out[:len(b)] += scale * b
    return out


def phase_series_monomial(x):
    """Expand A(t), B(t) into ascending monomial coefficients.

    Used only by the explicit small-N remainder-tree diagnostic.  The rapidly
    growing coefficient condition number is part of what this probe measures.
    """
    at, bu = phase_series(x)
    m = len(at) - 1

    # T_n(t/2): T_0=1, T_1=t/2, T_{n+1}=t*T_n-T_{n-1}.
    ts = [np.array([1.0]), np.array([0.0, 0.5])]
    for _ in range(1, m):
        ts.append(_poly_add_scaled(np.r_[0.0, ts[-1]], ts[-2], -1.0))
    a = np.zeros(1, dtype=np.float64)
    for c, basis in zip(at, ts[:m + 1]):
        a = _poly_add_scaled(a, basis, c)

    # U_n(t/2): U_0=1, U_1=t, same recurrence thereafter.
    us = [np.array([1.0])]
    if m >= 2:
        us.append(np.array([0.0, 1.0]))
    for _ in range(1, m - 1):
        us.append(_poly_add_scaled(np.r_[0.0, us[-1]], us[-2], -1.0))
    b = np.zeros(1, dtype=np.float64)
    for c, basis in zip(bu, us):
        b = _poly_add_scaled(b, basis, c)
    return a, b


@dataclass
class PhaseNode:
    """One consecutive phase interval and its monic product polynomial."""

    lo: int
    hi: int
    modulus: np.ndarray
    left: Optional["PhaseNode"] = None
    right: Optional["PhaseNode"] = None

    @property
    def leaf(self):
        return self.hi - self.lo == 1

    @property
    def bins(self):
        return range(self.lo, self.hi)


def build_phase_tree(n: int) -> PhaseNode:
    """Build the planned subproduct tree for t_k, k=0..N/2."""
    if not _power_of_two(n):
        raise ValueError("length must be a power of two >= 4")
    nodes = 2.0 * np.cos(2.0 * np.pi * np.arange(n // 2 + 1) / n)

    def rec(lo, hi):
        if hi - lo == 1:
            return PhaseNode(lo, hi, np.array([-nodes[lo], 1.0]))
        mid = (lo + hi) // 2
        left = rec(lo, mid)
        right = rec(mid, hi)
        return PhaseNode(lo, hi, np.convolve(left.modulus, right.modulus),
                         left, right)

    return rec(0, len(nodes))


def walk_nodes(root: PhaseNode) -> Iterator[PhaseNode]:
    yield root
    if root.left is not None:
        yield from walk_nodes(root.left)
        yield from walk_nodes(root.right)


def _poly_rem(f, monic_g):
    """Ascending-coefficient real polynomial remainder by a monic divisor."""
    if len(f) < len(monic_g):
        return f.copy()
    r = np.array(f, dtype=np.float64, copy=True)
    gd = len(monic_g) - 1
    for j in range(len(r) - 1, gd - 1, -1):
        c = r[j]  # monic divisor
        r[j - gd:j + 1] -= c * monic_g
    return r[:gd]


def _tree_eval(poly, node, out):
    rem = _poly_rem(poly, node.modulus)
    if node.leaf:
        out[node.lo] = rem[0]
        return
    _tree_eval(rem, node.left, out)
    _tree_eval(rem, node.right, out)


def dip2_rfft_remainder_tree(x):
    """Execute the literal phase-interval walk (small-N diagnostic).

    Monomial remainder trees are numerically unsafe at larger N; callers that
    need an oracle should use ``dip2_rfft``.  Refusing N>32 prevents a false
    impression that this basis is a viable production representation.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n > 32:
        raise ValueError("monomial phase tree is a diagnostic limited to N<=32")
    a, b = phase_series_monomial(x)
    root = build_phase_tree(n)
    av = np.empty(n // 2 + 1, dtype=np.float64)
    bv = np.empty_like(av)
    _tree_eval(a, root, av)
    _tree_eval(b, root, bv)
    theta = 2.0 * np.pi * np.arange(n // 2 + 1) / n
    return av - 2.0j * np.sin(theta) * bv


def tree_ledger(n: int):
    """Report ordering and density facts for the planned DIP2 tree."""
    root = build_phase_tree(n)
    internal = [q for q in walk_nodes(root) if not q.leaf]
    densities = []
    for q in internal:
        scale = max(float(np.max(np.abs(q.modulus))), 1.0)
        nz = np.count_nonzero(np.abs(q.modulus) > 64 * np.finfo(float).eps * scale)
        densities.append(nz / q.modulus.size)
    leaves = [q.lo for q in walk_nodes(root) if q.leaf]

    # Runtime coefficient updates for naive long division, assuming every
    # parent arrives as a remainder of degree < its interval size.  Product
    # polynomials are plan-time data and are not included.
    remainder_updates = 0
    for q in internal:
        parent_degree = (q.hi - q.lo) - 1
        for child in (q.left, q.right):
            child_degree = child.hi - child.lo
            quotient_terms = max(parent_degree - child_degree + 1, 0)
            remainder_updates += quotient_terms * (child_degree + 1)

    # A contiguous q-bin phase interval sees a q by (M+1) Chebyshev-
    # Vandermonde block.  Full row rank means that an exact independent
    # interval packet needs q scalar degrees of freedom.  Diagonal phase
    # twists cannot lower this rank.
    m = n // 2
    theta = 2.0 * np.pi * np.arange(m + 1) / n
    vandermonde = np.polynomial.chebyshev.chebvander(np.cos(theta), m)
    full_rank = True
    min_rank_margin = m + 1
    for q in walk_nodes(root):
        block = vandermonde[q.lo:q.hi]
        rank = int(np.linalg.matrix_rank(block))
        width = q.hi - q.lo
        full_rank &= rank == width
        min_rank_margin = min(min_rank_margin, rank - width)
    return {
        "leaves": leaves,
        "natural_leaf_order": leaves == list(range(n // 2 + 1)),
        "all_subtrees_contiguous": all(
            list(q.bins) == list(range(q.lo, q.hi)) for q in walk_nodes(root)),
        "mean_modulus_density": float(np.mean(densities)),
        "max_modulus_density": float(np.max(densities)),
        "naive_remainder_updates": int(remainder_updates),
        "all_intervals_full_row_rank": bool(full_rank),
        "min_rank_margin": int(min_rank_margin),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(82)
    print("DIP2: real Bruun phase-axis invariant")
    for n in (8, 16, 32, 64, 256, 1024):
        x = rng.standard_normal(n)
        err = np.max(np.abs(dip2_rfft(x) - np.fft.rfft(x)))
        print(f"  N={n:4d} stable phase-series error {err:.3e}")
    for n in (8, 16, 32):
        x = rng.standard_normal(n)
        err = np.max(np.abs(dip2_rfft_remainder_tree(x) - np.fft.rfft(x)))
        led = tree_ledger(n)
        print(f"  N={n:4d} phase-tree error {err:.3e}; "
              f"natural={led['natural_leaf_order']} "
              f"dense(max)={led['max_modulus_density']:.2f}")
