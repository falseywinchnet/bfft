#!/usr/bin/env python3
"""Certified spectral gating of the IPD support walk.

The census in fractional_channel_sharing.py shows chirp transport cost
concentrates at mid scales because the Cauchy-Schwarz energy bound cannot
prune off-resonance blocks: a chirp block has full energy everywhere, yet its
correlation against a distant frequency is tiny.

Fix: one ordinary FFT per dyadic block (a one-time pyramid, ~Sum_L (N/2)log L
butterflies) yields a certified frequency-dependent envelope.  With local
fractional frequency nu = k L/N and block spectrum G,

    V(B,k) = e^{i phi} (1/L) Sum_p G[p] K(nu-p),
    |K(nu-p)| = |sin(pi nu)| / |sin(pi (nu-p)/L)|,

so every term shares the numerator |sin(pi nu)|.  A certified bound Q(B,k)
uses a few exact near terms plus dyadic distance rings bounded through
circular prefix sums of |G|.  The prefix-max of a block then closes exactly:

    maxpre(B) <= max(maxpre(B_L), Q(B_L,k) + maxpre(B_R)),

unrolled to fixed depth with the energy bound at the recursion cut.  All
bounds are certified, so the selector remains exact; only the demand set
(and therefore the fractional-channel transport cost) shrinks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.intrinsic_dip_branchbound import make_cases
from experiments.fractional_channel_sharing import (
    census, collect_demands, direct_cost, shared_cost)


class BlockSpectra:
    """Per-block FFT magnitude tables with certified envelope queries."""

    def __init__(self, x, Lmin=16, near=2, lazy=False):
        self.x = np.asarray(x, np.complex128)
        self.N = self.x.size
        self.Lmin = Lmin
        self.near = near
        self.absG = {}
        self.pref = {}
        self.butterflies = 0
        if not lazy:
            L = Lmin
            while L <= self.N:
                for lo in range(0, self.N, L):
                    self._build(lo, L)
                L <<= 1

    def _build(self, lo, L):
        g = np.abs(np.fft.fft(self.x[lo:lo + L]))
        self.absG[(lo, L)] = g
        self.pref[(lo, L)] = np.concatenate(([0.0], np.cumsum(g)))
        self.butterflies += (L // 2) * int(np.log2(L))

    def _arc(self, key, a, b):
        """Sum of |G[p]| for integer p in the circular arc [a, b]."""
        if b < a:
            return 0.0
        L = self.absG[key].size
        if b - a + 1 >= L:
            return float(self.pref[key][L])
        a_m, b_m = a % L, b % L
        p = self.pref[key]
        if a_m <= b_m:
            return float(p[b_m + 1] - p[a_m])
        return float(p[L] - p[a_m] + p[b_m + 1])

    def envelope(self, lo, L, k):
        """Certified upper bound on |sum_{n in [lo,lo+L)} x[n] e^{-2pi i k n/N}|."""
        key = (lo, L)
        g = self.absG.get(key)
        if g is None:
            self._build(lo, L)
            g = self.absG[key]
        nu = (k * L / self.N) % L
        p0 = int(np.round(nu))
        if abs(nu - p0) < 1e-12:
            return float(g[p0 % L])
        s_nu = abs(np.sin(np.pi * nu))
        total = 0.0
        # exact near terms
        for p in range(p0 - self.near, p0 + self.near + 1):
            d = nu - p
            total += g[p % L] * s_nu / abs(np.sin(np.pi * d / L))
        # dyadic distance rings; within a ring 1/|sin(pi d/L)| is maximized
        # at the smallest circular distance d1.
        d1 = self.near + 1
        while d1 <= L // 2:
            d2 = min(2 * d1 - 1, L // 2)
            ring = (self._arc(key, p0 - d2, p0 - d1) +
                    self._arc(key, p0 + d1, p0 + d2))
            # circular distance from nu: at least d1 - |nu - p0| >= d1 - 0.5
            dmin = d1 - abs(nu - p0)
            total += ring * s_nu / abs(np.sin(np.pi * dmin / L))
            d1 = d2 + 1
        return total / L


def select_support_spectral(x, omega, spectra, depth=3, initial=None,
                            energy_prefix=None):
    """Exact selector with the certified spectral prefix-max gate.

    Returns (tau, C, score, used_blocks, nodes, pruned, q_queries).
    """
    x = np.asarray(x, np.complex128)
    N = x.size
    t = np.arange(N)
    c = np.cumsum(x * np.exp(-1j * omega * t))
    score = np.abs(c) ** 2 / (t + 1)
    if energy_prefix is None:
        energy_prefix = np.concatenate(([0.0], np.cumsum(np.abs(x) ** 2)))
    kk = omega * N / (2 * np.pi)
    q_queries = 0

    def ebound(lo, hi):
        return np.sqrt((hi - lo) *
                       max(energy_prefix[hi] - energy_prefix[lo], 0.0))

    def maxpre(lo, hi, d):
        nonlocal q_queries
        if d == 0 or hi - lo < 2 * spectra.Lmin:
            return ebound(lo, hi)
        mid = lo + (hi - lo) // 2
        q_queries += 1
        ql = spectra.envelope(lo, mid - lo, kk)
        return max(maxpre(lo, mid, d - 1), ql + maxpre(mid, hi, d - 1))

    used = set()
    seeds = []
    p = 1
    while p <= N:
        used.add((0, p))
        seeds.append(p - 1)
        p <<= 1
    if initial is not None and 1 <= initial <= N:
        lo2 = 0
        while lo2 < int(initial):
            L2 = 1
            while lo2 % (2 * L2) == 0 and lo2 + 2 * L2 <= int(initial):
                L2 <<= 1
            used.add((lo2, L2))
            lo2 += L2
        seeds.append(int(initial) - 1)
    jbest = max(seeds, key=lambda j: score[j])
    sbest = float(score[jbest])
    nodes = pruned = 0

    def visit(lo, hi):
        nonlocal jbest, sbest, nodes, pruned
        nodes += 1
        clen = hi - lo
        before = 0j if lo == 0 else c[lo - 1]
        u = min(ebound(lo, hi), maxpre(lo, hi, depth))
        if (abs(before) + u) ** 2 / (lo + 1) <= sbest * (1.0 + 1e-14):
            pruned += 1
            return
        if clen == 1:
            used.add((lo, 1))
            if score[lo] > sbest:
                sbest = float(score[lo])
                jbest = lo
            return
        mid = lo + clen // 2
        used.add((lo, mid - lo))
        jl, jr = mid - 1, hi - 1
        if score[jr] > score[jl]:
            visit(mid, hi)
            visit(lo, mid)
        else:
            visit(lo, mid)
            visit(mid, hi)

    visit(0, N)
    true = int(np.argmax(score))
    assert jbest == true and abs(sbest - score[true]) <= 1e-10 * (1 + sbest)
    return true + 1, c[true], sbest, used, nodes, pruned, q_queries


def collect_demands_spectral(x, spectra, warm=True):
    x = np.asarray(x, np.complex128)
    N = x.size
    mean = np.mean(np.abs(x) ** 2)
    gate = (np.log(N) + 2) * mean
    energy_prefix = np.concatenate(([0.0], np.cumsum(np.abs(x) ** 2)))
    demands = {}
    active = pairs = tot_nodes = tot_q = 0
    prev_tau = None
    for k in range(N):
        tau, _, score, used, nodes, _, qq = select_support_spectral(
            x, 2 * np.pi * k / N, spectra,
            initial=prev_tau if warm else None, energy_prefix=energy_prefix)
        if score <= gate:
            prev_tau = None
            continue
        active += 1
        prev_tau = tau
        tot_nodes += nodes
        tot_q += qq
        for blk in used:
            demands.setdefault(blk, set()).add(k)
            pairs += 1
    return demands, active, pairs, tot_nodes, tot_q


def select_support_hybrid(x, omega, spectra, Lf=64, depth=3, initial=None,
                          energy_prefix=None):
    """Two-regime certified walk.

    Coarse nodes (L > Lf) prune with the O(1) magnitude bound
    |before| + min(energy, spectral maxpre): coarse disks never carry the
    pruning power, so exact phase state is not transported there.  A fine
    node (L <= Lf) entered for the first time builds its exact local disk
    subtree once (3L ops: y slice + 2L disk combines); every descendant disk
    is then free, and pruning uses |before + c| + r, which is what collapses
    the Fresnel plateau.  All bounds are certified, so the walk stays exact.

    Returns (tau, C, score, coarse_used, nodes, fine_build_ops, q_queries).
    """
    x = np.asarray(x, np.complex128)
    N = x.size
    t = np.arange(N)
    y = x * np.exp(-1j * omega * t)
    c = np.cumsum(y)
    score = np.abs(c) ** 2 / (t + 1)
    if energy_prefix is None:
        energy_prefix = np.concatenate(([0.0], np.cumsum(np.abs(x) ** 2)))
    kk = omega * N / (2 * np.pi)
    q_queries = 0
    fine_build_ops = 0
    fine_trees = {}
    env_memo = {}

    def ebound(lo, hi):
        return np.sqrt((hi - lo) *
                       max(energy_prefix[hi] - energy_prefix[lo], 0.0))

    def envelope(lo, L):
        nonlocal q_queries
        v = env_memo.get((lo, L))
        if v is None:
            q_queries += 1
            v = spectra.envelope(lo, L, kk)
            env_memo[(lo, L)] = v
        return v

    def maxpre(lo, hi, d):
        if d == 0 or hi - lo < 2 * spectra.Lmin:
            return ebound(lo, hi)
        mid = lo + (hi - lo) // 2
        return max(maxpre(lo, mid, d - 1),
                   envelope(lo, mid - lo) + maxpre(mid, hi, d - 1))

    from experiments.intrinsic_dip_branchbound import build_phase_disk_tree

    def fine_tree(lo, L):
        nonlocal fine_build_ops
        node = fine_trees.get((lo, L))
        if node is None:
            node = build_phase_disk_tree(y[lo:lo + L])
            fine_build_ops += 3 * L
            fine_trees[(lo, L)] = node
        return node

    coarse_used = set()
    seeds = []
    p = 1
    while p <= N:
        coarse_used.add((0, p))
        seeds.append(p - 1)
        p <<= 1
    if initial is not None and 1 <= initial <= N:
        lo2 = 0
        while lo2 < int(initial):
            L2 = 1
            while lo2 % (2 * L2) == 0 and lo2 + 2 * L2 <= int(initial):
                L2 <<= 1
            coarse_used.add((lo2, L2))
            lo2 += L2
        seeds.append(int(initial) - 1)
    jbest = max(seeds, key=lambda j: score[j])
    sbest = float(score[jbest])
    nodes = 0

    def visit_fine(node, base, before):
        nonlocal jbest, sbest, nodes
        nodes += 1
        glo = base + node.lo
        if (abs(before + node.center) + node.radius) ** 2 / (glo + 1) \
                <= sbest * (1.0 + 1e-14):
            return
        if node.hi - node.lo == 1:
            if score[glo] > sbest:
                sbest = float(score[glo])
                jbest = glo
            return
        if score[base + node.hi - 1] > score[base + node.left.hi - 1]:
            visit_fine(node.right, base, before + node.left.total)
            visit_fine(node.left, base, before)
        else:
            visit_fine(node.left, base, before)
            visit_fine(node.right, base, before + node.left.total)

    def visit(lo, hi):
        nonlocal jbest, sbest, nodes
        clen = hi - lo
        before = 0j if lo == 0 else c[lo - 1]
        nodes += 1
        # The cheap certified magnitude bound gates everything, including
        # entry to fine nodes: a disk tree is built only when magnitude
        # pruning fails, i.e. inside the genuinely ambiguous zone.
        u = min(ebound(lo, hi), maxpre(lo, hi, depth))
        if (abs(before) + u) ** 2 / (lo + 1) <= sbest * (1.0 + 1e-14):
            return
        if clen <= Lf:
            visit_fine(fine_tree(lo, clen), lo, before)
            return
        mid = lo + clen // 2
        coarse_used.add((lo, mid - lo))
        jl, jr = mid - 1, hi - 1
        if score[jr] > score[jl]:
            visit(mid, hi)
            visit(lo, mid)
        else:
            visit(lo, mid)
            visit(mid, hi)

    visit(0, N)
    true = int(np.argmax(score))
    assert jbest == true and abs(sbest - score[true]) <= 1e-10 * (1 + sbest)
    return true + 1, c[true], sbest, coarse_used, nodes, fine_build_ops, q_queries


def evaluate_hybrid(N=1024, cases=("tone", "burst", "chirp", "noise"), Lf=64):
    rng = np.random.default_rng(22)
    fft = 0.5 * N * np.log2(N)
    all_cases = make_cases(N, rng)
    for name in cases:
        x = all_cases[name]
        spectra = BlockSpectra(x, lazy=True)
        mean = np.mean(np.abs(x) ** 2)
        gate = (np.log(N) + 2) * mean
        energy_prefix = np.concatenate(([0.0], np.cumsum(np.abs(x) ** 2)))
        demands = {}
        active = pairs = tot_nodes = tot_fine = tot_q = 0
        prev = None
        for k in range(N):
            tau, _, s, used, nodes, fine_ops, qq = select_support_hybrid(
                x, 2 * np.pi * k / N, spectra, Lf=Lf,
                initial=prev, energy_prefix=energy_prefix)
            if s <= gate:
                prev = None
                continue
            active += 1
            prev = tau
            tot_nodes += nodes
            tot_fine += fine_ops
            tot_q += qq
            for blk in used:
                demands.setdefault(blk, set()).add(k)
                pairs += 1
        shared, per_scale = shared_cost(demands, N)
        pyr = spectra.butterflies
        qcost = 8 * tot_q  # ~near terms + rings per envelope query
        total = shared + pyr + tot_fine + qcost
        print(f"[{name} N={N} Lf={Lf}] active={active} nodes/bin="
              f"{tot_nodes/max(active,1):.1f} coarse pairs={pairs}")
        print(f"  coarse V transport={shared/fft:.2f}  pyramid={pyr/fft:.2f}  "
              f"fine disk builds={tot_fine/fft:.2f}  Q-queries={qcost/fft:.2f}"
              f"  TOTAL={total/fft:.2f} /FFT  (DDC={active*N/fft:.1f})")


def evaluate(N=1024, cases=("tone", "burst", "chirp", "noise")):
    rng = np.random.default_rng(22)
    fft = 0.5 * N * np.log2(N)
    all_cases = make_cases(N, rng)
    for name in cases:
        x = all_cases[name]
        spectra = BlockSpectra(x)
        base_d, base_active, base_pairs = collect_demands(x, warm=True)
        base_shared, _ = shared_cost(base_d, N)
        demands, active, pairs, nodes, qq = collect_demands_spectral(x, spectra)
        assert active == base_active
        print(f"\n[{name} N={N}] active={active} "
              f"pairs {base_pairs}->{pairs} "
              f"nodes/bin={nodes/max(active,1):.1f} "
              f"Q-queries/bin={qq/max(active,1):.1f}")
        base = census(demands, N)
        shared, per_scale = shared_cost(demands, N)
        pyr = spectra.butterflies
        print(f"shared DP {base_shared/fft:.2f} -> {shared/fft:.2f} /FFT; "
              f"pyramid one-time={pyr/fft:.2f} /FFT; "
              f"total={(shared+pyr)/fft:.2f}")
        print("  DP ops by scale:", " ".join(
            f"L{L}:{per_scale[L]}" for L in sorted(per_scale, reverse=True)))


if __name__ == "__main__":
    if "--hybrid" in sys.argv:
        for N in (1024, 2048):
            evaluate_hybrid(N)
            print()
    else:
        evaluate()
