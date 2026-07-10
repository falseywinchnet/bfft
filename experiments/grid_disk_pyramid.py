#!/usr/bin/env python3
"""Cross-bin sharing of fine disk builds via a frequency-grid disk store.

The two-regime hybrid's dominant cost is one exact local disk tree per
(fine block, bin) with no sharing.  This experiment shares builds across
bins: disk trees are built exactly at grid frequencies k_g = g*dk with
dk = N/(Lf*G) bins, and a query at bin k snaps to the nearest grid tree.

Frequency translation is certified.  For a sub-block S with global center
n0 and Delta-omega = 2 pi (k-k_g)/N, every prefix phasor obeys

    |C_k(tau) - e^{-i dw n0} C_{k_g}(tau)| <= M1(S) |dw|,
    M1(S) = sum_{n in S} |x[n]| |n - n0|,

so the grid disk rotated by e^{-i dw n0} and inflated by M1|dw| contains
every prefix at k.  M1 is O(1) from two prefix-sum tables.  Because all
tree nodes combine at the same grid frequency, the build itself has zero
inflation; the only slack is the one-time query inflation
<= M1(S) pi/(Lf G).  The sharing factor per build is dk bins, so the
scheme pays nothing at N ~ Lf*G and improves linearly in N beyond that.

Walk semantics stay exact: inflation only loosens bounds (more visited
nodes at worst), incumbents come from exact scores, and every bin asserts
the certified global optimum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.intrinsic_dip_branchbound import (
    build_phase_disk_tree, make_cases)
from experiments.fractional_channel_sharing import shared_cost
from experiments.spectral_gate_selector import BlockSpectra


class GridDiskStore:
    """Exact disk trees at grid frequencies, shared across querying bins."""

    def __init__(self, x, G=16, Lf=64):
        self.x = np.asarray(x, np.complex128)
        self.N = self.x.size
        self.G = G
        self.Lf = Lf
        self.dk = self.N / (Lf * G)      # grid spacing in bins
        self.trees = {}
        self.build_ops = 0
        self.builds = 0
        a = np.abs(self.x)
        self.p_abs = np.concatenate(([0.0], np.cumsum(a)))
        self.p_nabs = np.concatenate(([0.0], np.cumsum(a * np.arange(self.N))))

    def m1(self, s, l):
        """sum |x[n]| |n-n0| over [s,s+l), n0 = s+(l-1)/2, in O(1)."""
        n0 = s + (l - 1) / 2
        m = int(np.ceil(n0))
        left = n0 * (self.p_abs[m] - self.p_abs[s]) - \
            (self.p_nabs[m] - self.p_nabs[s])
        right = (self.p_nabs[s + l] - self.p_nabs[m]) - \
            n0 * (self.p_abs[s + l] - self.p_abs[m])
        return left + right

    def snap(self, k):
        """Nearest grid index and the residual angular offset dw."""
        g = int(np.round(k / self.dk))
        return g, 2 * np.pi * (k - g * self.dk) / self.N

    def tree(self, lo, g):
        key = (lo, g)
        node = self.trees.get(key)
        if node is None:
            w = 2 * np.pi * (g * self.dk) / self.N
            n = np.arange(lo, lo + self.Lf)
            node = build_phase_disk_tree(self.x[lo:lo + self.Lf] *
                                         np.exp(-1j * w * n))
            self.trees[key] = node
            self.build_ops += 3 * self.Lf
            self.builds += 1
        return node

    def disk_at(self, base, node, dw):
        """Certified disk for the node's span at frequency k = k_g + dk_res."""
        s = base + node.lo
        l = node.hi - node.lo
        n0 = s + (l - 1) / 2
        c = node.center * np.exp(-1j * dw * n0)
        r = node.radius + self.m1(s, l) * abs(dw)
        return c, r


def select_support_grid(x, omega, spectra, store, depth=3, initial=None,
                        energy_prefix=None):
    """Hybrid walk with grid-shared fine disks.  Exact per-bin result.

    Returns (tau, C, score, used, nodes, env_queries, disk_queries).
    """
    x = np.asarray(x, np.complex128)
    N = x.size
    t = np.arange(N)
    c = np.cumsum(x * np.exp(-1j * omega * t))
    score = np.abs(c) ** 2 / (t + 1)
    if energy_prefix is None:
        energy_prefix = np.concatenate(([0.0], np.cumsum(np.abs(x) ** 2)))
    kk = omega * N / (2 * np.pi)
    Lf = store.Lf
    env_queries = 0
    disk_queries = 0
    env_memo = {}

    def ebound(lo, hi):
        return np.sqrt((hi - lo) *
                       max(energy_prefix[hi] - energy_prefix[lo], 0.0))

    def envelope(lo, L):
        nonlocal env_queries
        v = env_memo.get((lo, L))
        if v is None:
            env_queries += 1
            v = spectra.envelope(lo, L, kk)
            env_memo[(lo, L)] = v
        return v

    def maxpre(lo, hi, d):
        if d == 0 or hi - lo < 2 * spectra.Lmin:
            return ebound(lo, hi)
        mid = lo + (hi - lo) // 2
        return max(maxpre(lo, mid, d - 1),
                   envelope(lo, mid - lo) + maxpre(mid, hi, d - 1))

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
    nodes = 0

    def visit_fine(node, base, before, dw):
        nonlocal jbest, sbest, nodes, disk_queries
        nodes += 1
        disk_queries += 1
        glo = base + node.lo
        dc, dr = store.disk_at(base, node, dw)
        if (abs(before + dc) + dr) ** 2 / (glo + 1) <= sbest * (1.0 + 1e-14):
            return
        if node.hi - node.lo == 1:
            used.add((glo, 1))
            if score[glo] > sbest:
                sbest = float(score[glo])
                jbest = glo
            return
        # exact V(left child) is a transported demand, as in the coarse walk
        llo = base + node.left.lo
        llen = node.left.hi - node.left.lo
        used.add((llo, llen))
        vleft = c[llo + llen - 1] - (c[llo - 1] if llo else 0j)
        if score[base + node.hi - 1] > score[base + node.left.hi - 1]:
            visit_fine(node.right, base, before + vleft, dw)
            visit_fine(node.left, base, before, dw)
        else:
            visit_fine(node.left, base, before, dw)
            visit_fine(node.right, base, before + vleft, dw)

    def visit(lo, hi):
        nonlocal jbest, sbest, nodes
        clen = hi - lo
        before = 0j if lo == 0 else c[lo - 1]
        nodes += 1
        u = min(ebound(lo, hi), maxpre(lo, hi, depth))
        if (abs(before) + u) ** 2 / (lo + 1) <= sbest * (1.0 + 1e-14):
            return
        if clen <= Lf:
            g, dw = store.snap(kk)
            visit_fine(store.tree(lo, g), lo, before, dw)
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
    return true + 1, c[true], sbest, used, nodes, env_queries, disk_queries


def self_test():
    """Certified-containment property of grid-translated disks."""
    rng = np.random.default_rng(11)
    N, Lf, G = 256, 32, 8
    worst = -np.inf
    for _ in range(20):
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        store = GridDiskStore(x, G=G, Lf=Lf)
        for _ in range(10):
            lo = int(rng.integers(0, N // Lf)) * Lf
            k = float(rng.uniform(0, N))
            g, dw = store.snap(k)
            root = store.tree(lo, g)
            w = 2 * np.pi * k / N
            stack = [root]
            while stack:
                node = stack.pop()
                s = lo + node.lo
                l = node.hi - node.lo
                pre = np.cumsum(x[s:s + l] *
                                np.exp(-1j * w * np.arange(s, s + l)))
                dc, dr = store.disk_at(lo, node, dw)
                slack = np.max(np.abs(pre - dc)) - dr
                worst = max(worst, float(slack))
                assert slack <= 1e-9, (lo, l, k, slack)
                if node.left is not None:
                    stack.extend([node.left, node.right])
    print(f"grid disk containment certified: max |prefix-c|-r = {worst:.3e}")


def active_bins(x):
    """Activity gate used throughout this series (score-gated bins)."""
    x = np.asarray(x, np.complex128)
    N = x.size
    t = np.arange(N)
    mean = np.mean(np.abs(x) ** 2)
    gate = (np.log(N) + 2) * mean
    out = []
    for k in range(N):
        c = np.cumsum(x * np.exp(-2j * np.pi * k * t / N))
        if np.max(np.abs(c) ** 2 / (t + 1)) > gate:
            out.append(k)
    return out


def evaluate(N=1024, cases=("chirp",), G=16, Lf=64, warm=True, depth=3):
    """Active-bin accounting, consistent with the rest of the series: the
    store and all tallies see only activity-gated bins."""
    rng = np.random.default_rng(22)
    fft = 0.5 * N * np.log2(N)
    all_cases = make_cases(N, rng)
    out = {}
    for name in cases:
        x = all_cases[name]
        bins = active_bins(x)
        spectra = BlockSpectra(x, lazy=True)
        store = GridDiskStore(x, G=G, Lf=Lf)
        energy_prefix = np.concatenate(([0.0], np.cumsum(np.abs(x) ** 2)))
        demands = {}
        tot_nodes = tot_env = tot_disk = 0
        prev = None
        for k in bins:
            tau, _, s, used, nodes, qe, qd = select_support_grid(
                x, 2 * np.pi * k / N, spectra, store, depth=depth,
                initial=prev if warm else None, energy_prefix=energy_prefix)
            prev = tau
            tot_nodes += nodes
            tot_env += qe
            tot_disk += qd
            for blk in used:
                demands.setdefault(blk, set()).add(k)
        active = len(bins)
        dp, _ = shared_cost(demands, N)
        env_cost = 8 * tot_env
        disk_cost = 10 * tot_disk
        total = dp + spectra.butterflies + store.build_ops + env_cost + disk_cost
        print(f"[{name} N={N} G={G} Lf={Lf}] active={active} "
              f"nodes/bin={tot_nodes/max(active,1):.1f} "
              f"grid builds={store.builds} "
              f"({store.builds/max(active,1):.1f}/bin, dk={store.dk:.2f} bins)")
        print(f"  V-DP={dp/fft:.2f} spectral={spectra.butterflies/fft:.2f} "
              f"grid builds={store.build_ops/fft:.2f} env-q={env_cost/fft:.2f} "
              f"disk-q={disk_cost/fft:.2f}  TOTAL={total/fft:.2f} /FFT "
              f"(DDC={active*N/fft:.1f})")
        out[name] = total / fft
    return out


if __name__ == "__main__":
    self_test()
    for G in (8, 16, 32):
        evaluate(1024, G=G)
    print()
    for N in (2048, 4096):
        for G in (8, 16, 32):
            evaluate(N, G=G)
