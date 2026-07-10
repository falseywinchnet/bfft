#!/usr/bin/env python3
"""Certified phase-cone branch-and-bound for intrinsic leading-edge support.

For one frequency, a node [lo,hi) owns every candidate endpoint inside that
time interval.  Given the exact prefix phasor C(lo), Cauchy-Schwarz gives

    |C(tau)| <= |C(lo)| + sqrt(hi-lo) ||x[lo:hi]||_2,

and tau >= lo+1, hence a safe score bound.  If the bound cannot beat the
incumbent, the entire support packet is rejected without visiting endpoints.

This script uses direct prefix phasors only as an oracle for node totals; a
production walk must transport those totals in DIP/fractional-frequency
packets.  The measured quantity is the number of support leaves the nonlinear
selector actually needs after a linear phase-packet tower has supplied totals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Audit:
    nodes: int = 0
    pruned: int = 0
    leaves: int = 0
    opened: list = field(default_factory=list)


@dataclass
class PhaseDiskNode:
    lo: int
    hi: int
    total: complex
    center: complex
    radius: float
    left: "PhaseDiskNode | None" = None
    right: "PhaseDiskNode | None" = None


def enclosing_two_disks(c1, r1, c2, r2):
    """Smallest disk containing D(c1,r1) union D(c2,r2)."""
    d = abs(c2 - c1)
    if r1 >= d + r2:
        return c1, r1
    if r2 >= d + r1:
        return c2, r2
    R = 0.5 * (d + r1 + r2)
    if d <= 1e-30:
        return c1, R
    return c1 + ((R - r1) / d) * (c2 - c1), R


def build_phase_disk_tree(y):
    """Exact total phasor + certified disk containing every local prefix."""
    y = np.asarray(y, np.complex128)

    def rec(lo, hi):
        if hi - lo == 1:
            z = complex(y[lo])
            return PhaseDiskNode(lo, hi, z, z, 0.0)
        mid = lo + (hi - lo) // 2
        A = rec(lo, mid)
        B = rec(mid, hi)
        # Parent prefixes are A's prefixes or A.total + B's prefixes.
        c, r = enclosing_two_disks(A.center, A.radius,
                                   A.total + B.center, B.radius)
        return PhaseDiskNode(lo, hi, A.total + B.total, c, r, A, B)

    return rec(0, y.size)


def select_support(x, omega, seed_dyadic=True, initial=None):
    x = np.asarray(x, np.complex128)
    N = x.size
    t = np.arange(N)
    c = np.cumsum(x * np.exp(-1j * omega * t))
    score = np.abs(c) ** 2 / (t + 1)

    seeds = [N - 1]
    if seed_dyadic:
        p = 1
        while p <= N:
            seeds.append(p - 1)
            p <<= 1
    # A warm-start endpoint is scored at the *current* frequency, so the
    # incumbent is an achieved score and pruning stays exact.
    if initial is not None and 1 <= initial <= N:
        seeds.append(int(initial) - 1)
    jbest = max(seeds, key=lambda j: score[j])
    sbest = float(score[jbest])
    audit = Audit()

    energy_prefix = np.concatenate(([0.0], np.cumsum(np.abs(x) ** 2)))

    def visit(lo, hi):
        nonlocal jbest, sbest
        audit.nodes += 1
        audit.opened.append((lo, hi))
        clen = hi - lo
        before = 0j if lo == 0 else c[lo - 1]
        en = np.sqrt(max(energy_prefix[hi] - energy_prefix[lo], 0.0))
        upper = (abs(before) + np.sqrt(clen) * en) ** 2 / (lo + 1)
        if upper <= sbest * (1.0 + 1e-14):
            audit.pruned += 1
            return
        if clen == 1:
            audit.leaves += 1
            if score[lo] > sbest:
                sbest = float(score[lo])
                jbest = lo
            return
        mid = lo + clen // 2
        # Visit the child with the stronger cheap endpoint score first; its
        # improved incumbent can prune the sibling. This is packet scheduling,
        # not an approximation.
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
    return true + 1, c[true], sbest, audit


def select_support_disk(x, omega, seed_dyadic=True, initial=None):
    """Exact selector using only the compositional prefix-disk bound."""
    x = np.asarray(x, np.complex128)
    N = x.size
    t = np.arange(N)
    y = x * np.exp(-1j * omega * t)
    c = np.cumsum(y)
    score = np.abs(c) ** 2 / (t + 1)
    root = build_phase_disk_tree(y)

    seeds = [N - 1]
    if seed_dyadic:
        p = 1
        while p <= N:
            seeds.append(p - 1)
            p <<= 1
    jbest = max(seeds, key=lambda j: score[j])
    sbest = float(score[jbest])
    if initial is not None:
        ji = int(initial) - 1
        if 0 <= ji < N and score[ji] > sbest:
            jbest = ji
            sbest = float(score[ji])
    audit = Audit()

    def visit(node, before):
        nonlocal jbest, sbest
        audit.nodes += 1
        audit.opened.append((node.lo, node.hi))
        upper = (abs(before + node.center) + node.radius) ** 2 / (node.lo + 1)
        if upper <= sbest * (1.0 + 1e-14):
            audit.pruned += 1
            return
        if node.hi - node.lo == 1:
            audit.leaves += 1
            if score[node.lo] > sbest:
                sbest = float(score[node.lo])
                jbest = node.lo
            return
        # The right child sees every sample in the left child as preceding
        # context. Visit the stronger endpoint child first to raise incumbent.
        assert node.left is not None and node.right is not None
        if score[node.hi - 1] > score[node.left.hi - 1]:
            visit(node.right, before + node.left.total)
            visit(node.left, before)
        else:
            visit(node.left, before)
            visit(node.right, before + node.left.total)

    visit(root, 0j)
    true = int(np.argmax(score))
    assert jbest == true and abs(sbest - score[true]) <= 1e-10 * (1 + sbest)
    return true + 1, c[true], sbest, audit


def ipd_transform(x, activity=None):
    """Exact Intrinsic Phase-Disk transform reference.

    This is the semantic reference, not the fast joint-packet kernel: it builds
    one scalar-frequency disk tree per bin. ``activity`` is a score threshold;
    bins below it default to the full Fourier support. Returns
    ``(C, tau, nodes, leaves)`` in full complex FFT order.
    """
    x = np.asarray(x, np.complex128)
    N = x.size
    out = np.empty(N, np.complex128)
    tau = np.empty(N, np.int64)
    nodes = np.empty(N, np.int64)
    leaves = np.empty(N, np.int64)
    full = np.fft.fft(x)
    for k in range(N):
        tk, zk, sk, audit = select_support_disk(x, 2 * np.pi * k / N)
        if activity is not None and sk <= activity:
            tk, zk = N, full[k]
        tau[k] = tk
        out[k] = zk
        nodes[k] = audit.nodes
        leaves[k] = audit.leaves
    return out, tau, nodes, leaves


def make_cases(N, rng):
    t = np.arange(N)
    cases = {}
    cases["tone"] = np.exp(2j * np.pi * 91.3 * t / N)
    x = 0.01 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    m = (t >= 100) & (t < 490)
    x[m] += np.exp(2j * np.pi * 211.4 * t[m] / N)
    cases["burst"] = x
    cases["chirp"] = np.exp(2j * np.pi * (35 * t / N + 0.31 * t * t / N))
    cases["noise"] = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    return cases


def evaluate(N=1024):
    rng = np.random.default_rng(22)
    for name, x in make_cases(N, rng).items():
        mean = np.mean(np.abs(x) ** 2)
        rows = []
        for k in range(N):
            tau, z, s, a = select_support_disk(x, 2 * np.pi * k / N)
            if s > (np.log(N) + 2) * mean:
                rows.append((tau, s, a))
        if not rows:
            print(f"{name:8s}: no active bins")
            continue
        leaves = np.array([r[2].leaves for r in rows])
        nodes = np.array([r[2].nodes for r in rows])
        pruned = np.array([r[2].pruned for r in rows])
        print(f"{name:8s} disk: active={len(rows):4d} leaves med/p90/max="
              f"{np.median(leaves):.0f}/{np.percentile(leaves,90):.0f}/{leaves.max()} "
              f"({100*np.median(leaves)/N:.2f}% of endpoints), "
              f"nodes med={np.median(nodes):.0f}, pruned med={np.median(pruned):.0f}")


def grouped_fractional_cost(x):
    """Estimate shared linear work for certified energy-bound packet openings.

    A block of length L sees global bins k=p*(N/L)+delta. One dyadic-offset
    length-L transform for a fixed (block,delta) supplies every p, so openings
    from many bins are grouped by this exact fractional-channel type.
    """
    x = np.asarray(x, np.complex128)
    N = x.size
    mean = np.mean(np.abs(x) ** 2)
    channels = set()
    block_bins = {}
    active = 0
    raw_nodes = 0
    for k in range(N):
        _, _, score, audit = select_support(x, 2 * np.pi * k / N)
        if score <= (np.log(N) + 2) * mean:
            continue
        active += 1
        raw_nodes += audit.nodes
        for lo, hi in audit.opened:
            L = hi - lo
            e = N // L
            channels.add((lo, L, k % e))
            block_bins.setdefault((lo, L), set()).add(k)
    work = 0.0
    hybrid_work = 0.0
    pruned_work = 0.0
    by_level = {}
    for lo, L, delta in channels:
        cost = L * max(np.log2(L), 1.0)
        work += cost
        by_level[L] = by_level.get(L, 0) + 1
    for (lo, L), bins in block_bins.items():
        e = N // L
        residues = {k % e for k in bins}
        direct = len(bins) * L
        fractional = len(residues) * L * max(np.log2(L), 1.0)
        zero_padded = N * np.log2(N)
        hybrid_work += min(direct, fractional, zero_padded)
        # Exact pruned twisted cell count, grouped by fractional residue.
        from experiments.twisted_fractional_dip import pruned_cell_count
        for delta in residues:
            ps = [(k - delta) // e for k in bins if k % e == delta]
            pruned_work += pruned_cell_count(L, ps)
    print(f"grouped fractional channels: active={active}, raw opened nodes={raw_nodes}, "
          f"unique channels={len(channels)}, FFT-cell estimate={work:.0f}")
    print(f"  ratios: /NlogN={work/(N*np.log2(N)):.2f}, /N^2={work/(N*N):.4f}")
    print(f"  hybrid direct/fractional/full estimate={hybrid_work:.0f}: "
          f"/NlogN={hybrid_work/(N*np.log2(N)):.2f}, "
          f"/N^2={hybrid_work/(N*N):.4f}")
    fft_cells = 0.5 * N * np.log2(N)
    print(f"  exact pruned twisted butterflies={pruned_work:.0f}: "
          f"/{fft_cells:.0f} FFT butterflies={pruned_work/fft_cells:.2f}, "
          f"/N^2={pruned_work/(N*N):.4f}")
    print("  channels by block length:", " ".join(
        f"L{L}:{by_level[L]}" for L in sorted(by_level, reverse=True)))
    return pruned_work, channels, by_level


if __name__ == "__main__":
    evaluate()
