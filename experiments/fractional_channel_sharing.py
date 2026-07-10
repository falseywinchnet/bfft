#!/usr/bin/env python3
"""Shared transport of fractional channels for the IPD-DIP selector.

The certified selector demands total phasors V(B,k) for dyadic time blocks B.
The audited baseline evaluates each opened (block, delta=k mod N/L) channel by
an independent pruned twisted butterfly; chirps cost ~40x one FFT that way.

This script measures two exact reorganizations:

1.  Demand census: where (scale x residue) the chirp cost concentrates.
2.  Cross-scale sharing.  In global-phase coordinates the DIF identity

        V([lo,lo+2L), k) = V([lo,lo+L), k) + V([lo+L,lo+2L), k)

    is one complex add per (parent, k).  A demanded channel may therefore be
    SYNTHesized from its children instead of evaluated DIRECTly; children then
    inherit the parent's bin demands, which they share with their own demands
    and with the sibling subtree.  The min-cost assignment over the dyadic
    time tree is a small DP because inherited demand sets form a nested chain.

Also measured: warm-starting each bin's incumbent from the previous bin's
selected endpoint.  The seed score is an achieved score at the new frequency,
so pruning stays exact; for smooth-in-k support (chirps) it shrinks demand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.intrinsic_dip_branchbound import make_cases, select_support
from experiments.twisted_fractional_dip import pruned_cell_count


def dyadic_blocks(lo, hi):
    """Aligned dyadic decomposition of [lo,hi)."""
    out = []
    while lo < hi:
        L = 1
        while lo % (2 * L) == 0 and lo + 2 * L <= hi:
            L <<= 1
        out.append((lo, L))
        lo += L
    return out


def walk_demands(x, omega, initial=None):
    """Replicate the certified energy selector, recording only the block totals
    it actually consumes.

    The entry bound at a node uses the prefix phasor already carried down and
    the free energy table.  New values are consumed only when a non-pruned
    internal node hands ``before + V(left child)`` to its right child, when a
    leaf is scored, and when dyadic/warm seeds are scored.
    """
    x = np.asarray(x, np.complex128)
    N = x.size
    t = np.arange(N)
    c = np.cumsum(x * np.exp(-1j * omega * t))
    score = np.abs(c) ** 2 / (t + 1)
    energy_prefix = np.concatenate(([0.0], np.cumsum(np.abs(x) ** 2)))

    used = set()
    seeds = []
    p = 1
    while p <= N:
        used.add((0, p))          # prefix at a dyadic endpoint is one block
        seeds.append(p - 1)
        p <<= 1
    if initial is not None and 1 <= initial <= N:
        used.update(dyadic_blocks(0, int(initial)))
        seeds.append(int(initial) - 1)
    jbest = max(seeds, key=lambda j: score[j])
    sbest = float(score[jbest])
    nodes = pruned = 0

    def visit(lo, hi):
        nonlocal jbest, sbest, nodes, pruned
        nodes += 1
        clen = hi - lo
        before = 0j if lo == 0 else c[lo - 1]
        en = np.sqrt(max(energy_prefix[hi] - energy_prefix[lo], 0.0))
        if (abs(before) + np.sqrt(clen) * en) ** 2 / (lo + 1) \
                <= sbest * (1.0 + 1e-14):
            pruned += 1
            return
        if clen == 1:
            used.add((lo, 1))
            if score[lo] > sbest:
                sbest = float(score[lo])
                jbest = lo
            return
        mid = lo + clen // 2
        used.add((lo, mid - lo))  # right child consumes before + V(left)
        jl, jr = mid - 1, hi - 1
        if score[jr] > score[jl]:
            visit(mid, hi)
            visit(lo, mid)
        else:
            visit(lo, mid)
            visit(mid, hi)

    visit(0, N)
    true = int(np.argmax(score))
    assert jbest == true
    return true + 1, float(sbest), used, nodes, pruned


def collect_demands(x, warm=False, refined=True):
    """Gather (block,k) demands over active bins.

    refined=True counts only consumed totals (production accounting);
    refined=False charges every visited node (the earlier coarse audit).
    """
    x = np.asarray(x, np.complex128)
    N = x.size
    mean = np.mean(np.abs(x) ** 2)
    gate = (np.log(N) + 2) * mean
    demands = {}
    active = 0
    opened_pairs = 0
    prev_tau = None
    for k in range(N):
        if refined:
            tau, score, used, _, _ = walk_demands(
                x, 2 * np.pi * k / N, initial=prev_tau if warm else None)
        else:
            tau, _, score, audit = select_support(
                x, 2 * np.pi * k / N, initial=prev_tau if warm else None)
            used = {(lo, hi - lo) for lo, hi in audit.opened}
        if score <= gate:
            prev_tau = None
            continue
        active += 1
        prev_tau = tau
        for blk in used:
            demands.setdefault(blk, set()).add(k)
            opened_pairs += 1
    return demands, active, opened_pairs


def census(demands, N):
    by_scale = {}
    for (lo, L), ks in demands.items():
        e = N // L
        d = by_scale.setdefault(L, dict(blocks=0, pairs=0, chans=0, span=[]))
        d["blocks"] += 1
        d["pairs"] += len(ks)
        d["chans"] += len({k % e for k in ks})
        arr = np.sort(np.array(list(ks)))
        # circular span of demanded bins (band width around the block's content)
        if arr.size == 1:
            d["span"].append(1)
        else:
            gaps = np.diff(np.concatenate([arr, [arr[0] + N]]))
            d["span"].append(int(N - gaps.max() + 1))
    print(f"{'L':>5} {'blocks':>6} {'pairs':>7} {'channels':>8} "
          f"{'band med/max':>12} {'baseline cells':>14}")
    total = 0
    for L in sorted(by_scale, reverse=True):
        d = by_scale[L]
        cost = 0
        for (lo, LL), ks in demands.items():
            if LL != L:
                continue
            e = N // L
            for delta in {k % e for k in ks}:
                ps = [(k - delta) // e for k in ks if k % e == delta]
                cost += pruned_cell_count(L, ps) if L > 1 else len(ps)
        total += cost
        print(f"{L:5d} {d['blocks']:6d} {d['pairs']:7d} {d['chans']:8d} "
              f"{int(np.median(d['span'])):5d}/{max(d['span']):5d} {cost:14d}")
    fft = 0.5 * N * np.log2(N)
    print(f"baseline per-channel total={total}  /FFT-butterflies={total/fft:.2f}")
    return total


def direct_cost(N, L, ks):
    """Standalone pruned twisted butterflies to produce V(B,k) for k in ks."""
    if L == 1:
        return len(ks)
    e = N // L
    cost = 0
    for delta in {k % e for k in ks}:
        cost += pruned_cell_count(L, sorted((k - delta) // e
                                            for k in ks if k % e == delta))
    return cost


def shared_cost(demands, N):
    """Min-cost exact evaluation of all demanded values over the dyadic tree.

    solve(lo, L, inherited): inherited bins must be produced at this block.
    DIRECT: pruned twisted butterflies for inherited|own here; children only
    handle their own subtree demands.  SYNTH: one add per produced bin here;
    children each inherit the full set.  Inherited sets along any root path
    form a nested chain, so memoization keeps the DP small.
    """
    marked = set()
    for (lo, L) in demands:
        s, e = lo, L
        while True:
            marked.add((s, e))
            if e == N:
                break
            s, e = (s // (2 * e)) * (2 * e), 2 * e
    memo = {}

    def solve(lo, L, inherited):
        own = demands.get((lo, L), set())
        K = inherited | own if inherited else own
        if not K and (lo, L) not in marked:
            return 0
        key = (lo, L, frozenset(inherited))
        if key in memo:
            return memo[key][0]
        if L == 1:
            memo[key] = (len(K), "leaf")
            return len(K)
        half = L // 2
        direct = (direct_cost(N, L, K) +
                  solve(lo, half, frozenset()) +
                  solve(lo + half, half, frozenset()))
        best, act = direct, "direct"
        if K:
            fk = frozenset(K)
            synth = len(K) + solve(lo, half, fk) + solve(lo + half, half, fk)
            if synth < best:
                best, act = synth, "synth"
        memo[key] = (best, act)
        return best

    total = solve(0, N, frozenset())

    # Reconstruct chosen actions to attribute ops per scale.
    per_scale = {}
    stack = [(0, N, frozenset())]
    while stack:
        lo, L, inherited = stack.pop()
        own = demands.get((lo, L), set())
        K = inherited | own if inherited else own
        if not K and (lo, L) not in marked:
            continue
        act = memo[(lo, L, frozenset(inherited))][1]
        if L == 1:
            per_scale[1] = per_scale.get(1, 0) + len(K)
            continue
        half = L // 2
        if act == "direct":
            per_scale[L] = per_scale.get(L, 0) + direct_cost(N, L, K)
            stack.append((lo, half, frozenset()))
            stack.append((lo + half, half, frozenset()))
        else:
            per_scale[L] = per_scale.get(L, 0) + len(K)
            fk = frozenset(K)
            stack.append((lo, half, fk))
            stack.append((lo + half, half, fk))
    return total, per_scale


def evaluate(N=1024, cases=("tone", "burst", "chirp", "noise"), warms=(False, True)):
    rng = np.random.default_rng(22)
    fft = 0.5 * N * np.log2(N)
    all_cases = make_cases(N, rng)
    for name in cases:
        x = all_cases[name]
        for warm in warms:
            demands, active, pairs = collect_demands(x, warm=warm)
            tag = "warm" if warm else "cold"
            print(f"\n[{name} N={N} {tag}] active={active} consumed pairs={pairs}")
            base = census(demands, N)
            shared, per_scale = shared_cost(demands, N)
            print(f"shared DP total={shared}  /FFT={shared/fft:.2f}  "
                  f"vs baseline {base/fft:.2f}  gain x{base/max(shared,1):.2f}")
            print("  DP ops by scale:", " ".join(
                f"L{L}:{per_scale[L]}" for L in sorted(per_scale, reverse=True)))


def scaling(case="chirp", sizes=(512, 1024, 2048, 4096)):
    print(f"\n=== scaling: {case} ===")
    print(f"{'N':>6} {'active':>6} {'pairs':>8} {'pairs/NlogN':>11} "
          f"{'baseline/FFT':>12} {'shared/FFT':>10}")
    for N in sizes:
        rng = np.random.default_rng(22)
        x = make_cases(N, rng)[case]
        demands, active, pairs = collect_demands(x, warm=True)
        fft = 0.5 * N * np.log2(N)
        base = 0
        for (lo, L), ks in demands.items():
            base += direct_cost(N, L, ks)
        shared, _ = shared_cost(demands, N)
        print(f"{N:6d} {active:6d} {pairs:8d} {pairs/(N*np.log2(N)):11.2f} "
              f"{base/fft:12.2f} {shared/fft:10.2f}")


if __name__ == "__main__":
    evaluate()
    scaling()
