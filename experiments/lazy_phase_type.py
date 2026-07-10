#!/usr/bin/env python3
"""Lazy coarse phase type for the intrinsic leading-edge DIP transform.

This experiment removes two optimistic assumptions from the earlier cost
studies:

* every correlation value used by the selector is recorded as a dyadic block
  demand (the dense prefix array is only an oracle backing the experiment),
* inactive bins are not free -- their search demands remain in the ledger.

The new bound is a finite-depth compositional phase disk.  At the recursion
cut a block's local prefixes lie in the energy disk D(0, sqrt(L)||x||_2).
One level closes under the exact DIP prefix law

    P(A || B) = P(A) union (V(A,k) + P(B)),

by enclosing the two child disks.  Unlike a magnitude envelope, the complex
translation V(A,k) is retained.  Refinement is lazy and memoized per bin, so
only phase cells reached by the nonlinear walk demand fractional channels.
Across bins those demands are evaluated by the exact sharing DP in
``fractional_channel_sharing.py``.

This is a certified research reference, not yet a production kernel.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fractional_channel_sharing import dyadic_blocks, shared_cost
from experiments.intrinsic_dip_branchbound import (
    enclosing_two_disks,
    make_cases,
)


class DemandOracle:
    """Exact values for topology experiments, with every value demand logged.

    ``prefix`` deliberately decomposes a non-dyadic prefix into aligned
    dyadic blocks.  The backing cumsum is not part of the proposed algorithm;
    it only supplies values while the demand set states what a packet kernel
    must actually evaluate.
    """

    def __init__(self, x, omega):
        self.x = np.asarray(x, np.complex128)
        self.N = self.x.size
        n = np.arange(self.N)
        self._prefix = np.cumsum(self.x * np.exp(-1j * omega * n))
        self.used: set[tuple[int, int]] = set()

    def block(self, lo, length):
        self.used.add((int(lo), int(length)))
        hi = lo + length
        return self._prefix[hi - 1] - (self._prefix[lo - 1] if lo else 0j)

    def prefix(self, tau):
        blocks = dyadic_blocks(0, int(tau))
        return sum((self.block(lo, length) for lo, length in blocks), 0j)


def select_support_lazy_phase(x, omega, depth=0, initial=None,
                              half_edge=True, seed_dyadic=False,
                              activity=None, coupled=False,
                              ellipse_box=False):
    """Return the exact best support using a lazy finite-depth phase type.

    Returns ``(tau, C, score, used, nodes, phase_cells, half_tests)``.
    Correctness does not depend on ``depth``: depth zero is the SIMD candidate
    and larger depths only tighten its certified disk before deciding whether
    to split.
    """

    x = np.asarray(x, np.complex128)
    N = x.size
    oracle = DemandOracle(x, omega)
    energy = np.concatenate(([0.0], np.cumsum(np.abs(x) ** 2)))
    phase_cache: dict[tuple[int, int, int], tuple[complex, float]] = {}
    total_cache: dict[tuple[int, int], complex] = {}
    phase_cells = 0
    half_tests = 0

    def total(lo, length):
        key = (lo, length)
        if key not in total_cache:
            total_cache[key] = oracle.block(lo, length)
        return total_cache[key]

    def energy_radius(lo, length):
        e2 = max(float(energy[lo + length] - energy[lo]), 0.0)
        return np.sqrt(length * e2)

    def half_edge_upper(lo, length, before, vnode):
        """Certified max magnitude over the block's coherence ellipse.

        Writing y = V/L + r with Sum r = 0 gives, for u=j/L,

          P_j = u V + R_j,
          |R_j|^2 <= L u(1-u) (E-|V|^2/L).

        The union is contained by the ellipse centered at V/2 with major
        semiaxis sqrt(L E)/2 along V and minor semiaxis
        sqrt(L E-|V|^2)/2.  A coordinate box around that ellipse gives a
        cheap certified radial upper bound; the enclosing-circle bound is
        also evaluated and the tighter one is used.
        """
        e = max(float(energy[lo + length] - energy[lo]), 0.0)
        a = 0.5 * np.sqrt(length * e)
        h = before + 0.5 * vnode
        circle = abs(h) + a
        if not ellipse_box:
            return circle
        av = abs(vnode)
        if av <= 1e-30:
            return circle
        residual = max(length * e - av * av, 0.0)
        b = 0.5 * np.sqrt(residual)
        hr = h * np.conj(vnode) / av
        box = np.hypot(abs(hr.real) + a, abs(hr.imag) + b)
        return min(circle, box)

    def half_edge_prunable(lo, length, before, vnode, threshold):
        """Certified threshold test preserving endpoint/denominator.

        For u=j/L, A(u)=|before+uV|^2 and
        R(u)^2=D*u*(1-u), D=L*E-|V|^2.  The exact residual-disk bound is

            (sqrt(A(u)) + sqrt(R(u)^2))^2 / (lo + L*u).

        Put T(u)=threshold*(lo+L*u)-A(u)-R(u)^2.  The desired inequality
        holds wherever

            T(u) >= 0  and  H(u)=T(u)^2-4*A(u)*R(u)^2 >= 0.

        T is quadratic and H quartic.  Nonnegative Bernstein coefficients
        certify a polynomial on an interval; two fixed de Casteljau splits
        make the test tight while retaining branch-free, fixed-degree state.
        Failure merely means "split" and cannot change correctness.
        """
        nonlocal half_tests
        L = float(length)
        e = max(float(energy[lo + length] - energy[lo]), 0.0)
        vv = float(abs(vnode) ** 2)
        dres = max(L * e - vv, 0.0)
        a0 = float(abs(before) ** 2)
        a1 = float(2.0 * np.real(np.conj(before) * vnode))
        a2 = vv
        u0, u1 = 1.0 / L, 1.0

        def power_to_bernstein(power, ua, ub):
            """Power coefficients in u -> Bernstein coefficients on [ua,ub]."""
            n = len(power) - 1
            h = ub - ua
            # First compose u=ua+h*t into power coefficients in t.
            ct = np.zeros(n + 1)
            for i, ai in enumerate(power):
                for j in range(i + 1):
                    ct[j] += ai * math.comb(i, j) * ua ** (i - j) * h ** j
            b = np.zeros(n + 1)
            for k in range(n + 1):
                b[k] = sum(ct[i] * math.comb(k, i) / math.comb(n, i)
                           for i in range(k + 1))
            return b

        def bernstein_nonnegative(power, ua, ub, splits=2):
            b = power_to_bernstein(power, ua, ub)

            def rec(coeff, d):
                if np.min(coeff) >= 0.0:
                    return True
                if d == 0:
                    return False
                levels = [coeff]
                while len(levels[-1]) > 1:
                    q = levels[-1]
                    levels.append(0.5 * (q[:-1] + q[1:]))
                left = np.array([row[0] for row in levels])
                right = np.array([row[-1] for row in levels[::-1]])
                return rec(left, d - 1) and rec(right, d - 1)

            return rec(b, splits)

        A = np.array([a0, a1, a2])
        R = np.array([0.0, dres, -dres])
        T = np.array([threshold * lo - a0,
                      threshold * L - a1 - dres,
                      -a2 + dres])
        H = np.convolve(T, T) - 4.0 * np.convolve(A, R)
        half_tests += 1
        return (bernstein_nonnegative(T, u0, u1) and
                bernstein_nonnegative(H, u0, u1))

    def phase_disk(lo, length, d):
        """Disk containing every nonempty local prefix of this block."""
        nonlocal phase_cells
        key = (lo, length, d)
        cached = phase_cache.get(key)
        if cached is not None:
            return cached
        if d == 0 or length == 1:
            ans = (0j, energy_radius(lo, length))
        else:
            half = length // 2
            ca, ra = phase_disk(lo, half, d - 1)
            cb, rb = phase_disk(lo + half, half, d - 1)
            va = total(lo, half)
            ans = enclosing_two_disks(ca, ra, va + cb, rb)
            phase_cells += 1
        phase_cache[key] = ans
        return ans

    def achieved(tau):
        z = oracle.prefix(tau)
        return z, float(abs(z) ** 2 / tau)

    # The full-support value is supplied by the ordinary root FFT channel.
    # Dyadic seeds are optional: paying for all of them before the first gate
    # creates an avoidable N log N demand floor.
    seeds = [N]
    if seed_dyadic:
        p = 1
        while p <= N:
            seeds.append(p)
            p <<= 1
    if initial is not None and 1 <= int(initial) <= N:
        seeds.append(int(initial))
    vals = [(tau, *achieved(tau)) for tau in sorted(set(seeds))]
    tau_best, zbest, sbest = max(vals, key=lambda row: row[2])
    nodes = 0

    def visit(lo, length, before, vnode):
        nonlocal tau_best, zbest, sbest, nodes
        nodes += 1
        denom = lo + 1

        # Cheap first gate: it can avoid constructing any phase cells.
        er = energy_radius(lo, length)
        prune_score = max(sbest, float(activity) if activity is not None else sbest)
        if (abs(before) + er) ** 2 / denom <= prune_score * (1.0 + 1e-14):
            return

        # Intrinsic half-edge disk.  For a local prefix P and block total V,
        #
        #   P - V/2 = 1/2 Sum s_n y_n,  s_n in {+1,-1},
        #
        # hence |P-V/2| <= sqrt(L Sum|y|^2)/2.  The node total is already
        # carried by the walk, so this phase-aware bound costs no new channel.
        if half_edge:
            # Very cheap enclosing ellipse/circle first, then the stronger
            # denominator-coupled certificate only in the ambiguous region.
            if half_edge_upper(lo, length, before, vnode) ** 2 / denom \
                    <= prune_score * (1.0 + 1e-14):
                return
            if coupled and half_edge_prunable(
                    lo, length, before, vnode, prune_score):
                return

        dc, dr = phase_disk(lo, length, depth)
        if (abs(before + dc) + dr) ** 2 / denom \
                <= prune_score * (1.0 + 1e-14):
            return

        if length == 1:
            z = before + vnode
            s = float(abs(z) ** 2 / (lo + 1))
            if s > sbest:
                tau_best, zbest, sbest = lo + 1, z, s
            return

        half = length // 2
        vl = total(lo, half)
        vr = vnode - vl
        # Both endpoint values are already available from the carried parent
        # total and the one demanded left total; no oracle score is free.
        sl = abs(before + vl) ** 2 / (lo + half)
        sr = abs(before + vnode) ** 2 / (lo + length)
        if sr > sl:
            visit(lo + half, half, before + vl, vr)
            visit(lo, half, before, vl)
        else:
            visit(lo, half, before, vl)
            visit(lo + half, half, before + vl, vr)

    root_total = total(0, N)
    visit(0, N, 0j, root_total)

    # Brute force is a certification assertion only and contributes no values
    # to ``used`` or to the proposed execution ledger.
    n = np.arange(N)
    truth_c = np.cumsum(x * np.exp(-1j * omega * n))
    truth_s = np.abs(truth_c) ** 2 / (n + 1)
    truth = int(np.argmax(truth_s))
    if activity is None or truth_s[truth] > activity:
        assert tau_best == truth + 1
        assert abs(zbest - truth_c[truth]) <= 1e-9 * (1 + abs(zbest))
    else:
        # Below the semantic activity floor the transform deliberately emits
        # the ordinary full-support Fourier value.
        tau_best, zbest, sbest = N, root_total, float(abs(root_total) ** 2 / N)
    return (tau_best, zbest, sbest, oracle.used, nodes, phase_cells,
            half_tests)


def evaluate(N=1024, case="chirp", depths=(0, 1, 2, 3, 4), warm=True,
             half_edge=True, seed_dyadic=False, certified_activity=True,
             coupled=False, ellipse_box=False):
    """Corrected all-bin ledger for the shared fractional-channel kernel."""
    rng = np.random.default_rng(22)
    x = make_cases(N, rng)[case]
    mean = float(np.mean(np.abs(x) ** 2))
    gate = (np.log(N) + 2) * mean
    fft = 0.5 * N * np.log2(N)
    rows = {}
    for depth in depths:
        demands = {}
        active = nodes = cells = pairs = tests = 0
        prev = None
        for k in range(N):
            tau, _, score, used, nn, cc, ht = select_support_lazy_phase(
                x, 2 * np.pi * k / N, depth=depth,
                initial=prev if warm else None, half_edge=half_edge,
                seed_dyadic=seed_dyadic,
                activity=gate if certified_activity else None,
                coupled=coupled,
                ellipse_box=ellipse_box,
            )
            # Search work is retained for every bin.  Activity only controls
            # warm-start continuity and is reported as output semantics.
            if score > gate:
                active += 1
                prev = tau
            else:
                prev = None
            nodes += nn
            cells += cc
            tests += ht
            for block in used:
                demands.setdefault(block, set()).add(k)
                pairs += 1
        work, per_scale = shared_cost(demands, N)
        ratio = work / fft
        rows[depth] = ratio
        tag = ("coupled" if half_edge and coupled else
               "ellipse" if half_edge and ellipse_box else
               "disk" if half_edge else "energy")
        print(f"[{case} N={N} {tag} d={depth}] active={active}/{N} "
              f"nodes/bin={nodes/N:.1f} phase-cells/bin={cells/N:.1f} "
              f"half-tests/bin={tests/N:.1f} pairs={pairs} "
              f"shared-V={ratio:.2f} /FFT")
        print("  V ops by scale:", " ".join(
            f"L{L}:{per_scale[L]}" for L in sorted(per_scale, reverse=True)))
    return rows


if __name__ == "__main__":
    for size in (256, 512, 1024):
        evaluate(size)
