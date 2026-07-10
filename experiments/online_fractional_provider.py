#!/usr/bin/env python3
"""Executable online provider for intrinsic DIP block totals V(B,k).

The selector asks for

    V([lo,lo+L), k) = Sum x[n] exp(-2 pi i k n/N)

in an adaptive order.  This provider has no dense prefix-correlation oracle.
Every returned value is produced through one of two exact routes:

DIRECT
    Open an incremental pruned twisted-FFT channel for the block and residue
    delta = k mod (N/L).  Butterfly pairs are memoized, so later outputs in
    the same channel reuse every previously opened cell.  In any request
    order, the final cell count equals ``pruned_cell_count(L, requested_p)``.

SYNTH
    Use V(parent,k)=V(left,k)+V(right,k).  Child requests recursively choose
    their cheapest marginal route and the parent value costs one complex add.

The online router compares exact marginal cell counts against the current
cache.  It is not clairvoyant (the offline sharing DP remains a lower bound),
but it is executable, exact, and naturally exposes frequency-bin batches for
the half-edge SIMD selector.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fractional_channel_sharing import (
    direct_cost, dyadic_blocks, shared_cost)
from experiments.intrinsic_dip_branchbound import make_cases
from experiments.twisted_fractional_dip import pruned_cell_count


class IncrementalTwistedChannel:
    """One exact (block, fractional residue) output-pruned FFT channel."""

    def __init__(self, block, alpha):
        self.x = np.asarray(block, np.complex128)
        self.alpha = float(alpha)
        # key=(start,stride,length,r); value=(low output, high output)
        self.cells: dict[tuple[int, int, int, int], tuple[complex, complex]] = {}
        self.cell_count = 0
        self.twiddle_count = 0

    def _marginal(self, start, stride, length, p):
        if length == 1:
            return 0
        half = length // 2
        r = int(p % half)
        key = (start, stride, length, r)
        if key in self.cells:
            return 0
        return (1 + self._marginal(start, stride * 2, half, r) +
                self._marginal(start + stride, stride * 2, half, r))

    def marginal(self, p):
        return self._marginal(0, 1, self.x.size, int(p))

    def _output(self, start, stride, length, p):
        if length == 1:
            return complex(self.x[start])
        half = length // 2
        r = int(p % half)
        key = (start, stride, length, r)
        pair = self.cells.get(key)
        if pair is None:
            even = self._output(start, stride * 2, half, r)
            odd = self._output(start + stride, stride * 2, half, r)
            w = np.exp(-2j * np.pi * (r + self.alpha) / length)
            z = w * odd
            pair = (even + z, even - z)
            self.cells[key] = pair
            self.cell_count += 1
            self.twiddle_count += 1
        return pair[0] if p < half else pair[1]

    def output(self, p):
        return self._output(0, 1, self.x.size, int(p))


@dataclass
class ProviderStats:
    direct_values: int = 0
    synth_values: int = 0
    leaf_values: int = 0
    phase_multiplies: int = 0


class OnlineFractionalProvider:
    """Memoized exact service for dyadic ``V(block,k)`` requests."""

    def __init__(self, x, policy="marginal"):
        self.x = np.asarray(x, np.complex128)
        self.N = self.x.size
        assert self.N > 0 and self.N & (self.N - 1) == 0
        assert policy in ("direct", "synth", "marginal")
        self.policy = policy
        self.values: dict[tuple[int, int, int], complex] = {}
        self.routes: dict[tuple[int, int, int], str] = {}
        self.channels: dict[tuple[int, int, int], IncrementalTwistedChannel] = {}
        self.stats = ProviderStats()

    def _coords(self, lo, length, k):
        e = self.N // length
        delta = int(k % e)
        p = int((k - delta) // e)
        return e, delta, p

    def _channel(self, lo, length, delta):
        key = (lo, length, delta)
        channel = self.channels.get(key)
        if channel is None:
            e = self.N // length
            channel = IncrementalTwistedChannel(
                self.x[lo:lo + length], delta / e)
            self.channels[key] = channel
        return channel

    def direct_marginal(self, lo, length, k):
        key = (lo, length, int(k))
        if key in self.values:
            return 0
        if length == 1:
            return 1
        _, delta, p = self._coords(lo, length, k)
        channel = self.channels.get((lo, length, delta))
        cells = length - 1 if channel is None else channel.marginal(p)
        return cells + (lo != 0)

    def estimate(self, lo, length, k):
        """Current-state marginal estimate; does not mutate computed cells."""
        key = (lo, length, int(k))
        if key in self.values:
            return 0
        direct = self.direct_marginal(lo, length, k)
        if length == 1 or self.policy == "direct":
            return direct
        half = length // 2
        synth = (1 + self.estimate(lo, half, k) +
                 self.estimate(lo + half, half, k))
        if self.policy == "synth":
            return synth
        return min(direct, synth)

    def _direct(self, lo, length, k):
        if length == 1:
            self.stats.leaf_values += 1
            return self.x[lo] * np.exp(-2j * np.pi * k * lo / self.N)
        _, delta, p = self._coords(lo, length, k)
        local = self._channel(lo, length, delta).output(p)
        self.stats.direct_values += 1
        if lo:
            local *= np.exp(-2j * np.pi * k * lo / self.N)
            self.stats.phase_multiplies += 1
        return local

    def get(self, lo, length, k):
        lo, length, k = int(lo), int(length), int(k) % self.N
        key = (lo, length, k)
        cached = self.values.get(key)
        if cached is not None:
            return cached
        assert length > 0 and length & (length - 1) == 0
        assert lo % length == 0 and 0 <= lo < self.N and lo + length <= self.N

        direct = self.direct_marginal(lo, length, k)
        use_synth = False
        if length > 1 and self.policy != "direct":
            half = length // 2
            synth = (1 + self.estimate(lo, half, k) +
                     self.estimate(lo + half, half, k))
            use_synth = self.policy == "synth" or synth < direct

        if use_synth:
            half = length // 2
            value = self.get(lo, half, k) + self.get(lo + half, half, k)
            self.stats.synth_values += 1
            route = "synth"
        else:
            value = self._direct(lo, length, k)
            route = "direct"
        self.values[key] = complex(value)
        self.routes[key] = route
        return self.values[key]

    @property
    def butterfly_cells(self):
        return sum(ch.cell_count for ch in self.channels.values())

    @property
    def twiddles(self):
        return sum(ch.twiddle_count for ch in self.channels.values())

    @property
    def modeled_work(self):
        return (self.butterfly_cells + self.stats.synth_values +
                self.stats.leaf_values + self.stats.phase_multiplies)

    @property
    def cell_memory(self):
        return self.butterfly_cells


def select_support_online(x, k, provider, activity=None, initial=None):
    """Exact half-edge support selector using only provider-returned values."""
    x = np.asarray(x, np.complex128)
    N = x.size
    energy = np.concatenate(([0.0], np.cumsum(np.abs(x) ** 2)))
    consumed: set[tuple[int, int]] = set()
    nodes = 0

    def total(lo, length):
        consumed.add((lo, length))
        return provider.get(lo, length, k)

    def prefix(tau):
        return sum((total(lo, length)
                    for lo, length in dyadic_blocks(0, int(tau))), 0j)

    root = total(0, N)
    tau_best, zbest = N, root
    sbest = float(abs(root) ** 2 / N)
    if initial is not None and 1 <= int(initial) <= N:
        zi = prefix(int(initial))
        si = float(abs(zi) ** 2 / int(initial))
        if si > sbest:
            tau_best, zbest, sbest = int(initial), zi, si

    def visit(lo, length, before, vnode):
        nonlocal tau_best, zbest, sbest, nodes
        nodes += 1
        e = max(float(energy[lo + length] - energy[lo]), 0.0)
        er = np.sqrt(length * e)
        threshold = max(sbest, float(activity) if activity is not None else sbest)

        if (abs(before) + er) ** 2 / (lo + 1) \
                <= threshold * (1.0 + 1e-14):
            return
        if (abs(before + 0.5 * vnode) + 0.5 * er) ** 2 / (lo + 1) \
                <= threshold * (1.0 + 1e-14):
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
        sl = abs(before + vl) ** 2 / (lo + half)
        sr = abs(before + vnode) ** 2 / (lo + length)
        if sr > sl:
            visit(lo + half, half, before + vl, vr)
            visit(lo, half, before, vl)
        else:
            visit(lo, half, before, vl)
            visit(lo + half, half, before + vl, vr)

    visit(0, N, 0j, root)

    # Certification assertion only; it is not used by the selector/provider.
    n = np.arange(N)
    truth_c = np.cumsum(x * np.exp(-2j * np.pi * k * n / N))
    truth_s = np.abs(truth_c) ** 2 / (n + 1)
    truth = int(np.argmax(truth_s))
    if activity is None or truth_s[truth] > activity:
        assert tau_best == truth + 1
        assert abs(zbest - truth_c[truth]) <= 1e-9 * (1 + abs(zbest))
    else:
        tau_best, zbest, sbest = N, root, float(abs(root) ** 2 / N)
    return tau_best, zbest, sbest, consumed, nodes


def evaluate(N=256, case="chirp", policies=("direct", "marginal")):
    rng = np.random.default_rng(22)
    x = make_cases(N, rng)[case]
    gate = (np.log(N) + 2) * np.mean(np.abs(x) ** 2)
    fft = 0.5 * N * np.log2(N)
    out = {}
    for policy in policies:
        provider = OnlineFractionalProvider(x, policy=policy)
        demands = {}
        prev = None
        active = nodes = 0
        for k in range(N):
            tau, _, score, used, nn = select_support_online(
                x, k, provider, activity=gate, initial=prev)
            if score > gate:
                active += 1
                prev = tau
            else:
                prev = None
            nodes += nn
            for block in used:
                demands.setdefault(block, set()).add(k)
        offline, _ = shared_cost(demands, N)
        direct_ledger = sum(direct_cost(N, length, bins)
                            for (_, length), bins in demands.items())
        realized_direct = provider.butterfly_cells + provider.stats.leaf_values
        if policy == "direct":
            assert realized_direct == direct_ledger, (
                realized_direct, direct_ledger)
        ratio = provider.modeled_work / fft
        print(f"[{case} N={N} {policy}] active={active}/{N} "
              f"nodes/bin={nodes/N:.1f} values={len(provider.values)} "
              f"channels={len(provider.channels)} cells={provider.butterfly_cells}")
        print(f"  direct={provider.stats.direct_values} "
              f"synth={provider.stats.synth_values} "
              f"leaf={provider.stats.leaf_values} "
              f"phase-mul={provider.stats.phase_multiplies} "
              f"modeled={ratio:.2f}/FFT direct-ledger={direct_ledger/fft:.2f}/FFT "
              f"offline-lower={offline/fft:.2f}/FFT "
              f"gap={provider.modeled_work/max(offline, 1):.2f}x")
        out[policy] = ratio
    return out


def self_test():
    rng = np.random.default_rng(905)
    for length in (8, 32, 128):
        block = rng.standard_normal(length) + 1j * rng.standard_normal(length)
        channel = IncrementalTwistedChannel(block, 3 / 11)
        requested = []
        for p in rng.permutation(length):
            requested.append(int(p))
            before = channel.cell_count
            marginal = channel.marginal(p)
            got = channel.output(p)
            ref = np.sum(block * np.exp(
                -2j * np.pi * (p + 3 / 11) * np.arange(length) / length))
            assert abs(got - ref) <= 1e-9 * (1 + abs(ref))
            assert channel.cell_count - before == marginal
            assert channel.cell_count == pruned_cell_count(length, requested)
        print(f"PASS incremental channel L={length}: request-order exact, "
              f"full cells={channel.cell_count}")
    for N in (16, 32, 64):
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        t = np.arange(N)
        provider = OnlineFractionalProvider(x)
        for _ in range(500):
            length = 1 << int(rng.integers(0, int(np.log2(N)) + 1))
            lo = int(rng.integers(0, N // length)) * length
            k = int(rng.integers(0, N))
            got = provider.get(lo, length, k)
            ref = np.sum(x[lo:lo + length] *
                         np.exp(-2j * np.pi * k * t[lo:lo + length] / N))
            assert abs(got - ref) <= 1e-9 * (1 + abs(ref))
        print(f"PASS online provider N={N}: 500 mixed requests exact, "
              f"{provider.butterfly_cells} cached cells")


if __name__ == "__main__":
    self_test()
    for size in (128, 256, 512):
        evaluate(size)
