#!/usr/bin/env python3
"""Intrinsic support packets: exact set closure and compressed phase frontiers.

This is deliberately a scalar-frequency algebra experiment, not a fast
transform claim.  It answers the prerequisite question for a future DIP walk:
can the exact support-carrying packet type be compressed to a small width M?

For a local time block A, a packet carries

    total_A = sum x[t] exp(-i*w*t)
    frontier_A = {(tau, prefix_A(tau))}.

Adjacent blocks combine exactly by translating/rotating B's frontier through
A.  Pruning is based on the upper envelope under a possible preceding phasor
`a` and preceding length `L`:

    q_j(a,L) = |a + z_j|^2 / (L + tau_j).

A candidate earns retention when it wins one of a deterministic polar/length
probe family.  This is the correct geometry for future packet combination;
top-M local score is included only as a baseline.

The prototype compares bounded packet widths against the brute-force leading-
edge manifold on tones, bursts, chirps, crossings, and complex noise.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class Packet:
    length: int
    total: complex
    tau: np.ndarray
    z: np.ndarray
    work: int = 0


@dataclass
class ResidualDisk:
    center: complex
    radius: float
    tau_min: int
    tau_max: int


@dataclass
class BeamPacket(Packet):
    residual: ResidualDisk | None = None


@dataclass
class BandedBeamPacket(Packet):
    residuals: tuple = ()


def _enclose_disks(disks):
    if not disks:
        return None
    c = complex(disks[0].center)
    r = float(disks[0].radius)
    tmin = int(disks[0].tau_min)
    tmax = int(disks[0].tau_max)
    for dsk in disks[1:]:
        c2, r2 = complex(dsk.center), float(dsk.radius)
        d = abs(c2 - c)
        if r >= d + r2:
            pass
        elif r2 >= d + r:
            c, r = c2, r2
        else:
            R = 0.5 * (d + r + r2)
            if d > 1e-30:
                c = c + ((R - r) / d) * (c2 - c)
            r = R
        tmin = min(tmin, int(dsk.tau_min))
        tmax = max(tmax, int(dsk.tau_max))
    return ResidualDisk(c, r, tmin, tmax)


def _candidate_priority(tau, z, block_len, n_probe_angle=32):
    """Return upper-envelope win counts and margins for candidate prefixes."""
    m = tau.size
    if m <= 1:
        return np.ones(m, dtype=np.int64), np.ones(m)

    scale = max(float(np.max(np.abs(z))), 1e-15)
    angles = 2 * np.pi * (np.arange(n_probe_angle) + 0.5) / n_probe_angle
    # Cover translations smaller and larger than the block's own phasors.
    radii = scale * np.array([0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    # A block can occur under a short or much longer already-accepted prefix.
    lengths = block_len * np.array([0.0, 0.125, 0.5, 1.0, 2.0, 8.0])

    aa = []
    ll = []
    unit = np.exp(1j * angles)
    for length in lengths:
        aa.append(np.array([0j]))
        ll.append(np.array([length]))
        for rr in radii[1:]:
            aa.append(rr * unit)
            ll.append(np.full(unit.size, length))
    aa = np.concatenate(aa)
    ll = np.concatenate(ll)
    score = np.abs(aa[:, None] + z[None, :]) ** 2 / \
            (ll[:, None] + tau[None, :])
    winner = np.argmax(score, axis=1)
    pair = np.partition(score, -2, axis=1)[:, -2:]
    gap = pair[:, 1] - pair[:, 0]
    wins = np.bincount(winner, minlength=m).astype(np.int64)
    margin = np.zeros(m, dtype=float)
    np.maximum.at(margin, winner, gap)
    return wins, margin


def prune_frontier(tau, z, width, block_len, method="envelope"):
    if width is None or tau.size <= width:
        return tau, z
    score0 = np.abs(z) ** 2 / tau
    if method == "top":
        keep = np.argpartition(score0, -width)[-width:]
    elif method == "envelope":
        wins, margin = _candidate_priority(tau, z, block_len)
        # Required geometry anchors: earliest/latest, local winner, and angular
        # extrema. The remaining slots go to envelope win count then margin.
        required = {0, tau.size - 1, int(np.argmax(score0))}
        nang = max(4, min(16, width // 2))
        angle = (np.angle(z) + 2 * np.pi) % (2 * np.pi)
        for q in range(nang):
            mask = (angle >= 2 * np.pi * q / nang) & \
                   (angle < 2 * np.pi * (q + 1) / nang)
            ids = np.flatnonzero(mask)
            if ids.size:
                required.add(int(ids[np.argmax(np.abs(z[ids]))]))
        # Lexicographic ranking, with local score as the final tie breaker.
        order = np.lexsort((score0, margin, wins))[::-1]
        selected = list(required)
        for idx in order:
            if int(idx) not in required:
                selected.append(int(idx))
            if len(selected) >= width:
                break
        if len(selected) > width:
            # Never drop first/last/local-best; rank other angular anchors.
            core = [0, tau.size - 1, int(np.argmax(score0))]
            core = list(dict.fromkeys(core))
            rest = [j for j in selected if j not in core]
            rest.sort(key=lambda j: (wins[j], margin[j], score0[j]), reverse=True)
            selected = core + rest[:width - len(core)]
        keep = np.asarray(selected, dtype=np.intp)
    else:
        raise ValueError(method)
    keep = np.unique(keep)
    keep.sort()
    return tau[keep], z[keep]


def packet_tree(x, omega, width=None, method="envelope"):
    """Balanced exact-combine tree, with optional width-M frontier pruning."""
    x = np.asarray(x, dtype=np.complex128)

    def rec(lo, hi):
        n = hi - lo
        if n == 1:
            return Packet(1, complex(x[lo]), np.array([1], np.int64),
                          np.array([x[lo]], np.complex128), 1)
        mid = lo + n // 2
        A = rec(lo, mid)
        B = rec(mid, hi)
        rot = np.exp(-1j * omega * A.length)
        bz = A.total + rot * B.z
        tau = np.concatenate((A.tau, A.length + B.tau))
        z = np.concatenate((A.z, bz))
        tau, z = prune_frontier(tau, z, width, n, method)
        return Packet(n, A.total + rot * B.total, tau, z,
                      A.work + B.work + A.tau.size + B.tau.size)

    return rec(0, x.size)


def certified_beam_tree(x, omega, width=16, method="envelope"):
    """Width-M frontier plus one disk enclosing every discarded candidate.

    The root answer is certified exact whenever the residual disk's safe score
    bound cannot beat the retained winner. Uncertified roots must widen or fall
    back to phase-disk descent; they are never silently accepted.
    """
    x = np.asarray(x, dtype=np.complex128)

    def rec(lo, hi):
        n = hi - lo
        if n == 1:
            z = complex(x[lo])
            return BeamPacket(1, z, np.array([1], np.int64),
                              np.array([z], np.complex128), 1, None)
        mid = lo + n // 2
        A = rec(lo, mid)
        B = rec(mid, hi)
        rot = np.exp(-1j * omega * A.length)
        tau = np.concatenate((A.tau, A.length + B.tau))
        z = np.concatenate((A.z, A.total + rot * B.z))
        ktau, kz = prune_frontier(tau, z, width, n, method)
        keep = np.isin(tau, ktau)
        disks = []
        if A.residual is not None:
            disks.append(A.residual)
        if B.residual is not None:
            d = B.residual
            disks.append(ResidualDisk(A.total + rot * d.center, d.radius,
                                      A.length + d.tau_min,
                                      A.length + d.tau_max))
        for tj, zj in zip(tau[~keep], z[~keep]):
            disks.append(ResidualDisk(complex(zj), 0.0, int(tj), int(tj)))
        residual = _enclose_disks(disks)
        return BeamPacket(n, A.total + rot * B.total, ktau, kz,
                          A.work + B.work + tau.size, residual)

    return rec(0, x.size)


def certified_beam_answer(packet):
    tau, z, score = packet_answer(packet)
    if packet.residual is None:
        return tau, z, score, True, 0.0
    d = packet.residual
    upper = (abs(d.center) + d.radius) ** 2 / d.tau_min
    return tau, z, score, bool(upper <= score * (1 + 1e-14)), float(upper)


def _band_residuals(disks, phase_sectors=8):
    """Merge residual geometry by denominator scale and phase cone."""
    groups = {}
    for d in disks:
        band = int(np.floor(np.log2(max(d.tau_min, 1))))
        angle = (np.angle(d.center) + 2 * np.pi) % (2 * np.pi)
        sector = int(np.floor(phase_sectors * angle / (2 * np.pi))) % phase_sectors
        groups.setdefault((band, sector), []).append(d)
    return tuple(_enclose_disks(groups[b]) for b in sorted(groups))


def certified_banded_beam_tree(x, omega, width=16, method="envelope"):
    x = np.asarray(x, dtype=np.complex128)

    def rec(lo, hi):
        n = hi - lo
        if n == 1:
            z = complex(x[lo])
            return BandedBeamPacket(1, z, np.array([1], np.int64),
                                    np.array([z], np.complex128), 1, ())
        mid = lo + n // 2
        A = rec(lo, mid)
        B = rec(mid, hi)
        rot = np.exp(-1j * omega * A.length)
        tau = np.concatenate((A.tau, A.length + B.tau))
        z = np.concatenate((A.z, A.total + rot * B.z))
        ktau, kz = prune_frontier(tau, z, width, n, method)
        keep = np.isin(tau, ktau)
        disks = list(A.residuals)
        for d in B.residuals:
            disks.append(ResidualDisk(A.total + rot * d.center, d.radius,
                                      A.length + d.tau_min,
                                      A.length + d.tau_max))
        for tj, zj in zip(tau[~keep], z[~keep]):
            disks.append(ResidualDisk(complex(zj), 0.0, int(tj), int(tj)))
        return BandedBeamPacket(n, A.total + rot * B.total, ktau, kz,
                                A.work + B.work + tau.size,
                                _band_residuals(disks))

    return rec(0, x.size)


def certified_banded_beam_answer(packet):
    tau, z, score = packet_answer(packet)
    upper = 0.0
    for d in packet.residuals:
        upper = max(upper, (abs(d.center) + d.radius) ** 2 / d.tau_min)
    return tau, z, score, bool(upper <= score * (1 + 1e-14)), float(upper)


def packet_answer(packet):
    s = np.abs(packet.z) ** 2 / packet.tau
    j = int(np.argmax(s))
    return int(packet.tau[j]), packet.z[j], float(s[j])


def brute_answer(x, omega):
    t = np.arange(x.size)
    c = np.cumsum(x * np.exp(-1j * omega * t))
    s = np.abs(c) ** 2 / (t + 1)
    j = int(np.argmax(s))
    return j + 1, c[j], float(s[j])


def signals(N, rng):
    t = np.arange(N)
    out = {}
    out["tone"] = np.exp(2j * np.pi * 37.2 * t / N)
    burst = 0.015 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    m = (t >= 35) & (t < 171)
    burst[m] += np.exp(2j * np.pi * 83.4 * t[m] / N + 0.4j)
    out["burst"] = burst
    late = 0.01 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    m = (t >= N - 48) & (t < N - 7)
    late[m] += 1.4 * np.exp(-2j * np.pi * 51.0 * t[m] / N)
    out["late-short"] = late
    out["chirp"] = np.exp(2j * np.pi * (18 * t / N + 0.38 * t * t / N))
    cross = np.exp(2j * np.pi * (25 * t / N + 0.22 * t * t / N))
    cross += 0.85 * np.exp(2j * np.pi * (118 * t / N - 0.19 * t * t / N + 0.7))
    out["crossing"] = cross
    out["noise"] = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    return out


def evaluate(N=256, widths=(4, 8, 16, 32), methods=("top", "envelope")):
    rng = np.random.default_rng(20260709)
    kk = np.arange(N)
    cases = signals(N, rng)
    print("Intrinsic support frontier experiment")
    print(f"N={N}, {len(cases)} signals, {N} full complex bins")
    print("ratio = selected score / brute global score")
    print()
    results = []
    for name, x in cases.items():
        brute = [brute_answer(x, 2 * np.pi * k / N) for k in kk]
        brute_s = np.array([b[2] for b in brute])
        active = brute_s > (np.log(N) + 2.0) * np.mean(np.abs(x) ** 2)
        if not np.any(active):
            active[:] = True
        print(f"[{name}] active {active.sum()}/{N}")
        for method in methods:
            for width in widths:
                tau = np.empty(N, int)
                score = np.empty(N)
                work = 0
                t0 = time.perf_counter()
                for k in kk:
                    p = packet_tree(x, 2 * np.pi * k / N, width, method)
                    tau[k], _, score[k] = packet_answer(p)
                    work += p.work
                elapsed = time.perf_counter() - t0
                ratio = score[active] / (brute_s[active] + 1e-30)
                bt = np.array([b[0] for b in brute])
                terr = np.abs(tau[active] - bt[active])
                exact = np.mean(terr == 0)
                row = (name, method, width, float(np.median(ratio)),
                       float(np.percentile(ratio, 10)), float(exact),
                       float(np.median(terr)), work, elapsed)
                results.append(row)
                print(f"  {method:8s} M={width:2d}: ratio med {row[3]:.4f} "
                      f"p10 {row[4]:.4f} exact-tau {row[5]:.3f} "
                      f"tauerr-med {row[6]:.1f} work {work/1e6:.2f}M "
                      f"{elapsed:.2f}s")
        print()
    return results


def frontier_census(x, bins=None):
    """Probe-estimated width of the exact future-translation envelope by level."""
    x = np.asarray(x, np.complex128)
    N = x.size
    if bins is None:
        bins = np.arange(N)
    print("Probe-envelope census (unpruned exact prefix sets)")
    print("  L     nodes*bins   median   p90   p99   max")
    L = 2
    out = {}
    while L <= N:
        widths = []
        local_t = np.arange(L)
        for lo in range(0, N, L):
            block = x[lo:lo + L]
            for k in bins:
                z = np.cumsum(block * np.exp(-2j * np.pi * k * local_t / N))
                wins, _ = _candidate_priority(np.arange(1, L + 1), z, L)
                widths.append(int(np.count_nonzero(wins)))
        w = np.asarray(widths)
        stats = (float(np.median(w)), float(np.percentile(w, 90)),
                 float(np.percentile(w, 99)), int(np.max(w)))
        out[L] = stats
        print(f"  {L:4d} {w.size:12d} {stats[0]:8.1f} {stats[1]:6.1f} "
              f"{stats[2]:6.1f} {stats[3]:5d}")
        L <<= 1
    return out


def monte_carlo_noise(N=128, trials=12, width=16, method="envelope"):
    """Adversarial audit: no activity gate, every bin of complex Gaussian noise."""
    rng = np.random.default_rng(9917)
    ratios = []
    exact = []
    for trial in range(trials):
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        for k in range(N):
            omega = 2 * np.pi * k / N
            bt, _, bs = brute_answer(x, omega)
            p = packet_tree(x, omega, width, method)
            pt, _, ps = packet_answer(p)
            ratios.append(ps / (bs + 1e-30))
            exact.append(pt == bt)
    r = np.asarray(ratios)
    print(f"noise MC N={N} trials={trials} M={width} {method}: "
          f"ratio med={np.median(r):.5f} p10={np.percentile(r,10):.5f} "
          f"p1={np.percentile(r,1):.5f} min={r.min():.5f} "
          f"exact-tau={np.mean(exact):.3f}")
    return r, np.asarray(exact)


if __name__ == "__main__":
    evaluate()
