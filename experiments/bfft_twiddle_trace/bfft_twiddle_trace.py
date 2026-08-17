#!/usr/bin/env python3
"""Trace and plot the normalized Bruun twiddles used by BFFT.

The production kernels fuse, vectorize, and reorder the logical cells.  This
module follows the same logical normalized-basis cells while attaching a bitset
of twiddle-operation ancestry to every scalar state coordinate.  A plotted cell
therefore means

    output coordinate (row) structurally depends on twiddle operation (column).

The color is the Givens angle theta/pi.  Columns are operation groups rather
than table values: two operations which happen to reuse the same numerical
angle remain distinct.  Conversely, a SIMD/broadcast use of one operation over
several lanes is one column.  This gives all three N-point walks the same
sum(2**j - 1) column count and makes their recursive geometry comparable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Factor:
    group: int
    position: int
    angle: float
    label: str


@dataclass(frozen=True)
class Trace:
    form: str
    support: np.ndarray
    factors: tuple[Factor, ...]
    group_labels: tuple[str, ...]
    group_edges: tuple[int, ...]

    @property
    def angles_over_pi(self) -> np.ndarray:
        return np.asarray([f.angle / np.pi for f in self.factors])


def _check_size(n: int) -> None:
    if n < 8 or n & (n - 1):
        raise ValueError("N must be a power of two at least 8")


def _factor_layout(
    groups: Sequence[tuple[str, Iterable[tuple[int, float, str]]]],
) -> tuple[tuple[Factor, ...], dict[tuple[int, int], int], tuple[str, ...], tuple[int, ...]]:
    factors: list[Factor] = []
    index: dict[tuple[int, int], int] = {}
    labels: list[str] = []
    edges = [0]
    for group, (group_label, entries) in enumerate(groups):
        labels.append(group_label)
        for position, angle, label in entries:
            index[(group, position)] = len(factors)
            factors.append(Factor(group, position, angle, label))
        edges.append(len(factors))
    return tuple(factors), index, tuple(labels), tuple(edges)


def _masks_to_support(masks: Sequence[int], width: int) -> np.ndarray:
    packed = np.zeros((len(masks), (width + 7) // 8), dtype=np.uint8)
    for row, mask in enumerate(masks):
        packed[row] = np.frombuffer(
            int(mask).to_bytes(packed.shape[1], "little"), dtype=np.uint8
        )
    return np.unpackbits(packed, axis=1, count=width, bitorder="little").astype(bool)


def _packed_masks(dc: int, nyquist: int, pairs: Sequence[tuple[int, int]]) -> list[int]:
    out = [dc]
    for a, b in pairs:
        out.extend((a, b))
    out.append(nyquist)
    return out


def _bitrev(value: int, bits: int) -> int:
    out = 0
    for _ in range(bits):
        out = (out << 1) | (value & 1)
        value >>= 1
    return out


def _graydecode(value: int) -> int:
    out = value
    while value:
        value >>= 1
        out ^= value
    return out


def bruun_index(node: int, logn: int) -> int:
    """Python transcription of bruun_idx_int in bruun_simd_backend.hpp."""
    level = node.bit_length() - 1
    residue = node ^ (1 << level)
    return (2 * _graydecode(_bitrev(residue, level)) + 1) << (logn - 2 - level)


def _union4(a: int, b: int, c: int, d: int, bit: int) -> tuple[int, int, int, int]:
    merged = a | b | c | d | bit
    return merged, merged, merged, merged


def trace_dif(n: int) -> Trace:
    """Trace src/detail/bruun_dif_kernel.hpp::forward_residues_inplace."""
    _check_size(n)
    logn = n.bit_length() - 1
    groups = []
    for jj in range(1, logn - 1):
        entries = []
        for node in range(1, 1 << jj):
            idx = bruun_index(node, logn)
            entries.append((node, np.pi * idx / n, f"level={jj}, node={node}, k={idx}"))
        groups.append((f"L{jj}", entries))
    factors, factor_index, group_labels, edges = _factor_layout(groups)

    state = [0] * n
    for jj in range(logn - 1):
        span = n >> jj
        half = span >> 1
        quarter = span >> 2
        for i in range(half):
            merged = state[i] | state[half + i]
            state[i] = merged
            state[half + i] = merged
        if jj == 0:
            continue
        for node in range(1, 1 << jj):
            bit = 1 << factor_index[(jj - 1, node)]
            base = node * span
            for lane in range(quarter):
                merged = (
                    state[base + lane]
                    | state[base + quarter + lane]
                    | state[base + 2 * quarter + lane]
                    | state[base + 3 * quarter + lane]
                    | bit
                )
                state[base + lane] = merged
                state[base + quarter + lane] = merged
                state[base + 2 * quarter + lane] = merged
                state[base + 3 * quarter + lane] = merged

    dc_ny = state[0] | state[1]
    pairs: list[tuple[int, int]] = []
    inverse = [0] * (n // 2)
    for node in range(1, n // 2):
        inverse[bruun_index(node, logn)] = node
    for k in range(1, n // 2):
        node = inverse[k]
        pairs.append((state[2 * node], state[2 * node + 1]))
    masks = _packed_masks(dc_ny, dc_ny, pairs)
    return Trace("DIF", _masks_to_support(masks, len(factors)), factors, group_labels, edges)


def _dit_butterfly(
    q0: Sequence[int], q1: Sequence[int], q2: Sequence[int], q3: Sequence[int],
    i: int, bit2: int, bit_lo: int, bit_hi: int,
) -> tuple[int, ...]:
    h0 = _union4(q0[2 * i], q0[2 * i + 1], q1[2 * i], q1[2 * i + 1], bit2)
    h1 = _union4(q2[2 * i], q2[2 * i + 1], q3[2 * i], q3[2 * i + 1], bit2)
    low = _union4(h0[0], h0[1], h1[0], h1[1], bit_lo)
    high = _union4(h0[2], h0[3], h1[2], h1[3], bit_hi)
    return low + high


def _dit_write(block: list[int], half: int, quarter: int, i: int, out: Sequence[int]) -> None:
    slots = (i, half - i, quarter - i, quarter + i)
    for slot, pair in zip(slots, ((0, 1), (2, 3), (4, 5), (6, 7))):
        block[2 * slot] = out[pair[0]]
        block[2 * slot + 1] = out[pair[1]]


def _dit_merge4(block: list[int], span: int, bits: dict[tuple[int, int], int]) -> None:
    quarter = span >> 2
    half = span >> 1
    q0 = block[:quarter]
    q1 = block[quarter : 2 * quarter]
    q2 = block[2 * quarter : 3 * quarter]
    q3 = block[3 * quarter :]

    h0dc = q0[0] | q1[0]
    h0ny = q0[0] | q1[0]
    h1dc = q2[0] | q3[0]
    h1ny = q2[0] | q3[0]
    special = _union4(q0[1], q1[1], q2[1], q3[1], bits[(span, quarter // 2)])
    block[0] = h0dc | h1dc
    block[1] = h0dc | h1dc
    block[2 * quarter] = h0ny
    block[2 * quarter + 1] = h1ny
    block[quarter] = special[0]
    block[quarter + 1] = special[1]
    block[2 * (half - quarter // 2)] = special[2]
    block[2 * (half - quarter // 2) + 1] = special[3]

    visited: list[int] = []
    for i in range(1, quarter // 2):
        mirror = quarter // 2 - i
        if i > mirror:
            break
        visited.append(i)
        if mirror != i:
            visited.append(mirror)
    for i in visited:
        out = _dit_butterfly(
            q0, q1, q2, q3, i,
            bits[(span // 2, i)], bits[(span, i)], bits[(span, quarter - i)],
        )
        _dit_write(block, half, quarter, i, out)


def trace_dit(n: int) -> Trace:
    """Trace the logical cells of DIT compute_vwork_norm + terminal merge."""
    _check_size(n)
    stages = [1 << p for p in range(3, n.bit_length())]
    groups = [
        (f"s={span}", [(k, 2 * np.pi * k / span, f"span={span}, k={k}") for k in range(1, span // 4)])
        for span in stages
    ]
    factors, factor_index, group_labels, edges = _factor_layout(groups)
    bits = {
        (span, k): 1 << factor_index[(group, k)]
        for group, span in enumerate(stages)
        for k in range(1, span // 4)
    }

    # seed_merge2_pair contains only binomial operations, hence zero ancestry.
    state = [0] * n
    for span in (1 << p for p in range(4, n.bit_length()) if p % 2 == 0):
        for offset in range(0, n, span):
            block = state[offset : offset + span]
            _dit_merge4(block, span, bits)
            state[offset : offset + span] = block

    half = n // 2
    if (n.bit_length() - 1) % 2 == 1:
        quarter = half // 2
        dc_ny = state[0] | state[2 * quarter]
        pairs: list[tuple[int, int] | None] = [None] * (half - 1)
        pairs[quarter - 1] = (state[1], state[2 * quarter + 1])
        for i in range(1, quarter):
            merged = (
                state[2 * i]
                | state[2 * i + 1]
                | state[2 * (i + quarter)]
                | state[2 * (i + quarter) + 1]
                | bits[(n, i)]
            )
            pairs[i - 1] = (merged, merged)
            pairs[half - i - 1] = (merged, merged)
        assert all(pair is not None for pair in pairs)
        packed = _packed_masks(dc_ny, dc_ny, pairs)  # type: ignore[arg-type]
    else:
        packed = _packed_masks(state[0], state[1], [(state[2*k], state[2*k+1]) for k in range(1, half)])
    return Trace("DIT", _masks_to_support(packed, len(factors)), factors, group_labels, edges)


def trace_dip(n: int) -> Trace:
    """Trace the diagonal packet cells used by DIP fwd_ridge/fwd_tree."""
    _check_size(n)
    spans = [1 << p for p in range(2, n.bit_length() - 1)]
    groups = [
        (f"e={span}", [(d, np.pi * d / span, f"packet e={span}, d={d}") for d in range(1, span // 2)])
        for span in spans
    ]
    factors, factor_index, group_labels, edges = _factor_layout(groups)

    q = n
    dc = [0] * n
    ny: list[int] | None = None
    a: dict[int, list[int]] = {}
    b: dict[int, list[int]] = {}
    e = 1
    while e < n:
        q2 = q // 2
        ndc = [dc[i] | dc[q2 + i] for i in range(q2)]
        nny = ndc.copy()
        na: dict[int, list[int]] = {}
        nb: dict[int, list[int]] = {}
        if e >= 2:
            assert ny is not None
            na[e // 2] = ny[:q2]
            nb[e // 2] = ny[q2:]
        if e >= 4:
            group = e.bit_length() - 3
            for d in range(1, e // 2):
                bit = 1 << factor_index[(group, d)]
                merged = [
                    a[d][i] | b[d][i] | a[d][q2 + i] | b[d][q2 + i] | bit
                    for i in range(q2)
                ]
                na[d] = merged.copy()
                nb[d] = merged.copy()
                na[e - d] = merged.copy()
                nb[e - d] = merged.copy()
        dc, ny, a, b, q, e = ndc, nny, na, nb, q2, 2 * e

    assert ny is not None
    pairs = [(a[k][0], b[k][0]) for k in range(1, n // 2)]
    packed = _packed_masks(dc[0], ny[0], pairs)
    return Trace("DIP", _masks_to_support(packed, len(factors)), factors, group_labels, edges)


def build_traces(n: int) -> tuple[Trace, Trace, Trace]:
    return trace_dif(n), trace_dit(n), trace_dip(n)


def _reordered(trace: Trace, order: Sequence[int], suffix: str) -> Trace:
    if sorted(order) != list(range(len(trace.factors))):
        raise AssertionError(f"{trace.form} execution order is not a factor permutation")
    return Trace(
        f"{trace.form} {suffix}",
        trace.support[:, order],
        tuple(trace.factors[i] for i in order),
        (),
        (0, len(order)),
    )


def _dif_execution_order(trace: Trace, n: int) -> list[int]:
    # Below the production fused-tail threshold the scalar walk is already in
    # logical level order.
    if n < 32:
        return list(range(len(trace.factors)))
    lookup = {(f.group + 1, f.position): i for i, f in enumerate(trace.factors)}
    order: list[int] = []

    def emit(level: int, node: int) -> None:
        order.append(lookup[(level, node)])

    def d3(level: int, node: int) -> None:
        emit(level, node)
        emit(level + 1, 2 * node)
        emit(level + 1, 2 * node + 1)
        for child in range(4 * node, 4 * node + 4):
            emit(level + 2, child)

    def segment(base: int, quarter: int) -> None:
        span = 4 * quarter
        level = int(math.log2(n // span))
        node = base // span
        if quarter >= 16:
            emit(level, node)
            emit(level + 1, 2 * node)
            emit(level + 1, 2 * node + 1)
            child_quarter = quarter // 4
            for child in range(4):
                segment(base + child * quarter, child_quarter)
        elif quarter == 8:
            emit(level, node)
            d3(level + 1, 2 * node)
            d3(level + 1, 2 * node + 1)
        else:
            d3(level, node)

    for half in (n >> shift for shift in range(1, n.bit_length()) if (n >> shift) >= 32):
        segment(half, half // 4)
    # residue_spine_tail_fwd: d3 at 16, then the two- and one-level spine cells.
    spine_level = int(math.log2(n // 16))
    d3(spine_level, 1)
    emit(spine_level + 1, 1)
    emit(spine_level + 2, 2)
    emit(spine_level + 2, 3)
    emit(spine_level + 2, 1)
    return order


def _dit_execution_order(trace: Trace, n: int) -> list[int]:
    lookup = {(1 << (f.group + 3), f.position): i for i, f in enumerate(trace.factors)}
    order: list[int] = []
    seen: set[int] = set()

    def first(span: int, position: int) -> None:
        idx = lookup[(span, position)]
        if idx not in seen:
            seen.add(idx)
            order.append(idx)

    # merge4_v_norm: special Nyquist cell, then paired i/M-i butterflies.
    for span in (1 << p for p in range(4, n.bit_length()) if p % 2 == 0):
        quarter = span // 4
        first(span, quarter // 2)
        visited: list[int] = []
        for i in range(1, quarter // 2):
            mirror = quarter // 2 - i
            if i > mirror:
                break
            visited.append(i)
            if mirror != i:
                visited.append(mirror)
        for i in visited:
            first(span // 2, i)
            first(span, i)
            first(span, quarter - i)
    if (n.bit_length() - 1) % 2 == 1:
        for i in range(1, n // 4):
            first(n, i)
    return order


def _dip_execution_order(trace: Trace, n: int) -> list[int]:
    lookup = {(1 << (f.group + 2), f.position): i for i, f in enumerate(trace.factors)}
    order = [lookup[(4, 1)]]  # forward_seed8's pi/4 rotation

    def emit(e: int, d: int) -> None:
        order.append(lookup[(e, d)])

    def span8(d: int, e: int) -> None:
        emit(e, d)
        emit(2 * e, d)
        emit(2 * e, e - d)
        emit(4 * e, d)
        emit(4 * e, 2 * e - d)
        emit(4 * e, e - d)
        emit(4 * e, e + d)

    def tree(width: int, d: int, e: int) -> None:
        if width == 1:
            return
        if width >= 32:
            emit(e, d)
            emit(2 * e, d)
            emit(2 * e, e - d)
            quarter = width // 4
            tree(quarter, d, 4 * e)
            tree(quarter, 2 * e - d, 4 * e)
            tree(quarter, e - d, 4 * e)
            tree(quarter, e + d, 4 * e)
        elif width == 16:
            emit(e, d)
            tree(8, d, 2 * e)
            tree(8, e - d, 2 * e)
        elif width == 8:
            span8(d, e)
        elif width == 2:
            emit(e, d)
        else:
            emit(e, d)
            tree(width // 2, d, 2 * e)
            tree(width // 2, e - d, 2 * e)

    def ridge(width: int, e: int) -> None:
        if width == 1:
            return
        ridge(width // 2, 2 * e)
        tree(width // 2, e // 2, 2 * e)

    q = n // 8
    ridge(q, 8)
    tree(q, 2, 8)
    tree(q, 1, 8)
    tree(q, 3, 8)
    return order


def execution_order_traces(traces: Sequence[Trace], n: int) -> tuple[Trace, Trace, Trace]:
    dif, dit, dip = traces
    return (
        _reordered(dif, _dif_execution_order(dif, n), "execution"),
        _reordered(dit, _dit_execution_order(dit, n), "execution"),
        _reordered(dip, _dip_execution_order(dip, n), "execution"),
    )


def direct_dft_phase(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.int64)[:, None]
    sample = np.arange(n, dtype=np.int64)[None, :]
    return np.angle(np.exp(-2j * np.pi * k * sample / n)) / np.pi


def packed_real_dft_matrix(n: int) -> np.ndarray:
    """BFFT's N-by-N real output matrix: DC, (Re,-Im) pairs, Nyquist."""
    sample = np.arange(n, dtype=np.float64)
    rows = [np.ones(n, dtype=np.float64)]
    for k in range(1, n // 2):
        angle = 2 * np.pi * k * sample / n
        rows.append(np.cos(angle))
        rows.append(np.sin(angle))
    rows.append(np.cos(np.pi * sample))
    return np.asarray(rows)


def direct_dft_lenses(n: int) -> tuple[np.ndarray, np.ndarray]:
    sample = np.arange(n, dtype=np.float64)[None, :]
    frequency = np.arange(n, dtype=np.float64)[:, None]
    angle = 2 * np.pi * frequency * sample / n
    return np.cos(angle), -np.sin(angle)


def _dif_prefixes(n: int) -> list[tuple[str, np.ndarray]]:
    logn = n.bit_length() - 1
    state = np.eye(n, dtype=np.float64)
    prefixes = [("input", state.copy())]
    for jj in range(logn - 1):
        span = n >> jj
        half = span >> 1
        quarter = span >> 2
        left = state[:half].copy()
        right = state[half:span].copy()
        state[:half] = left + right
        state[half:span] = left - right
        for node in range(1, 1 << jj):
            base = node * span
            a0 = state[base : base + quarter].copy()
            b0 = state[base + quarter : base + 2 * quarter].copy()
            a1 = state[base + 2 * quarter : base + 3 * quarter].copy()
            b1 = state[base + 3 * quarter : base + 4 * quarter].copy()
            theta = np.pi * bruun_index(node, logn) / n
            c, s = np.cos(theta), np.sin(theta)
            r = c * b0 - s * b1
            imag = s * b0 + c * b1
            state[base : base + quarter] = a0 + r
            state[base + quarter : base + 2 * quarter] = a1 + imag
            state[base + 2 * quarter : base + 3 * quarter] = a0 - r
            state[base + 3 * quarter : base + 4 * quarter] = imag - a1
        prefixes.append((f"after DIF level {jj + 1}", state.copy()))
    prefixes.append(("standard output", packed_real_dft_matrix(n)))
    return prefixes


def _pair_reduce_rows(
    ea: np.ndarray, eb: np.ndarray, oa: np.ndarray, ob: np.ndarray, theta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c, s = np.cos(theta), np.sin(theta)
    r = c * oa - s * ob
    imag = s * oa + c * ob
    return ea + r, eb + imag, ea - r, imag - eb


def _dit_butterfly_rows(
    q0: np.ndarray, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, i: int, span: int
) -> tuple[np.ndarray, ...]:
    h0 = _pair_reduce_rows(
        q0[2*i], q0[2*i + 1], q1[2*i], q1[2*i + 1], 2*np.pi*i/(span//2)
    )
    h1 = _pair_reduce_rows(
        q2[2*i], q2[2*i + 1], q3[2*i], q3[2*i + 1], 2*np.pi*i/(span//2)
    )
    low = _pair_reduce_rows(*h0[:2], *h1[:2], 2*np.pi*i/span)
    high = _pair_reduce_rows(*h0[2:], *h1[2:], 2*np.pi*(span//4-i)/span)
    return low + high


def _dit_merge4_rows(block: np.ndarray, span: int) -> None:
    quarter = span // 4
    half = span // 2
    q0 = block[:quarter].copy()
    q1 = block[quarter : 2*quarter].copy()
    q2 = block[2*quarter : 3*quarter].copy()
    q3 = block[3*quarter :].copy()
    h0dc, h0ny = q0[0] + q1[0], q0[0] - q1[0]
    h1dc, h1ny = q2[0] + q3[0], q2[0] - q3[0]
    special = _pair_reduce_rows(
        q0[1], q1[1], q2[1], q3[1], 2*np.pi*(quarter//2)/span
    )
    block[0], block[1] = h0dc + h1dc, h0dc - h1dc
    block[2*quarter], block[2*quarter + 1] = h0ny, h1ny
    block[quarter], block[quarter + 1] = special[0], special[1]
    block[2*(half-quarter//2)], block[2*(half-quarter//2) + 1] = special[2], special[3]
    visited: list[int] = []
    for i in range(1, quarter // 2):
        mirror = quarter // 2 - i
        if i > mirror:
            break
        visited.append(i)
        if mirror != i:
            visited.append(mirror)
    for i in visited:
        out = _dit_butterfly_rows(q0, q1, q2, q3, i, span)
        for slot, pair in zip(
            (i, half-i, quarter-i, quarter+i), ((0, 1), (2, 3), (4, 5), (6, 7))
        ):
            block[2*slot], block[2*slot + 1] = out[pair[0]], out[pair[1]]


def _dit_prefixes(n: int) -> list[tuple[str, np.ndarray]]:
    if n != 512:
        raise ValueError("the production-order DIT cascade is currently defined for N=512")
    half = n // 2
    rev_bits = (n.bit_length() - 1) - 1
    basis = np.eye(n, dtype=np.float64)
    state = np.empty_like(basis)
    for b in range(0, half, 2):
        j0, j1 = _bitrev(b, rev_bits), _bitrev(b + 1, rev_bits)
        a0, c0 = basis[j0], basis[half + j0]
        a1, c1 = basis[j1], basis[half + j1]
        e0, e1 = a0 + c0, a1 + c1
        state[2*b], state[2*b + 1] = e0 + e1, e0 - e1
        state[2*b + 2], state[2*b + 3] = a0 - c0, a1 - c1
    prefixes = [("input", basis), ("after DIT seed (2 levels)", state.copy())]
    for span in (16, 64, 256):
        for offset in range(0, n, span):
            _dit_merge4_rows(state[offset : offset + span], span)
        prefixes.append((f"after DIT fused span {span}", state.copy()))
    # Verify the source-matched odd-power terminal before using the canonical F.
    out = np.empty_like(state)
    quarter = half // 2
    out[0], out[-1] = state[0] + state[2*quarter], state[0] - state[2*quarter]
    out[2*quarter - 1], out[2*quarter] = state[1], state[2*quarter + 1]
    for i in range(1, quarter):
        low = _pair_reduce_rows(
            state[2*i], state[2*i + 1],
            state[2*(i + quarter)], state[2*(i + quarter) + 1],
            2*np.pi*i/n,
        )
        lo_row, hi_row = 2*i - 1, 2*(half-i) - 1
        out[lo_row], out[lo_row + 1] = low[0], low[1]
        out[hi_row], out[hi_row + 1] = low[2], low[3]
    expected = packed_real_dft_matrix(n)
    if not np.allclose(out, expected, atol=2e-12):
        raise AssertionError(f"DIT cascade terminal mismatch: {np.max(np.abs(out-expected)):.3e}")
    prefixes.append(("standard output", expected))
    return prefixes


def _flatten_dip_state(
    dc: np.ndarray, ny: np.ndarray | None,
    a: dict[int, np.ndarray], b: dict[int, np.ndarray], e: int,
) -> np.ndarray:
    if e == 1:
        return dc.copy()
    assert ny is not None
    rows = [dc, ny]
    for d in range(1, e // 2):
        rows.extend((a[d], b[d]))
    return np.concatenate(rows, axis=0)


def _dip_prefixes(n: int) -> list[tuple[str, np.ndarray]]:
    q, e = n, 1
    dc = np.eye(n, dtype=np.float64)
    ny: np.ndarray | None = None
    a: dict[int, np.ndarray] = {}
    b: dict[int, np.ndarray] = {}
    prefixes = [("input", dc.copy())]
    while e < n:
        q2 = q // 2
        ndc, nny = dc[:q2] + dc[q2:q], dc[:q2] - dc[q2:q]
        na: dict[int, np.ndarray] = {}
        nb: dict[int, np.ndarray] = {}
        if e >= 2:
            assert ny is not None
            na[e//2], nb[e//2] = ny[:q2].copy(), ny[q2:q].copy()
        for d in range(1, e // 2):
            result = _pair_reduce_rows(
                a[d][:q2], b[d][:q2], a[d][q2:q], b[d][q2:q], np.pi*d/e
            )
            na[d], nb[d], na[e-d], nb[e-d] = result
        dc, ny, a, b, q, e = ndc, nny, na, nb, q2, 2*e
        prefixes.append((f"after DIP packet level e={e}", _flatten_dip_state(dc, ny, a, b, e)))
    expected = packed_real_dft_matrix(n)
    packed = np.vstack([dc[0], *[row for k in range(1, n//2) for row in (a[k][0], b[k][0])], ny[0]])
    if not np.allclose(packed, expected, atol=2e-12):
        raise AssertionError(f"DIP cascade terminal mismatch: {np.max(np.abs(packed-expected)):.3e}")
    prefixes.append(("standard output", expected))
    return prefixes


def reverse_factor_cascade(
    prefixes: Sequence[tuple[str, np.ndarray]], n: int
) -> list[tuple[str, np.ndarray]]:
    """Map every intermediate coordinate system forward to fixed output rows."""
    final = packed_real_dft_matrix(n)
    cascade = []
    for label, prefix in reversed(prefixes):
        norms = np.einsum("ij,ij->i", prefix, prefix)
        gram_offdiag = prefix @ prefix.T - np.diag(norms)
        if np.max(np.abs(gram_offdiag)) > 2e-9:
            raise AssertionError(f"{label} is not an orthogonal-row Bruun frame")
        suffix = (final @ prefix.T) / norms[None, :]
        if not np.allclose(suffix @ prefix, final, atol=3e-11):
            raise AssertionError(f"reverse factorization failed at {label}")
        cascade.append((label, suffix))
    return cascade


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg")


def plot_traces(traces: Sequence[Trace], n: int, path: Path) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True, constrained_layout=True)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("white")
    norm = Normalize(0.0, 0.5)
    image = None
    for ax, trace in zip(axes, traces):
        values = np.broadcast_to(trace.angles_over_pi, trace.support.shape)
        image = ax.imshow(
            np.ma.masked_where(~trace.support, values),
            origin="upper", aspect="auto", interpolation="nearest", cmap=cmap, norm=norm,
        )
        if trace.group_labels:
            for edge in trace.group_edges[1:-1]:
                ax.axvline(edge - 0.5, color="black", linewidth=0.35, alpha=0.45)
            centers = [(a + b - 1) / 2 for a, b in zip(trace.group_edges[:-1], trace.group_edges[1:])]
            ax.set_xticks(centers, trace.group_labels, rotation=55, ha="right", fontsize=8)
        ax.set_title(f"{trace.form}: {len(trace.factors)} normalized Bruun rotations")
        ax.set_xlabel("twiddle operation")
    axes[0].set_ylabel("packed real spectrum coordinate")
    ticks = [0, 127.5, 255.5, 383.5, 511] if n == 512 else [0, n / 4, n / 2, 3*n / 4, n - 1]
    labels = ["DC", "k=64 pair", "k=128 pair", "k=192 pair", "Nyquist"] if n == 512 else ["DC", "1/4", "1/2", "3/4", "Nyquist"]
    axes[0].set_yticks(ticks, labels)
    fig.suptitle(
        f"BFFT N={n} twiddle ancestry — rows are output coordinates; columns are contributing rotations",
        fontsize=15,
    )
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, location="bottom", shrink=0.62, pad=0.08, aspect=45)
    cbar.set_label(r"Givens angle $\theta/\pi$ (white = no structural dependency)")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_direct_lenses(n: int, path: Path) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    real, imag = direct_dft_lenses(n)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True)
    norm = Normalize(-1, 1)
    titles = (r"real lens: $\cos(2\pi kn/N)$", r"imaginary lens: $-\sin(2\pi kn/N)$")
    image = None
    for ax, matrix, title in zip(axes, (real, imag), titles):
        image = ax.imshow(matrix, origin="upper", interpolation="nearest", cmap="viridis", norm=norm)
        ax.set_title(title)
        ax.set_xlabel("input sample n")
    axes[0].set_ylabel("output bin k")
    fig.suptitle(f"Traditional complex DFT Fresnel lenses, N={n}", fontsize=15)
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, shrink=0.74, pad=0.03)
    cbar.set_label("DFT coefficient")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_cascade(form: str, cascade: Sequence[tuple[str, np.ndarray]], n: int, path: Path) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    columns = 3
    rows = math.ceil(len(cascade) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(12.5, 4.0 * rows), squeeze=False, constrained_layout=True
    )
    norm = Normalize(-1, 1)
    image = None
    for ax, (label, matrix) in zip(axes.flat, cascade):
        image = ax.imshow(matrix, origin="upper", interpolation="nearest", cmap="viridis", norm=norm)
        ax.set_title(label, fontsize=10)
        ax.set_xticks((0, n//2, n-1), ("0", f"{n//2}", f"{n-1}"), fontsize=8)
        ax.set_yticks((0, n//2, n-1), ("0", f"{n//2}", f"{n-1}"), fontsize=8)
        ax.set_xlabel("intermediate factor coordinate", fontsize=9)
        ax.set_ylabel("final packed output coordinate", fontsize=9)
    for ax in axes.flat[len(cascade):]:
        ax.set_visible(False)
    fig.suptitle(
        f"BFFT {form} reverse-factor cascade, N={n} — output boundary back to the full transform",
        fontsize=15,
    )
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, location="bottom", shrink=0.66, pad=0.025, aspect=45)
    cbar.set_label("signed coefficient (viridis)")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_data(
    traces: Sequence[Trace], execution_traces: Sequence[Trace], n: int, output_dir: Path
) -> None:
    arrays: dict[str, np.ndarray] = {"n": np.asarray(n)}
    summary: dict[str, object] = {"n": n, "packed_rows": n, "forms": {}}
    for trace, executed in zip(traces, execution_traces):
        key = trace.form.lower()
        arrays[f"{key}_support"] = trace.support
        arrays[f"{key}_angle_over_pi"] = trace.angles_over_pi
        arrays[f"{key}_factor_labels"] = np.asarray([f.label for f in trace.factors])
        arrays[f"{key}_group_edges"] = np.asarray(trace.group_edges)
        logical_index = {factor: i for i, factor in enumerate(trace.factors)}
        arrays[f"{key}_execution_order"] = np.asarray(
            [logical_index[factor] for factor in executed.factors], dtype=np.int64
        )
        row_counts = trace.support.sum(axis=1)
        summary["forms"][trace.form] = {  # type: ignore[index]
            "factors": len(trace.factors),
            "group_labels": list(trace.group_labels),
            "group_edges": list(trace.group_edges),
            "nonzero_dependencies": int(trace.support.sum()),
            "dependencies_per_nontrivial_row": {
                "min": int(row_counts[1:-1].min()),
                "max": int(row_counts[1:-1].max()),
            },
        }
    np.savez_compressed(output_dir / f"bfft_twiddle_trace_n{n}.npz", **arrays)
    (output_dir / f"bfft_twiddle_trace_n{n}.json").write_text(json.dumps(summary, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "bfft_twiddle_trace",
    )
    parser.add_argument("--formats", nargs="+", choices=("png", "svg"), default=("png", "svg"))
    parser.add_argument("--skip-direct-lens", action="store_true")
    args = parser.parse_args(argv)
    _check_size(args.n)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = build_traces(args.n)
    execution_traces = execution_order_traces(traces, args.n)
    write_data(traces, execution_traces, args.n, args.output_dir)
    cascades: dict[str, list[tuple[str, np.ndarray]]] = {}
    if args.n == 512:
        cascades = {
            "DIF": reverse_factor_cascade(_dif_prefixes(args.n), args.n),
            "DIT": reverse_factor_cascade(_dit_prefixes(args.n), args.n),
            "DIP": reverse_factor_cascade(_dip_prefixes(args.n), args.n),
        }
    for extension in args.formats:
        plot_traces(traces, args.n, args.output_dir / f"bfft_twiddle_trace_n{args.n}.{extension}")
        plot_traces(
            execution_traces, args.n,
            args.output_dir / f"bfft_twiddle_execution_n{args.n}.{extension}",
        )
        if not args.skip_direct_lens:
            plot_direct_lenses(args.n, args.output_dir / f"bfft_dft_lenses_n{args.n}.{extension}")
        for form, cascade in cascades.items():
            plot_cascade(
                form, cascade, args.n,
                args.output_dir / f"bfft_{form.lower()}_cascade_n{args.n}.{extension}",
            )
    for trace in traces:
        counts = trace.support.sum(axis=1)
        print(
            f"{trace.form}: rows={trace.support.shape[0]} factors={trace.support.shape[1]} "
            f"dependencies={int(trace.support.sum())} interior-row-range={int(counts[1:-1].min())}..{int(counts[1:-1].max())}"
        )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
