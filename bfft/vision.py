"""Exact kernels for BFFT-guided partition-of-unity image models.

The routines in this module deliberately operate on one *measured*
owner/runner assignment.  They do not enumerate candidate diagrams.  The
resulting block matrix is the renderer's own interaction graph: a block
``(i, j)`` exists exactly when cells ``i`` and ``j`` jointly explain a pixel.

The native extension supplies the hot accumulation and rendering loops when
available.  NumPy/Numba implementations are retained as a portable reference.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse

from ._core import (_check, _vision_assemble_normal, _vision_render_affine,
                    _vision_scan_residual_ridges, _vision_support_forward,
                    _vision_support_normal_apply, _vision_support_transpose)

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _identity(fn):  # pragma: no cover
    return fn


_compile = njit(cache=True, fastmath=False) if njit is not None else _identity


def _ptr(array, ctype):
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def vision_backend():
    """Human-readable hot-kernel backend used by this installation."""
    native = (
        _vision_assemble_normal is not None and
        _vision_render_affine is not None and
        _vision_scan_residual_ridges is not None)
    return "native C++" if native else (
        "Numba" if njit is not None else "portable Python")


def compact_support_operators(rows, sites, weight, basis_x, basis_y,
                              pixels, cells):
    """Return native compact-support ``A``, ``A.T`` and ``A.T A`` calls."""
    if (_vision_support_forward is None or
            _vision_support_transpose is None or
            _vision_support_normal_apply is None):
        return None
    rows = np.ascontiguousarray(rows, dtype=np.int32)
    sites = np.ascontiguousarray(sites, dtype=np.int32)
    weight = np.ascontiguousarray(weight, dtype=np.float64)
    basis_x = np.ascontiguousarray(basis_x, dtype=np.float64)
    basis_y = np.ascontiguousarray(basis_y, dtype=np.float64)
    sample_count = rows.size
    pixels = int(pixels)
    cells = int(cells)
    scratch = np.empty(pixels, dtype=np.float64)

    def forward(coefficient):
        coefficient = np.ascontiguousarray(coefficient, dtype=np.float64)
        output = np.empty(pixels, dtype=np.float64)
        _check(_vision_support_forward(
            sample_count, pixels, cells,
            _ptr(rows, ctypes.c_int32), _ptr(sites, ctypes.c_int32),
            _ptr(weight, ctypes.c_double), _ptr(basis_x, ctypes.c_double),
            _ptr(basis_y, ctypes.c_double),
            _ptr(coefficient, ctypes.c_double),
            _ptr(output, ctypes.c_double)),
            "bfft_vision_support_forward")
        return output

    def transpose(pixel):
        pixel = np.ascontiguousarray(pixel, dtype=np.float64)
        output = np.empty(3 * cells, dtype=np.float64)
        _check(_vision_support_transpose(
            sample_count, pixels, cells,
            _ptr(rows, ctypes.c_int32), _ptr(sites, ctypes.c_int32),
            _ptr(weight, ctypes.c_double), _ptr(basis_x, ctypes.c_double),
            _ptr(basis_y, ctypes.c_double), _ptr(pixel, ctypes.c_double),
            _ptr(output, ctypes.c_double)),
            "bfft_vision_support_transpose")
        return output

    def normal(coefficient):
        coefficient = np.ascontiguousarray(coefficient, dtype=np.float64)
        output = np.empty(3 * cells, dtype=np.float64)
        _check(_vision_support_normal_apply(
            sample_count, pixels, cells,
            _ptr(rows, ctypes.c_int32), _ptr(sites, ctypes.c_int32),
            _ptr(weight, ctypes.c_double), _ptr(basis_x, ctypes.c_double),
            _ptr(basis_y, ctypes.c_double),
            _ptr(coefficient, ctypes.c_double),
            _ptr(scratch, ctypes.c_double), _ptr(output, ctypes.c_double)),
            "bfft_vision_support_normal_apply")
        return output

    return forward, transpose, normal


@dataclass(frozen=True)
class CoownershipGraph:
    """Fixed block-CSR topology for one actual owner/runner assignment."""

    cells: int
    width: int
    block_row: np.ndarray
    block_col: np.ndarray
    position: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    diag_of: np.ndarray
    slot_forward: np.ndarray
    slot_reverse: np.ndarray

    @property
    def block_count(self) -> int:
        return int(self.block_row.size)

    @property
    def edge_count(self) -> int:
        return (self.block_count - self.cells) // 2


class SingleStageDecompositionObjective:
    """RGB + one-stage cartoon + one-stage texture reconstruction objective.

    The target split is computed exactly once.  Candidate reconstructions are
    decomposed on demand, which removes the invariant target decomposition
    from line searches without relying on an identity/checksum cache.
    """

    def __init__(self, target_rgb, *, lam=0.05, mu=40.0, passes=24,
                 threads=4, space="oklab_lc", solver=1):
        from .effects import meyer_channels

        self.target_rgb = np.ascontiguousarray(
            np.clip(target_rgb, 0.0, 1.0), dtype=np.float64)
        self.lam = float(lam)
        self.mu = float(mu)
        self.passes = int(passes)
        self.threads = int(threads)
        self.space = str(space)
        self.solver = int(solver)
        split = meyer_channels(
            self.target_rgb, space=self.space, lam=self.lam, mu=self.mu,
            passes=self.passes, threads=self.threads, solver=self.solver)
        scale = np.maximum(split.scale[None, None, :], 1e-12)
        self.target_cartoon = split.cartoon / scale
        self.target_texture = split.texture / scale
        self.last_residual_energy = None

    def evaluate(self, reconstruction_rgb):
        """Return the three MSE terms and their equally weighted sum."""
        from .effects import meyer_channels

        reconstruction = np.ascontiguousarray(
            np.clip(reconstruction_rgb, 0.0, 1.0), dtype=np.float64)
        if reconstruction.shape != self.target_rgb.shape:
            raise ValueError("reconstruction shape differs from target")
        split = meyer_channels(
            reconstruction, space=self.space, lam=self.lam, mu=self.mu,
            passes=self.passes, threads=self.threads, solver=self.solver)
        scale = np.maximum(split.scale[None, None, :], 1e-12)
        cartoon = split.cartoon / scale
        texture = split.texture / scale
        rgb_mse = float(np.mean((self.target_rgb - reconstruction) ** 2))
        cartoon_mse = float(np.mean(
            (self.target_cartoon - cartoon) ** 2))
        texture_mse = float(np.mean(
            (self.target_texture - texture) ** 2))
        self.last_residual_energy = (
            np.mean((self.target_rgb - reconstruction) ** 2, axis=2)
            + np.mean((self.target_cartoon - cartoon) ** 2, axis=2)
            + np.mean((self.target_texture - texture) ** 2, axis=2)
        )
        return {
            "rgb_mse": rgb_mse,
            "psnr": -10.0 * math.log10(max(rgb_mse, 1e-12)),
            "cartoon_mse": cartoon_mse,
            "texture_mse": texture_mse,
            "objective": rgb_mse + cartoon_mse + texture_mse,
        }


@_compile
def _fill_scalar_indices(block_col, row_offsets, indptr, indices,
                         cells, width):
    for cell in range(cells):
        start = row_offsets[cell]
        stop = row_offsets[cell + 1]
        for sub in range(width):
            cursor = indptr[width * cell + sub]
            for k in range(start, stop):
                column = width * block_col[k]
                for part in range(width):
                    indices[cursor + part] = column + part
                cursor += width


def coownership_graph(owner, other, valid, cells, width=3):
    """Construct the exact interaction topology of the current rendering.

    The topology may be shared between cartoon and texture fits only when
    they use this same current owner/runner assignment.  It is intentionally
    not cached across geometry updates or hypothetical candidate placements.
    """
    owner = np.ascontiguousarray(owner, dtype=np.int64)
    other = np.ascontiguousarray(other, dtype=np.int64)
    valid = np.ascontiguousarray(valid, dtype=np.bool_)
    cells = int(cells)
    width = int(width)
    visible = np.flatnonzero(valid)
    i = owner[visible]
    j = other[visible]
    low = np.minimum(i, j)
    high = np.maximum(i, j)
    keys = np.unique(low * cells + high)
    pair_a = keys // cells
    pair_b = keys % cells

    diagonal = np.arange(cells, dtype=np.int64)
    block_row = np.concatenate((diagonal, pair_a, pair_b))
    block_col = np.concatenate((diagonal, pair_b, pair_a))
    order = np.lexsort((block_col, block_row))
    block_row = np.ascontiguousarray(block_row[order])
    block_col = np.ascontiguousarray(block_col[order])
    relocated = np.empty(order.size, dtype=np.int64)
    relocated[order] = np.arange(order.size)
    edges = keys.size
    diag_of = np.ascontiguousarray(relocated[:cells])
    forward_of = relocated[cells:cells + edges]
    reverse_of = relocated[cells + edges:]

    counts = np.bincount(block_row, minlength=cells)
    row_offsets = np.zeros(cells + 1, dtype=np.int64)
    np.cumsum(counts, out=row_offsets[1:])
    position = np.ascontiguousarray(
        np.arange(block_row.size) - row_offsets[block_row])

    indptr = np.zeros(width * cells + 1, dtype=np.int64)
    np.cumsum(np.repeat(width * counts, width), out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    _fill_scalar_indices(
        block_col, row_offsets, indptr, indices, cells, width)

    slot_forward = np.full(owner.size, -1, dtype=np.int64)
    slot_reverse = np.full(owner.size, -1, dtype=np.int64)
    if visible.size:
        found = np.searchsorted(keys, low * cells + high)
        owner_is_low = i == low
        slot_forward[visible] = np.where(
            owner_is_low, forward_of[found], reverse_of[found])
        slot_reverse[visible] = np.where(
            owner_is_low, reverse_of[found], forward_of[found])
    return CoownershipGraph(
        cells, width, block_row, block_col, position, indptr, indices,
        diag_of, slot_forward, slot_reverse)


@_compile
def _accumulate_reference(owner, other, valid, w1, w2, first, second,
                          target, diag_of, slot_forward, slot_reverse,
                          blocks, rhs):
    width = first.shape[1]
    channels = target.shape[1]
    u = np.empty(width, dtype=np.float64)
    v = np.empty(width, dtype=np.float64)
    for p in range(owner.size):
        i = owner[p]
        if i < 0:
            continue
        for a in range(width):
            u[a] = w1[p] * first[p, a]
        diagonal = diag_of[i]
        for a in range(width):
            for b in range(width):
                blocks[diagonal, a, b] += u[a] * u[b]
            for channel in range(channels):
                rhs[i, a, channel] += u[a] * target[p, channel]
        if not valid[p]:
            continue
        j = other[p]
        for a in range(width):
            v[a] = w2[p] * second[p, a]
        diagonal = diag_of[j]
        for a in range(width):
            for b in range(width):
                blocks[diagonal, a, b] += v[a] * v[b]
            for channel in range(channels):
                rhs[j, a, channel] += v[a] * target[p, channel]
        forward = slot_forward[p]
        reverse = slot_reverse[p]
        for a in range(width):
            for b in range(width):
                blocks[forward, a, b] += u[a] * v[b]
                blocks[reverse, a, b] += v[a] * u[b]


@_compile
def _blocks_to_data(blocks, block_row, position, indptr, data, width):
    for k in range(blocks.shape[0]):
        cell = block_row[k]
        base = width * position[k]
        for a in range(width):
            cursor = indptr[width * cell + a] + base
            for b in range(width):
                data[cursor + b] = blocks[k, a, b]


def assemble_normal(owner, other, valid, w1, w2, first, second, target,
                    graph, regularization=None):
    """Fused exact normal assembly without materializing a design matrix."""
    w1 = np.ascontiguousarray(w1, dtype=np.float64)
    w2 = np.ascontiguousarray(w2, dtype=np.float64)
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    target = np.ascontiguousarray(target, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("target must have shape (pixels, channels)")
    if first.shape != second.shape:
        raise ValueError("first and second basis arrays must have equal shape")
    if first.shape[1] != graph.width:
        raise ValueError("basis width does not match graph width")

    blocks = np.zeros(
        (graph.block_count, graph.width, graph.width), dtype=np.float64)
    rhs = np.zeros(
        (graph.cells, graph.width, target.shape[1]), dtype=np.float64)
    if _vision_assemble_normal is not None:
        owner32 = np.ascontiguousarray(owner, dtype=np.int32)
        other32 = np.ascontiguousarray(other, dtype=np.int32)
        valid8 = np.ascontiguousarray(valid, dtype=np.uint8)
        _check(_vision_assemble_normal(
            owner32.size, graph.cells, graph.width, graph.block_count,
            _ptr(owner32, ctypes.c_int32), _ptr(other32, ctypes.c_int32),
            _ptr(valid8, ctypes.c_uint8), _ptr(w1, ctypes.c_double),
            _ptr(w2, ctypes.c_double), _ptr(first, ctypes.c_double),
            _ptr(second, ctypes.c_double), _ptr(target, ctypes.c_double),
            _ptr(graph.diag_of, ctypes.c_int64),
            _ptr(graph.slot_forward, ctypes.c_int64),
            _ptr(graph.slot_reverse, ctypes.c_int64),
            _ptr(blocks, ctypes.c_double), _ptr(rhs, ctypes.c_double)),
            "bfft_vision_assemble_normal")
    else:
        owner64 = np.ascontiguousarray(owner, dtype=np.int64)
        other64 = np.ascontiguousarray(other, dtype=np.int64)
        valid_bool = np.ascontiguousarray(valid, dtype=np.bool_)
        _accumulate_reference(
            owner64, other64, valid_bool, w1, w2, first, second, target,
            graph.diag_of, graph.slot_forward, graph.slot_reverse, blocks, rhs)
    if regularization is not None:
        reg = np.asarray(regularization, dtype=np.float64)
        if reg.shape == (graph.width,):
            reg = np.tile(reg, graph.cells)
        if reg.shape != (graph.cells * graph.width,):
            raise ValueError("regularization must have width or cells*width values")
        for part in range(graph.width):
            blocks[graph.diag_of, part, part] += reg[part::graph.width]

    data = np.empty(graph.indices.size, dtype=np.float64)
    _blocks_to_data(
        blocks, graph.block_row, graph.position, graph.indptr, data,
        graph.width)
    matrix = sparse.csr_matrix(
        (data, graph.indices, graph.indptr),
        shape=(graph.cells * graph.width, graph.cells * graph.width))
    return matrix.tocsc(), rhs.reshape(
        graph.cells * graph.width, target.shape[1]), blocks


@_compile
def _render_reference(coeff, owner, other, valid, w1, w2, first, second,
                      pred_first, pred_second, field):
    width = coeff.shape[1]
    channels = coeff.shape[2]
    for p in range(owner.size):
        i = owner[p]
        j = other[p] if valid[p] else i
        for channel in range(channels):
            left = 0.0
            right = 0.0
            for a in range(width):
                left += coeff[i, a, channel] * first[p, a]
                right += coeff[j, a, channel] * second[p, a]
            pred_first[p, channel] = left
            pred_second[p, channel] = right
            field[p, channel] = w1[p] * left + w2[p] * right


def render_partition(coeff, owner, other, valid, w1, w2, first, second):
    """Render both cell predictions and their partition-of-unity blend."""
    coeff = np.ascontiguousarray(coeff, dtype=np.float64)
    w1 = np.ascontiguousarray(w1, dtype=np.float64)
    w2 = np.ascontiguousarray(w2, dtype=np.float64)
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    pixel_count = np.asarray(owner).size
    shape = (pixel_count, coeff.shape[2])
    pred_first = np.empty(shape, dtype=np.float64)
    pred_second = np.empty(shape, dtype=np.float64)
    field = np.empty(shape, dtype=np.float64)
    if _vision_render_affine is not None:
        owner32 = np.ascontiguousarray(owner, dtype=np.int32)
        other32 = np.ascontiguousarray(
            np.where(valid, other, owner), dtype=np.int32)
        _check(_vision_render_affine(
            pixel_count, coeff.shape[0], coeff.shape[1],
            _ptr(owner32, ctypes.c_int32), _ptr(other32, ctypes.c_int32),
            _ptr(w1, ctypes.c_double), _ptr(w2, ctypes.c_double),
            _ptr(first, ctypes.c_double), _ptr(second, ctypes.c_double),
            _ptr(coeff, ctypes.c_double), _ptr(pred_first, ctypes.c_double),
            _ptr(pred_second, ctypes.c_double), _ptr(field, ctypes.c_double)),
            "bfft_vision_render_affine")
    else:
        owner64 = np.ascontiguousarray(owner, dtype=np.int64)
        other64 = np.ascontiguousarray(other, dtype=np.int64)
        valid_bool = np.ascontiguousarray(valid, dtype=np.bool_)
        _render_reference(
            coeff, owner64, other64, valid_bool, w1, w2, first, second,
            pred_first, pred_second, field)
    return field, pred_first, pred_second


@_compile
def _ridge_scan_reference(owner, weight, residual, dx, dy, cosines, sines,
                          spacing, cells, angles, bins, span,
                          channel_weights):
    stride = angles * bins * residual.shape[1]
    channels = residual.shape[1]
    accumulator = np.zeros(cells * stride, dtype=np.float64)
    mass = np.zeros(cells, dtype=np.float64)
    total = np.zeros((cells, channels), dtype=np.float64)
    scale = bins / (2.0 * span)
    for p in range(owner.size):
        cell = owner[p]
        if cell < 0:
            continue
        phi = weight[p]
        mass[cell] += phi
        for channel in range(channels):
            total[cell, channel] += phi * residual[p, channel]
        px = dx[p] / spacing
        py = dy[p] / spacing
        base = cell * stride
        for angle in range(angles):
            projection = px * cosines[angle] + py * sines[angle]
            index = int((projection + span) * scale)
            if index < 0:
                index = 0
            elif index >= bins:
                index = bins - 1
            slot = base + (angle * bins + index) * channels
            for channel in range(channels):
                accumulator[slot + channel] += (
                    phi * residual[p, channel])

    best_score = np.zeros(cells, dtype=np.float64)
    best_angle = np.zeros(cells, dtype=np.int64)
    best_bin = np.zeros(cells, dtype=np.int64)
    running = np.empty(channels, dtype=np.float64)
    for cell in range(cells):
        denominator = max(mass[cell], 1e-9)
        base = cell * stride
        seen = False
        for angle in range(angles):
            for channel in range(channels):
                running[channel] = 0.0
            for bin_index in range(bins):
                slot = base + (angle * bins + bin_index) * channels
                value = 0.0
                for channel in range(channels):
                    running[channel] += accumulator[slot + channel]
                    contrast = total[cell, channel] - 2.0 * running[channel]
                    value += channel_weights[channel] * contrast * contrast
                value /= denominator
                if not seen or value > best_score[cell]:
                    best_score[cell] = value
                    best_angle[cell] = angle
                    best_bin[cell] = bin_index
                    seen = True
    return best_score, best_angle, best_bin


def measure_residual_ridges(owner, weight, residual, dx, dy, spacing,
                            cells, angles=16, bins=41, span=2.5,
                            channel_weights=(1.0, 1.5, 1.5)):
    """Measure one bounded residual ridge per current cell in one image pass."""
    weight = np.ascontiguousarray(weight, dtype=np.float64)
    residual = np.ascontiguousarray(residual, dtype=np.float64)
    dx = np.ascontiguousarray(dx, dtype=np.float64)
    dy = np.ascontiguousarray(dy, dtype=np.float64)
    channel_weights = np.ascontiguousarray(
        channel_weights, dtype=np.float64)
    theta = np.linspace(0.0, np.pi, int(angles), endpoint=False)
    cosines = np.ascontiguousarray(np.cos(theta))
    sines = np.ascontiguousarray(np.sin(theta))
    if _vision_scan_residual_ridges is not None:
        owner32 = np.ascontiguousarray(owner, dtype=np.int32)
        score = np.empty(int(cells), dtype=np.float64)
        angle_index = np.empty(int(cells), dtype=np.int32)
        bin_index = np.empty(int(cells), dtype=np.int32)
        _check(_vision_scan_residual_ridges(
            owner32.size, int(cells), int(angles), int(bins),
            float(spacing), float(span), _ptr(owner32, ctypes.c_int32),
            _ptr(weight, ctypes.c_double), _ptr(residual, ctypes.c_double),
            _ptr(dx, ctypes.c_double), _ptr(dy, ctypes.c_double),
            _ptr(cosines, ctypes.c_double), _ptr(sines, ctypes.c_double),
            _ptr(channel_weights, ctypes.c_double),
            _ptr(score, ctypes.c_double), _ptr(angle_index, ctypes.c_int32),
            _ptr(bin_index, ctypes.c_int32)),
            "bfft_vision_scan_residual_ridges")
    else:
        owner64 = np.ascontiguousarray(owner, dtype=np.int64)
        score, angle_index, bin_index = _ridge_scan_reference(
            owner64, weight, residual, dx, dy, cosines, sines,
            float(spacing), int(cells), int(angles), int(bins), float(span),
            channel_weights)
    offset = ((bin_index + 0.5) / int(bins) * (2.0 * float(span))
              - float(span))
    return score, theta[angle_index], offset


@_compile
def _takahashi(indptr, indices, lower, diagonal_inverse,
                inverse_values, positions, accumulator):
    n = indptr.size - 1
    for j in range(n - 1, -1, -1):
        start = indptr[j]
        end = indptr[j + 1]
        base = start + 1
        count = end - base
        for a in range(count):
            accumulator[a] = 0.0
        for a in range(count):
            column = indices[base + a]
            la = lower[base + a]
            lo = indptr[column]
            hi = indptr[column + 1]
            for t in range(lo, hi):
                positions[indices[t]] = t
            for b in range(a, count):
                t = positions[indices[base + b]]
                if t < 0:
                    for u in range(lo, hi):
                        positions[indices[u]] = -1
                    return 1
                value = inverse_values[t]
                accumulator[a] += lower[base + b] * value
                if b != a:
                    accumulator[b] += la * value
            for t in range(lo, hi):
                positions[indices[t]] = -1
        value = diagonal_inverse[j]
        for a in range(count):
            inverse_values[base + a] = -accumulator[a]
            value -= lower[base + a] * inverse_values[base + a]
        inverse_values[start] = value
    return 0


@_compile
def _gather_diagonal_blocks(indptr, indices, inverse_values, permutation,
                            cells, width, out):
    for cell in range(cells):
        for a in range(width):
            original_a = permutation[width * cell + a]
            for b in range(a, width):
                original_b = permutation[width * cell + b]
                row = max(original_a, original_b)
                column = min(original_a, original_b)
                lo = indptr[column]
                hi = indptr[column + 1]
                found = -1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if indices[mid] == row:
                        found = mid
                        break
                    if indices[mid] < row:
                        lo = mid + 1
                    else:
                        hi = mid
                if found < 0:
                    return 1
                out[cell, a, b] = inverse_values[found]
                out[cell, b, a] = inverse_values[found]
    return 0


def selected_inverse_blocks(lu, cells, width, verify=True):
    """Return every exact diagonal cell block of ``G^-1`` without solves.

    SuperLU's symmetric factor assumptions are checked.  Callers can catch
    ``RuntimeError`` and fall back to batched solves on unusual pivoting.
    """
    lower = lu.L.tocsc(copy=True)
    lower.sort_indices()
    n = lower.shape[0]
    if not np.array_equal(lu.perm_r, lu.perm_c):
        raise RuntimeError("SuperLU used asymmetric permutations")
    if not np.all(lower.indices[lower.indptr[:-1]] == np.arange(n)):
        raise RuntimeError("SuperLU L is not diagonal-first")
    if verify:
        relation = (
            lu.U -
            sparse.diags(lu.U.diagonal(), format="csc") @ lower.T).tocsc()
        scale = max(float(abs(lu.U).max()), 1e-30)
        if relation.nnz and float(np.max(np.abs(relation.data))) > 1e-9 * scale:
            raise RuntimeError("SuperLU factors are not symmetric LDL^T")

    indptr = np.ascontiguousarray(lower.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(lower.indices, dtype=np.int64)
    values = np.ascontiguousarray(lower.data, dtype=np.float64)
    inverse_values = np.zeros_like(values)
    positions = np.full(n, -1, dtype=np.int64)
    accumulator = np.zeros(n, dtype=np.float64)
    status = _takahashi(
        indptr, indices, values, 1.0 / lu.U.diagonal(),
        inverse_values, positions, accumulator)
    if status:
        raise RuntimeError("selected inversion left the factor sparsity pattern")
    out = np.zeros((int(cells), int(width), int(width)), dtype=np.float64)
    permutation = np.ascontiguousarray(lu.perm_r, dtype=np.int64)
    if _gather_diagonal_blocks(
            indptr, indices, inverse_values, permutation,
            int(cells), int(width), out):
        raise RuntimeError("cell block was absent from the inverse subset")
    return out


def deletion_prices(lu, coefficients, channel_weights=(1.0, 1.5, 1.5)):
    """Exact regularized-objective price of deleting each current cell."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    cells, width, channels = coefficients.shape
    weights = np.asarray(channel_weights, dtype=np.float64)
    if weights.shape != (channels,):
        raise ValueError("one channel weight is required per coefficient channel")
    blocks = selected_inverse_blocks(lu, cells, width)
    schur = np.linalg.pinv(blocks)
    prices = np.einsum(
        "c,ikc,ikl,ilc->i", weights, coefficients, schur, coefficients)
    return prices, blocks
