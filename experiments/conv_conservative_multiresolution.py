"""Conservative multiresolution coordinates for CONV.

The point-value CONV interpolant and a raster scale state answer different
questions.  This module treats raster entries as cell averages.  A dyadic
parent is the exact average of its children; the complementary coordinate is
the centered child moment.  A compact CONV jet predicts that moment, and a
signed-current admission retains the largest proposal on each witnessed
variation lineage that can be carried without adding an alternating pair.

All analysis/synthesis identities below are algebraic.  The admission maps
only the zero-detail section and never enters the proof of perfect
reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.self_geometric_harmonic_interpolation import (
    _evaluate_quintic_variation_lineage_profile,
    _local_fourth_order_2jet,
    _quintic_variation_lineage_profile,
)


Array = np.ndarray


def _filled_sign(difference: Array) -> Array:
    """Fill zero secant signs without changing the order of nonzero signs."""

    delta = np.asarray(difference, dtype=np.float64)
    if delta.ndim != 1:
        raise ValueError("sign filling expects one scalar line")
    sign = np.sign(delta)
    if not np.any(sign):
        return sign
    for index in range(sign.size):
        if sign[index] != 0.0:
            continue
        left = next(
            (sign[j] for j in range(index - 1, -1, -1) if sign[j] != 0.0),
            0.0,
        )
        right = next(
            (sign[j] for j in range(index + 1, sign.size) if sign[j] != 0.0),
            0.0,
        )
        sign[index] = left if left != 0.0 else right
    first = int(np.flatnonzero(sign)[0])
    sign[:first] = sign[first]
    return sign


def _cell_lineage_sign(raw: Array, interval_sign: Array) -> Array:
    """Assign every parent-cell moment to one adjacent sign lineage.

    Away from a coarse turning point the assignment is forced.  At a turning
    point the raw transported moment chooses the adjacent lineage with the
    least sign disagreement.  This moves an existing extremum by at most one
    child cell but cannot create a second transition.
    """

    proposal = np.asarray(raw, dtype=np.float64)
    sign = np.asarray(interval_sign, dtype=np.float64)
    cells = proposal.size
    if sign.shape != (cells - 1,):
        raise ValueError("one interval sign is required between adjacent cells")
    if not np.any(sign):
        return np.zeros(cells, dtype=np.float64)
    result = np.empty(cells, dtype=np.float64)
    result[0] = sign[0]
    result[-1] = sign[-1]
    for index in range(1, cells - 1):
        left, right = sign[index - 1], sign[index]
        if left == right:
            result[index] = left
        elif proposal[index] * left > proposal[index] * right:
            result[index] = left
        elif proposal[index] * right > proposal[index] * left:
            result[index] = right
        else:
            # Reflection has no preferred side when the proposal is exactly
            # zero; either choice synthesizes the same zero moment.
            result[index] = left
    return result


def _run_labels(sign: Array) -> Array:
    """Return consecutive labels for a filled nonzero sign sequence."""

    value = np.asarray(sign, dtype=np.float64)
    label = np.zeros(value.size, dtype=np.int64)
    for index in range(1, value.size):
        label[index] = label[index - 1] + int(value[index] != value[index - 1])
    return label


def _admit_moment_scalar(coarse: Array, raw: Array) -> Array:
    """Admit a scalar child-moment proposal to the ordered-current polytope.

    For parent averages ``c`` and child moments ``q``, the fine currents are

        g[2i]   = 2 q[i],
        g[2i+1] = c[i+1] - c[i] - q[i] - q[i+1].

    A fixed lineage chamber is the convex polytope

        t[i] q[i] >= 0,
        s[i] (delta[i] - q[i] - q[i+1]) >= 0,

    where ``s`` is the filled coarse-current sign and ``t`` assigns a turning
    cell to one of its two adjacent lineages.  The implementation first takes
    the orthant projection of the raw jet, then intersects each independent
    sign-lineage ray with every cross-cell half-space.  The retained scale is
    the largest value in [0,1], so no free limiter parameter is introduced.
    """

    c = np.asarray(coarse, dtype=np.float64)
    proposal = np.asarray(raw, dtype=np.float64)
    if c.ndim != 1 or proposal.shape != c.shape or c.size < 2:
        raise ValueError("coarse and raw moment must be equal scalar lines")
    delta = np.diff(c)
    interval_sign = _filled_sign(delta)
    if not np.any(interval_sign):
        return np.zeros_like(c)
    cell_sign = _cell_lineage_sign(proposal, interval_sign)
    ray = cell_sign * np.maximum(cell_sign * proposal, 0.0)
    interval_run = _run_labels(interval_sign)
    run_count = int(interval_run[-1]) + 1
    scale = np.ones(run_count, dtype=np.float64)

    # A turning cell belongs to the lineage selected above.  Interior cells of
    # a run have the same label on both adjacent intervals.
    cell_run = np.empty(c.size, dtype=np.int64)
    cell_run[0] = interval_run[0]
    cell_run[-1] = interval_run[-1]
    for index in range(1, c.size - 1):
        if cell_sign[index] == interval_sign[index - 1]:
            cell_run[index] = interval_run[index - 1]
        else:
            cell_run[index] = interval_run[index]

    for index, secant in enumerate(delta):
        orientation = interval_sign[index]
        positive = 0.0
        for cell in (index, index + 1):
            if cell_run[cell] == interval_run[index]:
                positive += orientation * ray[cell]
        if positive > abs(secant) and positive > 0.0:
            inward = np.nextafter(abs(secant) / positive, 0.0)
            scale[interval_run[index]] = min(
                scale[interval_run[index]], inward
            )
    admitted = ray * scale[cell_run]
    return admitted


def conv_moment_proposal_1d(coarse: Array) -> Array:
    """Return the compact CONV half-cell transport proposal.

    The left and right children are averages over the two halves of a parent
    cell.  Each half-average of the admitted quintic is evaluated exactly by
    three-point Gauss-Legendre quadrature (degree five).  Their half
    difference is the proposed centered moment.  Endpoint cells use the
    corresponding compact one-sided first jet because one half-cell lies
    outside the represented domain.
    """

    value = np.asarray(coarse, dtype=np.float64)
    if value.shape[0] < 5:
        raise ValueError("CONV moment prediction requires at least five cells")
    _, first, _ = _local_fourth_order_2jet(value)
    profile = _quintic_variation_lineage_profile(value)
    nodes = np.array((-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)))
    weights = np.array((5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0))

    def half_average(interval: int, right_half: bool) -> Array:
        center = 0.75 if right_half else 0.25
        positions = interval + center + 0.25 * nodes
        values = np.stack([
            _evaluate_quintic_variation_lineage_profile(profile, position)
            for position in positions
        ])
        return np.tensordot(0.5 * weights, values, axes=(0, 0))

    proposal = first / 4.0
    for index in range(1, value.shape[0] - 1):
        left = half_average(index - 1, True)
        right = half_average(index, False)
        proposal[index] = 0.5 * (right - left)
    return proposal


def admit_moment_1d(coarse: Array, proposal: Array) -> Array:
    """Apply scalar ordered-current admission independently to all channels."""

    c = np.asarray(coarse, dtype=np.float64)
    raw = np.asarray(proposal, dtype=np.float64)
    if raw.shape != c.shape:
        raise ValueError("proposal must have the same shape as coarse")
    flat_c = c.reshape(c.shape[0], -1)
    flat_raw = raw.reshape(raw.shape[0], -1)
    result = np.empty_like(flat_raw)
    for component in range(flat_c.shape[1]):
        result[:, component] = _admit_moment_scalar(
            flat_c[:, component], flat_raw[:, component]
        )
    return result.reshape(c.shape)


def conv_moment_1d(coarse: Array) -> Array:
    """Return the admitted CONV child moment for a line of parent averages."""

    return admit_moment_1d(coarse, conv_moment_proposal_1d(coarse))


def conservative_restrict_1d(fine: Array, axis: int = 0) -> Array:
    """Return exact dyadic parent-cell averages."""

    value = np.moveaxis(np.asarray(fine, dtype=np.float64), axis, 0)
    if value.shape[0] % 2:
        raise ValueError("conservative restriction requires an even axis")
    coarse = 0.5 * (value[0::2] + value[1::2])
    return np.moveaxis(coarse, 0, axis)


@dataclass(frozen=True)
class ConservativeLineState:
    coarse: Array
    detail: Array


@dataclass(frozen=True)
class ConservativeLinePyramid:
    coarse: Array
    details: tuple[Array, ...]


def conservative_analysis_1d(fine: Array, axis: int = 0) -> ConservativeLineState:
    """Analyze fine cell averages into parent averages and moment residuals."""

    value = np.moveaxis(np.asarray(fine, dtype=np.float64), axis, 0)
    if value.shape[0] % 2 or value.shape[0] < 10:
        raise ValueError("analysis requires an even axis with at least ten cells")
    coarse = 0.5 * (value[0::2] + value[1::2])
    actual = 0.5 * (value[1::2] - value[0::2])
    predicted = conv_moment_1d(coarse)
    state = ConservativeLineState(coarse=coarse, detail=actual - predicted)
    return ConservativeLineState(
        coarse=np.moveaxis(state.coarse, 0, axis),
        detail=np.moveaxis(state.detail, 0, axis),
    )


def conservative_synthesis_1d(
    coarse: Array, detail: Array | None = None, axis: int = 0
) -> Array:
    """Synthesize child cell averages from a conservative CONV state."""

    parent = np.moveaxis(np.asarray(coarse, dtype=np.float64), axis, 0)
    predicted = conv_moment_1d(parent)
    if detail is None:
        residual = np.zeros_like(parent)
    else:
        residual = np.moveaxis(np.asarray(detail, dtype=np.float64), axis, 0)
        if residual.shape != parent.shape:
            raise ValueError("detail must have the same shape as coarse")
    moment = predicted + residual
    fine = np.empty((2 * parent.shape[0],) + parent.shape[1:], dtype=np.float64)
    fine[0::2] = parent - moment
    fine[1::2] = parent + moment
    return np.moveaxis(fine, 0, axis)


def conservative_multilevel_analysis_1d(
    fine: Array, levels: int | None = None, axis: int = 0
) -> ConservativeLinePyramid:
    """Analyze every admissible dyadic level, or exactly ``levels`` levels."""

    current = np.asarray(fine, dtype=np.float64)
    details: list[Array] = []
    while True:
        length = current.shape[axis]
        if length % 2 or length // 2 < 5:
            break
        if levels is not None and len(details) >= levels:
            break
        state = conservative_analysis_1d(current, axis=axis)
        current = state.coarse
        details.append(state.detail)
    if levels is not None and len(details) != levels:
        raise ValueError("requested levels exceed the admissible dyadic depth")
    return ConservativeLinePyramid(current, tuple(details))


def conservative_multilevel_synthesis_1d(
    pyramid: ConservativeLinePyramid, axis: int = 0
) -> Array:
    """Invert a conservative pyramid from coarsest to finest."""

    current = np.asarray(pyramid.coarse, dtype=np.float64)
    for detail in reversed(pyramid.details):
        current = conservative_synthesis_1d(current, detail, axis=axis)
    return current


def fine_currents_1d(coarse: Array, moment: Array) -> Array:
    """Return the child currents induced by a parent-average/moment pair."""

    c = np.asarray(coarse, dtype=np.float64)
    q = np.asarray(moment, dtype=np.float64)
    if c.ndim != 1 or q.shape != c.shape:
        raise ValueError("fine current certificate expects scalar equal lines")
    current = np.empty(2 * c.size - 1, dtype=np.float64)
    current[0::2] = 2.0 * q
    current[1::2] = np.diff(c) - q[:-1] - q[1:]
    return current


def sign_changes(value: Array) -> int:
    """Count sign changes after deleting binary64 roundoff zeros."""

    current = np.asarray(value, dtype=np.float64)
    tolerance = 1024.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(current), initial=0.0))
    )
    sign = np.sign(current[np.abs(current) > tolerance])
    return int(np.sum(sign[1:] != sign[:-1]))


def conservative_restrict_2d(fine: Array) -> Array:
    """Return exact 2x2 parent-cell averages on the first two axes."""

    value = np.asarray(fine, dtype=np.float64)
    if value.ndim < 2 or value.shape[0] % 2 or value.shape[1] % 2:
        raise ValueError("2-D restriction requires even first two axes")
    return 0.25 * (
        value[0::2, 0::2] + value[0::2, 1::2]
        + value[1::2, 0::2] + value[1::2, 1::2]
    )


@dataclass(frozen=True)
class BlockMoments:
    mean: Array
    horizontal: Array
    vertical: Array
    mixed: Array


def block_moments_2d(fine: Array) -> BlockMoments:
    """Apply the normalized 2x2 Walsh-Haar moment transform."""

    value = np.asarray(fine, dtype=np.float64)
    if value.ndim < 2 or value.shape[0] % 2 or value.shape[1] % 2:
        raise ValueError("2-D moments require even first two axes")
    x00 = value[0::2, 0::2]
    x01 = value[0::2, 1::2]
    x10 = value[1::2, 0::2]
    x11 = value[1::2, 1::2]
    return BlockMoments(
        mean=0.25 * (x00 + x01 + x10 + x11),
        horizontal=0.25 * (-x00 + x01 - x10 + x11),
        vertical=0.25 * (-x00 - x01 + x10 + x11),
        mixed=0.25 * (x00 - x01 - x10 + x11),
    )


def synthesize_block_moments_2d(moment: BlockMoments) -> Array:
    """Invert :func:`block_moments_2d` exactly in arithmetic."""

    c = np.asarray(moment.mean, dtype=np.float64)
    qx = np.asarray(moment.horizontal, dtype=np.float64)
    qy = np.asarray(moment.vertical, dtype=np.float64)
    qxy = np.asarray(moment.mixed, dtype=np.float64)
    if qx.shape != c.shape or qy.shape != c.shape or qxy.shape != c.shape:
        raise ValueError("all block moments must have the same shape")
    fine = np.empty((2 * c.shape[0], 2 * c.shape[1]) + c.shape[2:])
    fine[0::2, 0::2] = c - qx - qy + qxy
    fine[0::2, 1::2] = c + qx - qy - qxy
    fine[1::2, 0::2] = c - qx + qy - qxy
    fine[1::2, 1::2] = c + qx + qy + qxy
    return fine


def conv_moment_proposal_2d(coarse: Array) -> BlockMoments:
    """Return the tensor half-cell proposal for three centered child moments."""

    c = np.asarray(coarse, dtype=np.float64)
    if c.ndim < 2 or min(c.shape[:2]) < 5:
        raise ValueError("2-D CONV moment prediction requires at least 5x5 cells")
    xline = np.moveaxis(c, 1, 0)
    qx = np.moveaxis(conv_moment_proposal_1d(xline), 0, 1)
    qy = conv_moment_proposal_1d(c)
    qxy = conv_moment_proposal_1d(qx)
    return BlockMoments(c.copy(), qx, qy, qxy)


def _face_signs_scalar_2d(
    coarse: Array, proposal: BlockMoments
) -> tuple[Array, Array, Array, Array]:
    """Return horizontal/vertical interval and block lineage signs."""

    c = np.asarray(coarse, dtype=np.float64)
    height, width = c.shape
    sx = np.empty((height, width - 1), dtype=np.float64)
    tx = np.empty((height, width), dtype=np.float64)
    for row in range(height):
        sx[row] = _filled_sign(np.diff(c[row]))
        tx[row] = _cell_lineage_sign(proposal.horizontal[row], sx[row])
    sy = np.empty((height - 1, width), dtype=np.float64)
    ty = np.empty((height, width), dtype=np.float64)
    for column in range(width):
        sy[:, column] = _filled_sign(np.diff(c[:, column]))
        ty[:, column] = _cell_lineage_sign(
            proposal.vertical[:, column], sy[:, column]
        )
    return sx, tx, sy, ty


def _admit_moments_scalar_2d(
    coarse: Array, proposal: BlockMoments
) -> BlockMoments:
    """Intersect the tensor-jet ray with the complete face-current polytope.

    The internal-face cone is

        tx * (qx +/- qxy) >= 0,
        ty * (qy +/- qxy) >= 0.

    After orthant admission, each block receives the largest local ray scale
    left by all incident faces.  On one face, only proposal components opposed
    to the witnessed coarse sign consume its current capacity; the two
    incident blocks receive the common exact capacity ratio.  Taking the
    minimum over incident faces gives a finite local box inside the complete
    face-current polytope.  Helpful components are not needed for feasibility,
    so reducing a neighboring block can never invalidate an admitted face.
    """

    c = np.asarray(coarse, dtype=np.float64)
    sx, tx, sy, ty = _face_signs_scalar_2d(c, proposal)
    ax = np.maximum(tx * proposal.horizontal, 0.0)
    ay = np.maximum(ty * proposal.vertical, 0.0)
    mixed_cap = np.minimum(ax, ay)
    qxy = np.sign(proposal.mixed) * np.minimum(
        np.abs(proposal.mixed), mixed_cap
    )
    qx = tx * ax
    qy = ty * ay
    alpha = np.ones_like(c)

    def admit_face(
        first: tuple[int, int], second: tuple[int, int],
        base: float, orientation: float,
        residual_first: float, residual_second: float,
    ) -> None:
        if orientation == 0.0:
            if residual_first + residual_second != 0.0:
                alpha[first] = 0.0
                alpha[second] = 0.0
            return
        consumed = max(-orientation * residual_first, 0.0) + max(
            -orientation * residual_second, 0.0
        )
        if consumed > abs(base) and consumed > 0.0:
            cap = np.nextafter(abs(base) / consumed, 0.0)
            alpha[first] = min(alpha[first], cap)
            alpha[second] = min(alpha[second], cap)

    height, width = c.shape
    for row in range(height):
        for column in range(width - 1):
            a, b = (row, column), (row, column + 1)
            delta = c[b] - c[a]
            orientation = sx[row, column]
            # Upper cross-face current, equation (21), first line.
            admit_face(
                a, b, delta, orientation,
                -qx[a] + qy[a] + qxy[a],
                -qx[b] - qy[b] + qxy[b],
            )
            # Lower cross-face current, equation (21), second line.
            admit_face(
                a, b, delta, orientation,
                -qx[a] - qy[a] - qxy[a],
                -qx[b] + qy[b] - qxy[b],
            )

    for row in range(height - 1):
        for column in range(width):
            a, b = (row, column), (row + 1, column)
            delta = c[b] - c[a]
            orientation = sy[row, column]
            # Left cross-face current, equation (22), first line.
            admit_face(
                a, b, delta, orientation,
                qx[a] - qy[a] + qxy[a],
                -qx[b] - qy[b] + qxy[b],
            )
            # Right cross-face current, equation (22), second line.
            admit_face(
                a, b, delta, orientation,
                -qx[a] - qy[a] - qxy[a],
                qx[b] - qy[b] - qxy[b],
            )

    return BlockMoments(c.copy(), alpha * qx, alpha * qy, alpha * qxy)


def admit_moments_2d(coarse: Array, proposal: BlockMoments) -> BlockMoments:
    """Apply the face-current admission independently to every value channel."""

    c = np.asarray(coarse, dtype=np.float64)
    shape = c.shape
    flat_c = c.reshape(shape[0], shape[1], -1)
    flat_x = np.asarray(proposal.horizontal).reshape(shape[0], shape[1], -1)
    flat_y = np.asarray(proposal.vertical).reshape(shape[0], shape[1], -1)
    flat_xy = np.asarray(proposal.mixed).reshape(shape[0], shape[1], -1)
    out_x = np.empty_like(flat_x)
    out_y = np.empty_like(flat_y)
    out_xy = np.empty_like(flat_xy)
    for component in range(flat_c.shape[2]):
        admitted = _admit_moments_scalar_2d(
            flat_c[..., component],
            BlockMoments(
                flat_c[..., component], flat_x[..., component],
                flat_y[..., component], flat_xy[..., component],
            ),
        )
        out_x[..., component] = admitted.horizontal
        out_y[..., component] = admitted.vertical
        out_xy[..., component] = admitted.mixed
    return BlockMoments(
        c.copy(), out_x.reshape(shape), out_y.reshape(shape), out_xy.reshape(shape)
    )


def conv_moments_2d(coarse: Array) -> BlockMoments:
    """Return admitted horizontal, vertical, and mixed CONV child moments."""

    proposal = conv_moment_proposal_2d(coarse)
    return admit_moments_2d(coarse, proposal)


@dataclass(frozen=True)
class ConservativeBlockState:
    coarse: Array
    horizontal_detail: Array
    vertical_detail: Array
    mixed_detail: Array


def conservative_analysis_2d(fine: Array) -> ConservativeBlockState:
    """Analyze 2x2 child averages into one mean and three residual moments."""

    actual = block_moments_2d(fine)
    if min(actual.mean.shape[:2]) < 5:
        raise ValueError("2-D conservative analysis requires at least 5x5 parents")
    predicted = conv_moments_2d(actual.mean)
    return ConservativeBlockState(
        actual.mean,
        actual.horizontal - predicted.horizontal,
        actual.vertical - predicted.vertical,
        actual.mixed - predicted.mixed,
    )


def conservative_synthesis_2d(
    coarse: Array,
    horizontal_detail: Array | None = None,
    vertical_detail: Array | None = None,
    mixed_detail: Array | None = None,
) -> Array:
    """Synthesize a fine raster from a conservative block state."""

    c = np.asarray(coarse, dtype=np.float64)
    predicted = conv_moments_2d(c)
    details = []
    for supplied in (horizontal_detail, vertical_detail, mixed_detail):
        if supplied is None:
            details.append(np.zeros_like(c))
        else:
            value = np.asarray(supplied, dtype=np.float64)
            if value.shape != c.shape:
                raise ValueError("every 2-D detail must have the coarse shape")
            details.append(value)
    return synthesize_block_moments_2d(BlockMoments(
        c,
        predicted.horizontal + details[0],
        predicted.vertical + details[1],
        predicted.mixed + details[2],
    ))


def face_sign_certificate_2d(coarse: Array, moment: BlockMoments) -> dict[str, int]:
    """Count factorwise sign surplus in the synthesized face currents."""

    c = np.asarray(coarse, dtype=np.float64)
    horizontal, vertical = face_currents_2d(moment)
    horizontal_surplus = 0
    for row in range(c.shape[0]):
        coarse_changes = sign_changes(np.diff(c[row]))
        horizontal_surplus = max(
            horizontal_surplus,
            sign_changes(horizontal[2 * row]) - coarse_changes,
            sign_changes(horizontal[2 * row + 1]) - coarse_changes,
        )
    vertical_surplus = 0
    for column in range(c.shape[1]):
        coarse_changes = sign_changes(np.diff(c[:, column]))
        vertical_surplus = max(
            vertical_surplus,
            sign_changes(vertical[:, 2 * column]) - coarse_changes,
            sign_changes(vertical[:, 2 * column + 1]) - coarse_changes,
        )
    return {
        "horizontal_sign_surplus": int(horizontal_surplus),
        "vertical_sign_surplus": int(vertical_surplus),
    }


def face_currents_2d(moment: BlockMoments) -> tuple[Array, Array]:
    """Return every horizontal and vertical fine-grid face current.

    The returned arrays have shapes ``(2H,2W-1)`` and ``(2H-1,2W)``.
    This routine is also an executable certificate for the affine face-current
    formulas used by the two-dimensional admissible polytope.
    """

    fine = synthesize_block_moments_2d(moment)
    return np.diff(fine, axis=1), np.diff(fine, axis=0)
