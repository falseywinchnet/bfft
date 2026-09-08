"""Global spectral-moment transport for conservative CONV investigation.

The spectral continuation proposes subcell moments.  A single strictly convex
quadratic program then finds the nearest moment field in the complete fixed
face-current chamber.  This is the global polytope whose inexpensive local-ray
inner approximation is used by the current real-time implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from experiments.conv_conservative_multiresolution import (
    BlockMoments,
    _face_signs_scalar_2d,
    synthesize_block_moments_2d,
)


Array = np.ndarray


def _pixel_center_lanczos_matrix(count: int, radius: int = 3) -> Array:
    target = 2 * count
    coordinate = (np.arange(target, dtype=np.float64) + 0.5) / 2.0 - 0.5
    source = np.arange(count, dtype=np.float64)
    matrix = np.zeros((target, count), dtype=np.float64)
    for row, location in enumerate(coordinate):
        distance = location - source
        admitted = np.abs(distance) < radius
        matrix[row, admitted] = (
            np.sinc(distance[admitted])
            * np.sinc(distance[admitted] / radius)
        )
        matrix[row] /= np.sum(matrix[row])
    return matrix


def spectral_moment_proposal(coarse: Array) -> BlockMoments:
    """Extract conservative moments from a cell-centered Lanczos continuation."""

    value = np.asarray(coarse, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("the projection oracle accepts one scalar channel")
    wy = _pixel_center_lanczos_matrix(value.shape[0])
    wx = _pixel_center_lanczos_matrix(value.shape[1])
    fine = wy @ value @ wx.T
    x00 = fine[0::2, 0::2]
    x01 = fine[0::2, 1::2]
    x10 = fine[1::2, 0::2]
    x11 = fine[1::2, 1::2]
    return BlockMoments(
        value.copy(),
        0.25 * (-x00 + x01 - x10 + x11),
        0.25 * (-x00 - x01 + x10 + x11),
        0.25 * (x00 - x01 - x10 + x11),
    )


@dataclass(frozen=True)
class ProjectionResult:
    moments: BlockMoments
    status: str
    iterations: int
    primal_residual: float
    dual_residual: float
    objective: float


def project_complete_face_chamber(
    coarse: Array,
    proposal: BlockMoments,
    *,
    tolerance: float = 1.0e-9,
    maximum_iterations: int = 100_000,
    bound_range: bool = True,
) -> ProjectionResult:
    """Project onto every internal and cross-parent face-current constraint."""

    c = np.asarray(coarse, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("the projection oracle accepts one scalar channel")
    height, width = c.shape
    cells = height * width
    sx, tx, sy, ty = _face_signs_scalar_2d(c, proposal)
    row_index: list[int] = []
    column_index: list[int] = []
    coefficient: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    def variable(cell: int, component: int) -> int:
        return component * cells + cell

    def append(
        entries: list[tuple[int, float]],
        low: float,
        high: float = np.inf,
    ) -> None:
        nonlocal row
        for column, value in entries:
            if value != 0.0:
                row_index.append(row)
                column_index.append(column)
                coefficient.append(value)
        lower.append(low)
        upper.append(high)
        row += 1

    # Two internal horizontal and two internal vertical child faces per block.
    for y in range(height):
        for x in range(width):
            cell = y * width + x
            faces = (
                (tx[y, x], [(variable(cell, 0), 1.0), (variable(cell, 2), -1.0)]),
                (tx[y, x], [(variable(cell, 0), 1.0), (variable(cell, 2), 1.0)]),
                (ty[y, x], [(variable(cell, 1), 1.0), (variable(cell, 2), -1.0)]),
                (ty[y, x], [(variable(cell, 1), 1.0), (variable(cell, 2), 1.0)]),
            )
            for orientation, entries in faces:
                if orientation == 0.0:
                    append(entries, 0.0, 0.0)
                else:
                    append(
                        [(index, orientation * value) for index, value in entries],
                        0.0,
                    )

    # Upper and lower child currents crossing each horizontal parent face.
    for y in range(height):
        for x in range(width - 1):
            first = y * width + x
            second = first + 1
            delta = c[y, x + 1] - c[y, x]
            orientation = sx[y, x]
            faces = (
                [
                    (variable(first, 0), -1.0), (variable(first, 1), 1.0),
                    (variable(first, 2), 1.0), (variable(second, 0), -1.0),
                    (variable(second, 1), -1.0), (variable(second, 2), 1.0),
                ],
                [
                    (variable(first, 0), -1.0), (variable(first, 1), -1.0),
                    (variable(first, 2), -1.0), (variable(second, 0), -1.0),
                    (variable(second, 1), 1.0), (variable(second, 2), -1.0),
                ],
            )
            for entries in faces:
                if orientation == 0.0:
                    append(entries, -delta, -delta)
                else:
                    append(
                        [(index, orientation * value) for index, value in entries],
                        -orientation * delta,
                    )

    # Left and right child currents crossing each vertical parent face.
    for y in range(height - 1):
        for x in range(width):
            first = y * width + x
            second = first + width
            delta = c[y + 1, x] - c[y, x]
            orientation = sy[y, x]
            faces = (
                [
                    (variable(first, 0), 1.0), (variable(first, 1), -1.0),
                    (variable(first, 2), 1.0), (variable(second, 0), -1.0),
                    (variable(second, 1), -1.0), (variable(second, 2), 1.0),
                ],
                [
                    (variable(first, 0), -1.0), (variable(first, 1), -1.0),
                    (variable(first, 2), -1.0), (variable(second, 0), 1.0),
                    (variable(second, 1), -1.0), (variable(second, 2), -1.0),
                ],
            )
            for entries in faces:
                if orientation == 0.0:
                    append(entries, -delta, -delta)
                else:
                    append(
                        [(index, orientation * value) for index, value in entries],
                        -orientation * delta,
                    )

    if bound_range:
        minimum = float(np.min(c))
        maximum = float(np.max(c))
        # Each child value is c plus one signed combination of the three
        # centered moments.  These four slabs are precisely the inverse
        # Walsh--Haar range constraints; no limiter coefficient is introduced.
        child_signs = (
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        )
        for y in range(height):
            for x in range(width):
                cell = y * width + x
                for sign_x, sign_y, sign_xy in child_signs:
                    append(
                        [
                            (variable(cell, 0), sign_x),
                            (variable(cell, 1), sign_y),
                            (variable(cell, 2), sign_xy),
                        ],
                        minimum - c[y, x],
                        maximum - c[y, x],
                    )

    matrix = sparse.csc_matrix(
        (coefficient, (row_index, column_index)), shape=(row, 3 * cells)
    )
    proposed = np.concatenate((
        np.ravel(proposal.horizontal),
        np.ravel(proposal.vertical),
        np.ravel(proposal.mixed),
    ))
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.eye(3 * cells, format="csc"),
        q=-proposed,
        A=matrix,
        l=np.asarray(lower),
        u=np.asarray(upper),
        eps_abs=tolerance,
        eps_rel=tolerance,
        max_iter=maximum_iterations,
        polishing=True,
        verbose=False,
    )
    result = solver.solve()
    if not result.info.status.lower().startswith("solved"):
        raise RuntimeError(f"global transport projection failed: {result.info.status}")
    value = np.asarray(result.x)
    moments = BlockMoments(
        c.copy(),
        value[:cells].reshape(c.shape),
        value[cells:2 * cells].reshape(c.shape),
        value[2 * cells:].reshape(c.shape),
    )
    return ProjectionResult(
        moments=moments,
        status=result.info.status,
        iterations=int(result.info.iter),
        primal_residual=float(result.info.prim_res),
        dual_residual=float(result.info.dual_res),
        objective=float(result.info.obj_val),
    )


def spectral_transport_synthesis(coarse: Array) -> ProjectionResult:
    """One zero-detail level of spectral proposal and global admission."""

    proposal = spectral_moment_proposal(coarse)
    projected = project_complete_face_chamber(coarse, proposal)
    # Constructing the raster here also exercises every moment shape before the
    # caller uses the returned coordinates.
    synthesize_block_moments_2d(projected.moments)
    return projected
