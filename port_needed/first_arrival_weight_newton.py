"""One-label source-energy Newton step on the accepted cell graph.

For arrival fields

    T_i(x) = d_M(s_i, x) - w_i,

the winning first-arrival cell is ``argmin_i T_i``.  Increasing ``w_i`` makes
source i arrive earlier and expands its accepted mass.  If each site should
consume one equal quantum of a resource density, the mass mismatch is the
gradient of the semi-discrete transport dual.

The derivative is assembled only on literal hard interfaces.  For adjacent
pixels p and q with crossing action c, linear collision gives an interface
motion of ``delta(w_i-w_j)/(2c)``.  Hence the local conductance is
``rho_edge/(2c)`` and the Newton matrix is the Laplacian of the accepted cell
adjacency graph.  No runner-up arrival is evaluated or stored.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .continuous_eikonal_transport import (
    continuous_first_partition_prepared,
)


def accepted_mass(
    labels: np.ndarray,
    measure: np.ndarray,
    cells: int,
) -> np.ndarray:
    return np.bincount(
        np.asarray(labels, dtype=np.int32).ravel(),
        weights=np.asarray(measure, dtype=np.float64).ravel(),
        minlength=int(cells),
    ).astype(np.float64)


def frontier_laplacian(
    labels: np.ndarray,
    measure: np.ndarray,
    cardinal_costs: np.ndarray,
    gradient_x: np.ndarray | None = None,
    gradient_y: np.ndarray | None = None,
    cells: int | None = None,
) -> sparse.csr_matrix:
    """Assemble the first-arrival interface Hessian in two image passes.

    In the continuum, an interface between cells i and j moves by

        delta(w_i - w_j) / |p_i - p_j|,

    where p_i and p_j are the two one-sided arrival covectors.  A horizontal
    or vertical grid crossing samples projected interface length.  Combining
    both crossing families therefore gives the grid-Crofton discretization

        rho / (|p_i.x - p_j.x| + |p_i.y - p_j.y|).

    This uses only the two literal sides of the accepted hard interface.  It
    does not evaluate or retain a runner-up arrival.
    """
    labels = np.asarray(labels, dtype=np.int32)
    measure = np.asarray(measure, dtype=np.float64)
    costs = np.asarray(cardinal_costs, dtype=np.float64)
    gx = None if gradient_x is None else np.asarray(
        gradient_x, dtype=np.float64)
    gy = None if gradient_y is None else np.asarray(
        gradient_y, dtype=np.float64)
    if cells is None:
        cells = int(np.max(labels)) + 1
    rows = []
    columns = []
    values = []
    # cardinal_costs order: right, left, down, up.
    for dy, dx, direction in ((0, 1, 0), (1, 0, 2)):
        ys = slice(0, labels.shape[0] - dy)
        xs = slice(0, labels.shape[1] - dx)
        yd = slice(dy, labels.shape[0])
        xd = slice(dx, labels.shape[1])
        first = labels[ys, xs]
        second = labels[yd, xd]
        crossing = first != second
        if not np.any(crossing):
            continue
        first = first[crossing].astype(np.int32)
        second = second[crossing].astype(np.int32)
        density = 0.5 * (
            measure[ys, xs][crossing] + measure[yd, xd][crossing])
        if gx is not None and gy is not None:
            covector_l1_jump = (
                np.abs(gx[ys, xs][crossing] - gx[yd, xd][crossing])
                + np.abs(gy[ys, xs][crossing] - gy[yd, xd][crossing])
            )
            conductance = density / np.maximum(
                covector_l1_jump, 1e-6)
        else:
            action = costs[ys, xs][..., direction][crossing]
            conductance = density / np.maximum(2.0 * action, 1e-30)
        rows.extend((first, first, second, second))
        columns.extend((first, second, first, second))
        values.extend((
            conductance,
            -conductance,
            -conductance,
            conductance,
        ))
    if not rows:
        return sparse.csr_matrix((cells, cells), dtype=np.float64)
    return sparse.coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(cells, cells),
    ).tocsr()


def newton_source_energy(
    labels: np.ndarray,
    measure: np.ndarray,
    cardinal_costs: np.ndarray,
    target_mass: np.ndarray,
    *,
    gradient_x: np.ndarray | None = None,
    gradient_y: np.ndarray | None = None,
    regularization: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """Solve one gauge-fixed source-energy step from accepted mass error."""
    target = np.asarray(target_mass, dtype=np.float64)
    cells = len(target)
    mass = accepted_mass(labels, measure, cells)
    mismatch = target - mass
    mismatch -= np.mean(mismatch)
    laplacian = frontier_laplacian(
        labels,
        measure,
        cardinal_costs,
        gradient_x,
        gradient_y,
        cells,
    )
    # The nullspace is the irrelevant common source-time offset. Pin the last
    # source during the solve, then restore a zero-mean gauge.
    reduced = laplacian[:-1, :-1].tocsc()
    scale = max(float(np.mean(reduced.diagonal())), 1e-30)
    reduced = reduced + sparse.eye(
        cells - 1, format="csc") * (regularization * scale)
    delta = np.zeros(cells, dtype=np.float64)
    if cells > 1:
        delta[:-1] = spsolve(reduced, mismatch[:-1])
    delta -= np.mean(delta)
    diagnostic = {
        "mass": mass,
        "target_mass": target,
        "mismatch": mismatch,
        "mass_cv": float(np.std(mass) / max(np.mean(mass), 1e-30)),
        "maximum_relative_mass_error": float(np.max(
            np.abs(mismatch) / np.maximum(target, 1e-30))),
        "interface_edges": int(laplacian.nnz),
        "step_rms": float(np.sqrt(np.mean(delta * delta))),
        "step_max": float(np.max(np.abs(delta))),
    }
    return delta, diagnostic


def local_source_pressure_step(
    labels: np.ndarray,
    measure: np.ndarray,
    cardinal_costs: np.ndarray,
    target_mass: np.ndarray,
    *,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """One simultaneous local pressure response on the current topology.

    The diagonal Hessian ``H_ii`` is the permeability of cell i's entire
    present frontier. A pressure mismatch acting on one interface moves both
    incident sources, so the simultaneous Jacobi response carries the exact
    two-cell factor one half:

        delta w_i = (target_i - mass_i) / (2 H_ii).

    Unlike the global inverse, this cannot transmit pressure through a chain
    of cells before the intervening fronts have actually moved. A subsequent
    causal remarch is the topology refresh.
    """
    target = np.asarray(target_mass, dtype=np.float64)
    cells = len(target)
    mass = accepted_mass(labels, measure, cells)
    mismatch = target - mass
    mismatch -= np.mean(mismatch)
    laplacian = frontier_laplacian(
        labels,
        measure,
        cardinal_costs,
        gradient_x,
        gradient_y,
        cells,
    )
    permeability = np.asarray(laplacian.diagonal(), dtype=np.float64)
    active = permeability > 1e-30
    delta = np.zeros(cells, dtype=np.float64)
    delta[active] = 0.5 * mismatch[active] / permeability[active]
    delta -= np.mean(delta)
    return delta, {
        "mass": mass,
        "target_mass": target,
        "mismatch": mismatch,
        "permeability": permeability,
        "mass_cv": float(np.std(mass) / max(np.mean(mass), 1e-30)),
        "inactive_fronts": int(np.count_nonzero(~active)),
        "step_rms": float(np.sqrt(np.mean(delta * delta))),
        "step_max": float(np.max(np.abs(delta))),
    }


def equalize_first_arrival_mass(
    centers: np.ndarray,
    prepared_metric: dict[str, np.ndarray],
    measure: np.ndarray,
    *,
    passes: int = 1,
    damping: float = 1.0,
    initial_reach: np.ndarray | None = None,
) -> tuple[np.ndarray, dict, list[dict]]:
    """Apply global Newton energy updates separated by exact remarches."""
    centers = np.asarray(centers, dtype=np.float64)
    cells = len(centers)
    reach = (
        np.zeros(cells, dtype=np.float64)
        if initial_reach is None
        else np.asarray(initial_reach, dtype=np.float64).copy()
    )
    target = np.full(
        cells,
        float(np.sum(measure)) / max(cells, 1),
        dtype=np.float64,
    )
    trace = []
    partition = continuous_first_partition_prepared(
        centers, prepared_metric, reach)
    for iteration in range(max(int(passes), 0)):
        delta, diagnostic = newton_source_energy(
            partition["labels"],
            measure,
            prepared_metric["cardinal_costs"],
            target,
            gradient_x=partition["gradient_x"],
            gradient_y=partition["gradient_y"],
        )
        diagnostic["iteration"] = iteration + 1
        diagnostic["accepted_step_scale"] = float(damping)
        trace.append(diagnostic)
        reach += float(damping) * delta
        reach -= np.mean(reach)
        partition = continuous_first_partition_prepared(
            centers, prepared_metric, reach)
    final_mass = accepted_mass(partition["labels"], measure, cells)
    trace.append({
        "iteration": len(trace) + 1,
        "mass": final_mass,
        "target_mass": target,
        "mass_cv": float(
            np.std(final_mass) / max(np.mean(final_mass), 1e-30)),
        "maximum_relative_mass_error": float(np.max(
            np.abs(final_mass - target) / np.maximum(target, 1e-30))),
        "final": True,
    })
    return reach, partition, trace
