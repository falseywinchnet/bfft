"""Locally channel-complete global SDP relaxation.

The spectral relaxation can collapse its three sections at individual blocks.
This tighter orthogonal-synchronization relaxation cannot: every region has an
identity 3x3 diagonal Gram block.

    maximize  <H, X>
    subject to X >= 0,  X[ii] = I_3 for every bloom region i.

It is a convex semidefinite program.  Its optimum is a global upper bound on
the nonconvex O(3) phase-alignment objective, and the block identity constraints
encode the requested full utilization of all three channels at every region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral_relaxation import OrthogonalConnections, build_connections


@dataclass
class SDPSynchronizationResult:
    region_frames: np.ndarray
    block_frames: np.ndarray
    region_labels: np.ndarray
    gram: np.ndarray
    relaxation_value: float
    rounded_value: float
    rounding_gap: float
    dual_upper_bound: float
    certified_gap: float
    minimum_gram_eigenvalue: float
    maximum_diagonal_error: float
    gram_rank: int
    region_count: int
    edge_count: int
    solver: str
    solver_status: str

    def report(self) -> dict:
        return {
            "relaxation_value": self.relaxation_value,
            "rounded_value": self.rounded_value,
            "rounding_gap": self.rounding_gap,
            "dual_upper_bound": self.dual_upper_bound,
            "certified_gap": self.certified_gap,
            "minimum_gram_eigenvalue": self.minimum_gram_eigenvalue,
            "maximum_diagonal_error": self.maximum_diagonal_error,
            "gram_rank": self.gram_rank,
            "region_count": self.region_count,
            "edge_count": self.edge_count,
            "solver": self.solver,
            "solver_status": self.solver_status,
            "proof": (
                "Convex SDP optimum with X positive semidefinite and every "
                "3x3 diagonal block fixed to identity; the dual feasible "
                "objective is a global upper bound."
            ),
        }


def _coarsen_bloom_regions(
    block_connections: OrthogonalConnections,
    block_count: int,
    maximum_regions: int,
) -> np.ndarray:
    """Maximum-confidence connected forest cut, analogous to v3 phase bloom."""
    parent = np.arange(block_count, dtype=np.int32)
    size = np.ones(block_count, dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    regions = block_count
    order = np.argsort(-block_connections.weight, kind="stable")
    for edge in order:
        if regions <= maximum_regions:
            break
        left = find(int(block_connections.left[edge]))
        right = find(int(block_connections.right[edge]))
        if left == right:
            continue
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]
        regions -= 1
    roots = np.array([find(index) for index in range(block_count)])
    _, compact = np.unique(roots, return_inverse=True)
    return compact.astype(np.int32)


def _region_connections(
    block_connections: OrthogonalConnections,
    region_labels: np.ndarray,
) -> OrthogonalConnections:
    accumulators: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    for left, right, rotation, weight in zip(
        block_connections.left,
        block_connections.right,
        block_connections.rotation,
        block_connections.weight,
    ):
        i, j = int(region_labels[left]), int(region_labels[right])
        if i == j:
            continue
        if i > j:
            i, j = j, i
            rotation = rotation.T
        matrix, total = accumulators.get((i, j), (np.zeros((3, 3)), 0.0))
        accumulators[(i, j)] = (matrix + weight * rotation, total + weight)
    left, right, rotations, weights = [], [], [], []
    for (i, j), (matrix, total) in sorted(accumulators.items()):
        u, _, vt = np.linalg.svd(matrix, full_matrices=False)
        left.append(i)
        right.append(j)
        rotations.append(u @ vt)
        weights.append(total)
    return OrthogonalConnections(
        np.asarray(left, dtype=np.int32),
        np.asarray(right, dtype=np.int32),
        np.asarray(rotations, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _objective_matrix(region_count: int, connections: OrthogonalConnections) -> np.ndarray:
    matrix = np.zeros((3 * region_count, 3 * region_count), dtype=np.float64)
    for i, j, rotation, weight in zip(
        connections.left,
        connections.right,
        connections.rotation,
        connections.weight,
    ):
        si, sj = slice(3 * int(i), 3 * int(i) + 3), slice(3 * int(j), 3 * int(j) + 3)
        matrix[si, sj] += 0.5 * weight * rotation
        matrix[sj, si] += 0.5 * weight * rotation.T
    return matrix


def _rounded_objective(frames: np.ndarray, connections: OrthogonalConnections) -> float:
    value = 0.0
    for i, j, rotation, weight in zip(
        connections.left,
        connections.right,
        connections.rotation,
        connections.weight,
    ):
        gram = frames[int(i)] @ frames[int(j)].T
        value += float(weight * np.sum(rotation * gram))
    return value


def solve_region_sdp(
    region_count: int,
    connections: OrthogonalConnections,
    *,
    solver: str = "CLARABEL",
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    try:
        import cvxpy as cp
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "The globally certified SDP needs cvxpy: python -m pip install cvxpy"
        ) from error

    dimension = 3 * region_count
    objective_matrix = _objective_matrix(region_count, connections)
    gram_variable = cp.Variable((dimension, dimension), symmetric=True)
    diagonal_constraints = [
        gram_variable[3 * index:3 * index + 3, 3 * index:3 * index + 3] == np.eye(3)
        for index in range(region_count)
    ]
    constraints = [gram_variable >> 0, *diagonal_constraints]
    problem = cp.Problem(cp.Maximize(cp.trace(objective_matrix @ gram_variable)), constraints)
    if solver.upper() == "CLARABEL":
        problem.solve(
            solver=cp.CLARABEL,
            tol_gap_abs=tolerance,
            tol_gap_rel=tolerance,
            tol_feas=tolerance,
            max_iter=500,
            verbose=False,
        )
    else:
        problem.solve(solver=cp.SCS, eps=tolerance, max_iters=100_000, verbose=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"SDP solver failed with status {problem.status}")
    gram = np.asarray(gram_variable.value, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)

    # Construct a numerically feasible dual certificate from equality duals.
    dual_blocks = [np.asarray(constraint.dual_value, dtype=np.float64) for constraint in diagonal_constraints]
    dual_matrix = np.zeros_like(gram)
    for index, block in enumerate(dual_blocks):
        dual_matrix[3 * index:3 * index + 3, 3 * index:3 * index + 3] = block
    slack = dual_matrix - objective_matrix
    minimum_slack = float(np.linalg.eigvalsh(0.5 * (slack + slack.T))[0])
    correction = max(0.0, -minimum_slack + 10.0 * np.finfo(float).eps)
    dual_upper = float(sum(np.trace(block) for block in dual_blocks) + 3 * region_count * correction)

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    top = np.maximum(eigenvalues[order[:3]], 0.0)
    factor = eigenvectors[:, order[:3]] * np.sqrt(top)[None, :]
    frames = np.empty((region_count, 3, 3), dtype=np.float64)
    for region in range(region_count):
        block = factor[3 * region:3 * region + 3]
        u, _, vt = np.linalg.svd(block, full_matrices=False)
        frames[region] = u @ vt

    # Fix the global gauge nearest the physical YCbCr frame.
    aggregate = np.sum(frames, axis=0)
    u, _, vt = np.linalg.svd(aggregate.T, full_matrices=False)
    frames = frames @ (u @ vt)
    primal_value = float(np.sum(objective_matrix * gram))
    rounded_value = _rounded_objective(frames, connections)
    diagonal_error = max(
        float(np.max(np.abs(gram[3 * i:3 * i + 3, 3 * i:3 * i + 3] - np.eye(3))))
        for i in range(region_count)
    )
    positive_rank = int(np.count_nonzero(eigenvalues > max(eigenvalues[-1], 1.0) * 1e-7))
    return frames, gram, {
        "relaxation_value": primal_value,
        "rounded_value": rounded_value,
        "rounding_gap": max(0.0, primal_value - rounded_value),
        "dual_upper_bound": dual_upper,
        "certified_gap": max(0.0, dual_upper - primal_value),
        "minimum_gram_eigenvalue": float(eigenvalues[0]),
        "maximum_diagonal_error": diagonal_error,
        "gram_rank": positive_rank,
        "solver": solver.upper(),
        "solver_status": str(problem.status),
    }


def solve_sdp_relaxation(
    coefficients: np.ndarray,
    block_labels: np.ndarray,
    *,
    maximum_regions: int = 48,
    cross_region_weight: float = 0.05,
    solver: str = "CLARABEL",
    tolerance: float = 1e-7,
) -> SDPSynchronizationResult:
    block_connections = build_connections(
        coefficients, block_labels, cross_region_weight=cross_region_weight
    )
    region_labels = _coarsen_bloom_regions(
        block_connections, len(coefficients), maximum_regions
    )
    region_count = int(region_labels.max()) + 1
    connections = _region_connections(block_connections, region_labels)
    frames, gram, diagnostics = solve_region_sdp(
        region_count, connections, solver=solver, tolerance=tolerance
    )
    return SDPSynchronizationResult(
        region_frames=frames,
        block_frames=frames[region_labels],
        region_labels=region_labels,
        gram=gram,
        region_count=region_count,
        edge_count=len(connections.left),
        **diagnostics,
    )
