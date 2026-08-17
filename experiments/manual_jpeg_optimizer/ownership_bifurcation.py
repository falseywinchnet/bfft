"""Ownership-preserving phase transport and optimal bifurcation tree.

This is the JPEG analogue of the repository's relaxed-chip and Wasserstein
allocation methods:

* a maximum-confidence predecessor forest unwraps each DCT mode's sign phase;
* every coefficient vector remains an individually owned transport atom;
* an unstable cell bifurcates along its Courant--Fischer principal direction;
* the branch boundary is the exact conserved-mass weighted median;
* dynamic programming gives the global optimum over every pruning of the
  resulting ownership tree; and
* one inverse pass restores phase and YCbCr coordinates.

The proof scope is the generated causal tree.  There is no frame marginal and
no post-hoc reassignment of atoms to leaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

import numpy as np

from .core import quality_table


@dataclass(frozen=True)
class BifurcationConfig:
    rate_lambda: float = 0.05
    branch_penalty: float = 3.0
    maximum_depth: int = 8
    minimum_atoms: int = 256
    maximum_condition: float = 24.0
    cross_region_weight: float = 0.05


@dataclass
class OwnershipNode:
    index: int
    parent: int
    depth: int
    atoms: np.ndarray
    mass: float
    mean: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    frame: np.ndarray
    direction: np.ndarray
    boundary: float
    children: tuple[int, int] | None = None
    stop_cost: float = math.inf
    optimum_cost: float = math.inf
    selected_stop: bool = True
    use_constituent_route: bool = False


@dataclass
class OwnershipBifurcationResult:
    coefficients: np.ndarray
    leaf_of_atom: np.ndarray
    gauge: np.ndarray
    phase_parent: np.ndarray
    nodes: list[OwnershipNode]
    selected_leaves: np.ndarray
    objective: float
    stop_control_cost: float
    leaf_count: int
    maximum_depth: int
    phase_forest_hash: str
    ownership_hash: str
    channel_energy_fraction: np.ndarray
    prequantization_max_composition_error: float
    changed_quantized_coefficients: int
    routed_leaf_count: int
    config: BifurcationConfig

    def report(self) -> dict:
        return {
            "objective": self.objective,
            "stop_control_cost": self.stop_control_cost,
            "objective_gain": self.stop_control_cost - self.objective,
            "leaf_count": self.leaf_count,
            "tree_nodes": len(self.nodes),
            "maximum_depth": self.maximum_depth,
            "phase_forest_hash": self.phase_forest_hash,
            "ownership_hash": self.ownership_hash,
            "channel_energy_fraction": self.channel_energy_fraction.tolist(),
            "prequantization_max_composition_error": self.prequantization_max_composition_error,
            "changed_quantized_coefficients": self.changed_quantized_coefficients,
            "routed_leaf_count": self.routed_leaf_count,
            "identity_leaf_count": self.leaf_count - self.routed_leaf_count,
            "config": self.config.__dict__,
            "proof": (
                "Bellman recursion exactly minimizes identity versus "
                "composition-preserving constituent transport and every "
                "pruning of the stored causal bifurcation tree."
            ),
        }


def _grid_edges(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(height * width, dtype=np.int32).reshape(height, width)
    return (
        np.concatenate((grid[:, :-1].ravel(), grid[:-1, :].ravel())),
        np.concatenate((grid[:, 1:].ravel(), grid[1:, :].ravel())),
    )


def phase_predecessor_forest(
    coefficients: np.ndarray,
    block_labels: np.ndarray,
    *,
    cross_region_weight: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-confidence signed spanning tree for every non-DC DCT mode."""
    values = np.asarray(coefficients, dtype=np.float64)
    blocks, modes, channels = values.shape
    if modes != 64 or channels != 3 or block_labels.size != blocks:
        raise ValueError("expected blocks x 64 x 3 coefficients and block labels")
    left, right = _grid_edges(*block_labels.shape)
    same = block_labels.ravel()[left] == block_labels.ravel()[right]
    region_weight = np.where(same, 1.0, np.clip(cross_region_weight, 0.0, 1.0))
    gauge = np.ones((blocks, 64), dtype=np.float64)
    predecessor = np.full((blocks, 64), -1, dtype=np.int32)
    for mode in range(1, 64):
        first, second = values[left, mode], values[right, mode]
        dot = np.sum(first * second, axis=1)
        norm = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        confidence = region_weight * np.abs(dot) / np.maximum(norm, 1e-12)
        relation = np.where(dot < 0.0, -1.0, 1.0)
        order = np.argsort(-confidence, kind="stable")
        parent = np.arange(blocks, dtype=np.int32)
        size = np.ones(blocks, dtype=np.int32)
        adjacency: list[list[tuple[int, float]]] = [[] for _ in range(blocks)]

        def find(item: int) -> int:
            root = item
            while parent[root] != root:
                root = int(parent[root])
            while parent[item] != item:
                next_item = int(parent[item])
                parent[item] = root
                item = next_item
            return root

        selected = 0
        for edge in order:
            i, j = int(left[edge]), int(right[edge])
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            if size[ri] < size[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            size[ri] += size[rj]
            sign = float(relation[edge])
            adjacency[i].append((j, sign))
            adjacency[j].append((i, sign))
            selected += 1
            if selected == blocks - 1:
                break

        stack = [0]
        visited = np.zeros(blocks, dtype=bool)
        visited[0] = True
        while stack:
            i = stack.pop()
            for j, sign in adjacency[i]:
                if visited[j]:
                    continue
                visited[j] = True
                predecessor[j, mode] = i
                gauge[j, mode] = gauge[i, mode] * sign
                stack.append(j)
        # Degenerate one-pixel components retain their own root and unit gauge.
    return gauge, predecessor


def _weighted_statistics(
    vectors: np.ndarray,
    weights: np.ndarray,
    atoms: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_weight = weights[atoms]
    mass = float(np.sum(local_weight))
    mean = np.sum(local_weight[:, None] * vectors[atoms], axis=0) / max(mass, 1e-30)
    centered = vectors[atoms] - mean
    covariance = (centered.T * local_weight) @ centered / max(mass, 1e-30)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    for column in range(3):
        pivot = int(np.argmax(np.abs(eigenvectors[:, column])))
        if eigenvectors[pivot, column] < 0.0:
            eigenvectors[:, column] *= -1.0
    if np.linalg.det(eigenvectors) < 0.0:
        eigenvectors[:, -1] *= -1.0
    frame = _balanced_variance_frame(eigenvalues, eigenvectors)
    return mass, mean, covariance, eigenvalues, frame, eigenvectors[:, 0]


def _balanced_variance_frame(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Schur--Horn frame with exactly equal variance in all three channels.

    The uniform diagonal ``trace(C)/3`` is majorized by the eigenvalue vector,
    so it is feasible.  This construction attains zero channel-load variance,
    the global lower bound of that nonnegative objective.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)
    basis = np.asarray(eigenvectors, dtype=np.float64)
    mean = float(np.sum(values) / 3.0)
    span = float(values[0] - values[2])
    if span <= 1e-15:
        return basis.copy()
    cosine2 = np.clip((mean - values[2]) / span, 0.0, 1.0)
    cosine, sine = math.sqrt(cosine2), math.sqrt(1.0 - cosine2)
    first = cosine * basis[:, 0] + sine * basis[:, 2]
    w1 = basis[:, 1]
    w2 = -sine * basis[:, 0] + cosine * basis[:, 2]
    complement = np.column_stack((w1, w2))
    restricted = complement.T @ (basis @ np.diag(values) @ basis.T) @ complement
    _, rotation = np.linalg.eigh(restricted)
    a, b = complement @ rotation[:, 1], complement @ rotation[:, 0]
    second = (a + b) / math.sqrt(2.0)
    third = (a - b) / math.sqrt(2.0)
    frame = np.column_stack((first, second, third))
    if np.linalg.det(frame) < 0.0:
        frame[:, -1] *= -1.0
    return frame


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    slot = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(values[order[min(slot, len(order) - 1)]])


def build_ownership_tree(
    vectors: np.ndarray,
    weights: np.ndarray,
    config: BifurcationConfig,
) -> list[OwnershipNode]:
    nodes: list[OwnershipNode] = []

    def build(atoms: np.ndarray, parent: int, depth: int) -> int:
        mass, mean, covariance, eigenvalues, frame, direction = _weighted_statistics(
            vectors, weights, atoms
        )
        projection = (vectors[atoms] - mean) @ direction
        boundary = _weighted_median(projection, weights[atoms])
        index = len(nodes)
        node = OwnershipNode(
            index=index,
            parent=parent,
            depth=depth,
            atoms=atoms,
            mass=mass,
            mean=mean,
            covariance=covariance,
            eigenvalues=eigenvalues,
            frame=frame,
            direction=direction,
            boundary=boundary,
        )
        nodes.append(node)
        condition = eigenvalues[0] / max(eigenvalues[-1], 1e-12)
        unstable = condition > config.maximum_condition
        may_split = (
            depth < config.maximum_depth
            and len(atoms) >= 2 * config.minimum_atoms
            and unstable
        )
        if may_split:
            minus = atoms[projection <= boundary]
            plus = atoms[projection > boundary]
            if len(minus) < config.minimum_atoms or len(plus) < config.minimum_atoms:
                order = np.argsort(projection, kind="stable")
                cut = len(atoms) // 2
                minus, plus = atoms[order[:cut]], atoms[order[cut:]]
            left = build(minus, index, depth + 1)
            right = build(plus, index, depth + 1)
            node.children = (left, right)
        return index

    build(np.arange(len(vectors), dtype=np.int32), -1, 0)
    return nodes


def _constituent_quantize(
    vectors: np.ndarray,
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Quantize three routed constituents whose prequantized sum is invariant."""
    coordinates = vectors @ frame
    constituents = np.stack(
        [coordinates[:, channel, None] * frame[:, channel][None, :]
         for channel in range(3)],
        axis=1,
    )
    composition = np.sum(constituents, axis=1)
    composition_error = float(np.max(np.abs(composition - vectors)))
    quantized = np.sum(np.rint(constituents), axis=1)
    energy = np.sum(constituents * constituents, axis=(0, 2))
    return quantized, energy, composition_error


def optimize_tree_pruning(
    vectors: np.ndarray,
    weights: np.ndarray,
    nodes: list[OwnershipNode],
    config: BifurcationConfig,
) -> tuple[float, float]:
    for node in reversed(nodes):
        quantized, _, _ = _constituent_quantize(
            vectors[node.atoms], node.frame
        )
        direct = np.rint(vectors[node.atoms])
        routed_difference = vectors[node.atoms] - quantized
        direct_difference = vectors[node.atoms] - direct
        routed_cost = float(np.sum(
            weights[node.atoms, None]
            * 0.5 * routed_difference * routed_difference
            + config.rate_lambda * np.abs(quantized)
        ))
        direct_cost = float(np.sum(
            weights[node.atoms, None]
            * 0.5 * direct_difference * direct_difference
            + config.rate_lambda * np.abs(direct)
        ))
        node.use_constituent_route = routed_cost < direct_cost
        node.stop_cost = min(routed_cost, direct_cost)
        node.optimum_cost = node.stop_cost
        node.selected_stop = True
        if node.children is not None:
            split_cost = (
                config.branch_penalty
                + nodes[node.children[0]].optimum_cost
                + nodes[node.children[1]].optimum_cost
            )
            if split_cost < node.stop_cost:
                node.optimum_cost = split_cost
                node.selected_stop = False
    return nodes[0].optimum_cost, nodes[0].stop_cost


def _selected_leaves(nodes: list[OwnershipNode]) -> list[int]:
    result: list[int] = []
    stack = [0]
    while stack:
        index = stack.pop()
        node = nodes[index]
        if node.selected_stop or node.children is None:
            result.append(index)
        else:
            stack.extend(reversed(node.children))
    return result


def bifurcate_coefficients(
    coefficients: np.ndarray,
    block_labels: np.ndarray,
    quality: int,
    config: BifurcationConfig = BifurcationConfig(),
) -> OwnershipBifurcationResult:
    source = np.asarray(coefficients, dtype=np.float64)
    blocks = len(source)
    gauge, phase_parent = phase_predecessor_forest(
        source, block_labels, cross_region_weight=config.cross_region_weight
    )
    quantizers = np.stack((
        quality_table(quality, False).reshape(64),
        quality_table(quality, True).reshape(64),
        quality_table(quality, True).reshape(64),
    ), axis=-1)
    aligned = source[:, 1:, :] * gauge[:, 1:, None] / quantizers[None, 1:, :]
    vectors = aligned.reshape(-1, 3)
    weights = np.clip(np.linalg.norm(vectors, axis=1), 0.05, 20.0)
    nodes = build_ownership_tree(vectors, weights, config)
    objective, control = optimize_tree_pruning(vectors, weights, nodes, config)
    leaves = _selected_leaves(nodes)
    leaf_of_atom = np.full(len(vectors), -1, dtype=np.int32)
    relaxed_vectors = vectors.copy()
    channel_energy = np.zeros(3, dtype=np.float64)
    composition_error = 0.0
    for leaf in leaves:
        node = nodes[leaf]
        leaf_of_atom[node.atoms] = leaf
        quantized, energy, local_error = _constituent_quantize(
            vectors[node.atoms], node.frame
        )
        channel_energy += energy
        composition_error = max(composition_error, local_error)
        if node.use_constituent_route:
            relaxed_vectors[node.atoms] = quantized
        else:
            relaxed_vectors[node.atoms] = np.rint(vectors[node.atoms])
    if np.any(leaf_of_atom < 0):
        raise RuntimeError("an information atom lost ownership during unrelaxation")
    restored = relaxed_vectors.reshape(blocks, 63, 3)
    output = source.copy()
    output[:, 1:, :] = (
        restored * quantizers[None, 1:, :] * gauge[:, 1:, None]
    )
    phase_hash = hashlib.sha256(
        np.ascontiguousarray(phase_parent).view(np.uint8)
    ).hexdigest()
    ownership_hash = hashlib.sha256(
        np.ascontiguousarray(leaf_of_atom).view(np.uint8)
    ).hexdigest()
    energy_total = max(float(np.sum(channel_energy)), 1e-30)
    direct_quantized = np.rint(vectors)
    return OwnershipBifurcationResult(
        coefficients=output,
        leaf_of_atom=leaf_of_atom,
        gauge=gauge,
        phase_parent=phase_parent,
        nodes=nodes,
        selected_leaves=np.asarray(leaves, dtype=np.int32),
        objective=objective,
        stop_control_cost=control,
        leaf_count=len(leaves),
        maximum_depth=max(nodes[index].depth for index in leaves),
        phase_forest_hash=phase_hash,
        ownership_hash=ownership_hash,
        channel_energy_fraction=channel_energy / energy_total,
        prequantization_max_composition_error=composition_error,
        changed_quantized_coefficients=int(np.count_nonzero(
            relaxed_vectors != direct_quantized
        )),
        routed_leaf_count=sum(nodes[index].use_constituent_route for index in leaves),
        config=config,
    )
