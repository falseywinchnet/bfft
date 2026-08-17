"""Error-priority, mass-balanced regions for DCT ownership transport.

This adapts the PNG-to-SVG quotient-tree discipline to JPEG blocks:

* one immutable root owns the image initially;
* every proposal is a spatial half-plane split inside one parent;
* the boundary is an exact weighted median, so transported demand bifurcates
  evenly;
* proposals compete globally by exact reduction in standardized block-feature
  SSE; and
* children retain explicit parent lineage.

Unlike the former connected signature threshold, the number and minimum size
of ownership cells are controlled and isolated one-block regions cannot flood
the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .certified_relaxation import _coefficients
from .core import quality_table


@dataclass(frozen=True)
class BalancedRegionConfig:
    target_regions: int = 256
    minimum_blocks: int = 24
    demand_strength: float = 0.35
    angular_candidates: int = 12
    even_depth: bool = True


@dataclass
class RegionNode:
    index: int
    parent: int
    depth: int
    blocks: np.ndarray
    mass: float
    feature_sse: float
    split_gain: float = 0.0
    direction: np.ndarray | None = None
    boundary: float = 0.0
    children: tuple[int, int] | None = None


@dataclass
class BalancedRegionResult:
    labels: np.ndarray
    nodes: list[RegionNode]
    leaves: np.ndarray
    block_features: np.ndarray
    demand: np.ndarray
    config: BalancedRegionConfig

    def report(self) -> dict:
        sizes = np.bincount(self.labels.ravel())
        masses = np.bincount(
            self.labels.ravel(), weights=self.demand, minlength=len(sizes)
        )
        return {
            "regions": len(sizes),
            "minimum_blocks": int(np.min(sizes)),
            "median_blocks": float(np.median(sizes)),
            "maximum_blocks": int(np.max(sizes)),
            "block_size_cv": float(np.std(sizes) / np.mean(sizes)),
            "mass_cv": float(np.std(masses) / np.mean(masses)),
            "tree_nodes": len(self.nodes),
            "maximum_depth": int(max(self.nodes[i].depth for i in self.leaves)),
            "config": self.config.__dict__,
            "proof_boundary": (
                "Each accepted split is the best exact feature-SSE reduction "
                "among all current leaf proposals, and each proposal uses a "
                "mass-balanced boundary. This is not a global optimum over "
                "all possible planar partitions."
            ),
        }


def _block_features(ycc: np.ndarray, quality: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    coefficients = _coefficients(ycc)
    blocks = len(coefficients)
    height = (ycc.shape[0] + 7) // 8
    width = (ycc.shape[1] + 7) // 8
    quantizers = np.stack((
        quality_table(quality, False).reshape(64),
        quality_table(quality, True).reshape(64),
        quality_table(quality, True).reshape(64),
    ), axis=-1)
    normalized = coefficients / quantizers[None, :, :]
    u, v = np.mgrid[:8, :8]
    radius = np.sqrt(u * u + v * v).reshape(64)
    bands = ((radius > 0) & (radius <= 2.25),
             (radius > 2.25) & (radius <= 5.0),
             radius > 5.0)
    features = [normalized[:, 0, :]]
    for mask in bands:
        energy = np.mean(normalized[:, mask, :] ** 2, axis=1)
        features.append(np.log1p(energy))
    # Signed low-order phase is needed in addition to energy: two cells with
    # equal spectra but opposing structure should not share a permeable owner.
    features.append(np.tanh(normalized[:, 1:10, :]).reshape(blocks, -1))
    feature = np.concatenate(features, axis=1)
    median = np.median(feature, axis=0)
    scale = np.median(np.abs(feature - median), axis=0) * 1.4826 + 1e-6
    feature = np.clip((feature - median) / scale, -8.0, 8.0)
    entropy_demand = np.sum(np.log1p(np.abs(normalized[:, 1:, :])), axis=(1, 2))
    demand_scale = np.median(entropy_demand) + 1e-9
    demand = 1.0 + entropy_demand / demand_scale
    return feature, demand, (height, width)


def _weighted_sse(features: np.ndarray, weight: np.ndarray) -> float:
    mass = max(float(np.sum(weight)), 1e-30)
    mean = np.sum(weight[:, None] * features, axis=0) / mass
    return float(np.sum(weight[:, None] * (features - mean) ** 2))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    slot = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(values[order[min(slot, len(order) - 1)]])


def balanced_bifurcation_regions(
    ycc: np.ndarray,
    quality: int,
    config: BalancedRegionConfig = BalancedRegionConfig(),
) -> BalancedRegionResult:
    features, raw_demand, shape = _block_features(ycc, quality)
    demand = 1.0 + config.demand_strength * (raw_demand - 1.0)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    coordinates = np.column_stack((
        (xx.ravel() + 0.5) / max(shape[1], 1),
        (yy.ravel() + 0.5) / max(shape[0], 1),
    ))
    nodes: list[RegionNode] = []
    leaves: set[int] = set()
    heap: list[tuple[int, float, int, int, np.ndarray, float, np.ndarray, np.ndarray]] = []
    serial = 0

    def add_node(blocks: np.ndarray, parent: int, depth: int) -> int:
        nonlocal serial
        local_weight = demand[blocks]
        node = RegionNode(
            index=len(nodes), parent=parent, depth=depth, blocks=blocks,
            mass=float(np.sum(local_weight)),
            feature_sse=_weighted_sse(features[blocks], local_weight),
        )
        nodes.append(node)
        leaves.add(node.index)
        proposal = propose(node)
        if proposal is not None:
            gain, direction, boundary, minus, plus = proposal
            priority_depth = node.depth if config.even_depth else 0
            heapq.heappush(heap, (
                priority_depth, -gain, serial, node.index,
                direction, boundary, minus, plus,
            ))
            serial += 1
        return node.index

    def propose(node: RegionNode):
        blocks = node.blocks
        minimum = max(int(config.minimum_blocks), 1)
        if len(blocks) < 2 * minimum:
            return None
        xy = coordinates[blocks]
        local_features = features[blocks]
        weight = demand[blocks]
        mass = float(np.sum(weight))
        xy_mean = np.sum(weight[:, None] * xy, axis=0) / mass
        centered_xy = xy - xy_mean
        feature_mean = np.sum(weight[:, None] * local_features, axis=0) / mass
        centered_feature = local_features - feature_mean
        cross = (centered_xy.T * weight) @ centered_feature / mass
        metric = cross @ cross.T
        covariance = (centered_xy.T * weight) @ centered_xy / mass
        _, metric_vectors = np.linalg.eigh(metric + 1e-3 * covariance)
        _, spatial_vectors = np.linalg.eigh(covariance)
        candidates = [metric_vectors[:, -1], spatial_vectors[:, -1], np.array((1.0, 0.0)), np.array((0.0, 1.0))]
        for angle in np.linspace(0.0, np.pi, max(config.angular_candidates, 2), endpoint=False):
            candidates.append(np.array((np.cos(angle), np.sin(angle))))
        best = None
        for direction in candidates:
            direction = direction / max(float(np.linalg.norm(direction)), 1e-30)
            projection = centered_xy @ direction
            boundary = _weighted_median(projection, weight)
            side = projection > boundary
            if np.count_nonzero(side) < minimum or np.count_nonzero(~side) < minimum:
                order = np.argsort(projection, kind="stable")
                cut = len(blocks) // 2
                side = np.zeros(len(blocks), dtype=bool)
                side[order[cut:]] = True
                boundary = 0.5 * (
                    projection[order[cut - 1]] + projection[order[cut]]
                )
            minus, plus = blocks[~side], blocks[side]
            if len(minus) < minimum or len(plus) < minimum:
                continue
            new_sse = (
                _weighted_sse(features[minus], demand[minus])
                + _weighted_sse(features[plus], demand[plus])
            )
            gain = node.feature_sse - new_sse
            if best is None or gain > best[0]:
                best = (gain, direction.copy(), boundary, minus, plus)
        if best is None or best[0] <= 1e-12:
            return None
        return best

    add_node(np.arange(len(features), dtype=np.int32), -1, 0)
    while heap and len(leaves) < max(int(config.target_regions), 1):
        _priority_depth, neg_gain, _serial, index, direction, boundary, minus, plus = heapq.heappop(heap)
        if index not in leaves:
            continue
        parent = nodes[index]
        leaves.remove(index)
        left = add_node(minus, index, parent.depth + 1)
        right = add_node(plus, index, parent.depth + 1)
        parent.children = (left, right)
        parent.split_gain = -neg_gain
        parent.direction = direction
        parent.boundary = boundary

    ordered_leaves = np.asarray(sorted(leaves), dtype=np.int32)
    labels = np.full(len(features), -1, dtype=np.int32)
    for label, node_index in enumerate(ordered_leaves):
        labels[nodes[node_index].blocks] = label
    if np.any(labels < 0):
        raise RuntimeError("balanced bifurcation lost block ownership")
    return BalancedRegionResult(
        labels=labels.reshape(shape), nodes=nodes, leaves=ordered_leaves,
        block_features=features, demand=demand, config=config,
    )
