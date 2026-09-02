"""Energy-gated discovery and solution of the photonic transport field.

Unlike :mod:`transport`, this module never compiles a scene-wide potential
operator.  Every residual light frontier traverses the surface hierarchy with
its current energy.  A whole unopened receiver cluster is omitted when a
conservative upper bound on the energy it can receive fits inside the shared
remaining error budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .transport import (
    Array,
    Patch,
    SphereOccluder,
    TransportBlock,
    _build_cluster,
    _certified_clear,
    _cluster_pair_exchange_is_empty,
    _directed_factors,
    _field_from_blocks,
    _leaf_link,
    _leaf_visible,
    _planar_self_exchange_is_empty,
)


@dataclass(frozen=True)
class BudgetedApplyResult:
    incoming: Array
    discarded_incident_bound: float
    diagnostics: dict[str, int | float]


@dataclass(frozen=True)
class BudgetedSolveResult:
    outgoing: Array
    incident: Array
    depth_count: int
    depth_diagnostics: tuple[dict[str, int | float], ...]
    discarded_incident_bound: float
    residual_tail_bound: float
    outgoing_error_bound: float


class BudgetedTransportGeometry:
    """Immutable geometry whose transport links are discovered on demand."""

    def __init__(
        self,
        patches: Iterable[Patch],
        *,
        occluders: Iterable[SphereOccluder] = (),
        admissibility: float = 0.35,
        transport_scale: float = 1.0,
    ) -> None:
        self.patches = tuple(patches)
        self.occluders = tuple(occluders)
        if len(self.patches) < 2:
            raise ValueError("budgeted transport needs at least two patches")
        self.admissibility = float(admissibility)
        if not math.isfinite(self.admissibility) or self.admissibility <= 0.0:
            raise ValueError("admissibility must be positive")
        self.transport_scale = float(transport_scale)
        if not 0.0 < self.transport_scale <= 1.0:
            raise ValueError("transport scale must lie in (0, 1]")
        self.centers = np.stack([patch.center for patch in self.patches])
        self.normals = np.stack([patch.normal for patch in self.patches])
        self.areas = np.array(
            [patch.area for patch in self.patches], dtype=np.float64
        )
        radii = np.array(
            [patch.radius for patch in self.patches], dtype=np.float64
        )
        self.object_ids = np.array(
            [patch.object_id for patch in self.patches], dtype=np.int64
        )
        self.root = _build_cluster(
            np.arange(len(self.patches)),
            self.centers,
            radii,
            self.areas,
            self.object_ids,
        )

    def apply(
        self,
        outgoing: Array,
        *,
        incident_error_budget: float,
    ) -> BudgetedApplyResult:
        """Apply one bounce while lazily discovering only relevant blocks.

        The budget is a global L1 incident-energy allowance for this and all
        cluster pairs visited by the call.  Pruned bounds are accumulated, so
        many individually small clusters cannot collectively exceed it.
        """
        value = np.asarray(outgoing, dtype=np.float64)
        squeeze = value.ndim == 1
        if squeeze:
            value = value[:, None]
        if value.ndim != 2 or value.shape[0] != len(self.patches):
            raise ValueError("outgoing state must have shape N or NxC")
        if np.any(value < 0.0) or not np.all(np.isfinite(value)):
            raise ValueError("outgoing state must be finite and nonnegative")
        budget = float(incident_error_budget)
        if not math.isfinite(budget) or budget < 0.0:
            raise ValueError("incident error budget must be finite and nonnegative")

        node_energy = np.sum(value, axis=1)
        blocks: list[TransportBlock] = []
        discarded = 0.0
        stats: dict[str, int | float] = {
            "cluster_pair_visits": 0,
            "energy_pruned_pairs": 0,
            "energy_pruned_bound": 0.0,
            "zero_source_prunes": 0,
            "hemisphere_pair_prunes": 0,
            "coherent_pair_count": 0,
            "leaf_pair_count": 0,
            "exact_visibility_tests": 0,
            "occlusion_refinements": 0,
            "angular_refinements": 0,
            "occluded_leaf_pairs": 0,
        }

        def consume_bound(bound: float) -> bool:
            nonlocal discarded
            candidate = max(float(bound), 0.0)
            if candidate == 0.0:
                return True
            if discarded + candidate <= budget:
                discarded += candidate
                stats["energy_pruned_pairs"] += 1
                stats["energy_pruned_bound"] = discarded
                return True
            return False

        def cluster_energy(cluster) -> float:
            return float(np.sum(node_energy[cluster.indices]))

        def coarse_delivery_bound(source, receiver) -> float:
            energy = cluster_energy(source)
            if energy == 0.0:
                return 0.0
            distance = float(np.linalg.norm(receiver.center - source.center))
            minimum_distance = max(
                distance - source.radius - receiver.radius, 1.0e-15
            )
            fraction = min(
                receiver.area / (math.pi * minimum_distance * minimum_distance),
                1.0,
            )
            return energy * fraction

        def exact_block_delivery(block: TransportBlock) -> float:
            gathered = float(np.dot(
                block.source_factor,
                node_energy[block.source],
            ))
            return gathered * float(np.sum(block.receiver_factor))

        def admit_block(block: TransportBlock) -> None:
            if self.transport_scale != 1.0:
                block = TransportBlock(
                    source=block.source,
                    receiver=block.receiver,
                    source_factor=block.source_factor,
                    receiver_factor=(
                        self.transport_scale * block.receiver_factor
                    ),
                    kind=block.kind,
                )
            if consume_bound(exact_block_delivery(block)):
                return
            blocks.append(block)

        def visit(first, second) -> None:
            stats["cluster_pair_visits"] += 1
            if first is second:
                energy = cluster_energy(first)
                if energy == 0.0:
                    stats["zero_source_prunes"] += 1
                    return
                # Internal exchange cannot deliver more than all power leaving
                # this cluster.  This bound lets a sufficiently weak complete
                # frontier terminate at the root without discovering a link.
                if consume_bound(energy):
                    return
                if _planar_self_exchange_is_empty(
                    first, self.centers, self.normals
                ):
                    stats["hemisphere_pair_prunes"] += 1
                    return
                if first.leaf:
                    return
                visit(first.left, first.left)
                visit(first.left, first.right)
                visit(first.right, first.right)
                return

            if _cluster_pair_exchange_is_empty(
                first, second, self.centers, self.normals
            ):
                stats["hemisphere_pair_prunes"] += 1
                return

            first_energy = cluster_energy(first)
            second_energy = cluster_energy(second)
            first_relevant = first_energy > 0.0
            second_relevant = second_energy > 0.0
            if first_relevant:
                first_relevant = not consume_bound(
                    coarse_delivery_bound(first, second)
                )
            else:
                stats["zero_source_prunes"] += 1
            if second_relevant:
                second_relevant = not consume_bound(
                    coarse_delivery_bound(second, first)
                )
            else:
                stats["zero_source_prunes"] += 1
            if not first_relevant and not second_relevant:
                return

            direction = second.center - first.center
            distance = float(np.linalg.norm(direction))
            separated = (
                distance > 0.0
                and max(first.radius, second.radius)
                <= self.admissibility * distance
            )
            clear = separated and _certified_clear(
                first, second, self.occluders
            )
            if clear:
                before = len(blocks)
                valid = True
                if first_relevant:
                    factors = _directed_factors(
                        first,
                        second,
                        self.centers,
                        self.normals,
                        self.areas,
                    )
                    if factors is None:
                        valid = False
                    else:
                        admit_block(TransportBlock(
                            first.indices,
                            second.indices,
                            factors[0],
                            factors[1],
                            "budgeted_aggregate",
                        ))
                if second_relevant and valid:
                    factors = _directed_factors(
                        second,
                        first,
                        self.centers,
                        self.normals,
                        self.areas,
                    )
                    if factors is None:
                        valid = False
                    else:
                        admit_block(TransportBlock(
                            second.indices,
                            first.indices,
                            factors[0],
                            factors[1],
                            "budgeted_aggregate",
                        ))
                if valid:
                    if len(blocks) > before:
                        stats["coherent_pair_count"] += 1
                    return
                del blocks[before:]
                stats["angular_refinements"] += 1
            elif separated and self.occluders:
                stats["occlusion_refinements"] += 1

            if first.leaf and second.leaf:
                stats["leaf_pair_count"] += 1
                stats["exact_visibility_tests"] += 1
                i = int(first.indices[0])
                j = int(second.indices[0])
                if not _leaf_visible(
                    self.centers[i],
                    self.centers[j],
                    self.occluders,
                    int(self.object_ids[i]),
                    int(self.object_ids[j]),
                ):
                    stats["occluded_leaf_pairs"] += 1
                    return
                if first_relevant:
                    block = _leaf_link(
                        i, j, self.centers, self.normals, self.areas
                    )
                    if block is not None:
                        admit_block(block)
                if second_relevant:
                    block = _leaf_link(
                        j, i, self.centers, self.normals, self.areas
                    )
                    if block is not None:
                        admit_block(block)
                return

            split_first = not first.leaf and (
                second.leaf or first.radius >= second.radius
            )
            if split_first:
                visit(first.left, second)
                visit(first.right, second)
            else:
                visit(first, second.left)
                visit(first, second.right)

        visit(self.root, self.root)
        field = _field_from_blocks(len(self.patches), blocks, stats)
        incoming = field.apply(value)
        stats.update({
            "active_source_nodes": int(np.count_nonzero(node_energy)),
            "frontier_energy": float(np.sum(node_energy)),
            "block_count": len(blocks),
            "stored_coefficients": field.stored_coefficients,
        })
        return BudgetedApplyResult(
            incoming=incoming[:, 0] if squeeze else incoming,
            discarded_incident_bound=discarded,
            diagnostics=stats,
        )


def solve_budgeted_transport(
    geometry: BudgetedTransportGeometry,
    emission: Array,
    albedo: Array,
    *,
    outgoing_error_budget: float,
    frontier_tolerance: float = 1.0e-14,
    depth_budget_fraction: float = 0.5,
    tail_budget_fraction: float = 0.5,
    maximum_depth: int = 10000,
) -> BudgetedSolveResult:
    """Solve while sharing one certified error budget across all depths."""
    source = np.asarray(emission, dtype=np.float64)
    reflectance = np.asarray(albedo, dtype=np.float64)
    if source.ndim == 1:
        source = source[:, None]
    if reflectance.ndim == 1:
        reflectance = reflectance[:, None]
    if source.ndim != 2 or source.shape[0] != len(geometry.patches):
        raise ValueError("emission must have shape N or NxC")
    if reflectance.shape not in (source.shape, (source.shape[0], 1)):
        raise ValueError("albedo must broadcast over emission")
    if np.any(source < 0.0) or np.any((reflectance < 0.0) | (reflectance >= 1.0)):
        raise ValueError("emission and albedo must be nonnegative; albedo must be < 1")
    requested = float(outgoing_error_budget)
    tail_tolerance = float(frontier_tolerance)
    depth_fraction = float(depth_budget_fraction)
    tail_fraction = float(tail_budget_fraction)
    if requested < 0.0 or tail_tolerance < 0.0:
        raise ValueError("error budgets must be nonnegative")
    if not 0.0 < depth_fraction < 1.0:
        raise ValueError("depth budget fraction must lie strictly between zero and one")
    if not 0.0 <= tail_fraction < 1.0:
        raise ValueError("tail budget fraction must lie in [0, 1)")
    maximum_reflectance = float(np.max(reflectance, initial=0.0))
    if maximum_reflectance == 0.0:
        return BudgetedSolveResult(
            outgoing=source.copy(),
            incident=np.zeros_like(source),
            depth_count=0,
            depth_diagnostics=(),
            discarded_incident_bound=0.0,
            residual_tail_bound=0.0,
            outgoing_error_bound=0.0,
        )
    tail_outgoing_budget = tail_fraction * requested
    transport_outgoing_budget = requested - tail_outgoing_budget
    incident_budget = transport_outgoing_budget * (
        1.0 - maximum_reflectance
    ) / maximum_reflectance
    effective_tail_tolerance = max(
        tail_tolerance,
        tail_outgoing_budget * (1.0 - maximum_reflectance),
    )
    remaining_budget = incident_budget
    frontier = source.copy()
    outgoing = source.copy()
    incident = np.zeros_like(source)
    depth_records: list[dict[str, int | float]] = []
    discarded = 0.0
    residual_tail = 0.0
    for depth in range(maximum_depth):
        frontier_energy = float(np.sum(frontier))
        if frontier_energy <= effective_tail_tolerance:
            residual_tail = frontier_energy / (1.0 - maximum_reflectance)
            break
        applied = geometry.apply(
            frontier,
            # Reserve budget for later, weaker diffusion fronts.  Spending the
            # entire allowance at the first illuminated layer would make the
            # geometry denser as energy decays—the opposite of the intended
            # transport law.  This geometric reservation still sums below the
            # one shared global certificate.
            incident_error_budget=depth_fraction * remaining_budget,
        )
        discarded += applied.discarded_incident_bound
        remaining_budget = max(incident_budget - discarded, 0.0)
        received = np.asarray(applied.incoming, dtype=np.float64)
        incident += received
        frontier = reflectance * received
        outgoing += frontier
        record = dict(applied.diagnostics)
        record.update({
            "depth": depth,
            "remaining_incident_error_budget": remaining_budget,
            "next_frontier_energy": float(np.sum(frontier)),
        })
        depth_records.append(record)
        if not np.any(frontier):
            break
    else:
        raise RuntimeError("budgeted transport did not terminate")
    outgoing_bound = (
        maximum_reflectance
        / (1.0 - maximum_reflectance)
        * discarded
        + residual_tail
    )
    return BudgetedSolveResult(
        outgoing=outgoing,
        incident=incident,
        depth_count=len(depth_records),
        depth_diagnostics=tuple(depth_records),
        discarded_incident_bound=discarded,
        residual_tail_bound=residual_tail,
        outgoing_error_bound=outgoing_bound,
    )
