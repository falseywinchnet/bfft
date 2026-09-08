"""Hierarchical diffuse optical transport without enumerated light paths.

The module is deliberately small enough to audit.  Geometry compilation turns
surface patches into a sum of nonnegative rank-one transport blocks.  A block
maps outgoing RGB power from a source cluster to incident RGB power on a
receiver cluster.  Repeated application of that one operator produces every
bounce depth; paths are never represented as objects.

This first experiment is diffuse-only.  A directional basis can later replace
the RGB state without changing the hierarchy or residual-march interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


Array = np.ndarray


def _vector3(value: Array | Iterable[float], name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result


@dataclass(frozen=True)
class Patch:
    """One oriented diffuse surface cell represented at its centroid.

    ``area`` and ``radius`` are physical.  Radius bounds the patch about its
    centroid and is used only for conservative hierarchy/occlusion decisions.
    The RGB fields are linear-radiometric values, never display-encoded color.
    """

    center: Array
    normal: Array
    area: float
    albedo: Array
    emission: Array
    radius: float | None = None
    object_id: int = -1

    def __post_init__(self) -> None:
        center = _vector3(self.center, "patch center")
        normal = _vector3(self.normal, "patch normal")
        length = float(np.linalg.norm(normal))
        if length <= 0.0:
            raise ValueError("patch normal must be nonzero")
        area = float(self.area)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("patch area must be positive")
        albedo = _vector3(self.albedo, "patch albedo")
        emission = _vector3(self.emission, "patch emission")
        if np.any((albedo < 0.0) | (albedo >= 1.0)):
            raise ValueError("diffuse albedo must lie in [0, 1)")
        if np.any(emission < 0.0):
            raise ValueError("emission must be nonnegative")
        radius = (
            math.sqrt(area / math.pi)
            if self.radius is None
            else float(self.radius)
        )
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("patch radius must be finite and nonnegative")
        object_id = int(self.object_id)
        if object_id < -1:
            raise ValueError("patch object id must be -1 or nonnegative")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "normal", normal / length)
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "albedo", albedo)
        object.__setattr__(self, "emission", emission)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "object_id", object_id)


@dataclass(frozen=True)
class SphereOccluder:
    center: Array
    radius: float
    object_id: int = -1

    def __post_init__(self) -> None:
        center = _vector3(self.center, "occluder center")
        radius = float(self.radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("occluder radius must be positive")
        object_id = int(self.object_id)
        if object_id < -1:
            raise ValueError("occluder object id must be -1 or nonnegative")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "object_id", object_id)


@dataclass(frozen=True)
class _Cluster:
    indices: Array
    center: Array
    radius: float
    area: float
    lower: Array
    upper: Array
    object_ids: frozenset[int]
    left: "_Cluster | None" = None
    right: "_Cluster | None" = None

    @property
    def leaf(self) -> bool:
        return self.left is None


@dataclass(frozen=True)
class TransportBlock:
    """A nonnegative rank-one block ``A[dst, src] = v[dst] u[src]``."""

    source: Array
    receiver: Array
    source_factor: Array
    receiver_factor: Array
    kind: str

    def __post_init__(self) -> None:
        source = np.asarray(self.source, dtype=np.int64)
        receiver = np.asarray(self.receiver, dtype=np.int64)
        u = np.asarray(self.source_factor, dtype=np.float64)
        v = np.asarray(self.receiver_factor, dtype=np.float64)
        if source.ndim != 1 or receiver.ndim != 1:
            raise ValueError("block indices must be one-dimensional")
        if u.shape != source.shape or v.shape != receiver.shape:
            raise ValueError("block factors must align with their indices")
        if np.any(u < 0.0) or np.any(v < 0.0):
            raise ValueError("transport factors must be nonnegative")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "receiver", receiver)
        object.__setattr__(self, "source_factor", u)
        object.__setattr__(self, "receiver_factor", v)


@dataclass(frozen=True)
class HierarchicalTransportField:
    """A conservative diffuse transport operator stored as low-rank blocks."""

    node_count: int
    blocks: tuple[TransportBlock, ...]
    outgoing_fraction: Array
    diagnostics: dict[str, int | float]

    def __post_init__(self) -> None:
        fraction = np.asarray(self.outgoing_fraction, dtype=np.float64)
        if fraction.shape != (self.node_count,):
            raise ValueError("one outgoing fraction is required per node")
        tolerance = 64.0 * np.finfo(float).eps
        if np.any(fraction < -tolerance) or np.any(fraction > 1.0 + tolerance):
            worst = float(np.max(fraction, initial=0.0))
            worst_index = int(np.argmax(fraction))
            raise ValueError(
                "centroid form factors exceed one outgoing energy budget "
                f"(node {worst_index}, maximum {worst:.6g}); "
                "refine/integrate the patches"
            )
        object.__setattr__(self, "outgoing_fraction", np.clip(fraction, 0.0, 1.0))

    @property
    def escape_fraction(self) -> Array:
        return 1.0 - self.outgoing_fraction

    @property
    def stored_coefficients(self) -> int:
        return int(sum(
            block.source.size + block.receiver.size for block in self.blocks
        ))

    def apply(self, outgoing: Array, active: Array | None = None) -> Array:
        """Gather, transfer, and scatter one complete bounce in parallel form."""
        value = np.asarray(outgoing, dtype=np.float64)
        squeeze = value.ndim == 1
        if squeeze:
            value = value[:, None]
        if value.ndim != 2 or value.shape[0] != self.node_count:
            raise ValueError("outgoing state must have shape N or NxC")
        if not np.all(np.isfinite(value)):
            raise ValueError("outgoing state must be finite")
        if active is None:
            active_mask = None
        else:
            active_mask = np.asarray(active, dtype=bool)
            if active_mask.shape != (self.node_count,):
                raise ValueError("active mask must have shape N")
        incoming = np.zeros_like(value)
        for block in self.blocks:
            source_value = value[block.source]
            if active_mask is not None:
                selected = active_mask[block.source]
                if not np.any(selected):
                    continue
                source_value = source_value * selected[:, None]
            gathered = np.sum(
                block.source_factor[:, None] * source_value, axis=0
            )
            incoming[block.receiver] += (
                block.receiver_factor[:, None] * gathered[None, :]
            )
        return incoming[:, 0] if squeeze else incoming

    def dense_matrix(self) -> Array:
        """Materialize the operator for small controls only."""
        matrix = np.zeros((self.node_count, self.node_count), dtype=np.float64)
        for block in self.blocks:
            matrix[np.ix_(block.receiver, block.source)] += np.outer(
                block.receiver_factor, block.source_factor
            )
        return matrix


@dataclass(frozen=True)
class SolveResult:
    outgoing: Array
    incident: Array
    absorbed: Array
    escaped: Array
    residual: Array
    depth_count: int
    active_counts: tuple[int, ...]
    conservation_error: Array


def _build_cluster(
    indices: Array,
    centers: Array,
    radii: Array,
    areas: Array,
    object_ids: Array | None = None,
) -> _Cluster:
    selected = np.asarray(indices, dtype=np.int64)
    if object_ids is None:
        object_ids = np.full(centers.shape[0], -1, dtype=np.int64)
    weights = areas[selected]
    center = np.sum(centers[selected] * weights[:, None], axis=0) / np.sum(weights)
    radius = float(np.max(
        np.linalg.norm(centers[selected] - center[None, :], axis=1)
        + radii[selected]
    ))
    if selected.size == 1:
        point = centers[selected[0]].copy()
        return _Cluster(
            selected,
            center,
            radius,
            float(weights[0]),
            point,
            point,
            frozenset((int(object_ids[selected[0]]),)),
        )
    extent = np.ptp(centers[selected], axis=0)
    axis = int(np.argmax(extent))
    order = selected[np.argsort(centers[selected, axis], kind="stable")]
    middle = order.size // 2
    left = _build_cluster(order[:middle], centers, radii, areas, object_ids)
    right = _build_cluster(order[middle:], centers, radii, areas, object_ids)
    return _Cluster(
        selected,
        center,
        radius,
        float(np.sum(weights)),
        np.minimum(left.lower, right.lower),
        np.maximum(left.upper, right.upper),
        left.object_ids | right.object_ids,
        left,
        right,
    )


def _point_swept_box_distance(
    point: Array,
    first_lower: Array,
    first_upper: Array,
    second_lower: Array,
    second_upper: Array,
) -> float:
    """Exact distance to the convex hull of two axis-aligned boxes.

    At interpolation coordinate ``t``, the Minkowski interpolation of the two
    boxes is again a box with linearly interpolated bounds.  Point-to-box
    distance is piecewise quadratic in ``t``.  Its only regime changes occur
    when one interpolated bound crosses one point coordinate, so the finite
    breakpoint scan below obtains the exact minimum without sampling rays.
    """
    lower_delta = second_lower - first_lower
    upper_delta = second_upper - first_upper
    breaks = [0.0, 1.0]
    for axis in range(3):
        for origin, delta in (
            (first_lower[axis], lower_delta[axis]),
            (first_upper[axis], upper_delta[axis]),
        ):
            if abs(float(delta)) <= np.finfo(float).tiny:
                continue
            crossing = float((point[axis] - origin) / delta)
            if 0.0 < crossing < 1.0:
                breaks.append(crossing)
    breaks = sorted(set(breaks))

    def coefficients(t: float) -> tuple[Array, Array]:
        lower = first_lower + t * lower_delta
        upper = first_upper + t * upper_delta
        slope = np.zeros(3, dtype=np.float64)
        offset = np.zeros(3, dtype=np.float64)
        below = point < lower
        above = point > upper
        slope[below] = lower_delta[below]
        offset[below] = first_lower[below] - point[below]
        slope[above] = -upper_delta[above]
        offset[above] = point[above] - first_upper[above]
        return slope, offset

    best = math.inf
    for left, right in zip(breaks[:-1], breaks[1:]):
        middle = 0.5 * (left + right)
        slope, offset = coefficients(middle)
        denominator = float(np.dot(slope, slope))
        candidates = [left, right]
        if denominator > 0.0:
            stationary = -float(np.dot(slope, offset)) / denominator
            candidates.append(min(max(stationary, left), right))
        for t in candidates:
            delta = slope * t + offset
            best = min(best, float(np.dot(delta, delta)))
    return math.sqrt(max(best, 0.0))


def _certified_clear(
    first: _Cluster,
    second: _Cluster,
    occluders: tuple[SphereOccluder, ...],
) -> bool:
    """Conservatively certify all centroid rays with a swept-box beam hull.

    Every segment joining a source centroid to a receiver centroid lies in the
    convex hull of their two centroid bounding boxes.  The helper computes the
    exact point distance to that convex hull as a one-dimensional piecewise
    quadratic minimization.  Distance to this outer beam is a lower bound on
    distance to every represented segment.
    """
    for obstacle in occluders:
        if obstacle.object_id >= 0:
            first_endpoint = first.object_ids == frozenset((obstacle.object_id,))
            second_endpoint = second.object_ids == frozenset((obstacle.object_id,))
            if first_endpoint != second_endpoint:
                # The represented rays begin or end on this convex object; it
                # is not an intervening blocker.  Same-object cluster pairs
                # remain uncertain and refine because their chord is hidden.
                continue
        if _point_swept_box_distance(
            obstacle.center,
            first.lower,
            first.upper,
            second.lower,
            second.upper,
        ) <= obstacle.radius:
            return False
    return True


def _leaf_visible(
    first: Array,
    second: Array,
    occluders: tuple[SphereOccluder, ...],
    source_object_id: int = -1,
    receiver_object_id: int = -1,
) -> bool:
    direction = second - first
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 0.0:
        return False
    for obstacle in occluders:
        if obstacle.object_id >= 0:
            source_endpoint = obstacle.object_id == source_object_id
            receiver_endpoint = obstacle.object_id == receiver_object_id
            if source_endpoint != receiver_endpoint:
                continue
        parameter = float(np.dot(obstacle.center - first, direction) / length_squared)
        if parameter <= 1.0e-12 or parameter >= 1.0 - 1.0e-12:
            continue
        closest = first + parameter * direction
        if np.dot(obstacle.center - closest, obstacle.center - closest) <= (
            obstacle.radius * obstacle.radius
        ):
            return False
    return True


def _directed_factors(
    source: _Cluster,
    receiver: _Cluster,
    centers: Array,
    normals: Array,
    areas: Array,
) -> tuple[Array, Array] | None:
    direction = receiver.center - source.center
    distance = float(np.linalg.norm(direction))
    if distance <= 0.0:
        return None
    omega = direction / distance
    source_cosine = normals[source.indices] @ omega
    receiver_cosine = normals[receiver.indices] @ (-omega)
    # Mixed front/back membership is an angular discontinuity and must refine.
    if np.any(source_cosine <= 0.0) or np.any(receiver_cosine <= 0.0):
        return None
    u = source_cosine
    v = areas[receiver.indices] * receiver_cosine / (math.pi * distance * distance)
    return u, v


def _common_normal(indices: Array, normals: Array) -> Array | None:
    candidate = normals[int(indices[0])]
    if np.max(np.linalg.norm(normals[indices] - candidate[None, :], axis=1)) <= 1.0e-12:
        return candidate
    return None


def _planar_self_exchange_is_empty(
    cluster: _Cluster,
    centers: Array,
    normals: Array,
) -> bool:
    """Certify that one coplanar, equally oriented cluster cannot see itself."""
    normal = _common_normal(cluster.indices, normals)
    if normal is None:
        return False
    projection = centers[cluster.indices] @ normal
    scale = max(float(np.max(np.abs(projection), initial=0.0)), 1.0)
    return float(np.ptp(projection)) <= 32.0 * np.finfo(float).eps * scale


def _cluster_pair_exchange_is_empty(
    first: _Cluster,
    second: _Cluster,
    centers: Array,
    normals: Array,
) -> bool:
    """Use exact planar support intervals to reject an entire hemisphere pair."""
    first_normal = _common_normal(first.indices, normals)
    if first_normal is not None:
        maximum_departure = (
            float(np.max(centers[second.indices] @ first_normal))
            - float(np.min(centers[first.indices] @ first_normal))
        )
        if maximum_departure <= 0.0:
            return True
    second_normal = _common_normal(second.indices, normals)
    if second_normal is not None:
        maximum_return = (
            float(np.max(centers[first.indices] @ second_normal))
            - float(np.min(centers[second.indices] @ second_normal))
        )
        if maximum_return <= 0.0:
            return True
    return False


def _leaf_link(
    source_index: int,
    receiver_index: int,
    centers: Array,
    normals: Array,
    areas: Array,
) -> TransportBlock | None:
    direction = centers[receiver_index] - centers[source_index]
    distance = float(np.linalg.norm(direction))
    if distance <= 0.0:
        return None
    omega = direction / distance
    outgoing_cosine = float(np.dot(normals[source_index], omega))
    incoming_cosine = float(np.dot(normals[receiver_index], -omega))
    if outgoing_cosine <= 0.0 or incoming_cosine <= 0.0:
        return None
    # The differential-area kernel is singular when finite adjacent cells meet
    # at a corner.  The symmetric area term is a reciprocal finite-footprint
    # regularization: it converges to the ordinary centroid kernel in the far
    # field and preserves A_i F_ij = A_j F_ji exactly in the near field.
    denominator = (
        math.pi * distance * distance
        + 2.0 * (areas[source_index] + areas[receiver_index])
    )
    fraction = (
        areas[receiver_index]
        * outgoing_cosine
        * incoming_cosine
        / denominator
    )
    return TransportBlock(
        source=np.array((source_index,), dtype=np.int64),
        receiver=np.array((receiver_index,), dtype=np.int64),
        source_factor=np.array((1.0,), dtype=np.float64),
        receiver_factor=np.array((fraction,), dtype=np.float64),
        kind="leaf",
    )


def _field_from_blocks(
    node_count: int,
    blocks: list[TransportBlock],
    diagnostics: dict[str, int | float],
) -> HierarchicalTransportField:
    outgoing = np.zeros(node_count, dtype=np.float64)
    for block in blocks:
        outgoing[block.source] += (
            block.source_factor * float(np.sum(block.receiver_factor))
        )
    diagnostics = dict(diagnostics)
    diagnostics["block_count"] = len(blocks)
    diagnostics["stored_coefficients"] = int(sum(
        block.source.size + block.receiver.size for block in blocks
    ))
    diagnostics["dense_coefficients"] = int(node_count * node_count)
    return HierarchicalTransportField(
        node_count=node_count,
        blocks=tuple(blocks),
        outgoing_fraction=outgoing,
        diagnostics=diagnostics,
    )


def compile_hierarchical_field(
    patches: Iterable[Patch],
    *,
    occluders: Iterable[SphereOccluder] = (),
    admissibility: float = 0.35,
    minimum_fraction: float = 0.0,
    transport_scale: float = 1.0,
) -> HierarchicalTransportField:
    """Compile patches with dual-tree refinement around uncertain geometry.

    A cluster pair is aggregated only when it is geometrically separated,
    every member faces the representative direction, and a conservative
    swept-box beam hull proves all its centroid rays clear of every sphere. Any
    uncertainty recursively bifurcates the larger cluster.  Leaves receive an
    exact centroid segment test.  Thus occlusion boundaries refine while open
    far fields remain rank one.

    This is a research compiler, not yet a general mesh visibility engine.
    Its certified-clear sphere test is intentionally stricter than sampling.
    """
    surface = tuple(patches)
    blockers = tuple(occluders)
    if len(surface) < 2:
        raise ValueError("transport compilation needs at least two patches")
    eta = float(admissibility)
    threshold = float(minimum_fraction)
    scale = float(transport_scale)
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("admissibility must be positive")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum fraction must be nonnegative")
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("transport scale must lie in (0, 1]")

    centers = np.stack([patch.center for patch in surface])
    normals = np.stack([patch.normal for patch in surface])
    areas = np.array([patch.area for patch in surface], dtype=np.float64)
    radii = np.array([patch.radius for patch in surface], dtype=np.float64)
    object_ids = np.array(
        [patch.object_id for patch in surface], dtype=np.int64
    )
    root = _build_cluster(
        np.arange(len(surface)), centers, radii, areas, object_ids
    )
    blocks: list[TransportBlock] = []
    stats = {
        "cluster_pair_visits": 0,
        "coherent_pair_count": 0,
        "leaf_pair_count": 0,
        "exact_visibility_tests": 0,
        "occlusion_refinements": 0,
        "angular_refinements": 0,
        "hemisphere_pair_prunes": 0,
        "bound_prunes": 0,
        "occluded_leaf_pairs": 0,
    }

    def add_direction(source: _Cluster, receiver: _Cluster, kind: str) -> bool:
        center_distance = float(np.linalg.norm(receiver.center - source.center))
        minimum_distance = max(
            center_distance - source.radius - receiver.radius, 1.0e-15
        )
        bound = (
            scale
            * receiver.area
            / (math.pi * minimum_distance * minimum_distance)
        )
        if bound < threshold:
            stats["bound_prunes"] += 1
            return True
        factors = _directed_factors(
            source, receiver, centers, normals, areas
        )
        if factors is None:
            return False
        u, v = factors
        blocks.append(TransportBlock(
            source=source.indices,
            receiver=receiver.indices,
            source_factor=u,
            receiver_factor=scale * v,
            kind=kind,
        ))
        return True

    def visit(first: _Cluster, second: _Cluster) -> None:
        stats["cluster_pair_visits"] += 1
        if first is second:
            if _planar_self_exchange_is_empty(first, centers, normals):
                stats["hemisphere_pair_prunes"] += 1
                return
            if first.leaf:
                return
            assert first.left is not None and first.right is not None
            visit(first.left, first.left)
            visit(first.left, first.right)
            visit(first.right, first.right)
            return


        if _cluster_pair_exchange_is_empty(first, second, centers, normals):
            stats["hemisphere_pair_prunes"] += 1
            return

        direction = second.center - first.center
        distance = float(np.linalg.norm(direction))
        separated = (
            distance > 0.0
            and max(first.radius, second.radius) <= eta * distance
        )
        clear = separated and _certified_clear(first, second, blockers)
        if clear:
            before = len(blocks)
            first_ok = add_direction(first, second, "aggregate")
            second_ok = add_direction(second, first, "aggregate")
            if first_ok and second_ok:
                if len(blocks) > before:
                    stats["coherent_pair_count"] += 1
                return
            del blocks[before:]
            stats["angular_refinements"] += 1
        elif separated and blockers:
            stats["occlusion_refinements"] += 1

        if first.leaf and second.leaf:
            stats["leaf_pair_count"] += 1
            stats["exact_visibility_tests"] += 1
            i = int(first.indices[0])
            j = int(second.indices[0])
            if not _leaf_visible(
                centers[i],
                centers[j],
                blockers,
                int(object_ids[i]),
                int(object_ids[j]),
            ):
                stats["occluded_leaf_pairs"] += 1
                return
            forward = _leaf_link(i, j, centers, normals, areas)
            reverse = _leaf_link(j, i, centers, normals, areas)
            if (
                forward is not None
                and scale * forward.receiver_factor[0] >= threshold
            ):
                blocks.append(TransportBlock(
                    forward.source,
                    forward.receiver,
                    forward.source_factor,
                    scale * forward.receiver_factor,
                    forward.kind,
                ))
            if (
                reverse is not None
                and scale * reverse.receiver_factor[0] >= threshold
            ):
                blocks.append(TransportBlock(
                    reverse.source,
                    reverse.receiver,
                    reverse.source_factor,
                    scale * reverse.receiver_factor,
                    reverse.kind,
                ))
            return


        split_first = not first.leaf and (
            second.leaf or first.radius >= second.radius
        )
        if split_first:
            assert first.left is not None and first.right is not None
            visit(first.left, second)
            visit(first.right, second)
        else:
            assert second.left is not None and second.right is not None
            visit(first, second.left)
            visit(first, second.right)

    visit(root, root)
    stats["node_count"] = len(surface)
    stats["transport_scale"] = scale
    return _field_from_blocks(len(surface), blocks, stats)


def compile_dense_centroid_field(
    patches: Iterable[Patch],
    *,
    occluders: Iterable[SphereOccluder] = (),
    minimum_fraction: float = 0.0,
) -> HierarchicalTransportField:
    """Quadratic leaf-pair compiler used only as a correctness oracle."""
    surface = tuple(patches)
    blockers = tuple(occluders)
    if len(surface) < 2:
        raise ValueError("transport compilation needs at least two patches")
    centers = np.stack([patch.center for patch in surface])
    normals = np.stack([patch.normal for patch in surface])
    areas = np.array([patch.area for patch in surface], dtype=np.float64)
    object_ids = np.array(
        [patch.object_id for patch in surface], dtype=np.int64
    )
    blocks: list[TransportBlock] = []
    visible_pairs = 0
    for i in range(len(surface)):
        for j in range(i + 1, len(surface)):
            if not _leaf_visible(
                centers[i],
                centers[j],
                blockers,
                int(object_ids[i]),
                int(object_ids[j]),
            ):
                continue
            visible_pairs += 1
            forward = _leaf_link(i, j, centers, normals, areas)
            reverse = _leaf_link(j, i, centers, normals, areas)
            if forward is not None and forward.receiver_factor[0] >= minimum_fraction:
                blocks.append(forward)
            if reverse is not None and reverse.receiver_factor[0] >= minimum_fraction:
                blocks.append(reverse)
    return _field_from_blocks(len(surface), blocks, {
        "node_count": len(surface),
        "cluster_pair_visits": len(surface) * (len(surface) - 1) // 2,
        "leaf_pair_count": len(surface) * (len(surface) - 1) // 2,
        "exact_visibility_tests": len(surface) * (len(surface) - 1) // 2,
        "visible_leaf_pairs": visible_pairs,
        "coherent_pair_count": 0,
    })


def solve_transport(
    field: HierarchicalTransportField,
    emission: Array,
    albedo: Array,
    *,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-14,
    maximum_depth: int = 10000,
) -> SolveResult:
    """March residual light until no node can materially feed another.

    Each iteration is one gather--transfer--scatter over the fixed field.
    Nodes below the joint absolute/relative frontier threshold are removed
    from the active mask.  Cycles require no special path handling: their
    attenuated contributions revisit this same residual march until exhausted.
    """
    source = np.asarray(emission, dtype=np.float64)
    reflectance = np.asarray(albedo, dtype=np.float64)
    if source.ndim == 1:
        source = source[:, None]
    if reflectance.ndim == 1:
        reflectance = reflectance[:, None]
    if source.ndim != 2 or source.shape[0] != field.node_count:
        raise ValueError("emission must have shape N or NxC")
    if reflectance.shape not in (source.shape, (field.node_count, 1)):
        raise ValueError("albedo must broadcast over the emission channels")
    if np.any(source < 0.0) or not np.all(np.isfinite(source)):
        raise ValueError("emission must be finite and nonnegative")
    if (
        np.any((reflectance < 0.0) | (reflectance >= 1.0))
        or not np.all(np.isfinite(reflectance))
    ):
        raise ValueError("albedo must be finite and lie in [0, 1)")
    relative = float(relative_tolerance)
    absolute = float(absolute_tolerance)
    if relative < 0.0 or absolute < 0.0:
        raise ValueError("solver tolerances must be nonnegative")
    if maximum_depth < 1:
        raise ValueError("maximum depth must be positive")

    frontier = source.copy()
    outgoing = source.copy()
    incident = np.zeros_like(source)
    absorbed = np.zeros_like(source)
    escaped = np.zeros_like(source)
    initial_scale = np.max(source, axis=0)
    threshold = absolute + relative * initial_scale
    active_counts: list[int] = []
    depth_count = 0
    for depth in range(maximum_depth):
        active = np.any(frontier > threshold[None, :], axis=1)
        active_counts.append(int(np.count_nonzero(active)))
        if not np.any(active):
            break
        # Sub-threshold light is retained as the explicit residual rather than
        # silently disappearing from the energy ledger.
        marching = frontier * active[:, None]
        frontier = frontier - marching
        escaped += field.escape_fraction[:, None] * marching
        received = field.apply(marching, active=active)
        incident += received
        loss = (1.0 - reflectance) * received
        reflected = reflectance * received
        absorbed += loss
        outgoing += reflected
        frontier += reflected
        depth_count = depth + 1
    else:
        raise RuntimeError("transport residual did not converge within maximum_depth")

    residual = frontier
    initial_energy = np.sum(source, axis=0)
    accounted = (
        np.sum(absorbed, axis=0)
        + np.sum(escaped, axis=0)
        + np.sum(residual, axis=0)
    )
    conservation_error = accounted - initial_energy
    return SolveResult(
        outgoing=outgoing,
        incident=incident,
        absorbed=absorbed,
        escaped=escaped,
        residual=residual,
        depth_count=depth_count,
        active_counts=tuple(active_counts),
        conservation_error=conservation_error,
    )


def dense_fixed_point(field: HierarchicalTransportField, emission: Array, albedo: Array) -> Array:
    """Solve the materialized linear system for small test scenes."""
    source = np.asarray(emission, dtype=np.float64)
    reflectance = np.asarray(albedo, dtype=np.float64)
    if source.ndim == 1:
        source = source[:, None]
    if reflectance.ndim == 1:
        reflectance = reflectance[:, None]
    operator = field.dense_matrix()
    result = np.empty_like(source)
    identity = np.eye(field.node_count)
    for channel in range(source.shape[1]):
        rho = reflectance[:, min(channel, reflectance.shape[1] - 1)]
        result[:, channel] = np.linalg.solve(
            identity - rho[:, None] * operator,
            source[:, channel],
        )
    return result
