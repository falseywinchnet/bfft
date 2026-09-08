"""Cached two-port child-volume response for sparse photonic transport.

The parent sees only directional ingress and egress fibers. Fine child
geometry is represented by private internal nodes. A boundary fiber is
duplicated by role so light which has reached an egress port cannot
accidentally re-enter the child through the corresponding geometric patch.

For fixed child geometry and materials the response is

    L_out = R_V[L_in] + E_V.

Scalar-link children cache one unit response per ingress/mode family. A child
with retained rank-one blocks instead batches its active ingress field through
those blocks, so gradual illumination cannot reconstruct a dense response
matrix. Replacing child topology invalidates its response state without
touching any parent-scene transport structure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np

from .light_tensor import (
    LightTensor6,
    LightTransportRegister,
    SignedLightLedger,
    diffuse_boundary,
    mueller_boundary,
    volume_transport,
)
from .transport import (
    Patch,
    SphereOccluder,
    compile_dense_centroid_field,
    compile_hierarchical_field,
)


Array = np.ndarray


def _unit_vector(value: Array | Iterable[float], name: str) -> Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    result = vector / length
    result.setflags(write=False)
    return result


def _vector3(value: Array | Iterable[float], name: str) -> Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    result = vector.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AxisAlignedVolume:
    lower: Array
    upper: Array

    def __post_init__(self) -> None:
        lower = _vector3(self.lower, "volume lower bound")
        upper = _vector3(self.upper, "volume upper bound")
        if np.any(upper <= lower):
            raise ValueError("volume upper bound must exceed its lower bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains(self, position: Array, *, tolerance: float = 1.0e-12) -> bool:
        return bool(
            np.all(position >= self.lower - tolerance)
            and np.all(position <= self.upper + tolerance)
        )

    def validates_boundary_normal(
        self,
        position: Array,
        normal: Array,
        *,
        tolerance: float = 1.0e-10,
    ) -> bool:
        for axis in range(3):
            expected = np.zeros(3)
            if abs(position[axis] - self.lower[axis]) <= tolerance:
                expected[axis] = -1.0
                if np.linalg.norm(normal - expected) <= tolerance:
                    return True
            if abs(position[axis] - self.upper[axis]) <= tolerance:
                expected[axis] = 1.0
                if np.linalg.norm(normal - expected) <= tolerance:
                    return True
        return False


@dataclass(frozen=True)
class BoundaryFiber:
    """One spatial/angular cell on the invisible child boundary."""

    position: Array
    outward_normal: Array
    direction: Array
    area: float
    solid_angle: float

    def __post_init__(self) -> None:
        position = _vector3(self.position, "boundary-fiber position")
        normal = _unit_vector(self.outward_normal, "boundary-fiber normal")
        direction = _unit_vector(self.direction, "boundary-fiber direction")
        area = float(self.area)
        solid_angle = float(self.solid_angle)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("boundary-fiber area must be positive")
        if not math.isfinite(solid_angle) or not 0.0 < solid_angle <= 4.0 * math.pi:
            raise ValueError("boundary-fiber solid angle must lie in (0, 4pi]")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "outward_normal", normal)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "solid_angle", solid_angle)


@dataclass(frozen=True)
class VolumeMedium:
    """Mode-dependent coefficients for the unified extinction exponent."""

    density: float = 0.0
    extinction: float = 0.0
    linear_diffusive_extinction: float = 0.0
    circular_diffusive_extinction: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.density,
            self.extinction,
            self.linear_diffusive_extinction,
            self.circular_diffusive_extinction,
        )
        if not all(math.isfinite(float(value)) and value >= 0.0 for value in values):
            raise ValueError("volume-medium coefficients must be finite and nonnegative")

    def transport(self, packet: LightTensor6, distance: float) -> LightTensor6:
        return volume_transport(
            packet,
            distance=distance,
            density=self.density,
            extinction=self.extinction,
            linear_diffusive_extinction=self.linear_diffusive_extinction,
            circular_diffusive_extinction=self.circular_diffusive_extinction,
        )


@dataclass(frozen=True)
class InternalKernel:
    """Local extinction/generation law for one hidden internal surface."""

    kind: str = "identity"
    throughput: float = 1.0
    mueller_matrix: Array | None = None
    allow_gain: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"identity", "diffuse", "mueller"}:
            raise ValueError("internal kernel kind must be identity, diffuse, or mueller")
        throughput = float(self.throughput)
        if not math.isfinite(throughput) or throughput < 0.0:
            raise ValueError("internal-kernel throughput must be finite and nonnegative")
        if not self.allow_gain and throughput > 1.0:
            raise ValueError("passive internal-kernel throughput cannot exceed one")
        matrix = self.mueller_matrix
        if self.kind == "mueller":
            if matrix is None:
                raise ValueError("Mueller internal kernel requires a matrix")
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError("Mueller internal-kernel matrix must be finite and 4x4")
            matrix = matrix.copy()
            matrix.setflags(write=False)
        elif matrix is not None:
            raise ValueError("only a Mueller internal kernel may carry a matrix")
        object.__setattr__(self, "throughput", throughput)
        object.__setattr__(self, "mueller_matrix", matrix)

    def apply(self, ledger: SignedLightLedger) -> SignedLightLedger:
        output: list[LightTensor6] = []
        for packet in ledger.packets:
            if self.kind == "identity":
                generated = packet.scaled(self.throughput)
            elif self.kind == "diffuse":
                generated = diffuse_boundary(packet, self.throughput)
            else:
                assert self.mueller_matrix is not None
                generated = mueller_boundary(packet, self.mueller_matrix).outgoing
                generated = generated.scaled(self.throughput)
                if (
                    not self.allow_gain
                    and abs(generated.power) > abs(packet.power) * (1.0 + 1.0e-12)
                ):
                    raise ValueError("passive Mueller internal kernel generated gain")
            output.append(generated)
        return SignedLightLedger.from_packets(output)


@dataclass(frozen=True)
class InternalNode:
    position: Array
    kernel: InternalKernel = InternalKernel()
    emission: SignedLightLedger = SignedLightLedger()

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector3(self.position, "internal-node position"))


@dataclass(frozen=True)
class ChildSurface:
    """Hidden oriented facet used only by the child geometry compiler."""

    center: Array
    normal: Array
    area: float
    kernel: InternalKernel = InternalKernel()
    emission: SignedLightLedger = SignedLightLedger()
    radius: float | None = None
    object_id: int = -1

    def __post_init__(self) -> None:
        center = _vector3(self.center, "child-surface center")
        normal = _unit_vector(self.normal, "child-surface normal")
        area = float(self.area)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("child-surface area must be positive")
        radius = math.sqrt(area / math.pi) if self.radius is None else float(self.radius)
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("child-surface radius must be finite and nonnegative")
        if self.object_id < -1:
            raise ValueError("child-surface object id must be -1 or nonnegative")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "radius", radius)

    def internal_node(self) -> InternalNode:
        return InternalNode(self.center, self.kernel, self.emission)

    def patch(self) -> Patch:
        return Patch(
            center=self.center,
            normal=self.normal,
            area=self.area,
            radius=self.radius,
            albedo=np.zeros(3),
            emission=np.zeros(3),
            object_id=self.object_id,
        )


@dataclass(frozen=True)
class NodeAddress:
    role: str
    index: int

    def __post_init__(self) -> None:
        if self.role not in {"ingress", "internal", "egress"}:
            raise ValueError("child node role must be ingress, internal, or egress")
        if self.index < 0:
            raise ValueError("child node index must be nonnegative")


def ingress(index: int) -> NodeAddress:
    return NodeAddress("ingress", index)


def internal(index: int) -> NodeAddress:
    return NodeAddress("internal", index)


def egress(index: int) -> NodeAddress:
    return NodeAddress("egress", index)


@dataclass(frozen=True)
class ChildLink:
    source: NodeAddress
    receiver: NodeAddress
    geometric_gain: float
    distance: float
    medium: VolumeMedium = VolumeMedium()
    geometry_label: int = 0

    def __post_init__(self) -> None:
        gain = float(self.geometric_gain)
        distance = float(self.distance)
        if not math.isfinite(gain) or not 0.0 <= gain <= 1.0:
            raise ValueError("child-link geometric gain must lie in [0, 1]")
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("child-link distance must be finite and nonnegative")
        if self.geometry_label < 0:
            raise ValueError("child-link geometry label must be nonnegative")
        if self.source.role == "egress":
            raise ValueError("egress fibers are terminal and cannot be link sources")
        if self.receiver.role == "ingress":
            raise ValueError("ingress fibers cannot receive child transport")
        object.__setattr__(self, "geometric_gain", gain)
        object.__setattr__(self, "distance", distance)

    def transport(self, ledger: SignedLightLedger) -> SignedLightLedger:
        return SignedLightLedger.from_packets(
            self.medium.transport(packet, self.distance).scaled(self.geometric_gain)
            for packet in ledger.packets
        )


@dataclass(frozen=True)
class ChildTransportBlock:
    """A retained role-filtered rank-one child operator ``v (u^T L)``."""

    sources: tuple[NodeAddress, ...]
    receivers: tuple[NodeAddress, ...]
    source_factor: Array
    receiver_factor: Array
    distance: float
    medium: VolumeMedium = VolumeMedium()
    geometry_label: int = 0
    kind: str = "retained"

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        receivers = tuple(self.receivers)
        source_factor = np.asarray(self.source_factor, dtype=np.float64)
        receiver_factor = np.asarray(self.receiver_factor, dtype=np.float64)
        if not sources or not receivers:
            raise ValueError("retained child block must have sources and receivers")
        if source_factor.shape != (len(sources),):
            raise ValueError("retained child-block source factors do not align")
        if receiver_factor.shape != (len(receivers),):
            raise ValueError("retained child-block receiver factors do not align")
        if (
            not np.all(np.isfinite(source_factor))
            or not np.all(np.isfinite(receiver_factor))
            or np.any(source_factor < 0.0)
            or np.any(receiver_factor < 0.0)
        ):
            raise ValueError("retained child-block factors must be finite and nonnegative")
        if any(source.role == "egress" for source in sources):
            raise ValueError("egress fibers cannot source a retained child block")
        if any(receiver.role == "ingress" for receiver in receivers):
            raise ValueError("ingress fibers cannot receive a retained child block")
        if len(set(sources)) != len(sources) or len(set(receivers)) != len(receivers):
            raise ValueError("retained child-block addresses must be unique")
        distance = float(self.distance)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("retained child-block distance must be nonnegative")
        if self.geometry_label < 0:
            raise ValueError("retained child-block geometry label must be nonnegative")
        source_factor = source_factor.copy()
        receiver_factor = receiver_factor.copy()
        source_factor.setflags(write=False)
        receiver_factor.setflags(write=False)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "receivers", receivers)
        object.__setattr__(self, "source_factor", source_factor)
        object.__setattr__(self, "receiver_factor", receiver_factor)
        object.__setattr__(self, "distance", distance)

    @property
    def stored_coefficients(self) -> int:
        return len(self.sources) + len(self.receivers)

    @property
    def expanded_coefficients(self) -> int:
        return len(self.sources) * len(self.receivers)


@dataclass(frozen=True)
class ChildVolumeTopology:
    bounds: AxisAlignedVolume
    ingress_fibers: tuple[BoundaryFiber, ...]
    egress_fibers: tuple[BoundaryFiber, ...]
    internal_nodes: tuple[InternalNode, ...]
    links: tuple[ChildLink, ...]
    revision: int = 0
    blocks: tuple[ChildTransportBlock, ...] = ()

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("child-volume revision must be nonnegative")
        for fiber in self.ingress_fibers:
            self._validate_fiber(fiber, incoming=True)
        for fiber in self.egress_fibers:
            self._validate_fiber(fiber, incoming=False)
        for node in self.internal_nodes:
            if not self.bounds.contains(node.position):
                raise ValueError("internal node lies outside its child volume")
        identities: set[tuple[NodeAddress, NodeAddress, int]] = set()
        outgoing_gain: dict[NodeAddress, float] = {}
        for link in self.links:
            self._validate_address(link.source)
            self._validate_address(link.receiver)
            identity = (link.source, link.receiver, link.geometry_label)
            if identity in identities:
                raise ValueError("child-link identity must be unique")
            identities.add(identity)
            outgoing_gain[link.source] = outgoing_gain.get(link.source, 0.0) + link.geometric_gain
        block_labels: set[int] = set()
        for block in self.blocks:
            if block.geometry_label in block_labels:
                raise ValueError("retained child-block geometry label must be unique")
            block_labels.add(block.geometry_label)
            for source in block.sources:
                self._validate_address(source)
            for receiver in block.receivers:
                self._validate_address(receiver)
            receiver_sum = float(np.sum(block.receiver_factor))
            for source, factor in zip(block.sources, block.source_factor):
                outgoing_gain[source] = outgoing_gain.get(source, 0.0) + float(
                    factor * receiver_sum
                )
        for source, gain in outgoing_gain.items():
            if gain > 1.0 + 64.0 * np.finfo(float).eps:
                raise ValueError(
                    f"outgoing child transport gain exceeds one at {source}"
                )

    @property
    def retained_stored_coefficients(self) -> int:
        return sum(block.stored_coefficients for block in self.blocks)

    @property
    def retained_expanded_coefficients(self) -> int:
        return sum(block.expanded_coefficients for block in self.blocks)

    def _validate_fiber(self, fiber: BoundaryFiber, *, incoming: bool) -> None:
        if not self.bounds.validates_boundary_normal(fiber.position, fiber.outward_normal):
            raise ValueError("boundary fiber is not on the declared box face")
        orientation = float(np.dot(fiber.direction, fiber.outward_normal))
        if incoming and orientation >= -1.0e-12:
            raise ValueError("ingress-fiber direction must point into the volume")
        if not incoming and orientation <= 1.0e-12:
            raise ValueError("egress-fiber direction must point out of the volume")

    def _validate_address(self, address: NodeAddress) -> None:
        counts = {
            "ingress": len(self.ingress_fibers),
            "internal": len(self.internal_nodes),
            "egress": len(self.egress_fibers),
        }
        if address.index >= counts[address.role]:
            raise ValueError("child-link address lies outside its role partition")


@dataclass(frozen=True)
class CompiledChildVolume:
    topology: ChildVolumeTopology
    geometry_diagnostics: dict[str, int | float]


def _child_compilation_patches(
    incoming: tuple[BoundaryFiber, ...],
    outgoing: tuple[BoundaryFiber, ...],
    hidden: tuple[ChildSurface, ...],
) -> tuple[Patch, ...]:
    boundary_material = np.zeros(3)
    boundary_emission = np.zeros(3)

    def proxy(fiber: BoundaryFiber, normal: Array, object_id: int) -> Patch:
        return Patch(
            center=fiber.position,
            normal=normal,
            area=fiber.area,
            radius=math.sqrt(fiber.area / math.pi),
            albedo=boundary_material,
            emission=boundary_emission,
            object_id=object_id,
        )

    return (
        tuple(
            proxy(fiber, fiber.direction, 1_000_000 + index)
            for index, fiber in enumerate(incoming)
        )
        + tuple(surface.patch() for surface in hidden)
        + tuple(
            proxy(fiber, -fiber.direction, 2_000_000 + index)
            for index, fiber in enumerate(outgoing)
        )
    )


def _child_address(
    global_index: int,
    ingress_count: int,
    internal_count: int,
) -> NodeAddress:
    if global_index < ingress_count:
        return ingress(global_index)
    if global_index < ingress_count + internal_count:
        return internal(global_index - ingress_count)
    return egress(global_index - ingress_count - internal_count)


def compile_child_volume_topology(
    bounds: AxisAlignedVolume,
    ingress_fibers: Iterable[BoundaryFiber],
    egress_fibers: Iterable[BoundaryFiber],
    surfaces: Iterable[ChildSurface],
    *,
    occluders: Iterable[SphereOccluder] = (),
    medium: VolumeMedium = VolumeMedium(),
    coefficient_cutoff: float = 0.0,
    revision: int = 0,
) -> CompiledChildVolume:
    """Discover a child's legal links with the repository geometry compiler.

    Boundary proxies exist only during child compilation. Ingress proxies face
    along their inward directions; egress proxies face back into the child so
    they can receive outward-moving transport. The resulting dense oracle is
    immediately reduced to legal role-directed sparse links.
    """
    incoming = tuple(ingress_fibers)
    outgoing = tuple(egress_fibers)
    hidden = tuple(surfaces)
    if coefficient_cutoff < 0.0:
        raise ValueError("child coefficient cutoff must be nonnegative")
    patches = _child_compilation_patches(incoming, outgoing, hidden)
    field = compile_dense_centroid_field(patches, occluders=tuple(occluders))
    matrix = field.dense_matrix()
    ingress_count = len(incoming)
    internal_count = len(hidden)

    links: list[ChildLink] = []
    for receiver_index in range(len(patches)):
        receiver = _child_address(receiver_index, ingress_count, internal_count)
        if receiver.role == "ingress":
            continue
        for source_index in range(len(patches)):
            source = _child_address(source_index, ingress_count, internal_count)
            if source.role == "egress":
                continue
            gain = float(matrix[receiver_index, source_index])
            if gain <= coefficient_cutoff:
                continue
            distance = float(
                np.linalg.norm(patches[receiver_index].center - patches[source_index].center)
            )
            links.append(
                ChildLink(
                    source,
                    receiver,
                    gain,
                    distance,
                    medium,
                    receiver_index * len(patches) + source_index,
                )
            )
    topology = ChildVolumeTopology(
        bounds,
        incoming,
        outgoing,
        tuple(surface.internal_node() for surface in hidden),
        tuple(links),
        revision,
    )
    return CompiledChildVolume(topology, dict(field.diagnostics))


def compile_child_volume_rank_blocks(
    bounds: AxisAlignedVolume,
    ingress_fibers: Iterable[BoundaryFiber],
    egress_fibers: Iterable[BoundaryFiber],
    surfaces: Iterable[ChildSurface],
    *,
    occluders: Iterable[SphereOccluder] = (),
    medium: VolumeMedium = VolumeMedium(),
    admissibility: float = 0.35,
    minimum_fraction: float = 0.0,
    revision: int = 0,
) -> CompiledChildVolume:
    """Compile and retain role-filtered hierarchical rank-one blocks."""
    incoming = tuple(ingress_fibers)
    outgoing = tuple(egress_fibers)
    hidden = tuple(surfaces)
    patches = _child_compilation_patches(incoming, outgoing, hidden)
    field = compile_hierarchical_field(
        patches,
        occluders=tuple(occluders),
        admissibility=admissibility,
        minimum_fraction=minimum_fraction,
    )
    ingress_count = len(incoming)
    internal_count = len(hidden)
    patch_areas = np.array([patch.area for patch in patches], dtype=np.float64)
    blocks: list[ChildTransportBlock] = []
    for block_index, block in enumerate(field.blocks):
        source_addresses = tuple(
            _child_address(int(index), ingress_count, internal_count)
            for index in block.source
        )
        receiver_addresses = tuple(
            _child_address(int(index), ingress_count, internal_count)
            for index in block.receiver
        )
        source_mask = np.array(
            [address.role != "egress" for address in source_addresses],
            dtype=bool,
        )
        receiver_mask = np.array(
            [address.role != "ingress" for address in receiver_addresses],
            dtype=bool,
        )
        if not np.any(source_mask) or not np.any(receiver_mask):
            continue
        source_weights = patch_areas[block.source]
        receiver_weights = patch_areas[block.receiver]
        source_center = np.average(
            np.stack([patches[int(index)].center for index in block.source]),
            axis=0,
            weights=source_weights,
        )
        receiver_center = np.average(
            np.stack([patches[int(index)].center for index in block.receiver]),
            axis=0,
            weights=receiver_weights,
        )
        blocks.append(
            ChildTransportBlock(
                tuple(
                    address
                    for address, keep in zip(source_addresses, source_mask)
                    if keep
                ),
                tuple(
                    address
                    for address, keep in zip(receiver_addresses, receiver_mask)
                    if keep
                ),
                block.source_factor[source_mask],
                block.receiver_factor[receiver_mask],
                float(np.linalg.norm(receiver_center - source_center)),
                medium,
                block_index,
                block.kind,
            )
        )
    topology = ChildVolumeTopology(
        bounds,
        incoming,
        outgoing,
        tuple(surface.internal_node() for surface in hidden),
        (),
        revision,
        tuple(blocks),
    )
    diagnostics = dict(field.diagnostics)
    diagnostics.update(
        {
            "role_filtered_retained_blocks": len(blocks),
            "role_filtered_stored_coefficients": topology.retained_stored_coefficients,
            "role_filtered_expanded_coefficients": topology.retained_expanded_coefficients,
        }
    )
    return CompiledChildVolume(topology, diagnostics)


@dataclass(frozen=True)
class CompressedBoundaryField:
    """Exact mode-coalesced outgoing ledgers, one per parent-visible fiber."""

    ledgers: tuple[SignedLightLedger, ...]

    @property
    def active_ports(self) -> tuple[int, ...]:
        return tuple(index for index, ledger in enumerate(self.ledgers) if ledger.packets)

    @property
    def packet_count(self) -> int:
        return sum(len(ledger.packets) for ledger in self.ledgers)

    @property
    def signed_power(self) -> float:
        return sum(ledger.signed_power for ledger in self.ledgers)

    def scatter_to_parent(
        self,
        output_to_parent_node: Mapping[int, int],
    ) -> dict[int, SignedLightLedger]:
        """Gather child egress fibers into parent transport-node emissions."""
        parent: dict[int, SignedLightLedger] = {}
        for port, ledger in enumerate(self.ledgers):
            if not ledger.packets:
                continue
            if port not in output_to_parent_node:
                raise ValueError("active child egress fiber has no parent mapping")
            node = int(output_to_parent_node[port])
            if node < 0:
                raise ValueError("parent transport-node index must be nonnegative")
            parent[node] = parent.get(node, SignedLightLedger()).plus(ledger)
        return {
            node: ledger.coalesced()
            for node, ledger in parent.items()
        }


@dataclass(frozen=True)
class PackedBoundaryField:
    """Mode-major terminal field which postpones allocation of packet objects."""

    modes: tuple[LightTensor6, ...]
    powers: Array
    execution_backend: str
    block_applications: int
    gathered_source_coefficients: int
    scattered_receiver_coefficients: int

    def __post_init__(self) -> None:
        powers = np.asarray(self.powers, dtype=np.float64)
        if powers.ndim != 2 or powers.shape[0] != len(self.modes):
            raise ValueError("packed boundary powers must have shape modes by ports")
        if not np.all(np.isfinite(powers)):
            raise ValueError("packed boundary powers must be finite")
        powers = np.ascontiguousarray(powers)
        powers.setflags(write=False)
        object.__setattr__(self, "powers", powers)

    @property
    def port_count(self) -> int:
        return self.powers.shape[1]

    @property
    def mode_count(self) -> int:
        return self.powers.shape[0]

    def materialize(self, cutoff: float = 0.0) -> CompressedBoundaryField:
        """Create packet ledgers only at a consumer boundary that needs them."""
        ledgers = []
        for port in range(self.port_count):
            packets = []
            for mode, template in enumerate(self.modes):
                power = float(self.powers[mode, port])
                if abs(power) <= cutoff:
                    continue
                packets.append(
                    LightTensor6(
                        power,
                        template.centre_nm,
                        template.variance_nm2,
                        template.orientation_degrees,
                        template.linear_peakedness,
                        template.chirality,
                        template.provenance,
                    )
                )
            ledgers.append(SignedLightLedger.from_packets(packets))
        return CompressedBoundaryField(tuple(ledgers))


@dataclass(frozen=True)
class ChildVolumeSolveResult:
    outgoing: CompressedBoundaryField
    cache_hits: int
    cache_misses: int
    kernel_marches: int
    edge_applications: int
    residual_events: int
    discarded_absolute_power: float
    topology_revision: int
    block_applications: int = 0
    gathered_source_coefficients: int = 0
    scattered_receiver_coefficients: int = 0
    evaluation_strategy: str = "unit-response-cache"
    execution_backend: str = "python"
    native_wave_calls: int = 0
    contiguous_source_blocks: int = 0
    contiguous_receiver_blocks: int = 0


@dataclass(frozen=True)
class _MarchResult:
    outputs: tuple[SignedLightLedger, ...]
    edge_applications: int
    residual_events: int
    discarded_absolute_power: float
    block_applications: int = 0
    gathered_source_coefficients: int = 0
    scattered_receiver_coefficients: int = 0
    native_wave_calls: int = 0
    contiguous_source_blocks: int = 0
    contiguous_receiver_blocks: int = 0


def _mode_key(packet: LightTensor6) -> tuple[float | int, ...]:
    return (
        packet.centre_nm,
        packet.variance_nm2,
        packet.orientation_degrees,
        packet.linear_peakedness,
        packet.chirality,
        packet.provenance,
    )


class ChildVolumeResponseSystem:
    """Evaluate the private sparse response of one invisible child volume."""

    def __init__(
        self,
        topology: ChildVolumeTopology,
        *,
        absolute_cutoff: float = 1.0e-12,
        maximum_events: int = 1_000_000,
        cache_strategy: str = "auto",
        execution_backend: str = "auto",
    ) -> None:
        if absolute_cutoff < 0.0 or maximum_events < 1:
            raise ValueError("invalid child-volume residual configuration")
        self._topology = topology
        self.absolute_cutoff = float(absolute_cutoff)
        self.maximum_events = int(maximum_events)
        if cache_strategy not in {"auto", "retained"}:
            raise ValueError("child cache strategy must be auto or retained")
        if execution_backend not in {"auto", "python", "native"}:
            raise ValueError("child execution backend must be auto, python, or native")
        self._requested_strategy = cache_strategy
        self._requested_backend = execution_backend
        self._evaluation_strategy = "unit"
        self._execution_backend = "python"
        self._native_plan = None
        self._bound_packed_signature = None
        self._bound_packed_plan = None
        self._packed_terminal = False
        self._outgoing: dict[NodeAddress, list[ChildLink]] = {}
        self._blocks_by_source: dict[NodeAddress, list[int]] = {}
        self._compile_links()
        self._mode_responses: dict[
            tuple[int, tuple[float | int, ...]], tuple[SignedLightLedger, ...]
        ] = {}
        self._emission_response: _MarchResult | None = None

    @property
    def cached_mode_response_count(self) -> int:
        return len(self._mode_responses)

    @property
    def topology(self) -> ChildVolumeTopology:
        return self._topology

    @property
    def evaluation_strategy(self) -> str:
        return self._evaluation_strategy

    @property
    def execution_backend(self) -> str:
        return self._execution_backend

    def _address_index(self, address: NodeAddress) -> int:
        ingress_count = len(self.topology.ingress_fibers)
        internal_count = len(self.topology.internal_nodes)
        if address.role == "ingress":
            return address.index
        if address.role == "internal":
            return ingress_count + address.index
        return ingress_count + internal_count + address.index

    def _index_address(self, index: int) -> NodeAddress:
        return _child_address(
            index,
            len(self.topology.ingress_fibers),
            len(self.topology.internal_nodes),
        )

    def _compile_native_plan(self) -> None:
        if self._bound_packed_plan is not None:
            self._bound_packed_plan.close()
        self._bound_packed_signature = None
        self._bound_packed_plan = None
        self._native_plan = None
        self._execution_backend = "python"
        if not self.topology.blocks or self._requested_backend == "python":
            return
        try:
            from .native_retained import NativeRetainedPlan

            specifications = []
            for block in self.topology.blocks:
                medium = block.medium
                specifications.append(
                    (
                        np.array(
                            [self._address_index(source) for source in block.sources],
                            dtype=np.uint32,
                        ),
                        block.source_factor,
                        np.array(
                            [
                                self._address_index(receiver)
                                for receiver in block.receivers
                            ],
                            dtype=np.uint32,
                        ),
                        block.receiver_factor,
                        block.distance * medium.density,
                        medium.extinction,
                        medium.linear_diffusive_extinction,
                        medium.circular_diffusive_extinction,
                    )
                )
            node_count = (
                len(self.topology.ingress_fibers)
                + len(self.topology.internal_nodes)
                + len(self.topology.egress_fibers)
            )
            self._native_plan = NativeRetainedPlan(node_count, specifications)
            self._execution_backend = self._native_plan.backend
        except (OSError, RuntimeError, ValueError):
            if self._requested_backend == "native":
                raise

    def _compile_links(self) -> None:
        self._outgoing = {}
        for link in self.topology.links:
            self._outgoing.setdefault(link.source, []).append(link)
        self._blocks_by_source = {}
        for block_index, block in enumerate(self.topology.blocks):
            for source in block.sources:
                self._blocks_by_source.setdefault(source, []).append(block_index)
        if self.topology.blocks or self._requested_strategy == "retained":
            self._evaluation_strategy = "retained"
        else:
            self._evaluation_strategy = "unit"
        self._packed_terminal = (
            not self.topology.internal_nodes
            and not self.topology.links
            and all(
                source.role == "ingress"
                for block in self.topology.blocks
                for source in block.sources
            )
            and all(
                receiver.role == "egress"
                for block in self.topology.blocks
                for receiver in block.receivers
            )
        )
        self._compile_native_plan()

    def replace_topology(self, topology: ChildVolumeTopology) -> None:
        """Install changed child geometry/materials and invalidate only this child."""
        self._topology = topology
        self._compile_links()
        self._mode_responses.clear()
        self._emission_response = None

    def invalidate_ingress(self, ports: Iterable[int]) -> int:
        """Invalidate unit responses for selected scalar-link ingress fibers."""
        selected = set(int(port) for port in ports)
        if any(port < 0 or port >= len(self.topology.ingress_fibers) for port in selected):
            raise ValueError("invalid ingress port for child-cache invalidation")
        stale = [key for key in self._mode_responses if key[0] in selected]
        for key in stale:
            del self._mode_responses[key]
        return len(stale)

    def _apply_receiver(
        self,
        receiver: NodeAddress,
        ledger: SignedLightLedger,
    ) -> SignedLightLedger:
        if receiver.role == "internal":
            return self.topology.internal_nodes[receiver.index].kernel.apply(ledger)
        return ledger

    def _native_block_wave(
        self,
        frontier: Mapping[NodeAddress, SignedLightLedger],
    ) -> tuple[dict[NodeAddress, SignedLightLedger], object]:
        """Apply every retained block once while keeping modes as SIMD channels."""
        assert self._native_plan is not None
        templates: dict[tuple[float | int, ...], LightTensor6] = {}
        for ledger in frontier.values():
            for packet in ledger.packets:
                templates.setdefault(_mode_key(packet), packet)
        keys = sorted(templates)
        node_count = self._native_plan.node_count
        powers = np.zeros((len(keys), node_count), dtype=np.float64)
        key_index = {key: index for index, key in enumerate(keys)}
        active = np.zeros(node_count, dtype=np.uint8)
        for address, ledger in frontier.items():
            node = self._address_index(address)
            active[node] = 1
            for packet in ledger.packets:
                powers[key_index[_mode_key(packet)], node] = packet.power
        linear = np.array(
            [templates[key].linear_peakedness for key in keys], dtype=np.float64
        )
        circular = np.array(
            [templates[key].chirality for key in keys], dtype=np.float64
        )
        transported, stats = self._native_plan.apply(
            powers,
            active,
            linear,
            circular,
        )
        delivered: dict[NodeAddress, SignedLightLedger] = {}
        occupied = np.flatnonzero(np.any(transported != 0.0, axis=0))
        for raw_node in occupied:
            node = int(raw_node)
            packets = []
            for mode, key in enumerate(keys):
                power = float(transported[mode, node])
                if power == 0.0:
                    continue
                template = templates[key]
                packets.append(
                    LightTensor6(
                        power,
                        template.centre_nm,
                        template.variance_nm2,
                        template.orientation_degrees,
                        template.linear_peakedness,
                        template.chirality,
                        template.provenance,
                    )
                )
            delivered[self._index_address(node)] = SignedLightLedger.from_packets(
                packets
            )
        return delivered, stats

    def evaluate_packed(
        self,
        modes: Iterable[LightTensor6],
        ingress_powers: Array,
    ) -> PackedBoundaryField:
        """Evaluate a direct terminal child field without materializing ledgers.

        This is the renderer-facing path: a fixed set of optical modes remains
        packed while the camera or parent field consumes its boundary result.
        General children containing hidden surfaces continue through
        :meth:`evaluate`, where local kernels may create new modes.
        """
        mode = tuple(modes)
        powers = np.ascontiguousarray(ingress_powers, dtype=np.float64)
        ingress_count = len(self.topology.ingress_fibers)
        internal_count = len(self.topology.internal_nodes)
        egress_count = len(self.topology.egress_fibers)
        if not self._packed_terminal:
            raise ValueError("packed evaluation requires direct ingress-egress blocks")
        if powers.shape != (len(mode), ingress_count):
            raise ValueError("packed ingress powers must have shape modes by ingress")
        if not np.all(np.isfinite(powers)):
            raise ValueError("packed ingress powers must be finite")
        if len({_mode_key(packet) for packet in mode}) != len(mode):
            raise ValueError("packed optical modes must be unique")
        if self._native_plan is None:
            if self._requested_backend == "native":
                raise RuntimeError("native retained execution plan is unavailable")
            # The packed fallback uses exactly the same rank algebra and never
            # expands a block. It is an oracle/fallback, not the fast path.
            output = np.zeros((len(mode), egress_count), dtype=np.float64)
            block_applications = gathered = scattered = 0
            for block in self.topology.blocks:
                source_index = np.array(
                    [source.index for source in block.sources], dtype=np.int64
                )
                value = np.sum(
                    powers[:, source_index] * block.source_factor[None, :],
                    axis=1,
                )
                if not np.any(value != 0.0):
                    continue
                retention = np.exp(
                    -block.distance
                    * block.medium.density
                    * np.array(
                        [
                            block.medium.extinction
                            + block.medium.linear_diffusive_extinction
                            * packet.linear_peakedness
                            + block.medium.circular_diffusive_extinction
                            * abs(packet.chirality)
                            for packet in mode
                        ]
                    )
                )
                receiver_index = np.array(
                    [receiver.index for receiver in block.receivers], dtype=np.int64
                )
                output[:, receiver_index] += (
                    value * retention
                )[:, None] * block.receiver_factor[None, :]
                block_applications += 1
                gathered += len(block.sources)
                scattered += len(block.receivers)
            return PackedBoundaryField(
                mode,
                output,
                "numpy-f64",
                block_applications,
                gathered,
                scattered,
            )
        node_count = ingress_count + egress_count
        full_power = np.zeros((len(mode), node_count), dtype=np.float64)
        full_power[:, :ingress_count] = powers
        active = np.zeros(node_count, dtype=np.uint8)
        active[:ingress_count] = np.any(powers != 0.0, axis=0)
        signature = tuple(
            (packet.linear_peakedness, packet.chirality) for packet in mode
        )
        if signature != self._bound_packed_signature:
            if self._bound_packed_plan is not None:
                self._bound_packed_plan.close()
            self._bound_packed_plan = self._native_plan.bind_modes(
                np.array([packet.linear_peakedness for packet in mode]),
                np.array([packet.chirality for packet in mode]),
            )
            self._bound_packed_signature = signature
        transported, stats = self._bound_packed_plan.apply(full_power, active)
        return PackedBoundaryField(
            mode,
            transported[:, ingress_count:],
            self._execution_backend,
            stats.block_applications,
            stats.gathered_source_coefficients,
            stats.scattered_receiver_coefficients,
        )

    def _march(self, seeds: Mapping[NodeAddress, SignedLightLedger]) -> _MarchResult:
        frontier = {
            address: ledger.coalesced(self.absolute_cutoff)
            for address, ledger in seeds.items()
            if ledger.absolute_power > self.absolute_cutoff
        }
        outputs = [SignedLightLedger() for _ in self.topology.egress_fibers]
        applications = events = block_applications = 0
        gathered_coefficients = scattered_coefficients = 0
        native_wave_calls = 0
        contiguous_source_blocks = contiguous_receiver_blocks = 0
        discarded = 0.0
        while frontier:
            next_frontier: dict[NodeAddress, SignedLightLedger] = {}

            def deliver(receiver: NodeAddress, ledger: SignedLightLedger) -> None:
                nonlocal discarded
                delivered = self._apply_receiver(receiver, ledger)
                if delivered.absolute_power <= self.absolute_cutoff:
                    discarded += delivered.absolute_power
                    return
                if receiver.role == "egress":
                    port = receiver.index
                    outputs[port] = outputs[port].plus(delivered)
                else:
                    prior = next_frontier.get(receiver, SignedLightLedger())
                    next_frontier[receiver] = prior.plus(delivered)

            for source, source_ledger in frontier.items():
                if events >= self.maximum_events:
                    raise RuntimeError("child-volume residual event ceiling reached")
                events += 1
                for link in self._outgoing.get(source, ()):
                    applications += 1
                    deliver(link.receiver, link.transport(source_ledger))
            if self._native_plan is not None and self.topology.blocks:
                native_delivery, native_stats = self._native_block_wave(frontier)
                native_wave_calls += 1
                block_applications += native_stats.block_applications
                gathered_coefficients += native_stats.gathered_source_coefficients
                scattered_coefficients += native_stats.scattered_receiver_coefficients
                contiguous_source_blocks += native_stats.contiguous_source_blocks
                contiguous_receiver_blocks += native_stats.contiguous_receiver_blocks
                for receiver, ledger in native_delivery.items():
                    deliver(receiver, ledger)
            else:
                active_blocks: set[int] = set()
                for source in frontier:
                    active_blocks.update(self._blocks_by_source.get(source, ()))
                for block_index in sorted(active_blocks):
                    block = self.topology.blocks[block_index]
                    gathered_packets: list[LightTensor6] = []
                    for source, factor in zip(block.sources, block.source_factor):
                        source_ledger = frontier.get(source)
                        if source_ledger is None:
                            continue
                        gathered_coefficients += 1
                        gathered_packets.extend(
                            packet.scaled(float(factor))
                            for packet in source_ledger.packets
                        )
                    gathered = SignedLightLedger.from_packets(gathered_packets)
                    if not gathered.packets:
                        continue
                    block_applications += 1
                    transported = SignedLightLedger.from_packets(
                        block.medium.transport(packet, block.distance)
                        for packet in gathered.packets
                    )
                    for receiver, factor in zip(block.receivers, block.receiver_factor):
                        scattered_coefficients += 1
                        deliver(receiver, transported.scaled(float(factor)))
            frontier = {
                address: ledger.coalesced(self.absolute_cutoff)
                for address, ledger in next_frontier.items()
                if ledger.absolute_power > self.absolute_cutoff
            }
        return _MarchResult(
            tuple(outputs),
            applications,
            events,
            discarded,
            block_applications,
            gathered_coefficients,
            scattered_coefficients,
            native_wave_calls,
            contiguous_source_blocks,
            contiguous_receiver_blocks,
        )

    def _internal_emission(self) -> tuple[_MarchResult, bool]:
        if self._emission_response is None:
            seeds = {
                internal(index): node.emission
                for index, node in enumerate(self.topology.internal_nodes)
                if node.emission.packets
            }
            self._emission_response = self._march(seeds)
            return self._emission_response, bool(seeds)
        return self._emission_response, False

    def evaluate(
        self,
        incoming: Mapping[int, SignedLightLedger],
        *,
        include_emission: bool = True,
    ) -> ChildVolumeSolveResult:
        """Evaluate ``R_V[incoming] + E_V`` through scalar cache or rank blocks."""
        if include_emission:
            emission_response, emission_compiled = self._internal_emission()
            output = list(emission_response.outputs)
        else:
            emission_response = _MarchResult(
                tuple(SignedLightLedger() for _ in self.topology.egress_fibers),
                0,
                0,
                0.0,
            )
            emission_compiled = False
            output = list(emission_response.outputs)
        hits = misses = marches = applications = events = 0
        block_applications = gathered_coefficients = scattered_coefficients = 0
        native_wave_calls = 0
        contiguous_source_blocks = contiguous_receiver_blocks = 0
        discarded = 0.0
        if emission_compiled:
            marches += 1
            applications += emission_response.edge_applications
            events += emission_response.residual_events
            discarded += emission_response.discarded_absolute_power
            block_applications += emission_response.block_applications
            gathered_coefficients += emission_response.gathered_source_coefficients
            scattered_coefficients += emission_response.scattered_receiver_coefficients
            native_wave_calls += emission_response.native_wave_calls
            contiguous_source_blocks += emission_response.contiguous_source_blocks
            contiguous_receiver_blocks += emission_response.contiguous_receiver_blocks
        clean_incoming: dict[int, SignedLightLedger] = {}
        for port, ledger in incoming.items():
            if port < 0 or port >= len(self.topology.ingress_fibers):
                raise ValueError("incoming field addresses an invalid child ingress fiber")
            clean = ledger.coalesced(self.absolute_cutoff)
            if clean.packets:
                clean_incoming[port] = clean
        if self._evaluation_strategy == "retained":
            if clean_incoming:
                marched = self._march(
                    {ingress(port): ledger for port, ledger in clean_incoming.items()}
                )
                marches += 1
                applications += marched.edge_applications
                events += marched.residual_events
                discarded += marched.discarded_absolute_power
                block_applications += marched.block_applications
                gathered_coefficients += marched.gathered_source_coefficients
                scattered_coefficients += marched.scattered_receiver_coefficients
                native_wave_calls += marched.native_wave_calls
                contiguous_source_blocks += marched.contiguous_source_blocks
                contiguous_receiver_blocks += marched.contiguous_receiver_blocks
                for output_port, ledger in enumerate(marched.outputs):
                    output[output_port] = output[output_port].plus(ledger)
        else:
            for port, ledger in clean_incoming.items():
                for packet in ledger.packets:
                    key = (port, _mode_key(packet))
                    response = self._mode_responses.get(key)
                    if response is None:
                        unit = LightTensor6(
                            1.0,
                            packet.centre_nm,
                            packet.variance_nm2,
                            packet.orientation_degrees,
                            packet.linear_peakedness,
                            packet.chirality,
                            packet.provenance,
                        )
                        marched = self._march(
                            {ingress(port): SignedLightLedger.from_packets([unit])}
                        )
                        response = marched.outputs
                        self._mode_responses[key] = response
                        misses += 1
                        marches += 1
                        applications += marched.edge_applications
                        events += marched.residual_events
                        block_applications += marched.block_applications
                        gathered_coefficients += marched.gathered_source_coefficients
                        scattered_coefficients += marched.scattered_receiver_coefficients
                        native_wave_calls += marched.native_wave_calls
                        contiguous_source_blocks += marched.contiguous_source_blocks
                        contiguous_receiver_blocks += marched.contiguous_receiver_blocks
                        discarded += abs(packet.power) * marched.discarded_absolute_power
                    else:
                        hits += 1
                    for output_port, unit_ledger in enumerate(response):
                        output[output_port] = output[output_port].plus(
                            unit_ledger.scaled(packet.power)
                        )
        compressed = CompressedBoundaryField(
            tuple(ledger.coalesced(self.absolute_cutoff) for ledger in output)
        )
        return ChildVolumeSolveResult(
            compressed,
            hits,
            misses,
            marches,
            applications,
            events,
            discarded,
            self.topology.revision,
            block_applications,
            gathered_coefficients,
            scattered_coefficients,
            self._evaluation_strategy,
            self._execution_backend,
            native_wave_calls,
            contiguous_source_blocks,
            contiguous_receiver_blocks,
        )


@dataclass(frozen=True)
class ChildIngressBinding:
    parent_source: int
    child_port: int
    gain: float

    def __post_init__(self) -> None:
        if self.parent_source < 0 or self.child_port < 0:
            raise ValueError("child-ingress binding indices must be nonnegative")
        if not math.isfinite(self.gain) or not 0.0 <= self.gain <= 1.0:
            raise ValueError("child-ingress binding gain must lie in [0, 1]")


@dataclass(frozen=True)
class ChildEgressBinding:
    child_port: int
    parent_receiver: int
    gain: float

    def __post_init__(self) -> None:
        if self.child_port < 0 or self.parent_receiver < 0:
            raise ValueError("child-egress binding indices must be nonnegative")
        if not math.isfinite(self.gain) or not 0.0 <= self.gain <= 1.0:
            raise ValueError("child-egress binding gain must lie in [0, 1]")


@dataclass(frozen=True)
class ChildPortal:
    response: ChildVolumeResponseSystem
    ingress_bindings: tuple[ChildIngressBinding, ...]
    egress_bindings: tuple[ChildEgressBinding, ...]


@dataclass(frozen=True)
class CoupledVolumeSolveResult:
    parent_state: tuple[SignedLightLedger, ...]
    parent_edge_applications: int
    child_edge_applications: int
    child_response_evaluations: int
    child_cache_hits: int
    child_cache_misses: int
    residual_events: int
    discarded_absolute_power: float
    child_block_applications: int = 0
    child_gathered_source_coefficients: int = 0
    child_scattered_receiver_coefficients: int = 0


class CoupledVolumeTransportSystem:
    """Residual march over a parent register and sealed child response ports."""

    def __init__(
        self,
        parent: LightTransportRegister,
        portals: Iterable[ChildPortal],
    ) -> None:
        self.parent = parent
        self.portals = tuple(portals)
        self._ingress_by_parent: list[list[tuple[int, ChildIngressBinding]]] = [
            [] for _ in range(parent.node_count)
        ]
        parent_outgoing = [sum(edge.gain for edge in edges) for edges in parent.outgoing]
        for portal_index, portal in enumerate(self.portals):
            ingress_count = len(portal.response.topology.ingress_fibers)
            egress_count = len(portal.response.topology.egress_fibers)
            egress_gain: dict[int, float] = {}
            for binding in portal.ingress_bindings:
                if binding.parent_source >= parent.node_count:
                    raise ValueError("child ingress binds outside the parent register")
                if binding.child_port >= ingress_count:
                    raise ValueError("child ingress binds outside the child boundary")
                parent_outgoing[binding.parent_source] += binding.gain
                self._ingress_by_parent[binding.parent_source].append(
                    (portal_index, binding)
                )
            for binding in portal.egress_bindings:
                if binding.parent_receiver >= parent.node_count:
                    raise ValueError("child egress binds outside the parent register")
                if binding.child_port >= egress_count:
                    raise ValueError("child egress binds outside the child boundary")
                egress_gain[binding.child_port] = (
                    egress_gain.get(binding.child_port, 0.0) + binding.gain
                )
            if any(gain > 1.0 + 64.0 * np.finfo(float).eps for gain in egress_gain.values()):
                raise ValueError("child egress binding gains exceed one")
        if any(gain > 1.0 + 64.0 * np.finfo(float).eps for gain in parent_outgoing):
            raise ValueError("combined parent and child-ingress gains exceed one")

    def solve(
        self,
        emission: Mapping[int, SignedLightLedger],
        *,
        absolute_cutoff: float = 1.0e-12,
        maximum_events: int = 1_000_000,
    ) -> CoupledVolumeSolveResult:
        if absolute_cutoff < 0.0 or maximum_events < 1:
            raise ValueError("invalid coupled residual-march configuration")
        state = [SignedLightLedger() for _ in range(self.parent.node_count)]
        frontier: dict[int, SignedLightLedger] = {}

        def feed(node: int, ledger: SignedLightLedger) -> None:
            clean = ledger.coalesced(absolute_cutoff)
            if not clean.packets:
                return
            state[node] = state[node].plus(clean)
            frontier[node] = frontier.get(node, SignedLightLedger()).plus(clean)

        for node, ledger in emission.items():
            if node < 0 or node >= self.parent.node_count:
                raise ValueError("emission lies outside the parent register")
            feed(node, ledger)

        parent_applications = child_applications = child_evaluations = 0
        child_block_applications = 0
        child_gathered_coefficients = child_scattered_coefficients = 0
        child_hits = child_misses = events = 0
        discarded = 0.0
        for portal in self.portals:
            child_emission = portal.response.evaluate({}, include_emission=True)
            child_applications += child_emission.edge_applications
            child_block_applications += child_emission.block_applications
            child_gathered_coefficients += child_emission.gathered_source_coefficients
            child_scattered_coefficients += child_emission.scattered_receiver_coefficients
            discarded += child_emission.discarded_absolute_power
            if child_emission.kernel_marches:
                child_evaluations += 1
            for binding in portal.egress_bindings:
                feed(
                    binding.parent_receiver,
                    child_emission.outgoing.ledgers[binding.child_port].scaled(binding.gain),
                )

        while frontier:
            next_frontier: dict[int, SignedLightLedger] = {}

            def queue(node: int, ledger: SignedLightLedger) -> None:
                nonlocal discarded
                if ledger.absolute_power <= absolute_cutoff:
                    discarded += ledger.absolute_power
                    return
                next_frontier[node] = next_frontier.get(
                    node, SignedLightLedger()
                ).plus(ledger)

            child_incoming: list[dict[int, SignedLightLedger]] = [
                {} for _ in self.portals
            ]
            for node, ledger in frontier.items():
                if events >= maximum_events:
                    raise RuntimeError("coupled residual event ceiling reached")
                events += 1
                for edge in self.parent.outgoing[node]:
                    parent_applications += 1
                    queue(edge.receiver, ledger.scaled(edge.gain))
                for portal_index, binding in self._ingress_by_parent[node]:
                    delivered = ledger.scaled(binding.gain)
                    prior = child_incoming[portal_index].get(
                        binding.child_port, SignedLightLedger()
                    )
                    child_incoming[portal_index][binding.child_port] = prior.plus(delivered)
            for portal_index, incoming_field in enumerate(child_incoming):
                if not incoming_field:
                    continue
                portal = self.portals[portal_index]
                child_cutoff = max(absolute_cutoff, portal.response.absolute_cutoff)
                prepared: dict[int, SignedLightLedger] = {}
                for port, ledger in incoming_field.items():
                    clean = ledger.coalesced(child_cutoff)
                    if clean.absolute_power <= child_cutoff:
                        discarded += ledger.absolute_power
                    else:
                        prepared[port] = clean
                if not prepared:
                    continue
                response = portal.response.evaluate(
                    prepared,
                    include_emission=False,
                )
                child_evaluations += 1
                child_applications += response.edge_applications
                child_block_applications += response.block_applications
                child_gathered_coefficients += response.gathered_source_coefficients
                child_scattered_coefficients += response.scattered_receiver_coefficients
                child_hits += response.cache_hits
                child_misses += response.cache_misses
                discarded += response.discarded_absolute_power
                for binding in portal.egress_bindings:
                    queue(
                        binding.parent_receiver,
                        response.outgoing.ledgers[binding.child_port].scaled(binding.gain),
                    )
            frontier = {}
            for node, ledger in next_frontier.items():
                clean = ledger.coalesced(absolute_cutoff)
                if clean.packets:
                    state[node] = state[node].plus(clean)
                    frontier[node] = clean
        return CoupledVolumeSolveResult(
            tuple(ledger.coalesced(absolute_cutoff) for ledger in state),
            parent_applications,
            child_applications,
            child_evaluations,
            child_hits,
            child_misses,
            events,
            discarded,
            child_block_applications,
            child_gathered_coefficients,
            child_scattered_coefficients,
        )
