"""Signed six-coordinate spectral and polarization transport.

The physical packet is

    (signed power, wavelength centre, wavelength variance,
     central polarization orientation, linear peakedness, chirality).

Orientation is stored in degrees, peakedness in ``[0, 1]``, and signed
chirality in ``[-1, 1]``.  Stokes values are derived only when an oracle or
physical projection needs them; they are not transport coordinates. Source
provenance is registration metadata and deliberately is not a seventh
physical coordinate. A field is a sparse mixture of packets. Negative packets
are linear correction measures: they may propagate through linear transport,
but they are reconciled against positive radiance before a terminal boundary
can emit or display the field.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


Array = np.ndarray
SpectrumFunction = Callable[[Array], Array]


def trapezoidal_integral(values: Array, coordinates: Array, axis: int = -1) -> Array:
    """NumPy 1.x/2.x-compatible trapezoidal integration."""
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return implementation(values, coordinates, axis=axis)


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class LightTensor6:
    """One continuous Gaussian spectral packet and immutable mode identity.

    ``power`` is signed integrated radiant power.  A negative value denotes an
    adjustment packet, not negative physical illumination.  The polarization
    state consists of a central orientation, its linear peakedness, and signed
    circular chirality. Their combined degree must remain physically
    realizable.
    """

    power: float
    centre_nm: float
    variance_nm2: float
    orientation_degrees: float = 0.0
    linear_peakedness: float = 0.0
    chirality: float = 0.0
    provenance: int = 0

    def __post_init__(self) -> None:
        if not _finite(
            self.power,
            self.centre_nm,
            self.variance_nm2,
            self.orientation_degrees,
            self.linear_peakedness,
            self.chirality,
        ):
            raise ValueError("light tensor coordinates must be finite")
        if self.variance_nm2 <= 0.0:
            raise ValueError("spectral variance must be positive")
        if self.centre_nm <= 0.0:
            raise ValueError("wavelength centre must be positive")
        if self.provenance < 0:
            raise ValueError("provenance bitset must be nonnegative")
        if not 0.0 <= self.orientation_degrees < 360.0:
            raise ValueError("polarization orientation must lie in [0, 360)")
        if not 0.0 <= self.linear_peakedness <= 1.0:
            raise ValueError("linear peakedness must lie in [0, 1]")
        if not -1.0 <= self.chirality <= 1.0:
            raise ValueError("chirality must lie in [-1, 1]")
        degree2 = self.linear_peakedness**2 + self.chirality**2
        if degree2 > 1.0 + 1.0e-12:
            raise ValueError("combined polarization state is not realizable")

    @property
    def degree_of_polarization(self) -> float:
        return min(1.0, math.hypot(self.linear_peakedness, self.chirality))

    @property
    def linear_orientation_radians(self) -> float:
        return math.radians(self.orientation_degrees)

    @property
    def linear_fraction(self) -> float:
        return self.linear_peakedness

    @property
    def circular_fraction(self) -> float:
        return abs(self.chirality)

    @property
    def derived_stokes(self) -> Array:
        angle = 2.0 * self.linear_orientation_radians
        return np.array(
            [
                1.0,
                self.linear_peakedness * math.cos(angle),
                self.linear_peakedness * math.sin(angle),
                self.chirality,
            ],
            dtype=np.float64,
        )

    def scaled(self, factor: float) -> "LightTensor6":
        if not math.isfinite(factor):
            raise ValueError("transport factor must be finite")
        return LightTensor6(
            self.power * factor,
            self.centre_nm,
            self.variance_nm2,
            self.orientation_degrees,
            self.linear_peakedness,
            self.chirality,
            self.provenance,
        )

    def with_provenance(self, label: int) -> "LightTensor6":
        if label < 0:
            raise ValueError("provenance label must be nonnegative")
        return LightTensor6(
            self.power,
            self.centre_nm,
            self.variance_nm2,
            self.orientation_degrees,
            self.linear_peakedness,
            self.chirality,
            self.provenance | (1 << label),
        )

    def spectral_density(self, wavelength_nm: Array) -> Array:
        wavelength = np.asarray(wavelength_nm, dtype=np.float64)
        sigma = math.sqrt(self.variance_nm2)
        normalized = np.exp(-0.5 * ((wavelength - self.centre_nm) / sigma) ** 2)
        normalized /= sigma * math.sqrt(2.0 * math.pi)
        return self.power * normalized

    def stokes_density(self, wavelength_nm: Array) -> Array:
        intensity = self.spectral_density(wavelength_nm)
        return intensity[:, None] * self.derived_stokes[None, :]


@dataclass(frozen=True)
class MaterializedLight:
    wavelength_nm: Array
    stokes_density: Array
    cancelled_power: float
    unmet_negative_power: float
    polarization_projection_power: float


@dataclass(frozen=True)
class SignedLightLedger:
    packets: tuple[LightTensor6, ...] = ()

    @classmethod
    def from_packets(cls, packets: Iterable[LightTensor6]) -> "SignedLightLedger":
        return cls(tuple(packets)).coalesced()

    @property
    def absolute_power(self) -> float:
        return sum(abs(packet.power) for packet in self.packets)

    @property
    def signed_power(self) -> float:
        return sum(packet.power for packet in self.packets)

    @property
    def provenance(self) -> int:
        result = 0
        for packet in self.packets:
            result |= packet.provenance
        return result

    def scaled(self, factor: float) -> "SignedLightLedger":
        return SignedLightLedger.from_packets(packet.scaled(factor) for packet in self.packets)

    def plus(self, other: "SignedLightLedger") -> "SignedLightLedger":
        return SignedLightLedger.from_packets(self.packets + other.packets)

    def coalesced(self, cutoff: float = 0.0) -> "SignedLightLedger":
        """Fuse algebraically identical packets without merging spectral shape."""
        powers: dict[tuple[float | int, ...], float] = {}
        templates: dict[tuple[float | int, ...], LightTensor6] = {}
        for packet in self.packets:
            key = (
                packet.centre_nm,
                packet.variance_nm2,
                packet.orientation_degrees,
                packet.linear_peakedness,
                packet.chirality,
                packet.provenance,
            )
            powers[key] = powers.get(key, 0.0) + packet.power
            templates[key] = packet
        result = []
        for key, power in powers.items():
            if abs(power) <= cutoff:
                continue
            packet = templates[key]
            result.append(
                LightTensor6(
                    power,
                    packet.centre_nm,
                    packet.variance_nm2,
                    packet.orientation_degrees,
                    packet.linear_peakedness,
                    packet.chirality,
                    packet.provenance,
                )
            )
        result.sort(
            key=lambda packet: (
                packet.provenance,
                packet.centre_nm,
                packet.variance_nm2,
                packet.orientation_degrees,
                packet.linear_peakedness,
                packet.chirality,
            )
        )
        return SignedLightLedger(tuple(result))

    def materialize(self, wavelength_nm: Array) -> MaterializedLight:
        """Reconcile signed measures and project onto the physical Stokes cone."""
        wavelength = np.asarray(wavelength_nm, dtype=np.float64)
        if wavelength.ndim != 1 or wavelength.size < 2 or np.any(np.diff(wavelength) <= 0):
            raise ValueError("wavelength grid must be a strictly increasing vector")
        positive = np.zeros((wavelength.size, 4), dtype=np.float64)
        negative = np.zeros_like(positive)
        for packet in self.packets:
            density = packet.stokes_density(wavelength)
            if packet.power >= 0.0:
                positive += density
            else:
                negative -= density
        signed = positive - negative
        cancelled = np.minimum(positive[:, 0], negative[:, 0])
        unmet = np.maximum(-signed[:, 0], 0.0)
        signed[:, 0] = np.maximum(signed[:, 0], 0.0)
        signed[unmet > 0.0, 1:] = 0.0

        polarized_norm = np.linalg.norm(signed[:, 1:], axis=1)
        excess = np.maximum(polarized_norm - signed[:, 0], 0.0)
        projected = polarized_norm > signed[:, 0]
        safe_norm = np.maximum(polarized_norm[projected], 1.0e-300)
        signed[projected, 1:] *= (
            signed[projected, 0] / safe_norm
        )[:, None]
        return MaterializedLight(
            wavelength,
            signed,
            float(trapezoidal_integral(cancelled, wavelength)),
            float(trapezoidal_integral(unmet, wavelength)),
            float(trapezoidal_integral(excess, wavelength)),
        )


@dataclass(frozen=True)
class PolarizationBoundaryResult:
    """Extinction/generation ledger for one polarization boundary.

    ``incident`` is retained verbatim as the identity of the arriving mode
    family. ``extinguished`` is its signed removal at the incident port.
    ``outgoing`` is a newly selected or generated family at the outgoing port.
    No operation changes the polarization coordinates of ``incident``.
    """

    incident: LightTensor6
    extinguished: LightTensor6
    outgoing: LightTensor6

    @property
    def surviving_power_fraction(self) -> float:
        if abs(self.incident.power) <= 1.0e-300:
            return 0.0
        return self.outgoing.power / self.incident.power


def linear_polarizer(packet: LightTensor6, angle_radians: float) -> LightTensor6:
    """Feed forward through an ideal linear polarization filter.

    The filter establishes its exact central orientation and unit peakedness.
    The surviving power is the generalized Malus fraction of the incoming
    orientation distribution.
    """
    if not math.isfinite(angle_radians):
        raise ValueError("polarizer angle must be finite")
    c = math.cos(2.0 * angle_radians)
    s = math.sin(2.0 * angle_radians)
    matrix = 0.5 * np.array(
        [
            [1.0, c, s, 0.0],
            [c, c * c, c * s, 0.0],
            [s, c * s, s * s, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    selected = mueller_boundary(packet, matrix).outgoing
    return LightTensor6(
        selected.power,
        selected.centre_nm,
        selected.variance_nm2,
        math.degrees(angle_radians) % 360.0 if abs(selected.power) > 1.0e-300 else 0.0,
        selected.linear_peakedness,
        selected.chirality,
        selected.provenance,
    )


def circular_polarizer(packet: LightTensor6, handedness: int) -> LightTensor6:
    """Feed forward through a circular filter and establish handedness."""
    if handedness not in (-1, 1):
        raise ValueError("circular handedness must be -1 or +1")
    h = float(handedness)
    matrix = 0.5 * np.array(
        [
            [1.0, 0.0, 0.0, h],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [h, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return mueller_boundary(packet, matrix).outgoing


def mueller_boundary(packet: LightTensor6, matrix: Array) -> PolarizationBoundaryResult:
    """Resolve a Mueller boundary as extinction plus mode generation.

    Stokes coordinates are derived for this one local multiplication only.
    The matrix determines how much incident population survives and the mode
    distribution emitted at the outgoing port. The incident mode is neither
    rewritten nor marched in Stokes form. Sequential boundaries are therefore
    transitive: each consumes the preceding boundary's generated population.
    """
    outgoing = _mueller_generated_mode(packet, matrix)
    return PolarizationBoundaryResult(
        incident=packet,
        extinguished=packet.scaled(-1.0),
        outgoing=outgoing,
    )


def apply_mueller(packet: LightTensor6, matrix: Array) -> LightTensor6:
    """Return the outgoing mode generated by :func:`mueller_boundary`.

    This compatibility form does not imply mutation. New code needing the
    extinction ledger should call ``mueller_boundary`` explicitly.
    """
    return mueller_boundary(packet, matrix).outgoing


def _mueller_generated_mode(packet: LightTensor6, matrix: Array) -> LightTensor6:
    """Perform transient Mueller arithmetic and construct a new mode family."""
    operator = np.asarray(matrix, dtype=np.float64)
    if operator.shape != (4, 4) or not np.all(np.isfinite(operator)):
        raise ValueError("Mueller operator must be a finite 4x4 matrix")
    outgoing = operator @ (packet.power * packet.derived_stokes)
    power = float(outgoing[0])
    if abs(power) <= 1.0e-300:
        return LightTensor6(
            0.0,
            packet.centre_nm,
            packet.variance_nm2,
            provenance=packet.provenance,
        )
    q, u, chirality = (float(value / power) for value in outgoing[1:])
    peakedness = math.hypot(q, u)
    degree = math.hypot(peakedness, chirality)
    if degree > 1.0 + 1.0e-12:
        raise ValueError("Mueller operator generated a non-realizable mode distribution")
    if degree > 1.0:
        peakedness /= degree
        chirality /= degree
        q /= degree
        u /= degree
    orientation = (
        math.degrees(0.5 * math.atan2(u, q)) % 360.0
        if peakedness > 1.0e-15
        else 0.0
    )
    return LightTensor6(
        power,
        packet.centre_nm,
        packet.variance_nm2,
        orientation,
        peakedness,
        chirality,
        packet.provenance,
    )


def volume_transport(
    packet: LightTensor6,
    *,
    distance: float,
    density: float,
    extinction: float,
    linear_diffusive_extinction: float,
    circular_diffusive_extinction: float,
) -> LightTensor6:
    """Transport through a homogeneous volume with independent loss laws.

    The polarization-mode coordinates never change in flight. Extinction and
    diffusivity form one probabilistic extinction exponent for that mode.
    Apparent loss of aggregate polarization is the change in relative
    population among immutable modes with different extinction rates.
    """
    if min(
        distance,
        density,
        extinction,
        linear_diffusive_extinction,
        circular_diffusive_extinction,
    ) < 0.0:
        raise ValueError("volume coefficients and distance must be nonnegative")
    optical_depth = distance * density
    mode_extinction = (
        extinction
        + linear_diffusive_extinction * packet.linear_peakedness
        + circular_diffusive_extinction * abs(packet.chirality)
    )
    retention = math.exp(-mode_extinction * optical_depth)
    return LightTensor6(
        packet.power * retention,
        packet.centre_nm,
        packet.variance_nm2,
        packet.orientation_degrees,
        packet.linear_peakedness,
        packet.chirality,
        packet.provenance,
    )


def volume_transport_ledger(
    ledger: SignedLightLedger,
    *,
    distance: float,
    density: float,
    extinction: float,
    linear_diffusive_extinction: float,
    circular_diffusive_extinction: float,
) -> SignedLightLedger:
    """Feed all immutable mode families through the unified extinction law."""
    return SignedLightLedger.from_packets(
        volume_transport(
            packet,
            distance=distance,
            density=density,
            extinction=extinction,
            linear_diffusive_extinction=linear_diffusive_extinction,
            circular_diffusive_extinction=circular_diffusive_extinction,
        )
        for packet in ledger.packets
    )


def diffuse_boundary(packet: LightTensor6, reflectance: float) -> LightTensor6:
    """Extinguish the incident family and generate a diffuse outgoing family."""
    if not 0.0 <= reflectance <= 1.0:
        raise ValueError("reflectance must lie in [0, 1]")
    return LightTensor6(
        packet.power * reflectance,
        packet.centre_nm,
        packet.variance_nm2,
        0.0,
        0.0,
        0.0,
        packet.provenance,
    )


@dataclass(frozen=True)
class SpectralFilterResult:
    ledger: SignedLightLedger
    input_power: float
    output_power: float
    component_count: int
    maximum_closure_error: float


def adaptive_spectral_filter(
    packet: LightTensor6,
    transmission: SpectrumFunction,
    *,
    tolerance: float = 0.08,
    max_components: int = 8,
    quadrature_order: int = 32,
) -> SpectralFilterResult:
    """Apply a wavelength law and retain an adaptive Gaussian mixture.

    Centre and variance determine interaction exactly only for transfer laws
    closed under those two moments.  We therefore evaluate the transmitted
    continuous density on a deterministic local wavelength chart and fit one
    through ``max_components`` Gaussian packets.  The first mixture whose
    relative spectral L1 residual meets ``tolerance`` is retained.  This tests
    the represented distribution itself rather than assuming that matching a
    few higher moments also makes the density accurate.
    """
    if tolerance < 0.0 or max_components < 1 or quadrature_order < 8:
        raise ValueError("invalid adaptive spectral-filter configuration")
    sigma = math.sqrt(packet.variance_nm2)
    sample_count = max(257, quadrature_order * 16 + 1)
    wavelength = np.linspace(
        max(1.0, packet.centre_nm - 6.0 * sigma),
        packet.centre_nm + 6.0 * sigma,
        sample_count,
    )
    transmitted = np.asarray(transmission(wavelength), dtype=np.float64)
    if transmitted.shape != wavelength.shape or not np.all(np.isfinite(transmitted)):
        raise ValueError("spectral transmission must return one finite value per wavelength")
    if np.any(transmitted < 0.0) or np.any(transmitted > 1.0):
        raise ValueError("spectral transmission must lie in [0, 1]")
    base_density = np.exp(-0.5 * ((wavelength - packet.centre_nm) / sigma) ** 2)
    base_density /= sigma * math.sqrt(2.0 * math.pi)
    target_density = base_density * transmitted
    spacing = float(wavelength[1] - wavelength[0])
    integration_weight = np.full(sample_count, spacing)
    integration_weight[[0, -1]] *= 0.5
    sample_mass = target_density * integration_weight
    transmitted_fraction = float(np.sum(sample_mass))
    if transmitted_fraction <= 1.0e-300:
        return SpectralFilterResult(SignedLightLedger(), packet.power, 0.0, 0, 0.0)
    probability_mass = sample_mass / transmitted_fraction
    cumulative = np.cumsum(probability_mass)
    grid_variance_floor = spacing * spacing * 0.25

    def normal_density(centres: Array, variances: Array) -> Array:
        delta = wavelength[:, None] - centres[None, :]
        return np.exp(-0.5 * delta * delta / variances[None, :]) / np.sqrt(
            2.0 * math.pi * variances[None, :]
        )

    best: tuple[float, Array, Array, Array] | None = None
    for component_count in range(1, max_components + 1):
        quantiles = (np.arange(component_count, dtype=np.float64) + 0.5) / component_count
        centres = np.interp(quantiles, cumulative, wavelength)
        global_centre = float(np.dot(probability_mass, wavelength))
        global_variance = max(
            float(np.dot(probability_mass, (wavelength - global_centre) ** 2)),
            grid_variance_floor,
        )
        variances = np.full(
            component_count,
            max(global_variance / component_count, grid_variance_floor),
        )
        mixture_weight = np.full(component_count, 1.0 / component_count)
        for _ in range(128):
            component_density = normal_density(centres, variances)
            weighted_density = component_density * mixture_weight[None, :]
            normalizer = np.maximum(np.sum(weighted_density, axis=1), 1.0e-300)
            responsibility = weighted_density / normalizer[:, None]
            owned = probability_mass[:, None] * responsibility
            new_weight = np.maximum(np.sum(owned, axis=0), 1.0e-15)
            new_weight /= np.sum(new_weight)
            new_centre = np.sum(owned * wavelength[:, None], axis=0) / np.maximum(
                np.sum(owned, axis=0), 1.0e-300
            )
            delta = wavelength[:, None] - new_centre[None, :]
            new_variance = np.sum(owned * delta * delta, axis=0) / np.maximum(
                np.sum(owned, axis=0), 1.0e-300
            )
            new_variance = np.maximum(new_variance, grid_variance_floor)
            change = max(
                float(np.max(np.abs(new_centre - centres))) / max(sigma, 1.0),
                float(np.max(np.abs(new_variance - variances))) / max(global_variance, 1.0),
                float(np.max(np.abs(new_weight - mixture_weight))),
            )
            centres, variances, mixture_weight = new_centre, new_variance, new_weight
            if change < 1.0e-11:
                break
        reconstructed = transmitted_fraction * np.sum(
            normal_density(centres, variances) * mixture_weight[None, :], axis=1
        )
        error = float(np.sum(np.abs(reconstructed - target_density) * integration_weight))
        error /= transmitted_fraction
        candidate = (error, centres.copy(), variances.copy(), mixture_weight.copy())
        if best is None or error < best[0]:
            best = candidate
        if error <= tolerance:
            break

    assert best is not None
    closure_error, centres, variances, mixture_weight = best
    components = [
        LightTensor6(
            packet.power * transmitted_fraction * float(weight),
            float(centre),
            float(variance),
            packet.orientation_degrees,
            packet.linear_peakedness,
            packet.chirality,
            packet.provenance,
        )
        for centre, variance, weight in zip(centres, variances, mixture_weight)
        if weight > 1.0e-14
    ]
    output = SignedLightLedger.from_packets(components)
    return SpectralFilterResult(
        output,
        packet.power,
        output.signed_power,
        len(output.packets),
        closure_error,
    )


def sensor_rgb(materialized: MaterializedLight) -> Array:
    """Integrate a smooth three-sensor observer; output remains linear."""
    wavelength = materialized.wavelength_nm
    intensity = materialized.stokes_density[:, 0]
    centres = np.array([610.0, 545.0, 460.0])
    widths = np.array([48.0, 38.0, 34.0])
    responses = np.exp(
        -0.5 * ((wavelength[:, None] - centres[None, :]) / widths[None, :]) ** 2
    )
    return trapezoidal_integral(intensity[:, None] * responses, wavelength, axis=0)


@dataclass(frozen=True)
class RegisteredEdge:
    source: int
    receiver: int
    gain: float
    geometry_label: int

    def __post_init__(self) -> None:
        if self.source < 0 or self.receiver < 0 or self.geometry_label < 0:
            raise ValueError("transport indices and labels must be nonnegative")
        if not math.isfinite(self.gain) or self.gain < 0.0:
            raise ValueError("transport gain must be finite and nonnegative")

    @property
    def identity(self) -> tuple[int, int, int]:
        return self.source, self.receiver, self.geometry_label


@dataclass(frozen=True)
class RegisterSolveResult:
    state: tuple[SignedLightLedger, ...]
    edge_applications: int
    residual_events: int
    discarded_absolute_power: float


class LightTransportRegister:
    """Sparse source register with signed incremental correction transport."""

    def __init__(self, node_count: int, edges: Iterable[RegisteredEdge]) -> None:
        if node_count < 1:
            raise ValueError("transport register needs at least one node")
        self.node_count = node_count
        self.edges = tuple(edges)
        self.outgoing: list[list[RegisteredEdge]] = [[] for _ in range(node_count)]
        identities: set[tuple[int, int, int]] = set()
        for edge in self.edges:
            if edge.source >= node_count or edge.receiver >= node_count:
                raise ValueError("transport edge lies outside the register")
            if edge.identity in identities:
                raise ValueError("transport edge identity must be unique")
            identities.add(edge.identity)
            self.outgoing[edge.source].append(edge)
        for source, outgoing in enumerate(self.outgoing):
            total_gain = sum(edge.gain for edge in outgoing)
            if total_gain > 1.0 + 64.0 * np.finfo(float).eps:
                raise ValueError(
                    f"outgoing transport gain exceeds one at node {source}"
                )

    def propagate_residual(
        self,
        seeds: Mapping[int, SignedLightLedger],
        *,
        absolute_cutoff: float = 1.0e-12,
        maximum_events: int = 1_000_000,
    ) -> RegisterSolveResult:
        if absolute_cutoff < 0.0 or maximum_events < 1:
            raise ValueError("invalid residual march configuration")
        state = [SignedLightLedger() for _ in range(self.node_count)]
        frontier: dict[int, SignedLightLedger] = {}
        for node, ledger in seeds.items():
            if not 0 <= node < self.node_count:
                raise ValueError("seed node lies outside the register")
            clean = ledger.coalesced(absolute_cutoff)
            if clean.packets:
                state[node] = state[node].plus(clean)
                frontier[node] = frontier.get(node, SignedLightLedger()).plus(clean)
        applications = events = 0
        discarded = 0.0
        while frontier:
            next_frontier: dict[int, SignedLightLedger] = {}
            for node, node_frontier in frontier.items():
                if events >= maximum_events:
                    raise RuntimeError("residual event ceiling reached")
                events += 1
                for edge in self.outgoing[node]:
                    applications += 1
                    delivered = node_frontier.scaled(edge.gain)
                    if delivered.absolute_power <= absolute_cutoff:
                        discarded += delivered.absolute_power
                        continue
                    next_frontier[edge.receiver] = next_frontier.get(
                        edge.receiver, SignedLightLedger()
                    ).plus(delivered)
            frontier = {}
            for node, ledger in next_frontier.items():
                clean = ledger.coalesced(absolute_cutoff)
                if clean.packets:
                    state[node] = state[node].plus(clean)
                    frontier[node] = clean
        return RegisterSolveResult(tuple(state), applications, events, discarded)

    def solve(
        self,
        emission: Mapping[int, SignedLightLedger],
        *,
        absolute_cutoff: float = 1.0e-12,
    ) -> RegisterSolveResult:
        return self.propagate_residual(emission, absolute_cutoff=absolute_cutoff)

    def affected_closure(
        self,
        changed_edge_identities: Iterable[tuple[int, int, int]],
    ) -> frozenset[int]:
        changed = set(changed_edge_identities)
        roots = {identity[1] for identity in changed}
        reached = set(roots)
        queue = deque(roots)
        while queue:
            node = queue.popleft()
            for edge in self.outgoing[node]:
                if edge.receiver not in reached:
                    reached.add(edge.receiver)
                    queue.append(edge.receiver)
        return frozenset(reached)


def register_from_dense_operator(
    operator: Array,
    receiver_reflectance: Array,
    *,
    coefficient_cutoff: float = 0.0,
) -> LightTransportRegister:
    """Adapt an existing geometric transport operator to the light register.

    This is a dogfood bridge, not the production storage representation: it
    intentionally reads a dense oracle so the six-coordinate march can be
    compared with the repository's established geometric field.
    """
    matrix = np.asarray(operator, dtype=np.float64)
    reflectance = np.asarray(receiver_reflectance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transport operator must be square")
    if reflectance.shape != (matrix.shape[0],):
        raise ValueError("one receiver reflectance is required per node")
    if (
        not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(reflectance))
        or np.any(matrix < 0.0)
        or np.any((reflectance < 0.0) | (reflectance >= 1.0))
    ):
        raise ValueError("operator and reflectance must be finite and passive")
    edges = []
    node_count = matrix.shape[0]
    for receiver in range(node_count):
        for source in range(node_count):
            gain = float(reflectance[receiver] * matrix[receiver, source])
            if gain > coefficient_cutoff:
                edges.append(
                    RegisteredEdge(
                        source,
                        receiver,
                        gain,
                        receiver * node_count + source,
                    )
                )
    return LightTransportRegister(node_count, edges)


@dataclass(frozen=True)
class IncrementalUpdateResult:
    state: tuple[SignedLightLedger, ...]
    correction: tuple[SignedLightLedger, ...]
    affected_nodes: frozenset[int]
    edge_applications: int
    residual_events: int


@dataclass(frozen=True)
class PlannedUpdateResult:
    state: tuple[SignedLightLedger, ...]
    mode: str
    affected_nodes: frozenset[int]
    edge_applications: int
    residual_events: int


def _changed_edge_identities(
    old_register: LightTransportRegister,
    new_register: LightTransportRegister,
) -> tuple[
    dict[tuple[int, int, int], RegisteredEdge],
    dict[tuple[int, int, int], RegisteredEdge],
    set[tuple[int, int, int]],
]:
    old_edges = {edge.identity: edge for edge in old_register.edges}
    new_edges = {edge.identity: edge for edge in new_register.edges}
    changed = set()
    for identity in old_edges.keys() | new_edges.keys():
        old_gain = old_edges[identity].gain if identity in old_edges else 0.0
        new_gain = new_edges[identity].gain if identity in new_edges else 0.0
        if old_gain != new_gain:
            changed.add(identity)
    return old_edges, new_edges, changed


def _affected_nodes_for_change(
    new_register: LightTransportRegister,
    changed: Iterable[tuple[int, int, int]],
) -> frozenset[int]:
    roots = {identity[1] for identity in changed}
    reached = set(roots)
    queue = deque(roots)
    while queue:
        node = queue.popleft()
        for edge in new_register.outgoing[node]:
            if edge.receiver not in reached:
                reached.add(edge.receiver)
                queue.append(edge.receiver)
    return frozenset(reached)


def antimatter_update(
    old_register: LightTransportRegister,
    new_register: LightTransportRegister,
    old_state: Sequence[SignedLightLedger],
    *,
    absolute_cutoff: float = 1.0e-12,
) -> IncrementalUpdateResult:
    """Apply ``(I-T_new)^-1 (T_new-T_old) L_old`` as signed light.

    Removed transport is injected as negative residual power; new transport is
    injected as positive power.  Marching that correction through ``T_new``
    automatically reaches all descendants and returns through feedback loops.
    """
    if old_register.node_count != new_register.node_count:
        raise ValueError("old and new registers must have the same node count")
    if len(old_state) != old_register.node_count:
        raise ValueError("old state does not match the register")
    old_edges, new_edges, changed = _changed_edge_identities(
        old_register, new_register
    )
    seeds: dict[int, SignedLightLedger] = {}
    for identity in changed:
        old_gain = old_edges[identity].gain if identity in old_edges else 0.0
        new_gain = new_edges[identity].gain if identity in new_edges else 0.0
        source, receiver, _ = identity
        correction = old_state[source].scaled(new_gain - old_gain)
        seeds[receiver] = seeds.get(receiver, SignedLightLedger()).plus(correction)
    marched = new_register.propagate_residual(seeds, absolute_cutoff=absolute_cutoff)
    updated = tuple(
        old_state[node].plus(marched.state[node]).coalesced(absolute_cutoff)
        for node in range(new_register.node_count)
    )
    reached = _affected_nodes_for_change(new_register, changed)
    return IncrementalUpdateResult(
        updated,
        marched.state,
        reached,
        marched.edge_applications,
        marched.residual_events,
    )


def update_or_recompute(
    old_register: LightTransportRegister,
    new_register: LightTransportRegister,
    old_state: Sequence[SignedLightLedger],
    emission: Mapping[int, SignedLightLedger],
    *,
    maximum_incremental_node_fraction: float = 0.6,
    absolute_cutoff: float = 1.0e-12,
) -> PlannedUpdateResult:
    """Select signed correction only when the invalidation closure is local."""
    if not 0.0 <= maximum_incremental_node_fraction <= 1.0:
        raise ValueError("incremental node fraction must lie in [0, 1]")
    _, _, changed = _changed_edge_identities(old_register, new_register)
    affected = _affected_nodes_for_change(new_register, changed)
    if len(affected) > maximum_incremental_node_fraction * new_register.node_count:
        full = new_register.solve(emission, absolute_cutoff=absolute_cutoff)
        return PlannedUpdateResult(
            full.state,
            "full-remarch",
            affected,
            full.edge_applications,
            full.residual_events,
        )
    incremental = antimatter_update(
        old_register,
        new_register,
        old_state,
        absolute_cutoff=absolute_cutoff,
    )
    return PlannedUpdateResult(
        incremental.state,
        "antimatter",
        incremental.affected_nodes,
        incremental.edge_applications,
        incremental.residual_events,
    )
