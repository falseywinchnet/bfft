"""Closure of positive observation transports under composition.

The unknown-image model is one row-stochastic transport, not a pipeline label::

    y(p) = integral x(q) K(p, dq).

If two physical or digital transports occurred, their Chapman--Kolmogorov
composition is still one ``K``.  This module performs that composition exactly
for the discrete reflected/bilinear operators used by the deblurrer.  It also
provides affine positive measures, for which radial scale exposure has an
additive log-scale chart.

Nothing here classifies an image as warp, blur, resampling, inner, or outer.
The known generators are synthetic/oracle controls for the representation that
a blind estimator must eventually recover directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decomposition import image_fingerprint
from .spatial_transport import (
    SpatialExposureField,
    SpatialReflectedExposureOperator,
    _reflect_indices,
)


@dataclass(frozen=True)
class LocalObservationJet:
    """First and second local moments of one consolidated row measure."""

    barycentric_displacement_xy: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    principal_direction_xy: np.ndarray
    supported: np.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ConsolidatedInverseResult:
    image: np.ndarray
    adjoint_seed: np.ndarray
    uncertainty: np.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ObservationBounds:
    """Per-sample admissible preimage interval and continuous authority."""

    lower: np.ndarray
    upper: np.ndarray
    precision: np.ndarray
    diagnostics: dict[str, object]

    def validate(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        measured = np.asarray(observation, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        precision = np.asarray(self.precision, dtype=np.float64)
        if lower.shape != measured.shape or upper.shape != measured.shape:
            raise ValueError("observation bounds must match the measured raster")
        if precision.shape == measured.shape[:2] and measured.ndim == 3:
            precision = np.broadcast_to(precision[..., None], measured.shape)
        if precision.shape != measured.shape:
            raise ValueError("observation precision must match pixels or samples")
        if (
            np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(~np.isfinite(precision))
            or np.any(lower > upper)
            or np.any(precision < 0.0)
        ):
            raise ValueError("observation bounds must be finite, ordered, and positive")
        return lower, upper, precision


def _local_moment_jet_from_transport(transport) -> LocalObservationJet:
    """Derive row moments by transporting coordinate monomials."""
    height, width = transport.shape
    yy, xx = np.mgrid[:height, :width]
    source_x = transport.forward(xx.astype(np.float64))
    source_y = transport.forward(yy.astype(np.float64))
    source_xx = transport.forward((xx * xx).astype(np.float64))
    source_xy = transport.forward((xx * yy).astype(np.float64))
    source_yy = transport.forward((yy * yy).astype(np.float64))
    mean = np.stack((xx - source_x, yy - source_y), axis=-1)
    covariance = np.empty((height, width, 2, 2), dtype=np.float64)
    covariance[..., 0, 0] = np.maximum(
        source_xx - source_x * source_x, 0.0)
    covariance[..., 0, 1] = source_xy - source_x * source_y
    covariance[..., 1, 0] = covariance[..., 0, 1]
    covariance[..., 1, 1] = np.maximum(
        source_yy - source_y * source_y, 0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance.reshape((-1, 2, 2)))
    eigenvalues = eigenvalues[:, ::-1]
    principal = eigenvectors[:, :, 1]
    pivot_x = np.abs(principal[:, 0]) >= np.abs(principal[:, 1])
    sign = np.where(
        pivot_x,
        np.sign(principal[:, 0]),
        np.sign(principal[:, 1]),
    )
    sign[sign == 0.0] = 1.0
    principal *= sign[:, None]
    tolerance = (
        np.finfo(float).eps
        * max(transport.contribution_count, 1)
        * np.maximum(eigenvalues[:, 0], 1.0)
    )
    supported = eigenvalues[:, 0] > tolerance
    principal[~supported] = 0.0
    return LocalObservationJet(
        barycentric_displacement_xy=mean,
        covariance=covariance,
        eigenvalues=eigenvalues.reshape((height, width, 2)),
        principal_direction_xy=principal.reshape((height, width, 2)),
        supported=supported.reshape((height, width)),
        diagnostics={
            "basis": "local_moment_jet_of_one_positive_row_measure",
            "family_classification": False,
            "contribution_count": transport.contribution_count,
            "supported_fraction": float(np.mean(supported)),
            "direction_origin": "principal_axis_of_local_second_cumulant",
            "moment_evaluation": "transported_coordinate_monomials",
        },
    )


class PositiveObservationTransport:
    """Finite row-stochastic observation kernel with exact transpose scatter."""

    def __init__(
        self,
        name: str,
        shape: tuple[int, int],
        source_indices: np.ndarray,
        coefficients: np.ndarray,
    ) -> None:
        height, width = map(int, shape)
        indices = np.asarray(source_indices, dtype=np.int64)
        weights = np.asarray(coefficients, dtype=np.float64)
        pixels = height * width
        if indices.ndim != 2 or indices.shape[1] != pixels:
            raise ValueError("source indices must have shape Cx(HW)")
        if weights.shape != indices.shape:
            raise ValueError("transport coefficients must match source indices")
        if np.any(indices < 0) or np.any(indices >= pixels):
            raise ValueError("transport source index outside its raster")
        if np.any(~np.isfinite(weights)) or np.any(weights < -1e-15):
            raise ValueError("transport coefficients must be finite and positive")
        weights = np.maximum(weights, 0.0)
        row_mass = np.sum(weights, axis=0)
        if np.any(row_mass <= 0.0):
            raise ValueError("every observation row needs positive mass")
        # Preserve exact operator composition while correcting only accumulated
        # floating mass roundoff.  This is normalization, not fitted evidence.
        weights = weights / row_mass[None, :]
        self.name = str(name)
        self.shape = (height, width)
        self.source_indices = np.ascontiguousarray(indices)
        self.coefficients = np.ascontiguousarray(weights)

    @property
    def contribution_count(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def storage_bytes(self) -> int:
        return int(self.source_indices.nbytes + self.coefficients.nbytes)

    @classmethod
    def identity(
        cls,
        shape: tuple[int, int],
    ) -> "PositiveObservationTransport":
        pixels = int(shape[0]) * int(shape[1])
        return cls(
            "identity_observation_transport",
            shape,
            np.arange(pixels, dtype=np.int64)[None, :],
            np.ones((1, pixels), dtype=np.float64),
        )

    @classmethod
    def from_spatial_field(
        cls,
        field: SpatialExposureField,
    ) -> "PositiveObservationTransport":
        operator = SpatialReflectedExposureOperator(field)
        indices = operator._source_indices
        if operator._coefficients is None:
            assert operator._scalar_coefficients is not None
            coefficients = np.broadcast_to(
                operator._scalar_coefficients[:, None], indices.shape)
        else:
            coefficients = operator._coefficients
        return cls(field.name, field.shape, indices, coefficients)

    def _validate(self, image: np.ndarray) -> np.ndarray:
        value = np.asarray(image, dtype=np.float64)
        if value.ndim not in (2, 3) or value.shape[:2] != self.shape:
            raise ValueError("observation transport image shape mismatch")
        return value

    def forward(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        channels = 1 if value.ndim == 2 else value.shape[2]
        flat = value.reshape((-1, channels))
        output = np.zeros_like(flat)
        for source, coefficient in zip(
            self.source_indices, self.coefficients
        ):
            output += coefficient[:, None] * flat[source]
        return output.reshape(value.shape)

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        channels = 1 if value.ndim == 2 else value.shape[2]
        flat = value.reshape((-1, channels))
        output = np.zeros_like(flat)
        for source, coefficient in zip(
            self.source_indices, self.coefficients
        ):
            for channel in range(channels):
                output[:, channel] += np.bincount(
                    source,
                    weights=coefficient * flat[:, channel],
                    minlength=flat.shape[0],
                )
        return output.reshape(value.shape)

    def adjoint_normalization(self, channels: int | None = None) -> np.ndarray:
        shape = self.shape if channels is None else (*self.shape, int(channels))
        return self.adjoint(np.ones(shape, dtype=np.float64))

    def local_moment_jet(self) -> LocalObservationJet:
        """Measure local flow direction without selecting a transport family."""
        return _local_moment_jet_from_transport(self)

    def diagnostics(self) -> dict[str, object]:
        row_mass = np.sum(self.coefficients, axis=0)
        return {
            "formation": "one_consolidated_positive_observation_transport",
            "composition_law": "chapman_kolmogorov_row_measure",
            "family_classification": False,
            "contribution_count": self.contribution_count,
            "row_mass_range": [float(np.min(row_mass)), float(np.max(row_mass))],
            "storage_bytes": self.storage_bytes,
        }


class AffinePositiveObservationTransport:
    """Matrix-free positive gather/scatter for a finite affine-map measure."""

    def __init__(
        self,
        measure: "AffineObservationMeasure",
        shape: tuple[int, int],
    ) -> None:
        self.measure = measure
        self.name = measure.name
        self.shape = (int(shape[0]), int(shape[1]))

    @property
    def contribution_count(self) -> int:
        # Each continuous atom has at most four bilinear footprint atoms.
        return 4 * self.measure.atom_count

    @property
    def storage_bytes(self) -> int:
        return int(
            self.measure.matrices.nbytes
            + self.measure.offsets_xy.nbytes
            + self.measure.weights.nbytes)

    def _validate(self, image: np.ndarray) -> np.ndarray:
        value = np.asarray(image, dtype=np.float64)
        if value.ndim not in (2, 3) or value.shape[:2] != self.shape:
            raise ValueError("affine observation transport image shape mismatch")
        return value

    def _contributions(self):
        height, width = self.shape
        grid_y, grid_x = np.mgrid[:height, :width]
        destination = np.stack((grid_x, grid_y), axis=-1).astype(np.float64)
        for matrix, offset, atom_weight in zip(
            self.measure.matrices,
            self.measure.offsets_xy,
            self.measure.weights,
        ):
            source = np.einsum("ij,hwj->hwi", matrix, destination)
            source += offset
            source_x = source[..., 0]
            source_y = source[..., 1]
            x0 = np.floor(source_x).astype(np.int64)
            y0 = np.floor(source_y).astype(np.int64)
            fraction_x = source_x - x0
            fraction_y = source_y - y0
            for dx, dy, interpolation in (
                (0, 0, (1.0 - fraction_x) * (1.0 - fraction_y)),
                (1, 0, fraction_x * (1.0 - fraction_y)),
                (0, 1, (1.0 - fraction_x) * fraction_y),
                (1, 1, fraction_x * fraction_y),
            ):
                source_index = (
                    _reflect_indices(y0 + dy, height) * width
                    + _reflect_indices(x0 + dx, width)
                ).ravel()
                coefficient = (atom_weight * interpolation).ravel()
                yield source_index, coefficient

    def forward(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        channels = 1 if value.ndim == 2 else value.shape[2]
        flat = value.reshape((-1, channels))
        output = np.zeros_like(flat)
        for source, coefficient in self._contributions():
            output += coefficient[:, None] * flat[source]
        return output.reshape(value.shape)

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        channels = 1 if value.ndim == 2 else value.shape[2]
        flat = value.reshape((-1, channels))
        output = np.zeros_like(flat)
        for source, coefficient in self._contributions():
            for channel in range(channels):
                output[:, channel] += np.bincount(
                    source,
                    weights=coefficient * flat[:, channel],
                    minlength=flat.shape[0],
                )
        return output.reshape(value.shape)

    def adjoint_normalization(self, channels: int | None = None) -> np.ndarray:
        shape = self.shape if channels is None else (*self.shape, int(channels))
        return self.adjoint(np.ones(shape, dtype=np.float64))

    def local_moment_jet(self) -> LocalObservationJet:
        return _local_moment_jet_from_transport(self)

    def diagnostics(self) -> dict[str, object]:
        return {
            "formation": "one_matrix_free_affine_observation_transport",
            "composition_law": "positive_affine_row_measure",
            "family_classification": False,
            "contribution_count": self.contribution_count,
            "storage_bytes": self.storage_bytes,
            "matrix_free": True,
        }


class ComposedPositiveObservationTransport:
    """Lazy exact Chapman--Kolmogorov operator with bounded storage."""

    def __init__(self, first, second, *, name: str | None = None) -> None:
        if first.shape != second.shape:
            raise ValueError("composed transports must share one raster")
        self.first = first
        self.second = second
        self.name = name or f"composed_{second.name}_after_{first.name}"
        self.shape = first.shape

    @property
    def contribution_count(self) -> int:
        return self.first.contribution_count * self.second.contribution_count

    @property
    def storage_bytes(self) -> int:
        # Shared factors are counted once when the same object is composed.
        if self.first is self.second:
            return self.first.storage_bytes
        return self.first.storage_bytes + self.second.storage_bytes

    def _validate(self, image: np.ndarray) -> np.ndarray:
        return self.first._validate(image)

    def forward(self, image: np.ndarray) -> np.ndarray:
        return self.second.forward(self.first.forward(image))

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        return self.first.adjoint(self.second.adjoint(image))

    def adjoint_normalization(self, channels: int | None = None) -> np.ndarray:
        shape = self.shape if channels is None else (*self.shape, int(channels))
        return self.adjoint(np.ones(shape, dtype=np.float64))

    def local_moment_jet(self) -> LocalObservationJet:
        return _local_moment_jet_from_transport(self)

    def diagnostics(self) -> dict[str, object]:
        return {
            "formation": "one_lazy_consolidated_positive_observation_transport",
            "composition_law": "chapman_kolmogorov_row_measure",
            "family_classification": False,
            "factorization_exposed_to_inverse": False,
            "contribution_count": self.contribution_count,
            "storage_bytes": self.storage_bytes,
            "matrix_free_composition": True,
        }


def compose_positive_transports(
    first,
    second,
    *,
    name: str | None = None,
) -> PositiveObservationTransport | ComposedPositiveObservationTransport:
    """Return ``second(first(x))`` as one exact discrete positive transport."""
    if first.shape != second.shape:
        raise ValueError("composed transports must share one raster")
    if not (
        isinstance(first, PositiveObservationTransport)
        and isinstance(second, PositiveObservationTransport)
    ):
        return ComposedPositiveObservationTransport(first, second, name=name)
    materialized_entries = (
        first.contribution_count
        * second.contribution_count
        * first.shape[0]
        * first.shape[1]
    )
    if materialized_entries > 2_000_000:
        return ComposedPositiveObservationTransport(first, second, name=name)
    source_rows: list[np.ndarray] = []
    coefficient_rows: list[np.ndarray] = []
    for second_source, second_weight in zip(
        second.source_indices, second.coefficients
    ):
        for first_source, first_weight in zip(
            first.source_indices, first.coefficients
        ):
            source_rows.append(first_source[second_source])
            coefficient_rows.append(
                second_weight * first_weight[second_source])
    return PositiveObservationTransport(
        name or f"composed_{second.name}_after_{first.name}",
        first.shape,
        np.stack(source_rows),
        np.stack(coefficient_rows),
    )


@dataclass(frozen=True)
class AffineObservationMeasure:
    """Positive probability measure over affine destination-to-source maps."""

    name: str
    matrices: np.ndarray
    offsets_xy: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        matrices = np.asarray(self.matrices, dtype=np.float64)
        offsets = np.asarray(self.offsets_xy, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (2, 2):
            raise ValueError("affine matrices must have shape Kx2x2")
        if offsets.shape != (len(matrices), 2) or weights.shape != (len(matrices),):
            raise ValueError("affine offsets and weights must match atom count")
        if (
            np.any(~np.isfinite(matrices))
            or np.any(~np.isfinite(offsets))
            or np.any(~np.isfinite(weights))
            or np.any(weights < -1e-15)
        ):
            raise ValueError("affine observation measure must be finite and positive")
        mass = float(np.sum(np.maximum(weights, 0.0)))
        if mass <= 0.0:
            raise ValueError("affine observation measure needs positive mass")
        object.__setattr__(self, "matrices", np.ascontiguousarray(matrices))
        object.__setattr__(self, "offsets_xy", np.ascontiguousarray(offsets))
        object.__setattr__(
            self, "weights", np.ascontiguousarray(np.maximum(weights, 0.0) / mass))

    @property
    def atom_count(self) -> int:
        return int(len(self.weights))

    def to_spatial_field(self, shape: tuple[int, int]) -> SpatialExposureField:
        height, width = map(int, shape)
        yy, xx = np.mgrid[:height, :width]
        destination = np.stack((xx, yy), axis=-1).astype(np.float64)
        source = np.einsum(
            "kij,hwj->khwi", self.matrices, destination, optimize=True)
        source += self.offsets_xy[:, None, None, :]
        displacement = destination[None, ...] - source
        weights = np.broadcast_to(
            self.weights[:, None, None], (self.atom_count, height, width))
        return SpatialExposureField(
            self.name,
            np.ascontiguousarray(displacement),
            np.ascontiguousarray(weights),
        )

    def to_transport(self, shape: tuple[int, int]) -> AffinePositiveObservationTransport:
        return AffinePositiveObservationTransport(self, shape)

    def diagnostics(self) -> dict[str, object]:
        singular_values = np.linalg.svd(self.matrices, compute_uv=False)
        determinant = np.linalg.det(self.matrices)
        return {
            "formation": "positive_measure_over_affine_maps",
            "family_classification": False,
            "atom_count": self.atom_count,
            "determinant_range": [
                float(np.min(determinant)), float(np.max(determinant))],
            "log_singular_value_atoms": np.log(np.maximum(
                singular_values, np.finfo(float).tiny)).tolist(),
        }


def compose_affine_measures(
    first: AffineObservationMeasure,
    second: AffineObservationMeasure,
    *,
    name: str | None = None,
) -> AffineObservationMeasure:
    """Return the positive affine law for ``second(first(x))``.

    If ``q=A p+a`` is the destination-to-source map, composition gives
    ``A_first A_second p + A_first a_second + a_first``.  Radial scale atoms
    therefore multiply, and their logarithms add without a stage decision.
    """
    matrices = []
    offsets = []
    weights = []
    for second_matrix, second_offset, second_weight in zip(
        second.matrices, second.offsets_xy, second.weights
    ):
        for first_matrix, first_offset, first_weight in zip(
            first.matrices, first.offsets_xy, first.weights
        ):
            matrices.append(first_matrix @ second_matrix)
            offsets.append(first_matrix @ second_offset + first_offset)
            weights.append(first_weight * second_weight)
    return AffineObservationMeasure(
        name or f"composed_{second.name}_after_{first.name}",
        np.stack(matrices),
        np.stack(offsets),
        np.asarray(weights),
    )


def radial_scale_measure(
    shape: tuple[int, int],
    *,
    fractional_extent: float = 0.045,
    center_xy: tuple[float, float] | None = None,
    stages: int = 1,
) -> AffineObservationMeasure:
    """Wronski-binomial exposure on radial scale maps about one center."""
    height, width = map(int, shape)
    center = np.asarray(
        (0.5 * (width - 1), 0.5 * (height - 1))
        if center_xy is None else center_xy,
        dtype=np.float64,
    )
    extent = float(fractional_extent)
    if not np.isfinite(extent) or extent <= 0.0 or extent >= 1.0:
        raise ValueError("radial fractional extent must lie in (0,1)")
    scales = np.asarray((1.0 - extent, 1.0, 1.0 + extent))
    matrices = scales[:, None, None] * np.eye(2)[None, ...]
    offsets = (1.0 - scales[:, None]) * center[None, :]
    measure = AffineObservationMeasure(
        f"radial_scale_binomial_extent_{extent:g}",
        matrices,
        offsets,
        np.asarray((0.25, 0.5, 0.25)),
    )
    for stage in range(1, max(int(stages), 1)):
        measure = compose_affine_measures(
            measure,
            AffineObservationMeasure(
                f"radial_scale_stage_{stage + 1}",
                matrices,
                offsets,
                np.asarray((0.25, 0.5, 0.25)),
            ),
            name=f"radial_scale_binomial_extent_{extent:g}_stages_{stage + 1}",
        )
    return measure


def refine_consolidated_transport(
    observation: np.ndarray,
    transport: PositiveObservationTransport,
    *,
    passes: int = 64,
    ratio_limit: float = 4.0,
    descent_method: str = "optimal_positive_line",
    observation_bounds: ObservationBounds | None = None,
) -> ConsolidatedInverseResult:
    """Invert one supplied row measure without decomposing or classifying it."""
    measured = transport._validate(observation)
    before = image_fingerprint(measured)
    channels = None if measured.ndim == 2 else measured.shape[2]
    if observation_bounds is None:
        lower = measured
        upper = measured
        precision = np.ones_like(measured)
        bounds_diagnostics = {
            "method": "exact_sample_equality",
            "interval_censored": False,
        }
    else:
        lower, upper, precision = observation_bounds.validate(measured)
        bounds_diagnostics = {
            **observation_bounds.diagnostics,
            "interval_censored": True,
        }
    if not np.any(precision > 0.0):
        raise ValueError("consolidated inverse needs positive observation authority")
    normalization = np.maximum(
        transport.adjoint(precision), 1e-8)
    seed = np.clip(
        transport.adjoint(precision * measured) / normalization,
        1e-8,
        1.0,
    )
    latent = seed.copy()
    prediction = transport.forward(latent)
    limit = max(float(ratio_limit), 1.0)
    if descent_method not in ("multiplicative", "optimal_positive_line"):
        raise ValueError("unknown consolidated transport descent method")
    residual_trace: list[float] = []
    step_trace: list[float] = []
    stopped_by = "maximum_passes"
    for _ in range(max(int(passes), 0)):
        admissible_target = np.clip(prediction, lower, upper)
        ratio = np.clip(
            admissible_target / np.maximum(prediction, 1e-8),
            1.0 / limit,
            limit,
        )
        correction = np.maximum(
            transport.adjoint(precision * ratio) / normalization, 1e-8)
        proposed = np.clip(latent * correction, 0.0, 1.0)
        if descent_method == "optimal_positive_line":
            direction = proposed - latent
            transported_direction = transport.forward(direction)
            numerator = float(np.sum(
                precision
                * (admissible_target - prediction)
                * transported_direction))
            denominator = float(np.sum(
                precision
                * transported_direction * transported_direction))
            step = max(numerator / max(denominator, 1e-20), 0.0)
            negative = direction < 0.0
            if np.any(negative):
                step = min(step, 0.999 * float(np.min(
                    -latent[negative] / direction[negative])))
            positive = direction > 0.0
            if np.any(positive):
                step = min(step, max(0.999 * float(np.min(
                    (1.0 - latent[positive]) / direction[positive])), 0.0))
            latent = latent + step * direction
            prediction = prediction + step * transported_direction
        else:
            step = 1.0
            latent = proposed
            prediction = transport.forward(latent)
        violation = prediction - np.clip(prediction, lower, upper)
        residual_trace.append(float(np.sqrt(
            np.sum(precision * violation * violation)
            / max(float(np.sum(precision)), 1e-20))))
        step_trace.append(float(step))
        if descent_method == "optimal_positive_line" and step <= 1e-6:
            stopped_by = "optimal_positive_line_stationarity"
            break
    violation = prediction - np.clip(prediction, lower, upper)
    transported_residual = transport.adjoint(
        precision * np.abs(violation)) / normalization
    interval_sigma = (upper - lower) / np.sqrt(12.0)
    transported_interval = transport.adjoint(
        precision * interval_sigma) / normalization
    seed_gauge = np.abs(latent - seed)
    uncertainty = np.sqrt(
        transported_residual * transported_residual
        + transported_interval * transported_interval
        + seed_gauge * seed_gauge)
    after = image_fingerprint(measured)
    return ConsolidatedInverseResult(
        image=np.clip(latent, 0.0, 1.0),
        adjoint_seed=seed,
        uncertainty=uncertainty,
        diagnostics={
            "method": "one_consolidated_positive_observation_transport_inverse",
            "operator": transport.diagnostics(),
            "operator_decomposition": False,
            "family_classification": False,
            "descent_method": descent_method,
            "observation_bounds": bounds_diagnostics,
            "observation_authority_fraction": float(np.mean(precision > 0.0)),
            "mean_observation_precision": float(np.mean(precision)),
            "mean_interval_width": float(np.mean(upper - lower)),
            "passes_used": len(residual_trace),
            "stopped_by": stopped_by,
            "residual_trace": residual_trace,
            "step_trace": step_trace,
            "forward_rms": float(np.sqrt(
                np.sum(precision * violation * violation)
                / max(float(np.sum(precision)), 1e-20))),
            "forward_rms_semantics": "distance_to_admissible_observation_interval",
            "uncertainty_rms": float(np.sqrt(np.mean(
                uncertainty * uncertainty))),
            "observation_fingerprint_before": before,
            "observation_fingerprint_after": after,
            "observation_unchanged": bool(before == after),
        },
    )
