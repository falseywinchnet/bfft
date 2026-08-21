"""Spatial positive-exposure transport spanning deterministic warp and mixing.

At each destination pixel ``p`` a positive exposure field stores displacement
atoms ``d_j(p)`` and weights ``w_j(p)``.  The single formation law is

    y(p) = sum_j w_j(p) x(reflect(p - d_j(p))).

One atom is deterministic distortion.  Several coincident spatially constant
atoms are global convolutional blur.  Spatially varying atom clouds cover the
continuous interval between them without selecting a solver family.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .curvilinear import residual_discrepancy
from .kernels import TransportKernel


def _reflect_indices(indices: np.ndarray, size: int) -> np.ndarray:
    count = int(size)
    if count <= 1:
        return np.zeros_like(indices, dtype=np.int64)
    period = 2 * count
    folded = np.mod(np.asarray(indices, dtype=np.int64), period)
    return np.where(folded < count, folded, period - 1 - folded)


def _validated_covariance_components(covariance_field: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance_field, dtype=np.float64)
    if covariance.ndim == 4 and covariance.shape[-2:] == (2, 2):
        if not np.allclose(
            covariance[..., 0, 1], covariance[..., 1, 0],
            atol=1e-12, rtol=1e-12,
        ):
            raise ValueError("covariance exposure field must be symmetric")
        components = np.stack((
            covariance[..., 0, 0],
            0.5 * (covariance[..., 0, 1] + covariance[..., 1, 0]),
            covariance[..., 1, 1],
        ), axis=-1)
    elif covariance.ndim == 3 and covariance.shape[-1] == 3:
        components = covariance
    else:
        raise ValueError(
            "covariance exposure field must have shape HxWx2x2 or HxWx3")
    if np.any(~np.isfinite(components)):
        raise ValueError("covariance exposure field must be finite")
    determinant = (
        components[..., 0] * components[..., 2]
        - components[..., 1] * components[..., 1])
    scale = np.maximum(
        np.maximum(components[..., 0], components[..., 2]), 1.0)
    if (
        np.any(components[..., 0] < -1e-10)
        or np.any(components[..., 2] < -1e-10)
        or np.any(determinant < -1e-10 * scale * scale)
    ):
        raise ValueError("covariance exposure field must be positive")
    return np.ascontiguousarray(components)


@dataclass(frozen=True)
class SpatialExposureField:
    """A normalized positive displacement measure at every image position."""

    name: str
    displacements_xy: np.ndarray
    weights: np.ndarray
    compact_global: bool = False

    def __post_init__(self) -> None:
        displacement = np.asarray(self.displacements_xy, dtype=np.float64)
        weight = np.asarray(self.weights, dtype=np.float64)
        if displacement.ndim != 4 or displacement.shape[-1] != 2:
            raise ValueError("spatial displacements must have shape KxHxWx2")
        if weight.shape != displacement.shape[:3]:
            raise ValueError("spatial weights must have shape KxHxW")
        if np.any(~np.isfinite(displacement)) or np.any(~np.isfinite(weight)):
            raise ValueError("spatial exposure fields must be finite")
        if np.any(weight < -1e-15):
            raise ValueError("spatial exposure weights must be non-negative")
        mass = np.sum(np.maximum(weight, 0.0), axis=0)
        if np.any(mass <= 0.0):
            raise ValueError("every spatial exposure measure needs positive mass")
        normalized = np.maximum(weight, 0.0) / mass[None, ...]
        object.__setattr__(
            self, "displacements_xy", np.ascontiguousarray(displacement))
        object.__setattr__(self, "weights", np.ascontiguousarray(normalized))

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.weights.shape[1]), int(self.weights.shape[2]))

    @property
    def atom_count(self) -> int:
        return int(self.weights.shape[0])

    @property
    def barycentric_flow_xy(self) -> np.ndarray:
        return np.sum(
            self.weights[..., None] * self.displacements_xy, axis=0)

    @property
    def centered_displacements_xy(self) -> np.ndarray:
        return self.displacements_xy - self.barycentric_flow_xy[None, ...]

    @property
    def sensor_to_latent_jacobian_determinant(self) -> np.ndarray:
        """Determinant of ``p -> p - m(p)`` for the barycentric flow."""
        flow = self.barycentric_flow_xy
        dmx_dx = np.gradient(flow[..., 0], axis=1)
        dmx_dy = np.gradient(flow[..., 0], axis=0)
        dmy_dx = np.gradient(flow[..., 1], axis=1)
        dmy_dy = np.gradient(flow[..., 1], axis=0)
        return (
            (1.0 - dmx_dx) * (1.0 - dmy_dy) - dmx_dy * dmy_dx)

    @classmethod
    def from_global_kernel(
        cls,
        kernel: TransportKernel,
        shape: tuple[int, int],
    ) -> "SpatialExposureField":
        """Lift a global positive PSF into the spatial operator exactly."""
        height, width = map(int, shape)
        yy, xx = np.mgrid[: kernel.psf.shape[0], : kernel.psf.shape[1]]
        center = 0.5 * (np.asarray(kernel.psf.shape) - 1.0)
        mask = kernel.psf > 0.0
        atoms = np.column_stack((
            xx[mask] - center[1],
            yy[mask] - center[0],
        )).astype(np.float64)
        mass = kernel.psf[mask]
        displacement = np.broadcast_to(
            atoms[:, None, None, :], (len(atoms), height, width, 2))
        weights = np.broadcast_to(
            mass[:, None, None], (len(atoms), height, width))
        return cls(
            name=f"spatial_lift_{kernel.name}",
            displacements_xy=np.ascontiguousarray(displacement),
            weights=np.ascontiguousarray(weights),
        )

    @classmethod
    def from_barycentric_paths(
        cls,
        name: str,
        barycentric_flow_xy: np.ndarray,
        residual_displacements_xy: np.ndarray,
        weights: np.ndarray,
        compact_global: bool = False,
    ) -> "SpatialExposureField":
        """Compose a deterministic flow field with centered exposure atoms."""
        flow = np.asarray(barycentric_flow_xy, dtype=np.float64)
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError("barycentric flow must have shape HxWx2")
        residual = np.asarray(residual_displacements_xy, dtype=np.float64)
        if residual.ndim == 2 and residual.shape[1] == 2:
            residual = np.broadcast_to(
                residual[:, None, None, :],
                (len(residual), flow.shape[0], flow.shape[1], 2),
            )
        if residual.ndim != 4 or residual.shape[1:] != flow.shape:
            raise ValueError("residual paths must have shape KxHxWx2 or Kx2")
        mass = np.asarray(weights, dtype=np.float64)
        if mass.ndim == 1 and len(mass) == len(residual):
            mass = np.broadcast_to(
                mass[:, None, None], residual.shape[:3])
        displacement = flow[None, ...] + residual
        return cls(name, displacement, mass, compact_global=compact_global)

    def barycentric_field(self) -> "SpatialExposureField":
        flow = self.barycentric_flow_xy
        return SpatialExposureField(
            name=f"barycentric_{self.name}",
            displacements_xy=flow[None, ...],
            weights=np.ones((1, *self.shape), dtype=np.float64),
        )

    def diagnostics(self) -> dict[str, float | int | str]:
        flow = self.barycentric_flow_xy
        centered = self.centered_displacements_xy
        variance = np.sum(
            self.weights * np.sum(centered * centered, axis=-1), axis=0)
        flow_energy = np.sum(flow * flow, axis=-1)
        determinant = self.sensor_to_latent_jacobian_determinant
        return {
            "formation": "one_spatial_positive_exposure_measure",
            "atom_count": self.atom_count,
            "barycentric_flow_rms": float(np.sqrt(np.mean(flow_energy))),
            "centered_mixing_rms": float(np.sqrt(np.mean(variance))),
            "mapping_jacobian_min": float(np.min(determinant)),
            "mapping_jacobian_max": float(np.max(determinant)),
            "fold_fraction": float(np.mean(determinant <= 0.0)),
        }


@dataclass(frozen=True)
class CompactGlobalExposureField:
    """A translated global positive measure stored without spatial broadcast."""

    name: str
    raster_shape: tuple[int, int]
    residual_displacements_xy: np.ndarray
    residual_weights: np.ndarray
    barycentric_translation_xy: np.ndarray | None = None

    def __post_init__(self) -> None:
        height, width = map(int, self.raster_shape)
        points = np.asarray(self.residual_displacements_xy, dtype=np.float64)
        weights = np.asarray(self.residual_weights, dtype=np.float64)
        translation = np.asarray(
            np.zeros(2, dtype=np.float64)
            if self.barycentric_translation_xy is None
            else self.barycentric_translation_xy,
            dtype=np.float64,
        )
        if height < 2 or width < 2:
            raise ValueError("compact global exposure needs a 2-D raster")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("compact global paths must have shape Kx2")
        if weights.shape != (len(points),) or np.any(weights < -1e-15):
            raise ValueError("compact global weights must be K nonnegative values")
        if np.any(~np.isfinite(points)) or np.any(~np.isfinite(weights)):
            raise ValueError("compact global exposure must be finite")
        if translation.shape != (2,) or np.any(~np.isfinite(translation)):
            raise ValueError("compact global translation must be two finite values")
        mass = float(np.sum(np.maximum(weights, 0.0)))
        if mass <= 0.0:
            raise ValueError("compact global exposure needs positive mass")
        normalized = np.maximum(weights, 0.0) / mass
        centroid = np.sum(normalized[:, None] * points, axis=0)
        object.__setattr__(self, "raster_shape", (height, width))
        object.__setattr__(
            self, "residual_displacements_xy",
            np.ascontiguousarray(points - centroid[None, :]))
        object.__setattr__(
            self, "residual_weights", np.ascontiguousarray(normalized))
        object.__setattr__(
            self, "barycentric_translation_xy",
            np.ascontiguousarray(translation))

    @property
    def shape(self) -> tuple[int, int]:
        return self.raster_shape

    @property
    def atom_count(self) -> int:
        return len(self.residual_weights)

    @property
    def barycentric_flow_xy(self) -> np.ndarray:
        return np.broadcast_to(
            self.barycentric_translation_xy,
            (*self.raster_shape, 2),
        )

    @property
    def weights(self) -> np.ndarray:
        """Broadcast view retained for positive-measure diagnostics."""
        return np.broadcast_to(
            self.residual_weights[:, None, None],
            (self.atom_count, *self.raster_shape),
        )

    @property
    def centered_displacements_xy(self) -> np.ndarray:
        """Broadcast view retained for moment audits without storage cost."""
        return np.broadcast_to(
            self.residual_displacements_xy[:, None, None, :],
            (self.atom_count, *self.raster_shape, 2),
        )

    def centered_field(self) -> "CompactGlobalExposureField":
        return CompactGlobalExposureField(
            name=f"centered_{self.name}",
            raster_shape=self.raster_shape,
            residual_displacements_xy=self.residual_displacements_xy,
            residual_weights=self.residual_weights,
        )

    def barycentric_field(self) -> SpatialExposureField:
        return SpatialExposureField.from_barycentric_paths(
            name=f"barycentric_{self.name}",
            barycentric_flow_xy=np.broadcast_to(
                self.barycentric_translation_xy,
                (*self.raster_shape, 2),
            ),
            residual_displacements_xy=np.zeros((1, 2), dtype=np.float64),
            weights=np.ones(1, dtype=np.float64),
            compact_global=True,
        )

    def diagnostics(self) -> dict[str, float | int | str]:
        variance = float(np.sum(
            self.residual_weights
            * np.sum(self.residual_displacements_xy ** 2, axis=1)))
        return {
            "formation": "compact_global_positive_exposure_measure",
            "atom_count": self.atom_count,
            "barycentric_flow_rms": float(np.linalg.norm(
                self.barycentric_translation_xy)),
            "centered_mixing_rms": float(np.sqrt(variance)),
            "mapping_jacobian_min": 1.0,
            "mapping_jacobian_max": 1.0,
            "fold_fraction": 0.0,
            "stored_bytes": int(
                self.residual_displacements_xy.nbytes
                + self.residual_weights.nbytes
                + self.barycentric_translation_xy.nbytes),
        }


@dataclass(frozen=True)
class CovarianceExposureField:
    """Memory-compact positive mixing covariance plus deterministic flow."""

    name: str
    barycentric_flow_xy: np.ndarray
    covariance_components: np.ndarray
    axis_side_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        flow = np.asarray(self.barycentric_flow_xy, dtype=np.float64)
        covariance = np.asarray(self.covariance_components, dtype=np.float64)
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError("covariance exposure flow must have shape HxWx2")
        if covariance.ndim == 4 and covariance.shape[-2:] == (2, 2):
            covariance = np.stack((
                covariance[..., 0, 0],
                0.5 * (covariance[..., 0, 1] + covariance[..., 1, 0]),
                covariance[..., 1, 1],
            ), axis=-1)
        if covariance.shape != (*flow.shape[:2], 3):
            raise ValueError("covariance exposure must have shape HxWx3 or HxWx2x2")
        components = _validated_covariance_components(covariance)
        side_weights = np.ascontiguousarray(
            (np.full(2, 1.0 / 6.0)
             if self.axis_side_weights is None else self.axis_side_weights),
            dtype=np.float64,
        )
        if (
            side_weights.shape not in ((2,), (*flow.shape[:2], 2))
            or np.any(side_weights <= 0.0)
            or np.any(side_weights >= 0.5)
        ):
            raise ValueError("axis side weights must be two values in (0,1/2)")
        object.__setattr__(self, "barycentric_flow_xy", np.ascontiguousarray(flow))
        object.__setattr__(
            self, "covariance_components", components.copy())
        object.__setattr__(self, "axis_side_weights", side_weights)

    @property
    def shape(self) -> tuple[int, int]:
        return self.barycentric_flow_xy.shape[:2]

    @property
    def atom_count(self) -> int:
        return 9

    @property
    def sensor_to_latent_jacobian_determinant(self) -> np.ndarray:
        flow = self.barycentric_flow_xy
        dmx_dx = np.gradient(flow[..., 0], axis=1)
        dmx_dy = np.gradient(flow[..., 0], axis=0)
        dmy_dx = np.gradient(flow[..., 1], axis=1)
        dmy_dy = np.gradient(flow[..., 1], axis=0)
        return (
            (1.0 - dmx_dx) * (1.0 - dmy_dy) - dmx_dy * dmy_dx)

    def barycentric_field(self) -> SpatialExposureField:
        return SpatialExposureField(
            name=f"barycentric_{self.name}",
            displacements_xy=self.barycentric_flow_xy[None, ...],
            weights=np.ones((1, *self.shape), dtype=np.float64),
        )

    def diagnostics(self) -> dict[str, float | int | str]:
        flow_energy = np.sum(self.barycentric_flow_xy ** 2, axis=-1)
        trace = self.covariance_components[..., 0] + self.covariance_components[..., 2]
        determinant = self.sensor_to_latent_jacobian_determinant
        return {
            "formation": "generated_nine_atom_positive_covariance_measure",
            "atom_count": 9,
            "barycentric_flow_rms": float(np.sqrt(np.mean(flow_energy))),
            "centered_mixing_rms": float(np.sqrt(np.mean(trace))),
            "mapping_jacobian_min": float(np.min(determinant)),
            "mapping_jacobian_max": float(np.max(determinant)),
            "fold_fraction": float(np.mean(determinant <= 0.0)),
            "stored_bytes": int(
                self.barycentric_flow_xy.nbytes
                + self.covariance_components.nbytes
                + self.axis_side_weights.nbytes),
            "axis_side_weight_min": np.min(
                self.axis_side_weights, axis=tuple(
                    range(self.axis_side_weights.ndim - 1))).tolist(),
            "axis_side_weight_max": np.max(
                self.axis_side_weights, axis=tuple(
                    range(self.axis_side_weights.ndim - 1))).tolist(),
        }


class SpatialReflectedExposureOperator:
    """Exact bilinear gather and matched scatter for a spatial exposure field."""

    backend = "numpy_spatial_bilinear_gather_bincount_scatter"

    def __init__(self, field: SpatialExposureField) -> None:
        self.field = field
        self.shape = field.shape
        self.backend = "numpy_spatial_bilinear_gather_bincount_scatter"
        height, width = self.shape
        grid_y, grid_x = np.mgrid[:height, :width]
        source_indices: list[np.ndarray] = []
        coefficients: list[np.ndarray] = []
        scalar_coefficients: list[float] = []
        global_measure = bool(
            field.compact_global
            and np.all(field.displacements_xy == field.displacements_xy[:, :1, :1])
            and np.all(field.weights == field.weights[:, :1, :1])
        )
        for atom in range(field.atom_count):
            if global_measure:
                displacement = field.displacements_xy[atom, 0, 0]
                # Decompose one global offset once. Computing a fractional
                # coordinate as ``grid-offset-floor(grid-offset)`` makes it
                # vary with grid magnitude through floating cancellation.
                offset_x = -float(displacement[0])
                offset_y = -float(displacement[1])
                base_x = int(np.floor(offset_x))
                base_y = int(np.floor(offset_y))
                x0 = grid_x + base_x
                y0 = grid_y + base_y
                fx = offset_x - base_x
                fy = offset_y - base_y
                atom_weight: np.ndarray | float = float(field.weights[atom, 0, 0])
            else:
                source_x = grid_x - field.displacements_xy[atom, ..., 0]
                source_y = grid_y - field.displacements_xy[atom, ..., 1]
                atom_weight = field.weights[atom]
                x0 = np.floor(source_x).astype(np.int64)
                y0 = np.floor(source_y).astype(np.int64)
                fx = source_x - x0
                fy = source_y - y0
            for dx, dy, interpolation in (
                (0, 0, (1.0 - fx) * (1.0 - fy)),
                (1, 0, fx * (1.0 - fy)),
                (0, 1, (1.0 - fx) * fy),
                (1, 1, fx * fy),
            ):
                ix = _reflect_indices(x0 + dx, width)
                iy = _reflect_indices(y0 + dy, height)
                source_indices.append(np.ascontiguousarray(
                    (iy * width + ix).ravel()))
                if global_measure:
                    coefficient_scalar = float(
                        atom_weight * interpolation)
                    if coefficient_scalar > 0.0:
                        scalar_coefficients.append(coefficient_scalar)
                    else:
                        source_indices.pop()
                else:
                    coefficient = np.ascontiguousarray(
                        (atom_weight * interpolation).ravel())
                    if np.any(coefficient > 0.0):
                        coefficients.append(coefficient)
                    else:
                        source_indices.pop()
        self._source_indices = np.stack(source_indices, axis=0)
        self._scalar_coefficients = (
            np.asarray(scalar_coefficients, dtype=np.float64)
            if global_measure else None)
        self._coefficients = (
            None if global_measure else np.stack(coefficients, axis=0))
        self._native_plan = None
        try:
            from .native_backend import (
                NativeReflectedPathPlan,
                NativeSpatialExposurePlan,
                native_available,
            )
            if native_available():
                self._native_plan = (
                    NativeReflectedPathPlan(
                        self._source_indices,
                        self._scalar_coefficients,
                        self.shape,
                    )
                    if self._scalar_coefficients is not None else
                    NativeSpatialExposurePlan(
                        self._source_indices,
                        self._coefficients,
                        self.shape,
                    )
                )
                self.backend = self._native_plan.backend
        except (OSError, RuntimeError, ValueError):
            self._native_plan = None

    def _validate(self, image: np.ndarray) -> np.ndarray:
        value = np.asarray(image, dtype=np.float64)
        if value.ndim not in (2, 3) or value.shape[:2] != self.shape:
            raise ValueError("spatial exposure image shape does not match its field")
        return value

    def _forward_numpy(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        channels = 1 if value.ndim == 2 else value.shape[2]
        flat = value.reshape((-1, channels))
        output = np.zeros_like(flat)
        if self._scalar_coefficients is not None:
            for source, coefficient in zip(
                self._source_indices, self._scalar_coefficients
            ):
                output += coefficient * flat[source]
        else:
            assert self._coefficients is not None
            for source, coefficient in zip(
                self._source_indices, self._coefficients
            ):
                output += coefficient[:, None] * flat[source]
        return output.reshape(value.shape)

    def _adjoint_numpy(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        channels = 1 if value.ndim == 2 else value.shape[2]
        flat = value.reshape((-1, channels))
        output = np.zeros_like(flat)
        coefficient_sequence = (
            self._scalar_coefficients
            if self._scalar_coefficients is not None else self._coefficients)
        assert coefficient_sequence is not None
        for source, coefficient in zip(self._source_indices, coefficient_sequence):
            for channel in range(channels):
                output[:, channel] += np.bincount(
                    source,
                    weights=coefficient * flat[:, channel],
                    minlength=flat.shape[0],
                )
        return output.reshape(value.shape)

    def forward(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        if self._native_plan is not None:
            return self._native_plan.forward(value)
        return self._forward_numpy(value)

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        if self._native_plan is not None:
            return self._native_plan.adjoint(value)
        return self._adjoint_numpy(value)

    def adjoint_normalization(self, channels: int | None = None) -> np.ndarray:
        shape = self.shape if channels is None else (*self.shape, int(channels))
        return self.adjoint(np.ones(shape, dtype=np.float64))


class CompactGlobalReflectedExposureOperator:
    """Exact reflected global exposure via one even-extension circular FFT."""

    backend = "numpy_even_reflection_rfft_global_transport"

    def __init__(self, field: CompactGlobalExposureField) -> None:
        self.field = field
        self.shape = field.shape
        height, width = self.shape
        self._extended_shape = (2 * height, 2 * width)
        kernel = np.zeros(self._extended_shape, dtype=np.float64)
        for point, atom_weight in zip(
            field.residual_displacements_xy, field.residual_weights
        ):
            point = point + field.barycentric_translation_xy
            offset_x = -float(point[0])
            offset_y = -float(point[1])
            base_x = int(np.floor(offset_x))
            base_y = int(np.floor(offset_y))
            fraction_x = offset_x - base_x
            fraction_y = offset_y - base_y
            for delta_x, x_weight in (
                (0, 1.0 - fraction_x), (1, fraction_x)):
                for delta_y, y_weight in (
                    (0, 1.0 - fraction_y), (1, fraction_y)):
                    coefficient = float(atom_weight * x_weight * y_weight)
                    shift_x = base_x + delta_x
                    shift_y = base_y + delta_y
                    kernel[
                        (-shift_y) % self._extended_shape[0],
                        (-shift_x) % self._extended_shape[1],
                    ] += coefficient
        self._spectrum = np.fft.rfft2(kernel)
        self._reflect_y = _reflect_indices(
            np.arange(self._extended_shape[0]), height)
        self._reflect_x = _reflect_indices(
            np.arange(self._extended_shape[1]), width)

    @property
    def storage_bytes(self) -> int:
        return int(
            self.field.residual_displacements_xy.nbytes
            + self.field.residual_weights.nbytes
            + self.field.barycentric_translation_xy.nbytes
            + self._spectrum.nbytes
            + self._reflect_y.nbytes
            + self._reflect_x.nbytes)

    def _validate(self, image: np.ndarray) -> np.ndarray:
        value = np.asarray(image, dtype=np.float64)
        if value.ndim not in (2, 3) or value.shape[:2] != self.shape:
            raise ValueError("compact global exposure image shape mismatch")
        return value

    def _multiply_spectrum(
        self,
        transformed: np.ndarray,
        *,
        adjoint: bool,
    ) -> np.ndarray:
        spectrum = np.conjugate(self._spectrum) if adjoint else self._spectrum
        return transformed * (
            spectrum if transformed.ndim == 2 else spectrum[..., None])

    def forward(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        extended = value[
            self._reflect_y[:, None], self._reflect_x[None, :]]
        transformed = np.fft.rfft2(extended, axes=(0, 1))
        filtered = np.fft.irfft2(
            self._multiply_spectrum(transformed, adjoint=False),
            s=self._extended_shape,
            axes=(0, 1),
        )
        return filtered[:self.shape[0], :self.shape[1]]

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        extended_shape = (
            self._extended_shape if value.ndim == 2
            else (*self._extended_shape, value.shape[2]))
        embedded = np.zeros(extended_shape, dtype=np.float64)
        embedded[:self.shape[0], :self.shape[1]] = value
        transformed = np.fft.rfft2(embedded, axes=(0, 1))
        filtered = np.fft.irfft2(
            self._multiply_spectrum(transformed, adjoint=True),
            s=self._extended_shape,
            axes=(0, 1),
        )
        reverse_y = 2 * self.shape[0] - 1 - np.arange(self.shape[0])
        reverse_x = 2 * self.shape[1] - 1 - np.arange(self.shape[1])
        return (
            filtered[:self.shape[0], :self.shape[1]]
            + filtered[:self.shape[0], reverse_x]
            + filtered[reverse_y, :self.shape[1]]
            + filtered[reverse_y[:, None], reverse_x[None, :]]
        )

    def adjoint_normalization(self, channels: int | None = None) -> np.ndarray:
        shape = self.shape if channels is None else (*self.shape, int(channels))
        return self.adjoint(np.ones(shape, dtype=np.float64))


class CompactGlobalExposureOperatorBatch:
    """Bounded parallel execution of independent compact global measures."""

    backend = "numpy_parallel_even_reflection_rfft_global_transport"

    def __init__(
        self,
        operators: tuple[CompactGlobalReflectedExposureOperator, ...],
        *,
        maximum_workers: int = 4,
    ) -> None:
        if not operators:
            raise ValueError("compact global operator batch cannot be empty")
        if any(operator.shape != operators[0].shape for operator in operators):
            raise ValueError("compact global operator batch shapes must match")
        self.operators = operators
        self.shape = operators[0].shape
        self.maximum_workers = min(
            max(int(maximum_workers), 1), len(operators))

    def _validate(self, images: np.ndarray) -> np.ndarray:
        value = np.asarray(images, dtype=np.float64)
        if (
            value.ndim not in (3, 4)
            or value.shape[0] != len(self.operators)
            or value.shape[1:3] != self.shape
        ):
            raise ValueError("compact global batch images must be NxHxW[xC]")
        return value

    def _apply(self, images: np.ndarray, method: str) -> np.ndarray:
        value = self._validate(images)
        with ThreadPoolExecutor(max_workers=self.maximum_workers) as executor:
            outputs = executor.map(
                lambda pair: getattr(pair[0], method)(pair[1]),
                zip(self.operators, value),
            )
            return np.stack(tuple(outputs), axis=0)

    def forward(self, images: np.ndarray) -> np.ndarray:
        return self._apply(images, "forward")

    def adjoint(self, images: np.ndarray) -> np.ndarray:
        return self._apply(images, "adjoint")


class CovarianceReflectedExposureOperator:
    """Positive covariance measure without a materialized spatial atom plan.

    At each pixel, the covariance eigenaxes induce the positive 3x3 sigma
    measure with coordinates ``{-sqrt(3 lambda), 0, +sqrt(3 lambda)}`` and
    weights ``{1/6, 2/3, 1/6}``. Its mass is one, centroid is zero, and second
    moment is exactly the supplied covariance. ABI v6 derives bilinear
    coordinates inside the operator and stores only three covariance
    components per pixel.
    """

    backend = "numpy_generated_positive_covariance_transport"

    def __init__(
        self,
        covariance_field: np.ndarray,
        axis_side_weights: np.ndarray | None = None,
    ) -> None:
        components = _validated_covariance_components(covariance_field)
        self._covariance = components
        self.shape = (int(components.shape[0]), int(components.shape[1]))
        self._side_weights = np.ascontiguousarray(
            (np.full(2, 1.0 / 6.0)
             if axis_side_weights is None else axis_side_weights),
            dtype=np.float64,
        )
        if (
            self._side_weights.shape not in ((2,), (*self.shape, 2))
            or np.any(self._side_weights <= 0.0)
            or np.any(self._side_weights >= 0.5)
        ):
            raise ValueError("axis side weights must be two values in (0,1/2)")
        low_axis, high_axis = self._axis_displacements()
        self._axes = np.ascontiguousarray(np.concatenate(
            (low_axis, high_axis), axis=-1))
        del self._covariance
        self._native_plan = None
        self.backend = "numpy_generated_positive_covariance_transport"
        try:
            from .native_backend import (
                NativeCovarianceExposurePlan,
                native_available,
            )
            if native_available():
                self._native_plan = NativeCovarianceExposurePlan(
                    self._axes, self.shape, self._side_weights)
                self.backend = self._native_plan.backend
        except (OSError, RuntimeError, ValueError):
            self._native_plan = None

    @property
    def storage_bytes(self) -> int:
        return int(self._axes.nbytes + self._side_weights.nbytes)

    def _validate(self, image: np.ndarray) -> np.ndarray:
        value = np.asarray(image, dtype=np.float64)
        if value.ndim not in (2, 3) or value.shape[:2] != self.shape:
            raise ValueError("covariance exposure image shape mismatch")
        return value

    def _axis_displacements(self) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self, "_axes"):
            return self._axes[..., :2], self._axes[..., 2:]
        a = np.maximum(self._covariance[..., 0], 0.0)
        b = self._covariance[..., 1]
        c = np.maximum(self._covariance[..., 2], 0.0)
        middle = 0.5 * (a + c)
        radius = np.hypot(0.5 * (a - c), b)
        low_value = np.maximum(middle - radius, 0.0)
        high_value = np.maximum(middle + radius, 0.0)
        vector_x = b.copy()
        vector_y = high_value - a
        norm = np.hypot(vector_x, vector_y)
        degenerate = norm <= 1e-15
        vector_x = np.where(degenerate, (a >= c).astype(float), vector_x)
        vector_y = np.where(degenerate, (a < c).astype(float), vector_y)
        norm = np.where(degenerate, 1.0, norm)
        low_weight = (
            self._side_weights[0]
            if self._side_weights.ndim == 1 else self._side_weights[..., 0])
        high_weight = (
            self._side_weights[1]
            if self._side_weights.ndim == 1 else self._side_weights[..., 1])
        high_scale = np.sqrt(high_value / (2.0 * high_weight)) / norm
        low_scale = np.sqrt(low_value / (2.0 * low_weight)) / norm
        return (
            np.stack((-low_scale * vector_y, low_scale * vector_x), axis=-1),
            np.stack((high_scale * vector_x, high_scale * vector_y), axis=-1),
        )

    def _contributions(
        self,
    ):
        height, width = self.shape
        grid_y, grid_x = np.mgrid[:height, :width]
        low_axis, high_axis = self._axis_displacements()
        coordinates = (-1.0, 0.0, 1.0)
        low_weight = (
            self._side_weights[0]
            if self._side_weights.ndim == 1 else self._side_weights[..., 0])
        high_weight = (
            self._side_weights[1]
            if self._side_weights.ndim == 1 else self._side_weights[..., 1])
        low_sigma_weights = (
            low_weight,
            1.0 - 2.0 * low_weight,
            low_weight,
        )
        high_sigma_weights = (
            high_weight,
            1.0 - 2.0 * high_weight,
            high_weight,
        )
        for low_index, low_coordinate in enumerate(coordinates):
            for high_index, high_coordinate in enumerate(coordinates):
                displacement = (
                    low_coordinate * low_axis
                    + high_coordinate * high_axis)
                atom_weight = np.asarray(
                    low_sigma_weights[low_index]
                    * high_sigma_weights[high_index])
                source_x = grid_x - displacement[..., 0]
                source_y = grid_y - displacement[..., 1]
                x0 = np.floor(source_x).astype(np.int64)
                y0 = np.floor(source_y).astype(np.int64)
                fx = source_x - x0
                fy = source_y - y0
                for dx, dy, interpolation in (
                    (0, 0, (1.0 - fx) * (1.0 - fy)),
                    (1, 0, fx * (1.0 - fy)),
                    (0, 1, (1.0 - fx) * fy),
                    (1, 1, fx * fy),
                ):
                    ix = _reflect_indices(x0 + dx, width)
                    iy = _reflect_indices(y0 + dy, height)
                    yield (
                        np.ascontiguousarray((iy * width + ix).ravel()),
                        np.ascontiguousarray(
                            (atom_weight * interpolation).ravel()),
                    )

    def _forward_numpy(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        channels = 1 if value.ndim == 2 else value.shape[2]
        flat = value.reshape((-1, channels))
        output = np.zeros_like(flat)
        for source, coefficient in self._contributions():
            output += coefficient[:, None] * flat[source]
        return output.reshape(value.shape)

    def _adjoint_numpy(self, image: np.ndarray) -> np.ndarray:
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

    def forward(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        if self._native_plan is not None:
            return self._native_plan.forward(value)
        return self._forward_numpy(value)

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        if self._native_plan is not None:
            return self._native_plan.adjoint(value)
        return self._adjoint_numpy(value)

    def adjoint_normalization(self, channels: int | None = None) -> np.ndarray:
        shape = self.shape if channels is None else (*self.shape, int(channels))
        return self.adjoint(np.ones(shape, dtype=np.float64))


class CovarianceExposureOperatorBatch:
    """Batch exchangeable covariance operators, using parallel ABI v6."""

    def __init__(
        self,
        operators: tuple[CovarianceReflectedExposureOperator, ...],
    ) -> None:
        if not operators:
            raise ValueError("covariance operator batch cannot be empty")
        if any(operator.shape != operators[0].shape for operator in operators):
            raise ValueError("covariance operator batch shapes must match")
        self.operators = operators
        self.shape = operators[0].shape
        self.backend = "numpy_generated_covariance_operator_sequence"
        self._native_plan = None
        try:
            from .native_backend import NativeCovarianceExposureBatchPlan
            side_weights = [operator._side_weights for operator in operators]
            if all(weight.shape == side_weights[0].shape for weight in side_weights):
                self._native_plan = NativeCovarianceExposureBatchPlan(
                    np.stack([operator._axes for operator in operators]),
                    self.shape,
                    np.stack(side_weights),
                )
                self.backend = self._native_plan.backend
        except (OSError, RuntimeError, ValueError):
            self._native_plan = None

    def _validate(self, images: np.ndarray) -> np.ndarray:
        value = np.asarray(images, dtype=np.float64)
        if (
            value.ndim not in (3, 4)
            or value.shape[0] != len(self.operators)
            or value.shape[1:3] != self.shape
        ):
            raise ValueError("covariance batch images must be NxHxW[xC]")
        return value

    def forward(self, images: np.ndarray) -> np.ndarray:
        value = self._validate(images)
        if self._native_plan is not None:
            return self._native_plan.forward(value)
        return np.stack([
            operator.forward(image)
            for operator, image in zip(self.operators, value)
        ])

    def adjoint(self, images: np.ndarray) -> np.ndarray:
        value = self._validate(images)
        if self._native_plan is not None:
            return self._native_plan.adjoint(value)
        return np.stack([
            operator.adjoint(image)
            for operator, image in zip(self.operators, value)
        ])


class SpatialExposureOperatorBatch:
    """Exchangeable spatial operators with an exact loop fallback."""

    def __init__(
        self,
        operators: list[SpatialReflectedExposureOperator]
        | tuple[SpatialReflectedExposureOperator, ...],
    ) -> None:
        self.operators = tuple(operators)
        if not self.operators:
            raise ValueError("spatial operator batch needs at least one plan")
        self.shape = self.operators[0].shape
        if any(operator.shape != self.shape for operator in self.operators):
            raise ValueError("spatial operator batch plans must share one raster")
        self._native_plan = None
        self.backend = "python_exchangeable_spatial_operator_loop"
        try:
            from .native_backend import (
                NativeSpatialExposureBatchPlan,
                native_available,
            )
            if native_available():
                maximum_contributions = max(
                    operator._source_indices.shape[0]
                    for operator in self.operators)
                indices = np.zeros(
                    (len(self.operators), maximum_contributions,
                     self.shape[0] * self.shape[1]),
                    dtype=np.int64,
                )
                coefficients = np.zeros_like(indices, dtype=np.float64)
                contribution_counts = np.empty(
                    len(self.operators), dtype=np.int32)
                for plan, operator in enumerate(self.operators):
                    count = operator._source_indices.shape[0]
                    contribution_counts[plan] = count
                    indices[plan, :count] = operator._source_indices
                    if operator._scalar_coefficients is not None:
                        coefficients[plan, :count] = (
                            operator._scalar_coefficients[:, None])
                    else:
                        coefficients[plan, :count] = operator._coefficients
                self._native_plan = NativeSpatialExposureBatchPlan(
                    indices, coefficients, self.shape, contribution_counts)
                self.backend = self._native_plan.backend
        except (OSError, RuntimeError, ValueError):
            self._native_plan = None

    def _validate(self, images: np.ndarray) -> np.ndarray:
        value = np.asarray(images, dtype=np.float64)
        if value.ndim not in (3, 4):
            raise ValueError("spatial operator batch expects SxHxW or SxHxWxC")
        if value.shape[0] != len(self.operators) or value.shape[1:3] != self.shape:
            raise ValueError("spatial operator batch image shape mismatch")
        return value

    def forward(self, images: np.ndarray) -> np.ndarray:
        value = self._validate(images)
        if self._native_plan is not None:
            return self._native_plan.forward(value)
        return np.stack([
            operator.forward(image)
            for operator, image in zip(self.operators, value)
        ], axis=0)

    def adjoint(self, images: np.ndarray) -> np.ndarray:
        value = self._validate(images)
        if self._native_plan is not None:
            return self._native_plan.adjoint(value)
        return np.stack([
            operator.adjoint(image)
            for operator, image in zip(self.operators, value)
        ], axis=0)


@dataclass(frozen=True)
class SpatialInverseResult:
    image: np.ndarray
    barycentric_seed: np.ndarray
    uncertainty: np.ndarray
    diagnostics: dict[str, object]


def _sample_field_at_sensor_coordinates(
    value: np.ndarray,
    sensor_x: np.ndarray,
    sensor_y: np.ndarray,
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    array = np.asarray(value, dtype=np.float64)
    if array.shape[:2] != sensor_x.shape or sensor_x.shape != sensor_y.shape:
        raise ValueError("sensor coordinates must match the sampled field")
    if array.ndim == 2:
        return map_coordinates(
            array,
            (sensor_y, sensor_x),
            order=1,
            mode="reflect",
            prefilter=False,
        )
    return np.stack([
        map_coordinates(
            array[..., channel],
            (sensor_y, sensor_x),
            order=1,
            mode="reflect",
            prefilter=False,
        )
        for channel in range(array.shape[2])
    ], axis=2)


def barycentric_inverse_coordinates(
    field: SpatialExposureField,
    *,
    iterations: int = 20,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Solve ``p=q+m(p)`` once and return sensor coordinates over latent q."""
    height, width = field.shape
    latent_y, latent_x = np.mgrid[:height, :width]
    sensor_x = latent_x.astype(np.float64)
    sensor_y = latent_y.astype(np.float64)
    flow = field.barycentric_flow_xy
    convergence_trace: list[float] = []
    for _ in range(max(int(iterations), 1)):
        sampled_flow = _sample_field_at_sensor_coordinates(
            flow, sensor_x, sensor_y)
        next_x = latent_x + sampled_flow[..., 0]
        next_y = latent_y + sampled_flow[..., 1]
        update = float(max(
            np.max(np.abs(next_x - sensor_x)),
            np.max(np.abs(next_y - sensor_y)),
        ))
        convergence_trace.append(update)
        sensor_x = next_x
        sensor_y = next_y
        if update <= max(float(tolerance), 0.0):
            break
    sampled_flow = _sample_field_at_sensor_coordinates(flow, sensor_x, sensor_y)
    map_residual_x = sensor_x - sampled_flow[..., 0] - latent_x
    map_residual_y = sensor_y - sampled_flow[..., 1] - latent_y
    return sensor_x, sensor_y, {
        "iterations_used": len(convergence_trace),
        "convergence_trace": convergence_trace,
        "terminal_coordinate_residual_max": float(max(
            np.max(np.abs(map_residual_x)),
            np.max(np.abs(map_residual_y)),
        )),
    }


def pullback_barycentric_values(
    value: np.ndarray,
    field: SpatialExposureField,
    *,
    iterations: int = 20,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict[str, object]]:
    """Pull an arbitrary sensor-coordinate scalar or image into latent q."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape[:2] != field.shape:
        raise ValueError("barycentric value pullback must match the field")
    sensor_x, sensor_y, record = barycentric_inverse_coordinates(
        field, iterations=iterations, tolerance=tolerance)
    return _sample_field_at_sensor_coordinates(
        array, sensor_x, sensor_y), record


def pullback_compact_global_values(
    value: np.ndarray,
    field: CompactGlobalExposureField,
) -> tuple[np.ndarray, dict[str, object]]:
    """Exactly pull a constant-translation sensor chart into latent space."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape[:2] != field.shape:
        raise ValueError("compact global pullback must match the field")
    height, width = field.shape
    latent_y, latent_x = np.mgrid[:height, :width]
    translation = field.barycentric_translation_xy
    pulled = _sample_field_at_sensor_coordinates(
        array,
        latent_x + translation[0],
        latent_y + translation[1],
    )
    return pulled, {
        "method": "analytic_constant_translation_pullback",
        "iterations_used": 0,
        "translation_xy": translation.tolist(),
        "terminal_coordinate_residual_max": 0.0,
    }


def pullback_barycentric_coordinates(
    observation: np.ndarray,
    field: SpatialExposureField,
    *,
    iterations: int = 20,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, SpatialExposureField, dict[str, object]]:
    """Invert ``q=p-m(p)`` and express centered exposure in latent coordinates."""
    measured = np.asarray(observation, dtype=np.float64)
    if measured.shape[:2] != field.shape:
        raise ValueError("barycentric pullback must match the exposure field")
    height, width = field.shape
    latent_y, latent_x = np.mgrid[:height, :width]
    flow = field.barycentric_flow_xy
    sensor_x, sensor_y, inverse_record = barycentric_inverse_coordinates(
        field, iterations=iterations, tolerance=tolerance)
    inverse_displacement = np.stack((
        latent_x - sensor_x,
        latent_y - sensor_y,
    ), axis=2)
    inverse_field = SpatialExposureField(
        name=f"inverse_barycentric_{field.name}",
        displacements_xy=inverse_displacement[None, ...],
        weights=np.ones((1, height, width), dtype=np.float64),
    )
    pullback_operator = SpatialReflectedExposureOperator(inverse_field)
    pulled_observation = pullback_operator.forward(measured)

    centered_sensor = field.centered_displacements_xy
    if field.compact_global:
        # A global residual probability measure is coordinate-invariant.
        # Resampling it would only introduce roundoff variation and destroy
        # the exact compact scalar-coefficient representation downstream.
        pulled_displacements = np.broadcast_to(
            centered_sensor[:, :1, :1], centered_sensor.shape).copy()
        pulled_weights = np.broadcast_to(
            field.weights[:, :1, :1], field.weights.shape).copy()
    else:
        pulled_displacements = np.empty_like(centered_sensor)
        pulled_weights = np.empty_like(field.weights)
        for atom in range(field.atom_count):
            pulled_displacements[atom] = _sample_field_at_sensor_coordinates(
                centered_sensor[atom], sensor_x, sensor_y)
            pulled_weights[atom] = _sample_field_at_sensor_coordinates(
                field.weights[atom], sensor_x, sensor_y)
    pulled_weights = np.maximum(pulled_weights, 0.0)
    pulled_weights /= np.maximum(
        np.sum(pulled_weights, axis=0, keepdims=True), 1e-15)
    # Interpolation can introduce a tiny nonzero residual centroid. Remove it
    # so deterministic transport remains entirely in the barycentric stage.
    residual_centroid = np.sum(
        pulled_weights[..., None] * pulled_displacements, axis=0)
    pulled_displacements -= residual_centroid[None, ...]
    centered_field = SpatialExposureField(
        name=f"centered_pullback_{field.name}",
        displacements_xy=pulled_displacements,
        weights=pulled_weights,
        compact_global=field.compact_global,
    )
    barycentric_forward = SpatialReflectedExposureOperator(
        field.barycentric_field()).forward(pulled_observation)
    return pulled_observation, centered_field, {
        "method": "fixed_point_inverse_barycentric_coordinate_transport",
        **inverse_record,
        "barycentric_roundtrip_rms": float(np.sqrt(np.mean(
            (barycentric_forward - measured) ** 2))),
        "inverse_mapping_jacobian_min": inverse_field.diagnostics()[
            "mapping_jacobian_min"],
    }


def pullback_covariance_coordinates(
    observation: np.ndarray,
    field: CovarianceExposureField,
    *,
    iterations: int = 20,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, CovarianceExposureField, dict[str, object]]:
    """Invert deterministic flow and transport a compact covariance field."""
    measured = np.asarray(observation, dtype=np.float64)
    if measured.shape[:2] != field.shape:
        raise ValueError("covariance pullback must match the exposure field")
    barycentric = field.barycentric_field()
    height, width = field.shape
    latent_y, latent_x = np.mgrid[:height, :width]
    sensor_x, sensor_y, inverse_record = barycentric_inverse_coordinates(
        barycentric, iterations=iterations, tolerance=tolerance)
    inverse_displacement = np.stack((
        latent_x - sensor_x,
        latent_y - sensor_y,
    ), axis=2)
    inverse_field = SpatialExposureField(
        name=f"inverse_barycentric_{field.name}",
        displacements_xy=inverse_displacement[None, ...],
        weights=np.ones((1, height, width), dtype=np.float64),
    )
    pulled_observation = SpatialReflectedExposureOperator(
        inverse_field).forward(measured)
    pulled_covariance = _sample_field_at_sensor_coordinates(
        field.covariance_components, sensor_x, sensor_y)
    pulled_side_weights = (
        field.axis_side_weights
        if field.axis_side_weights.ndim == 1
        else _sample_field_at_sensor_coordinates(
            field.axis_side_weights, sensor_x, sensor_y))
    centered_field = CovarianceExposureField(
        name=f"centered_pullback_{field.name}",
        barycentric_flow_xy=np.zeros((height, width, 2), dtype=np.float64),
        covariance_components=pulled_covariance,
        axis_side_weights=pulled_side_weights,
    )
    roundtrip = SpatialReflectedExposureOperator(
        barycentric).forward(pulled_observation)
    return pulled_observation, centered_field, {
        "method": "fixed_point_inverse_barycentric_covariance_transport",
        **inverse_record,
        "barycentric_roundtrip_rms": float(np.sqrt(np.mean(
            (roundtrip - measured) ** 2))),
        "generated_covariance_storage_bytes": int(
            centered_field.covariance_components.nbytes),
    }


def barycentric_pullback_seed(
    observation: np.ndarray,
    field: SpatialExposureField,
) -> tuple[np.ndarray, dict[str, object]]:
    """Undo deterministic mean flow by inverting its coordinate map."""
    seed, _, record = pullback_barycentric_coordinates(observation, field)
    return np.clip(seed, 0.0, 1.0), record


def refine_spatial_exposure(
    observation: np.ndarray,
    field: SpatialExposureField,
    *,
    passes: int = 32,
    ratio_limit: float = 2.0,
    discrepancy_ratio: float = 1.1,
    initial: np.ndarray | None = None,
) -> SpatialInverseResult:
    """Pull back barycentric flow, then refine through the full exact operator."""
    measured = np.asarray(observation, dtype=np.float64)
    if measured.shape[:2] != field.shape or measured.ndim not in (2, 3):
        raise ValueError("spatial observation must match its exposure field")
    field_record = field.diagnostics()
    determinant = field.sensor_to_latent_jacobian_determinant
    fold_mask = determinant <= 0.0
    if np.any(fold_mask):
        # A folded sensor-to-latent map has no single-valued deterministic
        # inverse. Keep the measurement intact and expose the failed geometry
        # as uncertainty instead of feeding an arbitrary pullback to descent.
        barycentric_operator = SpatialReflectedExposureOperator(
            field.barycentric_field())
        channels = None if measured.ndim == 2 else measured.shape[2]
        normalization = np.maximum(
            barycentric_operator.adjoint_normalization(channels), 1e-8)
        adjoint_seed = np.clip(
            barycentric_operator.adjoint(measured) / normalization,
            0.0,
            1.0,
        )
        coordinate_gauge = np.abs(adjoint_seed - measured)
        geometric_failure = fold_mask.astype(np.float64)
        if measured.ndim == 3:
            geometric_failure = np.repeat(
                geometric_failure[..., None], measured.shape[2], axis=2)
        uncertainty = np.sqrt(
            coordinate_gauge * coordinate_gauge
            + geometric_failure * geometric_failure)
        return SpatialInverseResult(
            image=measured.copy(),
            barycentric_seed=measured.copy(),
            uncertainty=uncertainty,
            diagnostics={
                "method": (
                    "barycentric_first_spatial_positive_exposure_transport"),
                "estimation_decision": (
                    "abstain_noninvertible_barycentric_map"),
                "operator_backend": barycentric_operator.backend,
                "boundary": "exact_bilinear_half_sample_reflection",
                "field": field_record,
                "centered_pullback_field": None,
                "barycentric_pullback": {
                    "method": "not_attempted_noninvertible_map",
                    "iterations_used": 0,
                    "convergence_trace": [],
                    "terminal_coordinate_residual_max": None,
                },
                "passes_used": 0,
                "stopped_by": "geometry_fold_abstention",
                "residual_trace": [],
                "uncertainty_rms": float(np.sqrt(np.mean(
                    uncertainty * uncertainty))),
                "uncertainty_q95": float(np.quantile(uncertainty, 0.95)),
                "uncertainty_role": (
                    "noninvertible_geometry_mask_plus_coordinate_adjoint_"
                    "disagreement_not_calibrated_interval"),
                "coordinate_gauge_rms": float(np.sqrt(np.mean(
                    coordinate_gauge * coordinate_gauge))),
                "observation_unchanged": True,
            },
        )
    barycentric_seed, centered_field, seed_record = (
        pullback_barycentric_coordinates(measured, field))
    latent = np.clip(
        barycentric_seed if initial is None else np.asarray(initial, dtype=np.float64),
        1e-8,
        1.0,
    )
    if latent.shape != measured.shape:
        raise ValueError("spatial initial state must match the observation")
    operator = SpatialReflectedExposureOperator(centered_field)
    pulled_observation = barycentric_seed
    channels = None if measured.ndim == 2 else measured.shape[2]
    normalization = np.maximum(operator.adjoint_normalization(channels), 1e-8)
    limit = max(float(ratio_limit), 1.0)
    target = max(float(discrepancy_ratio), 1.0)
    prediction = operator.forward(latent)
    initial_discrepancy = residual_discrepancy(pulled_observation, prediction)
    terminal_discrepancy = initial_discrepancy
    residual_trace: list[float] = []
    stopped_by = "maximum_passes"
    if initial_discrepancy["total_to_read_ratio"] <= target:
        stopped_by = "noise_discrepancy"
    for _ in range(
        0 if stopped_by == "noise_discrepancy" else max(int(passes), 0)
    ):
        ratio = np.clip(
            pulled_observation / np.maximum(prediction, 1e-8),
            1.0 / limit,
            limit,
        )
        correction = np.maximum(operator.adjoint(ratio) / normalization, 1e-8)
        latent = np.clip(latent * correction, 0.0, 1.0)
        prediction = operator.forward(latent)
        residual_trace.append(float(np.sqrt(np.mean(
            (prediction - pulled_observation) ** 2))))
        terminal_discrepancy = residual_discrepancy(
            pulled_observation, prediction)
        if terminal_discrepancy["total_to_read_ratio"] <= target:
            stopped_by = "noise_discrepancy"
            break
    original_operator = SpatialReflectedExposureOperator(field)
    original_prediction = original_operator.forward(latent)
    original_normalization = np.maximum(
        original_operator.adjoint_normalization(channels), 1e-8)
    transported_residual = (
        original_operator.adjoint(np.abs(original_prediction - measured))
        / original_normalization
    )
    barycentric_operator = SpatialReflectedExposureOperator(
        field.barycentric_field())
    barycentric_normalization = np.maximum(
        barycentric_operator.adjoint_normalization(channels), 1e-8)
    adjoint_pullback = np.clip(
        barycentric_operator.adjoint(measured) / barycentric_normalization,
        0.0,
        1.0,
    )
    coordinate_gauge = np.abs(adjoint_pullback - barycentric_seed)
    uncertainty = np.sqrt(
        transported_residual * transported_residual
        + coordinate_gauge * coordinate_gauge
    )
    return SpatialInverseResult(
        image=latent,
        barycentric_seed=barycentric_seed,
        uncertainty=uncertainty,
        diagnostics={
            "method": "barycentric_first_spatial_positive_exposure_transport",
            "operator_backend": operator.backend,
            "boundary": "exact_bilinear_half_sample_reflection",
            "field": field_record,
            "centered_pullback_field": centered_field.diagnostics(),
            "barycentric_pullback": seed_record,
            "passes_used": len(residual_trace),
            "stopped_by": stopped_by,
            "discrepancy_ratio_target": target,
            "initial_discrepancy": initial_discrepancy,
            "terminal_discrepancy": terminal_discrepancy,
            "residual_trace": residual_trace,
            "centered_forward_rms": float(np.sqrt(np.mean(
                (prediction - pulled_observation) ** 2))),
            "original_forward_rms": float(np.sqrt(np.mean(
                (original_prediction - measured) ** 2))),
            "uncertainty_rms": float(np.sqrt(np.mean(
                uncertainty * uncertainty))),
            "uncertainty_q95": float(np.quantile(uncertainty, 0.95)),
            "uncertainty_role": (
                "backtransported_residual_plus_coordinate_pullback_gauge_"
                "not_calibrated_interval"
            ),
            "coordinate_gauge_rms": float(np.sqrt(np.mean(
                coordinate_gauge * coordinate_gauge))),
            "backtransported_residual_rms": float(np.sqrt(np.mean(
                transported_residual * transported_residual))),
            "observation_unchanged": True,
        },
    )


def shear_path_exposure(
    shape: tuple[int, int],
    *,
    shear: float,
    residual_length: float = 0.0,
    atoms: int = 9,
) -> SpatialExposureField:
    """Synthetic spatial warp plus centered horizontal exposure control."""
    height, width = map(int, shape)
    yy, _ = np.mgrid[:height, :width]
    flow = np.zeros((height, width, 2), dtype=np.float64)
    flow[..., 0] = float(shear) * (yy - 0.5 * (height - 1))
    count = max(int(atoms), 1)
    residual = np.zeros((count, 2), dtype=np.float64)
    residual[:, 0] = np.linspace(
        -0.5 * float(residual_length),
        0.5 * float(residual_length),
        count,
    )
    return SpatialExposureField.from_barycentric_paths(
        name=f"shear_{float(shear):g}_residual_{float(residual_length):g}",
        barycentric_flow_xy=flow,
        residual_displacements_xy=residual,
        weights=np.ones(count, dtype=np.float64),
    )


def rotational_exposure(
    shape: tuple[int, int],
    *,
    mean_angle_degrees: float,
    exposure_degrees: float = 0.0,
    atoms: int = 9,
) -> SpatialExposureField:
    """Positive camera-rotation exposure with spatially varying displacement."""
    height, width = map(int, shape)
    count = max(int(atoms), 1)
    angles = np.linspace(
        float(mean_angle_degrees) - 0.5 * float(exposure_degrees),
        float(mean_angle_degrees) + 0.5 * float(exposure_degrees),
        count,
        dtype=np.float64,
    )
    yy, xx = np.mgrid[:height, :width]
    center_x = 0.5 * (width - 1)
    center_y = 0.5 * (height - 1)
    centered_x = xx - center_x
    centered_y = yy - center_y
    displacement = np.empty((count, height, width, 2), dtype=np.float64)
    for atom, angle_degrees in enumerate(angles):
        angle = np.deg2rad(angle_degrees)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        # Sensor p observes latent R(-angle)(p-c)+c; d=p-q.
        source_x = cosine * centered_x + sine * centered_y + center_x
        source_y = -sine * centered_x + cosine * centered_y + center_y
        displacement[atom, ..., 0] = xx - source_x
        displacement[atom, ..., 1] = yy - source_y
    return SpatialExposureField(
        name=(
            f"rotation_mean_{float(mean_angle_degrees):g}_"
            f"exposure_{float(exposure_degrees):g}"
        ),
        displacements_xy=displacement,
        weights=np.ones((count, height, width), dtype=np.float64),
    )
