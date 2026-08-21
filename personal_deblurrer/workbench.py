"""Dear PyGui workbench and headless session for the personal deblurrer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
import traceback

import numpy as np
from PIL import Image, ImageOps

from .decomposition import (
    TwoStageDeblurResult,
    apply_reflect,
    image_fingerprint,
    two_stage_deblur_blind,
    two_stage_deblur_known,
)
from .kernels import (
    TransportKernel,
    curved_path_kernel,
    disk_kernel,
    gaussian_kernel,
    identity_kernel,
    line_kernel,
    path_kernel,
    translated_kernel,
)
from .solver import DeblurResult, fuse_transport_observations
from .multicapture_transport import (
    MultiCaptureTransportResult,
    deblur_multicapture_consensus as solve_multicapture_consensus,
)
from .multicapture_posterior import (
    MultiCapturePosteriorSolution,
    solve_multicapture_transport_posterior,
)
from .spatial_transport import (
    SpatialExposureField,
    SpatialInverseResult,
    SpatialReflectedExposureOperator,
    refine_spatial_exposure,
    rotational_exposure,
)
from .spatial_estimation import (
    SpatialConsensusResult,
    deblur_rotation_consensus as solve_rotation_consensus,
)
from .flow_fiber_estimation import (
    FlowFiberConsensusResult,
    deblur_flow_fiber_consensus as solve_dense_pair_consensus,
)
from .synthetic import degrade, random_camera_path
from .uncertainty import (
    PairPosterior,
    UncertainDeblurResult,
    deblur_pair_posterior,
    estimate_noise_discrepancy,
    estimate_pair_posterior,
    predict_observation,
)


V3_SKIMAGE_PORTFOLIO = (
    "brick", "grass", "gravel", "checkerboard", "text", "page", "coins",
    "moon", "camera", "cell", "clock", "astronaut", "chelsea", "coffee",
    "immunohistochemistry", "retina", "hubble_deep_field",
)


@dataclass(frozen=True)
class BlurSpec:
    kind: str = "Gaussian"
    sigma: float = 2.0
    radius: float = 3.0
    length: float = 11.0
    angle_degrees: float = 0.0
    bend: float = 4.0
    extent: float = 7.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    rotation_mean_degrees: float = 4.0
    rotation_exposure_degrees: float = 8.0
    rolling_mean_x: float = 4.0
    rolling_row_acceleration: float = 1.0
    rolling_exposure_extent: float = 2.0
    exposure_gain: float = 1.0
    seed: int = 0

    def kernel(self) -> TransportKernel:
        name = self.kind.strip().lower()
        if name in ("none", "identity"):
            base = identity_kernel()
            return translated_kernel(base, (self.shift_x, self.shift_y))
        if name == "gaussian":
            base = gaussian_kernel(self.sigma)
            return translated_kernel(base, (self.shift_x, self.shift_y))
        if name in ("disk", "defocus"):
            base = disk_kernel(self.radius)
            return translated_kernel(base, (self.shift_x, self.shift_y))
        if name in ("line", "linear motion"):
            base = line_kernel(self.length, self.angle_degrees)
            return translated_kernel(base, (self.shift_x, self.shift_y))
        if name in ("curve", "curved motion"):
            base = curved_path_kernel(
                self.length, self.angle_degrees, self.bend)
            return translated_kernel(base, (self.shift_x, self.shift_y))
        if name in ("random path", "camera path"):
            points = random_camera_path(
                extent=self.extent, seed=self.seed)
            base = path_kernel(
                points,
                name=f"random_path_extent_{self.extent:g}_seed_{self.seed}",
            )
            return translated_kernel(base, (self.shift_x, self.shift_y))
        raise ValueError(f"unknown blur family: {self.kind}")

    def spatial_field(
        self,
        shape: tuple[int, int],
    ) -> SpatialExposureField | None:
        name = self.kind.strip().lower()
        if name in ("rolling shutter", "rolling shutter exposure"):
            height, width = map(int, shape)
            yy = np.arange(height, dtype=np.float64)[:, None]
            normalized_y = (
                (yy - 0.5 * (height - 1)) / max(height - 1, 1))
            row_scale = 1.0 + self.rolling_row_acceleration * normalized_y
            flow = np.zeros((height, width, 2), dtype=np.float64)
            flow[..., 0] = self.rolling_mean_x * row_scale
            flow[..., 1] = 0.0
            times = np.linspace(-0.5, 0.5, 9, dtype=np.float64)
            residual = np.zeros((len(times), height, width, 2), dtype=np.float64)
            residual[..., 0] = (
                times[:, None, None]
                * self.rolling_exposure_extent
                * (1.0 + 0.5 * self.rolling_row_acceleration * normalized_y)
            )
            field = SpatialExposureField.from_barycentric_paths(
                name=(
                    f"rolling_shutter_mean_{self.rolling_mean_x:g}_"
                    f"row_{self.rolling_row_acceleration:g}_"
                    f"exposure_{self.rolling_exposure_extent:g}"),
                barycentric_flow_xy=flow,
                residual_displacements_xy=residual,
                weights=np.ones((len(times), height, width), dtype=np.float64),
            )
        elif name in ("rotation", "rotational exposure"):
            field = rotational_exposure(
                shape,
                mean_angle_degrees=self.rotation_mean_degrees,
                exposure_degrees=self.rotation_exposure_degrees,
                atoms=9,
            )
        else:
            return None
        if abs(self.shift_x) <= 1e-15 and abs(self.shift_y) <= 1e-15:
            return field
        displacement = field.displacements_xy.copy()
        displacement[..., 0] += self.shift_x
        displacement[..., 1] += self.shift_y
        return SpatialExposureField(
            name=(f"{field.name}_shift_{self.shift_x:g}_{self.shift_y:g}"),
            displacements_xy=displacement,
            weights=field.weights,
        )


@dataclass
class SourceRecord:
    label: str
    path: Path | None
    original: np.ndarray
    observation: np.ndarray
    kernel: TransportKernel | None = None
    spatial_field: SpatialExposureField | None = None
    mode: str = "loaded_source"
    synthetic_truth_available: bool = False
    read_noise_sigma: float = 0.0
    shot_peak: float = 0.0
    result: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


def _load_rgb(path: Path, maximum_dimension: int = 1024) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if max(image.size) > max(int(maximum_dimension), 1):
            image.thumbnail(
                (maximum_dimension, maximum_dimension), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64) / 255.0


def load_v3_skimage_source(name: str) -> np.ndarray:
    """Load one scikit-image raster from the Segmenter V3 source portfolio.

    These are input fixtures only.  No segmenter state, labels, or algorithm is
    imported into the deblurrer.
    """
    key = str(name).strip()
    if key not in V3_SKIMAGE_PORTFOLIO:
        raise ValueError(f"unknown V3 scikit-image portfolio source: {name}")
    try:
        from skimage import data
    except ImportError as error:
        raise RuntimeError(
            "The V3 portfolio requires scikit-image in the GUI environment") from error
    value = np.asarray(getattr(data, key)(), dtype=np.float64)
    if key == "retina":
        value = value[::2, ::2]
    if value.ndim == 3 and value.shape[2] > 3:
        value = value[..., :3]
    if float(np.max(value)) > 1.5:
        value /= 255.0
    return np.clip(value, 0.0, 1.0)


def workbench_catalog() -> list[TransportKernel]:
    result = [identity_kernel()]
    result.extend(gaussian_kernel(value) for value in (1.5, 2.5, 3.5))
    result.extend(disk_kernel(value) for value in (2.5, 3.5))
    result.extend(
        line_kernel(length, angle)
        for length in (7.0, 11.0, 15.0)
        for angle in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
    )
    return result


def _center_crop(image: np.ndarray, maximum: int = 256) -> np.ndarray:
    height, width = image.shape[:2]
    crop_h = min(height, max(int(maximum), 16))
    crop_w = min(width, max(int(maximum), 16))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


class DeblurSession:
    """Headless state used by both tests and the Dear PyGui front end."""

    def __init__(self) -> None:
        self.sources: list[SourceRecord] = []

    def load(self, paths: list[Path]) -> list[int]:
        indices = []
        for path in paths:
            rgb = _load_rgb(Path(path))
            indices.append(len(self.sources))
            self.sources.append(SourceRecord(
                label=Path(path).name,
                path=Path(path),
                original=rgb,
                observation=rgb.copy(),
            ))
        return indices

    def add_array(self, image: np.ndarray, label: str = "array") -> int:
        value = np.asarray(image, dtype=np.float64)
        if value.ndim not in (2, 3):
            raise ValueError("a source must be HxW or HxWxC")
        if value.ndim == 3 and value.shape[2] not in (1, 3, 4):
            raise ValueError("a color source must have one, three, or four channels")
        if value.ndim == 3 and value.shape[2] == 4:
            value = value[..., :3]
        index = len(self.sources)
        self.sources.append(SourceRecord(
            label=str(label), path=None, original=value.copy(),
            observation=value.copy()))
        return index

    def add_v3_portfolio_source(self, name: str) -> int:
        return self.add_array(
            load_v3_skimage_source(name),
            label=f"V3 portfolio - {name}",
        )

    def synthesize(
        self,
        index: int,
        spec: BlurSpec,
        *,
        read_noise_sigma: float = 0.0,
        shot_peak: float = 0.0,
        seed: int = 0,
    ) -> SourceRecord:
        record = self.sources[int(index)]
        spatial_field = spec.spatial_field(record.original.shape[:2])
        if spatial_field is None:
            kernel = spec.kernel()
            blurred = degrade(
                record.original,
                kernel,
                clip=False,
                boundary="reflect",
            )
            operator_name = kernel.name
        else:
            kernel = None
            blurred = SpatialReflectedExposureOperator(
                spatial_field).forward(record.original)
            operator_name = spatial_field.name
        blurred = max(float(spec.exposure_gain), 0.0) * blurred
        rng = np.random.default_rng(int(seed))
        if float(shot_peak) > 0.0:
            peak = float(shot_peak)
            blurred = rng.poisson(np.maximum(blurred, 0.0) * peak) / peak
        if float(read_noise_sigma) > 0.0:
            blurred = blurred + rng.normal(
                0.0, float(read_noise_sigma), blurred.shape)
        record.observation = np.clip(blurred, 0.0, 1.0)
        record.kernel = kernel
        record.spatial_field = spatial_field
        record.mode = "synthetic_blur"
        record.synthetic_truth_available = True
        record.read_noise_sigma = float(read_noise_sigma)
        record.shot_peak = float(shot_peak)
        record.result = None
        record.uncertainty = None
        record.diagnostics = {
            "operation": "synthetic_exposure",
            "operator": operator_name,
            "spatial": spatial_field is not None,
            "read_noise_sigma": float(read_noise_sigma),
            "shot_peak": float(shot_peak),
            "exposure_gain": float(spec.exposure_gain),
            "seed": int(seed),
            "boundary": "reflect",
        }
        return record

    def use_as_is(self, index: int) -> SourceRecord:
        """Declare the loaded pixels to be an unknown real observation.

        This makes no synthetic truth or known-kernel claim.  A fresh copy is
        used so later reconstruction state can never overwrite the loaded
        source buffer.
        """
        record = self.sources[int(index)]
        record.observation = record.original.copy()
        record.kernel = None
        record.spatial_field = None
        record.mode = "as_is_observation"
        record.synthetic_truth_available = False
        record.read_noise_sigma = 0.0
        record.shot_peak = 0.0
        record.result = None
        record.uncertainty = None
        record.diagnostics = {
            "operation": "use_loaded_pixels_as_unknown_observation",
            "observation_fingerprint": image_fingerprint(record.observation),
            "truth_available": False,
            "kernel_available": False,
        }
        return record

    def add_synthetic_capture(
        self,
        source_index: int,
        spec: BlurSpec,
        *,
        read_noise_sigma: float = 0.0,
        shot_peak: float = 0.0,
        seed: int = 0,
    ) -> int:
        source = self.sources[int(source_index)]
        index = self.add_array(
            source.original,
            label=f"{source.label} - capture {len(self.sources) + 1}")
        self.sources[index].path = source.path
        self.synthesize(
            index,
            spec,
            read_noise_sigma=read_noise_sigma,
            shot_peak=shot_peak,
            seed=seed,
        )
        return index

    def deblur_known(
        self,
        index: int,
        *,
        tv_weight: float = 0.0012,
        flux_penalty: float = 0.035,
        passes: int = 64,
    ) -> TwoStageDeblurResult | SpatialInverseResult:
        record = self.sources[int(index)]
        del tv_weight, flux_penalty  # retained in the API for older callers
        if record.kernel is None and record.spatial_field is None:
            raise ValueError(
                "this is an unknown as-is observation; use unified blind deblurring")
        if record.spatial_field is not None:
            result = refine_spatial_exposure(
                record.observation,
                record.spatial_field,
                passes=passes,
                ratio_limit=4.0,
            )
            prediction = SpatialReflectedExposureOperator(
                record.spatial_field).forward(result.image)
            discrepancy = estimate_noise_discrepancy(
                record.observation, prediction)
            before_mse = max(float(np.mean(
                (record.observation - record.original) ** 2)),
                np.finfo(float).tiny,
            )
            after_mse = max(float(np.mean(
                (result.image - record.original) ** 2)),
                np.finfo(float).tiny,
            )
            record.result = result.image
            record.uncertainty = result.uncertainty
            record.diagnostics = {
                **result.diagnostics,
                "noise_discrepancy": discrepancy.__dict__,
                "truth_role": "evaluation_only",
                "observation_psnr": float(-10.0 * np.log10(before_mse)),
                "result_psnr": float(-10.0 * np.log10(after_mse)),
                "psnr_gain": float(10.0 * np.log10(before_mse / after_mse)),
            }
            return result
        result = two_stage_deblur_known(
            record.observation,
            record.kernel,
            passes=passes,
            reference=(record.original if record.synthetic_truth_available else None),
        )
        prediction = apply_reflect(result.image, record.kernel)
        discrepancy = estimate_noise_discrepancy(record.observation, prediction)
        record.result = result.image
        record.uncertainty = result.uncertainty
        record.diagnostics = {
            **result.diagnostics,
            "noise_discrepancy": discrepancy.__dict__,
        }
        return result

    def deblur_active(
        self,
        index: int,
        *,
        passes: int = 64,
    ) -> TwoStageDeblurResult | SpatialInverseResult:
        """Run the same shift-first/center-mix inverse in known or blind mode."""
        record = self.sources[int(index)]
        if (
            (record.kernel is not None or record.spatial_field is not None)
            and record.synthetic_truth_available
        ):
            return self.deblur_known(index, passes=passes)
        result = two_stage_deblur_blind(record.observation, passes=passes)
        estimated_kernel = result.factorization.centered_mixing
        prediction = apply_reflect(result.image, estimated_kernel)
        discrepancy = estimate_noise_discrepancy(record.observation, prediction)
        record.result = result.image
        record.uncertainty = result.uncertainty
        record.diagnostics = {
            **result.diagnostics,
            "noise_discrepancy": discrepancy.__dict__,
            "truth_available": False,
        }
        return result

    def deblur_pair_uncertain(
        self,
        first_index: int,
        second_index: int,
        *,
        noise_sigma: float | None = None,
        credibility: float = 0.95,
        maximum_branches: int = 8,
        tv_weight: float = 0.0012,
        flux_penalty: float = 0.035,
        passes: int = 16,
    ) -> tuple[PairPosterior, UncertainDeblurResult]:
        first = self.sources[int(first_index)]
        second = self.sources[int(second_index)]
        if first.observation.shape != second.observation.shape:
            raise ValueError("registered pair captures must share one raster")
        sigma = (
            max(first.read_noise_sigma, second.read_noise_sigma)
            if noise_sigma is None else max(float(noise_sigma), 0.0)
        )
        posterior = estimate_pair_posterior(
            _center_crop(first.observation),
            _center_crop(second.observation),
            workbench_catalog(),
            noise_sigma=sigma,
        )
        if posterior.common_blur_unidentifiable:
            samples = np.stack((first.observation, second.observation), axis=0)
            mean = np.mean(samples, axis=0)
            standard_deviation = np.std(samples, axis=0)
            result = UncertainDeblurResult(
                image=mean,
                standard_deviation=standard_deviation,
                lower=np.minimum(first.observation, second.observation),
                upper=np.maximum(first.observation, second.observation),
                branch_images=(),
                branch_hypotheses=(),
                branch_weights=np.empty(0, dtype=np.float64),
                retained_probability=1.0,
                diagnostics={
                    "decision": "abstain_common_blur_gauge",
                    "posterior_entropy": posterior.entropy,
                    "effective_hypotheses": posterior.effective_hypotheses,
                    "selected_branches": 0,
                    "retained_probability": 1.0,
                    "common_blur_unidentifiable": True,
                    "mean_blur_variance": float(np.mean(
                        standard_deviation * standard_deviation)),
                },
            )
        else:
            result = deblur_pair_posterior(
                first.observation,
                second.observation,
                posterior,
                credibility=credibility,
                maximum_branches=maximum_branches,
                noise_sigma=sigma,
                tv_weight=tv_weight,
                flux_penalty=flux_penalty,
                passes=passes,
            )
        first.result = result.image
        first.uncertainty = result.standard_deviation
        first.diagnostics = {
            **result.diagnostics,
            "best_pair": [posterior.best.first.name, posterior.best.second.name],
            "best_probability": posterior.best.probability,
            "temperature": posterior.temperature,
        }
        return posterior, result

    def deblur_rotation_consensus(
        self,
        indices: list[int] | tuple[int, ...],
        *,
        reference_index: int | None = None,
        duty_cycle: float = 1.0,
        passes: int = 64,
    ) -> tuple[int, SpatialConsensusResult]:
        selected = tuple(int(index) for index in indices)
        if len(selected) < 2:
            raise ValueError("rotation consensus needs at least two observations")
        records = [self.sources[index] for index in selected]
        if any(
            record.observation.shape != records[0].observation.shape
            for record in records[1:]
        ):
            raise ValueError("rotation consensus observations must share one raster")
        if (
            all(record.synthetic_truth_available for record in records)
            and len({
                image_fingerprint(record.original) for record in records
            }) != 1
        ):
            raise ValueError(
                "synthetic consensus captures must come from one source truth")
        local_reference = (
            len(selected) // 2 if reference_index is None else int(reference_index))
        result = solve_rotation_consensus(
            [record.observation for record in records],
            reference_index=local_reference,
            duty_cycle=duty_cycle,
            passes=passes,
        )
        target_index = selected[result.estimate.reference_index]
        target = self.sources[target_index]
        target.result = result.image
        target.uncertainty = result.uncertainty
        diagnostics = dict(result.diagnostics)
        if all(record.synthetic_truth_available for record in records):
            truth = target.original
            before = min(float(np.mean(
                (record.observation - truth) ** 2)) for record in records)
            after = float(np.mean((result.image - truth) ** 2))
            diagnostics.update({
                "truth_role": "evaluation_only",
                "best_capture_psnr": float(-10.0 * np.log10(max(
                    before, np.finfo(float).tiny))),
                "result_psnr": float(-10.0 * np.log10(max(
                    after, np.finfo(float).tiny))),
                "psnr_gain_over_best_capture": float(10.0 * np.log10(
                    max(before, np.finfo(float).tiny)
                    / max(after, np.finfo(float).tiny))),
            })
        target.diagnostics = diagnostics
        return target_index, result

    def deblur_multicapture_posterior(
        self,
        indices: list[int] | tuple[int, ...],
        *,
        target_index: int | None = None,
        passes: int = 64,
    ) -> tuple[
        int,
        MultiCaptureTransportResult,
        MultiCapturePosteriorSolution,
    ]:
        """Run the unified center-first posterior on explicit same-scene views."""
        selected = tuple(int(index) for index in indices)
        if len(selected) < 3:
            raise ValueError("multi-capture posterior needs at least three observations")
        records = [self.sources[index] for index in selected]
        if any(
            record.observation.shape != records[0].observation.shape
            for record in records[1:]
        ):
            raise ValueError(
                "multi-capture posterior observations must share one raster")
        synthetic = [record.synthetic_truth_available for record in records]
        if any(synthetic) and not all(synthetic):
            raise ValueError(
                "do not mix synthetic-truth and as-is observations in one gauge")
        if all(synthetic) and len({
            image_fingerprint(record.original) for record in records
        }) != 1:
            raise ValueError(
                "synthetic multi-capture observations must share one source truth")
        minimum_extent = min(records[0].observation.shape[:2])
        patch_size = min(192, max(16, int(round(0.5 * minimum_extent))))
        stride = max(8, int(round((2.0 / 3.0) * patch_size)))
        inverse = solve_multicapture_consensus(
            [record.observation for record in records],
            passes=passes,
            descent_method="optimal_positive_line",
            mixing_patch_size=patch_size,
            mixing_stride=stride,
        )
        posterior = solve_multicapture_transport_posterior(inverse)
        target = selected[0] if target_index is None else int(target_index)
        if target not in selected:
            raise ValueError("posterior target must be one of its observations")
        record = self.sources[target]
        record.result = posterior.image
        record.uncertainty = posterior.uncertainty
        diagnostics = {
            **inverse.diagnostics,
            "posterior": posterior.diagnostics,
            "mixing_patch_size": patch_size,
            "mixing_stride": stride,
        }
        if all(synthetic):
            truth = records[0].original
            center_mse = float(np.mean(
                (posterior.center_image - truth) ** 2))
            result_mse = float(np.mean((posterior.image - truth) ** 2))
            diagnostics.update({
                "truth_role": "evaluation_only",
                "center_transport_psnr": float(-10.0 * np.log10(max(
                    center_mse, np.finfo(float).tiny))),
                "posterior_psnr": float(-10.0 * np.log10(max(
                    result_mse, np.finfo(float).tiny))),
            })
        record.diagnostics = diagnostics
        return target, inverse, posterior

    def deblur_dense_pair_consensus(
        self,
        first_index: int,
        second_index: int,
        *,
        target_index: int | None = None,
        duty_cycle: float = 0.5,
        passes: int = 64,
    ) -> tuple[int, FlowFiberConsensusResult]:
        first = self.sources[int(first_index)]
        second = self.sources[int(second_index)]
        if int(first_index) == int(second_index):
            raise ValueError("dense consensus needs two different observations")
        if first.observation.shape != second.observation.shape:
            raise ValueError("dense consensus observations must share one raster")
        if (
            first.synthetic_truth_available
            and second.synthetic_truth_available
            and image_fingerprint(first.original) != image_fingerprint(second.original)
        ):
            raise ValueError(
                "synthetic consensus captures must come from one source truth")
        result = solve_dense_pair_consensus(
            first.observation,
            second.observation,
            duty_cycle=duty_cycle,
            passes=passes,
        )
        target = (
            int(first_index) if target_index is None else int(target_index))
        if target not in (int(first_index), int(second_index)):
            raise ValueError("dense consensus target must be one of its observations")
        record = self.sources[target]
        record.result = result.image
        record.uncertainty = result.uncertainty
        diagnostics = dict(result.diagnostics)
        diagnostics["output_coordinate_gauge"] = "symmetric_pair_midpoint"
        # Truth-coordinate scoring is meaningful only when known synthetic
        # barycenters themselves close around that midpoint gauge.
        fields = (first.spatial_field, second.spatial_field)
        symmetric_truth_gauge = bool(
            first.synthetic_truth_available
            and second.synthetic_truth_available
            and all(field is not None for field in fields)
            and float(np.sqrt(np.mean(np.sum(
                (fields[0].barycentric_flow_xy
                 + fields[1].barycentric_flow_xy) ** 2,
                axis=2,
            )))) <= 1e-6
        )
        diagnostics["synthetic_truth_gauge_matches_output"] = symmetric_truth_gauge
        if symmetric_truth_gauge:
            truth = first.original
            best_mse = min(
                float(np.mean((first.observation - truth) ** 2)),
                float(np.mean((second.observation - truth) ** 2)),
            )
            after_mse = float(np.mean((result.image - truth) ** 2))
            diagnostics.update({
                "truth_role": "evaluation_only",
                "best_capture_psnr": float(-10.0 * np.log10(max(
                    best_mse, np.finfo(float).tiny))),
                "result_psnr": float(-10.0 * np.log10(max(
                    after_mse, np.finfo(float).tiny))),
                "psnr_gain_over_best_capture": float(10.0 * np.log10(
                    max(best_mse, np.finfo(float).tiny)
                    / max(after_mse, np.finfo(float).tiny))),
            })
        record.diagnostics = diagnostics
        return target, result


def _display_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    elif value.ndim == 3 and value.shape[2] == 1:
        value = np.repeat(value, 3, axis=2)
    return np.clip(value[..., :3], 0.0, 1.0)


def _uncertainty_rgb(uncertainty: np.ndarray) -> np.ndarray:
    value = np.asarray(uncertainty, dtype=np.float64)
    if value.ndim == 3:
        value = np.mean(value, axis=2)
    scale = max(float(np.quantile(value, 0.99)), 1e-8)
    normalized = np.clip(value / scale, 0.0, 1.0)
    # Perceptually ordered dark-blue -> cyan -> yellow map.
    red = np.clip(2.0 * normalized - 0.15, 0.0, 1.0)
    green = np.clip(1.8 * normalized, 0.0, 1.0)
    blue = np.clip(1.25 - 1.4 * normalized, 0.0, 1.0)
    return np.stack((red, green, blue), axis=2)


def _rgba_texture(image: np.ndarray) -> tuple[int, int, list[float]]:
    rgb = _display_rgb(image).astype(np.float32)
    alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
    rgba = np.concatenate((rgb, alpha), axis=2)
    return rgb.shape[1], rgb.shape[0], rgba.ravel().tolist()


class DeblurWorkbenchApp:
    def __init__(self, dpg) -> None:
        self.dpg = dpg
        self.session = DeblurSession()
        self.active = 0
        self.texture_tags: list[str] = []
        self._busy = threading.Lock()

    def _labels(self) -> list[str]:
        return [f"{index + 1}: {item.label}" for index, item in enumerate(self.session.sources)]

    def _refresh_source_controls(self) -> None:
        labels = self._labels()
        for tag in ("active_source", "pair_first", "pair_second"):
            self.dpg.configure_item(tag, items=labels)
        if labels:
            self.active = min(self.active, len(labels) - 1)
            self.dpg.set_value("active_source", labels[self.active])
            self.dpg.set_value("pair_first", labels[0])
            self.dpg.set_value("pair_second", labels[min(1, len(labels) - 1)])

    def _index_from_label(self, label: str) -> int:
        if not label:
            raise ValueError("choose a source")
        return int(label.split(":", 1)[0]) - 1

    def choose_sources(self, _sender=None, app_data=None) -> None:
        try:
            paths: list[Path] = []
            if app_data:
                selections = app_data.get("selections") or {}
                paths.extend(Path(value) for value in selections.values())
                if not paths and app_data.get("file_path_name"):
                    paths.append(Path(app_data["file_path_name"]))
            if not paths:
                return
            indices = self.session.load(paths)
            self.active = indices[-1]
            self._refresh_source_controls()
            self.render()
            self.status(f"Loaded {len(paths)} source image(s)")
        except Exception:
            self.status(traceback.format_exc())

    def choose_active(self, _sender=None, app_data=None) -> None:
        try:
            self.active = self._index_from_label(str(app_data))
            self.render()
        except Exception:
            self.status(traceback.format_exc())

    def add_v3_portfolio_source(self) -> None:
        try:
            name = self.dpg.get_value("v3_portfolio_source")
            self.active = self.session.add_v3_portfolio_source(name)
            self._refresh_source_controls()
            self.render()
            self.status(
                f"Loaded Segmenter V3 portfolio fixture {name} as source data only")
        except Exception:
            self.status(traceback.format_exc())

    def spec(self) -> BlurSpec:
        dpg = self.dpg
        def optional(tag: str, default: float) -> float:
            try:
                return float(dpg.get_value(tag))
            except Exception:
                return float(default)
        return BlurSpec(
            kind=dpg.get_value("blur_kind"),
            sigma=dpg.get_value("blur_sigma"),
            radius=dpg.get_value("blur_radius"),
            length=dpg.get_value("blur_length"),
            angle_degrees=dpg.get_value("blur_angle"),
            bend=dpg.get_value("blur_bend"),
            extent=dpg.get_value("blur_extent"),
            shift_x=dpg.get_value("shift_x"),
            shift_y=dpg.get_value("shift_y"),
            rotation_mean_degrees=optional("rotation_mean", 4.0),
            rotation_exposure_degrees=optional("rotation_exposure", 8.0),
            rolling_mean_x=optional("rolling_mean_x", 4.0),
            rolling_row_acceleration=optional(
                "rolling_row_acceleration", 1.0),
            rolling_exposure_extent=optional(
                "rolling_exposure_extent", 2.0),
            exposure_gain=optional("synthetic_exposure_gain", 1.0),
            seed=dpg.get_value("synthetic_seed"),
        )

    def synthesize(self, add_capture: bool = False) -> None:
        if not self.session.sources:
            self.status("Load a source image first")
            return
        try:
            kwargs = dict(
                read_noise_sigma=self.dpg.get_value("read_noise"),
                shot_peak=self.dpg.get_value("shot_peak"),
                seed=self.dpg.get_value("synthetic_seed"),
            )
            if add_capture:
                self.active = self.session.add_synthetic_capture(
                    self.active, self.spec(), **kwargs)
                self._refresh_source_controls()
            else:
                self.session.synthesize(self.active, self.spec(), **kwargs)
            self.render()
            record = self.session.sources[self.active]
            self.status(f"Synthesized {record.diagnostics['operator']}")
        except Exception:
            self.status(traceback.format_exc())

    def use_as_is(self) -> None:
        if not self.session.sources:
            self.status("Load a source image first")
            return
        try:
            record = self.session.use_as_is(self.active)
            self.render()
            self.status(
                "Using loaded pixels as the immutable deblurring observation; "
                "no clean truth and no kernel are assumed")
        except Exception:
            self.status(traceback.format_exc())

    def _start(self, label: str, operation) -> None:
        if not self._busy.acquire(blocking=False):
            self.status("A reconstruction is already running")
            return
        self.status(label)

        def work() -> None:
            try:
                message = operation()
                self.render()
                self.status(message)
            except Exception:
                self.status(traceback.format_exc())
            finally:
                self._busy.release()

        threading.Thread(target=work, daemon=True).start()

    def deblur_active(self) -> None:
        if not self.session.sources:
            self.status("Load an image first")
            return

        def operation() -> str:
            result = self.session.deblur_active(
                self.active,
                passes=self.dpg.get_value("solver_passes"),
            )
            diagnostic = self.session.sources[self.active].diagnostics
            if "field" in diagnostic:
                field = diagnostic["field"]
                if diagnostic.get("estimation_decision") == (
                    "abstain_noninvertible_barycentric_map"
                ):
                    return (
                        "Abstained: barycentric warp is not single-valued; "
                        f"folded pixels {100.0 * field['fold_fraction']:.2f}%; "
                        "the working observation was preserved and the fold "
                        "was transported to the sensitivity view"
                    )
                pullback = diagnostic["barycentric_pullback"]
                return (
                    f"{diagnostic['method']}; barycentric flow RMS "
                    f"{field['barycentric_flow_rms']:.3f} px; centered mixing RMS "
                    f"{field['centered_mixing_rms']:.3f} px; passes "
                    f"{diagnostic['passes_used']} ({diagnostic['stopped_by']}); "
                    f"coordinate residual "
                    f"{pullback['terminal_coordinate_residual_max']:.2e}; "
                    f"folds {100.0 * field['fold_fraction']:.2f}%; "
                    f"sensitivity RMS {diagnostic['uncertainty_rms']:.5f}; "
                    f"native/backend {diagnostic['operator_backend']}; "
                    f"measured synthetic gain {diagnostic['psnr_gain']:+.2f} dB"
                )
            shift = diagnostic["deterministic_shift_xy"]
            support = diagnostic["support_gate"]
            message = (
                f"{diagnostic['method']}; shift ({shift[0]:.2f}, {shift[1]:.2f}) px; "
                f"forward RMS {diagnostic['forward_rms']:.5f}; "
                f"observation unchanged={diagnostic['observation_unchanged']}; "
                f"unresolved spectrum {100.0 * support['dead_fraction']:.1f}%"
            )
            characteristic = diagnostic["characteristic_transport"]
            if characteristic.get("selected"):
                line_authority = characteristic.get(
                    "line_constraint_authority", 0.0)
                exact_used = characteristic.get(
                    "refinement_passes_used", 0)
                exact_stop = characteristic.get(
                    "refinement_stopped_by", "unknown")
                message += (
                    f"; {characteristic.get('method', 'exposure transport')} "
                    f"exact passes {exact_used} ({exact_stop}); "
                    f"line-constraint authority {line_authority:.3f}; "
                    f"exposure uncertainty RMS "
                    f"{characteristic['uncertainty_rms']:.5f}"
                )
                if "jacobian_max" in characteristic:
                    message += (
                        f"; path Jacobian "
                        f"{characteristic['jacobian_min']:.3f}-"
                        f"{characteristic['jacobian_max']:.3f}; "
                        f"endpoint seed/basin RMS "
                        f"{characteristic['endpoint_seed_basin_rms']:.5f}"
                    )
            else:
                message += (
                    f"; unsupported energy removed "
                    f"{100.0 * support['unsupported_energy_removed_fraction']:.1f}%"
                )
            if "psnr_gain" in diagnostic:
                message += f"; measured synthetic gain {diagnostic['psnr_gain']:+.2f} dB"
            else:
                message += (
                    f"; blind estimate {diagnostic['estimated_centered_kernel']} "
                    f"(confidence {diagnostic['estimation_confidence']:.2f}, "
                    f"inverse authority {diagnostic['blind_inverse_authority']:.2f})"
                )
            return message

        self._start(
            "Recovering deterministic transport, then centered mixing...",
            operation,
        )

    def deblur_pair(self) -> None:
        if len(self.session.sources) < 2:
            self.status("Load or synthesize two registered captures first")
            return
        try:
            first = self._index_from_label(self.dpg.get_value("pair_first"))
            second = self._index_from_label(self.dpg.get_value("pair_second"))
        except Exception:
            self.status(traceback.format_exc())
            return
        if first == second:
            self.status("Choose two different registered captures")
            return

        def operation() -> str:
            posterior, result = self.session.deblur_pair_uncertain(
                first,
                second,
                noise_sigma=self.dpg.get_value("pair_noise"),
                credibility=self.dpg.get_value("credibility"),
                maximum_branches=self.dpg.get_value("maximum_branches"),
                passes=self.dpg.get_value("solver_passes"),
            )
            self.active = first
            decision = result.diagnostics.get("decision", "transport_posterior")
            return (
                f"{decision}; pair {posterior.best.first.name} / "
                f"{posterior.best.second.name}; "
                f"p={posterior.best.probability:.3f}; effective hypotheses "
                f"{posterior.effective_hypotheses:.2f}; retained mass "
                f"{result.retained_probability:.3f}"
            )

        self._start("Estimating blur law and transporting uncertainty...", operation)

    def deblur_rotation_consensus(self) -> None:
        if len(self.session.sources) < 2:
            self.status("Load or synthesize at least two observations first")
            return
        try:
            first = self._index_from_label(self.dpg.get_value("pair_first"))
            second = self._index_from_label(self.dpg.get_value("pair_second"))
        except Exception:
            self.status(traceback.format_exc())
            return
        if first == second:
            self.status("Choose two different observations")
            return
        indices = [first, second]
        reference = indices.index(self.active) if self.active in indices else None

        def operation() -> str:
            target, result = self.session.deblur_rotation_consensus(
                indices,
                reference_index=reference,
                duty_cycle=self.dpg.get_value("rotation_duty_cycle"),
                passes=self.dpg.get_value("solver_passes"),
            )
            self.active = target
            diagnostic = self.session.sources[target].diagnostics
            gain = diagnostic.get("psnr_gain_over_best_capture")
            gain_text = "" if gain is None else f"; synthetic gain {gain:+.2f} dB"
            return (
                f"{diagnostic['estimation_decision']}; angles "
                f"{np.round(result.estimate.relative_mean_angles_degrees, 3).tolist()} deg; "
                f"extents {np.round(result.estimate.exposure_extents_degrees, 3).tolist()} deg; "
                f"cycle RMS {result.estimate.cycle_rms_degrees:.4f} deg; "
                f"confidence {result.estimate.confidence:.3f}; "
                f"uncertainty RMS {diagnostic['uncertainty_rms']:.5f}"
                f"{gain_text}"
            )

        self._start("Estimating rotational flow consensus...", operation)

    def deblur_all_multicapture(self) -> None:
        if len(self.session.sources) < 3:
            self.status("Load at least three same-scene working observations first")
            return
        indices = list(range(len(self.session.sources)))
        target = self.active if self.active in indices else indices[0]

        def operation() -> str:
            target_index, inverse, posterior = (
                self.session.deblur_multicapture_posterior(
                    indices,
                    target_index=target,
                    passes=self.dpg.get_value("solver_passes"),
                )
            )
            self.active = target_index
            diagnostic = self.session.sources[target_index].diagnostics
            synthetic_psnr = diagnostic.get("posterior_psnr")
            score_text = (
                "" if synthetic_psnr is None
                else f"; synthetic posterior {synthetic_psnr:.2f} dB"
            )
            return (
                f"unified {len(indices)}-capture posterior; inverse mass "
                f"{posterior.atlas_mass:.3f}; center mass "
                f"{posterior.center_mass:.3f}; noise mass "
                f"{posterior.denoise_mass:.3f}; passes "
                f"{inverse.diagnostics['passes_used']} "
                f"({inverse.diagnostics['stopped_by']}); uncertainty RMS "
                f"{np.sqrt(np.mean(posterior.uncertainty ** 2)):.5f}"
                f"{score_text}"
            )

        self._start(
            "Transporting all loaded same-scene observations through one posterior...",
            operation,
        )

    def deblur_dense_consensus(self) -> None:
        if len(self.session.sources) < 2:
            self.status("Load or synthesize at least two observations first")
            return
        try:
            first = self._index_from_label(self.dpg.get_value("pair_first"))
            second = self._index_from_label(self.dpg.get_value("pair_second"))
        except Exception:
            self.status(traceback.format_exc())
            return
        if first == second:
            self.status("Choose two different observations")
            return
        target = self.active if self.active in (first, second) else first

        def operation() -> str:
            target_index, result = self.session.deblur_dense_pair_consensus(
                first,
                second,
                target_index=target,
                duty_cycle=self.dpg.get_value("flow_duty_cycle"),
                passes=self.dpg.get_value("solver_passes"),
            )
            self.active = target_index
            diagnostic = self.session.sources[target_index].diagnostics
            gain = diagnostic.get("psnr_gain_over_best_capture")
            gain_text = "" if gain is None else f"; synthetic gain {gain:+.2f} dB"
            atlas_rms = diagnostic.get(
                "fourier_circle_atlas_transport", {}).get(
                    "flow_rms_pixels", 0.0)
            chart_records = diagnostic.get(
                "fourier_circle_atlas_transport", {}).get(
                    "chart_records", [])
            chart_x_std = float(np.std([
                chart["translation_xy"][0] for chart in chart_records
            ])) if chart_records else 0.0
            precision_mean = diagnostic.get("sensor_precision_mean", (1.0, 1.0))
            if not result.estimate.relative_motion_observable:
                return (
                    f"{diagnostic['estimation_decision']}; common warp/exposure "
                    "remains unidentifiable; observations preserved in their "
                    "mean gauge; zero inverse passes"
                )
            return (
                f"{diagnostic['estimation_decision']}; dense flow RMS "
                f"{diagnostic['flow_rms']:.3f} px; cycle RMS "
                f"{diagnostic['cycle_rms']:.3f} px; transported connection "
                f"{diagnostic['transport_authority_mean']:.3f}; passes "
                f"{diagnostic['passes_used']} ({diagnostic['stopped_by']}); "
                f"chart {diagnostic['execution_chart']}; max folds "
                f"{100.0 * max(diagnostic['fold_fractions']):.3f}%; "
                f"flow-measure entropy "
                f"{diagnostic.get('latent_measure_entropy_mean', 0.0):.3f}; "
                f"circle-atlas RMS {atlas_rms:.3f} px; "
                f"chart-X variation {chart_x_std:.3f} px; "
                f"atlas prior center "
                f"{diagnostic.get('flow_atlas_prior_center', 0.0) or 0.0:.3f}; "
                f"relative exposure "
                f"{diagnostic.get('relative_gain_second_over_first', 1.0):.3f}; "
                f"radiometric authority "
                f"{diagnostic.get('radiometric_authority', 0.0):.3f}; "
                f"sensor precision {np.round(precision_mean, 3).tolist()}; "
                f"positive-atlas authority "
                f"{diagnostic.get('correction_authority_mean', 0.0):.3f}; "
                f"jointly unsupported "
                f"{100.0 * diagnostic['unsupported_visibility_fraction']:.3f}%; "
                f"uncertainty RMS {diagnostic['uncertainty_rms']:.5f}; "
                "output gauge symmetric pair midpoint"
                f"{gain_text}"
            )

        self._start(
            "Estimating one continuous global/local flow atlas and transporting exposure...",
            operation,
        )

    def save_result(self) -> None:
        try:
            if not self.session.sources:
                raise ValueError("no source is loaded")
            record = self.session.sources[self.active]
            if record.result is None:
                raise ValueError("run deblurring before saving")
            value = self.dpg.get_value("output_path").strip()
            if not value:
                base = record.path.stem if record.path else "deblurred"
                value = str(Path.cwd() / f"{base}_deblurred.png")
                self.dpg.set_value("output_path", value)
            output = Path(value).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            rgb = np.rint(_display_rgb(record.result) * 255.0).astype(np.uint8)
            Image.fromarray(rgb, mode="RGB").save(output)
            self.status(f"Saved {output}")
        except Exception:
            self.status(traceback.format_exc())

    def status(self, message: str) -> None:
        self.dpg.set_value("status", str(message))

    def render(self) -> None:
        if not self.session.sources:
            return
        record = self.session.sources[self.active]
        views: list[tuple[str, np.ndarray]] = [
            ("Loaded source", record.original),
            ("Working observation", record.observation),
        ]
        if record.result is not None:
            views.append(("Deblurred", record.result))
        if record.uncertainty is not None:
            views.append((
                "Exposure-transport uncertainty (diagnostic)",
                _uncertainty_rgb(record.uncertainty),
            ))
        for tag in self.texture_tags:
            if self.dpg.does_item_exist(tag):
                self.dpg.delete_item(tag)
        self.texture_tags.clear()
        self.dpg.delete_item("view_tabs", children_only=True)
        with self.dpg.texture_registry(show=False):
            for index, (_name, image) in enumerate(views):
                width, height, data = _rgba_texture(image)
                tag = f"deblur_texture_{index}"
                self.dpg.add_static_texture(width, height, data, tag=tag)
                self.texture_tags.append(tag)
        for index, (name, image) in enumerate(views):
            height, width = image.shape[:2]
            scale = min(1.0, 1100.0 / max(width, 1), 590.0 / max(height, 1))
            with self.dpg.tab(label=name, parent="view_tabs"):
                self.dpg.add_text(f"{width} x {height}")
                self.dpg.add_image(
                    f"deblur_texture_{index}",
                    width=max(1, int(width * scale)),
                    height=max(1, int(height * scale)),
                )


def run_gui() -> None:
    try:
        import dearpygui.dearpygui as dpg
    except ImportError as error:  # pragma: no cover - desktop dependency
        raise SystemExit(
            "Dear PyGui is not installed. Use the repository .venv-jpeg "
            "environment or install dearpygui and Pillow."
        ) from error

    dpg.create_context()
    app = DeblurWorkbenchApp(dpg)
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=app.choose_sources,
        tag="source_dialog",
        width=760,
        height=520,
        modal=True,
    ):
        dpg.add_file_extension(".png", color=(90, 205, 255, 255))
        dpg.add_file_extension(".jpg", color=(245, 190, 85, 255))
        dpg.add_file_extension(".jpeg", color=(245, 190, 85, 255))
        dpg.add_file_extension(".tif", color=(180, 150, 255, 255))
        dpg.add_file_extension(".tiff", color=(180, 150, 255, 255))
        dpg.add_file_extension(".*")

    with dpg.window(tag="primary", label="Personal Deblurrer"):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Load source(s)...",
                callback=lambda: dpg.show_item("source_dialog"))
            dpg.add_combo(
                (), tag="active_source", label="Active source", width=430,
                callback=app.choose_active)
            dpg.add_input_text(
                tag="output_path", hint="Output PNG path", width=420)
            dpg.add_button(label="Save result", callback=lambda: app.save_result())
        with dpg.group(horizontal=True):
            dpg.add_combo(
                V3_SKIMAGE_PORTFOLIO,
                tag="v3_portfolio_source",
                label="Segmenter V3 scikit-image portfolio",
                default_value="astronaut",
                width=260,
            )
            dpg.add_button(
                label="Add portfolio source",
                callback=lambda: app.add_v3_portfolio_source(),
            )
            dpg.add_text(
                "Fixtures only - no V3 segmentation machinery is used.",
                color=(150, 175, 195, 255),
            )

        with dpg.collapsing_header(
            label="Input role and synthetic blur generator", default_open=True):
            dpg.add_text(
                "Blur image: loaded pixels are known synthetic truth.  "
                "Use as-is: loaded pixels are the unknown observation.",
                color=(180, 205, 220, 255),
            )
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    (
                        "None", "Gaussian", "Disk", "Line", "Curve",
                        "Random path", "Rotational exposure",
                        "Rolling shutter exposure",
                    ),
                    tag="blur_kind", label="Blur family", default_value="Line", width=150)
                dpg.add_slider_float(
                    tag="blur_sigma", label="Gaussian sigma", default_value=2.0,
                    min_value=0.1, max_value=8.0, width=180)
                dpg.add_slider_float(
                    tag="blur_radius", label="Defocus radius", default_value=3.0,
                    min_value=0.5, max_value=12.0, width=180)
                dpg.add_slider_float(
                    tag="blur_length", label="Path length", default_value=11.0,
                    min_value=1.0, max_value=41.0, width=180)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    tag="blur_angle", label="Angle", default_value=0.0,
                    min_value=0.0, max_value=179.0, width=180)
                dpg.add_slider_float(
                    tag="blur_bend", label="Curve bend", default_value=4.0,
                    min_value=-16.0, max_value=16.0, width=180)
                dpg.add_slider_float(
                    tag="blur_extent", label="Random extent", default_value=7.0,
                    min_value=1.0, max_value=24.0, width=180)
                dpg.add_input_int(
                    tag="synthetic_seed", label="Seed", default_value=0, width=100)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    tag="rotation_mean", label="Mean rotation (degrees)",
                    default_value=4.0, min_value=-12.0, max_value=12.0,
                    width=220)
                dpg.add_slider_float(
                    tag="rotation_exposure", label="Rotation exposure (degrees)",
                    default_value=8.0, min_value=0.0, max_value=24.0,
                    width=220)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    tag="rolling_mean_x", label="Rolling mean X",
                    default_value=4.0, min_value=-16.0, max_value=16.0,
                    width=190)
                dpg.add_slider_float(
                    tag="rolling_row_acceleration", label="Row acceleration",
                    default_value=1.0, min_value=-2.0, max_value=2.0,
                    width=190)
                dpg.add_slider_float(
                    tag="rolling_exposure_extent", label="Rolling exposure extent",
                    default_value=2.0, min_value=0.0, max_value=12.0,
                    width=210)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    tag="shift_x", label="Deterministic shift X", default_value=0.0,
                    min_value=-16.0, max_value=16.0, width=190)
                dpg.add_slider_float(
                    tag="shift_y", label="Deterministic shift Y", default_value=0.0,
                    min_value=-16.0, max_value=16.0, width=190)
                dpg.add_slider_float(
                    tag="read_noise", label="Read noise sigma", default_value=0.002,
                    min_value=0.0, max_value=0.05, format="%.5f", width=180)
                dpg.add_input_float(
                    tag="shot_peak", label="Poisson peak (0=off)",
                    default_value=0.0, min_value=0.0, width=150)
                dpg.add_slider_float(
                    tag="synthetic_exposure_gain", label="Exposure gain",
                    default_value=1.0, min_value=0.1, max_value=3.0,
                    format="%.3f", width=180)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Blur image", callback=lambda: app.synthesize(False),
                    width=220)
                dpg.add_button(
                    label="Use as-is for deblurring", callback=lambda: app.use_as_is(),
                    width=220)

        with dpg.collapsing_header(
            label="Unified shift-first reconstruction", default_open=True):
            dpg.add_text(
                "One inverse: deterministic coordinate transport first; "
                "positive mixing around the transported centers second.",
                color=(180, 205, 220, 255),
            )
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    tag="solver_passes", label="Positive transport passes", default_value=64,
                    min_value=1, max_value=100, width=120)
                dpg.add_button(
                    label="Deblur working observation",
                    callback=lambda: app.deblur_active(), width=260)

        with dpg.collapsing_header(
            label="Optional multi-observation estimation", default_open=False):
            dpg.add_text(
                "The explicitly chosen captures constrain one continuous positive "
                "transport atlas over dense, global-circle, and local-circle paths. "
                "Translation, affine motion, rotation, shear, curved layered motion, "
                "rolling-shutter acceleration, exposure gain, clipping, and local "
                "deformation coexist; no blur family or preferred frame is selected. "
                "Equal raster size alone never implies registration.",
                color=(150, 175, 195, 255),
            )
            dpg.add_button(
                label="Deblur all loaded same-scene observations (3+)",
                callback=lambda: app.deblur_all_multicapture(),
                width=430,
            )
            with dpg.group(horizontal=True):
                dpg.add_combo((), tag="pair_first", label="Pair A", width=300)
                dpg.add_combo((), tag="pair_second", label="Pair B", width=300)
                dpg.add_slider_float(
                    tag="flow_duty_cycle", label="Exposure / frame interval",
                    default_value=0.5, min_value=0.0, max_value=2.0,
                    format="%.3f", width=180)
                dpg.add_button(
                    label="Estimate continuous flow atlas + deblur Pair A / Pair B",
                    callback=lambda: app.deblur_dense_consensus(), width=420)

        dpg.add_text(
            "Load an image, choose exactly one input role, then deblur the "
            "working observation. The loaded source buffer is never overwritten.",
            tag="status", wrap=1400)
        dpg.add_separator()
        with dpg.tab_bar(tag="view_tabs"):
            pass

    # Make the built-in source library concrete on first launch.  This also
    # gives users an immediate image to blur without navigating a file dialog.
    try:
        app.active = app.session.add_v3_portfolio_source("astronaut")
        app._refresh_source_controls()
        app.render()
        app.status(
            "Loaded Segmenter V3 portfolio fixture astronaut as source data only")
    except Exception:
        app.status(traceback.format_exc())

    dpg.create_viewport(title="Personal Deblurrer", width=1500, height=920)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    run_gui()
