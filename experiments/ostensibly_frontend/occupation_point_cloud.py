"""Mass-preserving point clouds and constrained phone-geometry registration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree
from scipy.stats import rankdata


@dataclass(frozen=True)
class PointCloudConfig:
    """Resolution and sampling controls for an occupation-field lift."""

    row_upsample: int = 4
    frame_upsample: int = 8
    point_count: int = 32768
    height_percentile: float = 99.5
    mass_exponent: float = 0.5

    def __post_init__(self) -> None:
        if self.row_upsample < 1 or self.frame_upsample < 1:
            raise ValueError("upsampling factors must be positive integers")
        if self.point_count < 1:
            raise ValueError("point count must be positive")
        if not 0.0 < self.height_percentile <= 100.0:
            raise ValueError("invalid height percentile")
        if not 0.0 < self.mass_exponent <= 1.0:
            raise ValueError("mass exponent must lie in (0, 1]")


@dataclass(frozen=True)
class SupportGeometryConfig:
    """Quantization and jitter model for density-free ridge support."""

    row_bins: int = 256
    frame_bins: int = 256
    minimum_cell_count: int = 2
    jitter_sigma_cells: float = 2.0

    def __post_init__(self) -> None:
        if self.row_bins < 16 or self.frame_bins < 16:
            raise ValueError("support geometry needs at least 16 bins per axis")
        if self.minimum_cell_count < 1:
            raise ValueError("minimum cell count must be positive")
        if self.jitter_sigma_cells <= 0.0:
            raise ValueError("jitter sigma must be positive")


@dataclass(frozen=True)
class OccupationPointCloud:
    """A deterministic empirical measure over ``(row, frame, height)``."""

    points: np.ndarray
    high_resolution_field: np.ndarray
    native_shape: tuple[int, int]
    mass: float


@dataclass(frozen=True)
class CloudFitConfig:
    """Constrained registration and robust-distance controls."""

    fit_point_count: int = 4096
    row_metric_scale: float = 0.25
    frame_metric_scale: float = 1.0
    height_metric_scale: float = 2.0
    distance_mode: str = "chamfer"
    sliced_projection_count: int = 32
    sliced_projection_seed: int = 72031
    trim_fraction: float = 0.12
    row_scale_bounds: tuple[float, float] = (0.70, 1.45)
    frame_scale_bounds: tuple[float, float] = (0.55, 1.50)
    row_shift_bounds: tuple[float, float] = (-40.0, 40.0)
    frame_shift_bounds: tuple[float, float] = (-3.0, 3.0)
    pitch_drift_bounds: tuple[float, float] = (-2.0, 2.0)
    global_iterations: int = 80
    population_size: int = 12
    random_seed: int = 20260826

    def __post_init__(self) -> None:
        if self.fit_point_count < 32:
            raise ValueError("fit point count must be at least 32")
        if min(
            self.row_metric_scale,
            self.frame_metric_scale,
            self.height_metric_scale,
        ) <= 0.0:
            raise ValueError("metric scales must be positive")
        if self.distance_mode not in {"chamfer", "sliced_wasserstein"}:
            raise ValueError("unsupported cloud distance mode")
        if self.sliced_projection_count < 4:
            raise ValueError("sliced Wasserstein needs at least four projections")
        if not 0.0 <= self.trim_fraction < 0.5:
            raise ValueError("trim fraction must lie in [0, 0.5)")
        if self.global_iterations < 1 or self.population_size < 4:
            raise ValueError("invalid optimizer schedule")
        for bounds in (
            self.row_scale_bounds,
            self.frame_scale_bounds,
            self.row_shift_bounds,
            self.frame_shift_bounds,
            self.pitch_drift_bounds,
        ):
            if bounds[0] >= bounds[1]:
                raise ValueError("fit bounds must be increasing")


@dataclass(frozen=True)
class CloudFitResult:
    """Registration parameters and before/after geometric residuals."""

    row_scale: float
    frame_scale: float
    row_shift: float
    frame_shift: float
    pitch_drift: float
    prefit_distance: float
    postfit_distance: float
    transformed_points: np.ndarray
    moving_center: np.ndarray
    reference_center: np.ndarray
    optimizer_success: bool
    optimizer_evaluations: int


def _validate_field(field: np.ndarray) -> np.ndarray:
    source = np.asarray(field, dtype=np.float64)
    if source.ndim != 2 or not source.size or not np.all(np.isfinite(source)):
        raise ValueError("occupation field must be one finite, nonempty 2-D array")
    return np.maximum(source, 0.0)


def occupation_to_point_cloud(
    field: np.ndarray,
    config: PointCloudConfig = PointCloudConfig(),
) -> OccupationPointCloud:
    """Lift a continuous occupation field into a deterministic empirical cloud.

    Linear interpolation increases coordinate resolution without inventing the
    ringing that a higher-order interpolant can add.  Systematic inverse-CDF
    sampling preserves the field's mass: repeated points are intentional and
    represent repeated stochastic occupation rather than duplicate removal.
    """

    source = _validate_field(field)
    high_resolution = ndi.zoom(
        source,
        (config.row_upsample, config.frame_upsample),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    # Square-root occupation is the Hellinger amplitude of the empirical
    # measure.  It keeps weak recurrent ridges represented instead of allowing
    # one dominant harmonic to consume nearly the entire finite cloud.
    mass = np.power(high_resolution, config.mass_exponent).ravel()
    total = float(np.sum(mass))
    if total <= 0.0:
        return OccupationPointCloud(
            points=np.empty((0, 3), dtype=np.float64),
            high_resolution_field=high_resolution,
            native_shape=source.shape,
            mass=0.0,
        )

    cumulative = np.cumsum(mass / total)
    cumulative[-1] = 1.0
    quantiles = (
        np.arange(config.point_count, dtype=np.float64) + 0.5
    ) / config.point_count
    flat_indices = np.searchsorted(cumulative, quantiles, side="left")
    rows, frames = np.unravel_index(flat_indices, high_resolution.shape)
    positive = high_resolution[high_resolution > 0.0]
    gauge = max(
        float(np.percentile(positive, config.height_percentile)),
        1e-30,
    )
    heights = np.clip(high_resolution[rows, frames] / gauge, 0.0, 1.0)
    native_rows = (rows.astype(np.float64) + 0.5) / config.row_upsample - 0.5
    native_frames = (
        (frames.astype(np.float64) + 0.5) / config.frame_upsample - 0.5
    )
    points = np.column_stack((native_rows, native_frames, heights))
    return OccupationPointCloud(
        points=points,
        high_resolution_field=high_resolution,
        native_shape=source.shape,
        mass=total,
    )


def marginal_copula_cloud(points: np.ndarray) -> np.ndarray:
    """Remove monotone row/time marginals while retaining joint occupation.

    This is the empirical occupation copula.  It is invariant to arbitrary
    monotone frequency and timing reparameterizations, including nonuniform
    vocal-tract warps that a global affine fit cannot express.
    """

    cloud = _validate_points(points, "points")
    count = cloud.shape[0]
    canonical = cloud.copy()
    canonical[:, 0] = (rankdata(cloud[:, 0], method="average") - 0.5) / count
    canonical[:, 1] = (rankdata(cloud[:, 1], method="average") - 0.5) / count
    return canonical


def affine_marginal_cloud(points: np.ndarray) -> np.ndarray:
    """Remove global pitch and duration without flattening their marginals.

    Frequency is centered and scaled by robust quantiles; frame coordinates
    are mapped affinely onto ``[0, 1]``.  Unlike ``marginal_copula_cloud``,
    occupation density along both axes remains part of the empirical measure.
    """

    cloud = _validate_points(points, "points")
    canonical = cloud.copy()
    row_center = float(np.median(cloud[:, 0]))
    row_scale = float(
        np.quantile(cloud[:, 0], 0.9) - np.quantile(cloud[:, 0], 0.1)
    )
    canonical[:, 0] = (cloud[:, 0] - row_center) / max(row_scale, 1e-12)
    frame0 = float(np.min(cloud[:, 1]))
    frame_scale = float(np.max(cloud[:, 1]) - frame0)
    canonical[:, 1] = (cloud[:, 1] - frame0) / max(frame_scale, 1e-12)
    return canonical


def unique_support_cloud(
    points: np.ndarray,
    config: SupportGeometryConfig = SupportGeometryConfig(),
) -> np.ndarray:
    """Return one point per occupied high-resolution row/time cell.

    Repeated stochastic samples establish that a cell is supported, but they do
    not survive as multiplicity.  This prevents ridge energy or cloud density
    from becoming a terminal similarity shortcut.
    """

    cloud = _validate_points(points, "points")
    scaled = np.column_stack(
        (
            np.clip(cloud[:, 0], 0.0, np.nextafter(1.0, 0.0)) * config.row_bins,
            np.clip(cloud[:, 1], 0.0, np.nextafter(1.0, 0.0)) * config.frame_bins,
        )
    )
    cells, counts = np.unique(scaled.astype(np.int64), axis=0, return_counts=True)
    cells = cells[counts >= config.minimum_cell_count]
    if not cells.size:
        raise ValueError("no support cells survive the occupation threshold")
    return cells.astype(np.float64) + 0.5


def jittered_support_distance(
    moving: np.ndarray,
    reference: np.ndarray,
    config: SupportGeometryConfig = SupportGeometryConfig(),
) -> float:
    """Compare ridge support after analytically marginalizing local jitter.

    A Gaussian miss cost is the deterministic limit of superimposing random
    pitch/time-offset copies.  It rewards sub-feature agreement while
    saturating for unrelated clutter; the symmetric average requires coverage
    in both directions.
    """

    left = unique_support_cloud(moving, config)
    right = unique_support_cloud(reference, config)
    left_distance = cKDTree(right).query(left, k=1, workers=1)[0]
    right_distance = cKDTree(left).query(right, k=1, workers=1)[0]
    sigma = config.jitter_sigma_cells

    def miss(distance: np.ndarray) -> float:
        return float(np.mean(1.0 - np.exp(-0.5 * (distance / sigma) ** 2)))

    return 0.5 * (miss(left_distance) + miss(right_distance))


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not cloud.size:
        raise ValueError(f"{name} must be a nonempty N x 3 cloud")
    if not np.all(np.isfinite(cloud)):
        raise ValueError(f"{name} must contain only finite points")
    return cloud


def _even_subset(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] <= count:
        return points
    indices = np.linspace(0, points.shape[0] - 1, count, dtype=np.int64)
    return points[indices]


def transform_phone_cloud(
    points: np.ndarray,
    parameters: np.ndarray,
    moving_center: np.ndarray,
    reference_center: np.ndarray,
) -> np.ndarray:
    """Apply time/pitch-preserving registration to one phone cloud.

    Parameters are ``log(row scale), log(frame scale), row shift, frame shift,
    pitch drift``.  Drift is row displacement per native time frame.  No free
    3-D rotation or nonlinear warp is permitted.
    """

    cloud = np.asarray(points, dtype=np.float64)
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (5,):
        raise ValueError("phone-cloud transform requires five parameters")
    relative_row = cloud[:, 0] - moving_center[0]
    relative_frame = cloud[:, 1] - moving_center[1]
    transformed = cloud.copy()
    transformed[:, 0] = (
        reference_center[0]
        + np.exp(values[0]) * relative_row
        + values[2]
        + values[4] * relative_frame
    )
    transformed[:, 1] = (
        reference_center[1] + np.exp(values[1]) * relative_frame + values[3]
    )
    return transformed


def _metric_points(points: np.ndarray, config: CloudFitConfig) -> np.ndarray:
    return points * np.asarray(
        [config.row_metric_scale, config.frame_metric_scale, config.height_metric_scale]
    )


def _trimmed_mean_squared(values: np.ndarray, trim_fraction: float) -> float:
    if not values.size:
        return float("inf")
    keep = max(1, int(np.floor(values.size * (1.0 - trim_fraction))))
    if keep < values.size:
        values = np.partition(values, keep - 1)[:keep]
    return float(np.mean(values * values))


def symmetric_trimmed_chamfer(
    moving: np.ndarray,
    reference: np.ndarray,
    config: CloudFitConfig = CloudFitConfig(),
) -> float:
    """Return robust symmetric Chamfer distance in the declared phone metric."""

    left = _metric_points(_validate_points(moving, "moving"), config)
    right = _metric_points(_validate_points(reference, "reference"), config)
    left_distance = cKDTree(right).query(left, k=1, workers=1)[0]
    right_distance = cKDTree(left).query(right, k=1, workers=1)[0]
    squared = 0.5 * (
        _trimmed_mean_squared(left_distance, config.trim_fraction)
        + _trimmed_mean_squared(right_distance, config.trim_fraction)
    )
    return float(np.sqrt(squared))


@lru_cache(maxsize=16)
def _sliced_directions(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_directions = rng.normal(size=(count, 3))
    random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)
    return np.vstack((np.eye(3, dtype=np.float64), random_directions))


def sliced_wasserstein_projection(
    points: np.ndarray,
    config: CloudFitConfig = CloudFitConfig(distance_mode="sliced_wasserstein"),
) -> np.ndarray:
    """Project and sort one empirical cloud in the deterministic SWD basis."""

    metric_points = _metric_points(_validate_points(points, "points"), config)
    directions = _sliced_directions(
        config.sliced_projection_count, config.sliced_projection_seed
    )
    return np.sort(metric_points @ directions.T, axis=0)


def sliced_wasserstein_projection_distance(
    left_projection: np.ndarray,
    right_projection: np.ndarray,
) -> float:
    """Compare two preprojected empirical clouds at common quantiles."""

    left_projection = np.asarray(left_projection, dtype=np.float64)
    right_projection = np.asarray(right_projection, dtype=np.float64)
    if (
        left_projection.ndim != 2
        or right_projection.ndim != 2
        or not left_projection.size
        or not right_projection.size
        or left_projection.shape[1] != right_projection.shape[1]
    ):
        raise ValueError("SWD projections must be nonempty with equal direction count")
    if left_projection.shape[0] != right_projection.shape[0]:
        count = min(left_projection.shape[0], right_projection.shape[0])
        quantiles = (np.arange(count, dtype=np.float64) + 0.5) / count

        def resample(values: np.ndarray) -> np.ndarray:
            source_quantiles = (
                np.arange(values.shape[0], dtype=np.float64) + 0.5
            ) / values.shape[0]
            return np.column_stack(
                [
                    np.interp(quantiles, source_quantiles, values[:, column])
                    for column in range(values.shape[1])
                ]
            )

        left_projection = resample(left_projection)
        right_projection = resample(right_projection)
    difference = left_projection - right_projection
    return float(np.sqrt(np.mean(difference * difference)))


def resample_sliced_wasserstein_projection(
    projection: np.ndarray, quantile_count: int
) -> np.ndarray:
    """Compress a sorted SWD projection onto fixed midpoint quantiles."""

    projection = np.asarray(projection, dtype=np.float64)
    if projection.ndim != 2 or not projection.size:
        raise ValueError("SWD projection must be a nonempty matrix")
    if quantile_count < 1:
        raise ValueError("SWD quantile count must be positive")
    if projection.shape[0] == quantile_count:
        return projection.copy()
    quantiles = (np.arange(quantile_count, dtype=np.float64) + 0.5) / quantile_count
    source_quantiles = (
        np.arange(projection.shape[0], dtype=np.float64) + 0.5
    ) / projection.shape[0]
    return np.column_stack(
        [
            np.interp(quantiles, source_quantiles, projection[:, column])
            for column in range(projection.shape[1])
        ]
    )


def sliced_wasserstein_distance(
    moving: np.ndarray,
    reference: np.ndarray,
    config: CloudFitConfig = CloudFitConfig(distance_mode="sliced_wasserstein"),
) -> float:
    """Compare full empirical mass through deterministic 1-D projections.

    Unlike Chamfer, this distance cannot discard multiplicity: two clouds with
    identical geometric support but different occupation mass remain distinct.
    """

    return sliced_wasserstein_projection_distance(
        sliced_wasserstein_projection(moving, config),
        sliced_wasserstein_projection(reference, config),
    )


def cloud_distance(
    moving: np.ndarray,
    reference: np.ndarray,
    config: CloudFitConfig,
) -> float:
    if config.distance_mode == "chamfer":
        return symmetric_trimmed_chamfer(moving, reference, config)
    return sliced_wasserstein_distance(moving, reference, config)


def fit_phone_cloud(
    moving: np.ndarray,
    reference: np.ndarray,
    config: CloudFitConfig = CloudFitConfig(),
) -> CloudFitResult:
    """Fit a candidate phone cloud to a reference with five constrained DOF."""

    moving_full = _validate_points(moving, "moving")
    reference_full = _validate_points(reference, "reference")
    moving_fit = _even_subset(moving_full, config.fit_point_count)
    reference_fit = _even_subset(reference_full, config.fit_point_count)
    moving_center = np.median(moving_fit[:, :2], axis=0)
    reference_center = np.median(reference_fit[:, :2], axis=0)
    zero = np.zeros(5, dtype=np.float64)

    def objective(parameters: np.ndarray) -> float:
        transformed = transform_phone_cloud(
            moving_fit, parameters, moving_center, reference_center
        )
        distance = cloud_distance(transformed, reference_fit, config)
        # A weak physical prior only resolves nearly equivalent extreme warps.
        regularization = 2e-3 * (
            parameters[0] ** 2
            + parameters[1] ** 2
            + 0.02 * parameters[4] ** 2
        )
        return distance + regularization

    bounds = (
        tuple(np.log(config.row_scale_bounds)),
        tuple(np.log(config.frame_scale_bounds)),
        config.row_shift_bounds,
        config.frame_shift_bounds,
        config.pitch_drift_bounds,
    )
    global_result = differential_evolution(
        objective,
        bounds,
        maxiter=config.global_iterations,
        popsize=config.population_size,
        seed=config.random_seed,
        polish=False,
        updating="immediate",
        workers=1,
    )
    local_result = minimize(
        objective,
        global_result.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 500, "xtol": 1e-5, "ftol": 1e-6},
    )
    parameters = np.asarray(local_result.x, dtype=np.float64)
    transformed_full = transform_phone_cloud(
        moving_full, parameters, moving_center, reference_center
    )
    prefit = cloud_distance(
        transform_phone_cloud(moving_fit, zero, moving_center, reference_center),
        reference_fit,
        config,
    )
    postfit = cloud_distance(
        transform_phone_cloud(moving_fit, parameters, moving_center, reference_center),
        reference_fit,
        config,
    )
    return CloudFitResult(
        row_scale=float(np.exp(parameters[0])),
        frame_scale=float(np.exp(parameters[1])),
        row_shift=float(parameters[2]),
        frame_shift=float(parameters[3]),
        pitch_drift=float(parameters[4]),
        prefit_distance=prefit,
        postfit_distance=postfit,
        transformed_points=transformed_full,
        moving_center=moving_center,
        reference_center=reference_center,
        optimizer_success=bool(local_result.success),
        optimizer_evaluations=int(global_result.nfev + local_result.nfev),
    )
