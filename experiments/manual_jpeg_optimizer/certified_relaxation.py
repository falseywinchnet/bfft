"""Certified connection-valued convex relaxation for JPEG preprocessing.

For fixed block regions, local channel frames, and signed connections, solve

  min_z  1/2 ||z-c||_2^2
       + sum_i a_i |(Fz)_i|
       + sum_(e,k) b_(e,k) ||z[j,k] - s[e,k] z[i,k]||_2.

``F`` is blockwise orthogonal, so ``z`` retains all three YCbCr components.
The graph term is the relaxed-chip analogue: relative orientation is carried
on an incidence relation rather than collapsed into a scalar marginal.  The
problem is strongly convex and therefore has one global optimum.  A feasible
Fenchel dual supplies a rigorous lower bound and primal--dual gap certificate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .core import (
    JPEGConfig,
    _region_labels,
    block_dct,
    inverse_block_dct,
    rgb_to_ycc,
    ycc_to_rgb,
)


@dataclass(frozen=True)
class RelaxationConfig:
    rate_lambda: float = 0.20
    connection_lambda: float = 0.10
    frame_mode: str = "chroma"
    cartoon_sigma: float = 1.2
    region_threshold: float = 0.58
    cross_region_weight: float = 0.05
    iterations: int = 1000
    check_interval: int = 25
    relative_gap_tolerance: float = 1e-6


@dataclass
class RelaxationResult:
    rgb: np.ndarray
    coefficients: np.ndarray
    labels: np.ndarray
    iterations: int
    primal: float
    dual_lower_bound: float
    absolute_gap: float
    relative_gap: float
    converged: bool
    zero_fraction_before: float
    zero_fraction_after: float
    connection_residual_before: float
    connection_residual_after: float
    config: RelaxationConfig

    def report(self) -> dict:
        value = asdict(self)
        value.pop("rgb")
        value.pop("coefficients")
        value.pop("labels")
        return value


@dataclass(frozen=True)
class ConnectionGraph:
    left: np.ndarray
    right: np.ndarray
    sign: np.ndarray
    weight: np.ndarray


def _coefficients(ycc: np.ndarray) -> np.ndarray:
    channels = [block_dct(ycc[..., channel]) for channel in range(3)]
    # blocks x modes x channels
    packed = np.stack(channels, axis=-1)
    return packed.reshape(-1, 64, 3)


def _frames(coefficients: np.ndarray, labels: np.ndarray, mode: str) -> np.ndarray:
    if mode not in {"identity", "chroma", "full"}:
        raise ValueError("frame_mode must be identity, chroma, or full")
    blocks = len(coefficients)
    result = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], blocks, axis=0)
    if mode == "identity":
        return result
    flat_labels = labels.ravel()
    for region in range(int(flat_labels.max()) + 1):
        members = np.flatnonzero(flat_labels == region)
        if not len(members):
            continue
        samples = coefficients[members, 1:, :].reshape(-1, 3)
        if mode == "chroma":
            covariance = samples[:, 1:].T @ samples[:, 1:]
            values, vectors = np.linalg.eigh(covariance)
            vectors = vectors[:, np.argsort(values)[::-1]]
            for column in range(2):
                pivot = int(np.argmax(np.abs(vectors[:, column])))
                if vectors[pivot, column] < 0.0:
                    vectors[:, column] *= -1.0
            if np.linalg.det(vectors) < 0.0:
                vectors[:, 1] *= -1.0
            frame = np.eye(3)
            frame[1:, 1:] = vectors
        else:
            covariance = samples.T @ samples
            values, frame = np.linalg.eigh(covariance)
            frame = frame[:, np.argsort(values)[::-1]]
            for column in range(3):
                pivot = int(np.argmax(np.abs(frame[:, column])))
                if frame[pivot, column] < 0.0:
                    frame[:, column] *= -1.0
            if np.linalg.det(frame) < 0.0:
                frame[:, -1] *= -1.0
        result[members] = frame
    return result


def _to_local(global_coefficients: np.ndarray, frames: np.ndarray) -> np.ndarray:
    return np.einsum("bca,bmc->bma", frames, global_coefficients, optimize=True)


def _to_global(local_coefficients: np.ndarray, frames: np.ndarray) -> np.ndarray:
    return np.einsum("bca,bma->bmc", frames, local_coefficients, optimize=True)


def _graph(local_source: np.ndarray, labels: np.ndarray, cross_weight: float) -> ConnectionGraph:
    height, width = labels.shape
    grid = np.arange(height * width, dtype=np.int32).reshape(height, width)
    left = np.concatenate((grid[:, :-1].ravel(), grid[:-1, :].ravel()))
    right = np.concatenate((grid[:, 1:].ravel(), grid[1:, :].ravel()))
    same = labels.ravel()[left] == labels.ravel()[right]
    weight = np.where(same, 1.0, np.clip(cross_weight, 0.0, 1.0))
    correlation = np.sum(local_source[left] * local_source[right], axis=-1)
    sign = np.where(correlation < 0.0, -1.0, 1.0)
    # DC is an offset/mass coordinate, not a carrier phase.
    sign[:, 0] = 1.0
    return ConnectionGraph(left, right, sign, weight.astype(np.float64))


def _global_rate_weight() -> np.ndarray:
    u, v = np.mgrid[:8, :8]
    radial = np.sqrt(u * u + v * v).reshape(64)
    frequency = 0.75 + 0.18 * radial ** 1.35
    frequency[0] = 0.0  # never buy bytes by changing block means
    channel = np.array((1.0, 1.08, 1.08), dtype=np.float64)
    return frequency[:, None] * channel[None, :]


def _edge_mode_weight() -> np.ndarray:
    u, v = np.mgrid[:8, :8]
    radial = np.sqrt(u * u + v * v).reshape(64)
    # Connection agreement is strongest on stable low/mid modes.
    weight = 1.0 / (1.0 + 0.12 * radial)
    weight[0] = 0.0
    return weight


def _difference(value: np.ndarray, graph: ConnectionGraph) -> np.ndarray:
    return value[graph.right] - graph.sign[..., None] * value[graph.left]


def _adjoint(
    unary_dual: np.ndarray,
    edge_dual: np.ndarray,
    frames: np.ndarray,
    graph: ConnectionGraph,
) -> np.ndarray:
    result = np.einsum("bca,bmc->bma", frames, unary_dual, optimize=True)
    np.add.at(result, graph.right, edge_dual)
    np.add.at(result, graph.left, -graph.sign[..., None] * edge_dual)
    return result


def _primal_dual_values(
    value: np.ndarray,
    source: np.ndarray,
    unary_dual: np.ndarray,
    edge_dual: np.ndarray,
    frames: np.ndarray,
    graph: ConnectionGraph,
    unary_bound: np.ndarray,
    edge_bound: np.ndarray,
) -> tuple[float, float, float]:
    global_value = _to_global(value, frames)
    differences = _difference(value, graph)
    primal = (
        0.5 * float(np.sum((value - source) ** 2))
        + float(np.sum(unary_bound * np.abs(global_value)))
        + float(np.sum(edge_bound * np.linalg.norm(differences, axis=-1)))
    )
    adjoint = _adjoint(unary_dual, edge_dual, frames, graph)
    dual = float(np.sum(source * adjoint) - 0.5 * np.sum(adjoint * adjoint))
    gap = max(0.0, primal - dual)
    return primal, dual, gap


def solve_coefficients(
    global_source: np.ndarray,
    labels: np.ndarray,
    config: RelaxationConfig = RelaxationConfig(),
    *,
    fixed_frames: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Solve and certify the fixed-frame connection relaxation."""
    source_global = np.asarray(global_source, dtype=np.float64)
    if source_global.ndim != 3 or source_global.shape[1:] != (64, 3):
        raise ValueError("global_source must have shape (blocks, 64, 3)")
    if labels.size != len(source_global):
        raise ValueError("labels must contain one value per block")
    if fixed_frames is None:
        frames = _frames(source_global, labels, config.frame_mode)
    else:
        frames = np.asarray(fixed_frames, dtype=np.float64)
        if frames.shape != (len(source_global), 3, 3):
            raise ValueError("fixed_frames must have shape (blocks, 3, 3)")
        # The explicit multiply avoids accepting a lossy chart.
        gram = np.matmul(np.swapaxes(frames, -1, -2), frames)
        if not np.allclose(gram, np.eye(3)[None], atol=1e-7):
            raise ValueError("fixed_frames must be blockwise orthogonal")
    source = _to_local(source_global, frames)
    graph = _graph(source, labels, config.cross_region_weight)
    unary_bound = float(config.rate_lambda) * _global_rate_weight()[None, :, :]
    edge_bound = (
        float(config.connection_lambda)
        * graph.weight[:, None]
        * _edge_mode_weight()[None, :]
    )

    # ||[F;D]||^2 <= 1 + 2*d_max = 9 on a four-neighbor grid.
    step = 0.32
    value = source.copy()
    extrapolated = value.copy()
    unary_dual = np.zeros_like(source_global)
    edge_dual = np.zeros((len(graph.left), 64, 3), dtype=np.float64)
    primal = dual = gap = math.inf
    converged = False
    completed = 0
    for iteration in range(1, int(config.iterations) + 1):
        global_extrapolated = _to_global(extrapolated, frames)
        unary_dual += step * global_extrapolated
        unary_dual = np.clip(unary_dual, -unary_bound, unary_bound)

        edge_dual += step * _difference(extrapolated, graph)
        norms = np.linalg.norm(edge_dual, axis=-1)
        scale = np.maximum(1.0, norms / np.maximum(edge_bound, 1e-300))
        edge_dual /= scale[..., None]
        edge_dual[edge_bound <= 0.0] = 0.0

        previous = value
        descent = value - step * _adjoint(
            unary_dual, edge_dual, frames, graph
        )
        value = (descent + step * source) / (1.0 + step)
        extrapolated = 2.0 * value - previous
        completed = iteration

        if iteration % max(1, int(config.check_interval)) == 0 or iteration == config.iterations:
            primal, dual, gap = _primal_dual_values(
                value, source, unary_dual, edge_dual, frames, graph,
                unary_bound, edge_bound,
            )
            relative = gap / max(1.0, abs(primal), abs(dual))
            if relative <= float(config.relative_gap_tolerance):
                converged = True
                break

    global_value = _to_global(value, frames)
    relative = gap / max(1.0, abs(primal), abs(dual))
    before_connection = float(np.mean(np.linalg.norm(_difference(source, graph), axis=-1)))
    after_connection = float(np.mean(np.linalg.norm(_difference(value, graph), axis=-1)))
    diagnostics: dict[str, float | int | bool] = {
        "iterations": completed,
        "primal": primal,
        "dual_lower_bound": dual,
        "absolute_gap": gap,
        "relative_gap": relative,
        "converged": converged,
        "zero_fraction_before": float(np.mean(np.abs(source_global) < 0.5)),
        "zero_fraction_after": float(np.mean(np.abs(global_value) < 0.5)),
        "connection_residual_before": before_connection,
        "connection_residual_after": after_connection,
    }
    return global_value, diagnostics


def coefficients_to_rgb(
    coefficients: np.ndarray,
    block_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Invert packed three-channel block coefficients to ordinary RGB."""
    bh, bw = block_shape
    unpacked = np.asarray(coefficients, dtype=np.float64).reshape(
        bh, bw, 8, 8, 3
    )
    channels = [
        inverse_block_dct(unpacked[..., channel], image_shape)
        for channel in range(3)
    ]
    return ycc_to_rgb(np.stack(channels, axis=-1))


def relax_rgb(
    rgb: np.ndarray,
    config: RelaxationConfig = RelaxationConfig(),
) -> RelaxationResult:
    ycc = rgb_to_ycc(rgb)
    labels, _ = _region_labels(
        ycc, config.cartoon_sigma, config.region_threshold
    )
    source = _coefficients(ycc)
    relaxed, diagnostics = solve_coefficients(source, labels, config)
    bh, bw = labels.shape
    return RelaxationResult(
        rgb=coefficients_to_rgb(relaxed, (bh, bw), ycc.shape[:2]),
        coefficients=relaxed,
        labels=labels,
        config=config,
        **diagnostics,
    )
