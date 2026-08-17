"""Spatial/frequency ownership transport of signed Y/Cb/Cr DCT constituents.

The transport never changes channel or DCT frequency.  For every
``(frequency, Y/Cb/Cr, sign)`` field it moves nonnegative constituent mass
between adjacent blocks and adjacent positions of the within-block DCT
cascade, simultaneously. Positive and negative ownership are transported
separately, so apparent smoothing cannot be obtained by silently cancelling
opposite constituents. Frequency edges let spatial phase frustration pass
through the DCT triplet instead of jamming into blur.

For a source mass field ``m`` and block-graph Laplacian ``L`` the relaxation is

    min_z  1/2 ||z-m||^2 + lambda/2 z^T L z.

Its unique global minimizer is ``z=(I+lambda L)^-1 m``.  Because the inverse is
an M-matrix inverse and ``L 1=0``, nonnegativity and total mass are preserved.
The edge flow ``lambda*w*(z_left-z_right)`` has divergence exactly ``z-m`` and
is the explicit ownership-transfer record.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from .core import ZIGZAG, quality_table
from .ownership_bifurcation import _grid_edges


@dataclass(frozen=True)
class SpatialDCTTransportConfig:
    transport_lambda: float = 0.01
    frequency_weight: float = 0.1
    cross_region_weight: float = 0.05
    luma_mobility: float = 1.0
    cb_mobility: float = 1.0
    cr_mobility: float = 1.0
    tolerance: float = 1e-11
    maximum_iterations: int = 500


@dataclass
class SpatialDCTTransportResult:
    coefficients: np.ndarray
    positive_mass: np.ndarray
    negative_mass: np.ndarray
    positive_flow: np.ndarray
    negative_flow: np.ndarray
    edge_left: np.ndarray
    edge_right: np.ndarray
    edge_weight: np.ndarray
    edge_kind: np.ndarray
    objective: float
    fidelity_energy: float
    smoothness_energy: float
    kkt_residual: float
    flow_divergence_residual: float
    positive_mass_error: float
    negative_mass_error: float
    minimum_transported_mass: float
    rms_coefficient_displacement: float
    flow_hash: str
    iterations: tuple[int, ...]
    spatial_edges: int
    frequency_edges: int
    config: SpatialDCTTransportConfig

    def report(self) -> dict:
        return {
            "objective": self.objective,
            "fidelity_energy": self.fidelity_energy,
            "smoothness_energy": self.smoothness_energy,
            "kkt_residual": self.kkt_residual,
            "flow_divergence_residual": self.flow_divergence_residual,
            "positive_mass_error": self.positive_mass_error,
            "negative_mass_error": self.negative_mass_error,
            "minimum_transported_mass": self.minimum_transported_mass,
            "rms_coefficient_displacement": self.rms_coefficient_displacement,
            "edge_count": len(self.edge_left),
            "spatial_edges": self.spatial_edges,
            "frequency_edges": self.frequency_edges,
            "iterations": list(self.iterations),
            "flow_hash": self.flow_hash,
            "config": self.config.__dict__,
            "proof": (
                "Each signed constituent field is the unique global minimizer "
                "of a strictly convex block-graph transport relaxation. The "
                "reported edge-flow divergence reconstructs every ownership "
                "change, while positive and negative mass remain separately "
                "conserved."
            ),
        }


def _ownership_laplacian(
    labels: np.ndarray,
    config: SpatialDCTTransportConfig,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    modes = 63
    block_left, block_right = _grid_edges(*labels.shape)
    same = labels.ravel()[block_left] == labels.ravel()[block_right]
    block_weight = np.where(
        same, 1.0, np.clip(config.cross_region_weight, 0.0, 1.0)
    ).astype(np.float64)
    mode = np.arange(modes, dtype=np.int64)
    spatial_left = (block_left[:, None] * modes + mode[None, :]).ravel()
    spatial_right = (block_right[:, None] * modes + mode[None, :]).ravel()
    spatial_weight = np.repeat(block_weight, modes)

    raster = {pair: pair[0] * 8 + pair[1] for pair in ZIGZAG}
    order = np.asarray([raster[pair] - 1 for pair in ZIGZAG[1:]], dtype=np.int64)
    blocks = labels.size
    block = np.arange(blocks, dtype=np.int64)
    frequency_left = (block[:, None] * modes + order[:-1][None, :]).ravel()
    frequency_right = (block[:, None] * modes + order[1:][None, :]).ravel()
    rank_weight = config.frequency_weight / np.sqrt(1.0 + np.arange(62))
    frequency_weight = np.tile(rank_weight, blocks)

    left = np.concatenate((spatial_left, frequency_left))
    right = np.concatenate((spatial_right, frequency_right))
    weight = np.concatenate((spatial_weight, frequency_weight))
    kind = np.concatenate((
        np.zeros(len(spatial_left), dtype=np.uint8),
        np.ones(len(frequency_left), dtype=np.uint8),
    ))
    keep = weight > 0
    left, right, weight, kind = left[keep], right[keep], weight[keep], kind[keep]
    count = blocks * modes
    diagonal = np.bincount(
        np.concatenate((left, right)),
        weights=np.concatenate((weight, weight)),
        minlength=count,
    )
    laplacian = sparse.coo_matrix(
        (
            np.concatenate((diagonal, -weight, -weight)),
            (
                np.concatenate((np.arange(count), left, right)),
                np.concatenate((np.arange(count), right, left)),
            ),
        ),
        shape=(count, count),
    ).tocsr()
    return (
        laplacian, left, right, weight, kind,
        len(spatial_left), len(frequency_left),
    )


def _divergence(flow: np.ndarray, left: np.ndarray, right: np.ndarray, blocks: int) -> np.ndarray:
    result = np.zeros((blocks, *flow.shape[1:]), dtype=np.float64)
    np.add.at(result, left, -flow)
    np.add.at(result, right, flow)
    return result


def transport_spatial_dct(
    coefficients: np.ndarray,
    block_labels: np.ndarray,
    quality: int,
    config: SpatialDCTTransportConfig = SpatialDCTTransportConfig(),
) -> SpatialDCTTransportResult:
    source = np.asarray(coefficients, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (64, 3):
        raise ValueError("coefficients must have shape (blocks, 64, 3)")
    if block_labels.size != len(source):
        raise ValueError("block_labels must contain one label per block")
    if config.transport_lambda < 0 or config.frequency_weight < 0:
        raise ValueError("transport_lambda must be nonnegative")
    quantizers = np.stack((
        quality_table(quality, False).reshape(64),
        quality_table(quality, True).reshape(64),
        quality_table(quality, True).reshape(64),
    ), axis=-1)
    normalized = source[:, 1:, :] / quantizers[None, 1:, :]
    positive_source = np.maximum(normalized, 0.0)
    negative_source = np.maximum(-normalized, 0.0)
    laplacian, left, right, weight, kind, spatial_edges, frequency_edges = _ownership_laplacian(
        block_labels, config
    )
    nodes = len(source) * 63
    positive_flat = positive_source.reshape(nodes, 3)
    negative_flat = negative_source.reshape(nodes, 3)
    positive_solution = np.empty_like(positive_flat)
    negative_solution = np.empty_like(negative_flat)
    iteration_counts: list[int] = []
    mobilities = np.asarray((
        config.luma_mobility, config.cb_mobility, config.cr_mobility
    ), dtype=np.float64)
    if np.any(mobilities < 0):
        raise ValueError("channel mobilities must be nonnegative")
    operators: list[sparse.csr_matrix] = []
    preconditioners: list[LinearOperator] = []
    for mobility in mobilities:
        local_operator = (
            sparse.eye(nodes, format="csr")
            + config.transport_lambda * mobility * laplacian
        )
        inverse_diagonal = 1.0 / local_operator.diagonal()
        operators.append(local_operator)
        preconditioners.append(LinearOperator(
            (nodes, nodes), matvec=lambda value, diagonal=inverse_diagonal: diagonal * value
        ))
    for source_field, destination in (
        (positive_flat, positive_solution),
        (negative_flat, negative_solution),
    ):
        for channel in range(3):
            count = 0

            def callback(_value):
                nonlocal count
                count += 1

            solution, info = cg(
                operators[channel],
                source_field[:, channel],
                M=preconditioners[channel],
                rtol=config.tolerance,
                atol=0.0,
                maxiter=config.maximum_iterations,
                callback=callback,
            )
            if info != 0:
                raise RuntimeError(f"ownership transport CG did not converge: {info}")
            destination[:, channel] = solution
            iteration_counts.append(count)
    # The exact inverse preserves each channel's total mass because L1=0.
    # Remove the finite CG residual in that null direction explicitly.
    positive_solution += (
        np.sum(positive_flat, axis=0) - np.sum(positive_solution, axis=0)
    )[None, :] / nodes
    negative_solution += (
        np.sum(negative_flat, axis=0) - np.sum(negative_solution, axis=0)
    )[None, :] / nodes
    shape = positive_source.shape
    positive = positive_solution.reshape(shape)
    negative = negative_solution.reshape(shape)
    # Numerical noise from the direct solve is many orders below a JPEG bin;
    # retain it for the KKT certificate but report the nonnegativity margin.
    transported = positive - negative
    output = source.copy()
    output[:, 1:, :] = transported * quantizers[None, 1:, :]

    flow_scale = config.transport_lambda * weight[:, None] * mobilities[None, :]
    positive_flow = flow_scale * (
        positive_solution[left] - positive_solution[right]
    )
    negative_flow = flow_scale * (
        negative_solution[left] - negative_solution[right]
    )
    positive_delta = positive - positive_source
    negative_delta = negative - negative_source
    positive_divergence = _divergence(positive_flow, left, right, nodes).reshape(shape)
    negative_divergence = _divergence(negative_flow, left, right, nodes).reshape(shape)
    kkt_positive = np.column_stack([
        operators[channel] @ positive_solution[:, channel] - positive_flat[:, channel]
        for channel in range(3)
    ])
    kkt_negative = np.column_stack([
        operators[channel] @ negative_solution[:, channel] - negative_flat[:, channel]
        for channel in range(3)
    ])
    edge_positive = positive_solution[left] - positive_solution[right]
    edge_negative = negative_solution[left] - negative_solution[right]
    fidelity = 0.5 * float(np.sum(positive_delta ** 2) + np.sum(negative_delta ** 2))
    smoothness = 0.5 * config.transport_lambda * float(np.sum(
        weight[:, None] * mobilities[None, :]
        * (edge_positive ** 2 + edge_negative ** 2)
    ))
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(positive_flow).view(np.uint8))
    digest.update(np.ascontiguousarray(negative_flow).view(np.uint8))
    return SpatialDCTTransportResult(
        coefficients=output,
        positive_mass=positive,
        negative_mass=negative,
        positive_flow=positive_flow,
        negative_flow=negative_flow,
        edge_left=left,
        edge_right=right,
        edge_weight=weight,
        edge_kind=kind,
        objective=fidelity + smoothness,
        fidelity_energy=fidelity,
        smoothness_energy=smoothness,
        kkt_residual=max(
            float(np.max(np.abs(kkt_positive))),
            float(np.max(np.abs(kkt_negative))),
        ),
        flow_divergence_residual=max(
            float(np.max(np.abs(positive_delta - positive_divergence))),
            float(np.max(np.abs(negative_delta - negative_divergence))),
        ),
        positive_mass_error=float(np.max(np.abs(
            np.sum(positive, axis=(0, 1)) - np.sum(positive_source, axis=(0, 1))
        ))),
        negative_mass_error=float(np.max(np.abs(
            np.sum(negative, axis=(0, 1)) - np.sum(negative_source, axis=(0, 1))
        ))),
        minimum_transported_mass=float(min(np.min(positive), np.min(negative))),
        rms_coefficient_displacement=float(np.sqrt(np.mean(
            (transported - normalized) ** 2
        ))),
        flow_hash=digest.hexdigest(),
        iterations=tuple(iteration_counts),
        spatial_edges=spatial_edges,
        frequency_edges=frequency_edges,
        config=config,
    )
