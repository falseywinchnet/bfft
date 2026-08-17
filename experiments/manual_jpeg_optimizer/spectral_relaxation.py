"""Globally optimal connection-spectral relaxation of channel phase alignment.

The nonconvex field asks for one orthogonal three-channel frame per DCT block.
Its spectral relaxation keeps only global orthonormality::

    min tr(Z.T @ L @ Z)  subject to Z.T @ Z = I_3.

Ky Fan's minimum principle proves that the bottom three eigenvectors are a
global minimizer and that the sum of their eigenvalues is the attained global
lower bound.  Local polar projection is a single explicit unrelaxation step;
it is reported separately and is not included in the relaxation's proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


@dataclass
class SpectralRelaxationResult:
    frames: np.ndarray
    relaxed_sections: np.ndarray
    eigenvalues: np.ndarray
    relaxed_optimum: float
    lower_bound: float
    eigengap: float
    eigen_residual: float
    relaxed_connection_energy: float
    rounded_connection_energy: float
    edge_count: int
    mean_confidence: float

    def report(self) -> dict:
        return {
            "eigenvalues": self.eigenvalues.tolist(),
            "relaxed_optimum": self.relaxed_optimum,
            "lower_bound": self.lower_bound,
            "eigengap": self.eigengap,
            "eigen_residual": self.eigen_residual,
            "relaxed_connection_energy": self.relaxed_connection_energy,
            "rounded_connection_energy": self.rounded_connection_energy,
            "edge_count": self.edge_count,
            "mean_confidence": self.mean_confidence,
            "proof": (
                "Ky Fan minimum principle: the bottom three orthonormal "
                "eigenvectors globally minimize trace(Z^T L Z)."
            ),
        }


@dataclass(frozen=True)
class OrthogonalConnections:
    left: np.ndarray
    right: np.ndarray
    rotation: np.ndarray
    weight: np.ndarray


def _block_edges(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = labels.shape
    grid = np.arange(height * width, dtype=np.int32).reshape(height, width)
    left = np.concatenate((grid[:, :-1].ravel(), grid[:-1, :].ravel()))
    right = np.concatenate((grid[:, 1:].ravel(), grid[1:, :].ravel()))
    same = labels.ravel()[left] == labels.ravel()[right]
    return left, right, same


def build_connections(
    coefficients: np.ndarray,
    labels: np.ndarray,
    *,
    cross_region_weight: float = 0.05,
) -> OrthogonalConnections:
    """Measure one O(3) relative channel connection on every block edge."""
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (64, 3):
        raise ValueError("coefficients must have shape (blocks, 64, 3)")
    if labels.size != len(values):
        raise ValueError("labels must contain one entry per block")
    left, right, same = _block_edges(labels)

    # Equalize physical Y/Cb/Cr scale before measuring relative orientation.
    scale = np.sqrt(np.mean(values[:, 1:, :] ** 2, axis=(0, 1))) + 1e-9
    features = values[:, 1:, :] / scale[None, None, :]
    rotation = np.empty((len(left), 3, 3), dtype=np.float64)
    confidence = np.empty(len(left), dtype=np.float64)
    for edge, (i, j) in enumerate(zip(left, right)):
        fi, fj = features[i], features[j]
        cross = fi.T @ fj
        u, singular, vt = np.linalg.svd(cross, full_matrices=False)
        # R maps a section at j into i's measured channel chart.
        rotation[edge] = u @ vt
        denominator = np.linalg.norm(fi) * np.linalg.norm(fj) + 1e-12
        confidence[edge] = np.clip(np.sum(singular) / denominator, 0.0, 1.0)
    region_factor = np.where(same, 1.0, np.clip(cross_region_weight, 0.0, 1.0))
    # A small floor keeps the global chart connected without pretending weak
    # evidence is strong.
    weight = region_factor * (0.05 + 0.95 * confidence)
    return OrthogonalConnections(left, right, rotation, weight)


def connection_laplacian(
    block_count: int,
    connections: OrthogonalConnections,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    degree = np.zeros(block_count, dtype=np.float64)
    for i, j, rotation, weight in zip(
        connections.left,
        connections.right,
        connections.rotation,
        connections.weight,
    ):
        degree[i] += weight
        degree[j] += weight
        for a in range(3):
            for b in range(3):
                rows.extend((3 * int(i) + a, 3 * int(j) + b))
                columns.extend((3 * int(j) + b, 3 * int(i) + a))
                data.extend((-weight * rotation[a, b], -weight * rotation[a, b]))
    adjacency_part = sparse.coo_matrix(
        (data, (rows, columns)), shape=(3 * block_count, 3 * block_count)
    ).tocsr()
    diagonal = np.repeat(degree, 3)
    laplacian = adjacency_part + sparse.diags(diagonal)
    return laplacian, degree


def _polar_frames(sections: np.ndarray, degree: np.ndarray) -> np.ndarray:
    blocks = len(degree)
    fields = sections.reshape(blocks, 3, 3) / np.sqrt(
        np.maximum(degree, 1e-12)
    )[:, None, None]
    frames = np.empty_like(fields)
    for block, field in enumerate(fields):
        u, _, vt = np.linalg.svd(field, full_matrices=False)
        frames[block] = u @ vt

    # Fix the unobservable global O(3) gauge nearest standard YCbCr.
    aggregate = np.sum(frames, axis=0)
    u, _, vt = np.linalg.svd(aggregate.T, full_matrices=False)
    gauge = u @ vt
    return frames @ gauge


def connection_energy(field: np.ndarray, connections: OrthogonalConnections) -> float:
    value = np.asarray(field, dtype=np.float64)
    energy = 0.0
    for i, j, rotation, weight in zip(
        connections.left,
        connections.right,
        connections.rotation,
        connections.weight,
    ):
        difference = value[int(i)] - rotation @ value[int(j)]
        energy += float(weight * np.sum(difference * difference))
    return energy


def solve_spectral_relaxation(
    coefficients: np.ndarray,
    labels: np.ndarray,
    *,
    cross_region_weight: float = 0.05,
    tolerance: float = 1e-10,
) -> SpectralRelaxationResult:
    """Solve the globally optimal three-section phase relaxation."""
    connections = build_connections(
        coefficients, labels, cross_region_weight=cross_region_weight
    )
    laplacian, degree = connection_laplacian(len(coefficients), connections)
    inverse_root = np.repeat(1.0 / np.sqrt(np.maximum(degree, 1e-12)), 3)
    normalized = sparse.diags(inverse_root) @ laplacian @ sparse.diags(inverse_root)
    eigenvalues, sections = eigsh(normalized, k=4, which="SA", tol=tolerance)
    order = np.argsort(eigenvalues)
    eigenvalues, sections = eigenvalues[order], sections[:, order]
    relaxed = sections[:, :3]
    residual = normalized @ relaxed - relaxed * eigenvalues[:3][None, :]
    eigen_residual = float(np.linalg.norm(residual, ord="fro"))
    optimum = float(np.sum(eigenvalues[:3]))
    frames = _polar_frames(relaxed, degree)
    physical_relaxed = (
        relaxed.reshape(len(coefficients), 3, 3)
        / np.sqrt(np.maximum(degree, 1e-12))[:, None, None]
    )
    normalized_frames = frames / np.sqrt(max(float(np.sum(degree)), 1e-12))
    return SpectralRelaxationResult(
        frames=frames,
        relaxed_sections=physical_relaxed,
        eigenvalues=eigenvalues,
        relaxed_optimum=optimum,
        lower_bound=optimum,
        eigengap=float(eigenvalues[3] - eigenvalues[2]),
        eigen_residual=eigen_residual,
        relaxed_connection_energy=connection_energy(physical_relaxed, connections),
        rounded_connection_energy=connection_energy(normalized_frames, connections),
        edge_count=len(connections.left),
        mean_confidence=float(np.mean(connections.weight)),
    )


def frame_views(
    frames: np.ndarray,
    block_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Render each globally smooth channel section as an RGB direction map."""
    bh, bw = block_shape
    height, width = image_shape
    value = np.asarray(frames, dtype=np.float64).reshape(bh, bw, 3, 3)
    views = {}
    for channel in range(3):
        colors = np.uint8(np.clip(127.5 + 127.5 * value[..., :, channel], 0, 255))
        expanded = np.repeat(np.repeat(colors, 8, axis=0), 8, axis=1)
        views[f"global_channel_{channel + 1}_direction"] = expanded[:height, :width]
    return views
