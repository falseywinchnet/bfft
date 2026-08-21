"""Scale-space complex for typed proposal transport across V3 resolutions.

Each independently inferred V3 resolution remains one layer.  Within-layer
edges are the explicit off-diagonal support--manifold proposal connections.
Between every pair of layers, exact common-raster overlap is the incidence
measure.  A final symmetric degree normalization creates one scale-space
operator.  Thus paths retain their order (scale change, assembly transport,
scale return) rather than averaging already-collapsed region kernels.
"""

from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


def _labels_on_grid(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(labels, dtype=np.int32)
    height, width = shape
    source_y = np.rint(
        np.linspace(0.0, value.shape[0] - 1, height)).astype(np.int32)
    source_x = np.rint(
        np.linspace(0.0, value.shape[1] - 1, width)).astype(np.int32)
    return value[source_y[:, None], source_x[None, :]]


def normalized_region_overlap(
    first: np.ndarray,
    second: np.ndarray,
    shape: tuple[int, int],
) -> sparse.csr_matrix:
    """Return cosine-normalized exact population overlap on a common grid."""
    left = _labels_on_grid(first, shape).ravel()
    right = _labels_on_grid(second, shape).ravel()
    left_count = int(np.max(first, initial=-1)) + 1
    right_count = int(np.max(second, initial=-1)) + 1
    key = left.astype(np.int64) * right_count + right
    unique, population = np.unique(key, return_counts=True)
    row = (unique // right_count).astype(np.int32)
    column = (unique % right_count).astype(np.int32)
    left_area = np.bincount(left, minlength=left_count).astype(np.float64)
    right_area = np.bincount(right, minlength=right_count).astype(np.float64)
    denominator = np.sqrt(left_area[row] * right_area[column])
    data = population.astype(np.float64) / np.maximum(denominator, 1.0)
    return sparse.csr_matrix(
        (data, (row, column)), shape=(left_count, right_count))


def build_multiscale_connection(
    labels: dict[int, np.ndarray],
    proposal_connection: dict[int, np.ndarray],
    *,
    shuffled_alignment: bool = False,
    shuffle_key: str = "control",
) -> dict:
    """Build the normalized multiplex adjacency with every scale-pair link."""
    scales = tuple(sorted(labels))
    if scales != tuple(sorted(proposal_connection)):
        raise ValueError("label and proposal scales disagree")
    shape = tuple(max(labels[side].shape[axis] for side in scales)
                  for axis in (0, 1))
    counts = {
        side: int(np.max(labels[side], initial=-1)) + 1 for side in scales}
    offsets = {}
    total = 0
    for side in scales:
        offsets[side] = total
        total += counts[side]
    blocks: list[list[sparse.spmatrix | None]] = [
        [None for _ in scales] for _ in scales]
    for index, side in enumerate(scales):
        value = np.asarray(proposal_connection[side], dtype=np.float64)
        if value.shape != (counts[side], counts[side]):
            raise ValueError(f"proposal connection has wrong shape at {side}")
        blocks[index][index] = sparse.csr_matrix(value)

    permutations = {
        side: np.arange(counts[side], dtype=np.int32) for side in scales}
    if shuffled_alignment:
        for side in scales:
            digest = hashlib.sha256(
                f"{shuffle_key}:{side}".encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little")
            permutations[side] = np.random.default_rng(seed).permutation(
                counts[side])
    overlap_summary = {}
    for first_index, second_index in combinations(range(len(scales)), 2):
        first, second = scales[first_index], scales[second_index]
        overlap = normalized_region_overlap(
            labels[first], labels[second], shape)
        if shuffled_alignment:
            overlap = overlap[
                permutations[first]][:, permutations[second]]
        blocks[first_index][second_index] = overlap
        blocks[second_index][first_index] = overlap.T
        overlap_summary[f"{first}_{second}"] = {
            "nonzeros": int(overlap.nnz),
            "measure": float(np.sum(overlap.data)),
        }
    adjacency = sparse.bmat(blocks, format="csr")
    adjacency = 0.5 * (adjacency + adjacency.T)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(
        1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0.0)
    normalized = sparse.diags(inverse) @ adjacency @ sparse.diags(inverse)
    return {
        "scales": scales,
        "shape": shape,
        "counts": counts,
        "offsets": offsets,
        "adjacency": adjacency,
        "degree": degree,
        "normalized_connection": normalized.tocsr(),
        "overlap": overlap_summary,
    }


def multiscale_point_sources(
    connection: dict,
    labels: dict[int, np.ndarray],
    points: dict[str, dict],
) -> tuple[tuple[str, ...], sparse.csc_matrix]:
    """Represent every normalized point by its one region atom at every scale."""
    names = tuple(points)
    scales = connection["scales"]
    amplitude = 1.0 / np.sqrt(len(scales))
    row = []
    column = []
    data = []
    for point_index, name in enumerate(names):
        x_normalized, y_normalized = points[name]["xy"]
        for side in scales:
            raster = labels[side]
            x = int(np.clip(round(
                float(x_normalized) * (raster.shape[1] - 1)),
                0, raster.shape[1] - 1))
            y = int(np.clip(round(
                float(y_normalized) * (raster.shape[0] - 1)),
                0, raster.shape[0] - 1))
            row.append(connection["offsets"][side] + int(raster[y, x]))
            column.append(point_index)
            data.append(amplitude)
    return names, sparse.csc_matrix(
        (data, (row, column)),
        shape=(connection["normalized_connection"].shape[0], len(names)),
    )


def _normalized_gram(gram: np.ndarray) -> np.ndarray:
    value = 0.5 * (
        np.asarray(gram, dtype=np.float64)
        + np.asarray(gram, dtype=np.float64).T)
    diagonal = np.maximum(np.diag(value), 0.0)
    denominator = np.sqrt(diagonal[:, None] * diagonal[None, :])
    return np.divide(
        value, denominator, out=np.zeros_like(value),
        where=denominator > 1e-30)


def query_multiscale_bloom(
    connection: dict,
    sources: sparse.csc_matrix,
    base_kernels: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """Query heat and transported part Grams without forming a dense kernel."""
    normalized = connection["normalized_connection"]
    if normalized.shape[0] != sources.shape[0]:
        raise ValueError("sources do not fit the scale-space connection")
    half_flow = np.exp(-0.5) * expm_multiply(
        0.5 * normalized, sources.toarray())
    heat_gram = half_flow.T @ half_flow
    transported_gram = np.zeros_like(heat_gram)
    for side in connection["scales"]:
        offset = connection["offsets"][side]
        count = connection["counts"][side]
        flow = half_flow[offset:offset + count]
        base = np.asarray(base_kernels[side], dtype=np.float64)
        if base.shape != (count, count):
            raise ValueError(f"base kernel has wrong shape at {side}")
        transported_gram += flow.T @ base @ flow
    return {
        "heat_gram": heat_gram,
        "heat_similarity": _normalized_gram(heat_gram),
        "transported_base_gram": transported_gram,
        "transported_base_similarity": _normalized_gram(transported_gram),
    }


def audit_similarity(
    names: tuple[str, ...],
    similarity: np.ndarray,
    points: dict[str, dict],
) -> dict:
    pairs = []
    for first, second in combinations(range(len(names)), 2):
        value = float(similarity[first, second])
        pairs.append({
            "first": names[first],
            "second": names[second],
            "same_instance": (
                points[names[first]]["instance"]
                == points[names[second]]["instance"]),
            "similarity": value,
            "distance": 1.0 - value,
        })
    same = np.asarray([
        pair["distance"] for pair in pairs if pair["same_instance"]])
    different = np.asarray([
        pair["distance"] for pair in pairs if not pair["same_instance"]])
    auc = None
    if len(same) and len(different):
        comparison = same[:, None] - different[None, :]
        auc = float(
            np.mean(comparison < 0.0)
            + 0.5 * np.mean(comparison == 0.0))
    return {"closer_pair_auc": auc, "pairs": pairs}
