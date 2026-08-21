"""Seed-free analytical bloom on the V3 directed-incidence complex.

This is the first falsifiable recognition operator, not a finished object
partition.  It constructs an unweighted bipartite topology from exact V3
incidences, whitens every measured relation jointly across the control atlas,
and evaluates one normalized heat exponential.  No object seed, expected
object count, merge order, or affinity threshold enters the construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply, splu


@dataclass(frozen=True)
class EmpiricalWhitener:
    """Complete non-null covariance whitening with only a numerical cutoff."""

    mean: np.ndarray
    basis: np.ndarray
    names: tuple[str, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        value = np.asarray(values, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != len(self.names):
            raise ValueError("relation matrix does not match the whitener")
        return np.ascontiguousarray((value - self.mean) @ self.basis)


def fit_joint_whitener(
    matrices: Iterable[np.ndarray],
    names: Iterable[str],
) -> EmpiricalWhitener:
    """Fit one atlas-wide whitener and retain every numerical covariance mode."""
    labels = tuple(names)
    values = np.vstack([
        np.asarray(matrix, dtype=np.float64) for matrix in matrices
    ])
    if values.ndim != 2 or values.shape[1] != len(labels) or not len(values):
        raise ValueError("non-empty, schema-aligned relation matrices required")
    if not np.all(np.isfinite(values)):
        raise ValueError("relation matrices must be finite")
    mean = np.mean(values, axis=0, keepdims=True)
    centered = values - mean
    covariance = centered.T @ centered / max(len(values), 1)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    largest = float(np.max(eigenvalue, initial=0.0))
    numerical = (
        np.finfo(np.float64).eps
        * max(values.shape)
        * max(largest, 1.0)
    )
    retained = eigenvalue > numerical
    if not np.any(retained):
        raise ValueError("relation atlas has no non-constant covariance mode")
    basis = eigenvector[:, retained] / np.sqrt(eigenvalue[retained])[None, :]
    return EmpiricalWhitener(
        mean=np.ascontiguousarray(mean),
        basis=np.ascontiguousarray(basis),
        names=labels,
    )


def _append(
    columns: list[np.ndarray],
    names: list[str],
    name: str,
    value: np.ndarray,
) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        columns.append(array[:, None])
        names.append(name)
    elif array.ndim == 2:
        columns.append(array)
        names.extend(f"{name}_{index}" for index in range(array.shape[1]))
    else:
        raise ValueError(f"relation channel {name!r} must be one- or two-dimensional")


def _stable_seed(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def relation_features(
    complex_: dict,
    bundle: dict,
    *,
    include_fused: bool = True,
    shuffled_outside: bool = False,
    shuffle_key: str = "control",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Measure incidence relations without collapsing them to an affinity.

    ``shuffled_outside`` is a matched null: the ordinary topology is left
    untouched while outside-region identities are permuted and all directed
    transitions are recomputed from those false partners.
    """
    incidence = bundle["incidence"]
    node = complex_["node"]
    region = np.asarray(incidence["region"], dtype=np.int32)
    outside = np.asarray(incidence["outside"], dtype=np.int32)
    if shuffled_outside:
        outside = np.random.default_rng(_stable_seed(shuffle_key)).permutation(
            outside)

    columns: list[np.ndarray] = []
    names: list[str] = []

    directed_nodes = (
        ("target", "target_mean"),
        ("cartoon", "cartoon_mean"),
        ("texture_rms", "texture_target_rms"),
    )
    for feature_name, node_name in directed_nodes:
        if shuffled_outside:
            value = node[node_name][outside] - node[node_name][region]
        else:
            value = incidence[f"{feature_name}_transition"]
        _append(columns, names, f"directed_{feature_name}", value)

    if include_fused and "fused_cartoon_mean" in node:
        for feature_name in ("cartoon", "texture", "residual"):
            if shuffled_outside:
                value = (
                    node[f"fused_{feature_name}_mean"][outside]
                    - node[f"fused_{feature_name}_mean"][region]
                )
            else:
                value = incidence[f"fused_{feature_name}_transition"]
            _append(columns, names, f"directed_fused_{feature_name}", value)

    _append(columns, names, "normal", np.column_stack((
        incidence["normal_x"], incidence["normal_y"])))
    _append(columns, names, "log_arc_length", np.log1p(incidence["length"]))
    _append(columns, names, "closed_arc", incidence["closed"].astype(float))
    for name in (
        "boundary_confidence",
        "literal_target_jump",
        "literal_cartoon_jump",
        "literal_texture_jump",
    ):
        _append(columns, names, name, incidence[name])
    if include_fused and "literal_fused_cartoon_jump" in incidence:
        for feature_name in ("cartoon", "texture", "residual"):
            name = f"literal_fused_{feature_name}_jump"
            _append(columns, names, name, incidence[name])

    context = (
        ("log_area", np.log1p(node["area"])),
        ("log_thickness", np.log1p(node["thickness"])),
        ("frame", node["touches_frame"].astype(float)),
        ("structural_purity", node["structural_purity"]),
        ("structural_entropy", node["structural_entropy"]),
        ("log_structural_support", np.log1p(
            node["structural_support_count"])),
    )
    for name, value in context:
        _append(
            columns,
            names,
            f"directed_{name}",
            np.asarray(value)[outside] - np.asarray(value)[region],
        )
    same_structure = (
        node["structural_dominant"][outside]
        == node["structural_dominant"][region]
    )
    _append(columns, names, "same_dominant_structure", same_structure.astype(float))
    return np.ascontiguousarray(np.column_stack(columns)), tuple(names)


def incidence_topology(
    complex_: dict,
    bundle: dict,
    *,
    include_arc_crossing: bool = True,
    include_junctions: bool = True,
) -> tuple[sparse.csr_matrix, dict[str, int], np.ndarray]:
    """Build the exact unweighted incidence/hub topology.

    Region hubs allow all one-sided observations of one V3 region to
    participate.  Arc hubs preserve the two sides of each connected interface.
    Continuation hubs preserve each exact junction proposal separately.
    Normalized adjacency gives every hub family its combinatorial measure;
    there are no evidence weights here.
    """
    incidence = bundle["incidence"]
    continuation = bundle["continuation"]
    count = len(incidence["arc"])
    region_count = int(complex_["region_count"])
    arc_count = int(np.max(incidence["arc"], initial=-1)) + 1
    continuation_count = (
        len(continuation["junction"]) if include_junctions else 0)
    region_offset = count
    arc_offset = region_offset + region_count
    continuation_offset = arc_offset + (arc_count if include_arc_crossing else 0)
    total = continuation_offset + continuation_count

    row: list[int] = []
    column: list[int] = []

    def connect(first: int, second: int) -> None:
        row.extend((first, second))
        column.extend((second, first))

    for identifier, region in enumerate(incidence["region"]):
        connect(identifier, region_offset + int(region))
    if include_arc_crossing:
        for identifier, arc in enumerate(incidence["arc"]):
            connect(identifier, arc_offset + int(arc))

    if include_junctions:
        lookup = {
            (int(arc), int(region)): identifier
            for identifier, (arc, region) in enumerate(zip(
                incidence["arc"], incidence["region"]))
        }
        for identifier, (first_arc, second_arc, shared) in enumerate(zip(
            continuation["first_arc"],
            continuation["second_arc"],
            continuation["region"],
        )):
            hub = continuation_offset + identifier
            first = lookup[(int(first_arc), int(shared))]
            second = lookup[(int(second_arc), int(shared))]
            connect(first, hub)
            connect(second, hub)

    adjacency = sparse.coo_matrix(
        (np.ones(len(row), dtype=np.float64), (row, column)),
        shape=(total, total),
    ).tocsr()
    # Repeated exact records carry repeated combinatorial measure.  This is
    # retained rather than silently binarized.
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree),
        where=degree > 0.0,
    )
    normalized = sparse.diags(inverse) @ adjacency @ sparse.diags(inverse)
    return normalized.tocsr(), {
        "incidences": count,
        "regions": region_count,
        "arcs": arc_count if include_arc_crossing else 0,
        "continuations": continuation_count,
        "total_states": total,
    }, degree


def analytical_bloom(
    whitened_incidence: np.ndarray,
    topology: sparse.csr_matrix,
    degree: np.ndarray,
) -> np.ndarray:
    """Evaluate ``exp(-(I-S))`` once on every empirical relation coordinate."""
    value = np.asarray(whitened_incidence, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] > topology.shape[0]:
        raise ValueError("incidence values must fit the topology")
    expanded = np.zeros((topology.shape[0], value.shape[1]), dtype=np.float64)
    expanded[:len(value)] = value

    # Remove the stationary degree mode before blooming.  This is the exact
    # configuration-model baseline of the unweighted graph, not a fitted cue.
    graph_degree = np.asarray(degree, dtype=np.float64)
    if graph_degree.shape != (topology.shape[0],):
        raise ValueError("degree vector must match the topology")
    stationary = np.sqrt(graph_degree)
    norm = float(np.linalg.norm(stationary))
    if norm > 0.0:
        stationary /= norm
        expanded -= stationary[:, None] * (stationary @ expanded)[None, :]
    flowed = np.exp(-1.0) * expm_multiply(topology, expanded)
    return np.ascontiguousarray(flowed[:len(value)])


def aggregate_region_embedding(
    incidence_values: np.ndarray,
    region: np.ndarray,
    region_count: int,
) -> np.ndarray:
    """Retain first and second relation moments for every V3 region."""
    value = np.asarray(incidence_values, dtype=np.float64)
    labels = np.asarray(region, dtype=np.int32)
    count = np.bincount(labels, minlength=region_count).astype(np.float64)
    mean = np.column_stack([
        np.bincount(labels, weights=value[:, index], minlength=region_count)
        / np.maximum(count, 1.0)
        for index in range(value.shape[1])
    ])
    second = np.column_stack([
        np.bincount(
            labels,
            weights=value[:, index] * value[:, index],
            minlength=region_count,
        ) / np.maximum(count, 1.0)
        for index in range(value.shape[1])
    ])
    rms = np.sqrt(np.maximum(second, 0.0))
    return np.ascontiguousarray(np.column_stack((mean, rms)))


def bloom_region_embedding(
    complex_: dict,
    bundle: dict,
    whitened_incidence: np.ndarray,
    *,
    include_arc_crossing: bool = True,
    include_junctions: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    topology, summary, degree = incidence_topology(
        complex_,
        bundle,
        include_arc_crossing=include_arc_crossing,
        include_junctions=include_junctions,
    )
    flowed = analytical_bloom(whitened_incidence, topology, degree)
    embedding = aggregate_region_embedding(
        flowed,
        bundle["incidence"]["region"],
        int(complex_["region_count"]),
    )
    return embedding, summary


def signed_incidence_connection(
    complex_: dict,
    bundle: dict,
    whitened_incidence: np.ndarray,
    *,
    mode: str = "signed",
    include_arc_crossing: bool = True,
    include_junctions: bool = True,
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, int]]:
    """Build a scalar connection Laplacian without an affinity threshold.

    Each topologically lawful pair receives its complete whitened empirical
    inner product, divided only by fibre dimension.  Its sign is connection
    phase and its magnitude is observed relation measure.  Absolute degree
    normalization makes ``I-S`` the standard positive signed Laplacian.

    ``unsigned`` is the matched holonomy ablation and ``topology`` discards all
    evidence while retaining the exact same incidence graph.
    """
    if mode not in ("signed", "unsigned", "topology"):
        raise ValueError("connection mode must be signed, unsigned, or topology")
    value = np.asarray(whitened_incidence, dtype=np.float64)
    incidence = bundle["incidence"]
    continuation = bundle["continuation"]
    count, dimension = value.shape
    if count != len(incidence["arc"]) or dimension < 1:
        raise ValueError("whitened incidence state does not match the bundle")

    def weights(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        if mode == "topology":
            return np.ones(len(first), dtype=np.float64)
        result = np.sum(value[first] * value[second], axis=1) / dimension
        return np.abs(result) if mode == "unsigned" else result

    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    data: list[np.ndarray] = []

    def add(first: np.ndarray, second: np.ndarray) -> None:
        first = np.asarray(first, dtype=np.int32).ravel()
        second = np.asarray(second, dtype=np.int32).ravel()
        measured = weights(first, second)
        finite = np.isfinite(measured) & (measured != 0.0)
        first, second, measured = first[finite], second[finite], measured[finite]
        rows.append(first)
        columns.append(second)
        data.append(measured)
        off_diagonal = first != second
        rows.append(second[off_diagonal])
        columns.append(first[off_diagonal])
        data.append(measured[off_diagonal])

    region = np.asarray(incidence["region"], dtype=np.int32)
    order = np.argsort(region, kind="stable")
    ordered_region = region[order]
    starts = np.flatnonzero(np.r_[
        True, ordered_region[1:] != ordered_region[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        member = order[start:end]
        # The complete region fibre is a Gram block, including its measured
        # diagonal.  No arbitrary nearest-neighbour count is introduced.
        first_index, second_index = np.triu_indices(len(member))
        first = member[first_index]
        second = member[second_index]
        add(first, second)

    lookup = {
        (int(arc), int(region_id)): identifier
        for identifier, (arc, region_id) in enumerate(zip(
            incidence["arc"], incidence["region"]))
    }
    if include_arc_crossing:
        by_arc: dict[int, list[int]] = {}
        for identifier, arc in enumerate(incidence["arc"]):
            by_arc.setdefault(int(arc), []).append(identifier)
        pair = [members for members in by_arc.values() if len(members) == 2]
        if pair:
            add(
                np.asarray([item[0] for item in pair], dtype=np.int32),
                np.asarray([item[1] for item in pair], dtype=np.int32),
            )
    if include_junctions:
        first_record = []
        second_record = []
        for first_arc, second_arc, shared in zip(
            continuation["first_arc"],
            continuation["second_arc"],
            continuation["region"],
        ):
            first_record.append(lookup[(int(first_arc), int(shared))])
            second_record.append(lookup[(int(second_arc), int(shared))])
        if first_record:
            add(np.asarray(first_record), np.asarray(second_record))

    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    column = np.concatenate(columns) if columns else np.empty(0, dtype=np.int32)
    measured = np.concatenate(data) if data else np.empty(0, dtype=np.float64)
    connection = sparse.coo_matrix(
        (measured, (row, column)), shape=(count, count)).tocsr()
    connection.sum_duplicates()
    absolute_degree = np.asarray(np.abs(connection).sum(axis=1)).ravel()
    inverse = np.divide(
        1.0,
        np.sqrt(absolute_degree),
        out=np.zeros_like(absolute_degree),
        where=absolute_degree > 0.0,
    )
    normalized = sparse.diags(inverse) @ connection @ sparse.diags(inverse)
    return normalized.tocsr(), absolute_degree, {
        "incidences": count,
        "connection_nonzeros": int(connection.nnz),
        "isolated_incidences": int(np.count_nonzero(absolute_degree == 0.0)),
        "mode": mode,
    }


def region_source_matrix(
    incidence_region: np.ndarray,
    regions: np.ndarray,
) -> sparse.csc_matrix:
    """Give each requested region one unit-norm, uniform incidence source."""
    labels = np.asarray(incidence_region, dtype=np.int32)
    selected = np.asarray(regions, dtype=np.int32)
    row = []
    column = []
    data = []
    for source, region in enumerate(selected):
        member = np.flatnonzero(labels == region)
        if not len(member):
            continue
        row.extend(member.tolist())
        column.extend([source] * len(member))
        data.extend([1.0 / np.sqrt(len(member))] * len(member))
    return sparse.csc_matrix(
        (data, (row, column)), shape=(len(labels), len(selected)))


def connection_heat_gram(
    normalized_connection: sparse.csr_matrix,
    sources: sparse.csc_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact queried Gram of the all-source unit-time heat kernel."""
    if normalized_connection.shape[0] != sources.shape[0]:
        raise ValueError("connection and source state spaces disagree")
    source = sources.toarray()
    half_flow = np.exp(-0.5) * expm_multiply(
        0.5 * normalized_connection, source)
    gram = half_flow.T @ half_flow
    diagonal = np.maximum(np.diag(gram), 0.0)
    denominator = np.sqrt(diagonal[:, None] * diagonal[None, :])
    similarity = np.divide(
        gram,
        denominator,
        out=np.zeros_like(gram),
        where=denominator > 1e-30,
    )
    return np.ascontiguousarray(gram), np.ascontiguousarray(similarity)


def connection_heat_response(
    normalized_connection: sparse.csr_matrix,
    source: sparse.csc_matrix,
) -> np.ndarray:
    """Query participatory amplitude without changing the all-source kernel."""
    if source.shape[1] != 1:
        raise ValueError("response query expects one source column")
    response = np.exp(-1.0) * expm_multiply(
        normalized_connection, source.toarray())
    return np.ascontiguousarray(response[:, 0])


def connection_green_gram(
    normalized_connection: sparse.csr_matrix,
    sources: sparse.csc_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | bool | int]]:
    """Query the all-scale Green participation kernel by one direct solve.

    The operator is the Moore--Penrose inverse of the signed normalized
    Laplacian ``L = I-S``.  If a genuine numerical null mode exists, sources
    are projected off it and one coordinate is grounded only to choose a
    gauge; quadratic forms on the projected sources are gauge invariant.
    There is no diffusion time, damping factor, or path-length cutoff.
    """
    if normalized_connection.shape[0] != sources.shape[0]:
        raise ValueError("connection and source state spaces disagree")
    size = normalized_connection.shape[0]
    symmetric = 0.5 * (normalized_connection + normalized_connection.T)
    largest_value, largest_vector = eigsh(
        symmetric, k=1, which="LA", return_eigenvectors=True)
    maximum = float(largest_value[0])
    null_tolerance = (
        64.0 * np.finfo(np.float64).eps * max(size, 1))
    has_null = abs(1.0 - maximum) <= null_tolerance
    source = sources.toarray()
    null_projection_norm = 0.0
    if has_null:
        null = largest_vector[:, 0]
        null /= max(float(np.linalg.norm(null)), 1e-30)
        projection = null @ source
        null_projection_norm = float(np.linalg.norm(projection))
        source = source - null[:, None] * projection[None, :]

    laplacian = sparse.eye(size, format="csr") - symmetric
    if has_null:
        # Ground the most visible coordinate for numerical conditioning.  The
        # projected right-hand sides make every resulting energy gauge-free.
        ground = int(np.argmax(np.abs(largest_vector[:, 0])))
        keep = np.ones(size, dtype=bool)
        keep[ground] = False
        reduced = laplacian[keep][:, keep].tocsc()
        factor = splu(reduced)
        solution = np.zeros_like(source)
        solution[keep] = factor.solve(source[keep])
    else:
        ground = -1
        factor = splu(laplacian.tocsc())
        solution = factor.solve(source)

    gram = source.T @ solution
    gram = 0.5 * (gram + gram.T)
    diagonal = np.maximum(np.diag(gram), 0.0)
    denominator = np.sqrt(diagonal[:, None] * diagonal[None, :])
    similarity = np.divide(
        gram,
        denominator,
        out=np.zeros_like(gram),
        where=denominator > 1e-30,
    )
    resistance = np.maximum(
        diagonal[:, None] + diagonal[None, :] - 2.0 * gram,
        0.0,
    )
    return (
        np.ascontiguousarray(gram),
        np.ascontiguousarray(similarity),
        np.ascontiguousarray(resistance),
        {
            "largest_connection_eigenvalue": maximum,
            "null_tolerance": null_tolerance,
            "projected_null_mode": bool(has_null),
            "grounded_coordinate": ground,
            "source_null_projection_norm": null_projection_norm,
        },
    )
