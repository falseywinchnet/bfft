#!/usr/bin/env python3
"""Circle-valued closure transport with warm multi-source consensus.

For one Fourier phase chart ``z[k] = exp(i phi[k])``, a normalized
bispectrum sample on the triad ``e = (k, s, k+s)`` obeys

    beta[e] = z[k] z[s] conj(z[k+s]).

The triads are the analogue of spatial gradient edges in Split Bregman.
Their wrapped closure defects are persistent dual/transport state, and the
three adjoint messages from every triad are the analogue of a divergence.
Independent source charts take one local sweep at a time and are coupled to
one shared phase chart through a second circular Bregman constraint.

No clean image is consulted.  Translation is the exact two-dimensional
linear-phase nullspace of the closure operator.  Each sweep synchronizes that
gauge by a convolution over all supported Fourier bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class ClosureTransportConfig:
    sweeps: int = 24
    consensus_weight: float = 0.35
    data_penalty: float = 1.0
    relaxation: float = 0.65
    minimum_edge_weight: float = 1e-5
    minimum_sources: int = 2
    publication_coherence: float = 0.70
    publication_residual: float = 0.70


@dataclass
class SourceTransportState:
    phase: np.ndarray
    closure_dual: np.ndarray
    closure_slack: np.ndarray
    consensus_dual: np.ndarray
    active: np.ndarray
    node_weight: np.ndarray
    gauge_shift_yx: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class ClosureEdges:
    first: np.ndarray
    second: np.ndarray
    total: np.ndarray
    measurement: np.ndarray
    weight: np.ndarray

    @property
    def count(self) -> int:
        return int(len(self.weight))


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex128)
    return values / np.maximum(np.abs(values), 1e-30)


def _hermitian_phase(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    opposite = np.conj(values[
        (-np.arange(height)) % height][:, (-np.arange(width)) % width])
    result = _unit(values + opposite)
    result[0, 0] = 1.0 + 0.0j
    return result


def _soft_threshold(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    return np.sign(values) * np.maximum(magnitude - threshold, 0.0)


def build_closure_edges(
    bispectrum: np.ndarray,
    coherence: np.ndarray,
    steps: list[tuple[int, int]],
    support: np.ndarray,
    minimum_weight: float = 1e-5,
) -> ClosureEdges:
    """Build the sparse closure hypergraph admitted by one source."""
    if bispectrum.shape != coherence.shape:
        raise ValueError("bispectrum and coherence shapes must agree")
    if bispectrum.shape[0] != len(steps):
        raise ValueError("bispectrum and step counts differ")
    height, width = bispectrum.shape[-2:]
    if support.shape != (height, width):
        raise ValueError("support must match the Fourier grid")

    active = support > 0.0
    ky, kx = np.nonzero(active)
    first: list[int] = []
    second: list[int] = []
    total: list[int] = []
    measurement: list[complex] = []
    weight: list[float] = []
    for step_index, (sy, sx) in enumerate(steps):
        if not active[sy, sx]:
            continue
        ty = (ky + sy) % height
        tx = (kx + sx) % width
        admitted = active[ty, tx] & ~((ky == 0) & (kx == 0))
        if not np.any(admitted):
            continue
        ay = ky[admitted]
        ax = kx[admitted]
        cy = ty[admitted]
        cx = tx[admitted]
        edge_weight = (
            np.clip(coherence[step_index, ay, ax], 0.0, 1.0) ** 3
            * support[ay, ax]
            * float(support[sy, sx])
            * support[cy, cx]
        )
        nonzero = edge_weight >= minimum_weight
        for y, x, yy, xx, value, confidence in zip(
            ay[nonzero],
            ax[nonzero],
            cy[nonzero],
            cx[nonzero],
            bispectrum[step_index, ay[nonzero], ax[nonzero]],
            edge_weight[nonzero],
        ):
            first.append(int(y * width + x))
            second.append(int(sy * width + sx))
            total.append(int(yy * width + xx))
            measurement.append(complex(value))
            weight.append(float(confidence))
    return ClosureEdges(
        first=np.asarray(first, dtype=np.int32),
        second=np.asarray(second, dtype=np.int32),
        total=np.asarray(total, dtype=np.int32),
        measurement=_unit(np.asarray(measurement, dtype=np.complex128)),
        weight=np.asarray(weight, dtype=np.float64),
    )


def _anchored_unit_graph(
    bispectrum: np.ndarray,
    coherence: np.ndarray,
    steps: list[tuple[int, int]],
    support: np.ndarray,
    minimum_weight: float,
) -> tuple[coo_matrix, np.ndarray, dict[int, int]]:
    """Connection Laplacian from the two translation-gauge unit steps."""
    height, width = support.shape
    flat_active = np.flatnonzero(support.ravel() > 0.0)
    node = {int(value): index for index, value in enumerate(flat_active)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    diagonal = np.zeros(len(flat_active), dtype=np.float64)
    unit_indices = [
        index
        for index, step in enumerate(steps)
        if step in ((0, 1), (1, 0))
    ]
    for step_index in unit_indices:
        sy, sx = steps[step_index]
        for flat in flat_active:
            y, x = divmod(int(flat), width)
            if y == 0 and x == 0:
                continue
            target = ((y + sy) % height) * width + (x + sx) % width
            if target not in node:
                continue
            confidence = float(
                np.clip(coherence[step_index, y, x], 0.0, 1.0) ** 3
                * support[y, x]
                * support[target // width, target % width]
            )
            if confidence < minimum_weight:
                continue
            relation = np.conj(_unit(
                np.asarray([bispectrum[step_index, y, x]]))[0])
            first = node[int(flat)]
            second = node[int(target)]
            diagonal[first] += confidence
            diagonal[second] += confidence
            rows.extend((first, second))
            columns.extend((second, first))
            values.extend(
                (-confidence * np.conj(relation), -confidence * relation))
    rows.extend(range(len(flat_active)))
    columns.extend(range(len(flat_active)))
    values.extend(diagonal.astype(np.complex128))
    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(flat_active), len(flat_active)),
        dtype=np.complex128,
    )
    return matrix, flat_active, node


def connection_phase_seed(
    bispectrum: np.ndarray,
    coherence: np.ndarray,
    steps: list[tuple[int, int]],
    support: np.ndarray,
    minimum_weight: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Nucleate a chart by simultaneous unit-step angular synchronization."""
    height, width = support.shape
    laplacian, flat_active, node = _anchored_unit_graph(
        bispectrum, coherence, steps, support, minimum_weight)
    if not len(flat_active):
        raise ValueError("source has no active closure support")

    adjacency = laplacian.copy().tocsr()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    component_count, labels = connected_components(
        abs(adjacency), directed=False, return_labels=True)
    anchors = [
        value
        for value in (1, width)
        if value in node
    ]
    if not anchors:
        raise ValueError("support does not contain a unit-frequency nucleus")
    anchor_component = int(labels[node[anchors[0]]])
    admitted_local = np.flatnonzero(labels == anchor_component)
    admitted_flat = flat_active[admitted_local]
    submatrix = laplacian.tocsr()[admitted_local][:, admitted_local]

    if len(admitted_local) == 1:
        vector = np.ones(1, dtype=np.complex128)
        eigenvalue = 0.0
    elif len(admitted_local) <= 4:
        eigenvalues, eigenvectors = np.linalg.eigh(submatrix.toarray())
        selected = int(np.argmin(eigenvalues))
        eigenvalue = float(eigenvalues[selected])
        vector = _unit(eigenvectors[:, selected])
    else:
        # Complex ARPACK can return the first positive mode for k=1 even
        # when the connection Laplacian has an exact zero mode.  Request a
        # tiny cluster and select its algebraically smallest member.
        eigen_count = min(3, len(admitted_local) - 1)
        eigenvalues, eigenvectors = eigsh(
            submatrix, k=eigen_count, which="SA", tol=1e-8)
        selected = int(np.argmin(eigenvalues))
        eigenvalue = float(eigenvalues[selected])
        vector = _unit(eigenvectors[:, selected])
    local_index = {
        int(flat): index for index, flat in enumerate(admitted_flat)}
    anchor = next(value for value in anchors if value in local_index)
    vector *= np.conj(vector[local_index[anchor]])

    phase = np.ones((height, width), dtype=np.complex128)
    active = np.zeros((height, width), dtype=bool)
    phase.ravel()[admitted_flat] = vector
    active.ravel()[admitted_flat] = True
    phase = _hermitian_phase(phase)
    active |= active[
        (-np.arange(height)) % height][:, (-np.arange(width)) % width]
    active[0, 0] = True
    phase[0, 0] = 1.0 + 0.0j
    return phase, active, {
        "unit_graph_components": int(component_count),
        "nucleated_frequencies": int(np.count_nonzero(active)),
        "connection_eigenvalue": eigenvalue,
    }


def synchronize_translation_gauge(
    source: np.ndarray,
    reference: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Align one phase chart by convolution over its translation nullspace."""
    if source.shape != reference.shape or weight.shape != source.shape:
        raise ValueError("phase charts and gauge weights must agree")
    height, width = source.shape
    cross = weight * source * np.conj(reference)
    correlation = np.fft.ifft2(cross).real
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy = int(peak[0] if peak[0] <= height // 2 else peak[0] - height)
    dx = int(peak[1] if peak[1] <= width // 2 else peak[1] - width)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    ramp = np.exp(2j * np.pi * (fy * dy + fx * dx))
    return _hermitian_phase(source * ramp), (dy, dx)


def _closure_residual(
    phase: np.ndarray,
    edges: ClosureEdges,
) -> np.ndarray:
    flat = phase.ravel()
    predicted = (
        flat[edges.first]
        * flat[edges.second]
        * np.conj(flat[edges.total])
    )
    return np.angle(predicted * np.conj(edges.measurement))


def _node_weights(
    edges: ClosureEdges,
    shape: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(shape[0] * shape[1], dtype=np.float64)
    for index in (edges.first, edges.second, edges.total):
        np.add.at(result, index, edges.weight)
    positive = result[result > 0.0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    return (result / max(scale, 1e-12)).reshape(shape)


def _source_sweep(
    state: SourceTransportState,
    edges: ClosureEdges,
    consensus: np.ndarray,
    config: ClosureTransportConfig,
) -> None:
    residual = _closure_residual(state.phase, edges)
    transported = np.angle(np.exp(1j * (
        residual + state.closure_dual)))
    threshold = (
        config.data_penalty
        * edges.weight
        / max(float(np.median(edges.weight)), 1e-12)
    )
    state.closure_slack = _soft_threshold(transported, threshold)
    state.closure_dual = np.angle(np.exp(1j * (
        state.closure_dual + residual - state.closure_slack)))
    target = edges.measurement * np.exp(
        1j * (state.closure_slack - state.closure_dual))

    flat = state.phase.ravel()
    messages = np.zeros_like(flat)
    denominator = np.zeros(flat.shape, dtype=np.float64)
    first_value = flat[edges.first]
    second_value = flat[edges.second]
    total_value = flat[edges.total]
    first_message = target * np.conj(second_value) * total_value
    second_message = target * np.conj(first_value) * total_value
    total_message = np.conj(target) * first_value * second_value
    for index, value in (
        (edges.first, first_message),
        (edges.second, second_message),
        (edges.total, total_message),
    ):
        np.add.at(messages, index, edges.weight * value)
        np.add.at(denominator, index, edges.weight)

    consensus_target = consensus * np.exp(-1j * state.consensus_dual)
    consensus_weight = (
        config.consensus_weight * state.node_weight.ravel())
    messages += consensus_weight * consensus_target.ravel()
    denominator += consensus_weight
    candidate = _unit(messages).reshape(state.phase.shape)
    movable = state.active & (denominator.reshape(state.phase.shape) > 0.0)
    blended = (
        (1.0 - config.relaxation) * state.phase[movable]
        + config.relaxation * candidate[movable]
    )
    state.phase[movable] = _unit(blended)
    state.phase = _hermitian_phase(state.phase)


def solve_closure_transport_consensus(
    bispectra: np.ndarray,
    coherences: np.ndarray,
    steps: list[tuple[int, int]],
    source_support: np.ndarray,
    config: ClosureTransportConfig = ClosureTransportConfig(),
    initial_phases: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, list[SourceTransportState]]:
    """Interleave local closure transport and shared gauge consensus."""
    if bispectra.ndim != 4:
        raise ValueError("expected [source, step, y, x] bispectra")
    if bispectra.shape != coherences.shape:
        raise ValueError("bispectra and coherences must agree")
    input_source_count, _, height, width = bispectra.shape
    if source_support.shape != (input_source_count, height, width):
        raise ValueError("source support must match source Fourier grids")
    if input_source_count < 1:
        raise ValueError("at least one source is required")

    edges: list[ClosureEdges] = []
    states: list[SourceTransportState] = []
    retained_support = []
    retained_sources = []
    seed_info = []
    for source in range(input_source_count):
        source_edges = build_closure_edges(
            bispectra[source],
            coherences[source],
            steps,
            source_support[source],
            config.minimum_edge_weight,
        )
        if source_edges.count == 0:
            continue
        edges.append(source_edges)
        retained_support.append(source_support[source])
        retained_sources.append(source)
        if initial_phases is None:
            phase, active, info = connection_phase_seed(
                bispectra[source],
                coherences[source],
                steps,
                source_support[source],
                config.minimum_edge_weight,
            )
        else:
            phase = _hermitian_phase(initial_phases[source])
            active = source_support[source] > 0.0
            info = {
                "unit_graph_components": 0,
                "nucleated_frequencies": int(np.count_nonzero(active)),
                "connection_eigenvalue": 0.0,
            }
        seed_info.append(info)
        states.append(SourceTransportState(
            phase=phase,
            closure_dual=np.zeros(source_edges.count, dtype=np.float64),
            closure_slack=np.zeros(source_edges.count, dtype=np.float64),
            consensus_dual=np.zeros((height, width), dtype=np.float64),
            active=active,
            node_weight=_node_weights(source_edges, (height, width)),
        ))

    if not states:
        phase = np.ones((height, width), dtype=np.complex128)
        publication = np.zeros((height, width), dtype=np.float64)
        publication[0, 0] = 1.0
        return phase, publication, {
            "operator": "warm_closure_bregman_transport_consensus",
            "sweeps": 0,
            "retained_sources": [],
            "source_edges": [],
            "source_seed": [],
            "source_gauge_shift_yx": [],
            "published_frequencies": 1,
            "mean_publication_gain": 1.0,
            "trace": [],
            "blocked": "no source formed a connected closure nucleus",
        }, []

    source_count = len(states)
    retained_support_array = np.asarray(retained_support)
    consensus = states[0].phase.copy()
    trace = []
    for sweep in range(config.sweeps):
        gauge_shifts = []
        for state in states:
            gauge_weight = (
                state.node_weight
                * state.active
                * np.maximum(np.abs(consensus), 1e-12)
            )
            state.phase, shift = synchronize_translation_gauge(
                state.phase, consensus, gauge_weight)
            state.gauge_shift_yx = (
                state.gauge_shift_yx[0] + shift[0],
                state.gauge_shift_yx[1] + shift[1],
            )
            gauge_shifts.append(shift)
        for state, source_edges in zip(states, edges):
            _source_sweep(state, source_edges, consensus, config)

        message = np.zeros((height, width), dtype=np.complex128)
        denominator = np.zeros((height, width), dtype=np.float64)
        for state in states:
            weight = state.node_weight * state.active
            message += (
                weight * state.phase
                * np.exp(1j * state.consensus_dual)
            )
            denominator += weight
        candidate = _unit(message)
        admitted = denominator > 0.0
        blended = (
            (1.0 - config.relaxation) * consensus[admitted]
            + config.relaxation * candidate[admitted]
        )
        consensus[admitted] = _unit(blended)
        consensus = _hermitian_phase(consensus)
        # Fix only the common, unobservable translation gauge.
        consensus[0, 1] = 1.0 + 0.0j
        consensus[0, -1] = 1.0 + 0.0j
        consensus[1, 0] = 1.0 + 0.0j
        consensus[-1, 0] = 1.0 + 0.0j

        closure_error = []
        consensus_error = []
        for state, source_edges in zip(states, edges):
            difference = np.angle(state.phase * np.conj(consensus))
            state.consensus_dual = np.angle(np.exp(1j * (
                state.consensus_dual + difference)))
            residual = _closure_residual(state.phase, source_edges)
            closure_error.append(float(np.average(
                np.abs(residual), weights=source_edges.weight)))
            selected = state.active & (state.node_weight > 0.0)
            consensus_error.append(float(np.average(
                np.abs(difference[selected]),
                weights=state.node_weight[selected],
            )))
        trace.append({
            "sweep": sweep + 1,
            "closure_residual": float(np.mean(closure_error)),
            "consensus_residual": float(np.mean(consensus_error)),
            "gauge_motion": int(sum(
                abs(y) + abs(x) for y, x in gauge_shifts)),
        })

    source_weight = np.asarray([
        state.node_weight * state.active for state in states])
    source_phase = np.asarray([state.phase for state in states])
    total_weight = np.sum(source_weight, axis=0)
    agreement = (
        np.abs(np.sum(source_weight * source_phase, axis=0))
        / np.maximum(total_weight, 1e-12)
    )
    source_presence = np.sum(source_weight > 0.0, axis=0)
    residual_sum = np.zeros((height, width), dtype=np.float64)
    residual_weight = np.zeros((height, width), dtype=np.float64)
    for state, source_edges in zip(states, edges):
        residual = np.abs(_closure_residual(state.phase, source_edges))
        for index in (
            source_edges.first,
            source_edges.second,
            source_edges.total,
        ):
            np.add.at(
                residual_sum.ravel(),
                index,
                source_edges.weight * residual,
            )
            np.add.at(
                residual_weight.ravel(), index, source_edges.weight)
    node_residual = residual_sum / np.maximum(residual_weight, 1e-12)
    residual_score = np.exp(-node_residual / max(
        config.publication_residual, 1e-12))
    coherence_gain = np.clip(
        (agreement - config.publication_coherence)
        / max(1.0 - config.publication_coherence, 1e-12),
        0.0,
        1.0,
    )
    availability = np.mean(
        np.clip(retained_support_array, 0.0, 1.0), axis=0)
    # Source evidence decides whether a node exists in the closure complex.
    # It is not an amplitude multiplier once the node is admitted.  The
    # expected phase phasor is attenuated only by measured cross-source
    # coherence and incident closure residual.
    publication = coherence_gain * residual_score
    publication[availability <= 0.0] = 0.0
    publication[source_presence < min(
        config.minimum_sources, source_count)] = 0.0
    publication[0, 0] = 1.0

    return consensus, publication, {
        "operator": "warm_closure_bregman_transport_consensus",
        "sweeps": config.sweeps,
        "retained_sources": retained_sources,
        "source_edges": [value.count for value in edges],
        "source_seed": seed_info,
        "source_gauge_shift_yx": [
            list(state.gauge_shift_yx) for state in states
        ],
        "published_frequencies": int(np.count_nonzero(publication > 0.0)),
        "mean_publication_gain": float(np.mean(
            publication[publication > 0.0])),
        "trace": trace,
    }, states
