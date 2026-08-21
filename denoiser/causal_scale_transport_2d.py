"""Continuous coarse-to-fine transport filtration for 2-D scene evidence.

This is a scale experiment, not a promoted denoiser.  It separates two acts
that conventional multiscale denoisers usually confound:

1. decompose the complete observation without deleting anything;
2. transport evidence from the coarse end of that decomposition toward its
   progressively finer generations.

The decomposition is the continuous isotropic Selling heat orbit

    u(t) = exp(-t L_0) y,

where ``L_0`` is the unit-metric Selling Laplacian.  Its reversed increments
are an exact telescoping recomposition of ``y``.  The returned finite times are
only dyadic samples of that continuum and select no physical image band.

At each reversed generation, a metric made from the scene already composed
and the still unresolved second moment transports confidence.  Reciprocal
parity charts provide a signed phase birth measure, while a positive tensor
overlap measures inheritance from the coarser scene.  Directional and
isotropic tensor actions remain separate.  Consequently a fine component is
never called noise merely because it is fine, and no component is removed
until the complete decomposition has been constructed.  Nested refinement has
falsified the present per-generation confidence recurrence as a continuum
posterior.  It is retained specifically to measure the missing scale-density
law; only the semigroup decomposition is currently representation-invariant.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.fft import dctn, idctn
from scipy.sparse import linalg as sparse_linalg

from .compressed_eikonal_observer_2d import (
    _COMPLETE_PARITY_COVECTORS,
    interlaced_scene_views_2d,
)
from .continual_eikonal_noise_transport_2d import (
    _continual_flux_laplacian,
    _shift_symmetric,
    continual_transport_metric,
)
from .witnessed_characteristic_transport_2d import _validate


def _isotropic_selling_spectrum(shape: tuple[int, int]) -> np.ndarray:
    """Eigenvalues of the unit-metric Selling Laplacian with reflection.

    Selling reduction of the identity metric emits the horizontal and
    vertical nearest-neighbour fluxes with coefficient ``1/4``.  The
    half-sample symmetric graph boundary is diagonalized exactly by DCT-II.
    """
    height, width = shape
    row = 0.5 * (1.0 - np.cos(np.pi * np.arange(height) / height))
    column = 0.5 * (1.0 - np.cos(np.pi * np.arange(width) / width))
    return row[:, None] + column[None, :]


def _heat_state(
    coefficient: np.ndarray,
    spectrum: np.ndarray,
    time: float,
) -> np.ndarray:
    """Evaluate one exact unit-Selling heat-semigroup state."""
    multiplier = np.exp(-float(time) * spectrum)
    if coefficient.ndim == 2:
        return idctn(multiplier * coefficient, type=2, norm="ortho")
    return np.stack([
        idctn(multiplier * member, type=2, norm="ortho")
        for member in coefficient
    ])


def _transport_times(
    spectrum: np.ndarray,
    refinement: int = 0,
) -> np.ndarray:
    """Return a nested numerical trace from identity to roundoff equilibrium."""
    refinement = int(refinement)
    if refinement < 0:
        raise ValueError("scale-trace refinement must be nonnegative")
    positive = spectrum[spectrum > 0.0]
    if not positive.size:
        return np.asarray((0.0,), dtype=np.float64)
    fastest_time = 1.0 / float(np.max(positive))
    equilibrium_time = (
        -np.log(np.finfo(float).eps) / float(np.min(positive)))
    times = [0.0]
    time = fastest_time
    while time < equilibrium_time:
        times.append(time)
        time *= 2.0
        if len(times) > 64:
            raise RuntimeError("dyadic semigroup trace failed to reach equilibrium")
    times.append(equilibrium_time)
    trace = np.asarray(times, dtype=np.float64)
    for _ in range(refinement):
        midpoint = 0.5 * (trace[:-1] + trace[1:])
        refined = np.empty(trace.size + midpoint.size, dtype=np.float64)
        refined[0::2] = trace
        refined[1::2] = midpoint
        trace = refined
    return trace


def _heat_tensor(
    field: np.ndarray,
    spectrum: np.ndarray,
    transport_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transport the complete first-jet second moment at its own scale."""
    gx = 0.5 * (
        _shift_symmetric(field, 0, 1)
        - _shift_symmetric(field, 0, -1))
    gy = 0.5 * (
        _shift_symmetric(field, 1, 0)
        - _shift_symmetric(field, -1, 0))
    members = np.stack((gx * gx, gx * gy, gy * gy))
    coefficient = np.stack([
        dctn(member, type=2, norm="ortho") for member in members
    ])
    transported = _heat_state(coefficient, spectrum, transport_time)
    return (
        np.maximum(transported[0], 0.0),
        transported[1],
        np.maximum(transported[2], 0.0),
    )


def _tensor_coordinates(
    tensor: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, np.ndarray | float]:
    """Split a PSD 2x2 field into directional and isotropic action."""
    xx, xy, yy = tensor
    trace = np.maximum(xx + yy, 0.0)
    gap = np.minimum(np.hypot(xx - yy, 2.0 * xy), trace)
    directional = np.maximum(gap, 0.0)
    isotropic = np.maximum(trace - gap, 0.0)
    total = float(np.sum(trace))
    directional_fraction = (
        float(np.sum(directional)) / total
        if total > np.finfo(float).tiny else 0.0)
    energy_square = float(np.sum(trace * trace))
    participation = (
        total * total / (trace.size * energy_square)
        if energy_square > np.finfo(float).tiny else 0.0)
    return {
        "trace": trace,
        "directional_action": directional,
        "isotropic_action": isotropic,
        "directional_fraction": float(np.clip(
            directional_fraction, 0.0, 1.0)),
        "isotropic_fraction": float(np.clip(
            1.0 - directional_fraction, 0.0, 1.0)),
        "spatial_participation": float(np.clip(participation, 0.0, 1.0)),
        "total_tensor_action": total,
    }


def _tensor_lineage_overlap(
    parent: tuple[np.ndarray, np.ndarray, np.ndarray],
    child: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Positive, scale-balanced overlap of two transported PSD jet laws."""
    pxx, pxy, pyy = parent
    cxx, cxy, cyy = child
    parent_norm = np.sqrt(pxx * pxx + 2.0 * pxy * pxy + pyy * pyy)
    child_norm = np.sqrt(cxx * cxx + 2.0 * cxy * cxy + cyy * cyy)
    product_norm = parent_norm * child_norm
    frobenius_cosine = np.divide(
        pxx * cxx + 2.0 * pxy * cxy + pyy * cyy,
        product_norm,
        out=np.zeros_like(parent_norm),
        where=product_norm > np.finfo(float).tiny,
    )
    balance = np.divide(
        2.0 * np.sqrt(product_norm),
        parent_norm + child_norm,
        out=np.zeros_like(parent_norm),
        where=(parent_norm + child_norm) > np.finfo(float).tiny,
    )
    return np.clip(frobenius_cosine * balance, 0.0, 1.0)


def _screened_transport(
    laplacian: sparse.csr_matrix,
    scale_time: float,
    fields: np.ndarray,
) -> np.ndarray:
    """Positive implicit transport over exactly one scale interval."""
    values = np.asarray(fields, dtype=np.float64)
    leading_shape = values.shape[:-2]
    height, width = values.shape[-2:]
    matrix = (
        sparse.eye(height * width, format="csr")
        + float(scale_time) * laplacian
    )
    rhs = values.reshape((-1, height * width)).T
    solution = np.asarray(sparse_linalg.spsolve(matrix, rhs))
    if solution.ndim == 1:
        solution = solution[:, None]
    return solution.T.reshape(leading_shape + (height, width))


def _scale_phase_birth(
    chart_components: np.ndarray,
    laplacian: sparse.csr_matrix,
    scale_time: float,
    *,
    hadamard_transport_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Transport phase and Hadamard pull of reciprocal chart moments.

    For a reciprocal pair ``(a,b)``, positive transport of its outer product
    gives a PSD matrix ``C``.  Hadamard's determinant inequality supplies

    ``pull = 1 - det(C)/(C00*C11) = C01^2/(C00*C11)``.

    The two eigenvalues of the normalized correlation matrix are
    ``1 +/- sqrt(pull)``.  Thus pull is one when transport collapses the pair
    onto one supported direction and zero when it retains independent volume.
    Unlike the phase coordinate ``2*C01/(C00+C11)``, it does not penalize a
    coherent relation merely because the two chart amplitudes are unequal.
    """
    numerator = []
    first_action = []
    second_action = []
    for pair in range(len(_COMPLETE_PARITY_COVECTORS)):
        first = chart_components[2 * pair]
        second = chart_components[2 * pair + 1]
        numerator.append(first * second)
        first_action.append(first * first)
        second_action.append(second * second)
    statistics = np.concatenate((
        np.stack(numerator),
        np.stack(first_action),
        np.stack(second_action),
    ), axis=0)
    phase_transported = _screened_transport(
        laplacian, scale_time, statistics)
    hadamard_time = (
        float(scale_time)
        if hadamard_transport_time is None
        else float(hadamard_transport_time)
    )
    hadamard_transported = (
        phase_transported
        if hadamard_time == float(scale_time)
        else _screened_transport(laplacian, hadamard_time, statistics)
    )
    count = len(_COMPLETE_PARITY_COVECTORS)
    phase_cross = phase_transported[:count]
    phase_first = np.maximum(
        phase_transported[count:2 * count], 0.0)
    phase_second = np.maximum(
        phase_transported[2 * count:], 0.0)
    transported_denominator = phase_first + phase_second
    phase = np.divide(
        2.0 * phase_cross,
        transported_denominator,
        out=np.zeros_like(phase_cross),
        where=transported_denominator > np.finfo(float).tiny,
    )
    phase = np.clip(phase, -1.0, 1.0)
    certainty = phase * phase
    union = 1.0 - np.prod(1.0 - certainty, axis=0)
    hadamard_cross = hadamard_transported[:count]
    hadamard_first = np.maximum(
        hadamard_transported[count:2 * count], 0.0)
    hadamard_second = np.maximum(
        hadamard_transported[2 * count:], 0.0)
    hadamard_denominator = hadamard_first * hadamard_second
    hadamard_pull = np.divide(
        hadamard_cross * hadamard_cross,
        hadamard_denominator,
        out=np.zeros_like(hadamard_cross),
        where=hadamard_denominator > np.finfo(float).tiny,
    )
    hadamard_pull = np.clip(hadamard_pull, 0.0, 1.0)
    spectral_phase_denominator = hadamard_first + hadamard_second
    spectral_phase = np.divide(
        2.0 * hadamard_cross,
        spectral_phase_denominator,
        out=np.zeros_like(hadamard_cross),
        where=spectral_phase_denominator > np.finfo(float).tiny,
    )
    spectral_phase = np.clip(spectral_phase, -1.0, 1.0)
    spectral_phase_certainty = spectral_phase * spectral_phase
    spectral_phase_union = 1.0 - np.prod(
        1.0 - spectral_phase_certainty, axis=0)
    # The complete covector fibre is integrated with its normalized counting
    # measure.  A union would let one chance-correlated chart certify noise.
    mean_hadamard_pull = np.mean(hadamard_pull, axis=0)
    effective_rank = 2.0 / (1.0 + hadamard_pull)
    yy, xx = np.mgrid[
        :chart_components.shape[-2], :chart_components.shape[-1]]
    eigenprojected_charts = []
    spectral_gap_fraction = []
    for pair, (covector_y, covector_x) in enumerate(
        _COMPLETE_PARITY_COVECTORS
    ):
        first = chart_components[2 * pair]
        second = chart_components[2 * pair + 1]
        first_moment = hadamard_first[pair]
        second_moment = hadamard_second[pair]
        cross_moment = hadamard_cross[pair]
        trace = first_moment + second_moment
        gap = np.minimum(np.hypot(
            first_moment - second_moment,
            2.0 * cross_moment,
        ), trace)
        low = 0.5 * (trace - gap)
        k00 = np.divide(
            first_moment - low,
            trace,
            out=np.zeros_like(trace),
            where=trace > np.finfo(float).tiny,
        )
        k01 = np.divide(
            cross_moment,
            trace,
            out=np.zeros_like(trace),
            where=trace > np.finfo(float).tiny,
        )
        k11 = np.divide(
            second_moment - low,
            trace,
            out=np.zeros_like(trace),
            where=trace > np.finfo(float).tiny,
        )
        projected_first = k00 * first + k01 * second
        projected_second = k01 * first + k11 * second
        first_owner = (
            (covector_y * yy + covector_x * xx) & 1) == 0
        eigenprojected_charts.append(np.where(
            first_owner, projected_first, projected_second))
        spectral_gap_fraction.append(np.divide(
            gap,
            trace,
            out=np.zeros_like(trace),
            where=trace > np.finfo(float).tiny,
        ))
    eigenprojected_component = np.mean(
        np.stack(eigenprojected_charts), axis=0)
    spectral_gap_fraction_array = np.stack(spectral_gap_fraction)
    return (
        np.clip(union, 0.0, 1.0),
        np.clip(mean_hadamard_pull, 0.0, 1.0),
        np.clip(spectral_phase_union, 0.0, 1.0),
        eigenprojected_component,
        {
        "covector_signed_phase_mean": tuple(
            float(np.mean(member)) for member in phase),
        "covector_phase_certainty_mean": tuple(
            float(np.mean(member)) for member in certainty),
        "covector_hadamard_pull_mean": tuple(
            float(np.mean(member)) for member in hadamard_pull),
        "covector_spectral_phase_certainty_mean": tuple(
            float(np.mean(member)) for member in spectral_phase_certainty),
        "covector_effective_rank_mean": tuple(
            float(np.mean(member)) for member in effective_rank),
        "mean_phase_birth": float(np.mean(union)),
        "mean_hadamard_pull": float(np.mean(mean_hadamard_pull)),
        "mean_spectral_phase_birth": float(np.mean(spectral_phase_union)),
        "mean_hadamard_surviving_volume": float(
            np.mean(1.0 - mean_hadamard_pull)),
        "mean_hadamard_effective_rank": float(np.mean(
            2.0 / (1.0 + mean_hadamard_pull))),
        "mean_spectral_gap_fraction": float(np.mean(
            spectral_gap_fraction_array)),
        "hadamard_transport_time": hadamard_time,
        },
    )


def _selling_jet_hadamard_pull(
    chart_components: np.ndarray,
    laplacian: sparse.csr_matrix,
    transport_time: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure reciprocal support in the complete normalized Selling jet.

    Let ``Q=L/d`` be the graph generator normalized by its maximum degree and
    let ``Gamma_Q`` be its local carre-du-champ.  At every vertex, two chart
    fields ``a`` and ``b`` have the positive Sobolev-jet Gram law

    ``J(a,b) = a*b + Gamma_Q(a,b) + (Q*a)*(Q*b)``.

    The three summands are respectively value, first-flux, and curvature
    agreement.  They are not fitted weights: they are the consecutive
    differential actions of the dimensionless Selling generator.  Positive
    resolvent transport preserves the resulting 2x2 PSD cone, so its
    Hadamard pull is bounded in ``[0,1]`` without a noise threshold.
    """
    charts = np.asarray(chart_components, dtype=np.float64)
    height, width = charts.shape[-2:]
    flat = charts.reshape((charts.shape[0], height * width))
    degree = np.asarray(laplacian.diagonal(), dtype=np.float64)
    maximum_degree = float(np.max(degree)) if degree.size else 0.0
    if maximum_degree <= np.finfo(float).tiny:
        # With no transport geometry the differential lift reduces exactly
        # to the value Gram law.
        normalized_action = np.zeros_like(flat)
        gamma = np.zeros(
            (charts.shape[0], charts.shape[0], height * width),
            dtype=np.float64,
        )
    else:
        normalized_action = np.stack([
            np.asarray(laplacian @ member) / maximum_degree
            for member in flat
        ])
        # Gamma_Q(a,b)=1/(2d) sum_j w_ij(a_i-a_j)(b_i-b_j).
        # Computing it from the generator identity avoids choosing a raster
        # derivative and therefore follows the evolving Selling stencil.
        product_action = np.empty(
            (charts.shape[0], charts.shape[0], height * width),
            dtype=np.float64,
        )
        for first in range(charts.shape[0]):
            for second in range(first, charts.shape[0]):
                product = flat[first] * flat[second]
                action = np.asarray(laplacian @ product) / maximum_degree
                value = 0.5 * (
                    flat[first] * normalized_action[second]
                    + flat[second] * normalized_action[first]
                    - action
                )
                product_action[first, second] = value
                product_action[second, first] = value
        gamma = product_action

    cross = []
    first_action = []
    second_action = []
    count = len(_COMPLETE_PARITY_COVECTORS)
    for pair in range(count):
        first_index = 2 * pair
        second_index = first_index + 1
        first = flat[first_index]
        second = flat[second_index]
        cross.append(
            first * second
            + gamma[first_index, second_index]
            + normalized_action[first_index] * normalized_action[second_index]
        )
        first_action.append(
            first * first
            + gamma[first_index, first_index]
            + normalized_action[first_index] ** 2
        )
        second_action.append(
            second * second
            + gamma[second_index, second_index]
            + normalized_action[second_index] ** 2
        )
    statistics = np.concatenate((
        np.stack(cross),
        np.stack(first_action),
        np.stack(second_action),
    ), axis=0).reshape((3 * count, height, width))
    transported = _screened_transport(
        laplacian, float(transport_time), statistics)
    transported_cross = transported[:count]
    transported_first = np.maximum(transported[count:2 * count], 0.0)
    transported_second = np.maximum(transported[2 * count:], 0.0)
    denominator = transported_first * transported_second
    covector_pull = np.divide(
        transported_cross * transported_cross,
        denominator,
        out=np.zeros_like(transported_cross),
        where=denominator > np.finfo(float).tiny,
    )
    covector_pull = np.clip(covector_pull, 0.0, 1.0)
    pull = np.mean(covector_pull, axis=0)
    return np.clip(pull, 0.0, 1.0), {
        "selling_jet_maximum_degree": maximum_degree,
        "covector_selling_jet_pull_mean": tuple(
            float(np.mean(member)) for member in covector_pull),
        "mean_selling_jet_pull": float(np.mean(pull)),
        "mean_selling_jet_surviving_volume": float(np.mean(1.0 - pull)),
    }


def _krylov_spectral_purity(
    component: np.ndarray,
    laplacian: sparse.csr_matrix,
    transport_time: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Measure spectral purity of the three-state local transport orbit.

    With ``P=I-L/d`` at the largest conservative Markov step, a transport
    eigenfunction obeys ``Pg=lambda*g``.  Therefore the positive local Gram
    law of ``(g,Pg)`` has rank one for *both* smooth and alternating ordered
    phases.  The transported Gram law of ``(g,Pg,P^2g)`` generally has higher
    rank for a superposition of unrelated transport directions.  Its
    normalized purity ``(3 tr(C^2)/tr(C)^2-1)/2`` maps rank-one support to one
    and an isotropic three-direction law to zero.
    """
    field = np.asarray(component, dtype=np.float64)
    flat = field.reshape(-1)
    degree = np.asarray(laplacian.diagonal(), dtype=np.float64)
    maximum_degree = float(np.max(degree)) if degree.size else 0.0
    if maximum_degree <= np.finfo(float).tiny:
        first_transport = field.copy()
        second_transport = field.copy()
    else:
        first_flat = (
            flat - np.asarray(laplacian @ flat) / maximum_degree
        )
        second_flat = (
            first_flat
            - np.asarray(laplacian @ first_flat) / maximum_degree
        )
        first_transport = first_flat.reshape(field.shape)
        second_transport = second_flat.reshape(field.shape)
    orbit = (field, first_transport, second_transport)
    moment_fields = []
    moment_indices = []
    for first in range(3):
        for second in range(first, 3):
            moment_fields.append(orbit[first] * orbit[second])
            moment_indices.append((first, second))
    moments = _screened_transport(
        laplacian,
        float(transport_time),
        np.stack(moment_fields),
    )
    matrix = np.zeros((3, 3) + field.shape, dtype=np.float64)
    for value, (first, second) in zip(moments, moment_indices):
        matrix[first, second] = value
        matrix[second, first] = value
    diagonal = np.stack((matrix[0, 0], matrix[1, 1], matrix[2, 2]))
    diagonal = np.maximum(diagonal, 0.0)
    trace = np.sum(diagonal, axis=0)
    frobenius_square = np.sum(diagonal * diagonal, axis=0) + 2.0 * (
        matrix[0, 1] ** 2
        + matrix[0, 2] ** 2
        + matrix[1, 2] ** 2
    )
    purity = np.divide(
        frobenius_square,
        trace * trace,
        out=np.full_like(trace, 1.0 / 3.0),
        where=trace > np.finfo(float).tiny,
    )
    normalized_purity = np.clip(0.5 * (3.0 * purity - 1.0), 0.0, 1.0)
    return normalized_purity, {
        "mean_krylov_spectral_purity": float(np.mean(normalized_purity)),
        "mean_krylov_spectral_impurity": float(
            np.mean(1.0 - normalized_purity)),
    }


def _spectral_measure_coordinates(
    field: np.ndarray,
    laplacian: sparse.csr_matrix,
    maximum_degree: float,
) -> dict[str, float]:
    """Return mean and normalized breadth of the generator spectral law.

    For ``Q=L/d`` the graph spectrum lies in ``[0,2]``.  The field-induced
    spectral probability has moments ``<g,Qg>/<g,g>`` and
    ``<Qg,Qg>/<g,g>``.  Bhatia--Davis gives

    ``Var(lambda) <= mu*(2-mu)``.

    The quotient is therefore a parameter-free measure of how many transport
    rates the field asks the operator to explain.  It is zero for every exact
    eigenphase, including both smooth and alternating endpoints.
    """
    flat = np.asarray(field, dtype=np.float64).reshape(-1)
    action = float(flat @ flat)
    if maximum_degree <= 0.0 or action <= np.finfo(float).tiny:
        return {
            "spectral_measure_mean": 0.0,
            "spectral_measure_second_moment": 0.0,
            "spectral_measure_variance": 0.0,
            "normalized_spectral_dispersion": 0.0,
            "normalized_spectral_concentration": 1.0,
        }
    generator_action = np.asarray(
        laplacian @ flat) / float(maximum_degree)
    mean = float(flat @ generator_action) / action
    second = float(generator_action @ generator_action) / action
    mean = float(np.clip(mean, 0.0, 2.0))
    variance = max(second - mean * mean, 0.0)
    capacity = mean * (2.0 - mean)
    dispersion = (
        variance / capacity
        if capacity > np.finfo(float).tiny else 0.0
    )
    dispersion = float(np.clip(dispersion, 0.0, 1.0))
    return {
        "spectral_measure_mean": mean,
        "spectral_measure_second_moment": second,
        "spectral_measure_variance": variance,
        "normalized_spectral_dispersion": dispersion,
        "normalized_spectral_concentration": 1.0 - dispersion,
    }


def causal_scale_transport_observation_2d(
    scene: np.ndarray,
    *,
    trace_refinement: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Decompose completely, then transport confidence cartoon-to-texture.

    The returned readout is deliberately provisional.  It is formed only
    after all telescoping components exist.  During decomposition no sample,
    residual, or fine-scale component is altered or discarded.
    """
    field = _validate(scene)
    spectrum = _isotropic_selling_spectrum(field.shape)
    times = _transport_times(spectrum, trace_refinement)

    chart_fields = []
    for covector in _COMPLETE_PARITY_COVECTORS:
        chart_fields.extend(interlaced_scene_views_2d(
            field, parity_covector=covector))
    complete_fields = np.stack((field, *chart_fields))
    coefficients = np.stack([
        dctn(member, type=2, norm="ortho") for member in complete_fields
    ])
    snapshots = np.stack([
        _heat_state(coefficients, spectrum, time) for time in times
    ])

    # Build the complete difference measure before confidence or removal is
    # allowed to inspect it.  Reversal is coarse/cartoon -> fine/texture.
    coarse = snapshots[-1, 0].copy()
    scene_components = snapshots[:-1, 0] - snapshots[1:, 0]
    chart_components = snapshots[:-1, 1:] - snapshots[1:, 1:]
    reverse_indices = range(scene_components.shape[0] - 1, -1, -1)
    exact_recomposition = coarse + np.sum(scene_components, axis=0)
    decomposition_error = float(np.max(np.abs(exact_recomposition - field)))

    reconstruction = coarse.copy()
    confidence = np.ones_like(field)
    preceding_phase = np.ones_like(field)
    confident_readout = coarse.copy()
    generations = []
    confidence_fields = []
    hadamard_pull_fields = []
    spectral_phase_fields = []
    spectral_connection_fields = []
    transport_pull_fields = []
    pulled_phase_fields = []
    hadamard_pulled_fields = []
    eigenprojected_components = []
    phase_susceptibility_fields = []
    selling_jet_pull_fields = []
    unresolved_krylov_purity_fields = []
    component_fields = []
    for ordinal, index in enumerate(reverse_indices):
        component = scene_components[index]
        scale_time = float(times[index + 1] - times[index])
        # ``reconstruction`` is the exact telescoping audit and necessarily
        # contains every preceding component.  Geometry is different: only
        # the confidence-bearing scene may orient later transport.  Otherwise
        # an unsupported coarse noise component writes a false fine metric.
        unresolved_before = field - confident_readout

        metric = continual_transport_metric(
            confident_readout, unresolved_before * unresolved_before)
        laplacian, _markov, stencil = _continual_flux_laplacian(
            metric, np.ones_like(field))
        transported_confidence = _screened_transport(
            laplacian, scale_time, confidence[None, ...])[0]

        parent_tensor = _heat_tensor(
            confident_readout, spectrum, scale_time)
        component_tensor = _heat_tensor(component, spectrum, scale_time)
        component_coordinates = _tensor_coordinates(component_tensor)
        lineage_overlap = _tensor_lineage_overlap(
            parent_tensor, component_tensor)
        maximum_degree = float(stencil["maximum_degree"])
        spectral_pull_time = (
            1.0 / maximum_degree if maximum_degree > 0.0 else 0.0)
        (
            phase_birth,
            hadamard_pull,
            spectral_phase_birth,
            eigenprojected_component,
            phase_diagnostic,
        ) = _scale_phase_birth(
            chart_components[index],
            laplacian,
            scale_time,
            hadamard_transport_time=spectral_pull_time,
        )
        selling_jet_pull, selling_jet_diagnostic = (
            _selling_jet_hadamard_pull(
                chart_components[index],
                laplacian,
                spectral_pull_time,
            )
        )
        # The witness must see the complete unresolved observation.  Applying
        # Krylov rank to one heat increment would certify the spectral
        # localization introduced by the numerical trace itself.
        unresolved_krylov_purity, krylov_diagnostic = _krylov_spectral_purity(
            unresolved_before,
            laplacian,
            spectral_pull_time,
        )
        # Hellinger coupling of spectral rank collapse and signed phase.  Both
        # coordinates are measured at the same operator-normalized time, so
        # this connection is independent of the stored scale interval.
        spectral_connection = np.sqrt(
            hadamard_pull * spectral_phase_birth)
        pulled_moments = _screened_transport(
            laplacian,
            spectral_pull_time,
            np.stack((component, component * component)),
        )
        pulled_mean = pulled_moments[0]
        pulled_second = np.maximum(pulled_moments[1], 0.0)
        transport_pull = np.divide(
            pulled_mean * pulled_mean,
            pulled_second,
            out=np.zeros_like(pulled_mean),
            where=pulled_second > np.finfo(float).tiny,
        )
        transport_pull = np.clip(transport_pull, 0.0, 1.0)
        # A component may be a slowly varying transported value or a signed
        # oscillatory phase.  Their smooth union is the complete pull witness;
        # neither a hard choice nor a selected texture band is introduced.
        pulled_phase = 1.0 - (
            (1.0 - transport_pull) * (1.0 - spectral_phase_birth))
        pulled_phase = np.clip(pulled_phase, 0.0, 1.0)
        hadamard_pulled = np.sqrt(hadamard_pull * pulled_phase)
        component_flat = component.reshape(-1)
        component_action_for_spectrum = float(
            component_flat @ component_flat)
        normalized_rayleigh = (
            float(component_flat @ (laplacian @ component_flat))
            / (maximum_degree * component_action_for_spectrum)
            if maximum_degree > 0.0
            and component_action_for_spectrum > np.finfo(float).tiny
            else 0.0
        )
        normalized_rayleigh = float(np.clip(normalized_rayleigh, 0.0, 2.0))
        component_spectral_measure = _spectral_measure_coordinates(
            component, laplacian, maximum_degree)
        unresolved_spectral_measure = _spectral_measure_coordinates(
            unresolved_before, laplacian, maximum_degree)
        refusal_distance = min(
            normalized_rayleigh, 2.0 - normalized_rayleigh)
        stable_spectral_order = 1.0 - refusal_distance
        spectral_susceptibility = (
            4.0 * refusal_distance * (1.0 - refusal_distance))
        phase_susceptibility = (
            (1.0 - spectral_susceptibility) * stable_spectral_order
            + spectral_susceptibility * spectral_phase_birth
        )
        phase_susceptibility = np.clip(
            phase_susceptibility, 0.0, 1.0)
        transported_preceding_phase = _screened_transport(
            laplacian, scale_time, preceding_phase[None, ...])[0]
        trace = np.asarray(component_coordinates["trace"])
        directional_share = np.divide(
            np.asarray(component_coordinates["directional_action"]),
            trace,
            out=np.zeros_like(trace),
            where=trace > np.finfo(float).tiny,
        )
        isotropic_share = 1.0 - directional_share
        # Directional evidence has an oriented jet and can be born at this
        # generation.  Isotropic evidence has no such identity; it enters only
        # through phase that survived transport from the coarser generation.
        # This is a continuous persistence coordinate, not a scale threshold.
        phase_persistence = np.clip(
            phase_birth * transported_preceding_phase, 0.0, 1.0)
        scale_birth = phase_birth * directional_share + (
            phase_persistence * isotropic_share)
        scale_birth = np.clip(scale_birth, 0.0, 1.0)
        inherited = np.clip(
            transported_confidence * lineage_overlap, 0.0, 1.0)
        # Same-orbit tensor ancestry is geometry, not fresh evidence.  It may
        # route an independently witnessed phase birth but must never admit a
        # component by itself.  With no parent overlap the new birth stands on
        # its own.  With complete overlap, the geometric coupling is the
        # parameter-free Hellinger product of incoming and inherited
        # confidence.
        confidence = (
            (1.0 - lineage_overlap) * scale_birth
            + lineage_overlap * np.sqrt(
                scale_birth * transported_confidence)
        )
        confidence = np.clip(confidence, 0.0, 1.0)
        preceding_phase = phase_birth

        reconstruction = reconstruction + component
        confident_readout = confident_readout + confidence * component
        component_action = component * component
        total_component_action = float(np.sum(component_action))
        confidence_fields.append(confidence.copy())
        hadamard_pull_fields.append(hadamard_pull.copy())
        spectral_phase_fields.append(spectral_phase_birth.copy())
        spectral_connection_fields.append(spectral_connection.copy())
        transport_pull_fields.append(transport_pull.copy())
        pulled_phase_fields.append(pulled_phase.copy())
        hadamard_pulled_fields.append(hadamard_pulled.copy())
        eigenprojected_components.append(eigenprojected_component.copy())
        phase_susceptibility_fields.append(phase_susceptibility.copy())
        selling_jet_pull_fields.append(selling_jet_pull.copy())
        unresolved_krylov_purity_fields.append(
            unresolved_krylov_purity.copy())
        component_fields.append(component.copy())
        generations.append({
            "ordinal_coarse_to_fine": int(ordinal),
            "transport_time_coarse": float(times[index + 1]),
            "transport_time_fine": float(times[index]),
            "scale_time": scale_time,
            "component_rms": float(np.sqrt(np.mean(component_action))),
            "component_action": total_component_action,
            "directional_fraction": component_coordinates[
                "directional_fraction"],
            "isotropic_fraction": component_coordinates[
                "isotropic_fraction"],
            "spatial_participation": component_coordinates[
                "spatial_participation"],
            "mean_lineage_overlap": float(np.mean(lineage_overlap)),
            "mean_inherited_confidence": float(np.mean(inherited)),
            "mean_phase_persistence": float(np.mean(phase_persistence)),
            "mean_scale_birth": float(np.mean(scale_birth)),
            "mean_confidence": float(np.mean(confidence)),
            "action_weighted_confidence": (
                float(np.sum(confidence * component_action))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "action_weighted_hadamard_pull": (
                float(np.sum(hadamard_pull * component_action))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "action_weighted_spectral_phase": (
                float(np.sum(spectral_phase_birth * component_action))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "action_weighted_spectral_connection": (
                float(np.sum(spectral_connection * component_action))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "action_weighted_transport_pull": (
                float(np.sum(transport_pull * component_action))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "action_weighted_pulled_phase": (
                float(np.sum(pulled_phase * component_action))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "action_weighted_hadamard_pulled": (
                float(np.sum(hadamard_pulled * component_action))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "eigenprojected_component_action_ratio": (
                float(np.sum(eigenprojected_component ** 2))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "normalized_rayleigh_eigenvalue": (
                normalized_rayleigh
            ),
            "component_spectral_measure": component_spectral_measure,
            "unresolved_spectral_measure": unresolved_spectral_measure,
            "spectral_refusal_distance": refusal_distance,
            "stable_spectral_order": stable_spectral_order,
            "spectral_susceptibility": spectral_susceptibility,
            "action_weighted_phase_susceptibility": (
                float(np.sum(phase_susceptibility * component_action))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "action_weighted_selling_jet_pull": (
                float(np.sum(selling_jet_pull * component_action))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "action_weighted_unresolved_krylov_purity": (
                float(np.sum(unresolved_krylov_purity * component_action))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "resolvent_energy_survival": (
                float(np.sum(pulled_mean * pulled_mean))
                / total_component_action
                if total_component_action > np.finfo(float).tiny else 0.0
            ),
            "action_weighted_isotropic_unresolved": (
                float(np.sum(
                    (1.0 - confidence)
                    * np.asarray(component_coordinates["isotropic_action"])))
                / max(float(component_coordinates["total_tensor_action"]),
                      np.finfo(float).tiny)
            ),
            "mean_metric_anisotropy": float(np.mean(
                np.hypot(
                    np.asarray(metric["metric_xx"])
                    - np.asarray(metric["metric_yy"]),
                    2.0 * np.asarray(metric["metric_xy"]),
                ) / (
                    np.asarray(metric["metric_xx"])
                    + np.asarray(metric["metric_yy"])
                )
            )),
            **phase_diagnostic,
            **selling_jet_diagnostic,
            **krylov_diagnostic,
            "stencil_maximum_degree": float(stencil["maximum_degree"]),
        })

    reconstruction_error = float(np.max(np.abs(reconstruction - field)))
    residual = field - confident_readout
    hadamard_readout = coarse + np.sum(
        np.stack(hadamard_pull_fields) * np.stack(component_fields), axis=0)
    spectral_phase_readout = coarse + np.sum(
        np.stack(spectral_phase_fields) * np.stack(component_fields), axis=0)
    spectral_connection_readout = coarse + np.sum(
        np.stack(spectral_connection_fields)
        * np.stack(component_fields),
        axis=0,
    )
    transport_pull_readout = coarse + np.sum(
        np.stack(transport_pull_fields) * np.stack(component_fields), axis=0)
    pulled_phase_readout = coarse + np.sum(
        np.stack(pulled_phase_fields) * np.stack(component_fields), axis=0)
    hadamard_pulled_readout = coarse + np.sum(
        np.stack(hadamard_pulled_fields) * np.stack(component_fields), axis=0)
    eigenprojection_readout = coarse + np.sum(
        np.stack(eigenprojected_components), axis=0)
    phase_susceptibility_readout = coarse + np.sum(
        np.stack(phase_susceptibility_fields)
        * np.stack(component_fields),
        axis=0,
    )
    selling_jet_pull_readout = coarse + np.sum(
        np.stack(selling_jet_pull_fields)
        * np.stack(component_fields),
        axis=0,
    )
    unresolved_krylov_purity_readout = coarse + np.sum(
        np.stack(unresolved_krylov_purity_fields)
        * np.stack(component_fields),
        axis=0,
    )
    return confident_readout, residual, {
        "status": (
            "complete isotropic Selling semigroup decomposition followed by "
            "coarse-to-fine anisotropic confidence transport"
        ),
        "transport_times": times,
        "trace_refinement": int(trace_refinement),
        "coarse_endpoint": coarse,
        "components_coarse_to_fine": np.stack(component_fields),
        "confidence_coarse_to_fine": np.stack(confidence_fields),
        "hadamard_pull_coarse_to_fine": np.stack(hadamard_pull_fields),
        "spectral_phase_coarse_to_fine": np.stack(spectral_phase_fields),
        "spectral_connection_coarse_to_fine": np.stack(
            spectral_connection_fields),
        "transport_pull_coarse_to_fine": np.stack(transport_pull_fields),
        "pulled_phase_coarse_to_fine": np.stack(pulled_phase_fields),
        "hadamard_pulled_coarse_to_fine": np.stack(hadamard_pulled_fields),
        "eigenprojected_components_coarse_to_fine": np.stack(
            eigenprojected_components),
        "phase_susceptibility_coarse_to_fine": np.stack(
            phase_susceptibility_fields),
        "selling_jet_pull_coarse_to_fine": np.stack(
            selling_jet_pull_fields),
        "unresolved_krylov_purity_coarse_to_fine": np.stack(
            unresolved_krylov_purity_fields),
        "generations": tuple(generations),
        "decomposition_maximum_error": decomposition_error,
        "reconstruction_maximum_error": reconstruction_error,
        "endpoint_distance_from_mean": float(np.max(np.abs(
            coarse - np.mean(field)))),
        "exact_bookkeeping_maximum_error": float(np.max(np.abs(
            confident_readout + residual - field))),
        "mean_retained_component_confidence": float(np.mean(
            np.stack(confidence_fields))),
        "readouts": {
            "hadamard_pull": hadamard_readout,
            "spectral_phase": spectral_phase_readout,
            "spectral_connection": spectral_connection_readout,
            "transport_pull": transport_pull_readout,
            "pulled_phase": pulled_phase_readout,
            "hadamard_pulled": hadamard_pulled_readout,
            "hadamard_eigenprojection": eigenprojection_readout,
            "phase_susceptibility": phase_susceptibility_readout,
            "selling_jet_pull": selling_jet_pull_readout,
            "unresolved_krylov_purity": unresolved_krylov_purity_readout,
        },
        "theory_status": (
            "exact continuous semigroup decomposition; the provisional "
            "per-generation confidence readout fails nested trace refinement "
            "and must be replaced by a continuous scale-density law"
        ),
    }


__all__ = [
    "causal_scale_transport_observation_2d",
]
