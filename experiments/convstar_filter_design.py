"""Bin-driven finite impulse response design by spectral projection.

This module makes the filter-design interpretation of CONV* explicit.  A
real FIR response is a linear map from taps to complex frequency bins.  The
weighted least-squares design is therefore a metric projection onto the
chosen realizable tap space.  Exact bin constraints are imposed by reducing
to their affine nullspace before the projection is solved.

The implementation supports 16 through 512 taps, arbitrary frequencies and
complex targets, and the four real-FIR linear-phase types through even/odd
tap symmetry.  The normal matrix is formed from its Toeplitz correlation
sequence, so memory is O(M L + L^2) only within a bounded bin chunk rather
than O(M L) for the complete design.

For real linear-phase amplitude specifications, ``design_minimax_from_bins``
solves the *discrete* weighted L-infinity problem as a linear program.  It is
kept separate from the general complex L2 projection because a complex
magnitude minimax problem is a second-order cone program, not a Remez or LP
problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray
MIN_FILTER_LENGTH = 16
MAX_FILTER_LENGTH = 512


@dataclass(frozen=True)
class DesignCertificate:
    """Numerical optimality certificate for a bin-driven design."""

    objective: str
    weighted_error: float
    weighted_rms_error: float
    maximum_weighted_error: float
    stationarity_residual: float
    exact_constraint_residual: float
    symmetry_residual: float
    iterations: int
    solve_seconds: float
    solver: str


@dataclass(frozen=True)
class FIRDesign:
    """A designed real FIR and the data needed to audit it."""

    taps: Array
    frequencies: Array
    desired: Array
    weights: Array
    response: Array
    symmetry: str
    certificate: DesignCertificate


def _as_vector(name: str, value: object, dtype: object) -> Array:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_length(length: int) -> int:
    value = int(length)
    if value != length or not MIN_FILTER_LENGTH <= value <= MAX_FILTER_LENGTH:
        raise ValueError(
            f"length must be an integer in [{MIN_FILTER_LENGTH}, "
            f"{MAX_FILTER_LENGTH}]"
        )
    return value


def _validate_problem(
    desired: object,
    length: int,
    frequencies: Optional[object],
    weights: Optional[object],
) -> Tuple[Array, int, Array, Array]:
    target = _as_vector("desired", desired, np.complex128)
    taps = _validate_length(length)
    if frequencies is None:
        frequency = np.linspace(0.0, 0.5, target.size, dtype=np.float64)
    else:
        frequency = _as_vector("frequencies", frequencies, np.float64)
    if frequency.shape != target.shape:
        raise ValueError("frequencies and desired must have the same shape")
    if np.any((frequency < 0.0) | (frequency > 0.5)):
        raise ValueError("frequencies must lie in [0, 0.5] cycles/sample")
    if weights is None:
        weight = np.ones(target.size, dtype=np.float64)
    else:
        weight = _as_vector("weights", weights, np.float64)
        if weight.shape != target.shape:
            raise ValueError("weights and desired must have the same shape")
        if np.any(weight < 0.0):
            raise ValueError("weights must be nonnegative")
    if not np.any(weight > 0.0):
        raise ValueError("at least one weight must be positive")
    return target, taps, frequency, weight


def _tap_parameterization(length: int, symmetry: str) -> Array:
    kind = str(symmetry).lower()
    if kind == "none":
        return np.eye(length, dtype=np.float64)
    if kind == "even":
        count = (length + 1) // 2
        mapping = np.zeros((length, count), dtype=np.float64)
        for index in range(count):
            mapping[index, index] = 1.0
            mapping[length - 1 - index, index] = 1.0
        return mapping
    if kind == "odd":
        count = length // 2
        mapping = np.zeros((length, count), dtype=np.float64)
        for index in range(count):
            mapping[index, index] = 1.0
            mapping[length - 1 - index, index] = -1.0
        return mapping
    raise ValueError("symmetry must be 'none', 'even', or 'odd'")


def frequency_response(
    taps: object, frequencies: object, *, chunk_size: int = 8192
) -> Array:
    """Evaluate a real FIR at frequencies measured in cycles/sample."""

    impulse = _as_vector("taps", taps, np.float64)
    frequency = _as_vector("frequencies", frequencies, np.float64)
    if np.any((frequency < 0.0) | (frequency > 0.5)):
        raise ValueError("frequencies must lie in [0, 0.5] cycles/sample")
    result = np.empty(frequency.size, dtype=np.complex128)
    coordinate = np.arange(impulse.size, dtype=np.float64)
    for start in range(0, frequency.size, int(chunk_size)):
        stop = min(start + int(chunk_size), frequency.size)
        angle = 2.0 * np.pi * frequency[start:stop, None] * coordinate[None, :]
        result[start:stop] = (
            np.cos(angle) @ impulse - 1j * (np.sin(angle) @ impulse)
        )
    return result


def linear_phase_target(
    amplitude: object, frequencies: object, length: int, *, symmetry: str = "even"
) -> Array:
    """Turn a real zero-phase amplitude target into a causal complex target."""

    value = _as_vector("amplitude", amplitude, np.float64)
    frequency = _as_vector("frequencies", frequencies, np.float64)
    taps = _validate_length(length)
    if value.shape != frequency.shape:
        raise ValueError("amplitude and frequencies must have the same shape")
    phase = np.exp(-2j * np.pi * frequency * ((taps - 1) / 2.0))
    kind = str(symmetry).lower()
    if kind == "even":
        return value.astype(np.complex128) * phase
    if kind == "odd":
        return -1j * value.astype(np.complex128) * phase
    raise ValueError("linear_phase_target requires even or odd symmetry")


def _normal_system(
    desired: Array,
    frequencies: Array,
    weights: Array,
    length: int,
    *,
    chunk_size: int,
) -> Tuple[Array, Array]:
    """Form Re(E* W E) and Re(E* W d) using the Toeplitz structure."""

    coordinate = np.arange(length, dtype=np.float64)
    correlation = np.zeros(length, dtype=np.float64)
    right = np.zeros(length, dtype=np.float64)
    for start in range(0, frequencies.size, int(chunk_size)):
        stop = min(start + int(chunk_size), frequencies.size)
        angle = (
            2.0 * np.pi * frequencies[start:stop, None] * coordinate[None, :]
        )
        cosine = np.cos(angle)
        sine = np.sin(angle)
        local_weight = weights[start:stop, None]
        correlation += np.sum(local_weight * cosine, axis=0)
        local_target = desired[start:stop]
        right += np.sum(
            local_weight
            * (
                local_target.real[:, None] * cosine
                - local_target.imag[:, None] * sine
            ),
            axis=0,
        )
    indices = np.arange(length, dtype=np.intp)
    gram = correlation[np.abs(indices[:, None] - indices[None, :])]
    return gram, right


def _exact_constraints(
    length: int,
    mapping: Array,
    frequencies: Optional[object],
    values: Optional[object],
) -> Tuple[Array, Array]:
    if frequencies is None and values is None:
        return np.empty((0, mapping.shape[1])), np.empty(0)
    if frequencies is None or values is None:
        raise ValueError("exact_frequencies and exact_values must be supplied together")
    exact_frequency = _as_vector("exact_frequencies", frequencies, np.float64)
    exact_value = _as_vector("exact_values", values, np.complex128)
    if exact_frequency.shape != exact_value.shape:
        raise ValueError("exact frequencies and values must have the same shape")
    if np.any((exact_frequency < 0.0) | (exact_frequency > 0.5)):
        raise ValueError("exact frequencies must lie in [0, 0.5]")
    coordinate = np.arange(length, dtype=np.float64)
    basis = np.exp(
        -2j * np.pi * exact_frequency[:, None] * coordinate[None, :]
    ) @ mapping
    rows = np.concatenate((basis.real, basis.imag), axis=0)
    rhs = np.concatenate((exact_value.real, exact_value.imag), axis=0)

    row_scale = np.linalg.norm(rows, axis=1)
    tolerance = 256.0 * np.finfo(np.float64).eps * max(
        float(np.max(row_scale, initial=0.0)), 1.0
    )
    impossible = (row_scale <= tolerance) & (np.abs(rhs) > tolerance)
    if np.any(impossible):
        raise ValueError("exact bin value is incompatible with the selected symmetry")
    keep = row_scale > tolerance
    return rows[keep], rhs[keep]


def _constraint_reduction(
    constraints: Array, rhs: Array, parameters: int
) -> Tuple[Array, Array, int]:
    """Return a particular point and orthonormal nullspace of C q = b."""

    if constraints.shape[0] == 0:
        return np.zeros(parameters), np.eye(parameters), 0
    u, singular, vh = np.linalg.svd(constraints, full_matrices=True)
    tolerance = max(constraints.shape) * np.finfo(np.float64).eps * max(
        float(singular[0]), 1.0
    )
    rank = int(np.sum(singular > tolerance))
    projected = u[:, :rank].T @ rhs
    particular = vh[:rank].T @ (projected / singular[:rank])
    residual = constraints @ particular - rhs
    if np.max(np.abs(residual), initial=0.0) > 2.0e3 * tolerance:
        raise ValueError("exact bin constraints are mutually inconsistent")
    return particular, vh[rank:].T.copy(), rank


def _solve_spd_refined(matrix: Array, rhs: Array) -> Tuple[Array, int, str]:
    if matrix.size == 0:
        return np.empty(0), 0, "fully constrained"
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        factor = np.linalg.cholesky(symmetric)

        def solve(value: Array) -> Array:
            return np.linalg.solve(factor.T, np.linalg.solve(factor, value))

        solution = solve(rhs)
        iterations = 0
        scale = max(float(np.linalg.norm(rhs)), 1.0)
        for _ in range(3):
            residual = rhs - symmetric @ solution
            if float(np.linalg.norm(residual)) <= 2.0e-12 * scale:
                break
            solution += solve(residual)
            iterations += 1
        return solution, iterations, "Toeplitz normal projection (Cholesky)"
    except np.linalg.LinAlgError:
        solution, _, _, _ = np.linalg.lstsq(symmetric, rhs, rcond=None)
        return solution, 0, "Toeplitz normal projection (SVD fallback)"


def design_from_bins(
    desired: object,
    *,
    length: int,
    frequencies: Optional[object] = None,
    weights: Optional[object] = None,
    symmetry: str = "none",
    exact_frequencies: Optional[object] = None,
    exact_values: Optional[object] = None,
    regularization: float = 0.0,
    chunk_size: int = 8192,
) -> FIRDesign:
    """Design the globally optimal weighted-L2 real FIR for requested bins.

    ``weights[k]`` multiplies ``abs(H(f[k]) - desired[k])**2``.  A positive
    ``regularization`` adds that value times ``sum(taps**2)``.  Exact bins are
    enforced as complex equalities, subject to the selected tap symmetry.
    """

    started = perf_counter()
    target, taps, frequency, weight = _validate_problem(
        desired, length, frequencies, weights
    )
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and nonnegative")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")

    kind = str(symmetry).lower()
    mapping = _tap_parameterization(taps, kind)
    tap_gram, tap_right = _normal_system(
        target, frequency, weight, taps, chunk_size=int(chunk_size)
    )
    if kind == "none":
        parameter_gram = tap_gram.copy()
        parameter_right = tap_right.copy()
        regularizer = np.eye(taps)
    else:
        parameter_gram = mapping.T @ tap_gram @ mapping
        parameter_right = mapping.T @ tap_right
        regularizer = mapping.T @ mapping
    parameter_gram += float(regularization) * regularizer

    constraints, constraint_rhs = _exact_constraints(
        taps, mapping, exact_frequencies, exact_values
    )
    particular, nullspace, _ = _constraint_reduction(
        constraints, constraint_rhs, mapping.shape[1]
    )
    reduced_gram = nullspace.T @ parameter_gram @ nullspace
    reduced_rhs = nullspace.T @ (
        parameter_right - parameter_gram @ particular
    )
    reduced, iterations, solver = _solve_spd_refined(reduced_gram, reduced_rhs)
    parameters = particular + nullspace @ reduced
    impulse = mapping @ parameters
    response = frequency_response(impulse, frequency, chunk_size=int(chunk_size))
    error = response - target

    gradient = parameter_gram @ parameters - parameter_right
    if constraints.shape[0]:
        multiplier, _, _, _ = np.linalg.lstsq(
            constraints.T, -gradient, rcond=None
        )
        stationarity = gradient + constraints.T @ multiplier
        constraint_residual = float(
            np.max(np.abs(constraints @ parameters - constraint_rhs), initial=0.0)
        )
    else:
        stationarity = gradient
        constraint_residual = 0.0
    stationarity_scale = max(
        float(np.linalg.norm(parameter_right)),
        float(np.linalg.norm(parameter_gram @ parameters)),
        1.0,
    )
    if kind == "even":
        symmetry_residual = float(np.max(np.abs(impulse - impulse[::-1])))
    elif kind == "odd":
        symmetry_residual = float(np.max(np.abs(impulse + impulse[::-1])))
    else:
        symmetry_residual = 0.0
    weighted_square = weight * np.abs(error) ** 2
    certificate = DesignCertificate(
        objective="weighted_l2",
        weighted_error=float(np.sum(weighted_square)),
        weighted_rms_error=float(
            np.sqrt(np.sum(weighted_square) / np.sum(weight))
        ),
        maximum_weighted_error=float(
            np.max(np.sqrt(weight) * np.abs(error), initial=0.0)
        ),
        stationarity_residual=float(
            np.linalg.norm(stationarity) / stationarity_scale
        ),
        exact_constraint_residual=constraint_residual,
        symmetry_residual=symmetry_residual,
        iterations=iterations,
        solve_seconds=perf_counter() - started,
        solver=solver,
    )
    return FIRDesign(
        taps=impulse,
        frequencies=frequency,
        desired=target,
        weights=weight,
        response=response,
        symmetry=kind,
        certificate=certificate,
    )


def _linear_phase_amplitude_basis(
    frequencies: Array, length: int, symmetry: str, mapping: Array
) -> Array:
    coordinate = np.arange(length, dtype=np.float64)
    complex_basis = np.exp(
        -2j * np.pi * frequencies[:, None] * coordinate[None, :]
    ) @ mapping
    undo_delay = np.exp(
        2j * np.pi * frequencies * ((length - 1) / 2.0)
    )[:, None]
    if symmetry == "even":
        rotated = undo_delay * complex_basis
    elif symmetry == "odd":
        rotated = 1j * undo_delay * complex_basis
    else:
        raise ValueError("minimax amplitude design requires even or odd symmetry")
    imaginary_residual = float(np.max(np.abs(rotated.imag), initial=0.0))
    if imaginary_residual > 2.0e-10:
        raise RuntimeError("linear-phase basis failed to become real")
    return rotated.real


def design_minimax_from_bins(
    amplitude: object,
    *,
    length: int,
    frequencies: Optional[object] = None,
    weights: Optional[object] = None,
    symmetry: str = "even",
    exact_frequencies: Optional[object] = None,
    exact_amplitudes: Optional[object] = None,
) -> FIRDesign:
    """Solve the discrete weighted-L-infinity linear-phase amplitude problem.

    This uses SciPy/HiGHS lazily.  Its certificate applies to the supplied
    finite bin set; it does not by itself bound response peaks between bins.
    """

    started = perf_counter()
    real_target = _as_vector("amplitude", amplitude, np.float64)
    target, taps, frequency, weight = _validate_problem(
        real_target.astype(np.complex128), length, frequencies, weights
    )
    kind = str(symmetry).lower()
    mapping = _tap_parameterization(taps, kind)
    basis = _linear_phase_amplitude_basis(frequency, taps, kind, mapping)
    count = mapping.shape[1]

    weighted_basis = weight[:, None] * basis
    inequality = np.concatenate((
        np.column_stack((weighted_basis, -np.ones(frequency.size))),
        np.column_stack((-weighted_basis, -np.ones(frequency.size))),
    ), axis=0)
    inequality_rhs = np.concatenate((weight * real_target, -weight * real_target))

    equality = None
    equality_rhs = None
    if exact_frequencies is not None or exact_amplitudes is not None:
        if exact_frequencies is None or exact_amplitudes is None:
            raise ValueError(
                "exact_frequencies and exact_amplitudes must be supplied together"
            )
        exact_frequency = _as_vector(
            "exact_frequencies", exact_frequencies, np.float64
        )
        exact_value = _as_vector("exact_amplitudes", exact_amplitudes, np.float64)
        if exact_frequency.shape != exact_value.shape:
            raise ValueError("exact frequencies and amplitudes must have the same shape")
        exact_basis = _linear_phase_amplitude_basis(
            exact_frequency, taps, kind, mapping
        )
        equality = np.column_stack((exact_basis, np.zeros(exact_frequency.size)))
        equality_rhs = exact_value

    try:
        from scipy.optimize import linprog
    except ImportError as error:
        raise RuntimeError(
            "discrete minimax design requires scipy.optimize.linprog"
        ) from error
    objective = np.zeros(count + 1)
    objective[-1] = 1.0
    bounds: Sequence[Tuple[Optional[float], Optional[float]]] = (
        [(None, None)] * count + [(0.0, None)]
    )
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=inequality_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1.0e-10,
            "dual_feasibility_tolerance": 1.0e-10,
            "ipm_optimality_tolerance": 1.0e-10,
        },
    )
    if not result.success:
        raise RuntimeError(f"minimax design failed: {result.message}")
    parameters = result.x[:-1]
    impulse = mapping @ parameters
    response = frequency_response(impulse, frequency)
    complex_target = linear_phase_target(
        real_target, frequency, taps, symmetry=kind
    )
    error = response - complex_target
    rotated_response = basis @ parameters
    amplitude_error = rotated_response - real_target
    maximum = float(np.max(weight * np.abs(amplitude_error), initial=0.0))
    epigraph_residual = abs(maximum - float(result.x[-1]))
    exact_residual = 0.0
    if equality is not None:
        exact_residual = float(
            np.max(np.abs(equality[:, :-1] @ parameters - equality_rhs), initial=0.0)
        )
    if kind == "even":
        symmetry_residual = float(np.max(np.abs(impulse - impulse[::-1])))
    else:
        symmetry_residual = float(np.max(np.abs(impulse + impulse[::-1])))
    weighted_square = weight * np.abs(error) ** 2
    certificate = DesignCertificate(
        objective="discrete_weighted_linf",
        weighted_error=float(np.sum(weighted_square)),
        weighted_rms_error=float(
            np.sqrt(np.sum(weighted_square) / np.sum(weight))
        ),
        maximum_weighted_error=maximum,
        # HiGHS supplies the dual optimality proof.  The reported scalar is
        # the independently recomputed primal epigraph gap.
        stationarity_residual=epigraph_residual,
        exact_constraint_residual=exact_residual,
        symmetry_residual=symmetry_residual,
        iterations=int(result.nit),
        solve_seconds=perf_counter() - started,
        solver="HiGHS discrete minimax LP",
    )
    return FIRDesign(
        taps=impulse,
        frequencies=frequency,
        desired=complex_target,
        weights=weight,
        response=response,
        symmetry=kind,
        certificate=certificate,
    )
