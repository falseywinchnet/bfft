"""Continual 2-D radiance, noise-law, and eikonal-statistic transport.

This experiment is deliberately not an FMMT variant and does not use image
patches, named noise classes, fixed bands, owner cells, or a prescribed
smoothing depth.  Its state is

``(radiance, noise centre, noise dispersion, noise radius, transport metric)``

and every accepted numerical step evolves all five fields.

Four paired, target-excluded directional characteristics provide a small
bounded mixture of *noise witnesses*.  They are not treated as independent
observations.  Their centre, total mixture variance, and outer radius are
transported through the same positive Markov operator as the image geometry.
The incoming witness law and transported law are joined by the law of total
variance, so iteration can never manufacture confidence by repeatedly seeing
the same pixels.

The evolving excess structure tensor defines a positive Riemannian metric.
The V3 segmenter's FM-LBR/Selling reduction converts its inverse into three
nonnegative local lattice fluxes.  Symmetrization produces a conservative
Dirichlet Laplacian.  At iteration ``t`` the radiance step minimizes the
frozen Back-to-Basics action

    E_t(x) = 1/2 ||x - (y - a_t mu_t)||^2 + 1/2 x^T L_t x.

``a_t`` is continuous agreement of the bounded noise witnesses, not a noise
label or posterior probability.  A Gershgorin majorizer of ``I + L_t`` gives
the step exactly; the only iteration count is a numerical failure ceiling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math

import numpy as np
from scipy import sparse

from .continuous_source_transport import selling_decomposition


_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True)
class ContinualEikonalResolution:
    """Numerical guards; neither member selects a physical smoothing scale."""

    maximum_iterations: int = 64
    convergence_multiplier: float = 32.0


def _validate_image(observation: np.ndarray) -> np.ndarray:
    image = np.asarray(observation, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("continual eikonal transport expects an HxW field")
    if not np.all(np.isfinite(image)):
        raise ValueError("continual eikonal transport requires finite samples")
    return image


def _shift_symmetric(field: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer translate with half-sample symmetric boundary continuation."""
    height, width = field.shape
    pad_y = abs(int(dy))
    pad_x = abs(int(dx))
    padded = np.pad(field, ((pad_y, pad_y), (pad_x, pad_x)), mode="symmetric")
    y0 = pad_y + int(dy)
    x0 = pad_x + int(dx)
    return padded[y0:y0 + height, x0:x0 + width]


def directional_noise_witnesses(
    observation: np.ndarray,
    radiance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Measure the bounded paired-characteristic noise mixture.

    For direction ``d``, the target is never read by its clean prediction:

        eta_d(p) = y(p) - (x(p-d) + x(p+d)) / 2.

    The median is only the location readout of the four-member set.  The
    returned variance is the complete mixture variance around that readout;
    it is not divided by direction count.
    """
    y = _validate_image(observation)
    x = np.asarray(radiance, dtype=np.float64)
    if x.shape != y.shape or not np.all(np.isfinite(x)):
        raise ValueError("radiance must be a finite field aligned with observation")
    members = []
    for dy, dx in _DIRECTIONS:
        minus = _shift_symmetric(x, -dy, -dx)
        plus = _shift_symmetric(x, dy, dx)
        members.append(y - 0.5 * (minus + plus))
    mixture = np.stack(members, axis=0)
    centre = np.median(mixture, axis=0)
    centered = mixture - centre[None, ...]
    variance = np.mean(centered * centered, axis=0)
    radius = np.max(np.abs(centered), axis=0)
    scale = max(float(np.max(np.abs(y))), float(np.ptp(y)), 1.0)
    floor = np.finfo(float).eps * scale * scale
    agreement = centre * centre / (centre * centre + variance + floor)
    return centre, variance, radius, {
        "mean_absolute_centre": float(np.mean(np.abs(centre))),
        "mean_dispersion": float(np.mean(variance)),
        "mean_radius": float(np.mean(radius)),
        "mean_agreement": float(np.mean(agreement)),
    }


def _reflect_index(index: np.ndarray, size: int) -> np.ndarray:
    """Half-sample symmetric continuation for arbitrary integer indices."""
    period = 2 * int(size)
    wrapped = np.mod(index, period)
    return np.where(wrapped < size, wrapped, period - 1 - wrapped)


def eikonal_noise_witnesses(
    observation: np.ndarray,
    radiance: np.ndarray,
    metric: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Measure paired noise characteristics on the local Selling superbase."""
    y = _validate_image(observation)
    x = np.asarray(radiance, dtype=np.float64)
    if x.shape != y.shape or not np.all(np.isfinite(x)):
        raise ValueError("radiance must be a finite field aligned with observation")
    decomposition = selling_decomposition(
        np.asarray(metric["metric_xx"]),
        np.asarray(metric["metric_xy"]),
        np.asarray(metric["metric_yy"]),
    )
    vectors = np.asarray(decomposition["vectors"], dtype=np.int64)
    coefficient = np.asarray(decomposition["coefficient"], dtype=np.float64)
    weight = coefficient / np.maximum(
        np.sum(coefficient, axis=-1, keepdims=True), np.finfo(float).tiny)
    height, width = x.shape
    yy, xx = np.mgrid[:height, :width]
    members = []
    for basis in range(3):
        dx = vectors[..., basis, 0]
        dy = vectors[..., basis, 1]
        minus_y = _reflect_index(yy - dy, height)
        minus_x = _reflect_index(xx - dx, width)
        plus_y = _reflect_index(yy + dy, height)
        plus_x = _reflect_index(xx + dx, width)
        members.append(y - 0.5 * (
            x[minus_y, minus_x] + x[plus_y, plus_x]))
    mixture = np.stack(members, axis=-1)
    centre = np.sum(weight * mixture, axis=-1)
    centered = mixture - centre[..., None]
    variance = np.sum(weight * centered * centered, axis=-1)
    active = weight > 64.0 * np.finfo(float).eps
    radius = np.max(np.where(active, np.abs(centered), 0.0), axis=-1)
    scale = max(float(np.max(np.abs(y))), float(np.ptp(y)), 1.0)
    floor = np.finfo(float).eps * scale * scale
    agreement = centre * centre / (centre * centre + variance + floor)
    reach = np.max(np.abs(vectors), axis=(-1, -2))
    return centre, variance, radius, {
        "mean_absolute_centre": float(np.mean(np.abs(centre))),
        "mean_dispersion": float(np.mean(variance)),
        "mean_radius": float(np.mean(radius)),
        "mean_agreement": float(np.mean(agreement)),
        "selling_reach_p90": float(np.percentile(reach, 90.0)),
        "selling_reach_maximum": float(np.max(reach)),
    }


def phase_covector_sufficient_statistics(
    radiance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired one-sided phase numerators and denominators.

    The direction order is horizontal, vertical, positive diagonal, negative
    diagonal.  First differences remove the unknown radiance offset before
    correlation is measured.  Numerator and denominator remain separate so
    positive transport commutes with accumulation; phase angles never do.
    """
    x = np.asarray(radiance, dtype=np.float64)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("phase covectors require a finite 2-D radiance field")
    numerator = []
    denominator = []
    for dy, dx in _DIRECTIONS:
        minus = x - _shift_symmetric(x, -dy, -dx)
        plus = _shift_symmetric(x, dy, dx) - x
        numerator.append(minus * plus)
        denominator.append(0.5 * (minus * minus + plus * plus))
    return np.stack(numerator), np.stack(denominator)


def phase_covector_noise_authority(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Measure departure from a single integrable full-band wave covector.

    If ``Cx=cos(wx)`` and ``Cy=cos(wy)``, the two diagonal correlations must
    be the unordered pair ``cos(wx+wy), cos(wx-wy)``.  Their exact algebraic
    values are ``Cx*Cy +/- sqrt((1-Cx^2)(1-Cy^2))``.  Energy lying on this
    manifold is coherent phase and receives no noise authority; only the
    normalized transverse defect can authorize denoising.
    """
    n = np.asarray(numerator, dtype=np.float64)
    d = np.asarray(denominator, dtype=np.float64)
    if n.ndim != 3 or n.shape[0] != 4 or d.shape != n.shape:
        raise ValueError("phase statistics must have shape 4xHxW")
    magnitude = max(float(np.max(d)), 1.0)
    floor = np.finfo(float).eps * magnitude
    correlation = np.divide(
        n, d, out=np.zeros_like(n), where=d > floor)
    correlation = np.clip(correlation, -1.0, 1.0)
    cx, cy, diagonal_positive, diagonal_negative = correlation
    product = cx * cy
    quadrature = np.sqrt(np.maximum(
        (1.0 - cx * cx) * (1.0 - cy * cy), 0.0))
    predicted_high = product + quadrature
    predicted_low = product - quadrature
    observed_high = np.maximum(diagonal_positive, diagonal_negative)
    observed_low = np.minimum(diagonal_positive, diagonal_negative)
    defect = (
        (observed_high - predicted_high) ** 2
        + (observed_low - predicted_low) ** 2
    )
    phase_energy = np.sum(correlation * correlation, axis=0)
    coherent_fraction = phase_energy / (phase_energy + defect + floor)
    authority = np.clip(1.0 - coherent_fraction, 0.0, 1.0)
    return authority, {
        "mean_phase_noise_authority": float(np.mean(authority)),
        "mean_phase_coherent_fraction": float(np.mean(coherent_fraction)),
        "mean_phase_covector_defect": float(np.mean(defect)),
        "mean_phase_correlation_energy": float(np.mean(phase_energy)),
    }


def _transport_phase_statistics(
    operator: sparse.csr_matrix,
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    directions, height, width = numerator.shape
    transported_numerator = (
        operator @ numerator.reshape(directions, -1).T
    ).T.reshape(directions, height, width)
    transported_denominator = (
        operator @ denominator.reshape(directions, -1).T
    ).T.reshape(directions, height, width)
    return transported_numerator, np.maximum(transported_denominator, 0.0)


def _positive_semidefinite_excess(
    xx: np.ndarray,
    xy: np.ndarray,
    yy: np.ndarray,
    isotropic_uncertainty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the positive part of a symmetric 2x2 tensor field."""
    a = xx - isotropic_uncertainty
    b = xy
    c = yy - isotropic_uncertainty
    trace = a + c
    gap = np.hypot(a - c, 2.0 * b)
    high = np.maximum(0.5 * (trace + gap), 0.0)
    low = np.maximum(0.5 * (trace - gap), 0.0)
    angle = 0.5 * np.arctan2(2.0 * b, a - c)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    out_xx = high * cosine * cosine + low * sine * sine
    out_xy = (high - low) * cosine * sine
    out_yy = high * sine * sine + low * cosine * cosine
    return out_xx, out_xy, out_yy


def continual_transport_metric(
    radiance: np.ndarray,
    noise_variance: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Build an SPD eikonal metric from structure above noise uncertainty."""
    x = np.asarray(radiance, dtype=np.float64)
    variance = np.maximum(np.asarray(noise_variance, dtype=np.float64), 0.0)
    if x.ndim != 2 or variance.shape != x.shape:
        raise ValueError("radiance and noise variance must be aligned 2-D fields")
    gx = 0.5 * (
        _shift_symmetric(x, 0, 1) - _shift_symmetric(x, 0, -1))
    gy = 0.5 * (
        _shift_symmetric(x, 1, 0) - _shift_symmetric(x, -1, 0))
    # A central first difference of independent noise has variance sigma^2/2.
    exx, exy, eyy = _positive_semidefinite_excess(
        gx * gx, gx * gy, gy * gy, 0.5 * variance)
    magnitude = max(float(np.max(np.abs(x))), float(np.ptp(x)), 1.0)
    numerical = np.finfo(float).eps * magnitude * magnitude
    denominator = variance + np.sqrt(
        np.maximum(exx * eyy - exy * exy, 0.0)) + numerical
    mxx = 1.0 + exx / denominator
    mxy = exy / denominator
    myy = 1.0 + eyy / denominator
    determinant = mxx * myy - mxy * mxy
    if np.any(mxx <= 0.0) or np.any(determinant <= 0.0):
        raise RuntimeError("continual statistic metric left the SPD cone")
    return {
        "metric_xx": np.ascontiguousarray(mxx),
        "metric_xy": np.ascontiguousarray(mxy),
        "metric_yy": np.ascontiguousarray(myy),
        "metric_determinant_minimum": float(np.min(determinant)),
        "metric_condition_p90": float(np.percentile(
            (mxx + myy + np.hypot(mxx - myy, 2.0 * mxy))
            / np.maximum(
                mxx + myy - np.hypot(mxx - myy, 2.0 * mxy), numerical),
            90.0,
        )),
    }


def continual_anisotropic_noise_metric(
    radiance: np.ndarray,
    noise_centre: np.ndarray,
    noise_variance: np.ndarray,
    phase_noise_authority: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Subtract a phase-vetted directional nuisance tensor from structure.

    The central directional dispersion supplies isotropic uncertainty.  The
    gradient of the transported residual centre supplies correlated nuisance
    uncertainty, but only to the extent that its phase is incoherent.  Thus a
    row-correlated residual can stop masquerading as an orthogonal image edge
    without declaring a row-noise class.
    """
    x = np.asarray(radiance, dtype=np.float64)
    centre = np.asarray(noise_centre, dtype=np.float64)
    variance = np.maximum(np.asarray(noise_variance, dtype=np.float64), 0.0)
    phase_noise = np.clip(
        np.asarray(phase_noise_authority, dtype=np.float64), 0.0, 1.0)
    if (
        x.ndim != 2
        or centre.shape != x.shape
        or variance.shape != x.shape
        or phase_noise.shape != x.shape
    ):
        raise ValueError("anisotropic metric fields must be aligned and 2-D")
    gx = 0.5 * (
        _shift_symmetric(x, 0, 1) - _shift_symmetric(x, 0, -1))
    gy = 0.5 * (
        _shift_symmetric(x, 1, 0) - _shift_symmetric(x, -1, 0))
    rx = 0.5 * (
        _shift_symmetric(centre, 0, 1)
        - _shift_symmetric(centre, 0, -1))
    ry = 0.5 * (
        _shift_symmetric(centre, 1, 0)
        - _shift_symmetric(centre, -1, 0))
    nxx = 0.5 * variance + phase_noise * rx * rx
    nxy = phase_noise * rx * ry
    nyy = 0.5 * variance + phase_noise * ry * ry
    exx, exy, eyy = _positive_semidefinite_excess(
        gx * gx - nxx,
        gx * gy - nxy,
        gy * gy - nyy,
        np.zeros_like(x),
    )
    magnitude = max(float(np.max(np.abs(x))), float(np.ptp(x)), 1.0)
    numerical = np.finfo(float).eps * magnitude * magnitude
    nuisance_scale = 0.5 * (nxx + nyy)
    denominator = nuisance_scale + np.sqrt(
        np.maximum(exx * eyy - exy * exy, 0.0)) + numerical
    mxx = 1.0 + exx / denominator
    mxy = exy / denominator
    myy = 1.0 + eyy / denominator
    determinant = mxx * myy - mxy * mxy
    if np.any(mxx <= 0.0) or np.any(determinant <= 0.0):
        raise RuntimeError("anisotropic statistic metric left the SPD cone")
    return {
        "metric_xx": np.ascontiguousarray(mxx),
        "metric_xy": np.ascontiguousarray(mxy),
        "metric_yy": np.ascontiguousarray(myy),
        "metric_determinant_minimum": float(np.min(determinant)),
        "metric_condition_p90": float(np.percentile(
            (mxx + myy + np.hypot(mxx - myy, 2.0 * mxy))
            / np.maximum(
                mxx + myy - np.hypot(mxx - myy, 2.0 * mxy), numerical),
            90.0,
        )),
        "mean_correlated_nuisance_trace": float(np.mean(
            phase_noise * (rx * rx + ry * ry))),
        "mean_isotropic_nuisance_trace": float(np.mean(variance)),
    }


def _continual_flux_laplacian(
    metric: dict[str, np.ndarray | float],
    authority: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, float]]:
    """Return symmetric eikonal flux ``L`` and its pure transport ``P``."""
    decomposition = selling_decomposition(
        np.asarray(metric["metric_xx"]),
        np.asarray(metric["metric_xy"]),
        np.asarray(metric["metric_yy"]),
    )
    vectors = np.asarray(decomposition["vectors"], dtype=np.int64)
    coefficient = np.asarray(decomposition["coefficient"], dtype=np.float64)
    height, width = coefficient.shape[:2]
    pixels = height * width
    yy, xx = np.mgrid[:height, :width]
    vertex = np.arange(pixels, dtype=np.int64).reshape(height, width)
    # Local normalization removes a metric-strength knob while retaining the
    # exact directional second moment represented by the Selling basis.
    coefficient /= np.maximum(
        2.0 * np.sum(coefficient, axis=-1, keepdims=True),
        np.finfo(float).tiny,
    )
    gate = np.clip(np.asarray(authority, dtype=np.float64), 0.0, 1.0)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for basis in range(3):
        for sign in (-1, 1):
            nx = xx + sign * vectors[..., basis, 0]
            ny = yy + sign * vectors[..., basis, 1]
            valid = (0 <= nx) & (nx < width) & (0 <= ny) & (ny < height)
            rows.append(vertex[valid])
            columns.append((ny[valid] * width + nx[valid]).astype(np.int64))
            values.append((coefficient[..., basis] * gate)[valid])
    directed = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(pixels, pixels),
    ).tocsr()
    directed.sum_duplicates()
    conductance = (0.5 * (directed + directed.T)).tocsr()
    conductance.setdiag(0.0)
    conductance.eliminate_zeros()
    degree = np.asarray(conductance.sum(axis=1)).ravel()
    laplacian = (sparse.diags(degree) - conductance).tocsr()
    maximum_degree = float(np.max(degree))
    if maximum_degree == 0.0:
        transport = sparse.eye(pixels, format="csr")
    else:
        # The largest nonnegative step makes I-tau*L a conservative symmetric
        # Markov operator.  This is statistic transport, not the radiance MM
        # step, whose Hessian also contains the identity data term.
        transport = (
            sparse.eye(pixels, format="csr")
            - laplacian / maximum_degree
        ).tocsr()
    return laplacian, transport, {
        "selling_maximum_reconstruction_error": float(
            decomposition["maximum_reconstruction_error"]),
        "selling_minimum_coefficient": float(
            decomposition["minimum_coefficient"]),
        "maximum_degree": maximum_degree,
        "laplacian_row_sum_error": float(np.max(np.abs(
            np.asarray(laplacian.sum(axis=1)).ravel()))),
        "transport_row_sum_error": float(np.max(np.abs(
            np.asarray(transport.sum(axis=1)).ravel() - 1.0))),
        "transport_column_sum_error": float(np.max(np.abs(
            np.asarray(transport.sum(axis=0)).ravel() - 1.0))),
        "undirected_edge_count": int(conductance.nnz // 2),
    }


def _mixture_moment_fusion(
    prior_centre: np.ndarray,
    prior_variance: np.ndarray,
    prior_radius: np.ndarray,
    witness_centre: np.ndarray,
    witness_variance: np.ndarray,
    witness_radius: np.ndarray,
    witness_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join two bounded laws without pretending they are independent data."""
    fraction = float(np.clip(witness_fraction, 0.0, 1.0))
    prior_fraction = 1.0 - fraction
    centre = prior_fraction * prior_centre + fraction * witness_centre
    variance = (
        prior_fraction * (
            prior_variance + (prior_centre - centre) ** 2)
        + fraction * (
            witness_variance + (witness_centre - centre) ** 2)
    )
    radius = np.maximum.reduce((
        np.abs(prior_centre - prior_radius - centre),
        np.abs(prior_centre + prior_radius - centre),
        np.abs(witness_centre - witness_radius - centre),
        np.abs(witness_centre + witness_radius - centre),
    ))
    return centre, np.maximum(variance, 0.0), radius


def _frozen_action(
    state: np.ndarray,
    target: np.ndarray,
    laplacian: sparse.csr_matrix,
) -> float:
    flat = state.ravel()
    difference = flat - target.ravel()
    return 0.5 * float(
        difference @ difference + flat @ (laplacian @ flat))


def _bounded_noise_violation(
    observation: np.ndarray,
    radiance: np.ndarray,
    centre: np.ndarray,
    radius: np.ndarray,
) -> float:
    """Squared distance of the exact residual from a bounded noise law."""
    residual = np.asarray(observation) - np.asarray(radiance)
    gap = np.maximum(np.abs(residual - centre) - radius, 0.0)
    return float(np.mean(gap * gap))


def _phase_constrained_noise_action(
    observation: np.ndarray,
    radiance: np.ndarray,
    centre: np.ndarray,
    radius: np.ndarray,
    phase_transport: sparse.csr_matrix | None,
) -> tuple[float, dict[str, float]]:
    """Require the exact residual to be bounded *and* phase-incoherent."""
    residual = np.asarray(observation) - np.asarray(radiance)
    bounded = _bounded_noise_violation(
        observation, radiance, centre, radius)
    if phase_transport is None or not np.any(residual):
        coherent_penalty = 0.0
        phase = {
            "mean_phase_noise_authority": 1.0,
            "mean_phase_coherent_fraction": 0.0,
            "mean_phase_covector_defect": 0.0,
            "mean_phase_correlation_energy": 0.0,
        }
    else:
        numerator, denominator = phase_covector_sufficient_statistics(residual)
        numerator, denominator = _transport_phase_statistics(
            phase_transport, numerator, denominator)
        noise_authority, phase = phase_covector_noise_authority(
            numerator, denominator)
        # V3's nonexpansive envelope constrains Dirichlet rather than carrier
        # amplitude.  Charging coherent residual gradient energy measures the
        # precise structural quantity a denoising move removes from radiance.
        coherent_fraction = 1.0 - noise_authority
        local_dirichlet = np.mean(denominator, axis=0)
        coherent_carrier = float(np.mean(
            residual * residual * coherent_fraction))
        coherent_dirichlet = float(np.mean(
            local_dirichlet * coherent_fraction))
        # The geometric mean is the scale-dual value/first-jet (Sasaki)
        # balance.  It has intensity-squared units and introduces no relative
        # penalty coefficient.
        coherent_penalty = math.sqrt(
            max(coherent_carrier * coherent_dirichlet, 0.0))
    return bounded + coherent_penalty, {
        "bounded_set_violation": bounded,
        "coherent_residual_penalty": coherent_penalty,
        **phase,
    }


def denoise_continual_eikonal_noise_transport_2d(
    observation: np.ndarray,
    resolution: ContinualEikonalResolution = ContinualEikonalResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evolve radiance and its bounded noise law to numerical equilibrium."""
    image = _validate_image(observation)
    ceiling = int(resolution.maximum_iterations)
    if ceiling < 1:
        raise ValueError("maximum_iterations must be positive")
    if resolution.convergence_multiplier <= 0.0:
        raise ValueError("convergence_multiplier must be positive")
    state = image.copy()
    centre, variance, radius, initial_witness = directional_noise_witnesses(
        image, state)
    # V3 measures carrier phase on the exact post-cartoon residual, never on
    # the cartoon itself.  ``centre`` is the present residual-law analogue;
    # using radiance here rewards smoothing and is a falsified placement.
    phase_numerator, phase_denominator = phase_covector_sufficient_statistics(
        centre)
    validation_centre = centre
    validation_radius = radius
    contractor_action, _initial_contractor = _phase_constrained_noise_action(
        image, state, validation_centre, validation_radius, None)
    lower = float(np.min(image))
    upper = float(np.max(image))
    scale = max(float(np.ptp(image)), float(np.max(np.abs(image))), 1.0)
    tolerance = (
        float(resolution.convergence_multiplier)
        * math.sqrt(np.finfo(float).eps) * scale
    )
    records: list[dict[str, Any]] = []
    equilibrium = False
    for iteration in range(ceiling):
        numerical = np.finfo(float).eps * scale * scale
        agreement = centre * centre / (centre * centre + variance + numerical)
        metric = continual_transport_metric(state, variance)
        _phase_laplacian, phase_transport, phase_flux_diagnostic = (
            _continual_flux_laplacian(metric, np.ones_like(state)))
        transported_phase_numerator, transported_phase_denominator = (
            _transport_phase_statistics(
                phase_transport, phase_numerator, phase_denominator))
        phase_authority, phase_diagnostic = phase_covector_noise_authority(
            transported_phase_numerator, transported_phase_denominator)
        laplacian, statistic_transport, flux_diagnostic = (
            _continual_flux_laplacian(metric, agreement * phase_authority))
        maximum_degree = flux_diagnostic["maximum_degree"]
        target = image - agreement * centre
        before = _frozen_action(state, target, laplacian)
        flat = state.ravel()
        gradient = flat - target.ravel() + laplacian @ flat
        # lambda_max(L) <= 2 max degree.  This is the exact scalar MM
        # majorizer of the frozen quadratic, analogous to the ISTA constraint
        # c > ||D^T D|| in the Back-to-Basics derivation.
        step = 1.0 / (1.0 + 2.0 * maximum_degree)
        candidate = np.clip(
            (flat - step * gradient).reshape(image.shape), lower, upper)
        after = _frozen_action(candidate, target, laplacian)
        slack = np.finfo(float).eps * max(before, 1.0) * image.size
        if after > before + slack:
            raise RuntimeError("eikonal MM step violated frozen-action descent")

        transported_centre = (
            statistic_transport @ centre.ravel()).reshape(image.shape)
        transported_second = (
            statistic_transport @ (variance + centre * centre).ravel()
        ).reshape(image.shape)
        transported_variance = np.maximum(
            transported_second - transported_centre * transported_centre, 0.0)
        transported_radius = (
            statistic_transport @ radius.ravel()).reshape(image.shape)
        witness_centre, witness_variance, witness_radius, witness_diag = (
            directional_noise_witnesses(image, candidate))
        witness_phase_numerator, witness_phase_denominator = (
            phase_covector_sufficient_statistics(witness_centre))
        validation_centre = witness_centre
        validation_radius = witness_radius
        candidate_contractor_action, contractor_diagnostic = (
            _phase_constrained_noise_action(
                image,
                candidate,
                validation_centre,
                validation_radius,
                phase_transport,
            ))
        next_centre, next_variance, next_radius = _mixture_moment_fusion(
            transported_centre,
            transported_variance,
            transported_radius,
            witness_centre,
            witness_variance,
            witness_radius,
            step,
        )
        next_phase_numerator = (
            (1.0 - step) * transported_phase_numerator
            + step * witness_phase_numerator
        )
        next_phase_denominator = (
            (1.0 - step) * transported_phase_denominator
            + step * witness_phase_denominator
        )
        state_change = float(np.max(np.abs(candidate - state)))
        statistic_change = max(
            float(np.max(np.abs(next_centre - centre))),
            float(np.max(np.abs(next_variance - variance))) / scale,
        )
        residual = image - candidate
        set_slack = np.finfo(float).eps * max(
            contractor_action, scale * scale, 1.0)
        accepted = candidate_contractor_action < contractor_action - set_slack
        record = {
            "iteration": iteration,
            "accepted": accepted,
            "frozen_action_before": before,
            "frozen_action_after": after,
            "noise_contractor_action_before": contractor_action,
            "noise_contractor_action_after": candidate_contractor_action,
            "majorizer_step": step,
            "maximum_state_change": state_change,
            "maximum_statistic_change": statistic_change,
            "mean_noise_centre": float(np.mean(next_centre)),
            "mean_noise_variance": float(np.mean(next_variance)),
            "mean_noise_radius": float(np.mean(next_radius)),
            "mean_transport_authority": float(np.mean(
                agreement * phase_authority)),
            "observation_identity_error": float(np.max(np.abs(
                image - (candidate + residual)))),
            "metric_condition_p90": metric["metric_condition_p90"],
            **flux_diagnostic,
            "phase_transport_row_sum_error": phase_flux_diagnostic[
                "transport_row_sum_error"],
            "phase_transport_column_sum_error": phase_flux_diagnostic[
                "transport_column_sum_error"],
            **phase_diagnostic,
            **{
                f"candidate_{key}": value
                for key, value in contractor_diagnostic.items()
            },
            **witness_diag,
        }
        records.append(record)
        if not accepted:
            equilibrium = True
            break
        state = candidate
        centre = next_centre
        variance = next_variance
        radius = next_radius
        phase_numerator = next_phase_numerator
        phase_denominator = next_phase_denominator
        contractor_action = candidate_contractor_action
        if max(state_change, statistic_change) <= tolerance:
            equilibrium = True
            break

    return np.clip(state, lower, upper), {
        "status": (
            "continual radiance/noise/statistic transport equilibrium"
            if equilibrium
            else "numerical iteration ceiling reached; equilibrium unresolved"
        ),
        "theory_status": (
            "first fused continual eikonal-noise checkpoint; not promoted"
        ),
        "accepted_iterations": int(sum(
            record["accepted"] for record in records)),
        "iteration_ceiling_hit": not equilibrium,
        "initial_witness": initial_witness,
        "final_noise_centre_mean": float(np.mean(centre)),
        "final_noise_variance_mean": float(np.mean(variance)),
        "final_noise_radius_mean": float(np.mean(radius)),
        "final_noise_contractor_action": contractor_action,
        "maximum_observation_identity_error": float(max(
            (record["observation_identity_error"] for record in records),
            default=0.0,
        )),
        "iterations": records,
        "numerical_resolution": asdict(resolution),
        "laws": {
            "descent": (
                "Gershgorin-majorized frozen convex action; no fitted step"
            ),
            "noise": (
                "paired target-excluded characteristic mixture; transported "
                "total variance and outer radius"
            ),
            "geometry": (
                "continual excess structure tensor; FM-LBR metric reduction; "
                "nonnegative symmetric Selling flux"
            ),
            "phase": (
                "transported paired-correlation sufficient statistics; noise "
                "authority is transverse defect from the full-band cosine "
                "covector manifold"
            ),
            "uncertainty": (
                "mixture moment transport, never repeated-data precision; "
                "only distance outside the paired-covector bounded set can "
                "reject the eikonal transport"
            ),
        },
        "unresolved": [
            "four paired covectors are a seed quadrature, not the circle limit",
            "shared-basis eikonal witnesses were rejected as self-confirming",
            "outer-radius transport is safe but may be overconservative",
            "the metric-reduced lattice basis changes discretely at cell boundaries",
            "colour distribution transport has not yet been lifted from luminance",
        ],
    }
