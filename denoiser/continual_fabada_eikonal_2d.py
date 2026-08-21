"""FABADA-order continual eikonal averaging experiment.

The primitive radiance operation here is a positive local average.  The
eikonal Laplacian is only its infinitesimal generator; it is not inserted as a
screened radiance solve.  A pointwise trapezoidal integral over the surviving
averaging trajectory is the readout.  Noise mixture moments, bounded radius,
phase sufficient statistics, and uncertainty about smoothing depth evolve on
the same physical time intervals.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal
import math

import numpy as np
from scipy import sparse

from .continual_eikonal_noise_transport_2d import (
    ContinualEikonalResolution,
    _bounded_noise_violation,
    continual_anisotropic_noise_metric,
    _continual_flux_laplacian,
    _mixture_moment_fusion,
    _phase_constrained_noise_action,
    _transport_phase_statistics,
    _validate_image,
    continual_transport_metric,
    directional_noise_witnesses,
    phase_covector_noise_authority,
    phase_covector_sufficient_statistics,
)


def _dirichlet_action(field: np.ndarray, laplacian: sparse.csr_matrix) -> float:
    flat = np.asarray(field, dtype=np.float64).ravel()
    return 0.5 * float(flat @ (laplacian @ flat))


def _candidate_survival_mass(
    observation: np.ndarray,
    candidate: np.ndarray,
    centre: np.ndarray,
    variance: np.ndarray,
    radius: np.ndarray,
    phase_transport: sparse.csr_matrix,
) -> tuple[np.ndarray, dict[str, float]]:
    """Contract a smoothing-depth mass only where noise feasibility fails."""
    residual = np.asarray(observation) - np.asarray(candidate)
    gap = np.maximum(np.abs(residual - centre) - radius, 0.0)
    numerator, denominator = phase_covector_sufficient_statistics(residual)
    numerator, denominator = _transport_phase_statistics(
        phase_transport, numerator, denominator)
    phase_noise, phase_diagnostic = phase_covector_noise_authority(
        numerator, denominator)
    coherent = 1.0 - phase_noise
    local_dirichlet = np.mean(denominator, axis=0)
    # Local value/jet Sasaki action.  The product is evaluated pointwise so a
    # coherent thin feature cannot hide inside image-global noise energy.
    coherent_action = (
        np.abs(residual) * np.sqrt(np.maximum(local_dirichlet, 0.0)) * coherent)
    violation = gap * gap + coherent_action
    magnitude = max(
        float(np.max(np.abs(observation))), float(np.ptp(observation)), 1.0)
    floor = np.finfo(float).eps * magnitude * magnitude
    capacity = centre * centre + variance + radius * radius + floor
    survival = capacity / (capacity + violation)
    return np.clip(survival, 0.0, 1.0), {
        "mean_candidate_survival": float(np.mean(survival)),
        "minimum_candidate_survival": float(np.min(survival)),
        "mean_candidate_set_gap_squared": float(np.mean(gap * gap)),
        "mean_candidate_coherent_action": float(np.mean(coherent_action)),
        **phase_diagnostic,
    }


def denoise_continual_fabada_eikonal_2d(
    observation: np.ndarray,
    resolution: ContinualEikonalResolution = ContinualEikonalResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Average along the surviving positive eikonal smoothing trajectory."""
    image = _validate_image(observation)
    ceiling = int(resolution.maximum_iterations)
    if ceiling < 1:
        raise ValueError("maximum_iterations must be positive")
    if resolution.convergence_multiplier <= 0.0:
        raise ValueError("convergence_multiplier must be positive")

    path = image.copy()
    readout = image.copy()
    centre, variance, radius, initial_witness = directional_noise_witnesses(
        image, readout)
    phase_numerator, phase_denominator = phase_covector_sufficient_statistics(
        centre)
    contractor_action, _ = _phase_constrained_noise_action(
        image, readout, centre, radius, None)

    # These are physical trajectory integrals, initially empty.  The first
    # accepted segment includes both y and A_0 y by trapezoidal quadrature.
    accumulated_value = np.zeros_like(image)
    accumulated_mass = np.zeros_like(image)
    path_survival = np.ones_like(image)
    identity_mass_initialized = False
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
        metric = continual_transport_metric(readout, variance)
        laplacian, _unused_transport, flux = _continual_flux_laplacian(
            metric, np.ones_like(image))
        maximum_degree = flux["maximum_degree"]
        if maximum_degree <= np.finfo(float).tiny:
            equilibrium = True
            break

        # lambda_max(L) <= 2*dmax.  This is the largest Back-to-Basics
        # gradient step certified by that majorizer.  It is also small enough
        # that A=I-dt*L is a positive, symmetric averaging operator.
        delta_time = 1.0 / (2.0 * maximum_degree)
        averaging = (
            sparse.eye(image.size, format="csr") - delta_time * laplacian)
        path_before = _dirichlet_action(path, laplacian)
        next_path = (averaging @ path.ravel()).reshape(image.shape)
        path_after = _dirichlet_action(next_path, laplacian)
        numerical = np.finfo(float).eps * max(path_before, 1.0) * image.size
        if path_after > path_before + numerical:
            raise RuntimeError("positive averaging step increased Dirichlet action")

        transported_centre = (
            averaging @ centre.ravel()).reshape(image.shape)
        transported_second = (
            averaging @ (variance + centre * centre).ravel()
        ).reshape(image.shape)
        transported_variance = np.maximum(
            transported_second - transported_centre * transported_centre, 0.0)
        transported_radius = (averaging @ radius.ravel()).reshape(image.shape)
        witness_centre, witness_variance, witness_radius, witness_diag = (
            directional_noise_witnesses(image, next_path))
        # Exact continuous-time fraction for one unit-rate incoming witness
        # law over this physical averaging interval.
        witness_fraction = -math.expm1(-delta_time)
        next_centre, next_variance, next_radius = _mixture_moment_fusion(
            transported_centre,
            transported_variance,
            transported_radius,
            witness_centre,
            witness_variance,
            witness_radius,
            witness_fraction,
        )

        transported_phase_numerator, transported_phase_denominator = (
            _transport_phase_statistics(
                averaging, phase_numerator, phase_denominator))
        current_phase_noise, _current_phase_diag = (
            phase_covector_noise_authority(
                transported_phase_numerator, transported_phase_denominator))
        current_agreement = centre * centre / np.maximum(
            centre * centre + variance,
            np.finfo(float).eps * scale * scale,
        )
        current_density = path_survival * current_agreement * current_phase_noise
        witness_phase_numerator, witness_phase_denominator = (
            phase_covector_sufficient_statistics(witness_centre))
        next_phase_numerator = (
            (1.0 - witness_fraction) * transported_phase_numerator
            + witness_fraction * witness_phase_numerator)
        next_phase_denominator = (
            (1.0 - witness_fraction) * transported_phase_denominator
            + witness_fraction * witness_phase_denominator)

        next_survival, survival_diag = _candidate_survival_mass(
            image,
            next_path,
            next_centre,
            next_variance,
            next_radius,
            averaging,
        )
        next_phase_noise, _next_phase_diag = phase_covector_noise_authority(
            next_phase_numerator, next_phase_denominator)
        next_agreement = next_centre * next_centre / np.maximum(
            next_centre * next_centre + next_variance,
            np.finfo(float).eps * scale * scale,
        )
        next_density = next_survival * next_agreement * next_phase_noise
        if not identity_mass_initialized:
            # The posterior complement is the explicit no-transport mode.  It
            # is entered once, not once per iteration, so elapsed numerical
            # depth cannot manufacture evidence for the observation.
            identity_mass = np.clip(1.0 - current_density, 0.0, 1.0)
            accumulated_mass = identity_mass.copy()
            accumulated_value = identity_mass * image
        segment_mass = 0.5 * delta_time * (
            current_density + next_density)
        segment_value = 0.5 * delta_time * (
            current_density * path + next_density * next_path)
        proposal_mass = accumulated_mass + segment_mass
        proposal_value = accumulated_value + segment_value
        proposal = np.divide(
            proposal_value,
            proposal_mass,
            out=readout.copy(),
            where=proposal_mass > np.finfo(float).tiny,
        )
        proposal = np.clip(proposal, lower, upper)
        validation_centre, _validation_variance, validation_radius, _ = (
            directional_noise_witnesses(image, proposal))
        proposal_contractor, contractor_diag = _phase_constrained_noise_action(
            image,
            proposal,
            validation_centre,
            validation_radius,
            averaging,
        )
        slack = np.finfo(float).eps * max(
            contractor_action, scale * scale, 1.0)
        accepted = proposal_contractor < contractor_action - slack
        path_change = float(np.max(np.abs(next_path - path)))
        readout_change = float(np.max(np.abs(proposal - readout)))
        records.append({
            "iteration": iteration,
            "accepted": accepted,
            "delta_time": delta_time,
            "path_dirichlet_before": path_before,
            "path_dirichlet_after": path_after,
            "noise_contractor_action_before": contractor_action,
            "noise_contractor_action_after": proposal_contractor,
            "maximum_path_change": path_change,
            "maximum_readout_change": readout_change,
            "mean_accumulated_mass_after": float(np.mean(proposal_mass)),
            "minimum_accumulated_mass_after": float(np.min(proposal_mass)),
            "averaging_row_sum_error": float(np.max(np.abs(
                np.asarray(averaging.sum(axis=1)).ravel() - 1.0))),
            "averaging_column_sum_error": float(np.max(np.abs(
                np.asarray(averaging.sum(axis=0)).ravel() - 1.0))),
            "averaging_minimum_diagonal": float(np.min(averaging.diagonal())),
            **flux,
            **survival_diag,
            **{
                f"candidate_{key}": value
                for key, value in contractor_diag.items()
            },
            **witness_diag,
        })
        if not accepted:
            equilibrium = True
            break

        path = next_path
        readout = proposal
        accumulated_mass = proposal_mass
        accumulated_value = proposal_value
        path_survival = next_survival
        identity_mass_initialized = True
        centre = next_centre
        variance = next_variance
        radius = next_radius
        phase_numerator = next_phase_numerator
        phase_denominator = next_phase_denominator
        contractor_action = proposal_contractor
        if max(path_change, readout_change) <= tolerance:
            equilibrium = True
            break

    return np.clip(readout, lower, upper), {
        "status": (
            "continual FABADA-order eikonal averaging equilibrium"
            if equilibrium
            else "numerical iteration ceiling reached; equilibrium unresolved"
        ),
        "theory_status": "pure-averaging hierarchy experiment; not promoted",
        "accepted_iterations": int(sum(row["accepted"] for row in records)),
        "evaluated_iterations": len(records),
        "iteration_ceiling_hit": not equilibrium,
        "final_noise_contractor_action": contractor_action,
        "final_mean_trajectory_mass": float(np.mean(accumulated_mass)),
        "maximum_observation_identity_error": float(np.max(np.abs(
            image - (readout + (image - readout))))),
        "initial_witness": initial_witness,
        "iterations": records,
        "numerical_resolution": asdict(resolution),
        "laws": {
            "radiance": "positive eikonal nearest-neighbour averaging only",
            "readout": (
                "pointwise trapezoidal integral over surviving smoothing depth"
            ),
            "statistics": (
                "same averaging pushforward plus total-mixture-variance fusion"
            ),
            "descent": (
                "operator-norm Dirichlet descent and joint amplitude/phase "
                "contractor"
            ),
        },
    }


def _zero_noise_mixture(
    centre: np.ndarray,
    variance: np.ndarray,
    radius: np.ndarray,
    noise_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix a bounded residual law with the exact zero-noise atom.

    This is a Bernoulli mixture, not a shrinkage prescription.  In particular,
    the variance contains the between-hypothesis term and therefore does not
    manufacture confidence when noise authority is uncertain.
    """
    probability = np.clip(np.asarray(noise_probability, dtype=np.float64), 0, 1)
    mean = probability * centre
    second = probability * (variance + centre * centre)
    mixture_variance = np.maximum(second - mean * mean, 0.0)
    mixture_radius = np.maximum(
        np.abs(mean),
        np.maximum(
            np.abs(centre - radius - mean),
            np.abs(centre + radius - mean),
        ),
    )
    return mean, mixture_variance, mixture_radius


def denoise_continual_residual_posterior_2d(
    observation: np.ndarray,
    resolution: ContinualEikonalResolution = ContinualEikonalResolution(),
    metric_noise_moment: Literal[
        "central", "complete", "anisotropic"
    ] = "central",
    contractor_mode: Literal["phase", "bounded"] = "phase",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer a transported residual law on the positive averaging path.

    ``path`` is evolved only by positive Selling/eikonal averaging.  Each
    depth proposes a Bernoulli mixture of the exact zero-noise atom and the
    transported residual witness law.  Those complete mixture moments are
    transported and fused in physical time.  The radiance readout is the
    operator-split state ``x = A_t u_t - E[delta N_t]``; it is neither a bare
    path barycenter nor a screened reaction-diffusion step.
    """
    image = _validate_image(observation)
    if metric_noise_moment not in {"central", "complete", "anisotropic"}:
        raise ValueError(
            "metric_noise_moment must be 'central', 'complete', or 'anisotropic'")
    if contractor_mode not in {"phase", "bounded"}:
        raise ValueError("contractor_mode must be 'phase' or 'bounded'")
    ceiling = int(resolution.maximum_iterations)
    if ceiling < 1:
        raise ValueError("maximum_iterations must be positive")
    if resolution.convergence_multiplier <= 0.0:
        raise ValueError("convergence_multiplier must be positive")

    path = image.copy()
    readout = image.copy()
    centre, variance, radius, initial_witness = directional_noise_witnesses(
        image, readout)
    phase_numerator, phase_denominator = phase_covector_sufficient_statistics(
        centre)
    # This law is the residual drift accumulated along physical averaging
    # time.  The exact zero law is therefore its initial condition; the first
    # witness enters over the first nonzero interval rather than being counted
    # both as an initial atom and an incoming law.
    posterior_centre = np.zeros_like(image)
    posterior_variance = np.zeros_like(image)
    posterior_radius = np.zeros_like(image)
    if contractor_mode == "phase":
        contractor_action, _ = _phase_constrained_noise_action(
            image, readout, centre, radius, None)
    else:
        contractor_action = _bounded_noise_violation(
            image, readout, centre, radius)
    lower = float(np.min(image))
    upper = float(np.max(image))
    scale = max(float(np.ptp(image)), float(np.max(np.abs(image))), 1.0)
    numerical = np.finfo(float).eps * scale * scale
    tolerance = (
        float(resolution.convergence_multiplier)
        * math.sqrt(np.finfo(float).eps) * scale
    )
    records: list[dict[str, Any]] = []
    equilibrium = False

    for iteration in range(ceiling):
        current_phase_noise, _current_phase_diagnostic = (
            phase_covector_noise_authority(
                phase_numerator, phase_denominator))
        metric_noise_variance = (
            variance
            if metric_noise_moment == "central"
            else variance + centre * centre)
        if metric_noise_moment == "anisotropic":
            metric = continual_anisotropic_noise_metric(
                readout, centre, variance, current_phase_noise)
        else:
            metric = continual_transport_metric(readout, metric_noise_variance)
        laplacian, _unused_transport, flux = _continual_flux_laplacian(
            metric, np.ones_like(image))
        maximum_degree = flux["maximum_degree"]
        if maximum_degree <= np.finfo(float).tiny:
            equilibrium = True
            break
        delta_time = 1.0 / (2.0 * maximum_degree)
        averaging = sparse.eye(image.size, format="csr") - delta_time * laplacian

        path_before = _dirichlet_action(path, laplacian)
        next_path = (averaging @ path.ravel()).reshape(image.shape)
        path_after = _dirichlet_action(next_path, laplacian)
        descent_slack = (
            np.finfo(float).eps * max(path_before, 1.0) * image.size)
        if path_after > path_before + descent_slack:
            raise RuntimeError("positive averaging step increased Dirichlet action")

        transported_centre = (averaging @ centre.ravel()).reshape(image.shape)
        transported_second = (
            averaging @ (variance + centre * centre).ravel()
        ).reshape(image.shape)
        transported_variance = np.maximum(
            transported_second - transported_centre * transported_centre, 0.0)
        transported_radius = (averaging @ radius.ravel()).reshape(image.shape)
        witness_centre, witness_variance, witness_radius, witness_diag = (
            directional_noise_witnesses(image, next_path))
        witness_fraction = -math.expm1(-delta_time)
        next_centre, next_variance, next_radius = _mixture_moment_fusion(
            transported_centre,
            transported_variance,
            transported_radius,
            witness_centre,
            witness_variance,
            witness_radius,
            witness_fraction,
        )

        transported_phase_numerator, transported_phase_denominator = (
            _transport_phase_statistics(
                averaging, phase_numerator, phase_denominator))
        witness_phase_numerator, witness_phase_denominator = (
            phase_covector_sufficient_statistics(witness_centre))
        next_phase_numerator = (
            (1.0 - witness_fraction) * transported_phase_numerator
            + witness_fraction * witness_phase_numerator)
        next_phase_denominator = (
            (1.0 - witness_fraction) * transported_phase_denominator
            + witness_fraction * witness_phase_denominator)
        phase_noise, phase_diag = phase_covector_noise_authority(
            next_phase_numerator, next_phase_denominator)
        agreement = next_centre * next_centre / (
            next_centre * next_centre + next_variance + numerical)
        # Agreement is the Bernoulli mass of the nonzero residual branch.
        # Phase is an independent structural falsifier in the contractor; it
        # must not shrink the same residual mean a second time.
        noise_probability = agreement
        branch_centre, branch_variance, branch_radius = _zero_noise_mixture(
            next_centre, next_variance, next_radius, noise_probability)

        transported_posterior_centre = (
            averaging @ posterior_centre.ravel()).reshape(image.shape)
        transported_posterior_second = (
            averaging @ (
                posterior_variance + posterior_centre * posterior_centre
            ).ravel()
        ).reshape(image.shape)
        transported_posterior_variance = np.maximum(
            transported_posterior_second
            - transported_posterior_centre * transported_posterior_centre,
            0.0,
        )
        transported_posterior_radius = (
            averaging @ posterior_radius.ravel()).reshape(image.shape)
        next_posterior_centre, next_posterior_variance, next_posterior_radius = (
            _mixture_moment_fusion(
                transported_posterior_centre,
                transported_posterior_variance,
                transported_posterior_radius,
                branch_centre,
                branch_variance,
                branch_radius,
                witness_fraction,
            ))

        # Lie splitting of geometric transport and residual inference.  The
        # only smoothing act is ``A_t path``; posterior noise mean is the
        # contemporaneous source term, not a second smoothing operator.
        proposal = np.clip(
            next_path - next_posterior_centre, lower, upper)
        validation_centre, _validation_variance, validation_radius, _ = (
            directional_noise_witnesses(image, proposal))
        if contractor_mode == "phase":
            proposal_contractor, contractor_diag = (
                _phase_constrained_noise_action(
                    image,
                    proposal,
                    validation_centre,
                    validation_radius,
                    averaging,
                ))
        else:
            proposal_contractor = _bounded_noise_violation(
                image, proposal, validation_centre, validation_radius)
            contractor_diag = {
                "bounded_set_violation": proposal_contractor,
                "coherent_residual_penalty": 0.0,
                "mean_phase_noise_authority": 0.0,
                "mean_phase_coherent_fraction": 0.0,
                "mean_phase_covector_defect": 0.0,
                "mean_phase_correlation_energy": 0.0,
            }
        action_slack = np.finfo(float).eps * max(
            contractor_action, scale * scale, 1.0)
        accepted = proposal_contractor < contractor_action - action_slack
        path_change = float(np.max(np.abs(next_path - path)))
        readout_change = float(np.max(np.abs(proposal - readout)))
        posterior_change = max(
            float(np.max(np.abs(
                next_posterior_centre - posterior_centre))),
            float(np.max(np.abs(
                next_posterior_variance - posterior_variance))) / scale,
        )
        records.append({
            "iteration": iteration,
            "accepted": accepted,
            "delta_time": delta_time,
            "path_dirichlet_before": path_before,
            "path_dirichlet_after": path_after,
            "noise_contractor_action_before": contractor_action,
            "noise_contractor_action_after": proposal_contractor,
            "maximum_path_change": path_change,
            "maximum_readout_change": readout_change,
            "maximum_posterior_change": posterior_change,
            "mean_noise_probability": float(np.mean(noise_probability)),
            "mean_metric_noise_second_moment": float(np.mean(
                metric_noise_variance)),
            "mean_posterior_noise_centre": float(np.mean(
                next_posterior_centre)),
            "mean_posterior_noise_variance": float(np.mean(
                next_posterior_variance)),
            "minimum_posterior_noise_variance": float(np.min(
                next_posterior_variance)),
            "posterior_identity_error": float(np.max(np.abs(
                image - (proposal + (image - proposal))))),
            "averaging_row_sum_error": float(np.max(np.abs(
                np.asarray(averaging.sum(axis=1)).ravel() - 1.0))),
            "averaging_column_sum_error": float(np.max(np.abs(
                np.asarray(averaging.sum(axis=0)).ravel() - 1.0))),
            "averaging_minimum_diagonal": float(np.min(averaging.diagonal())),
            **flux,
            **phase_diag,
            **{
                f"candidate_{key}": value
                for key, value in contractor_diag.items()
            },
            **witness_diag,
        })
        if not accepted:
            equilibrium = True
            break

        path = next_path
        readout = proposal
        centre = next_centre
        variance = next_variance
        radius = next_radius
        phase_numerator = next_phase_numerator
        phase_denominator = next_phase_denominator
        posterior_centre = next_posterior_centre
        posterior_variance = next_posterior_variance
        posterior_radius = next_posterior_radius
        contractor_action = proposal_contractor
        if max(path_change, readout_change, posterior_change) <= tolerance:
            equilibrium = True
            break

    return np.clip(readout, lower, upper), {
        "status": (
            "continual residual posterior equilibrium"
            if equilibrium
            else "numerical iteration ceiling reached; equilibrium unresolved"
        ),
        "theory_status": "transported residual-risk posterior experiment",
        "metric_noise_moment": metric_noise_moment,
        "contractor_mode": contractor_mode,
        "accepted_iterations": int(sum(row["accepted"] for row in records)),
        "evaluated_iterations": len(records),
        "iteration_ceiling_hit": not equilibrium,
        "final_noise_contractor_action": contractor_action,
        "final_posterior_noise_centre_mean": float(np.mean(posterior_centre)),
        "final_posterior_noise_variance_mean": float(np.mean(
            posterior_variance)),
        "maximum_observation_identity_error": float(max(
            (row["posterior_identity_error"] for row in records), default=0.0)),
        "initial_witness": initial_witness,
        "iterations": records,
        "numerical_resolution": asdict(resolution),
        "laws": {
            "transport": "positive symmetric Selling/eikonal averaging only",
            "metric_uncertainty": (
                "directional central dispersion"
                if metric_noise_moment == "central"
                else (
                    "complete directional residual second moment"
                    if metric_noise_moment == "complete"
                    else "phase-vetted anisotropic residual second moment"
                )
            ),
            "hypotheses": (
                "exact zero-noise atom plus transported residual mixture law"
            ),
            "posterior": (
                "complete mixture moments transported and fused in physical time"
            ),
            "readout": (
                "positive averaged path minus posterior residual-drift mean"
            ),
            "descent": (
                "joint amplitude/phase residual contractor"
                if contractor_mode == "phase"
                else "bounded residual-law contractor without phase ontology"
            ),
        },
    }


def denoise_complete_moment_residual_posterior_2d(
    observation: np.ndarray,
    resolution: ContinualEikonalResolution = ContinualEikonalResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Let the complete residual second moment shape the Selling metric."""
    estimate, diagnostic = denoise_continual_residual_posterior_2d(
        observation,
        resolution=resolution,
        metric_noise_moment="complete",
    )
    diagnostic["theory_status"] = (
        "complete-posterior-moment metric ablation; not promoted")
    return estimate, diagnostic


def denoise_anisotropic_moment_residual_posterior_2d(
    observation: np.ndarray,
    resolution: ContinualEikonalResolution = ContinualEikonalResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Let evolving residual correlation shape the Selling metric tensor."""
    estimate, diagnostic = denoise_continual_residual_posterior_2d(
        observation,
        resolution=resolution,
        metric_noise_moment="anisotropic",
    )
    diagnostic["theory_status"] = (
        "anisotropic posterior-moment metric ablation; not promoted")
    return estimate, diagnostic


def denoise_bounded_complete_moment_posterior_2d(
    observation: np.ndarray,
    resolution: ContinualEikonalResolution = ContinualEikonalResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use complete moment geometry with distribution bounds as sole veto."""
    estimate, diagnostic = denoise_continual_residual_posterior_2d(
        observation,
        resolution=resolution,
        metric_noise_moment="complete",
        contractor_mode="bounded",
    )
    diagnostic["theory_status"] = (
        "phase-free bounded-contractor ablation; not promoted")
    return estimate, diagnostic
