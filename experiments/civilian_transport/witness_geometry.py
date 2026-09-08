"""Observation-supported local correction geometry, without a denoising loop.

The baseline and direction fields must be functions of training data only.
Witness errors must be independent of that training data, zero-mean Gaussian
with known isotropic sigma. These certificates concern summed clean squared
error at witness positions; they do not certify an unobserved continuation.
"""
import numpy as np
from scipy.stats import chi2, norm


def _inputs(baseline, directions, witnesses, observations, sigma):
    baseline = np.asarray(baseline, float)
    directions = np.asarray(directions, float)
    witnesses = np.asarray(witnesses, int)
    observations = np.asarray(observations, float)
    if (baseline.ndim != 2 or baseline.shape[1] != 3 or directions.ndim != 3
            or directions.shape[:2] != baseline.shape
            or observations.shape != (len(witnesses), 3)
            or len(witnesses) == 0 or len(np.unique(witnesses)) != len(witnesses)
            or not np.isfinite(baseline).all() or not np.isfinite(directions).all()
            or not np.isfinite(observations).all() or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError('Invalid field, direction, witness or noise specification')
    if np.any(witnesses < 0) or np.any(witnesses >= len(baseline)):
        raise ValueError('Witness outside query field')
    return baseline, directions, witnesses, observations


def supported_subspace(baseline, directions, witnesses, observations, sigma, alpha=.05):
    """Uniform confidence geometry over a training-defined correction subspace.

    D is the stacked witness direction matrix / sigma, D=U diag(s) V^T.
    v=U^T(y-f0)/sigma, c=sqrt(chi2_r(1-alpha)). The entire projected unknown
    clean residual is within c of v with probability 1-alpha, conditional on
    training. The correction z=(1-c/||v||)_+ v maximizes the worst-case clean
    risk improvement over that ball. Its lower gain is sigma^2(||v||-c)_+^2.
    The SVD is a finite coordinate decomposition, not iterative denoising.
    """
    baseline, directions, witnesses, observations = _inputs(
        baseline, directions, witnesses, observations, sigma)
    if not 0 < alpha < 1:
        raise ValueError('Require 0<alpha<1')
    count = directions.shape[-1]
    d = directions[witnesses].reshape(-1, count)/sigma
    u, s, vt = np.linalg.svd(d, full_matrices=True)
    tolerance = max(d.shape)*np.finfo(float).eps*(s[0] if len(s) else 0.)
    rank = int(np.count_nonzero(s > tolerance))
    ur = u[:, :rank]
    vr = vt[:rank].T
    singular = s[:rank]
    whitened = ((observations-baseline[witnesses])/sigma).reshape(-1)
    projected = ur.T @ whitened
    radius = float(np.sqrt(chi2.ppf(1-alpha, rank))) if rank else 0.
    magnitude = float(np.linalg.norm(projected))
    factor = max(0., 1-radius/magnitude) if magnitude else 0.
    # Map independent witness coordinates to the entire current query field.
    transport = np.einsum('qdk,kr->qdr', directions, vr/singular) if rank else np.zeros((*baseline.shape, 0))
    correction = np.einsum('qdr,r->qd', transport, factor*projected)
    coefficients = (vr/singular) @ (factor*projected) if rank else np.zeros(count)
    null = vt[rank:].T
    null_response = np.einsum('qdk,kr->qdr', directions, null)
    endpoint_amplification = float(np.linalg.norm(transport[-1], 2)/sigma) if rank else 0.
    return dict(mean=baseline+correction, correction=correction, coefficients=coefficients,
                rank=rank, witness_basis=ur, projected=projected, radius=radius,
                factor=factor, lower_gain=float(sigma*sigma*max(0., magnitude-radius)**2),
                certified=bool(magnitude>radius and rank>0), singular_values=singular,
                endpoint_amplification=endpoint_amplification,
                endpoint_null_response=float(np.linalg.norm(null_response[-1])),
                transport=transport, alpha=alpha)


def supported_ray(baseline, direction, witnesses, observations, sigma, alpha=.05):
    """A simultaneous bound for all nonnegative scales of one trained direction.

    Restrict the chosen scale to [0,1]: retain or shorten the proposed correction.
    This gives a finite conservative correction, with no iterative application.
    """
    baseline, ds, witnesses, observations = _inputs(
        baseline, np.asarray(direction)[..., None], witnesses, observations, sigma)
    if not 0 < alpha < 1:
        raise ValueError('Require 0<alpha<1')
    direction = ds[..., 0]
    dw = direction[witnesses]
    energy = float(np.sum(dw*dw))
    score = float(np.sum(dw*(observations-baseline[witnesses])))
    uncertainty = float(norm.ppf(1-alpha)*sigma*np.sqrt(energy))
    lower_slope = score-uncertainty
    factor = float(np.clip(lower_slope/energy, 0, 1)) if energy else 0.
    lower_gain = factor*(2*lower_slope-factor*energy)
    return dict(mean=baseline+factor*direction, correction=factor*direction, factor=factor,
                lower_gain=float(lower_gain), certified=bool(factor>0), energy=energy,
                score=score, uncertainty=uncertainty, alpha=alpha)


def clean_gain(baseline, corrected, truth, indices):
    """Terminal evaluation only. Never pass truth to either construction above."""
    return float(np.sum((truth[indices]-baseline[indices])**2)
                 -np.sum((truth[indices]-corrected[indices])**2))
