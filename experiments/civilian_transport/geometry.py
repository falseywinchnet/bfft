"""Finite rotation/integrated motion maps and invariant transition action."""
import numpy as np


def rotate_and_integrate(direction, omega, dt):
    """Return exp(dt [omega]x) u and integral_0^dt exp(t [omega]x) u dt.

    Broadcasting over arbitrary leading dimensions; stable at zero turn.
    This is a finite group action, not a truncated position derivative series.
    """
    u, w = np.broadcast_arrays(np.asarray(direction, float), np.asarray(omega, float))
    rate = np.linalg.norm(w, axis=-1, keepdims=True)
    theta = rate*dt
    axis = w/np.maximum(rate, 1e-100)
    parallel = axis*np.sum(axis*u, axis=-1, keepdims=True)
    normal = u-parallel
    transverse = np.cross(axis, u)
    sinc = np.sinc(theta/np.pi)
    # (1-cos(theta))/theta, evaluated without cancellation.
    cosc = .5*theta*np.sinc(theta/(2*np.pi))**2
    turned = parallel+np.cos(theta)*normal+np.sin(theta)*transverse
    displacement = dt*(parallel+sinc*normal+cosc*transverse)
    return turned, displacement


def action(old_log_speed, new_log_speed, old_omega, new_omega,
           heading_rotation, dt, speed_diffusion, turn_diffusion, heading_diffusion):
    """Negative-log Gaussian proposal action up to parameter-only constants.

    Resistance lives on relative log progression and rotational changes. A zero
    diffusion represents a hard constraint. This is a path-action prototype,
    not a derived Finsler eikonal equation or an inferred arbitrary action field.
    """
    result = 0.
    for delta, power in [(new_log_speed-old_log_speed, speed_diffusion),
                         (np.linalg.norm(new_omega-old_omega, axis=-1), turn_diffusion),
                         (np.linalg.norm(heading_rotation, axis=-1), heading_diffusion)]:
        if power == 0:
            result = result+np.where(np.abs(delta) < 1e-14, 0., np.inf)
        else:
            result = result+.5*np.square(delta)/(power**2*dt)
    return result


def unit_rows(x):
    return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-100)


def logsumexp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    result = m+np.log(np.exp(x-m).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def systematic(weights, n, rng):
    cdf = np.cumsum(weights)
    cdf[-1] = 1.
    return np.searchsorted(cdf, (rng.random()+np.arange(n))/n)
