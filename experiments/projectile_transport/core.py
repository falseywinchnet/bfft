"""Small Gaussian-mixture EKF; all state/component cross covariances are kept.

This is an IMM approximation, not an exact nonlinear Bayesian filter or a
zonotopic enclosure. No observation is used to tune its own prior covariance.
"""
from dataclasses import dataclass
from itertools import product
import copy
import math

import numpy as np


GRAVITY = np.array([0., -9.81])
DIM = 10  # position, velocity, residual acceleration, jerk, camera bias
H = np.zeros((2, DIM))
H[:, :2] = np.eye(2)
H[:, 8:10] = np.eye(2)


@dataclass(frozen=True)
class Mode:
    drag: float
    sigma: float
    jerk_power: float


def moments(weights, means, covariances):
    mean = weights @ means
    delta = means - mean
    cov = np.einsum('i,ijk->jk', weights, covariances)
    cov += np.einsum('i,ij,ik->jk', weights, delta, delta)
    return mean, (cov + cov.T) * .5


def gaussian_logpdf(residual, covariance):
    chol = np.linalg.cholesky(covariance)
    white = np.linalg.solve(chol, residual)
    return -.5 * (len(residual)*math.log(2*math.pi)
                  + 2*np.log(np.diag(chol)).sum() + white @ white)


def logsumexp(values):
    maximum = np.max(values)
    return float(maximum + np.log(np.exp(values - maximum).sum()))


def flow(mean, mode, dt, augmented=True):
    """Second-order position transport with frozen drag and evolving residual jet."""
    mean = np.asarray(mean)
    velocity = mean[2:4]
    speed = np.linalg.norm(velocity)
    drag = -mode.drag * speed * velocity
    ddrag = -mode.drag * (speed*np.eye(2) + np.outer(velocity, velocity)
                         / max(speed, 1e-12))
    a = mean[4:6] if augmented else np.zeros(2)
    j = mean[6:8] if augmented else np.zeros(2)
    result = mean.copy()
    result[:2] += dt*velocity + .5*dt**2*(GRAVITY+drag+a) + dt**3*j/6
    result[2:4] += dt*(GRAVITY+drag+a) + .5*dt**2*j
    jac = np.eye(DIM)
    jac[:2, 2:4] = dt*np.eye(2) + .5*dt**2*ddrag
    jac[2:4, 2:4] += dt*ddrag
    if augmented:
        result[4:6] += dt*j
        result[6:8] *= math.exp(-dt/1.5)
        result[8:10] *= math.exp(-dt/3.)
        jac[:2, 4:6] = .5*dt**2*np.eye(2)
        jac[:2, 6:8] = dt**3/6*np.eye(2)
        jac[2:4, 4:6] = dt*np.eye(2)
        jac[2:4, 6:8] = .5*dt**2*np.eye(2)
        jac[4:6, 6:8] = dt*np.eye(2)
        jac[6:8, 6:8] *= math.exp(-dt/1.5)
        jac[8:10, 8:10] *= math.exp(-dt/3.)
    return result, jac


def process_covariance(mode, dt, augmented=True):
    q = np.zeros((DIM, DIM))
    if augmented:
        # Integrated white snap covariance of the undamped local jet. Jerk
        # mean reversion is in the drift; this PSD Q defines the discrete model.
        for i in range(4):
            for j in range(4):
                value = mode.jerk_power * dt**(7-i-j) / (
                    math.factorial(3-i)*math.factorial(3-j)*(7-i-j))
                q[2*i:2*i+2, 2*j:2*j+2] = value*np.eye(2)
        q[8:10, 8:10] = .12**2*3/2*(1-math.exp(-2*dt/3))*np.eye(2)
    else:
        for i in range(2):
            for j in range(2):
                q[2*i:2*i+2, 2*j:2*j+2] = (
                    mode.jerk_power*dt**(3-i-j)/(3-i-j))*np.eye(2)
    return q


class JointFilter:
    def __init__(self, modes, dt=.05, augmented=True, transition=None):
        self.modes = tuple(modes)
        self.dt = dt
        self.augmented = augmented
        n = len(modes)
        self.weights = np.ones(n)/n
        self.means = np.tile([0., 30., 12., 18., 0., 0., 0., 0., 0., 0.], (n, 1))
        diagonal = [1., 1., 9., 9., 4., 4., 4., 4., .25, .25]
        if not augmented:
            diagonal[4:] = [0.]*6
        self.covs = np.tile(np.diag(diagonal), (n, 1, 1))
        if transition is None:
            transition = np.ones((n, n))
            for i, left in enumerate(modes):
                for j, right in enumerate(modes):
                    transition[i, j] = np.prod([
                        1. if left.drag == right.drag else .0005,
                        1. if left.sigma == right.sigma else .015,
                        1. if left.jerk_power == right.jerk_power else .008])
            transition /= transition.sum(axis=1, keepdims=True)
        self.transition = np.asarray(transition, dtype=float)
        if self.transition.shape != (n, n) or np.any(self.transition < 0) or not np.allclose(self.transition.sum(axis=1), 1):
            raise ValueError('transition must be a row-stochastic mode matrix')
        self.last_noise = None

    def predict(self):
        joint = self.weights[:, None]*self.transition
        weights = joint.sum(axis=0)
        means, covs = [], []
        for j, mode in enumerate(self.modes):
            if weights[j] > 0:
                mean, cov = moments(joint[:, j]/weights[j], self.means, self.covs)
            else:
                mean, cov = self.means[j], self.covs[j]
            mean, jac = flow(mean, mode, self.dt, self.augmented)
            cov = jac @ cov @ jac.T + process_covariance(mode, self.dt, self.augmented)
            means.append(mean)
            covs.append((cov+cov.T)*.5)
        self.weights, self.means, self.covs = weights, np.array(means), np.array(covs)
        self.last_noise = None

    def update(self, observation):
        if observation is None:
            self.last_noise = None
            return None
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (2,) or not np.isfinite(observation).all():
            raise ValueError('observation must be a finite 2-vector, or None')
        logs, noises, noise_covs, state_noise_covs = [], [], [], []
        for i, mode in enumerate(self.modes):
            prior_mean, prior_cov = self.means[i].copy(), self.covs[i].copy()
            r = mode.sigma**2*np.eye(2)
            residual = observation-H @ prior_mean
            s = H @ prior_cov @ H.T+r
            logs.append(math.log(max(self.weights[i], 1e-300))+gaussian_logpdf(residual, s))
            gain = np.linalg.solve(s, H @ prior_cov).T
            self.means[i] = prior_mean+gain @ residual
            a = np.eye(DIM)-gain @ H
            self.covs[i] = a @ prior_cov @ a.T + gain @ r @ gain.T
            self.covs[i] = (self.covs[i]+self.covs[i].T)*.5
            noises.append(r @ np.linalg.solve(s, residual))
            noise_covs.append(r-r @ np.linalg.solve(s, r))
            state_noise_covs.append(-gain @ r)
        log_evidence = logsumexp(logs)
        self.weights = np.exp(np.array(logs)-log_evidence)
        # Preserve joint state/current-observation-noise posterior, including
        # negative cross covariance. Independent marginals would double count.
        joined_means = np.concatenate([self.means, np.array(noises)], axis=1)
        joined_covs = np.zeros((len(self.modes), DIM+2, DIM+2))
        joined_covs[:, :DIM, :DIM] = self.covs
        joined_covs[:, DIM:, DIM:] = noise_covs
        joined_covs[:, :DIM, DIM:] = state_noise_covs
        joined_covs[:, DIM:, :DIM] = np.array(state_noise_covs).transpose(0, 2, 1)
        self.last_noise = moments(self.weights, joined_means, joined_covs)
        return log_evidence

    def state(self):
        return moments(self.weights, self.means, self.covs)

    def forecast(self, steps):
        future = copy.deepcopy(self)
        for _ in range(steps):
            future.predict()
        return future

    def position_logpdf(self, truth):
        return logsumexp(np.array([
            math.log(max(w, 1e-300))+gaussian_logpdf(truth-m[:2], p[:2, :2])
            for w, m, p in zip(self.weights, self.means, self.covs)]))

    def components(self):
        """Moment transport of [drag acceleration, residual acceleration, bias]."""
        means, covs = [], []
        for mode, mean, cov in zip(self.modes, self.means, self.covs):
            v = mean[2:4]
            speed = np.linalg.norm(v)
            jac = np.zeros((6, DIM))
            jac[:2, 2:4] = -mode.drag*(speed*np.eye(2)+np.outer(v, v)/max(speed, 1e-12))
            jac[2:4, 4:6] = np.eye(2)
            jac[4:6, 8:10] = np.eye(2)
            means.append(np.r_[-mode.drag*speed*v, mean[4:6], mean[8:10]])
            covs.append(jac @ cov @ jac.T)
        return moments(self.weights, np.array(means), np.array(covs))


def make_filter(name, dt=.05):
    if name == 'ballistic_kf':
        return JointFilter([Mode(0., .4, 2.)], dt, augmented=False)
    if name == 'fixed_joint_ekf':
        return JointFilter([Mode(.012, .4, .5)], dt)
    drags = [0., .012, .024] if name != 'noise_only' else [.012]
    sigmas = [.4, 2.] if name != 'motion_only' else [.4]
    if name not in ('joint_transport', 'motion_only', 'noise_only'):
        raise ValueError(name)
    powers = [.5, 8.] if name != 'noise_only' else [.5]
    return JointFilter([Mode(*v) for v in product(drags, sigmas, powers)], dt)
