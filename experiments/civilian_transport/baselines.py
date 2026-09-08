"""Auditable CV/CA Kalman, CV/CA/turn IMM, and continuous-turn UKF baselines.

Literature-family implementations, not claimed replicas of a particular recent
paper's reported scores. All initialize and update from position only.
"""
import copy
import math
import numpy as np
from .geometry import rotate_and_integrate, logsumexp, systematic


D = 12
H = np.zeros((3, D))
H[:, :3] = np.eye(3)
H[:, 9:] = np.eye(3)


def mixture_moments(weights, means, covs):
    mean = weights @ means
    delta = means-mean
    cov = np.einsum('n,nij->ij', weights, covs)+np.einsum('n,ni,nj->ij', weights, delta, delta)
    return mean, (cov+cov.T)*.5


def measurement(mean, covariance, y, sigma, robust):
    residual = y-H @ mean
    hp = H @ covariance
    components = [(1., 1.)] if not robust else [(.97, 1.), (.03, 100.)]
    logs, means, covs, noises = [], [], [], []
    for prob, scale in components:
        r = sigma**2*scale*np.eye(3)
        s = hp @ H.T+r
        chol = np.linalg.cholesky(s)
        white = np.linalg.solve(chol, residual)
        logs.append(np.log(prob)-.5*(3*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+white @ white))
        k = np.linalg.solve(s, hp).T
        means.append(mean+k @ residual)
        a = np.eye(D)-k @ H
        covs.append(a @ covariance @ a.T+k @ r @ k.T)
        noises.append(r @ np.linalg.solve(s, residual))
    evidence = float(logsumexp(np.array(logs)))
    weights = np.exp(np.array(logs)-evidence)
    mean, cov = mixture_moments(weights, np.array(means), np.array(covs))
    return mean, cov, evidence, weights @ np.array(noises)


def linear_model(kind, dt, q):
    f = np.eye(D)
    process = np.zeros((D, D))
    if isinstance(kind, tuple):
        omega = np.array(kind)
        rotated, integrated = rotate_and_integrate(np.eye(3), omega, dt)
        f[:3, 3:6] = integrated.T
        f[3:6, 3:6] = rotated.T
        f[6:9, 6:9] = 0.
        order = 2
    elif kind == 'ca':
        f[:3, 3:6] = dt*np.eye(3)
        f[:3, 6:9] = dt*dt/2*np.eye(3)
        f[3:6, 6:9] = dt*np.eye(3)
        order = 3
    else:
        f[:3, 3:6] = dt*np.eye(3)
        f[6:9, 6:9] = 0.
        order = 2
    for i in range(order):
        for j in range(order):
            exponent = 2*order-1-i-j
            process[3*i:3*i+3, 3*j:3*j+3] = (q*dt**exponent / (
                math.factorial(order-1-i)*math.factorial(order-1-j)*exponent))*np.eye(3)
    rho = np.exp(-dt/6.)
    f[9:, 9:] = rho*np.eye(3)
    process[9:, 9:] = .06**2*3*(1-rho*rho)*np.eye(3)
    process[:3, :3] += .04**2*dt*np.eye(3)
    return f, process


class ConventionalFilter:
    def __init__(self, observation, method='imm', sigma=.35, q=1.,
                 turn_diffusion=.35, speed_diffusion=.16):
        self.method = method
        self.sigma, self.q = sigma, q
        self.turn_diffusion, self.speed_diffusion = turn_diffusion, speed_diffusion
        self.robust = method in ('robust_imm', 'turn_ukf')
        if method in ('imm', 'robust_imm'):
            self.modes = ['cv', 'ca']+[tuple(sign*.65*axis) for axis in np.eye(3) for sign in (-1, 1)]
        else:
            self.modes = ['ca' if method == 'ca' else 'cv']
        n = len(self.modes)
        self.weights = np.ones(n)/n
        self.means = np.zeros((n, D))
        self.means[:, :3] = observation
        self.covs = np.tile(np.eye(D), (n, 1, 1))
        self.covs[:, :3, :3] *= sigma*sigma+.25
        self.covs[:, 3:6, 3:6] *= 4*np.exp(2*.7**2)/3
        self.covs[:, 6:9, 6:9] *= .36 if method == 'turn_ukf' else 4.
        self.covs[:, 9:, 9:] *= .25
        self.covs[:, :3, 9:] = -.25*np.eye(3)
        self.covs[:, 9:, :3] = -.25*np.eye(3)
        self.last_noise = None

    def transition(self, dt):
        n = len(self.modes)
        return np.exp(-dt/3)*np.eye(n)+(1-np.exp(-dt/3))*np.ones((n, n))/n

    def _turn_flow(self, states, dt):
        values = np.array(states, copy=True)
        velocity, increment = rotate_and_integrate(values[..., 3:6], values[..., 6:9], dt)
        values[..., :3] += increment
        values[..., 3:6] = velocity
        values[..., 9:] *= np.exp(-dt/6)
        return values

    def _turn_q(self, state, dt):
        q = np.zeros((D, D))
        v = state[3:6]
        speed = np.linalg.norm(v)
        u = v/max(speed, 1e-12)
        q[3:6, 3:6] = speed**2*dt*(self.speed_diffusion**2*np.outer(u, u)
                                      +.035**2*(np.eye(3)-np.outer(u, u)))
        q[6:9, 6:9] = self.turn_diffusion**2*dt*np.eye(3)
        return q

    def _ukf_predict(self, dt):
        mean, cov = self.means[0], self.covs[0]
        cov = cov+self._turn_q(mean, dt)
        chol = np.linalg.cholesky(cov+np.eye(D)*1e-10)
        points = np.vstack([mean, mean+np.sqrt(D)*chol.T, mean-np.sqrt(D)*chol.T])
        values = self._turn_flow(points, dt)
        wm = np.r_[0., np.full(2*D, 1/(2*D))]
        wc = wm.copy()
        wc[0] = 2.  # alpha=1, beta=2, kappa=0 scaled unscented transform
        mean = wm @ values
        delta = values-mean
        cov = np.einsum('n,ni,nj->ij', wc, delta, delta)
        cov[:3, :3] += .04**2*dt*np.eye(3)
        cov[9:, 9:] += .06**2*3*(1-np.exp(-dt/3))*np.eye(3)
        self.means[0], self.covs[0] = mean, (cov+cov.T)*.5

    def predict(self, dt):
        if dt <= 0:
            raise ValueError('elapsed time must be positive')
        count = max(1, int(np.ceil(dt/.1-1e-10)))
        for _ in range(count):
            step = dt/count
            if self.method == 'turn_ukf':
                self._ukf_predict(step)
            else:
                joint = self.weights[:, None]*self.transition(step)
                weights = joint.sum(axis=0)
                means, covs = [], []
                for index, kind in enumerate(self.modes):
                    mean, cov = mixture_moments(joint[:, index]/weights[index], self.means, self.covs)
                    f, q = linear_model(kind, step, self.q)
                    means.append(f @ mean)
                    covs.append(f @ cov @ f.T+q)
                self.weights, self.means, self.covs = weights, np.array(means), np.array(covs)
        self.last_noise = None

    def update(self, y):
        if y is None:
            self.last_noise = None
            return None
        logs, noises = [], []
        for i in range(len(self.modes)):
            mean, cov, evidence, noise = measurement(self.means[i], self.covs[i], np.asarray(y), self.sigma, self.robust)
            self.means[i], self.covs[i] = mean, cov
            logs.append(np.log(max(self.weights[i], 1e-300))+evidence)
            noises.append(noise)
        evidence = float(logsumexp(np.array(logs)))
        self.weights = np.exp(np.array(logs)-evidence)
        self.last_noise = self.weights @ np.array(noises)
        return evidence

    def state(self):
        mean, cov = mixture_moments(self.weights, self.means, self.covs)
        return mean[:3], cov[:3, :3]

    def diagnostics(self):
        return {'bias': self.weights @ self.means[:, 9:], 'noise': self.last_noise,
                'sigma2': self.sigma**2}

    def forecast_paths(self, intervals, samples=256, seed=0, ablation=None):
        rng = np.random.default_rng(seed)
        indices = systematic(self.weights, samples, rng)
        chol = np.linalg.cholesky(self.covs[indices]+np.eye(D)[None]*1e-10)
        states = self.means[indices]+np.einsum('nij,nj->ni', chol, rng.normal(size=(samples, D)))
        path = [states[:, :3].copy()]
        for dt in intervals:
            count = max(1, int(np.ceil(dt/.1-1e-10)))
            for _ in range(count):
                step = dt/count
                if self.method == 'turn_ukf':
                    v = states[:, 3:6]
                    u = v/np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                    eta = rng.normal(size=(samples, 3))
                    parallel = u*np.sum(u*eta, axis=1, keepdims=True)
                    states[:, 3:6] += np.linalg.norm(v, axis=1)[:, None]*np.sqrt(step)*(
                        self.speed_diffusion*parallel+.035*(eta-parallel))
                    states[:, 6:9] += self.turn_diffusion*np.sqrt(step)*rng.normal(size=(samples, 3))
                    states = self._turn_flow(states, step)
                    states[:, :3] += .04*np.sqrt(step)*rng.normal(size=(samples, 3))
                else:
                    transition = self.transition(step)
                    indices = np.sum(rng.random(samples)[:, None]>np.cumsum(transition[indices], axis=1), axis=1)
                    for index, kind in enumerate(self.modes):
                        mask = indices == index
                        n = np.count_nonzero(mask)
                        if n:
                            f, q = linear_model(kind, step, self.q)
                            cholq = np.linalg.cholesky(q+np.eye(D)*1e-12)
                            states[mask] = states[mask] @ f.T+rng.normal(size=(n, D)) @ cholq.T
            path.append(states[:, :3].copy())
        return np.stack(path, axis=1)
