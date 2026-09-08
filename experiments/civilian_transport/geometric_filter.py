"""Joint distribution of geometric continuations and observation-noise laws.

    nonlinear support: direction on S2, positive progression, rotation generator
    conditional Gaussian: position and persistent observation bias
    stochastic noise law: log sensor scale

All observations are 3-D position only. There is no trajectory library,
ground-truth initialization, measured attitude, or derivative preprocessing.
"""
import copy
import numpy as np

from .geometry import rotate_and_integrate, unit_rows, logsumexp, systematic


class GeometricFilter:
    def __init__(self, observation, sigma=.35, turn_diffusion=.35,
                 speed_diffusion=.16, heading_diffusion=.035, particles=1024,
                 seed=0, adaptive_noise=True):
        self.rng = np.random.default_rng(seed)
        self.n = particles
        self.capacity = particles
        self.sigma = sigma
        self.turn_diffusion = turn_diffusion
        self.speed_diffusion = speed_diffusion
        self.heading_diffusion = heading_diffusion
        self.adaptive_noise = adaptive_noise
        self.u = unit_rows(self.rng.normal(size=(particles, 3)))
        self.log_speed = self.rng.normal(np.log(2.), .7, particles)
        self.omega = self.rng.normal(0., .6, (particles, 3))
        self.log_sigma = np.full(particles, np.log(sigma))
        self.weights = np.ones(particles)/particles
        # Isotropic conditional blocks, with full p-b dependence. Initialization
        # conditions once on y0 under a flat position prior and b~N(0,.25I).
        self.mean = np.zeros((particles, 2, 3))
        self.mean[:, 0] = observation
        self.cov = np.tile([[sigma*sigma+.25, -.25], [-.25, .25]], (particles, 1, 1))
        self.last_noise = None
        self.last_log_evidence = None
        self.resamples = 0

    def _advance(self, dt, rng, random=True):
        if random:
            self.log_speed += self.speed_diffusion*np.sqrt(dt)*rng.normal(size=self.n)
            self.omega += self.turn_diffusion*np.sqrt(dt)*rng.normal(size=(self.n, 3))
            wobble = self.heading_diffusion*np.sqrt(dt)*rng.normal(size=(self.n, 3))
            self.u = rotate_and_integrate(self.u, wobble, 1.)[0]
            if self.adaptive_noise:
                self.log_sigma += -.1*dt*(self.log_sigma-np.log(self.sigma))
                self.log_sigma += .5*np.sqrt(dt)*rng.normal(size=self.n)
        # Explicit numerical support bounds, far outside nominal civilian speeds.
        self.log_speed = np.clip(self.log_speed, np.log(.015), np.log(30.))
        self.log_sigma = np.clip(self.log_sigma, np.log(.03), np.log(8.))
        self.u, integral = rotate_and_integrate(self.u, self.omega, dt)
        self.mean[:, 0] += np.exp(self.log_speed)[:, None]*integral
        rho = np.exp(-dt/6.)
        self.mean[:, 1] *= rho
        self.cov[:, 0, 1] *= rho
        self.cov[:, 1, 0] *= rho
        self.cov[:, 1, 1] *= rho*rho
        self.cov[:, 0, 0] += .04**2*dt
        self.cov[:, 1, 1] += .06**2*3*(1-rho*rho)
        self.last_noise = None
        self.last_log_evidence = None

    def predict(self, dt):
        if dt <= 0:
            raise ValueError('elapsed time must be positive')
        # Resample only before a new prediction. The reported posterior remains
        # weighted; forecast draws never modify filtering ancestry or its RNG.
        if self.n > self.capacity or 1/np.sum(self.weights**2) < .65*self.n:
            indices = systematic(self.weights, self.capacity, self.rng)
            for name in ('u', 'log_speed', 'omega', 'log_sigma', 'mean', 'cov'):
                setattr(self, name, getattr(self, name)[indices].copy())
            self.n = self.capacity
            self.weights = np.ones(self.n)/self.n
            self.resamples += 1
        # At most 0.1 s action increments, also used during observation gaps.
        count = max(1, int(np.ceil(dt/.1-1e-10)))
        for _ in range(count):
            self._advance(dt/count, self.rng)

    def update(self, observation):
        if observation is None:
            self.last_noise = None
            return None
        y = np.asarray(observation, float)
        if y.shape != (3,) or not np.isfinite(y).all():
            raise ValueError('expected a finite 3-D observation or None')
        residual = y-self.mean.sum(axis=1)
        base = np.exp(2*self.log_sigma)
        # Same contamination likelihood used in the robust conventional baselines.
        variances = base[:, None]*np.array([1., 100.])[None, :]
        innovation_var = self.cov.sum(axis=(1, 2))[:, None]+variances
        loglike = -.5*(3*np.log(2*np.pi*innovation_var)
                       +np.sum(residual**2, axis=1)[:, None]/innovation_var)
        loglike += np.log([.97, .03])[None, :]
        marginal = logsumexp(loglike, axis=1)
        hcov = self.cov.sum(axis=2)
        gain = hcov[:, None, :]/innovation_var[:, :, None]
        means = self.mean[:, None, :, :]+gain[:, :, :, None]*residual[:, None, None, :]
        covs = self.cov[:, None]-hcov[:, None, :, None]*gain[:, :, None, :]
        total_logs = np.log(np.maximum(self.weights, 1e-300))[:, None]+loglike
        evidence = float(logsumexp(total_logs))
        self.weights = np.exp(total_logs-evidence).reshape(-1)
        noise_means = (variances/innovation_var)[:, :, None]*residual[:, None, :]
        self.last_noise = self.weights @ noise_means.reshape(-1, 3)
        # Retain both observation-noise hypotheses. There is no covariance or
        # Gaussian collapse of these branches; Monte Carlo resampling happens
        # before the next transition, and its budget is explicitly reported.
        self.mean = means.reshape(-1, 2, 3)
        self.cov = covs.reshape(-1, 2, 2)
        self.cov = (self.cov+self.cov.transpose(0, 2, 1))*.5
        for name in ('u', 'log_speed', 'omega', 'log_sigma'):
            setattr(self, name, np.repeat(getattr(self, name), 2, axis=0))
        self.n *= 2
        self.last_log_evidence = evidence
        return evidence

    def state(self):
        p = self.mean[:, 0]
        center = self.weights @ p
        delta = p-center
        covariance = np.einsum('n,ni,nj->ij', self.weights, delta, delta)
        covariance += np.eye(3)*(self.weights @ self.cov[:, 0, 0])
        return center, covariance

    def diagnostics(self):
        return {'bias': self.weights @ self.mean[:, 1],
                'noise': self.last_noise,
                'sigma2': float(self.weights @ np.exp(2*self.log_sigma)),
                'ess': float(1/np.sum(self.weights**2)),
                'direction_resultant': float(np.linalg.norm(self.weights @ self.u)),
                'turn_mean': self.weights @ self.omega}

    def forecast_paths(self, intervals, samples=256, seed=0, ablation=None):
        rng = np.random.default_rng(seed)
        indices = systematic(self.weights, samples, rng)
        model = copy.copy(self)
        model.n = samples
        for name in ('u', 'log_speed', 'omega', 'log_sigma', 'mean', 'cov'):
            setattr(model, name, getattr(self, name)[indices].copy())
        if ablation == 'uncoupled':
            # Preserve each marginal but erase p/direction/progression/turn coupling.
            for name in ('u', 'log_speed', 'omega'):
                value = getattr(model, name)
                setattr(model, name, value[rng.permutation(samples)])
        if ablation == 'straight':
            model.omega[:] = 0.
            model.turn_diffusion = 0.
        chol = np.linalg.cholesky(model.cov+np.eye(2)[None]*1e-12)
        errors = np.einsum('nij,nja->nia', chol, rng.normal(size=(samples, 2, 3)))
        model.mean += errors
        path = [model.mean[:, 0].copy()]
        for dt in intervals:
            count = max(1, int(np.ceil(dt/.1-1e-10)))
            for _ in range(count):
                model._advance(dt/count, rng)
                model.mean[:, 0] += .04*np.sqrt(dt/count)*rng.normal(size=(samples, 3))
            path.append(model.mean[:, 0].copy())
        return np.stack(path, axis=1)
