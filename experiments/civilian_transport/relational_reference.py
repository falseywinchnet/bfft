"""Finite L2 currents, Gaussian relational priors, and exact evidence accounting.

A declared Matérn-3/2 current hypothesis; no claim that Zak/BTB imply this law.
The spatial prior is isotropic. Each component has a full temporal covariance;
the hyperparameter mixture retains between-component spatial covariance.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

LENGTHS = (.3, .7, 1.5, 3., 6.)


def design(times, end=8., cells=64):
    """Exact integration of orthonormal piecewise-constant current functions."""
    times = np.asarray(times, float)
    if np.any(times < 0) or np.any(times > end):
        raise ValueError('Query outside represented interval')
    h = end/cells
    return np.c_[np.ones(len(times)),
                 np.clip(times[:, None]-np.arange(cells)*h, 0, h)/np.sqrt(h)]


def prior(length, sigma, end=8., cells=64, sever_at=None):
    """Cellwise current values have variance 4+4 (m/s)^2.

    Midpoint covariance defines this finite model exactly; increasing resolution
    approximates a continuous Matérn current, rather than changing time units.
    """
    h = end/cells
    t = (np.arange(cells)+.5)*h
    d = np.sqrt(3)*np.abs(t[:, None]-t)/length
    kernel = 4.+4.*(1+d)*np.exp(-d)
    if sever_at is not None:
        past = t < sever_at
        kernel = kernel * (past[:, None] == past[None, :])
    p = np.zeros((cells+1, cells+1))
    p[0, 0] = sigma*sigma
    p[1:, 1:] = h*kernel
    return p


def zak(cells=64, rows=8):
    """Unitary analysis matrix; full complex covariance stays conjugacy-aware."""
    if cells % rows:
        raise ValueError('Lattice must factor coefficient dimension')
    cols = cells//rows
    u = np.zeros((cells, cells), complex)
    for delta in range(rows):
        for r in range(cols):
            for m in range(rows):
                u[delta*cols+r, r+m*cols] = np.exp(-2j*np.pi*delta*m/rows)/np.sqrt(rows)
    full = np.eye(cells+1, dtype=complex)
    full[1:, 1:] = u
    return full


def condition(mean, covariance, b, y, variance):
    """Real measurements on a possibly conjugate-constrained complex state.

    The measurement covariance is real because b maps the state back to real
    positions. No proper-complex Gaussian likelihood is asserted or used.
    """
    pb = covariance @ b.conj().T
    s = (b @ pb).real + np.diag(np.broadcast_to(variance, len(b)))
    chol = cho_factor(s, lower=True)
    innovation = y-(b @ mean).real
    k = cho_solve(chol, pb.conj().T).conj().T
    new_mean = mean+k @ innovation
    new_cov = covariance-k @ pb.conj().T
    new_cov = (new_cov+new_cov.conj().T)/2
    loglike = -.5*(3*(len(b)*np.log(2*np.pi)+2*np.log(np.diag(chol[0])).sum())
                      +np.sum(innovation*cho_solve(chol, innovation)))
    return new_mean, new_cov, float(loglike)


def component(times, observations, sigma, queries, length, cells=64,
              coordinates='zak', beta_steps=1, sever_at=None):
    times, observations = np.asarray(times), np.asarray(observations)
    if len(times)<2 or times[0]!=0 or np.any(np.diff(times)<=0):
        raise ValueError('Need increasing history beginning at zero')
    if not np.isfinite(observations).all() or sigma<=0 or beta_steps<1:
        raise ValueError('Require finite observations, positive noise and steps')
    end = max(8., float(np.max(queries)), float(times[-1]))
    b = design(times[1:], end, cells)
    f = design(queries, end, cells)
    p = prior(length, sigma, end, cells, sever_at)
    m = np.zeros((cells+1, 3))
    # Flat absolute anchor prior: condition on y0 once, then use independent
    # later sensor errors. Work relative to y0 for exact translation equivariance.
    y = observations[1:]-observations[0]
    if coordinates == 'zak':
        u = zak(cells)
        m, p, b, f = u @ m, u @ p @ u.conj().T, b @ u.conj().T, f @ u.conj().T
    elif coordinates != 'current':
        raise ValueError('Unknown coordinate representation')
    # True marginal likelihood always computed once under the original prior.
    exact_m, exact_p, loglike = condition(m, p, b, y, sigma*sigma)
    if beta_steps == 1:
        m, p = exact_m, exact_p
    else:
        for _ in range(beta_steps):
            m, p, _ = condition(m, p, b, y, sigma*sigma*beta_steps)
    means = (f @ m).real+observations[0]
    time_cov = (f @ p @ f.conj().T).real
    return means, (time_cov+time_cov.T)/2, loglike


def forecast(times, observations, sigma, queries, cells=64, coordinates='zak',
             beta_steps=1, lengths=LENGTHS, sever_at=None):
    records = [component(times, observations, sigma, queries, length, cells,
                         coordinates, beta_steps, sever_at) for length in lengths]
    means = np.array([r[0] for r in records])
    variances = np.array([np.diag(r[1]) for r in records])
    logs = np.array([r[2] for r in records])
    weights = np.exp(logs-logsumexp(logs))  # fixed uniform hyperprior
    mean = np.einsum('k,ktd->td', weights, means)
    delta = means-mean
    covariance = (np.einsum('k,kt,ij->tij', weights, variances, np.eye(3))
                  +np.einsum('k,kti,ktj->tij', weights, delta, delta))
    return {'mean': mean, 'covariance': covariance, 'weights': weights,
            'component_mean': means, 'component_variance': variances,
            'log_evidence': float(logsumexp(logs)-np.log(len(logs)))}


def btb_proximal_check(times, observations, sigma, cells=64, length=1.5, steps=2000):
    """Actual relaxed proximal iteration for ONE fixed Gaussian posterior.

    Whiten current+anchor prior; D_tau(v)=v/(1+tau). This is a concrete Gaussian
    BTB-compatible F, not a claim of Pereg's nonlinear denoising mechanism.
    Eigen-coordinates only accelerate repeated matrix applications; no direct
    posterior answer initializes the iteration. Return a convergence diagnostic.
    """
    b = design(np.asarray(times)[1:], cells=cells)
    p = prior(length, sigma, cells=cells)
    root = np.linalg.cholesky(p)
    w = b @ root/sigma
    d = (observations[1:]-observations[0])/sigma
    eig, v = np.linalg.eigh(w.T @ w)
    eig = np.maximum(eig, 0)
    rhs = v.T @ w.T @ d
    tau = 1/max(float(eig[-1]), 1.)
    mu = .8
    x = np.zeros_like(rhs)
    for _ in range(steps):
        denoised = (x-tau*(eig[:, None]*x-rhs))/(1+tau)
        x = (1-mu)*x+mu*denoised
    residual = (1+eig[:, None])*x-rhs
    exact = rhs/(1+eig[:, None])
    return {'steps': steps, 'relative_residual': float(np.linalg.norm(residual)/np.linalg.norm(rhs)),
            'relative_mean_error': float(np.linalg.norm(x-exact)/np.linalg.norm(exact)),
            'worst_contraction': float(1-mu*tau/(1+tau))}


class RelationalKalman:
    """Streaming belief over a fixed current interval, retaining all covariance.

    The known representation horizon is not an observed future. Queries inside
    it can smooth history or forecast future; extending it needs an explicit
    prior extension and is deliberately not performed by cyclic Zak wrapping.
    """
    def __init__(self, observation, sigma=.35, end=8., cells=64,
                 lengths=LENGTHS, coordinates='zak', sever_at=None):
        self.anchor = np.asarray(observation, float).copy()
        if self.anchor.shape != (3,) or not np.isfinite(self.anchor).all() or sigma<=0:
            raise ValueError('Require a finite 3-D observation and positive sigma')
        self.sigma, self.end, self.cells = sigma, end, cells
        self.lengths = tuple(lengths)
        self.u = zak(cells) if coordinates=='zak' else np.eye(cells+1)
        if coordinates not in ('zak','current') or not self.lengths:
            raise ValueError('Invalid representation or prior')
        self.means = [self.u @ np.zeros((cells+1,3)) for _ in self.lengths]
        self.covs = [self.u @ prior(l,sigma,end,cells,sever_at) @ self.u.conj().T for l in self.lengths]
        self.logs = np.full(len(self.lengths), -np.log(len(self.lengths)))
        self.last_time = 0.

    def update(self, time, observation, beta_steps=1):
        y = np.asarray(observation, float)
        if not self.last_time < time <= self.end or y.shape!=(3,) or not np.isfinite(y).all():
            raise ValueError('Require new increasing observation within interval')
        if not isinstance(beta_steps, int) or beta_steps<1:
            raise ValueError('Positive integer refinement count required')
        b = design([time], self.end, self.cells) @ self.u.conj().T
        for i in range(len(self.lengths)):
            m,p,loglike = condition(self.means[i],self.covs[i],b,(y-self.anchor)[None],self.sigma**2)
            self.logs[i] += loglike
            if beta_steps>1:
                m,p=self.means[i],self.covs[i]
                for _ in range(beta_steps):
                    m,p,_=condition(m,p,b,(y-self.anchor)[None],self.sigma**2*beta_steps)
            self.means[i],self.covs[i]=m,p
        self.last_time = float(time)

    def forecast(self, queries):
        f = design(queries,self.end,self.cells) @ self.u.conj().T
        means=np.array([(f @ m).real+self.anchor for m in self.means])
        variances=np.array([np.diag((f @ p @ f.conj().T).real) for p in self.covs])
        weights=np.exp(self.logs-logsumexp(self.logs))
        mean=np.einsum('k,ktd->td',weights,means)
        delta=means-mean
        covariance=(np.einsum('k,kt,ij->tij',weights,variances,np.eye(3))
                    +np.einsum('k,kti,ktj->tij',weights,delta,delta))
        return {'mean':mean,'covariance':covariance,'weights':weights,
                'component_mean':means,'component_variance':variances,
                'log_evidence':float(logsumexp(self.logs))}
