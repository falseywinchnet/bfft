"""Richardson-Lucy (ML-EM Poisson deconvolution) as a mirror/Bregman map.

Log coordinates z=log x.  F(z) = z + log(A^T(y/(A e^z + b))), A^T 1 = 1.
Exact tangent: J = I - Q P with P = diag(1/(Ax)) A diag(x) (row stochastic
up to the background share) and Q = diag(1/r) A^T diag(y/(Ax+b)^2) ... ; the
weighted form diag(x r) J is symmetric, so the natural metric is x_next.
"""
import time
import numpy as np


class Blur:
    def __init__(self, shape, sigma):
        ky = np.fft.fftfreq(shape[0]); kx = np.fft.fftfreq(shape[1])
        self.H = np.exp(-2*(np.pi*sigma)**2*(ky[:, None]**2+kx[None, :]**2))
        self.H = self.H[:, :shape[1]//2+1] if False else self.H
        self.shape = shape

    def __call__(self, x):  # symmetric kernel: A = A^T, A 1 = 1
        return np.fft.irfft2(np.fft.rfft2(x.reshape(self.shape))*self.H[:, :self.shape[1]//2+1],
                             s=self.shape).ravel()


class RL:
    def __init__(self, y, blur, background):
        self.y = y.ravel(); self.A = blur; self.b = background
        self.initial = np.log(np.full(self.y.size, max(self.y.mean()-background, 1e-3)))
        self.reset_counts()

    def reset_counts(self):
        self.calls = self.actions = self.objs = 0

    def _fwd(self, z):
        x = np.exp(z); m = self.A(x)+self.b
        return x, m

    def step(self, z):
        self.calls += 1
        x, m = self._fwd(z)
        return z+np.log(np.maximum(self.A(self.y/m), 1e-300))

    def linearize(self, z):
        from .core import Linearization
        self.calls += 1
        x, m = self._fwd(z); r = self.A(self.y/m); nxt = z+np.log(r)
        w = self.y/m**2
        def action(h):
            self.actions += 1
            return h - self.A(w*self.A(x*h))/r
        xn = x*r
        return Linearization(nxt, action, lambda v: xn*v)

    def objective(self, z):
        self.objs += 1
        x, m = self._fwd(z)
        return float(np.sum(m-self.y*np.log(m)))


def solve(model, method, target, budget=20000, depth=4, horizon=64, guard=True,
          check_every=8, natural=True, tau=0.02):
    """Time to reach objective <= target. Work = map calls + tangent actions.
    Objective evaluations (one blur each) are counted separately and timed."""
    from .core import finite_flow, Anderson
    model.reset_counts(); z = model.initial.copy(); aa = Anderson(depth=depth)
    acc = rej = 0; horizons_used = []; nxt = 0; status = 'budget'; f = model.objective(z); trace = []
    t0 = time.perf_counter()
    while model.calls+model.actions < budget:
        if method == 'ordinary' or model.calls < 4:
            z = model.step(z)
        elif method == 'anderson':
            s = model.step(z); c = aa.advance(z, s)
            if guard and model.objective(c) > model.objective(s):
                c = s; rej += 1
            z = c
        elif method == 'certified':
            from .core import local_arnoldi
            basis, lin = local_arnoldi(model, z, depth, natural)
            H = certified_horizon(basis, tau)
            s = lin.next_state
            if H < 2:
                z = s; rej += 1
            else:
                cand = z+basis.displacement(H); fc = model.objective(cand)
                if not np.isfinite(fc) or fc > model.objective(s):
                    z = s; rej += 1
                else:
                    z = cand; acc += 1; horizons_used.append(H)
        elif method == 'krylov':
            cand, basis, lin = finite_flow(model, z, depth=depth, horizon=horizon,
                                           natural_metric=natural)
            s = lin.next_state
            if guard:
                fc = model.objective(cand)
                if not np.isfinite(fc) or fc > model.objective(s):
                    z = s; rej += 1; continue_ok = False
                else:
                    z = cand; acc += 1
            else:
                z = cand; acc += 1
        work = model.calls+model.actions
        if work >= nxt:
            f = model.objective(z); nxt = work+check_every
            trace.append((work, time.perf_counter()-t0, f))
            if f <= target:
                status = 'target'; break
            if not np.isfinite(f):
                status = 'invalid'; break
    return z, dict(method=method, status=status, seconds=time.perf_counter()-t0,
                   calls=model.calls, actions=model.actions, objs=model.objs, horizons=horizons_used,
                   accepted=acc, rejected=rej, objective=f, trace=trace)


def remainder_parts(model, z, d):
    """Exact split F(z+d)-F(z)-Jd = -Q L_P(d~) + L_Q(-e); background = column with d=0."""
    x, m = model._fwd(z); A = model.A; y = model.y
    ratio = (A(x*np.exp(d))+model.b)/m          # P~ exp(d~)
    e = np.log(ratio); Pd = A(x*d)/m             # P~ d~
    r = A(y/m)
    LP = e-Pd
    Qexp = A(y/m*np.exp(-e))/r; Qe = A(y/m*e)/r
    LQ = np.log(Qexp)+Qe
    QLP = A(y/m*LP)/r
    return -QLP, LQ


def osc0(v):
    return max(float(v.max()), 0.)-min(float(v.min()), 0.)


def certified_horizon(basis, tau, horizons=(16, 32, 64, 128, 256, 512, 1024)):
    """Largest H with sum_j [osc(s_j)^2/8 + ||E c_j||_inf] <= tau*||s_H||_inf.

    osc^2/8 bounds each frozen-J one-step defect exactly (log-moment bound);
    E c_j is the Arnoldi compression defect. Summing them is a trajectory
    certificate only under a contraction we have not proved for I-QP, so
    this is a certificate-derived gate, and the objective guard stays on.
    """
    k = basis.H.shape[0]; c = np.zeros(k); src = np.zeros(k)
    if k == 0: return 0
    # Triangle-inequality bounds: O(k) per step instead of O(nk).
    oq = np.array([float(basis.Q[:, i].max()-basis.Q[:, i].min()) for i in range(k)])
    er = np.abs(basis.remainder).max(axis=0)
    src[0] = basis.beta; acc = 0.; best = 0; hs = set(horizons)
    for j in range(max(horizons)):
        a = np.abs(c)
        acc += float(oq@a)**2/8+float(er@a)
        c = src+basis.H@c
        if j+1 in hs:
            if acc <= tau*float(np.abs(basis.Q@c).max()): best = j+1
            else: break
    return best
