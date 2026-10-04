"""Four independently specified geometries, each with a known minimizer.

Planted optima define the synthetic objectives and exact measurement gaps;
the optimizer does not access them to choose a Krylov step.
ADMM is a splitting iteration with Bregman memory, not identified with mirror
descent. The simplex uses n-1 log ratios, avoiding a singular gauge metric.
"""
import numpy as np
from .core import Linearization, identity


def orthogonal(rng, n):
    return np.linalg.qr(rng.normal(size=(n, n)))[0]


class QuadraticMirror:
    name = "quadratic_mirror"
    def __init__(self, n=96, seed=0, condition=1000., scale=1000.):
        rng = np.random.default_rng(seed)
        self.g = np.geomspace(1., scale, n)
        self.root = np.sqrt(self.g)
        Q = orthogonal(rng, n)
        self.B = (Q * np.geomspace(1. / condition, 1., n)) @ Q.T
        self.target = rng.normal(size=n) * self.root
        self.alpha = .9
        self.initial = np.zeros(n)
        self.dimension = n

    def step(self, y):
        return y - self.alpha * self.root * (self.B @ ((y - self.target) / self.root))

    def linearize(self, y):
        def action(h):
            return h - self.alpha * self.root * (self.B @ (h / self.root))
        return Linearization(self.step(y), action, lambda h: h / self.g)

    def gap(self, y):
        e = (y - self.target) / self.root
        return max(0., .5 * float(e @ self.B @ e))

    def primal(self, y):
        return y / self.g


class EntropicSeparable:
    name = "entropy_dual_affine"
    def __init__(self, n=96, seed=0, condition=1000.):
        rng = np.random.default_rng(seed)
        self.target = rng.normal(size=n) * .8
        self.xstar = np.exp(self.target)
        self.a = np.geomspace(1. / condition, 1., n)
        rng.shuffle(self.a)
        self.alpha = .9
        self.initial = np.zeros(n)
        self.dimension = n

    def step(self, y):
        return y - self.alpha * self.a * (y - self.target)

    def linearize(self, y):
        x = np.exp(y)
        return Linearization(self.step(y), lambda h: (1. - self.alpha * self.a) * h,
                             lambda h: x * h)

    def gap(self, y):
        e = y - self.target
        # exp(e)*e - expm1(e) = (1+expm1(e))*e - expm1(e).
        value = self.xstar * (np.exp(e) * e - np.expm1(e))
        return max(0., float(self.a @ value))

    def primal(self, y):
        return np.exp(y)


class EntropicPrimal:
    """Same separable mirror iteration represented in primal coordinates."""
    name = "entropy_primal_ablation"
    def __init__(self, n=96, seed=0, condition=1000.):
        self.dual = EntropicSeparable(n, seed, condition)
        self.initial = np.ones(n)
        self.dimension = n

    def step(self, x):
        if np.any(x <= 0):
            raise FloatingPointError("primal extrapolate outside positive orthant")
        return np.exp(self.dual.step(np.log(x)))

    def linearize(self, x):
        nxt = self.step(x)
        diagonal = nxt * (1 - self.dual.alpha * self.dual.a) / x
        return Linearization(nxt, lambda h: diagonal * h, lambda h: h / x)

    def gap(self, x):
        if np.any(x <= 0):
            return float("inf")
        return self.dual.gap(np.log(x))

    def primal(self, x):
        return x


class EntropicSimplex:
    name = "entropy_simplex"
    def __init__(self, n=96, seed=0, condition=1000.):
        rng = np.random.default_rng(seed)
        Q = orthogonal(rng, n)
        self.A = (Q * (n * np.geomspace(1. / condition, 1., n))) @ Q.T
        target_log = rng.normal(size=n) * 1.2
        target_log -= np.max(target_log)
        self.xstar = np.exp(target_log)
        self.xstar /= self.xstar.sum()
        self.logstar = np.log(self.xstar)
        self.tau = .05
        # diag(x)-xx^T <= .5 I, hence this is a global relative-smoothness bound.
        self.alpha = .9 / (.5 * n + self.tau)
        self.initial = np.zeros(n - 1)
        self.dimension = n - 1

    def _decode(self, y):
        logits = np.append(y, 0.)
        logits -= np.max(logits)
        logx = logits - np.log(np.exp(logits).sum())
        return np.exp(logx), logx

    def _terms(self, y):
        x, logx = self._decode(y)
        gradient = self.A @ (x - self.xstar) + self.tau * (logx - self.logstar)
        return x, logx, y - self.alpha * (gradient[:-1] - gradient[-1])

    def step(self, y):
        return self._terms(y)[2]

    def linearize(self, y):
        x, logx, nxt = self._terms(y)
        def metric(h):
            return x[:-1] * h - x[:-1] * float(x[:-1] @ h)
        def action(h):
            augmented = np.append(h, 0.)
            dx = x * (augmented - float(x @ augmented))
            dh = self.A @ dx
            # Entropy gradient differences are exactly tau*y, avoiding dx/x.
            return (1. - self.alpha * self.tau) * h - self.alpha * (dh[:-1] - dh[-1])
        return Linearization(nxt, action, metric)

    def gap(self, y):
        x, logx = self._decode(y)
        e = x - self.xstar
        return max(0., .5 * float(e @ self.A @ e) + self.tau * float(x @ (logx - self.logstar)))

    def primal(self, y):
        return self._decode(y)[0]


class ADMMLasso:
    name = "admm_lasso"
    def __init__(self, n=96, seed=0, condition=1000., rho=.1):
        rng = np.random.default_rng(seed)
        Q = orthogonal(rng, n)
        eigenvalues = np.geomspace(1. / condition, 1., n)
        self.H = (Q * eigenvalues) @ Q.T
        self.rho = rho
        self.lam = .05
        self.xstar = rng.normal(size=n)
        self.xstar[rng.uniform(size=n) > .25] = 0.
        self.subgradient = rng.uniform(-.6, .6, n)
        active = self.xstar != 0
        self.subgradient[active] = np.sign(self.xstar[active])
        self.q = self.H @ self.xstar + self.lam * self.subgradient
        # Reusable exact screened inverse; setup is reported separately.
        self.R = (Q * (1. / (eigenvalues + rho))) @ Q.T
        self.initial = np.zeros(2 * n)
        self.dimension = 2 * n
        self.n = n

    def _terms(self, z):
        d, b = z[:self.n], z[self.n:]
        x = self.R @ (self.q + self.rho * (d - b))
        s = x + b
        dnext = np.sign(s) * np.maximum(np.abs(s) - self.lam / self.rho, 0.)
        return s, np.concatenate((dnext, s - dnext))

    def step(self, z):
        return self._terms(z)[1]

    def linearize(self, z):
        s, nxt = self._terms(z)
        mask = np.abs(s) > self.lam / self.rho
        def action(h):
            hd, hb = h[:self.n], h[self.n:]
            ds = self.rho * (self.R @ (hd - hb)) + hb
            return np.concatenate((mask * ds, (~mask) * ds))
        return Linearization(nxt, action, identity)

    def gap(self, z):
        d = z[:self.n]
        e = d - self.xstar
        nonsmooth = np.abs(d).sum() - np.abs(self.xstar).sum() - self.subgradient @ e
        return max(0., .5 * float(e @ self.H @ e) + self.lam * float(nonsmooth))

    def primal(self, z):
        return z[:self.n]


class EntropicBoundary:
    """Linear simplex objective: primal boundary optimum, no finite dual root.

    This is an exactly solvable control, not a new way of solving a linear
    program. It exposes why a finite polynomial need not invert I-J.
    """
    name = "entropy_boundary_drift"
    def __init__(self, n=96, seed=0, condition=None):
        rng=np.random.default_rng(seed)
        self.c=np.linspace(0.,1.,n);rng.shuffle(self.c)
        self.r=-(self.c[:-1]-self.c[-1])
        self.initial=np.zeros(n-1)
        self.dimension=n-1

    def step(self,y):return y+self.r

    def primal(self,y):
        a=np.append(y,0.);a-=np.max(a)
        x=np.exp(a);return x/x.sum()

    def linearize(self,y):
        x=self.primal(y)[:-1]
        return Linearization(self.step(y),identity,lambda h:x*h-x*float(x@h))

    def gap(self,y):return float(self.c@self.primal(y))


class ADMMShadow:
    """Exact pre-shrink ADMM state, encoding BOTH split and dual variables.

    On the ordinary proximal graph, d=shrink(s), b=s-d. The shadow map is
    the Douglas-Rachford representation of this ADMM, already classical.
    Its acceleration is a representation ablation, not a new base solver.
    """
    name='admm_shadow'
    def __init__(self,n=96,seed=0,condition=1000.,rho=.1):
        self.full=ADMMLasso(n,seed,condition,rho)
        self.initial=np.zeros(n)
        self.dimension=n

    def decode(self,s):
        threshold=self.full.lam/self.full.rho
        d=np.sign(s)*np.maximum(np.abs(s)-threshold,0.)
        return np.concatenate((d,s-d))

    def step(self,s):
        z=self.decode(s);n=self.dimension
        d,b=z[:n],z[n:]
        return self.full.R@(self.full.q+self.full.rho*(d-b))+b

    def linearize(self,s):
        mask=np.abs(s)>self.full.lam/self.full.rho
        def action(h):
            hd=mask*h;hb=h-hd
            return self.full.rho*(self.full.R@(hd-hb))+hb
        return Linearization(self.step(s),action,identity)

    def gap(self,s):return self.full.gap(self.decode(s))

    def primal(self,s):return self.decode(s)[:self.dimension]


FAMILIES = {cls.name: cls for cls in
            [QuadraticMirror, EntropicSeparable, EntropicSimplex, ADMMLasso,
             EntropicPrimal, EntropicBoundary, ADMMShadow]}
