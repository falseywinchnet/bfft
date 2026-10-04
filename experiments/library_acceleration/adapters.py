"""Analytic adapters for scikit-image 0.24 restoration recurrences.

These are independently expressed maps, checked against the public library.
Richardson--Lucy uses scipy.signal.convolve's actual zero-padded 'same'
operation. Chambolle uses forward differences with zero terminal components,
not the periodic Fourier geometry of the earlier Meyer experiment.
"""
import numpy as np
from scipy.signal import convolve
from experiments.krylov_bregman.core import Linearization


class Chambolle:
    def __init__(self, image, weight=.1):
        self.image = np.asarray(image, dtype=float)
        self.shape = self.image.shape
        self.ndim = self.image.ndim
        self.weight = weight
        self.tau = 1 / (2 * self.ndim)
        self.initial = np.zeros((self.ndim,) + self.shape).ravel()

    def grad(self, u):
        g = np.zeros((self.ndim,) + self.shape)
        for ax in range(self.ndim):
            sl = [slice(None)] * self.ndim
            sl[ax] = slice(0, -1)
            g[(ax,) + tuple(sl)] = np.diff(u, axis=ax)
        return g

    def divergence(self, p):
        p = p.reshape((self.ndim,) + self.shape)
        d = -p.sum(0)
        for ax in range(self.ndim):
            dst = [slice(None)] * self.ndim
            src = dst.copy()
            dst[ax] = slice(1, None)
            src[ax] = slice(0, -1)
            d[tuple(dst)] += p[(ax,) + tuple(src)]
        return d

    def output(self, p):
        return self.image + self.divergence(p)

    def step(self, p):
        g = self.grad(self.output(p))
        den = 1 + self.tau / self.weight * np.linalg.norm(g, axis=0)
        return ((p.reshape(g.shape) - self.tau * g) / den).ravel()

    def linearize(self, p):
        g = self.grad(self.output(p))
        length = np.linalg.norm(g, axis=0)
        den = 1 + self.tau / self.weight * length
        nxt = (p.reshape(g.shape) - self.tau * g) / den
        # At zero gradient and nonzero numerator, this selected action is
        # not a Frechet derivative. Such anchors are explicitly rejected.
        if np.any((length == 0) & (np.linalg.norm(nxt, axis=0) != 0)):
            raise ValueError('nondifferentiable Chambolle anchor')
        def action(h):
            dg = self.grad(self.divergence(h))
            dn = np.divide(np.sum(g * dg, axis=0), length,
                           out=np.zeros_like(length), where=length > 0)
            return ((h.reshape(g.shape) - self.tau * dg
                     - nxt * (self.tau / self.weight * dn)) / den).ravel()
        return Linearization(nxt.ravel(), action)

    def valid(self, p):
        return np.isfinite(p).all() and np.max(np.linalg.norm(
            p.reshape((self.ndim,) + self.shape), axis=0)) <= self.weight * (1 + 1e-12)


class RichardsonLucy:
    def __init__(self, image, psf):
        self.image = np.asarray(image, dtype=float)
        self.shape = self.image.shape
        self.psf = np.asarray(psf, dtype=float)
        self.mirror = np.flip(self.psf)
        self.initial = np.full(self.image.size, .5)

    def forward(self, x):
        return convolve(x.reshape(self.shape), self.psf, mode='same')

    def backward(self, x):
        return convolve(x, self.mirror, mode='same')

    def step(self, x):
        c = self.forward(x) + 1e-12
        return (x.reshape(self.shape) * self.backward(self.image / c)).ravel()

    def linearize(self, x):
        c = self.forward(x) + 1e-12
        b = self.backward(self.image / c)
        def action(h):
            return (h.reshape(self.shape) * b - x.reshape(self.shape) *
                    self.backward((self.image / c**2) * self.forward(h))).ravel()
        return Linearization((x.reshape(self.shape) * b).ravel(), action)

    def output(self, x):
        return x.reshape(self.shape)

    def valid(self, x):
        return np.isfinite(x).all() and np.min(x) >= 0
