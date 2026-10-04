"""Proof-oracle operations for finite energy bounds and second-jet transport.

All certificate inequalities concern exact arithmetic. Floating-point checks
are regression evidence, not outward-rounded numerical certificates.
These routines do not implement a production accelerator.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import logsumexp


def osc(x):
    return float(np.ptp(x))


def phi(order, x):
    """Entire phi function (exp(x)-sum_{j<order} x^j/j!)/x^order."""
    x = float(x)
    if x < 0:
        raise ValueError('This proof oracle only needs nonnegative arguments.')
    if x < 1.:
        term = 1./math.factorial(order)
        value = term
        for j in range(1, 80):
            term *= x/(order+j)
            value += term
            if abs(term) <= 2e-17*value:
                break
        return value
    return (math.exp(x)-sum(x**j/math.factorial(j) for j in range(order)))/x**order


def linear_factor(radius):
    return 2*phi(2, 2*radius)


def quadratic_factor(radius):
    return 5*radius*phi(3, 2*radius)


@dataclass
class Geometry:
    log_kernel: np.ndarray
    a: np.ndarray
    b: np.ndarray
    y: np.ndarray
    f: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    c: np.ndarray

    @property
    def J(self):
        return self.Q@self.P

    def energy(self, h):
        mu = self.P@h
        return float(np.sum(self.a[:, None]*self.P*(h[None, :]-mu[:, None])**2))

    def energy_matrix(self):
        return np.diag(self.c)-self.P.T@(self.a[:, None]*self.P)

    def bilinear(self, u, v):
        pu, pv = self.P@u, self.P@v
        covp = np.sum(self.P*(u[None, :]-pu[:, None])*(v[None, :]-pv[:, None]), axis=1)
        ju, jv = self.Q@pu, self.Q@pv
        covq = np.sum(self.Q*(pu[None, :]-ju[:, None])*(pv[None, :]-jv[:, None]), axis=1)
        return .5*(self.Q@covp-covq)

    def third(self, h):
        """D^3 F[h,h,h], without the factorial 1/6."""
        ph = self.P@h
        dp = h[None, :]-ph[:, None]
        varp = np.sum(self.P*dp**2, axis=1)
        kappap = np.sum(self.P*dp**3, axis=1)
        jh = self.Q@ph
        dq = ph[None, :]-jh[:, None]
        covq = np.sum(self.Q*dq*varp[None, :], axis=1)
        kappaq = np.sum(self.Q*dq**3, axis=1)
        return self.Q@kappap-3*covq+kappaq

    def weighted_l1(self, x):
        return float(self.c@np.abs(x))

    def center(self, x):
        return x-float(self.c@x/self.c.sum())

    def quadratic_tensor(self):
        n = len(self.y)
        eye = np.eye(n)
        return np.stack([self.bilinear(eye[i], eye[j])
                         for i in range(n) for j in range(n)], axis=1)


def geometry(log_kernel, a, b, y):
    log_kernel, a, b, y = map(lambda x: np.asarray(x, dtype=float), (log_kernel, a, b, y))
    row = logsumexp(log_kernel+y[None, :], axis=1)
    P = np.exp(log_kernel+y[None, :]-row[:, None])
    x = np.log(a)-row
    col = logsumexp(log_kernel+x[:, None], axis=0)
    Q = np.exp(log_kernel+x[:, None]-col[None, :]).T
    return Geometry(log_kernel, a, b, y.copy(), np.log(b)-col, P, Q, a@P)


def second_jet_matrix(g):
    """At a fixed point: [J, B; 0, J tensor J] on (h,h tensor h).

    This is the second jet of the iterated map, not exact iteration of the
    quadratic Taylor polynomial. Its tensor coordinate is (J^j h)^tensor2.
    """
    J = g.J
    n = len(J)
    L = np.zeros((n+n*n, n+n*n))
    L[:n, :n] = J
    L[:n, n:] = g.quadratic_tensor()
    L[n:, n:] = np.kron(J, J)
    return L


def power_and_gramian(L, W, horizon):
    """L^H and sum_{j=0}^{H-1} (L^j)^T W L^j by binary composition.

    No inverse or spectral separation is used; unit and defective modes are
    allowed. Complexity O(d^3 log H) for dense d-dimensional matrices.
    """
    if horizon < 0 or int(horizon) != horizon:
        raise ValueError('horizon must be a nonnegative integer')
    P, S = np.eye(len(L)), np.zeros_like(W)
    block, energy = L.copy(), W.copy()
    left = int(horizon)
    while left:
        if left & 1:
            S = S+P.T@energy@P
            P = block@P
        energy = energy+block.T@energy@block
        block = block@block
        left >>= 1
    return P, .5*(S+S.T)


def uniform_bounds(g, h):
    """All-horizon bounds at a FIXED POINT only, in osc norm.

    Probability marginals are required. Uniformity is in horizon and spectral
    gap, not dimension, minimum marginal mass, or amplitude.
    """
    h = g.center(h)
    A2 = float(g.c@(h*h)); A = math.sqrt(A2)
    minimum = float(g.c.min()); R = osc(h)
    first = linear_factor(R)*A2/minimum
    S = A2/math.sqrt(minimum)
    second = (2*A*S+S*S+quadratic_factor(R+A2/minimum)*(A+S)**2)/minimum
    return dict(linear=first, second_jet=second, amplitude=A, radius=R,
                cmin=minimum, second_jet_radius=R+A2/minimum)


def observable_rows(J, output, tol=1e-12):
    """Numerical orthonormal basis of the minimal J-invariant output row space."""
    rows = np.atleast_2d(output).copy()
    basis = []
    for _ in range(len(J)):
        added = []
        for row in rows:
            w = row.copy()
            original = np.linalg.norm(w)
            for _ in range(2):
                for v in basis:
                    w -= v*(v@w)
            size = np.linalg.norm(w)
            if size > tol*max(1., original):
                w /= size
                basis.append(w)
                added.append(w)
        if not added:
            break
        rows = np.array(added)@J
    return np.array(basis)
