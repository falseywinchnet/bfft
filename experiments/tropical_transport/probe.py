"""Does the tropical (max) algebra describe slow Sinkhorn trajectories where
frozen-Jacobian Krylov transport does not?  A representation diagnostic:
actual future sweeps are computed only to measure prediction error.

Metric: osc(v)=max v-min v, the Hilbert projective metric in log coordinates
(gauge invariant). Sinkhorn F and its tropical limit T are both osc-
nonexpansive, which gives the two rigorous bounds recorded here:
  tropical:  osc(F^H y - T^H y) <= H (log n + log m)
  Krylov:    osc(F^H y - y - s_H) <= sum_j [osc(s_j)^2/4 + osc(E c_j)]
"""
import argparse, json
import numpy as np
from scipy.special import logsumexp
from experiments.krylov_em.core import arnoldi


def osc(v):
    return float(v.max()-v.min())


class Pair:
    def __init__(self, C, p, q, eps):
        self.L = -C/eps; self.lp = np.log(p); self.lq = np.log(q)
        self.n, self.m = C.shape

    def F(self, y):
        x = self.lp-logsumexp(self.L+y[None, :], axis=1)
        return self.lq-logsumexp(self.L+x[:, None], axis=0)

    def T(self, y):
        x = self.lp-np.max(self.L+y[None, :], axis=1)
        return self.lq-np.max(self.L+x[:, None], axis=0)

    def piece(self, y):
        """Frozen argmax/argmin piece of T at y: T(z)=z[sigma]+c there."""
        k = np.argmax(self.L+y[None, :], axis=1)
        x = self.lp-self.L[np.arange(self.n), k]-y[k]
        i = np.argmax(self.L+x[:, None], axis=0)
        j = np.arange(self.m)
        sigma = k[i]
        c = self.lq-self.L[i, j]-self.lp[i]+self.L[i, sigma]
        return sigma, c

    def jac(self, y):
        R = self.L+y[None, :]; P = np.exp(R-logsumexp(R, axis=1)[:, None])
        x = self.lp-logsumexp(R, axis=1)
        S = self.L+x[:, None]; Q = np.exp(S-logsumexp(S, axis=0)[None, :]).T
        return lambda h: Q@(P@h)


def problem(n, seed):
    rng = np.random.default_rng(seed)
    X, Y = rng.random((n, 2)), rng.random((n, 2))
    C = ((X[:, None, :]-Y[None, :, :])**2).sum(-1); C /= C.max()
    def mass(Z, centers):
        w = sum(np.exp(-((Z-c)**2).sum(1)/0.02) for c in centers)+0.05
        return w/w.sum()
    return C, mass(X, rng.random((2, 2))), mass(Y, rng.random((3, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--eps', default='0.1,0.03,0.01,0.003')
    ap.add_argument('--anchors', default='0,10,100,1000')
    ap.add_argument('--horizons', default='1,4,16,64,256,1024')
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    anchors = list(map(int, a.anchors.split(','))); Hs = list(map(int, a.horizons.split(',')))
    rows = []
    for seed in range(a.seeds):
        C, p, q = problem(a.n, seed)
        for eps in map(float, a.eps.split(',')):
            S = Pair(C, p, q, eps); y = np.zeros(a.n); k = 0
            for anc in anchors:
                while k < anc: y = S.F(y); k += 1
                # actual future
                fut = {0: y}; z = y
                for h in range(1, max(Hs)+1):
                    z = S.F(z); fut[h] = z
                # Krylov (frozen tangent, Euclidean basis)
                J = S.jac(y); B = arnoldi(J, S.F(y)-y, a.depth)
                kk = B.H.shape[0]; cc = np.zeros(kk); src = np.zeros(kk)
                if kk: src[0] = B.beta
                kryl = {}; kb = 0.
                # exact tropical and frozen piece
                sigma, cpiece = S.piece(y); zt = y.copy(); zp = y.copy()
                trop = {}; piece = {}; changes = 0; prev = sigma
                for h in range(1, max(Hs)+1):
                    s = B.Q@cc if kk else 0*y
                    kb += osc(s)**2/4+(osc(B.remainder@cc) if kk else 0.)
                    cc = src+B.H@cc if kk else cc
                    sg, _ = S.piece(zt)
                    changes += int(np.any(sg != prev)); prev = sg
                    zt = S.T(zt); zp = zp[sigma]+cpiece
                    if h in Hs:
                        kryl[h] = (y+(B.Q@cc if kk else 0), kb)
                        trop[h] = zt.copy(); piece[h] = zp.copy()
                for h in Hs:
                    act = fut[h]
                    r = dict(seed=seed, eps=eps, anchor=anc, H=h,
                             motion=osc(act-y),
                             step=osc(fut[1]-y),
                             krylov_err=osc(kryl[h][0]-act), krylov_bound=kryl[h][1],
                             trop_err=osc(trop[h]-act),
                             trop_bound=h*(np.log(a.n)+np.log(a.n)),
                             piece_err=osc(piece[h]-act),
                             piece_vs_trop=osc(piece[h]-trop[h]),
                             trop_piece_changes_upto_maxH=changes)
                    rows.append(r)
                    print(f"s{seed} eps={eps:<6} k={anc:<5} H={h:<5} motion={r['motion']:.3g} "
                          f"kryl={r['krylov_err']:.3g}({r['krylov_bound']:.2g}) "
                          f"trop={r['trop_err']:.3g}({r['trop_bound']:.2g}) piece={r['piece_err']:.3g}",
                          flush=True)
    json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
