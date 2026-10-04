"""Formal test: slow Sinkhorn as a near-parabolic map (Koenigs vs Fatou).

Uses the fixed point y* as an ORACLE: this is a structure diagnostic, not a
solver. At y*: J*=QP, c-self-adjoint; second-order term
  B(u,w) = 1/2 [ Q Cov_P(u,w) - Cov_Q(Pu,Pw) ].
Slow mode v1 (largest non-gauge eigenvalue lam, c-orthonormal), scalar
coefficient beta=<v1,B(v1,v1)>_c, Koenigs radius a*=(1-lam)/|beta|.
"""
import argparse, json
import numpy as np
from scipy.special import logsumexp
from .probe import Pair, problem, osc


def conditionals(S, y):
    R = S.L+y[None, :]; lr = logsumexp(R, axis=1); P = np.exp(R-lr[:, None])
    x = S.lp-lr; T = S.L+x[:, None]; Q = np.exp(T-logsumexp(T, axis=0)[None, :]).T
    return P, Q


def bilinear(P, Q, u, w):
    Pu, Pw = P@u, P@w
    covP = P@(u*w)-Pu*Pw
    covQ = Q@(Pu*Pw)-(Q@Pu)*(Q@Pw)
    return 0.5*(Q@covP-covQ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--eps', default='0.03,0.01,0.003,0.001')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = []
    for seed in range(a.seeds):
        C, p, q = problem(a.n, seed)
        for eps in map(float, a.eps.split(',')):
            S = Pair(C, p, q, eps); y = np.zeros(a.n); traj = [y]
            for k in range(400000):
                y = S.F(y); y = y-y.mean(); traj.append(y)
                if osc(traj[-1]-traj[-2]) < 1e-13: break
            ystar = y; K = len(traj)-1
            P, Q = conditionals(S, ystar); J = Q@P
            c = (np.exp(S.L+(S.lp-logsumexp(S.L+ystar[None, :], axis=1))[:, None]+ystar[None, :])).sum(0)
            w = np.sqrt(c); Sym = (w[:, None]*J)/w[None, :]
            Sym = 0.5*(Sym+Sym.T); lam, U = np.linalg.eigh(Sym)
            order = np.argsort(-lam); lam = lam[order]; U = U[:, order]
            # drop the gauge eigenvalue 1 (constant direction)
            lam, U = lam[1:], U[:, 1:]; V = U/w[:, None]     # c-orthonormal
            v1 = V[:, 0]; l1 = lam[0]
            # check B by finite difference
            t = 1e-4; fd = (S.F(ystar+t*v1)+S.F(ystar-t*v1)-2*S.F(ystar))/(2*t*t)
            Bv = bilinear(P, Q, v1, v1); fd -= fd@c/c.sum(); Bv2 = Bv-Bv@c/c.sum()
            beta = float(v1@(c*Bv)); astar = (1-l1)/abs(beta)
            amp = np.array([float(v1@(c*(z-ystar))) for z in traj])
            rest = np.array([np.sqrt(max(0., float((z-ystar)@(c*(z-ystar)))-float(v1@(c*(z-ystar)))**2)) for z in traj])
            # scalar model predictions from anchors
            preds = []
            for k0 in sorted(set([10, 30, 100, 300, 1000, 3000, 10000])):
                if k0 >= K: continue
                for H in (16, 64, 256, 1024):
                    if k0+H > K: continue
                    a0 = amp[k0]; g = a0
                    for _ in range(H): g = l1*g+beta*g*g
                    anchor_rate = l1+2*beta*a0
                    preds.append(dict(k=k0, H=H, a0=a0, actual=amp[k0+H], linear=l1**H*a0,
                                      quadratic=g, frozen_anchor=anchor_rate**H*a0,
                                      rest0=rest[k0]))
            vinf=float(np.abs(v1).max()); Pvinf=float(np.abs(P@v1).max())
            rec = dict(vinf=vinf, beta_bound=0.5*(Pvinf+l1*vinf)*(1-l1), seed=seed, eps=eps, sweeps_to_1e13=K, lam1=l1, lam2=float(lam[1]),
                       gap=1-l1, beta=beta, astar=astar,
                       B_fd_relerr=float(np.linalg.norm(fd-Bv2)/np.linalg.norm(Bv2)),
                       amp=amp[::max(1, K//400)].tolist(), rest=rest[::max(1, K//400)].tolist(),
                       stride=max(1, K//400), preds=preds)
            out.append(rec)
            print(f"s{seed} eps={eps} K={K} lam1={l1:.6f} lam2={lam[1]:.6f} gap={1-l1:.2e} "
                  f"beta={beta:.3g} bound={rec['beta_bound']:.3g} 1/vinf={1/vinf:.3g} a*={astar:.3g} |a0|={abs(amp[0]):.3g} Bfd={rec['B_fd_relerr']:.1e}", flush=True)
            for r in preds:
                print(f"   k={r['k']:<6}H={r['H']:<5}a0={r['a0']:+.3e} rest={r['rest0']:.2e} actual={r['actual']:+.3e} "
                      f"lin={r['linear']:+.3e} quad={r['quadratic']:+.3e} frozen={r['frozen_anchor']:+.3e}")
    json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
