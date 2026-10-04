"""Mixed homological coefficients of Sinkhorn at y* (oracle diagnostic).

kappa_kij = <v_k, B(v_i,v_j)>_c / (lam_i lam_j - lam_k) over the r slowest
non-gauge modes; compared with the polarized Dirichlet bound
|<v_k,B(v_i,v_j)>_c| <= ||v_k||_inf sqrt(delta_i delta_j).
Supports: random 2-D points, and a uniform 1-D grid (diffusive slow spectrum).
"""
import argparse, json
import numpy as np
from scipy.special import logsumexp
from .probe import Pair, problem, osc
from .normal_form import conditionals, bilinear


def grid1d(n):
    x = (np.arange(n)+.5)/n; C = (x[:, None]-x[None, :])**2
    p = np.exp(-(x-.3)**2/.02)+.1; q = np.exp(-(x-.7)**2/.05)+.1
    return C, p/p.sum(), q/q.sum()


def fixed(S, n):
    y = np.zeros(n)
    for _ in range(400000):
        z = S.F(y); z -= z.mean()
        if osc(z-y) < 1e-13: return z
        y = z
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=96); ap.add_argument('--r', type=int, default=10)
    ap.add_argument('--eps', default='0.03,0.01,0.003'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = []
    for name in ('points2d', 'grid1d'):
        C, p, q = problem(a.n, 0) if name == 'points2d' else grid1d(a.n)
        for eps in map(float, a.eps.split(',')):
            S = Pair(C, p, q, eps); ys = fixed(S, a.n); P, Q = conditionals(S, ys); J = Q@P
            c = np.exp(S.lp)@P; w = np.sqrt(c)
            M = (w[:, None]*J)/w[None, :]; lam, U = np.linalg.eigh(.5*(M+M.T))
            o = np.argsort(-lam)[1:a.r+1]; lam = lam[o]; V = U[:, o]/w[:, None]; d = 1-lam
            worst = (0, None); worst_bound_ratio = 0; mind = (np.inf, None)
            for i in range(a.r):
                for j in range(i, a.r):
                    Bij = bilinear(P, Q, V[:, i], V[:, j])
                    for k in range(a.r):
                        num = float(V[:, k]@(c*Bij)); den = lam[i]*lam[j]-lam[k]
                        bound = np.abs(V[:, k]).max()*np.sqrt(d[i]*d[j])
                        worst_bound_ratio = max(worst_bound_ratio, abs(num)/bound)
                        kap = abs(num/den)
                        if kap > worst[0]: worst = (kap, (i, j, k, num, den, float(np.abs(V[:, k]).max())))
                        if abs(den) < mind[0]: mind = (abs(den), (i, j, k, num))
            rec = dict(support=name, eps=eps, gaps=d.tolist(), max_kappa=worst[0], argmax=worst[1],
                       min_divisor=mind[0], at_min_divisor=mind[1], max_num_over_bound=worst_bound_ratio,
                       vinf=[float(np.abs(V[:, t]).max()) for t in range(a.r)])
            out.append(rec)
            print(f"{name:8} eps={eps:<6} gaps={np.round(d[:5],4)} max|kappa|={worst[0]:.3g} at(i,j,k,num,den,vinf)="
                  f"{tuple(np.round(worst[1],5))} min|den|={mind[0]:.2e} num there={mind[1][3]:.2e} "
                  f"num/bound<={worst_bound_ratio:.3f}", flush=True)
    json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
