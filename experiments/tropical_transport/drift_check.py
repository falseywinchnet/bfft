"""Scoring-only check: is the slow Ritz drift first-order multi-mode?
theta(y_k)-lam* vs 2<v1,B(u,v1)>_c (all modes) vs 2 beta a1 (slow mode only),
with u=y_k-y*. Exact J(y_k) slow eigenvalue used (dense), no Krylov error."""
import numpy as np
from .probe import Pair, problem, osc
from .normal_form import conditionals, bilinear


def slow(S, y):
    P, Q = conditionals(S, y); c = np.exp(S.lp)@P; J = Q@P; w = np.sqrt(c)
    M = (w[:, None]*J)/w[None, :]; lam, U = np.linalg.eigh(.5*(M+M.T))
    o = np.argsort(-lam)[1]; return lam[o], U[:, o]/w, P, Q, c


for seed, eps in ((0, .01), (0, .003), (1, .01), (1, .003), (0, .001)):
    C, p, q = problem(128, seed); S = Pair(C, p, q, eps); y = np.zeros(128); tr = [y]
    while True:
        y = S.F(y); y -= y.mean(); tr.append(y)
        if osc(tr[-1]-tr[-2]) < 1e-13: break
    K = len(tr)-1; ys = tr[-1]; lam, v1, P, Q, c = slow(S, ys)
    beta = v1@(c*bilinear(P, Q, v1, v1))
    print(f"s{seed} eps={eps} K={K}")
    for f in (.05, .1, .15, .3):
        k = int(f*K); u = tr[k]-ys; a1 = v1@(c*u)
        th = slow(S, tr[k])[0]
        full = 2*v1@(c*bilinear(P, Q, u, v1)); single = 2*beta*a1
        print(f"   k={k:<5} theta-lam*={th-lam:+.3e}  all-mode 1st order={full:+.3e}  slow-only={single:+.3e}")
