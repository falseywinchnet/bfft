"""Single-mode acquisition identifiability test (oracle used ONLY for scoring).

At anchors y_k on the ordinary trajectory: Arnoldi of J(y_k) in the state's
c-metric on the c-mean-free residual; slowest Ritz pair (theta, z), Ritz
residual rho, Kato-Temple bound rho^2/(theta1-theta2), signed coordinate
a_hat = <z,r>_c/(theta-1) with z sign-aligned to the first anchor, and the
fast-leakage share of r outside z. Fit theta = lam + 2 beta a on two anchors,
predict the held-out third. Then advance the scalar quadratic slow model from
the second anchor and compare with the frozen-anchor rate.
"""
import argparse, json
import numpy as np
from scipy.special import logsumexp
from .probe import Pair, problem, osc
from .normal_form import conditionals


def cstate(S, y):
    P, Q = conditionals(S, y); c = np.exp(S.lp)@P
    return P, Q, c


def ritz(S, y, depth):
    P, Q, c = cstate(S, y); r = S.F(y)-y
    cm = lambda v: v-(c@v)/c.sum()
    J = lambda h: cm(Q@(P@h))
    v = cm(r); V = [v/np.sqrt(v@(c*v))]; Hm = np.zeros((depth+1, depth))
    for j in range(depth):
        w = J(V[j])
        for _ in range(2):
            for i in range(j+1):
                h = V[i]@(c*w); Hm[i, j] += h; w = w-h*V[i]
        nrm = np.sqrt(w@(c*w)); Hm[j+1, j] = nrm
        if nrm < 1e-14: depth = j+1; break
        V.append(w/nrm)
    Vm = np.column_stack(V[:depth]); T = Hm[:depth, :depth]; T = .5*(T+T.T)
    th, W = np.linalg.eigh(T); o = np.argsort(-th); th = th[o]; W = W[:, o]
    z = Vm@W[:, 0]; rho = abs(Hm[depth, depth-1]*W[-1, 0])
    rc = cm(r); proj = z@(c*rc)
    leak = np.sqrt(max(0., rc@(c*rc)-proj**2))/np.sqrt(rc@(c*rc))
    return dict(theta=th[0], theta2=th[1], rho=rho, z=z, c=c, proj=proj, leak=leak,
                kt=rho**2/max(th[0]-th[1], 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=128); ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--eps', default='0.01,0.003,0.001'); ap.add_argument('--depth', type=int, default=12)
    ap.add_argument('--fractions', default='0.05,0.10,0.15')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = []
    fr = list(map(float, a.fractions.split(',')))
    for seed in range(a.seeds):
        C, p, q = problem(a.n, seed)
        for eps in map(float, a.eps.split(',')):
            S = Pair(C, p, q, eps); y = np.zeros(a.n); traj = [y]
            while True:
                y = S.F(y); y = y-y.mean(); traj.append(y)
                if osc(traj[-1]-traj[-2]) < 1e-13 or len(traj) > 400000: break
            K = len(traj)-1; ys = traj[-1]
            # oracle (scoring only): slow mode, lambda*, beta at y*
            P, Q, c = cstate(S, ys); J = Q@P; w = np.sqrt(c)
            lam, U = np.linalg.eigh(.5*((w[:, None]*J)/w[None, :]+((w[:, None]*J)/w[None, :]).T))
            o = np.argsort(-lam)[1:]; lam1 = lam[o[0]]; v1 = U[:, o[0]]/w
            from .normal_form import bilinear
            beta = float(v1@(c*bilinear(P, Q, v1, v1)))
            ks = [max(2, int(f*K)) for f in fr]
            R = [ritz(S, traj[k], a.depth) for k in ks]
            sgn = [1.]
            for Rk in R[1:]:
                sgn.append(np.sign(Rk['z']@(R[0]['c']*R[0]['z'])))
            ahat = [s*Rk['proj']/(Rk['theta']-1) for s, Rk in zip(sgn, R)]
            s1 = np.sign(v1@(R[0]['c']*R[0]['z']))   # oracle orientation, scoring only
            atrue = [s1*float(v1@(c*(traj[k]-ys))) for k in ks]
            th = [Rk['theta'] for Rk in R]
            bhat = (th[1]-th[0])/(2*(ahat[1]-ahat[0])); lhat = th[0]-2*bhat*ahat[0]
            pred3 = lhat+2*bhat*ahat[2]
            # prediction from anchor 2 over H=ceil(1/(1-lhat))
            H = int(np.ceil(1/max(1e-9, 1-lhat)))
            g = ahat[1]
            for _ in range(H): g = lhat*g+bhat*g*g
            frozen = th[1]**H*ahat[1]
            kH = min(K, ks[1]+H); actual = s1*float(v1@(c*(traj[kH]-ys)))
            rec = dict(seed=seed, eps=eps, K=K, anchors=ks, lam_true=lam1, beta_true=beta*s1,
                       gap=1-lam1, theta=th, theta2=[Rk['theta2'] for Rk in R],
                       kato_temple=[Rk['kt'] for Rk in R], ritz_resid=[Rk['rho'] for Rk in R],
                       leak=[Rk['leak'] for Rk in R], a_hat=ahat, a_true=atrue,
                       drift=th[1]-th[0], beta_hat=bhat, lam_hat=lhat,
                       theta3_pred=pred3, theta3_obs=th[2], H=H,
                       quad_pred=g, frozen_pred=frozen, actual=actual,
                       scale=[abs(x)*float(np.abs(v1).max()) for x in atrue])
            out.append(rec)
            print(f"s{seed} eps={eps} K={K} gap={1-lam1:.3e} anchors={ks} a|v|={np.round(rec['scale'],3)}")
            print(f"   theta={np.round(th,7)} KT={np.array(rec['kato_temple'])} leak={np.round(rec['leak'],3)}")
            print(f"   a_hat={np.round(ahat,4)} a_true={np.round(atrue,4)}  drift={th[1]-th[0]:.2e} vs KT {max(rec['kato_temple'][:2]):.1e}")
            print(f"   beta_hat={bhat:.3e} beta_true={beta*s1:.3e} |err|={abs(bhat-beta*s1):.2e} vs delta/a={(1-lam1)/abs(ahat[1]):.2e}"
                  f"   lam_hat={lhat:.7f} lam_true={lam1:.7f}")
            print(f"   held-out theta3 pred={pred3:.7f} obs={th[2]:.7f} miss={pred3-th[2]:.2e} (KT {rec['kato_temple'][2]:.1e}; drift scale {abs(th[2]-th[1]):.1e})")
            print(f"   H={H}: actual={actual:.4e} quad={g:.4e} frozen={frozen:.4e}  "
                  f"rel err quad={abs(g-actual)/abs(actual):.3f} frozen={abs(frozen-actual)/abs(actual):.3f}", flush=True)
    json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
