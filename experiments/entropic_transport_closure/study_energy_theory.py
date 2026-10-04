"""Focused proof probes, not an optimizer benchmark or an acquisition method.

The acquisition audit reproduces the public local problem generator in
experiments/tropical_transport/probe.py and its retained anchors. All dense
spectra, fixed points and branch diagnostics are explicit scoring oracles.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .energy_theory import (geometry, osc, linear_factor, quadratic_factor,
                            second_jet_matrix, uniform_bounds, power_and_gramian)


def map_value(logK,a,b,y):
    x=np.log(a)-logsumexp(logK+y[None,:],axis=1)
    return np.log(b)-logsumexp(logK+x[:,None],axis=0)


def finite_probe():
    rng=np.random.default_rng(2030); rows=[]
    for seed in range(10):
        m,n=7,9
        logK=rng.normal(size=(m,n))*rng.uniform(.5,4.)
        a=rng.random(m)+.1; a/=a.sum(); b=rng.random(n)+.1; b/=b.sum()
        y=rng.normal(size=n)*2; g=geometry(logK,a,b,y)
        v=rng.normal(size=n); v/=osc(v)
        for R in [.02,.1,.4,1.,2.]:
            h=R*v; shifted=map_value(logK,a,b,y+h)
            r1=shifted-g.f-g.J@h; r2=r1-g.bilinear(h,h)
            E=g.energy(h)
            rows.append(dict(seed=seed,radius=R,energy=E,
                             linear_l1=g.weighted_l1(r1),linear_bound=linear_factor(R)*E,
                             quadratic_l1=g.weighted_l1(r2),quadratic_bound=quadratic_factor(R)*E))
    return rows


def long_horizon_probe():
    n=3; a=np.ones(n)/n
    mixing=np.array([[.50,.35,.15],[.20,.45,.35],[.30,.20,.50]])
    families=[('small_gap',eps,(1-eps)*np.eye(n)+eps*mixing)
              for eps in [.3,.1,.03,.01,.003,.001]]
    families.append(('exact_resonance',None,np.array([[281,101,218],[101,281,218],[218,218,164]])/600.))
    rows=[]
    for name,eps,K in families:
        g=geometry(np.log(K),a,a,np.zeros(n)); J=g.J
        eigen=np.linalg.eigvalsh(.5*(J+J.T)); gap=1-eigen[-2]
        H=int(math.ceil(3/gap)); L=second_jet_matrix(g)
        W=np.zeros_like(L); W[:n,:n]=g.energy_matrix()
        power,gram=power_and_gramian(L,W,H)
        for amplitude in [.2,.1,.05,.025]:
            h=amplitude*np.array([1.,-.6,-.4]); upper=uniform_bounds(g,h)
            ell=h.copy(); q=np.zeros(n); actual=h.copy()
            e1=e2=certificate=radius=energy_sum=linear_energy=0.
            for _ in range(H):
                z=ell+q; radius=max(radius,osc(z))
                energy=g.energy(z); energy_sum+=energy; linear_energy+=g.energy(ell)
                defect=g.bilinear(z,z)-g.bilinear(ell,ell)
                certificate+=osc(defect)+quadratic_factor(osc(z))*energy/a.min()
                q=J@q+g.bilinear(ell,ell); ell=J@ell
                actual=map_value(np.log(K),a,a,actual); actual-=actual.mean()
                e1=max(e1,osc(actual-ell)); e2=max(e2,osc(actual-ell-q))
            start=np.r_[h,np.outer(h,h).ravel()]
            ledger=float(start@gram@start)
            powered=(power@start)[:n]
            rows.append(dict(family=name,mixing=eps,gap=float(gap),horizon=H,
                             amplitude=amplitude,linear_max_error=e1,second_jet_max_error=e2,
                             linear_uniform_bound=upper['linear'],second_uniform_bound=upper['second_jet'],
                             finite_path_certificate=certificate,initial_energy_norm2=float(a@(h*h)),
                             linear_energy_sum=linear_energy,jet_energy_sum=energy_sum,
                             doubled_energy_sum=ledger,ledger_difference=abs(ledger-energy_sum),
                             power_difference=osc(powered-ell-q),max_jet_radius=radius,
                             certified_jet_radius=upper['second_jet_radius']))
    return rows


def source_problem(n,seed):
    # Exactly the same RNG order and formula as Claude's acquisition study.
    rng=np.random.default_rng(seed)
    X,Y=rng.random((n,2)),rng.random((n,2))
    cost=((X[:,None,:]-Y[None,:,:])**2).sum(-1); cost/=cost.max()
    def mass(Z,centers):
        weights=sum(np.exp(-((Z-center)**2).sum(1)/.02) for center in centers)+.05
        return weights/weights.sum()
    return cost,mass(X,rng.random((2,2))),mass(Y,rng.random((3,2)))


def spectrum(g):
    root=np.sqrt(g.c)
    S=root[:,None]*g.J/root[None,:]
    values,U=np.linalg.eigh(.5*(S+S.T)); order=np.argsort(values)[::-1][1:]
    return values[order],U[:,order]/root[:,None]


def in_fixed_metric(V,weights):
    V=V-np.ones((len(weights),1))@(weights@V)[None,:]
    Z=np.sqrt(weights)[:,None]*V
    return Z/np.linalg.norm(Z,axis=0)[None,:]


def spectral_audit(records):
    rows=[]
    for record in records:
        seed,eps=record['seed'],record['eps']
        cost,a,b=source_problem(128,seed); logK=-cost/eps
        y=np.zeros(128); trajectory=[y.copy()]
        for _ in range(400000):
            new=map_value(logK,a,b,y); new-=new.mean()
            trajectory.append(new.copy())
            if osc(new-y)<1e-13: break
            y=new
        else: raise RuntimeError('reference trajectory did not converge')
        ys=trajectory[-1]; base=geometry(logK,a,b,ys)
        lam,V=spectrum(base); fixed=in_fixed_metric(V,b)
        for index,k in enumerate(record['anchors']):
            endpoint=trajectory[k]; g=geometry(logK,a,b,endpoint)
            vals,modes=spectrum(g); common=in_fixed_metric(modes,b)
            overlaps=np.abs(fixed[:,0]@common)**2
            coverage={}
            for r in [1,2,4,8]:
                QR=np.linalg.qr(common[:,:r])[0]
                coverage[str(r)]=float(np.linalg.norm(QR.T@fixed[:,0])**2)
            disp=endpoint-ys
            full=float(2*V[:,0]@(b*base.bilinear(disp,V[:,0])))
            beta=float(V[:,0]@(b*base.bilinear(V[:,0],V[:,0])))
            amp=float(V[:,0]@(b*disp))
            gammas=np.array([2*V[:,0]@(b*base.bilinear(V[:,j],V[:,0])) for j in range(len(lam))])
            amplitudes=V.T@(b*disp); coeff=gammas*amplitudes
            sequence=np.array([np.sum(coeff*lam**j) for j in range(40)])
            hankel=np.array([sequence[j:j+16] for j in range(16)])
            singular=np.linalg.svd(hankel,compute_uv=False)
            dense_error=abs(float(vals[0])-record['theta'][index])
            row=dict(seed=seed,eps=eps,anchor=k,run_length=len(trajectory)-1,
                     recorded_run_length=record['K'],lambda_star=float(lam[0]),
                     dense_lambda=float(vals[0]),ritz_value=record['theta'][index],
                     dense_ritz_error=dense_error,heuristic_temple=record['kato_temple'][index],
                     eigen_gap=float(vals[0]-vals[1]),
                     top_mode_reference_overlap2=float(overlaps[0]),
                     best_reference_match_rank=int(np.argmax(overlaps))+1,
                     best_reference_overlap2=float(overlaps.max()),projector_coverage=coverage,
                     all_mode_first_order=full,slow_only_first_order=2*beta*amp,
                     actual_drift=float(vals[0]-lam[0]),
                     hankel_singular_values=singular.tolist(),
                     first_order_rule_coefficients=coeff.tolist())
            # Challenge the specific alleged swap in the middle .001 anchor.
            # The path is explicitly interpolation to y*, not ordinary time.
            if eps==.001 and index==1:
                paths=[]
                for steps in [32,64]:
                    prev=fixed[:,0]; min_overlap=1.; min_gap=1.; min_cov2=1.
                    rank_changes=0; tracked=fixed[:,0]; maximum_rank=1
                    for t in np.linspace(0,1,steps+1):
                        gt=geometry(logK,a,b,ys+t*disp)
                        lt,vt=spectrum(gt); ct=in_fixed_metric(vt,b)
                        min_overlap=min(min_overlap,float(abs(prev@ct[:,0])**2))
                        min_gap=min(min_gap,float(lt[0]-lt[1]))
                        q2=np.linalg.qr(ct[:,:2])[0]
                        min_cov2=min(min_cov2,float(np.linalg.norm(q2.T@fixed[:,0])**2))
                        j=int(np.argmax(np.abs(tracked@ct)))
                        rank_changes+=int(j!=0); maximum_rank=max(maximum_rank,j+1)
                        tracked=ct[:,j]; prev=ct[:,0]
                    paths.append(dict(steps=steps,min_adjacent_top_overlap2=min_overlap,
                                      min_top_gap=min_gap,min_reference_top2_coverage=min_cov2,
                                      overlap_continuation_nonleading_samples=rank_changes,
                                      maximum_continued_rank=maximum_rank))
                row['interpolation_continuation']=paths
            rows.append(row)
            print('audit',seed,eps,k,'top overlap',overlaps[0],'best rank',np.argmax(overlaps)+1,flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--acquisition-json')
    args=parser.parse_args()
    result=dict(scope='theory diagnostics; no acquisition or speedup claim',
                finite_remainders=finite_probe(),long_horizons=long_horizon_probe())
    if args.acquisition_json:
        result['acquisition_audit']=spectral_audit(json.loads(Path(args.acquisition_json).read_text()))
    Path(args.out).write_text(json.dumps(result,indent=2))
    print('saved',args.out,flush=True)


if __name__=='__main__': main()
