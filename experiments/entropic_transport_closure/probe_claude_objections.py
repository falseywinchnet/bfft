"""Falsifiers for cluster size, observability, excitation, radius, and cost.

Dense fixed-point information is explicitly an oracle. No accelerator is
implemented. Partial timing excludes acquisition/certification and is not an
end-to-end speedup. The density sweep uses nested clouds and fixed centers.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from .energy_theory import geometry, osc, power_and_gramian
from .study_energy_theory import source_problem, spectrum, in_fixed_metric, map_value


def geometric_sum(products, horizon):
    products=np.asarray(products)
    answer=np.full_like(products,float(horizon))
    mask=np.abs(products-1)>1e-14
    answer[mask]=-np.expm1(horizon*np.log(np.maximum(products[mask],1e-300)))/(1-products[mask])
    return answer


def observability_gramian(lam, output, horizon):
    return (output.T@output)*geometric_sum(lam[:,None]*lam[None,:],horizon)


def rule_outputs(g,V,r):
    """Rows G_ij in nonconstant eigen-coordinates: 2<vi,B(V x,vj)>c."""
    J=g.J; rows=[]
    for j in range(r):
        v=V[:,j]; pv=g.P@v
        twiceB=J*v[None,:]-2*(g.Q*pv[None,:])@g.P+(J@v)[:,None]*J
        rows.extend((V[:,:r].T*g.c[None,:])@twiceB@V)
    return np.asarray(rows)


def gramian_summary(lam,O,H):
    W=observability_gramian(lam,O,H)
    w,U=np.linalg.eigh((W+W.T)/2); w=np.maximum(w[::-1],0); U=U[:,::-1]
    total=float(w.sum())
    def rank(frac): return int(np.searchsorted(np.cumsum(w),frac*total)+1) if total>0 else 0
    r=rank(.99); basis=U[:,:r]
    leakage=lam[:,None]*basis-basis@(basis.T@(lam[:,None]*basis))
    return dict(horizon=H,rank99=rank(.99),rank999=rank(.999),
                relative_spectral_rank_1e4=int(np.sum(w>1e-4*w[0])),
                numerical_rank_1e10=int(np.sum(w>1e-10*w[0])),
                gramian_trace=total,eigenvalues=w.tolist(),
                rank99_invariance_defect=float(np.linalg.norm(leakage,2)),
                rank99_tail_fraction=float(w[r:].sum()/total) if total else 0.)


def capture_curve(g,refV,weights):
    vals,V=spectrum(g); current=in_fixed_metric(V,weights)
    reference=in_fixed_metric(refV,weights)
    U=np.linalg.qr(current)[0]
    coefficients=U.T@reference
    cumulative=np.cumsum(coefficients**2,axis=0)
    # Worst principal-direction capture of a reference cluster requires its
    # orthonormal basis, not normalized but nonorthogonal column averages.
    Z=np.linalg.qr(reference)[0]
    curves=[]
    for r in range(1,min(32,len(vals))+1):
        singular=np.linalg.svd(U[:,:r].T@Z,compute_uv=False)
        worst=0. if r<Z.shape[1] else float(singular[-1]**2)
        curves.append(worst)
    found=[i+1 for i,x in enumerate(curves) if x>=.99]
    return dict(rank99=found[0] if found else None,rank_limit=32,
                worst_capture_by_rank=curves,top_reference_capture_by_rank=cumulative[:,0].tolist(),
                cluster_external_gap_at_rank99=float(vals[found[0]-1]-vals[found[0]]) if found and found[0]<len(vals) else None)


def log_second_bound(g,h):
    h=g.center(h); A2=float(g.c@(h*h)); A=math.sqrt(A2); m=float(g.c.min())
    S=A2/math.sqrt(m); R=osc(h); radius=R+A2/m
    if radius==0: logC=-math.inf
    elif 2*radius>50: logC=math.log(5*radius)+2*radius-3*math.log(2*radius)
    else:
        from .energy_theory import quadratic_factor
        logC=math.log(quadratic_factor(radius))
    first=2*A*S+S*S
    logbound=float(np.logaddexp(math.log(first) if first else -math.inf,
                              logC+2*math.log(A+S) if A+S else -math.inf)-math.log(m))
    return dict(amplitude=A,cmin=m,actual_initial_radius=R,certified_jet_radius=radius,
                log10_uniform_bound=logbound/math.log(10),radius_inflation=radius/R if R else 1.)


def median_time(fn,repeats=5):
    fn(); ts=[]
    for _ in range(repeats):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return float(np.median(ts))


def partial_cost(g,V,H):
    """Optimistic supplied-basis cost, excluding acquisition and acceptance."""
    baseline=median_time(lambda:map_value(g.log_kernel,g.a,g.b,g.y),11)
    rows=[]
    J=g.J
    for r in [1,2,4,8,16]:
        basis=V[:,:r]; L=np.zeros((r+r*r,r+r*r))
        Jr=(basis.T*g.c)@J@basis
        L[:r,:r]=Jr; L[r:,r:]=np.kron(Jr,Jr)
        def build():
            for i in range(r):
                for j in range(i,r):
                    v=(basis.T*g.c)@g.bilinear(basis[:,i],basis[:,j])
                    L[:r,r+i*r+j]=v; L[:r,r+j*r+i]=v
        construct=median_time(build,3)
        powered=median_time(lambda:power_and_gramian(L,np.eye(len(L)),H),3)
        rows.append(dict(rank=r,lift_dimension=len(L),bilinear_actions=r*(r+1)//2,
                         construct_seconds=construct,power_seconds=powered,
                         ordinary_step_seconds=baseline,
                         partial_equivalent_steps=(construct+powered)/baseline,
                         proposed_horizon=H,partial_cost_over_H_steps=(construct+powered)/(baseline*H)))
    return rows


def history_prediction(sequence,order,noise,rng):
    train=np.array(sequence[:32],copy=True); truth=np.asarray(sequence[32:])
    scale=float(np.sqrt(np.mean(np.asarray(sequence)**2)))
    train+=rng.normal(size=len(train))*noise*scale
    X=np.array([train[j-order:j][::-1] for j in range(order,len(train))]); target=train[order:]
    coef,_,_,singular=np.linalg.lstsq(X,target,rcond=1e-12)
    path=train.tolist()
    for _ in truth: path.append(float(coef@np.asarray(path[-order:][::-1])))
    err=np.asarray(path[32:])-truth
    return dict(relative_future_rms=float(np.sqrt(np.mean(err*err))/max(np.sqrt(np.mean(truth*truth)),1e-300)),
                absolute_future_rms=float(np.sqrt(np.mean(err*err))),
                train_condition=float(singular[0]/max(singular[-1],1e-300)))


def nested_problem(n):
    rng=np.random.default_rng(731)
    X,Y=rng.random((256,2)),rng.random((256,2))
    ca,cb=rng.random((2,2)),rng.random((3,2))
    C=((X[:,None,:]-Y[None,:,:])**2).sum(-1); C/=C.max()
    def mass(Z,centers):
        v=sum(np.exp(-((Z-z)**2).sum(1)/.02) for z in centers)+.05
        return v/v.sum()
    return C[:n,:n],mass(X[:n],ca),mass(Y[:n],cb)


def run_case(n,seed,eps,density=False):
    cost,a,b=nested_problem(n) if density else source_problem(n,seed)
    logK=-cost/eps; y=np.zeros(n); anchors=[]; initial=None; levels=[.1,.01,.001]
    for k in range(100000):
        new=map_value(logK,a,b,y); new-=new.mean(); residual=osc(new-y)
        if initial is None: initial=residual
        while levels and residual<=initial*levels[0]:
            anchors.append((k,levels.pop(0),y.copy(),residual))
        if residual<1e-12: break
        y=new
    else: raise RuntimeError('reference did not converge')
    ys=new; base=geometry(logK,a,b,ys); lam,V=spectrum(base); gap=1-lam[0]
    H=min(4096,int(math.ceil(1/gap)))
    case=dict(n=n,seed=seed,eps=eps,density=density,reference_iterations=k+1,
              gap=float(gap),horizon=H,relaxation_horizon_capped=H==4096,
              slow_mode_counts={str(t):int(np.sum(lam>=t)) for t in [.5,.9,.99]},anchors=[],observability=[])
    for r in [1,2,4]:
        state=np.eye(n-1)[:r]; rule=rule_outputs(base,V,r)
        for name,O in [('state',state),('rule',rule),('joint',np.vstack((state/np.linalg.norm(state),rule/max(np.linalg.norm(rule),1e-300))))]:
            for horizon in sorted(set([32,H])):
                row=gramian_summary(lam,O,horizon); row.update(output=name,reference_rank=r,output_frobenius=float(np.linalg.norm(O)))
                case['observability'].append(row)
    gamma=rule_outputs(base,V,1)[0]
    for k,level,y,residual in anchors:
        g=geometry(logK,a,b,y); h=base.center(y-ys); alpha=V.T@(base.c*h)
        row=dict(iteration=k,residual_fraction=level,residual=residual,
                 captures={str(r):capture_curve(g,V[:,:r],base.c) for r in [1,2,4]},
                 bound=log_second_bound(base,h),history=[])
        ell=h.copy(); q=np.zeros(n); actual=y.copy(); maxq=maxz=err=0.
        for j in range(64):
            z=ell+q; maxq=max(maxq,osc(q)); maxz=max(maxz,osc(z))
            q=base.J@q+base.bilinear(ell,ell); ell=base.J@ell
            actual=map_value(logK,a,b,actual)
            err=max(err,osc(actual-ys-ell-q))
        row.update(max_observed_q_radius_64=maxq,max_observed_jet_radius_64=maxz,second_jet_error_64=err)
        if not density:
            for stride in [1,max(2,int(round(.1/gap)))]:
                coeff=gamma*alpha
                seq=np.array([np.sum(coeff*lam**(stride*j)) for j in range(64)])
                singular=np.linalg.svd(np.array([seq[j:j+16] for j in range(16)]),compute_uv=False)
                for order in [2,4,8]:
                    for noise in [0.,1e-8,1e-6]:
                        stats=[history_prediction(seq,order,noise,np.random.default_rng(200+s)) for s in range(3 if noise else 1)]
                        row['history'].append(dict(stride=stride,order=order,noise_relative_to_window_rms=noise,
                                                   signal_rms=float(np.sqrt(np.mean(seq*seq))),
                                                   future_signal_rms=float(np.sqrt(np.mean(seq[32:]**2))),
                                                   worst_window_normalized_future_rms=max(s['absolute_future_rms'] for s in stats)/max(float(np.sqrt(np.mean(seq*seq))),1e-300),
                                                   hankel_singular_values=singular.tolist(),
                                                   median_relative_future_rms=float(np.median([s['relative_future_rms'] for s in stats])),
                                                   worst_relative_future_rms=max(s['relative_future_rms'] for s in stats),
                                                   median_train_condition=float(np.median([s['train_condition'] for s in stats]))))
        case['anchors'].append(row)
    case['heat_norms']=[]
    for j in sorted(set([0,1,8,32,128,H])):
        norms=np.sqrt(np.sum(V*V*lam[None,:]**(2*j),axis=1))
        case['heat_norms'].append(dict(steps=j,centered_c2_to_infinity=float(norms.max()),
                                       trivial_c2_to_infinity=1/math.sqrt(base.c.min())))
    if not density:
        middle=anchors[1][2]; g=geometry(logK,a,b,middle); _,currentV=spectrum(g)
        case['partial_cost']=partial_cost(g,currentV,H)
    print('finished',n,seed,eps,'density',density,'steps',case['reference_iterations'],flush=True)
    return case


def duplication_probe():
    """Refine the representation without changing any physical transition."""
    K=np.array([[.50,.35,.15],[.20,.45,.35],[.30,.20,.50]])
    mass=np.ones(3)/3; h=.2*np.array([1.,-.6,-.4]); rows=[]
    for copies in [1,2,4,8,16,32]:
        kernel=np.repeat(np.repeat(K,copies,axis=0),copies,axis=1)
        c=np.repeat(mass/copies,copies); v=np.repeat(h,copies)
        g=geometry(np.log(kernel),c,c,np.zeros(len(c)))
        ell=v.copy(); q=np.zeros(len(c)); actual=v.copy(); err=0.
        for _ in range(64):
            q=g.J@q+g.bilinear(ell,ell); ell=g.J@ell
            actual=map_value(g.log_kernel,c,c,actual)
            err=max(err,osc(actual-ell-q))
        J=g.J
        heat=float(np.sqrt(np.maximum(np.sum(J*J/c[None,:],axis=1)-1,0)).max())
        row=log_second_bound(g,v)
        row.update(copies=copies,states=len(c),energy=g.energy(v),second_jet_error_64=err,one_step_centered_c2_to_infinity=heat)
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); parser.add_argument('--duplication-only',action='store_true'); args=parser.parse_args()
    if args.duplication_only:
        Path(args.out).write_text(json.dumps(dict(scope='identical physical chain represented with duplicated atoms',cases=duplication_probe()),indent=2,allow_nan=False))
        return
    rows=[]
    for seed in [0,1]:
        for eps in [.03,.01,.003,.001]: rows.append(run_case(128,seed,eps))
    for n in [64,128,256]:
        for eps in [.003,.001]: rows.append(run_case(n,731,eps,True))
    result=dict(scope='oracle diagnostics and partial costs, not acceleration',
                anchor_rule='first observable residual crossings of .1, .01, .001 times initial; reference fixed point used only in scoring',
                history_scope='favorable oracle linearized rule signal; excludes moving-anchor error, includes relative observation-noise sensitivity',
                density_scope='nested 2D clouds, same centers and cost normalization; one seed',cases=rows)
    Path(args.out).write_text(json.dumps(result,indent=2,allow_nan=False))
    print('saved',args.out,flush=True)


if __name__=='__main__': main()
