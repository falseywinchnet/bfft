"""Apply exact directional-capture limits to the SIX original failed cases.

A finite sample covariance describes only that stated sample ensemble. A
finite-horizon observability Gramian describes squared prediction activity
averaged over isotropic unit covariance input perturbations in the fixed-point
metric. Neither establishes a lower bound for arbitrary nonlinear algorithms.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .energy_theory import geometry
from .study_energy_theory import source_problem, spectrum, map_value, osc
from .probe_claude_objections import rule_outputs, observability_gramian


def spectral_summary(M):
    values=np.maximum(np.linalg.eigvalsh((M+M.T)/2)[::-1],0)
    total=float(values.sum())
    if total==0:
        return dict(trace=0.,effective_dimension=0.,ranks={},capture=[],eigenvalues=values.tolist())
    fractions=values/total; capture=np.cumsum(fractions); effective=1/fractions[0]
    ranks={str(eta):int(np.searchsorted(capture,eta)+1) for eta in [.9,.95,.99,.999,.9999]}
    return dict(trace=total,effective_dimension=float(effective),
                best_scalar_capture=float(fractions[0]),best_scalar_rms_loss=float(np.sqrt(max(0,1-fractions[0]))),
                ranks=ranks,necessary_rank99_from_effective_dimension=int(math.ceil(.99*effective-1e-12)),
                capture=capture.tolist(),eigenvalues=values.tolist())


def activity_summary(samples,normalized=False):
    X=np.asarray(samples); norms=np.linalg.norm(X,axis=1)
    if normalized:
        keep=norms>max(float(norms.max())*1e-12,1e-300)
        X=X[keep]/norms[keep,None]
    M=X.T@X/len(X); row=spectral_summary(M); mean=X.mean(axis=0)
    dispersion=float(np.mean(np.sum((X-mean)**2,axis=1)))
    row.update(sample_count=len(X),equal_weight_samples=True,normalized_directions=normalized,
               mean_squared_norm=float(mean@mean),dispersion=dispersion,
               coherence=float(mean@mean/row['trace']) if row['trace'] else None,
               cancellation_identity_error=abs(row['trace']-float(mean@mean)-dispersion),
               minimum_original_norm=float(norms.min()),maximum_original_norm=float(norms.max()))
    return row


def oracle_replay(lam,O,alpha,H):
    """Best input directions are oracle Gramian directions; projected dynamics
    must still carry them. Also measure a leading-eigenmode subspace, which is
    invariant and has exact projected linear dynamics.
    """
    W=observability_gramian(lam,O,H); row=spectral_summary(W)
    values,U=np.linalg.eigh((W+W.T)/2); U=U[:,::-1]
    total=row['trace']; row['leading_mode_capture']=float(W[0,0]/total)
    alpha=np.asarray(alpha); j=np.arange(H)
    states=alpha[:,None]*lam[:,None]**j[None,:]
    target=O@states; actual_energy=float(np.sum(target*target)); trials=[]
    for rank in [1,2,4,8,16]:
        rank=min(rank,len(lam)); basis=U[:,:rank]
        # Best Gramian projection of the initial displacement, evolved in full
        # space: a favorable oracle, NOT a closed rank-r computational scheme.
        init=basis@(basis.T@alpha)
        projected=O@(init[:,None]*lam[:,None]**j[None,:])
        # Closed Galerkin evolution on the same learned directions.
        T=basis.T@(lam[:,None]*basis); reduced=basis.T@alpha
        path=[]
        for _ in range(H):
            path.append(O@basis@reduced); reduced=T@reduced
        closed=np.array(path).T
        modes=O[:,:rank]@states[:rank]
        denom=max(actual_energy,1e-300)
        trials.append(dict(rank=rank,
                           isotropic_input_rms_floor=float(np.sqrt(max(0,1-row['capture'][rank-1]))),
                           actual_orbit_oracle_projection_relative_rms=float(np.linalg.norm(projected-target)/math.sqrt(denom)),
                           actual_orbit_closed_relative_rms=float(np.linalg.norm(closed-target)/math.sqrt(denom)),
                           actual_orbit_leading_modes_relative_rms=float(np.linalg.norm(modes-target)/math.sqrt(denom)),
                           invariance_defect=float(np.linalg.norm(lam[:,None]*basis-basis@T,2))))
    row.update(orbit_output_energy=actual_energy,trials=trials)
    return row


def affine_minimax_three(x,y):
    """Exact minimax residual of any affine y=b+s*x on three distinct x's.

    w is orthogonal to both 1 and x. Necessity: |w.y|<=||w||_1 ||res||inf.
    Sufficiency: res=sign(w.y)*sign(w)*bound makes y-res affine in x.
    """
    x,y=np.asarray(x),np.asarray(y)
    w=np.array([x[1]-x[2],x[2]-x[0],x[0]-x[1]])
    bound=float(abs(w@y)/np.abs(w).sum())
    residual=np.sign(w@y)*np.sign(w)*bound
    adjusted=y-residual
    slope=(adjusted[1]-adjusted[0])/(x[1]-x[0]); intercept=adjusted[0]-slope*x[0]
    return dict(absolute_minimax_error=bound,
                error_over_last_observed_change=bound/max(abs(float(y[2]-y[1])),1e-300),
                optimal_slope=float(slope),optimal_intercept=float(intercept),
                attaining_errors=(y-(intercept+slope*x)).tolist())


def fit_three(x,y):
    x,y=np.asarray(x),np.asarray(y)
    slope=(y[1]-y[0])/(x[1]-x[0]); pred=y[0]+slope*(x[2]-x[0])
    return dict(predicted_third=float(pred),actual_third=float(y[2]),absolute_error=float(abs(pred-y[2])),
                error_over_last_observed_change=float(abs(pred-y[2])/max(abs(y[2]-y[1]),1e-300)))


def run_record(record):
    seed,eps=record['seed'],record['eps']; n=128
    cost,a,b=source_problem(n,seed); logK=-cost/eps
    y=np.zeros(n); trajectory=[y.copy()]
    for _ in range(400000):
        new=map_value(logK,a,b,y); new-=new.mean(); trajectory.append(new.copy())
        if osc(new-y)<1e-13: break
        y=new
    else: raise RuntimeError('reference convergence failed')
    base=geometry(logK,a,b,trajectory[-1]); lam,V=spectrum(base)
    ks=record['anchors']; rule=rule_outputs(base,V,4)
    scalar=rule_outputs(base,V,1)
    # Reference modes, metric, and analytic rule derivative are scoring oracles.
    signals={name:[] for name in ['state_displacement','actual_increment','quadratic_forcing','actual_rule_block4','linearized_rule_block4']}
    anchor_rows=[]
    for k in range(ks[0],ks[-1]+1):
        y=trajectory[k]; h=base.center(y-base.y); alpha=V.T@(base.c*h)
        inc=base.center(trajectory[k+1]-y)
        forcing=base.center(base.bilinear(h,h))
        current=geometry(logK,a,b,y)
        block=(V[:,:4].T*base.c)@(current.Q@(current.P@V[:,:4]))-np.diag(lam[:4])
        # rule_outputs has j outer, i inner: column-major vectorization.
        signals['state_displacement'].append(alpha)
        signals['actual_increment'].append(V.T@(base.c*inc))
        signals['quadratic_forcing'].append(V.T@(base.c*forcing))
        signals['actual_rule_block4'].append(block.T.ravel())
        signals['linearized_rule_block4'].append(rule@alpha)
        if k in ks:
            actual_eig,_=spectrum(current)
            anchor_rows.append(dict(iteration=k,slow_amplitude=float(alpha[0]),
                                    actual_leading_eigenvalue=float(actual_eig[0]),
                                    reference_rayleigh=float(lam[0]+block[0,0]),
                                    first_order_all_modes=float(lam[0]+(scalar@alpha)[0]),
                                    first_order_slow_only=float(lam[0]+scalar[0,0]*alpha[0])))
    activity={name:{'amplitude_weighted':activity_summary(xs),'direction_only':activity_summary(xs,True)} for name,xs in signals.items()}
    h=base.center(trajectory[ks[1]]-base.y); alpha=V.T@(base.c*h)
    horizons=sorted(set([ks[-1]-ks[1]+1,int(math.ceil(1/(1-lam[0]))),record['H']]))
    observability=[]
    X=np.asarray(signals['state_displacement'])
    Sigma=X.T@X/len(X)
    w,U=np.linalg.eigh(Sigma); root=(U*np.sqrt(np.maximum(w,0)))@U.T
    directions=X/np.linalg.norm(X,axis=1)[:,None]
    Sigma_unit=directions.T@directions/len(directions)
    w,U=np.linalg.eigh(Sigma_unit); root_unit=(U*np.sqrt(np.maximum(w,0)))@U.T
    for H in horizons:
        for r in [1,2,4]:
            O=rule_outputs(base,V,r)
            row=oracle_replay(lam,O,alpha,H)
            W=observability_gramian(lam,O,H)
            row['empirical_excitation']=spectral_summary(root@W@root)
            row['unit_direction_excitation']=spectral_summary(root_unit@W@root_unit)
            row.update(horizon=H,reference_rule_rank=r,output='rule_only',output_rows=r*r)
            observability.append(row)
    x=[r['slow_amplitude'] for r in anchor_rows]
    perfect_eigen=fit_three(x,[r['actual_leading_eigenvalue'] for r in anchor_rows])
    perfect_reference=fit_three(x,[r['reference_rayleigh'] for r in anchor_rows])
    original=dict(predicted_third=record['theta3_pred'],actual_third=record['theta3_obs'],
                  error_over_last_observed_change=abs(record['theta3_pred']-record['theta3_obs'])/abs(record['theta'][2]-record['theta'][1]))
    result=dict(seed=seed,eps=eps,n=n,anchors=ks,reference_iterations=len(trajectory)-1,
                original_proposed_horizon=record['H'],true_relaxation_horizon=int(math.ceil(1/(1-lam[0]))),
                anchor_values=anchor_rows,activity=activity,observability=observability,
                affine_fit_original=original,affine_fit_exact_amplitudes_and_eigenvalues=perfect_eigen,
                affine_fit_exact_amplitudes_and_reference_rayleigh=perfect_reference,
                affine_minimax_exact_eigenvalues=affine_minimax_three(x,[r['actual_leading_eigenvalue'] for r in anchor_rows]),
                affine_minimax_exact_reference_rayleigh=affine_minimax_three(x,[r['reference_rayleigh'] for r in anchor_rows]))
    print('done',seed,eps,'scalar best capture',[(r['horizon'],round(r['best_scalar_capture'],4)) for r in observability if r['reference_rule_rank']==1],flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); args=ap.parse_args()
    source=Path(__file__).parent/'results/theory/acquisition_input.json'
    records=json.loads(source.read_text())
    result=dict(scope='original six failed scalar acquisition cases; ordinary Sinkhorn converges; exact directional limits for stated linear output families and finite activity ensembles',
                observability_input_metric='identity covariance on centered fixed-point c-orthonormal eigen-coordinates',
                activity_window='every actual trajectory step from original first through original third anchor, equal temporal weights',
                direction_normalization='separate amplitude-weighted and unit-direction spectra; no mixture-count monotonicity assumed',
                cases=[run_record(r) for r in records])
    Path(args.out).write_text(json.dumps(result,indent=2,allow_nan=False))
    print('saved',args.out,flush=True)


if __name__=='__main__': main()
