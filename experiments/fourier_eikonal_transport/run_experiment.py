#!/usr/bin/env python3
"""Multiview Eikonal transport on Fourier shells.

The experiment reconstructs a conjugate-symmetric polar Fourier field from
several noisy, partially masked observations. Cross-observation agreement
estimates a complex connection between neighboring angular coefficients and an
edge cost. A redundant seed atlas transports values around each shell.
"""

from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path
import numpy as np


def smooth_random(rng, n, cutoff=7):
    coeff=np.zeros(n,dtype=complex)
    for k in range(cutoff+1):
        scale=1/(1+k)**1.4
        coeff[k]=scale*(rng.normal()+1j*rng.normal())
        if k: coeff[-k]=np.conj(coeff[k])
    return np.fft.ifft(coeff).real*n


def make_field(radii=24, angles=96, seed=0):
    """Smooth oriented spectral sections with exact real-image symmetry."""
    rng=np.random.default_rng(seed); half=angles//2
    theta=np.arange(half)*2*np.pi/angles
    field=np.zeros((radii,angles),complex)
    centers=rng.uniform(0,np.pi,4); widths=rng.uniform(.16,.45,4)
    phase_base=smooth_random(rng,half,5)
    for r in range(radii):
        rr=(r+1)/radii
        envelope=np.exp(-2.6*rr)*(1-np.exp(-7*rr))
        amp=.05+sum((.4+.6*np.cos((j+1)*rr*np.pi)**2)*
                    np.exp(-.5*(np.angle(np.exp(1j*(theta-c))))**2/w**2)
                    for j,(c,w) in enumerate(zip(centers,widths)))
        amp*=envelope*(1+.12*smooth_random(rng,half,4))
        phase=.35*phase_base + (1.2+2.5*rr)*theta + .15*r*smooth_random(rng,half,3)
        first=np.maximum(amp,1e-4)*np.exp(1j*phase)
        field[r,:half]=first
        field[r,half:]=np.conj(first)
    return field


def circular_wedge_mask(angles, center, width):
    t=np.arange(angles)*2*np.pi/angles
    delta=np.abs(np.angle(np.exp(1j*(t-center))))
    return delta>width/2


def observations(clean, views=5, noise=.08, seed=0):
    rng=np.random.default_rng(seed); R,T=clean.shape
    masks=np.ones((views,R,T),bool)
    # A shared blind wedge forces actual continuation. Its antipode preserves
    # the transfer symmetry of a real image.
    shared=rng.uniform(0,np.pi); shared_width=rng.uniform(.30,.48)
    shared_mask=circular_wedge_mask(T,shared,shared_width)&circular_wedge_mask(T,shared+np.pi,shared_width)
    for a in range(views):
        masks[a]&=shared_mask[None]
        for _ in range(2):
            c=rng.uniform(0,np.pi); w=rng.uniform(.16,.34)
            m=circular_wedge_mask(T,c,w)&circular_wedge_mask(T,c+np.pi,w)
            # Vary private wedge extent with radius.
            cutoff=rng.integers(R//3,R+1)
            masks[a,:cutoff]&=m[None]
        drop=rng.random((R,T))<.06
        drop[:,T//2:]=drop[:,:T//2]
        masks[a]&=~drop
    sigma=noise*np.sqrt(np.mean(np.abs(clean)**2))
    eps=sigma/math.sqrt(2)*(rng.normal(size=(views,R,T))+1j*rng.normal(size=(views,R,T)))
    y=masks*clean[None]+masks*eps
    return y,masks,sigma


def circular_interpolate(values, known):
    n=len(values); idx=np.flatnonzero(known)
    if len(idx)==0: return np.zeros(n)
    if len(idx)==1: return np.full(n,values[idx[0]])
    extended=np.r_[idx[-1]-n,idx,idx[0]+n]
    vals=np.r_[values[idx[-1]],values[idx],values[idx[0]]]
    return np.interp(np.arange(n),extended,vals)


def shell_evidence(y, masks):
    """Fused coefficients, reliability, complex connection, and edge costs."""
    count=masks.sum(0); fused=np.divide(y.sum(0),count,out=np.zeros_like(y[0]),where=count>0)
    R,T=fused.shape; edge_phase=np.empty((R,T)); edge_logamp=np.empty((R,T)); cost=np.empty((R,T))
    for r in range(R):
        left=y[:,r]; right=np.roll(left,-1,axis=1)
        joint=masks[:,r]&np.roll(masks[:,r],-1,axis=1)
        cross=(right*np.conj(left)*joint).sum(0)
        el=(np.abs(left)**2*joint).sum(0); er=(np.abs(right)**2*joint).sum(0)
        coherence=np.divide(np.abs(cross),np.sqrt(el*er)+1e-12)
        phase=np.angle(cross); logamp=.5*np.log((er+1e-10)/(el+1e-10))
        known=joint.sum(0)>0
        # Phase increments are local and can be unwrapped safely before gaps.
        phase_known=np.unwrap(phase[np.flatnonzero(known)]) if known.any() else np.array([])
        phase_fill=np.zeros(T)
        if known.any():
            temp=np.zeros(T); temp[known]=phase_known
            phase_fill=circular_interpolate(temp,known)
        edge_phase[r]=phase_fill
        edge_logamp[r]=circular_interpolate(logamp,known)
        joint_count=joint.sum(0)
        cost[r]=(1+.8/(joint_count+.25))/(coherence+.08)
        cost[r,~known]*=3.5
    # Enforce antipodal connection symmetry softly.
    half=T//2
    edge_phase[:,half:]=-edge_phase[:,:half]
    edge_logamp[:,half:]=edge_logamp[:,:half]
    cost[:,half:]=cost[:,:half]
    return fused,count,edge_phase,edge_logamp,cost


def select_seeds(reliability, number=10):
    n=len(reliability); score=reliability.astype(float).copy(); seeds=[]
    for _ in range(number):
        s=int(np.argmax(score)); seeds.append(s)
        d=np.minimum((np.arange(n)-s)%n,(s-np.arange(n))%n)
        score[d<max(2,n//number//2)]=-1
    return seeds


def path_transport(seed, target, value, phase, logamp, cost, metric=True):
    n=len(phase)
    cw=(target-seed)%n; ccw=(seed-target)%n
    cw_edges=(seed+np.arange(cw))%n
    ccw_edges=(target+np.arange(ccw))%n
    dcw=float(cost[cw_edges].sum()) if metric else float(cw)
    dccw=float(cost[ccw_edges].sum()) if metric else float(ccw)
    if dcw<=dccw:
        lp=logamp[cw_edges].sum(); ph=phase[cw_edges].sum(); distance=dcw
    else:
        lp=-logamp[ccw_edges].sum(); ph=-phase[ccw_edges].sum(); distance=dccw
    lp=float(np.clip(lp,-5,5))
    return value*np.exp(lp+1j*ph),distance


def atlas_transport(fused,reliability,phase,logamp,cost,mode='eikonal',seeds=10):
    R,T=fused.shape; out=np.zeros_like(fused)
    for r in range(R):
        seed_idx=select_seeds(reliability[r],seeds)
        observed=np.abs(fused[r,reliability[r]>0])
        amp_lo=max(float(np.quantile(observed,.05))*.5,1e-8)
        amp_hi=max(float(np.quantile(observed,.95))*1.5,amp_lo)
        if mode=='shuffled':
            rng=np.random.default_rng(8000+r); active_cost=cost[r,rng.permutation(T)]
        else: active_cost=cost[r]
        scale=np.median(active_cost)*T/max(seeds,1)
        for t in range(T):
            estimates=[]; weights=[]
            for s in seed_idx:
                est,dist=path_transport(s,t,fused[r,s],phase[r],logamp[r],active_cost,mode!='isotropic')
                # Non-unitary amplitude errors compound along a path. Keep the
                # transported section inside the shell's observed amplitude
                # envelope; phase remains genuinely connection-transported.
                est=np.clip(np.abs(est),amp_lo,amp_hi)*np.exp(1j*np.angle(est))
                estimates.append(est); weights.append((reliability[r,s]+.2)*np.exp(-dist/(scale+1e-8)))
            weights=np.maximum(weights,1e-12); estimates=np.asarray(estimates)
            transported=np.average(estimates,weights=weights)
            agreement=np.abs(np.sum(weights*estimates/(np.abs(estimates)+1e-12)))/np.sum(weights)
            transported*=agreement
            direct_weight=reliability[r,t]/(reliability[r,t]+2.0)
            out[r,t]=direct_weight*fused[r,t]+(1-direct_weight)*transported
    half=T//2; out[:,half:]=np.conj(out[:,:half])
    return out


def radial_control(fused,reliability):
    out=fused.copy(); R,T=out.shape
    for r in range(R):
        known=reliability[r]>0
        if known.any():
            # Radial pooling has no justified phase for a missing direction.
            magnitude=np.mean(np.abs(fused[r,known]))
            out[r,~known]=magnitude
    return out


def metrics(estimate,clean,reliability):
    missing=reliability==0
    scale=np.sum(np.abs(clean)**2)+1e-12
    mse=np.sum(np.abs(estimate-clean)**2)/scale
    mmse=np.sum(np.abs(estimate[missing]-clean[missing])**2)/(np.sum(np.abs(clean[missing])**2)+1e-12)
    corr=np.abs(np.vdot(estimate,clean))/(np.linalg.norm(estimate)*np.linalg.norm(clean)+1e-12)
    return {'nmse':float(mse),'missing_nmse':float(mmse),'complex_correlation':float(corr),
            'missing_fraction':float(missing.mean())}


def run_trial(seed,R,T,views,noise):
    clean=make_field(R,T,seed); y,masks,sigma=observations(clean,views,noise,10000+seed)
    fused,reliability,phase,logamp,cost=shell_evidence(y,masks)
    estimates={
        'direct_fusion':fused,
        'radial_fill':radial_control(fused,reliability),
        'isotropic_atlas':atlas_transport(fused,reliability,phase,logamp,cost,'isotropic'),
        'shuffled_metric':atlas_transport(fused,reliability,phase,logamp,cost,'shuffled'),
        'eikonal_atlas':atlas_transport(fused,reliability,phase,logamp,cost,'eikonal'),
    }
    return clean,reliability,estimates,{k:metrics(v,clean,reliability) for k,v in estimates.items()}


def write_example_npz(path,clean,reliability,estimates):
    np.savez_compressed(path,clean=clean,reliability=reliability,**estimates)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('experiments/fourier_eikonal_transport/results'))
    ap.add_argument('--seeds',type=int,default=40); ap.add_argument('--radii',type=int,default=24)
    ap.add_argument('--angles',type=int,default=96); ap.add_argument('--views',type=int,default=5)
    ap.add_argument('--noise',type=float,default=.10); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); start=time.time(); rows=[]
    for seed in range(args.seeds):
        clean,rel,est,trial=run_trial(seed,args.radii,args.angles,args.views,args.noise)
        for model,values in trial.items(): rows.append({'seed':seed,'model':model,**values})
        if seed==0: write_example_npz(args.out/'example.npz',clean,rel,est)
        print(json.dumps({'seed':seed,'metrics':trial}),flush=True)
    summary=[]
    for model in sorted({r['model'] for r in rows}):
        rs=[r for r in rows if r['model']==model]
        summary.append({'model':model,
            'nmse_mean':float(np.mean([r['nmse'] for r in rs])), 'nmse_std':float(np.std([r['nmse'] for r in rs])),
            'nmse_median':float(np.median([r['nmse'] for r in rs])),
            'nmse_p90':float(np.quantile([r['nmse'] for r in rs],.9)),
            'missing_nmse_mean':float(np.mean([r['missing_nmse'] for r in rs])),
            'missing_nmse_std':float(np.std([r['missing_nmse'] for r in rs])),
            'missing_nmse_median':float(np.median([r['missing_nmse'] for r in rs])),
            'missing_nmse_p90':float(np.quantile([r['missing_nmse'] for r in rs],.9)),
            'correlation_mean':float(np.mean([r['complex_correlation'] for r in rs])),
            'win_rate_vs_isotropic':float(np.mean([r['missing_nmse'] < next(q['missing_nmse'] for q in rows if q['seed']==r['seed'] and q['model']=='isotropic_atlas') for r in rs]))})
    with (args.out/'runs.json').open('w') as f: json.dump({'runtime_seconds':time.time()-start,'config':vars(args)|{'out':str(args.out)},'runs':rows},f,indent=2)
    with (args.out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    print(json.dumps({'runtime_seconds':time.time()-start,'summary':summary},indent=2))


if __name__=='__main__': main()
