"""Benefit/cost of smoothly increasing irregular observation density.

One retained relational Kalman system uses every observation once. The ordinary
covariance-weighted innovation is the shortening rule in this first acquisition
experiment. Policies differ only in sampling cadence, not in tracker selection.
"""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from .adaptive_acquisition import BoundarySampler, BoundaryFixture
from .relational_kalman import RelationalKalman, design
from .data import FAMILIES
from .study import save

EVALUATION=np.linspace(0,12,121)
POLICIES=('constant_2','adaptive_4','adaptive_8','adaptive_16','adaptive_32','constant_32')


def current_mean(model,t):
    if hasattr(model,'position'):
        return model.position(t)
    weights=np.exp(model.logs-logsumexp(model.logs))
    mean=np.einsum('k,knd->nd',weights,np.array(model.means))
    row=design([t],model.end,model.cells) @ model.u.conj().T
    return (row @ mean).real[0]+model.anchor


def simulate(fixture,sigma,policy,seed,cells=128,keep_trace=False,tracker_class=RelationalKalman):
    rng=np.random.default_rng(seed+3100000)
    is_constant=policy.startswith('constant')
    target=float(policy.split('_')[-1])
    sampler=BoundarySampler(low_rate=2.,high_rate=max(2.,target),ramp_seconds=1.,jitter=.2,
                            seed=seed+3200000,constant_rate=target if is_constant else None)
    y=fixture.position(0.)+sigma*rng.normal(size=3)
    start=time.perf_counter_ns()
    model=tracker_class(y,sigma,end=14.,cells=cells)
    initialization_ns=time.perf_counter_ns()-start
    sampler.observe_side(0.,float(y[0]))
    start=time.perf_counter_ns();next_sample=sampler.next_time(0.);scheduler_ns=time.perf_counter_ns()-start
    update_ns=0;readout_ns=0;diagnostic_ns=0
    event_records=[dict(time=0.,observation=y,update_ms=0.,readout_ms=0.,correction=0.,innovation=0.,projected_gain=0.,requested_rate=sampler.rate(0.))]
    estimates=[];forecasts=[];radii=[];coverage=[];forecast_coverage=[];forecast_radii=[]
    for now in EVALUATION:
        while next_sample<=now+1e-12:
            at=float(next_sample)
            observation=fixture.position(at)+sigma*rng.normal(size=3)
            start=time.perf_counter_ns();before=current_mean(model,at);diagnostic_ns+=time.perf_counter_ns()-start
            start=time.perf_counter_ns();model.update(at,observation);cost=time.perf_counter_ns()-start;update_ns+=cost
            start=time.perf_counter_ns();after=current_mean(model,at);read_cost=time.perf_counter_ns()-start;readout_ns+=read_cost
            # Trigger from the tracker, at an acquired observation, never clean truth.
            sampler.observe_side(at,float(after[0]))
            delta=after-before;innovation=observation-before
            norm2=float(innovation @ innovation)
            event_records.append(dict(time=at,observation=observation,estimate=after,
                update_ms=cost/1e6,readout_ms=read_cost/1e6,correction=float(np.linalg.norm(delta)),
                innovation=float(np.sqrt(norm2)),projected_gain=float(delta @ innovation/norm2) if norm2 else 0.,
                requested_rate=sampler.rate(at)))
            start=time.perf_counter_ns();next_sample=sampler.next_time(at);scheduler_ns+=time.perf_counter_ns()-start
        start=time.perf_counter_ns();result=model.forecast([float(now),float(now+2.)]);readout_ns+=time.perf_counter_ns()-start
        estimates.append(result['mean'][0]);forecasts.append(result['mean'][1])
        error=result['mean'][0]-fixture.position(now)
        forecast_error=result['mean'][1]-fixture.position(now+2.)
        cov=result['covariance']
        coverage.append(bool(error @ np.linalg.solve(cov[0],error)<=7.814727903))
        forecast_coverage.append(bool(forecast_error @ np.linalg.solve(cov[1],forecast_error)<=7.814727903))
        radii.append(float(np.sqrt(7.814727903)*np.linalg.det(cov[0])**(1/6)))
        forecast_radii.append(float(np.sqrt(7.814727903)*np.linalg.det(cov[1])**(1/6)))
    estimates=np.array(estimates);forecasts=np.array(forecasts)
    truth=fixture.position(EVALUATION);future_truth=fixture.position(EVALUATION+2.)
    track_sq=np.sum((estimates-truth)**2,axis=1)
    forecast_sq=np.sum((forecasts-future_truth)**2,axis=1)
    cross=fixture.crossing_time
    stages={'approach':(-2.,0.),'first_second':(0.,1.),'settling':(1.,3.),'later':(3.,6.),'post':(0.,6.)}
    stage_records={}
    for name,(a,b) in stages.items():
        mask=(EVALUATION>=cross+a)&(EVALUATION<cross+b)
        events=[e for e in event_records if cross+a<=e['time']<cross+b]
        duration=max(0.,min(12.,cross+b)-max(0.,cross+a))
        stage_records[name]=dict(track_sq=float(np.mean(track_sq[mask])),forecast_sq=float(np.mean(forecast_sq[mask])),
            mean_radius=float(np.mean(np.array(radii)[mask])),coverage=float(np.mean(np.array(coverage)[mask])),
            mean_forecast_radius=float(np.mean(np.array(forecast_radii)[mask])),
            forecast_coverage=float(np.mean(np.array(forecast_coverage)[mask])),samples=len(events),
            update_ms=sum(e['update_ms'] for e in events),
            readout_ms=sum(e['readout_ms'] for e in events),duration=duration,
            mean_correction=float(np.mean([e['correction'] for e in events])) if events else 0.,
            max_correction=max([e['correction'] for e in events],default=0.),
            mean_projected_gain=float(np.mean([e['projected_gain'] for e in events])) if events else 0.)
    trace=dict(times=EVALUATION,truth=truth,estimates=estimates,forecasts=forecasts,future_truth=future_truth,
               track_sq=track_sq,forecast_sq=forecast_sq,radii=radii,events=event_records,
               crossing_time=cross,trigger_time=sampler.trigger_time,
               rate=[sampler.rate(float(t)) for t in EVALUATION]) if keep_trace else None
    result=dict(policy=policy,sigma=sigma,cells=cells,stages=stage_records,actual_crossing=cross,
                trigger_time=sampler.trigger_time,
                trigger_delay=None if sampler.trigger_time is None else sampler.trigger_time-cross,
                samples=len(event_records),updates=len(event_records)-1,initialization_ms=initialization_ns/1e6,
                update_ms=update_ns/1e6,readout_ms=readout_ns/1e6,scheduler_ms=scheduler_ns/1e6,
                diagnostic_ms=diagnostic_ns/1e6,
                total_online_ms=(initialization_ns+update_ns+readout_ns+scheduler_ns)/1e6,
                mean_update_ms=update_ns/1e6/(len(event_records)-1),
                maximum_correction=max(e['correction'] for e in event_records),
                trace=trace)
    return result


def run(out,seeds=8):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[];traces=[];audits=[]
    started=time.perf_counter()
    for fi,family in enumerate(FAMILIES):
        for repeat in range(seeds):
            seed=110000+1000*fi+repeat
            fixture=BoundaryFixture(family,seed)
            for sigma in (.35,1.4):
                results=[]
                for policy in POLICIES:
                    # Retain two example trajectories plus all aggregate records.
                    r=simulate(fixture,sigma,policy,seed,keep_trace=repeat==0)
                    trace=r.pop('trace')
                    r.update(family=family,seed=seed)
                    rows.append(r);results.append(r)
                    if trace is not None:
                        trace.update(family=family,seed=seed,sigma=sigma,policy=policy)
                        traces.append(trace)
                # All adaptive rates share exactly the same low-rate history and
                # trigger. This checks the rate is not leaking into state inference.
                triggers=[r['trigger_time'] for r in results if r['policy'].startswith('adaptive')]
                assert all(t==triggers[0] for t in triggers)
                assert all(r['trigger_time'] is not None for r in results)
            save(out/'partial.json',dict(runs=rows))
            print(f'{family} {repeat+1}/{seeds} elapsed {time.perf_counter()-started:.1f}s',flush=True)
    # Independent representation check after the frozen main run; no selection.
    for family in ('helix','switching'):
        fi=FAMILIES.index(family);seed=110000+1000*fi
        fixture=BoundaryFixture(family,seed)
        for policy in ('constant_2','adaptive_32'):
            base=next(r for r in rows if (r['family'],r['seed'],r['sigma'],r['policy'])==(family,seed,.35,policy))
            fine=simulate(fixture,.35,policy,seed,cells=256)
            audits.append(dict(family=family,policy=policy,base_cells=128,fine_cells=256,
                post_track_rmse_difference=float(np.sqrt(fine['stages']['post']['track_sq'])-np.sqrt(base['stages']['post']['track_sq'])),
                post_forecast_rmse_difference=float(np.sqrt(fine['stages']['post']['forecast_sq'])-np.sqrt(base['stages']['post']['forecast_sq'])),
                trigger_difference=float(fine['trigger_time']-base['trigger_time']),
                sample_count_difference=fine['samples']-base['samples']))
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['adaptive_acquisition.py','acquisition_study.py','relational_kalman.py','data.py']}
    save(out/'results.json',dict(metadata=dict(seeds=seeds,seed_base=110000,cells=128,
          interval=[0.,14.],evaluated_interval=[0.,12.],forecast_horizon=2.,noise_sigmas=[.35,1.4],
          low_rate=2.,high_rates=[4.,8.,16.,32.],ramp_seconds=1.,phase_jitter=.2,
          trigger='estimated x>=0 after an acquired observation, latched',source_hashes=hashes,
          update='one ordinary covariance-weighted relational Kalman update per observation',
          price='acquisition cost is reported as counts, not assigned a hardware price',
          timing='measured M4 single-thread scopes; initialization, assimilation, readout, scheduler separate'),
          runs=rows,resolution_audits=audits))
    save(out/'traces.json',traces)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--seeds',type=int,default=8)
    args=parser.parse_args()
    run(args.out,args.seeds)
