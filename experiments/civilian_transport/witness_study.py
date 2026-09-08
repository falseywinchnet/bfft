"""Explore locally witnessed correction and the limit of its continuation.

No BTB algorithm is used. No threshold is selected using clean trajectories.
The sensor-drift/maneuver pair has exactly identical measurements by design.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import chi2, norm
from .data import FAMILIES, make_case
from .relational_kalman import RelationalKalman
from .witness_geometry import supported_subspace, supported_ray, clean_gain
from .study import save

TIMES=np.linspace(0,8,81)
PREFIX=np.arange(41)
TRAIN=np.arange(42,61,2)
WITNESS=np.arange(41,60,2)
FUTURE=np.arange(61,81)
KINDS=('ordinary','sensor_spike','maneuver','sensor_drift')


def fixture(family, seed, sigma, kind):
    case=make_case(family,seed,samples=len(TIMES),observation_times=TIMES)
    base=case['truth']
    rng=np.random.default_rng(seed+1900000)
    noise=sigma*rng.normal(size=base.shape)
    direction=rng.normal(size=3); direction/=np.linalg.norm(direction)
    elapsed=np.maximum(TIMES-4.5,0.)
    displacement=2.*(elapsed-.3*(1-np.exp(-elapsed/.3)))[:,None]*direction
    # Construct identical observation arrays for the two incompatible explanations.
    shared_observations=base+displacement+noise
    if kind=='maneuver':
        truth=base+displacement
        observations=shared_observations
    elif kind=='sensor_drift':
        truth=base.copy()
        observations=shared_observations
    elif kind=='sensor_spike':
        truth=base.copy(); observations=base+noise
        observations[50]+=3.*direction  # training only, never a witness
    elif kind=='ordinary':
        truth=base.copy(); observations=base+noise
    else:
        raise ValueError(kind)
    return dict(truth=truth, observations=observations, event=displacement,
                witness_contract=kind!='sensor_drift')


def propose(observations,sigma):
    """All directions depend only on prefix+TRAIN, never on WITNESS values.

    Each direction is the entire current mean change induced by a new training
    observation. Their sum telescopes to the raw training-only correction.
    """
    model=RelationalKalman(observations[0],sigma)
    for i in PREFIX[1:]:
        model.update(TIMES[i],observations[i])
    baseline=model.forecast(TIMES)['mean']
    previous=baseline
    directions=[]
    for i in TRAIN:
        model.update(TIMES[i],observations[i])
        current=model.forecast(TIMES)['mean']
        directions.append(current-previous)
        previous=current
    return baseline,np.stack(directions,axis=-1),previous


def null_probe(seed=431,repeats=4000):
    rng=np.random.default_rng(seed)
    train=rng.normal(size=(repeats,30))
    witness=rng.normal(size=(repeats,30))
    length=np.linalg.norm(train,axis=1)
    projection=np.sum(train*witness,axis=1)/length
    return dict(repeats=repeats, dimension=30, alpha=.05,
                honest_ray_false_acceptance=float(np.mean(projection>norm.ppf(.95))),
                honest_span_false_acceptance=float(np.mean(np.abs(projection)>np.sqrt(chi2.ppf(.95,1)))),
                reused_ray_false_acceptance=float(np.mean(length>norm.ppf(.95))),
                reused_span_false_acceptance=float(np.mean(length>np.sqrt(chi2.ppf(.95,1)))))


def run(out,seeds=12):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows,certificates,traces=[],[],[]
    started=time.perf_counter()
    for fi,family in enumerate(FAMILIES):
        for repeat in range(seeds):
            seed=90000+1000*fi+repeat
            for sigma in (.35,1.4):
                pair_records={}
                for kind in KINDS:
                    case=fixture(family,seed,sigma,kind)
                    y=case['observations']
                    # At prediction time only y[:61] is supplied. Other array rows
                    # passed here are deliberately NaN, detecting accidental reads.
                    supplied=y.copy();supplied[61:]=np.nan
                    base,directions,raw=propose(supplied,sigma)
                    ray=supported_ray(base,raw-base,WITNESS,supplied[WITNESS],sigma)
                    span=supported_subspace(base,directions,WITNESS,supplied[WITNESS],sigma)
                    reused=supported_subspace(base,directions,TRAIN,supplied[TRAIN],sigma)
                    all_model=RelationalKalman(y[0],sigma)
                    for i in range(1,61):
                        all_model.update(TIMES[i],y[i])
                    full=all_model.forecast(TIMES)['mean']
                    means={'prefix':base,'raw_train':raw,'witness_ray':ray['mean'],
                           'witness_span':span['mean'],'reused_witness':reused['mean'],'all_history':full}
                    truth=case['truth']
                    for method,mean in means.items():
                        rows.append(dict(family=family,seed=seed,sigma=sigma,kind=kind,method=method,
                            witness_sq=float(np.mean(np.sum((mean[WITNESS]-truth[WITNESS])**2,axis=1))),
                            forecast_sq=float(np.sum((mean[-1]-truth[-1])**2)),
                            path_sq=float(np.mean(np.sum((mean[FUTURE]-truth[FUTURE])**2,axis=1)))))
                    for method,result in [('witness_ray',ray),('witness_span',span),('reused_witness',reused)]:
                        check_indices=TRAIN if method=='reused_witness' else WITNESS
                        actual=clean_gain(base,result['mean'],truth,check_indices)
                        witness_gain=clean_gain(base,result['mean'],truth,WITNESS)
                        future_gain=clean_gain(base,result['mean'],truth,[-1])
                        cert=dict(family=family,seed=seed,sigma=sigma,kind=kind,method=method,
                             certified=result['certified'],lower_gain=result['lower_gain'],actual_gain=actual,
                             witness_gain=witness_gain,future_gain=future_gain,factor=result['factor'],
                             lower_bound_failed=bool(actual+1e-8<result['lower_gain']),
                             witness_contract=case['witness_contract'] and method!='reused_witness')
                        if method!='witness_ray':
                            noise=((y[check_indices]-truth[check_indices])/sigma).reshape(-1)
                            projection=float(np.linalg.norm(result['witness_basis'].T @ noise))
                            cert.update(rank=result['rank'],noise_projection=projection,
                                        confidence_radius=result['radius'],confidence_event=bool(projection<=result['radius']),
                                        endpoint_amplification=result['endpoint_amplification'],
                                        endpoint_null_response=result['endpoint_null_response'])
                            if projection<=result['radius']+1e-10:
                                assert actual+1e-7>=result['lower_gain']
                        certificates.append(cert)
                    if kind in ('maneuver','sensor_drift'):
                        pair_records[kind]=(y[:61].copy(),span['mean'].copy(),truth[-1].copy())
                    if repeat==0 and sigma==.35:
                        traces.append(dict(family=family,kind=kind,times=TIMES,truth=truth,
                            observations=y[:61],methods=means,ray_factor=ray['factor'],span_factor=span['factor'],
                            endpoint_amplification=span['endpoint_amplification']))
                a,b=pair_records['maneuver'],pair_records['sensor_drift']
                np.testing.assert_array_equal(a[0],b[0])
                np.testing.assert_array_equal(a[1],b[1])
                assert np.linalg.norm(a[2]-b[2])>6.
            save(out/'partial.json',dict(runs=rows,certificates=certificates))
            print(f'{family} {repeat+1}/{seeds} elapsed {time.perf_counter()-started:.1f}s',flush=True)
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['witness_geometry.py','witness_study.py','relational_kalman.py','data.py']}
    save(out/'results.json',dict(metadata=dict(seeds=seeds,seed_base=90000,prefix=PREFIX,train=TRAIN,
              witness=WITNESS,future=FUTURE,alpha=.05,noise_sigmas=[.35,1.4],source_hashes=hashes,
              direction_source='sequential training-only current mean revisions',
              scope='local clean squared-error certificate at witness times; no future guarantee'),
              runs=rows,certificates=certificates,null_probe=null_probe()))
    save(out/'traces.json',traces)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--seeds',type=int,default=12)
    args=parser.parse_args()
    run(args.out,args.seeds)
