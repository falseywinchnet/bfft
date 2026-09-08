"""Matched exact-algebra cost study, with frozen acquisition replay."""
import argparse,hashlib,time,platform
from pathlib import Path
import numpy as np
from .relational_reference import RelationalKalman as Dense
from .relational_markov import RelationalKalman as Markov, transitions
from .relational_native import RelationalKalman as Native
from .acquisition_study import simulate,POLICIES
from .adaptive_acquisition import BoundaryFixture
from .data import FAMILIES
from .study import save


def micro(repeats):
    rng=np.random.default_rng(60021)
    times=np.r_[0,np.cumsum(rng.uniform(.04,.06,160))];y=rng.normal(size=(161,3))
    results=[]
    for cells in (32,128,512):
        for name,cls,coords in [('dense_zak',Dense,'zak'),('dense_current',Dense,'current'),('markov_python',Markov,'current'),('markov_native',Native,'current')]:
            if cells==512 and name.startswith('dense'):nr=min(repeats,5)
            else:nr=repeats
            timings=[];cold=[]
            for rep in range(nr+1):
                if name.startswith('markov'):transitions.cache_clear()
                start=time.perf_counter_ns();model=cls(y[0],.35,14.,cells,coordinates=coords);init=time.perf_counter_ns()-start
                costs=[0,0,0]
                for t,v in zip(times[1:],y[1:]):
                    start=time.perf_counter_ns();model.update(t,v);costs[0]+=time.perf_counter_ns()-start
                    start=time.perf_counter_ns()
                    if hasattr(model,'position'):model.position(t)
                    else:
                        from .acquisition_study import current_mean
                        current_mean(model,t)
                    costs[1]+=time.perf_counter_ns()-start
                    start=time.perf_counter_ns();model.forecast([t,t+2]);costs[2]+=time.perf_counter_ns()-start
                if rep:timings.append(np.array(costs)/160/1000);cold.append(init/1e6)
            results.append(dict(backend=name,cells=cells,repeats=nr,cold_initialization_ms=np.median(cold),
                median_us=np.median(timings,axis=0),p10_us=np.percentile(timings,10,axis=0),p90_us=np.percentile(timings,90,axis=0)))
            print('micro',results[-1],flush=True)
    return results


def run(out,seeds,repeats):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    timing=micro(repeats);save(out/'micro.json',timing)
    records=[];maxima=dict(mean=0.,forecast=0.,radii=0.,event_estimate=0.);mismatches=0
    for fi,family in enumerate(FAMILIES):
        for rep in range(seeds):
            seed=110000+1000*fi+rep;fixture=BoundaryFixture(family,seed)
            for sigma in (.35,1.4):
                for policy in POLICIES:
                    pair={}
                    # Alternate ordering to balance temperature/order effects.
                    classes=[('dense',Dense),('native',Native)]
                    if (rep+fi)%2:classes.reverse()
                    for name,cls in classes:
                        pair[name]=simulate(fixture,sigma,policy,seed,keep_trace=True,tracker_class=cls)
                    a,b=pair['dense'],pair['native'];ta,tb=a['trace'],b['trace']
                    assert a['samples']==b['samples'] and a['trigger_time']==b['trigger_time']
                    np.testing.assert_array_equal([e['time'] for e in ta['events']],[e['time'] for e in tb['events']])
                    np.testing.assert_array_equal([e['observation'] for e in ta['events']],[e['observation'] for e in tb['events']])
                    for key,tracekey in [('mean','estimates'),('forecast','forecasts'),('radii','radii')]:
                        delta=float(np.max(np.abs(np.array(ta[tracekey])-np.array(tb[tracekey]))));maxima[key]=max(maxima[key],delta)
                        np.testing.assert_allclose(ta[tracekey],tb[tracekey],atol=2e-8,rtol=2e-8)
                    delta=float(np.max(np.abs(np.array([e['estimate'] for e in ta['events'][1:]])-np.array([e['estimate'] for e in tb['events'][1:]]))))
                    maxima['event_estimate']=max(maxima['event_estimate'],delta)
                    for name,r in pair.items():
                        r.pop('trace');r.update(backend=name,family=family,seed=seed);records.append(r)
            save(out/'partial.json',dict(runs=records,max_absolute_difference=maxima))
            print('replay',family,rep+1,seeds,'maxima',maxima,flush=True)
    hashes={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('relational_*.py')}
    hashes['markov.cpp']=hashlib.sha256((Path(__file__).parent/'native/markov.cpp').read_bytes()).hexdigest()
    save(out/'results.json',dict(metadata=dict(platform=platform.platform(),seeds=seeds,micro_repeats=repeats,source_hashes=hashes,
         cost_fields='micro: update, position, two-query forecast in microseconds; no sensor cost',precision='float64, no fast-math'),
         micro=timing,runs=records,max_absolute_difference=maxima,decision_or_timestamp_mismatches=mismatches))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,default=8);p.add_argument('--repeats',type=int,default=31)
    a=p.parse_args();run(a.out,a.seeds,a.repeats)
