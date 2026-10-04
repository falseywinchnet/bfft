"""Controlled primitive-query experiment, not an ordinary Sinkhorn trajectory.
Vary independently excited input directions and Gaussianity of their fields.
"""
import json,argparse,time
from pathlib import Path
import numpy as np
from .primitive_transport import ResponseCarrier
from .study_energy_theory import source_problem

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--fixed-range',action='store_true');args=p.parse_args();rows=[]
    for seed in [0,1]:
        C,a,b=source_problem(128,seed);K=np.exp(-C/.003)
        for d in [1,2,4,8,16]:
            for gaussian in [0.,.001,.01,.1,.5,1.]:
                rng=np.random.default_rng(1900+seed)
                normal=rng.normal(size=(128,16));binary=rng.choice([-1.,1.],size=(128,16))
                W=np.sqrt(gaussian)*normal[:,:d]+np.sqrt(1-gaussian)*binary[:,:d]
                W-=W.mean(axis=0);W/=np.sqrt(np.mean(W*W,axis=0))
                omega=np.linspace(.7,1.3,d);phase=rng.uniform(0,2*np.pi,d)
                carrier=ResponseCarrier(K,32);error=0.;bound=0.;ranges=[];kurt=[];skew=[]
                start=time.perf_counter()
                for t in np.linspace(0,20,256):
                    h=1.4/np.sqrt(d)*(W@np.sin(omega*t+phase));h-=h.max()
                    if args.fixed_range:h*=6/max(np.ptp(h),1e-300)
                    x=np.exp(h);pred,e,_=carrier.apply(x,1e-8)
                    exact=K@x # offline audit: time is intentionally NOT a benchmark
                    error=max(error,float(np.ptp(np.log(pred/exact))));bound=max(bound,e)
                    centered=h-h.mean();v=np.mean(centered**2)
                    ranges.append(float(np.ptp(h)));kurt.append(float(np.mean(centered**4)/max(v*v,1e-300)-3))
                    skew.append(float(np.mean(centered**3)/max(v,1e-300)**1.5))
                rows.append(dict(seed=seed,fixed_range=args.fixed_range,directions=d,gaussian_fraction=gaussian,products=carrier.products,
                    reuses=carrier.reuses,resets=carrier.resets,max_error=error,max_bound=bound,
                    mean_range=float(np.mean(ranges)),max_range=max(ranges),
                    mean_abs_skew=float(np.mean(np.abs(skew))),mean_abs_excess_kurtosis=float(np.mean(np.abs(kurt)))))
        print('finished seed',seed,flush=True);Path(args.out).write_text(json.dumps(rows))
if __name__=='__main__':main()
