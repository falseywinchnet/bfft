"""Known-family control versus causal learning of its log generators."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .multiplicative_algebra import AlgebraCarrier
from .primitive_transport import ResponseCarrier
from .study_energy_theory import source_problem

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=128)
    p.add_argument('--budget',type=int,default=64);p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args();rows=[]
    for seed in [0,1]:
        C,a,b=source_problem(args.size,seed);K=np.exp(-C/.003)
        for d in [1,2,4]:
            for gaussian in [0.,1.]:
                rng=np.random.default_rng(1900+seed)
                W=rng.normal(size=(args.size,d)) if gaussian else rng.choice([-1.,1.],size=(args.size,d))
                W-=W.mean(axis=0);W/=np.sqrt(np.mean(W*W,axis=0))
                omega=np.linspace(.7,1.3,d);phase=rng.uniform(0,2*np.pi,d)
                requests=[]
                for t in np.linspace(0,20,256):
                    h=1.4/np.sqrt(d)*(W@np.sin(omega*t+phase));h-=h.max();requests.append(np.exp(h))
                for method in ['ordinary','cache','axis','mixed','learned']:
                    times=[]
                    for _ in range(args.repeats):
                        start=time.perf_counter()
                        if method=='cache':carrier=ResponseCarrier(K,32)
                        elif method!='ordinary':
                            carrier=AlgebraCarrier(K,args.budget,generator_count=d,mixed=method!='axis',
                                supplied_generators=None if method=='learned' else W,refresh=False)
                        outputs=[];bounds=[]
                        for x in requests:
                            if method=='ordinary':out=K@x;e=0.
                            else:out,e=carrier.apply(x,1e-8)[:2]
                            outputs.append(out);bounds.append(e)
                        times.append(time.perf_counter()-start)
                    errors=[float(np.ptp(np.log(out/(K@x)))) for out,x in zip(outputs,requests)]
                    row=dict(seed=seed,n=args.size,directions=d,gaussian=gaussian,method=method,budget=args.budget,
                        repeat_seconds=times,products=256 if method=='ordinary' else carrier.products,
                        reuses=0 if method=='ordinary' else carrier.reuses,max_error=max(errors),
                        max_bound_violation=max(e-b for e,b in zip(errors,bounds)),
                        events=getattr(carrier,'events',[]) if method!='ordinary' else [])
                    rows.append(row);Path(args.out).write_text(json.dumps(rows))
                print(seed,d,gaussian,[(r['method'],r['products'],r['reuses']) for r in rows[-5:]],flush=True)
if __name__=='__main__':main()
