"""Test event-aware jumps without disabling SPORCO's adaptive rho."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from .study_bpdn import problem
from .solve_bpdn import solve as library_solve
from .adaptive_bpdn import solve


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();rows=[];dest=Path(a.out)
    for source in ['camera','coins','moon']:
     for seed in [1,2]:
      for signals in [1,16]:
       D,S=problem(16,signals,source,seed)
       for lam in [.02,.1]:
        funcs=dict(library=lambda:library_solve(D,S,lam,'library_adaptive',4096,1e-5),
                   native_driver=lambda:solve(D,S,lam,accelerated=False),
                   engine=lambda:solve(D,S,lam,accelerated=True))
        for fn in funcs.values():fn()
        timings={k:[] for k in funcs};outputs={};stats={}
        for rep in range(a.repeats):
         for k in np.random.default_rng(rep).permutation(list(funcs)):
          start=perf_counter();outputs[k],stats[k]=funcs[k]();timings[k].append(perf_counter()-start)
        row=dict(source=source,seed=seed,signals=signals,lmbda=lam,timing=timings,stats=stats,
                 driver_max_error=float(np.max(np.abs(outputs['library']-outputs['native_driver']))),
                 engine_output_difference=float(np.linalg.norm(outputs['engine']-outputs['library'])),
                 speedup=float(np.median(timings['library'])/np.median(timings['engine'])),
                 engine_only_speedup=float(np.median(timings['native_driver'])/np.median(timings['engine'])),
                 all_converged=all(s['converged'] for s in stats.values()))
        rows.append(row);dest.write_text(json.dumps(dict(protocol=dict(repeats=a.repeats,target=1e-5,
             check_every=64,relaxation='SPORCO default',rho='SPORCO default adaptive',admission=.02),rows=rows),indent=2))
        print({k:v for k,v in row.items() if k not in ('timing','stats')},
              {k:(s['passes'],s.get('accepted'),s.get('rejected')) for k,s in stats.items()},flush=True)
    print('saved',dest,flush=True)

if __name__=='__main__':main()
