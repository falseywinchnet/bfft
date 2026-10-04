"""Complete solves to a common duality gap, including unmodified adaptive rho."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from sporco.admm.bpdn import BPDN
from .study_bpdn import problem,gap
from .bpdn_adapter import make_solver,SparseCoding
from .certified_transport import run_certified


def solve(D,S,lam,method,maxiter,target,check_every=64):
    target_abs=target*.5*float(np.sum(S*S))
    log=[]
    def stop(Y):
        obj,g=gap(D,S,Y,lam);log.append(dict(objective=obj,gap=g));return g<=target_abs
    if method in ('library_fixed','library_adaptive'):
        def callback(solver):
            if (solver.k+1)%check_every==0 or solver.k+1==maxiter:return stop(solver.Y)
            return False
        opt=BPDN.Options(dict(Verbose=False,MaxMainIter=maxiter,FastSolve=True,
            AutoRho=dict(Enabled=method=='library_adaptive'),
            # The adaptive comparator retains the library's default relaxation.
            RelStopTol=0.,AbsStopTol=0.,ReturnX=False,Callback=callback))
        if method=='library_fixed':opt['RelaxParam']=1.
        s=BPDN(D,S,lam,opt);out=s.solve()
        stats=dict(passes=s.k,stopping_checks=len(log),converged=log[-1]['gap']<=target_abs)
    else:
        model=SparseCoding(make_solver(D,S,lam,maxiter))
        out,stats=run_certified(model,maxiter,tolerance=.02,depths=(2,4,8),stop=stop,
                                check_every=check_every,accelerated=method=='certified')
    stats.update(target_gap=target_abs,final_gap=log[-1]['gap'],objective=log[-1]['objective'])
    return out,stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--sources',default='camera,coins,moon')
    p.add_argument('--seeds',default='1,2');p.add_argument('--signals',default='1,16');p.add_argument('--side',type=int,default=16)
    p.add_argument('--maxiter',type=int,default=4096);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--target',type=float,default=1e-5);a=p.parse_args();rows=[];dest=Path(a.out)
    for source in a.sources.split(','):
     for seed in map(int,a.seeds.split(',')):
      for signals in map(int,a.signals.split(',')):
       D,S=problem(a.side,signals,source,seed)
       for lam in [.02,.1]:
        methods=['library_fixed','quotient','certified','library_adaptive']
        timing={k:[] for k in methods};stats={};outputs={}
        # Untimed warm-up uses a distinct short iteration cap.
        for key in methods:solve(D,S,lam,key,64,a.target)
        for rep in range(a.repeats):
         for key in np.random.default_rng(rep).permutation(methods):
          start=perf_counter();outputs[key],stats[key]=solve(D,S,lam,key,a.maxiter,a.target)
          timing[key].append(perf_counter()-start)
        row=dict(source=source,seed=seed,signals=signals,side=a.side,lmbda=lam,timing=timing,stats=stats,
                 speedup_fixed=float(np.median(timing['library_fixed'])/np.median(timing['certified'])),
                 speedup_quotient=float(np.median(timing['quotient'])/np.median(timing['certified'])),
                 speedup_adaptive=float(np.median(timing['library_adaptive'])/np.median(timing['certified'])),
                 all_converged=all(v['converged'] for v in stats.values()))
        rows.append(row);dest.write_text(json.dumps(dict(protocol=vars(a),rows=rows),indent=2))
        print({k:v for k,v in row.items() if k not in ('timing','stats')},
              {k:(v['passes'],v['final_gap'],v['converged']) for k,v in stats.items()},flush=True)
    print('saved',dest,flush=True)

if __name__=='__main__':main()
