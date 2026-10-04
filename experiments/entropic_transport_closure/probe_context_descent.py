import argparse
import json
from pathlib import Path
from .study_energy_theory import source_problem
from .context_descent import solve, ordinary_blas

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--repeats',type=int,default=3); p.add_argument('--size',type=int,default=128)
    args=p.parse_args(); rows=[]
    for seed in [0,1]:
        cost,a,b=source_problem(args.size,seed)
        for eps in [.01,.003,.001]:
            for method in ['ordinary_blas','ordinary','gradient','scalar','lbfgs','adaptive','frozen']:
                runs=[ordinary_blas(-cost/eps,a,b) if method=='ordinary_blas' else solve(-cost/eps,a,b,method) for _ in range(args.repeats)]
                assert all(x['converged'] for x in runs), (seed,eps,method)
                out=runs[0]; out['repeat_seconds']=[x['seconds'] for x in runs]
                out['repeat_residuals']=[x['residual'] for x in runs]
                out.update(seed=seed,eps=eps,n=args.size); rows.append(out)
                Path(args.out).write_text(json.dumps(rows))
                print(seed,eps,method,out['accepted_steps'],out['residual'],out['seconds'],out['failure'],flush=True)
if __name__=='__main__': main()
