import argparse
import json
import gzip
from pathlib import Path
from .study_energy_theory import source_problem
from .context_descent import solve as scalar_solve,ordinary_blas
from .enclosed_continuation import solve,audit

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--repeats',type=int,default=3); p.add_argument('--rank',type=int,default=4)
    p.add_argument('--size',type=int,default=128); p.add_argument('--seeds',default='0,1'); p.add_argument('--eps',default='.01,.003,.001')
    args=p.parse_args(); rows=[]
    for seed in map(int,args.seeds.split(',')):
        cost,a,b=source_problem(args.size,seed)
        for eps in map(float,args.eps.split(',')):
            L=-cost/eps
            for method in ['ordinary','scalar','exact','box','merged']:
                runs=[]
                for _ in range(args.repeats):
                    if method=='ordinary': out=ordinary_blas(L,a,b)
                    elif method=='scalar': out=scalar_solve(L,a,b,method='scalar')
                    else: out=solve(L,a,b,enclosure=method,rank=args.rank)
                    runs.append(out)
                row=runs[0]; row.update(seed=seed,eps=eps,n=args.size,method=method,
                    repeat_seconds=[r['seconds'] for r in runs],repeat_residuals=[r['residual'] for r in runs])
                if method not in ['ordinary','scalar']: row['audit']=audit(L,a,b,row)
                rows.append(row)
                if args.out.endswith('.gz'):
                    with gzip.open(args.out,'wt') as stream: json.dump(rows,stream)
                else: Path(args.out).write_text(json.dumps(rows))
                print(seed,eps,method,row['converged'],row['evaluations'],len(row.get('blocks',[])),row['seconds'],row.get('audit'),flush=True)
if __name__=='__main__': main()
