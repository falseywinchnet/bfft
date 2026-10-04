import argparse,json
from pathlib import Path
import numpy as np
from .multiplicative_algebra import run
from .frame_transport import run as control
from .study_energy_theory import source_problem

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=128)
    p.add_argument('--seeds',default='0,1');p.add_argument('--eps',default='.01,.003,.001')
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--budget',type=int,default=32)
    args=p.parse_args();rows=[]
    for seed in map(int,args.seeds.split(',')):
        C,a,b=source_problem(args.size,seed)
        for eps in map(float,args.eps.split(',')):
            K=np.exp(-C/eps);truth=control(K,a,b,method='ordinary')
            for method in ['ordinary','cache','axis2','mixed2','mixed4']:
                runs=[]
                for _ in range(args.repeats):
                    out=control(K,a,b,method=method) if method in ['ordinary','cache'] else run(K,a,b,budget=args.budget,generators=int(method[-1]),mixed=method.startswith('mixed'))
                    runs.append(out)
                row=runs[0];row.update(seed=seed,n=args.size,eps=eps,method=method,budget=args.budget,
                    error=float(np.ptp(row['y']-truth['y'])),repeat_seconds=[v['seconds'] for v in runs]);del row['y']
                rows.append(row);print({k:v for k,v in row.items() if k not in ['events','repeat_seconds','rejections']},flush=True)
                Path(args.out).write_text(json.dumps(rows))
if __name__=='__main__':main()
