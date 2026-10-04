import argparse,json
from pathlib import Path
import numpy as np
from .frame_transport import run
from .study_energy_theory import source_problem

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=128)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--audit',action='store_true')
    p.add_argument('--seeds',default='0,1');p.add_argument('--eps',default='.01,.003,.001')
    args=p.parse_args();rows=[]
    for seed in map(int,args.seeds.split(',')):
        C,a,b=source_problem(args.size,seed)
        for eps in map(float,args.eps.split(',')):
            K=np.exp(-C/eps);control=run(K,a,b,method='ordinary')
            for method in ['ordinary','cache','reset','guarded','spectral','atomic']:
                repeats=[run(K,a,b,method=method,audit=args.audit) for _ in range(args.repeats)]
                row=repeats[0];row.update(n=args.size,seed=seed,eps=eps,method=method,
                    repeat_seconds=[r['seconds'] for r in repeats],error=float(np.ptp(row['y']-control['y'])))
                del row['y'];rows.append(row)
                print({k:v for k,v in row.items() if k not in ['events','repeat_seconds']},flush=True)
                Path(args.out).write_text(json.dumps(rows))
if __name__=='__main__':main()
