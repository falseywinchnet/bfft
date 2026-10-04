import argparse,json
from pathlib import Path
import numpy as np
from .study_energy_theory import source_problem
from .primitive_transport import finite_run

def geometry_diagnostic(K,a,b):
    def frame(v):
        kv=K@v;u=a/kv;ktu=K.T@u
        return K*v[None,:]/kv[:,None],K.T*u[None,:]/ktu[:,None],b/ktu
    v0=np.ones(len(b));P0,Q0,_=frame(v0)
    v=v0.copy()
    for _ in range(16):
        _,_,v=frame(v);v/=v.max()
    P,Q,nxt=frame(v);h=np.log(nxt/v);h-=h.mean()
    old=Q0@(P0@h);fresh=Q@(P@h)
    actual_u=a/(K@v);base_u=a/(K@v0)
    Pt=P0*v[None,:];Pt/=Pt.sum(axis=1)[:,None]
    Qt=Q0*(actual_u/base_u)[None,:];Qt/=Qt.sum(axis=1)[:,None]
    return dict(frozen_rule_relative_defect=float(np.linalg.norm(old-fresh)/max(np.linalg.norm(fresh),1e-300)),
        transported_relation_max_error=float(max(np.max(np.abs(P-Pt)),np.max(np.abs(Q-Qt)))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--size',type=int,default=128);p.add_argument('--passes',type=int,default=256)
    p.add_argument('--transport',action='store_true',help='REJECTED diagnostic: whole-basis frame transfer is numerically unstable on hard cases');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--rank',type=int,default=32);p.add_argument('--budget',type=float,default=1e-8)
    args=p.parse_args();rows=[]
    for seed in [0,1]:
        C,a,b=source_problem(args.size,seed)
        for eps in [.01,.003,.001]:
            K=np.exp(-C/eps)
            controls=[];runs=[]
            for repeat in range(args.repeats):
                control=finite_run(K,a,b,passes=args.passes,carried=False)
                run=finite_run(K,a,b,passes=args.passes,rank=args.rank,step_budget=args.budget,transport=args.transport)
                controls.append(control['seconds']);runs.append(run['seconds'])
            error=float(np.ptp(run['y']-control['y']))
            row=dict(seed=seed,eps=eps,n=args.size,passes=args.passes,rank=args.rank,
                geometry=geometry_diagnostic(K,a,b),step_budget=args.budget,error=error,transport=args.transport,control_seconds=control['seconds'],control_repeats=controls,repeat_seconds=runs,
                **{k:v for k,v in run.items() if k!='y'})
            rows.append(row);print({k:v for k,v in row.items() if k!='trace'},flush=True)
            Path(args.out).write_text(json.dumps(rows))
if __name__=='__main__':main()
