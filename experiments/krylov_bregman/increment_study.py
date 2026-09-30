"""Both geometries, finite increment recurrence with all acquisition charged."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .increment_transport import IncrementChart
from .curved_meyer import MeyerQuotient
from .curved_study import scenes
from .sinkhorn_study import problem
from .sinkhorn_transport import Sinkhorn
from .core import finite_flow
from experiments.meyer_transport_audit.certificate import certificate


def solve(m,kind,method,budget=4096,target=1e-4):
    z=m.initial.copy();calls=actions=accepted=rejected=0;nextcheck=32;status='budget';trace=[];ranks=[];clips=[]
    if kind=='meyer':
        full=m.full.initial();gap0=certificate(m.full,full)['gap']
    def step(x):
        nonlocal calls,full
        calls+=1
        if kind=='meyer':full=m.next_full(x);return m.encode(full)
        return m.step(x)
    def metric(x):
        if kind=='meyer':return certificate(m.full,full)['gap']/gap0
        return m.error(x)
    start=time.perf_counter()
    while calls+actions<budget:
        if method=='ordinary' or calls<4:
            z=step(z)
        elif method=='linear':
            candidate,b,_=finite_flow(m,z,4,16,False);calls+=1;actions+=b.actions;z=step(candidate)
        else:
            states=[z.copy()]
            for _ in range(8):states.append(step(states[-1]))
            z=states[-1]
            if kind=='meyer':anchor_full=full
            chart=IncrementChart.fit(states,maximum_rank=4,contractive=kind=='meyer' and method!='increment_raw')
            candidate,horizon,_,_=chart.propose(step)
            ranks.append(chart.rank);clips.append(chart.clipping)
            if horizon:
                z=step(candidate);accepted+=1
            else:
                # Probe calls must not replace the actual retained ordinary state.
                if kind=='meyer':full=anchor_full
                rejected+=1
        work=calls+actions
        if work>=nextcheck:
            error=metric(z)
            trace.append(dict(work=work,error=float(error),seconds=time.perf_counter()-start))
            nextcheck=(work//32+1)*32
            if error<=target:status='target';break
            if not np.isfinite(error):status='invalid';break
    return dict(method=method,status=status,seconds=time.perf_counter()-start,calls=calls,actions=actions,
        accepted=accepted,rejected=rejected,ranks=ranks,clipping=clips,work=calls+actions,trace=trace)


def run(args):
    data={'configuration':vars(args),'runs':[], 'note':'all ordinary acquisition, SVD fitting, live probes, rejection and settling charged; no Jacobian in increment proposals; probes not global certificates'}
    jobs=[]
    for name,im in scenes(args.size):jobs.append(('meyer',name,im))
    for name in ('image','mixture'):jobs.append(('sinkhorn',name,None))
    for kind,name,im in jobs:
        methods=['ordinary','linear','increment']+(['increment_raw'] if kind=='meyer' else [])
        def make():return MeyerQuotient(im) if kind=='meyer' else Sinkhorn(*problem(512,0,name),.003)
        target=args.target if kind=='meyer' else 1e-8
        for method in methods:solve(make(),kind,method,budget=48,target=0)
        rng=np.random.default_rng(2026)
        for repeat in range(args.repeats):
            batch=[]
            for method in rng.permutation(methods):
                row=solve(make(),kind,str(method),args.budget,target);row.update(kind=kind,scene=name,repeat=repeat)
                data['runs'].append(row);batch.append((method,row['status'],round(row['seconds'],3),row['work'],row['accepted'],row['rejected']))
            Path(args.out).write_text(json.dumps(data,indent=2));print(kind,name,repeat,batch,flush=True)
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=64)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--budget',type=int,default=4096);p.add_argument('--target',type=float,default=1e-4)
    run(p.parse_args())
