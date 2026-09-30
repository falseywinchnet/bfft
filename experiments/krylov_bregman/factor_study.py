"""Compare frozen tangent and exact-primitive reduced flow on original Meyer."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .curved_meyer import MeyerQuotient
from .curved_study import scenes
from .factor_transport import FactorChart,enrich_from_defect
from .core import finite_flow
from experiments.meyer_transport_audit.certificate import certificate


def solve(image,method,budget=4096,target=1e-4):
    m=MeyerQuotient(image);full=m.full.initial();z=m.encode(full)
    gap0=certificate(m.full,full)['gap'];calls=actions=adjoints=geometry=0
    nextcheck=32;status='budget';trace=[]
    start=time.perf_counter()
    while calls+actions+adjoints<budget:
        if method=='ordinary' or calls<4:
            full=m.next_full(z);z=m.encode(full);calls+=1
        elif method=='linear16':
            candidate,b,_=finite_flow(m,z,4,16,False)
            full=m.next_full(candidate);z=m.encode(full);calls+=2;actions+=b.actions
        else:
            horizon=64 if method=='enriched64' else int(method[6:]);chart=FactorChart.acquire(m,z,4)
            if method=='enriched64':
                chart=enrich_from_defect(chart);calls+=1;geometry+=8
            candidate=chart.candidate(horizon)
            full=m.next_full(candidate);z=m.encode(full);calls+=2
            actions+=chart.basis.actions;adjoints+=chart.adjoint_actions;geometry+=horizon
        work=calls+actions+adjoints
        if work>=nextcheck:
            gap=certificate(m.full,full)['gap']/gap0
            trace.append(dict(work=work,gap_ratio=float(gap),seconds=time.perf_counter()-start))
            nextcheck=(work//32+1)*32
            if gap<=target:status='target';break
            if not np.isfinite(gap):status='invalid';break
    return dict(method=method,status=status,seconds=time.perf_counter()-start,calls=calls,
        actions=actions,adjoint_actions=adjoints,geometry_steps=geometry,work=calls+actions+adjoints,trace=trace)


def run(args):
    data={'configuration':vars(args),'runs':[],'anchors':[],
          'note':'original Meyer objective and quotient; all acquisition/adjoints/disk evaluations/settling/gap checks timed; common initialization excluded'}
    for name,image in scenes(args.size):
        if name not in args.scenes.split(','):continue
        m=MeyerQuotient(image);z=m.initial.copy();prefixlast=0
        for prefix in (4,32,128):
            for _ in range(prefix-prefixlast):z=m.step(z)
            prefixlast=prefix
            chart=FactorChart.acquire(m,z,4);b=chart.basis
            enriched=enrich_from_defect(chart)
            actual=z.copy()
            for j in range(1,65):
                actual=m.step(actual)
                if j in (16,64):
                    linear=z+b.displacement(j);factor=chart.candidate(j)
                    scale=float(np.linalg.norm(actual-z))
                    data['anchors'].append(dict(scene=name,prefix=prefix,horizon=j,
                        displacement=scale,linear_error=float(np.linalg.norm(linear-actual)),
                        factor_error=float(np.linalg.norm(factor-actual)),
                        enriched_error=float(np.linalg.norm(enriched.candidate(j)-actual)),
                        enriched_depth=enriched.basis.actions))
        methods=['ordinary','linear16','factor16','factor64','enriched64']
        for method in methods:solve(image,method,budget=48,target=0)
        rng=np.random.default_rng(2026)
        for repeat in range(args.repeats):
            batch=[]
            for method in rng.permutation(methods):
                row=solve(image,str(method),args.budget,args.target);row.update(scene=name,repeat=repeat)
                data['runs'].append(row);batch.append((method,row['status'],round(row['seconds'],3),row['work']))
            Path(args.out).write_text(json.dumps(data,indent=2));print(name,repeat,batch,flush=True)
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=64)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--budget',type=int,default=4096);p.add_argument('--target',type=float,default=1e-4)
    p.add_argument('--scenes',default='camera,camera_permuted,barbara,straight_carrier,crossing')
    run(p.parse_args())
