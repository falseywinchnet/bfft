"""Complete original-Meyer solves, including curved certificate acquisition."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .curved_study import scenes
from .curved_meyer import MeyerQuotient,discover_curved
from .core import finite_flow,Linearization
from experiments.meyer_transport_audit.certificate import certificate


class FullAdapter:
    def __init__(self,model):self.model=model
    def step(self,z):return self.model.pack(self.model.step(self.model.unpack(z)))
    def linearize(self,z):
        state=self.model.unpack(z)
        return Linearization(self.step(z),lambda h:self.model.pack(self.model.tangent(state,self.model.unpack(h))))


def solve(image,method,budget=4096,target=1e-4):
    model=MeyerQuotient(image);full=model.full.initial();z=model.encode(full)
    adapter=FullAdapter(model.full)
    gap0=certificate(model.full,full)['gap'];calls=0;actions=0;accepted=0;rejected=0;points=0
    cooldown=0;check=32;trace=[];status='budget';max_relative=0.;selected=[]
    start=time.perf_counter()
    while calls+actions<budget:
        remaining=budget-calls-actions
        if calls+actions<4 or remaining<12 or method=='ordinary' or cooldown>0:
            full=model.next_full(z);z=model.encode(full);calls+=1;cooldown=max(0,cooldown-1)
        elif method=='full_fixed':
            candidate,b,_=finite_flow(adapter,model.full.pack(full),4,16,False)
            full=model.full.step(model.full.unpack(candidate));z=model.encode(full);calls+=2;actions+=b.actions
        elif method=='quotient_fixed':
            candidate,b,_=finite_flow(model,z,4,16,False)
            full=model.next_full(candidate);z=model.encode(full);calls+=2;actions+=b.actions
        else:
            d=discover_curved(model,z);calls+=d.map_calls;actions+=d.tangent_actions;points+=d.geometry_points
            if d.accepted:
                full=model.next_full(d.candidate);z=model.encode(full);calls+=1;accepted+=1
                max_relative=max(max_relative,d.relative_bound);selected.append((d.depth,d.horizon))
            else:
                # d.candidate is the already-paid next quotient. Its full
                # output is not retained by discovery. Recover it exactly
                # with a charged next step, then cool down acquisition.
                full=model.next_full(d.candidate);z=model.encode(full);calls+=1
                rejected+=1;cooldown=32
        if calls+actions>=check or calls+actions>=budget:
            gap=certificate(model.full,full)['gap']/max(gap0,1e-300)
            trace.append({'work':calls+actions,'gap_ratio':float(gap),'seconds':time.perf_counter()-start})
            check=(calls+actions)//32*32+32
            if gap<=target:status='target';break
            if not np.isfinite(gap):status='invalid';break
    return {'method':method,'status':status,'calls':calls,'actions':actions,'work':calls+actions,
            'seconds':time.perf_counter()-start,'accepted':accepted,'rejected':rejected,
            'geometry_points':points,'maximum_admitted_relative_bound':max_relative,
            'selected_depth_horizons':selected,'trace':trace}


def main(args):
    data={'configuration':vars(args),'note':'Python prototypes, same original map; full cost includes geometry and gap checks; setup excluded.','runs':[]}
    for name,image in scenes(args.size):
        methods=['ordinary','full_fixed','quotient_fixed','curved_discovered']
        for method in methods:solve(image,method,budget=48,target=0.)
        rng=np.random.default_rng(2026)
        for repeat in range(args.repeats):
            for method in rng.permutation(methods):
                row=solve(image,str(method),args.budget,args.target);row.update(scene=name,repeat=repeat)
                data['runs'].append(row)
            Path(args.out).write_text(json.dumps(data,indent=2))
            print(name,repeat,[(r['method'],r['status'],r['work'],round(r['seconds'],3),r['accepted']) for r in data['runs'][-4:]],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=64)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--budget',type=int,default=4096);p.add_argument('--target',type=float,default=1e-4)
    main(p.parse_args())
