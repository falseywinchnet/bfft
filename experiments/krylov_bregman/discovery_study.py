"""Independent validation of discovered finite transport, including image data.

Full future trajectories and objective gaps are diagnostic only. The scanner
cannot access them. Natural/permuted images share exactly the same histogram.
"""
import argparse
import json
from pathlib import Path
import time
import platform
import numpy as np
from PIL import Image
from .core import Linearization, finite_flow, Anderson
from .discovery import discover
from .problems import QuadraticMirror, EntropicSimplex, ADMMShadow
from .study import json_safe
from experiments.meyer_transport_audit.model import ReducedMeyerMap
from experiments.meyer_transport_audit.certificate import certificate


class Meyer:
    def __init__(self,image):
        self.model=ReducedMeyerMap(image,.05,40)
        self.initial=self.model.pack(self.model.initial())
    def step(self,z):return self.model.pack(self.model.step(self.model.unpack(z)))
    def linearize(self,z):
        state=self.model.unpack(z)
        return Linearization(self.step(z),lambda h:self.model.pack(self.model.tangent(state,self.model.unpack(h))))
    def gap(self,z):return certificate(self.model,self.model.unpack(z))['gap']


def families(size,seed):
    # Same metric conditioning, eigenvalue endpoints, dimension, and seed;
    # only the number of distinct eigenvalues changes.
    for rank in [2,4,8,32]:
        model=QuadraticMirror(64,seed,condition=1000.)
        vals,V=np.linalg.eigh(model.B)
        levels=np.geomspace(.001,1.,rank)
        model.B=(V*levels[np.minimum(np.arange(64)*rank//64,rank-1)])@V.T
        yield 'spectral_degree_'+str(rank),model
    yield 'entropy_simplex',EntropicSimplex(64,seed)
    for rho in [.01,1.,10.]:
        yield 'admm_shadow_'+str(rho),ADMMShadow(64,seed,rho=rho)
    root=Path(__file__).resolve().parents[2]
    for name,file in [('camera','personal_deblurrer/source_assets/v3_skimage/camera.png'),
                      ('barbara','paper/fast_meyer_bregman/assets/barbara_512.tif')]:
        image=np.asarray(Image.open(root/file).convert('L').resize((size,size),Image.Resampling.BOX),dtype=float)
        if seed==0:yield name,Meyer(image)
        shuffled=np.random.default_rng(seed).permutation(image.ravel()).reshape(image.shape)
        yield name+'_permuted',Meyer(shuffled)


def advance(model,z,count):
    z=z.copy()
    for _ in range(count):z=model.step(z)
    return z


def probe(model,prefix):
    z=advance(model,model.initial,prefix)
    start=time.perf_counter();result=discover(model,z);elapsed=time.perf_counter()-start
    # Diagnostics below are separately counted, excluded from discovery time.
    start=time.perf_counter();actual=advance(model,z,result.horizon);baseline_seconds=time.perf_counter()-start
    scale=max(np.linalg.norm(actual-z),1e-30)
    error=float(np.linalg.norm(result.candidate-actual)/scale)
    fixed,basis,_=finite_flow(model,z,4,16,False)
    fixedactual=advance(model,z,16)
    fixederror=float(np.linalg.norm(fixed-fixedactual)/max(np.linalg.norm(fixedactual-z),1e-30))
    cost=result.map_calls+result.tangent_actions+(1 if result.accepted else 0)
    candidate=model.step(result.candidate) if result.accepted else result.candidate
    equalcost=advance(model,z,cost)
    return {'prefix':prefix,'accepted':result.accepted,'depth':result.depth,
            'horizon':result.horizon,'sampled_score':result.sampled_score,
            'relative_trajectory_error':error,'fixed4_h16_error':fixederror,
            'discovery_map_calls':result.map_calls,'tangent_actions':result.tangent_actions,
            'work_with_settling':cost,'equivalent_ordinary_steps':result.horizon+int(result.accepted),
            'seconds_discovery':elapsed,'seconds_ordinary_horizon':baseline_seconds,
            'settled_gap_ratio_to_equalcost':float(model.gap(candidate)/max(model.gap(equalcost),1e-30)),
            'trials':result.trials,
            'post_discovery_diagnostic_map_calls':result.horizon+1+16+int(result.accepted)+cost,
            'post_discovery_diagnostic_actions':basis.actions}


def solve(model,method,budget=512):
    z=model.initial.copy();calls=0;actions=0;accepted=0;rejected=0;history=Anderson(4)
    start=time.perf_counter();initialgap=model.gap(z);diagnostic_seconds=time.perf_counter()-start
    rows=[];start=time.perf_counter();check=64
    while calls+actions<budget:
        remaining=budget-calls-actions
        if calls+actions<4 or remaining<64 or method=='ordinary':
            z=model.step(z);calls+=1
        elif method=='anderson':
            z=history.advance(z,model.step(z));calls+=1
        elif method=='fixed':
            z,b,_=finite_flow(model,z,4,16,False);calls+=1;actions+=b.actions
            z=model.step(z);calls+=1
        else:
            d=discover(model,z);calls+=d.map_calls;actions+=d.tangent_actions
            z=d.candidate
            if d.accepted:z=model.step(z);calls+=1;accepted+=1
            else:rejected+=1
        if calls+actions>=check or calls+actions>=budget:
            t=time.perf_counter();gap=model.gap(z);diagnostic_seconds+=time.perf_counter()-t
            rows.append({'work':calls+actions,'gap_ratio':float(gap/max(initialgap,1e-30)),
                         'seconds_including_checks':time.perf_counter()-start})
            check=(calls+actions)//64*64+64
        if not np.isfinite(z).all():break
    return {'method':method,'calls':calls,'actions':actions,'accepted':accepted,'rejected':rejected,
            'seconds_including_checks':time.perf_counter()-start,
            'objective_diagnostic_seconds_including_initial':diagnostic_seconds,'trace':rows}


def run(out,size=32,seeds=2):
    data={'configuration':{'size':size,'seeds':seeds,'prefixes':[4,32,128],
          'depths':[2,4,8,12],'horizons':[64,32,16,8],'sampled_tolerance':.05,
          'minimum_work_gain':1.5,'python':platform.python_version(),
          'numpy':np.__version__,'timing_repeats':1,
          'note':'Discovery/falsification screen; exploratory timings, no certified admission or performance claim.'},
          'probes':[],'solves':[]}
    for seed in range(seeds):
        for name,model in families(size,seed):
            for prefix in [4,32,128]:
                row=probe(model,prefix);row.update(family=name,seed=seed);data['probes'].append(row)
            # Solver records ask whether local predictive acceptance survives
            # restarts; targets/objective evidence are not used for acceptance.
            if seed==0 and name in ['camera','barbara','camera_permuted','entropy_simplex','admm_shadow_1.0']:
                for method in ['ordinary','fixed','discovered','anderson']:
                    row=solve(model,method);row.update(family=name,seed=seed);data['solves'].append(row)
            Path(out).write_text(json.dumps(json_safe(data),indent=2))
            print(seed,name,[(r['prefix'],r['accepted'],r['depth'],r['horizon'],round(r['relative_trajectory_error'],4)) for r in data['probes'][-3:]],flush=True)
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=32);p.add_argument('--seeds',type=int,default=2)
    a=p.parse_args();run(a.out,a.size,a.seeds)
