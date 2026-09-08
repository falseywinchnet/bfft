"""Sampled time-to-certificate gate, with all update costs charged."""
from pathlib import Path
import argparse
import json
import time
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.defect_relaxation import DefectRelaxation, scenes
from experiments.meyer_transport_audit.model import ReducedMeyerMap
from experiments.meyer_transport_audit.certificate import certificate


def run(size, steps, seed, out, methods):
    result = {'size':size,'max_passes':steps,'seed':seed,'sample_every':8,'cases':{},
              'note':'Common certificate checks excluded from update timing. Update timing includes projections, gradient stencils, history, dot tests and allocations. First hits are sampled every eight passes; no interpolation.'}
    order_rng = np.random.default_rng(seed+71)
    for lam,mu in [(.02,20),(.05,40),(.1,80)]:
        for name,f in scenes(size,seed).items():
            m = ReducedMeyerMap(f,lam,mu)
            key = f'{name}_lambda{lam}_mu{mu}'
            case = {}
            for method in order_rng.permutation(methods):
                op = DefectRelaxation(m,method)
                z = m.initial()
                rows = []
                elapsed = 0.
                for k in range(2,steps+1):
                    t = time.perf_counter()
                    z = op.step(z)
                    elapsed += time.perf_counter()-t
                    if k%8 == 0 or k == steps:
                        rows.append({'pass':k,'seconds':elapsed,'gap':certificate(m,z)['gap_per_pixel']})
                case[method] = rows
            result['cases'][key] = case
            Path(out).write_text(json.dumps(result,indent=2))
            print(key, {v:float(f'{r[-1]["gap"]:.4g}') for v,r in case.items()},flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--steps',type=int,default=1024)
    p.add_argument('--seed',type=int,default=1987)
    p.add_argument('--out',required=True)
    p.add_argument('--methods',default='ordinary,refresh,aligned_refresh,remembered_refresh,stale_refresh')
    a = p.parse_args()
    run(a.size,a.steps,a.seed,a.out,a.methods.split(','))
