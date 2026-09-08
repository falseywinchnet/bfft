"""Follow the difficult one-dimensional edge through the convergence tail."""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap, grad
from experiments.meyer_transport_audit.defect_relaxation import DefectRelaxation
from experiments.meyer_transport_audit.certificate import certificate


def run():
    result = {}
    for n,lam,mu in [(64,.05,40),(128,.1,80)]:
        f = 50.+150.*(np.arange(n)[None,:]>n//2)
        m = ReducedMeyerMap(f,lam,mu)
        case = {}
        for method in ['ordinary','refresh','aligned_refresh','remembered_refresh']:
            op = DefectRelaxation(m,method)
            z = m.initial()
            first = {str(t):None for t in [.01,.001,.0001,.00001]}
            release = None
            rows = []
            for k in range(2,4097):
                z = op.step(z)
                if release is None and k>64 and grad(z.u)[0][0,n//2+1] < 0:
                    release = k
                if k%8 == 0:
                    gap = certificate(m,z)['gap_per_pixel']
                    for t in first:
                        if first[t] is None and gap<=float(t):first[t] = k
                    if k%128 == 0:rows.append({'pass':k,'gap':gap})
            case[method] = {'first_negative_shoulder_gradient_after64':release,
                            'first_sampled_pass':first,'checkpoints':rows}
        result[f'n{n}_lambda{lam}_mu{mu}'] = case
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    a = p.parse_args()
    result = run()
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{m:{'release':r['first_negative_shoulder_gradient_after64'],'hits':r['first_sampled_pass'],'final':r['checkpoints'][-1]} for m,r in c.items()} for k,c in result.items()},indent=2))
