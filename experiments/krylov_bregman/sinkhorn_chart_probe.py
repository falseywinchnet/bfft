"""Check transfer of the latest rational chart grammar without altering solves.

Eight acquisition steps are charged; future trajectories are diagnostic only.
This intentionally tests the actual coupled Sinkhorn map in its current dual
coordinates, rather than planting rational dynamics or learning from optima.
"""
import argparse,json
import numpy as np
from .sinkhorn_study import problem
from .sinkhorn_transport import Sinkhorn,gauge
from .general_geometry import discover_mobius,mobius_transport
from .core import finite_flow

def run():
    rows=[]
    for kind in ('image','mixture'):
      for eps in (.003,.01,.03):
        m=Sinkhorn(*problem(128,0,kind),eps)
        for prefix in (4,32,128):
            z=m.initial.copy()
            for _ in range(prefix):z=m.step(z)
            snapshots=[z.copy()]
            for _ in range(8):snapshots.append(m.step(snapshots[-1]))
            actual=z.copy()
            for _ in range(64):actual=m.step(actual)
            pred,b,_=finite_flow(m,z,4,64,False)
            scale=float(np.linalg.norm(actual-z))
            stationary=scale<1e-10
            row=dict(kind=kind,epsilon=eps,prefix=prefix,horizon=64,
                acquisition_calls=8,krylov_depth=b.actions,
                ordinary_displacement=scale,
                krylov_absolute_trajectory_error=float(np.linalg.norm(pred-actual)),
                krylov_relative_trajectory_error=None if stationary else float(np.linalg.norm(pred-actual)/scale))
            try:
                roots,rates,info=discover_mobius(snapshots)
                candidate=gauge(mobius_transport(z,roots,rates,64))
                row.update(chart_status='fitted',relation_defect=info['maximum_relation_defect'],
                    chart_absolute_trajectory_error=float(np.linalg.norm(candidate-actual)),
                    chart_relative_trajectory_error=None if stationary else float(np.linalg.norm(candidate-actual)/scale),
                    chart_marginal_error=m.error(candidate))
            except ValueError as exc:
                row.update(chart_status='rejected',reason=str(exc))
            rows.append(row)
    return {'note':'18 anchors; diagnostic replay not a runtime certificate; no rational chart is used in main timing study', 'rows':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    with open(a.out,'w') as f:json.dump(run(),f,indent=2)
