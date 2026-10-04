"""Controlled discovery experiment, not a universal performance benchmark."""
import argparse,json,time
import numpy as np
from .general_geometry import RationalMirror,observable,fit_recurrence,predict,discover_mobius,mobius_transport

def run():
    rows=[]
    for seed in range(3):
        for kind in ('euclidean','quartic','entropy','hypentropy'):
            m=RationalMirror(kind,seed=seed)
            x=m.initial.copy(); states=[]
            start=time.perf_counter()
            for j in range(9):
                states.append(m.encode(x))
                if j<8: x=m.step(x)
            acquisition=time.perf_counter()-start
            for name in ('dual','log','reciprocal'):
                start=time.perf_counter()
                data=[observable(s,name) for s in states]
                Q,T,info=fit_recurrence(data)
                fitting=time.perf_counter()-start
                for horizon in (16,64,128):
                    start=time.perf_counter(); lifted=predict(Q,T,data[0],horizon)
                    if name=='reciprocal': s=1/lifted[1:]
                    elif name=='log': s=np.exp(lifted[1:])
                    else: s=lifted[1:]
                    predicted=m.decode(s)
                    transport=time.perf_counter()-start
                    reference=m.initial.copy()
                    start=time.perf_counter()
                    for _ in range(horizon): reference=m.step(reference)
                    ordinary=time.perf_counter()-start
                    rows.append(dict(kind=kind,seed=seed,observable=name,horizon=horizon,
                        **info,relative_primal_error=float(np.linalg.norm(predicted-reference)/
                        max(np.linalg.norm(reference),1e-30)),acquisition_seconds=acquisition,
                        fitting_seconds=fitting,transport_seconds=transport,
                        ordinary_seconds=ordinary,total_identification_seconds=acquisition+fitting+transport))
            start=time.perf_counter()
            roots,rates,info=discover_mobius(states)
            fitting=time.perf_counter()-start
            for horizon in (16,64,128):
                start=time.perf_counter()
                predicted=m.decode(mobius_transport(states[0],roots,rates,horizon))
                transport=time.perf_counter()-start
                reference=m.initial.copy()
                start=time.perf_counter()
                for _ in range(horizon): reference=m.step(reference)
                ordinary=time.perf_counter()-start
                rows.append(dict(kind=kind,seed=seed,observable='synthesized_crossratio',
                    horizon=horizon,**info,
                    relative_primal_error=float(np.linalg.norm(predicted-reference)/
                        max(np.linalg.norm(reference),1e-30)),acquisition_seconds=acquisition,
                    fitting_seconds=fitting,transport_seconds=transport,
                    ordinary_seconds=ordinary,total_identification_seconds=acquisition+fitting+transport))
    return {'scope':'constructed convex nonlinear mirror family; supplied three-chart dictionary and degree-(1,1) rational grammar; '
             '8 training transitions; timing diagnostic only, no repeated performance gate',
            'rows':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    with open(a.out,'w') as f: json.dump(run(),f,indent=2)
