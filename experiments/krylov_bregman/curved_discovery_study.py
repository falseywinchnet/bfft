"""Local amortization of the live curved certificate, with settled output."""
import argparse,json,time,statistics
from pathlib import Path
import numpy as np
from .curved_meyer import MeyerQuotient,discover_curved
from .curved_study import scenes


def main(args):
    data={'configuration':vars(args),'rows':[]}
    for name,image in scenes(args.size):
        model=MeyerQuotient(image);z=model.initial.copy();last=0
        for prefix in [4,32,128,512]:
            for _ in range(prefix-last):z=model.step(z)
            last=prefix
            samples=[];baseline=[]
            for repeat in range(args.repeats):
                start=time.perf_counter();d=discover_curved(model,z)
                candidate=model.step(d.candidate);samples.append(time.perf_counter()-start)
                actual=z.copy();start=time.perf_counter()
                for _ in range(d.horizon+1):actual=model.step(actual)
                baseline.append(time.perf_counter()-start)
            row={'scene':name,'prefix':prefix,'accepted':d.accepted,'depth':d.depth,'horizon':d.horizon,
                 'bound':d.bound,'relative_bound':d.relative_bound,'actual_settled_error':float(np.linalg.norm(actual-candidate)),
                 'geometry_points':d.geometry_points,'work':d.map_calls+d.tangent_actions+1,
                 'seconds_discovery_settle':statistics.median(samples),
                 'seconds_ordinary_horizon_settle':statistics.median(baseline)}
            data['rows'].append(row)
            print(name,prefix,d.accepted,d.depth,d.horizon,round(row['seconds_ordinary_horizon_settle']/row['seconds_discovery_settle'],2),flush=True)
        Path(args.out).write_text(json.dumps(data,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=64);p.add_argument('--repeats',type=int,default=5)
    main(p.parse_args())
