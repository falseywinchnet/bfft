import argparse,json,pathlib
import numpy as np
from walk import Gate,serialize_gates
from gauge import best_gauge
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--out',default='/tmp/fourier_gauge_walk');args=p.parse_args();out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True);rows=[]
for path in sorted(pathlib.Path(args.source).glob('n*_seed*.json')):
    record=json.loads(path.read_text());n=record['n'];gates=[Gate(g['i'],g['j'],np.array(g['matrix'])) for g in record['physical_gates']]
    original,best=best_gauge(gates,n,64)
    row={'source':path.name,'N':n,'original':original,**{k:v for k,v in best.items() if k not in ('gates','lambdas')}}
    (out/path.name).write_text(json.dumps(dict(row,physical_gates=serialize_gates(best['gates']),lambdas=best['lambdas']),indent=2)+'\n')
    rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
