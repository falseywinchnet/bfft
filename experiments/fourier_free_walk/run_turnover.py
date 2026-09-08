import argparse,json,pathlib,time
from turnover import search
from walk import serialize_gates

p=argparse.ArgumentParser();p.add_argument('--general',action='store_true');p.add_argument('--start-kind',choices=['sparse','qr'],default='sparse');p.add_argument('--sizes',default='4,6,8,12,16');p.add_argument('--seeds',type=int,default=4);p.add_argument('--steps',type=int,default=5000);p.add_argument('--out',default='/tmp/fourier_turnover_walk');args=p.parse_args()
out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True);rows=[]
for n in map(int,args.sizes.split(',')):
    for seed in range(args.seeds):
        start=time.monotonic();result=search(n,seed,args.steps,start_kind=args.start_kind,general_frames=args.general)
        row={k:v for k,v in result.items() if k not in ('reverse_gates','physical_gates','history')};row.update(seed=seed,steps=args.steps,seconds=time.monotonic()-start)
        (out/f'n{n}_seed{seed}.json').write_text(json.dumps(dict(row,history=result['history'],reverse_gates=serialize_gates(result['reverse_gates']),physical_gates=serialize_gates(result['physical_gates'])),indent=2)+'\n')
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
