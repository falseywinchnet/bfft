import argparse,json,pathlib,time
import numpy as np
from walk import synthesize,orient,verify,serialize_gates


def main():
    p=argparse.ArgumentParser();p.add_argument('--sizes',default='4,6,8,12,16');p.add_argument('--seeds',type=int,default=3);p.add_argument('--out',default='/tmp/fourier_free_walk');args=p.parse_args()
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for n in map(int,args.sizes.split(',')):
        for oblique in (False,True):
            for seed in range(args.seeds):
                start=time.monotonic()
                walk=synthesize(n,seed,oblique,explore=0 if seed==0 else 1.0)
                route=orient(walk['permutation'],[(g.i,g.j) for g in reversed(walk['gates'])]) if walk['complete'] else {'success':False,'exhaustive':False}
                row={'N':n,'seed':seed,'oblique_allowed':oblique,'reduction_complete':walk['complete'],
                     'gate_count':len(walk['gates']),'gates_over_N_log2N':len(walk['gates'])/(n*np.log2(n)),
                     'terminal_permutation':walk['permutation'],'orientation':route,
                     'attempted_cells':walk['attempted_cells'],'peak_condition':walk['peak_condition'],
                     'oblique_cells_used':sum(t['oblique'] for t in walk['trace']),
                     'fill_in_steps':sum(t['support_gain']<0 for t in walk['trace'])}
                if route['success']:
                    physical,error=verify(walk,route);row['max_matrix_error']=error
                    label=f'n{n}_{"gl" if oblique else "orth"}_seed{seed}'
                    (out/(label+'.json')).write_text(json.dumps({'summary':row,'reverse_gates':serialize_gates(walk['gates']),
                        'physical_gates':serialize_gates(physical),'trace':walk['trace']},indent=2)+'\n')
                row['seconds']=time.monotonic()-start;rows.append(row)
                (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps(row),flush=True)

if __name__=='__main__':main()
