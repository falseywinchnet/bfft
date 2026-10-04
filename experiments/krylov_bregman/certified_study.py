"""Cost-matched Huber inverse-problem study; no hidden target in discovery."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
from PIL import Image
from .certified_piece import HuberMirror,discover_piece
from .core import Anderson,finite_flow


def make_model(side=8,seed=0,kind='positive',amplitude=1.):
    root=Path(__file__).resolve().parents[2]
    source=np.asarray(Image.open(root/'personal_deblurrer/source_assets/v3_skimage/camera.png').convert('L').resize((side,side),Image.Resampling.BOX),dtype=float).ravel()*amplitude
    rng=np.random.default_rng(seed);n=len(source);A=rng.normal(size=(4*n,n))/np.sqrt(4*n)
    if kind=='positive':A=np.abs(A)
    g=np.geomspace(1.,16.,n)
    return HuberMirror(A/np.sqrt(g),A@source,mirror_diagonal=g)


def solve(model,method,budget=20000,target=1e-3):
    z=model.initial.copy();gap0=model.gap(z);calls=0;actions=0;accepted=0;rejected=0
    aa=Anderson(4);cooldown=0;check=16;records=[];status='budget';represented=0
    started=time.perf_counter()
    while calls+actions<budget:
        if method=='certified' and cooldown==0 and budget-calls-actions>=10:
            d=discover_piece(model,z,maximum_horizon=4096)
            calls+=d.calls;actions+=d.actions;z=d.state;represented+=d.horizon
            if d.accepted:accepted+=1
            else:rejected+=1;cooldown=16
        elif method=='fixed' and calls+actions>=4 and budget-calls-actions>=6:
            z,b,_=finite_flow(model,z,4,16,False);calls+=1;actions+=b.actions;represented+=16
            z=model.step(z);calls+=1;represented+=1
        else:
            nxt=model.step(z);calls+=1;represented+=1
            z=aa.advance(z,nxt) if method=='anderson' else nxt
            cooldown=max(0,cooldown-1)
        if calls+actions>=check or calls+actions>=budget:
            ratio=model.gap(z)/gap0
            records.append({'work':calls+actions,'gap_ratio':float(ratio),'seconds':time.perf_counter()-started})
            check=(calls+actions)//16*16+16
            if ratio<=target:status='target';break
            if not np.isfinite(ratio) or ratio>1e10:status='diverged';break
    return {'method':method,'status':status,'seconds':time.perf_counter()-started,
            'calls':calls,'actions':actions,'work':calls+actions,'accepted':accepted,
            'rejected':rejected,'represented_ordinary_steps':represented,'trace':records}


def main(args):
    data={'configuration':vars(args),'implementation':'v2 block-doubled reduced path and batched feature checks','python':platform.python_version(),'numpy':np.__version__,'runs':[],'first_blocks':[]}
    for kind in ['positive','signed']:
        for seed in range(args.seeds):
            model=make_model(args.side,seed,kind)
            d=discover_piece(model,model.initial,maximum_horizon=4096)
            # Independent verification of first event skip, outside solve time.
            x=model.initial.copy()
            for _ in range(d.horizon):x=model.step(x)
            data['first_blocks'].append({'kind':kind,'seed':seed,'horizon':d.horizon,
                'depth':d.depth,'work':d.calls+d.actions,'accepted':d.accepted,
                'absolute_error':float(np.linalg.norm(x-d.state)),'bound':d.error_bound,'reason':d.reason})
            methods=['ordinary','fixed','certified','anderson']
            # Warm every method once, then alternate shuffled orders.
            for method in methods:solve(model,method,budget=64)
            rng=np.random.default_rng(200+seed)
            for repeat in range(args.repeats):
                for method in rng.permutation(methods):
                    row=solve(model,str(method),args.budget,args.target)
                    row.update(kind=kind,seed=seed,repeat=repeat);data['runs'].append(row)
                print(kind,seed,repeat,[(r['method'],r['status'],r['work'],round(r['seconds'],4)) for r in data['runs'][-4:]],flush=True)
                Path(args.out).write_text(json.dumps(data,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--side',type=int,default=8)
    p.add_argument('--seeds',type=int,default=3);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--budget',type=int,default=20000);p.add_argument('--target',type=float,default=1e-3)
    main(p.parse_args())
