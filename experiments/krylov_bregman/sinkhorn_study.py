"""Predeclared transfer benchmark, dense unstructured entropic OT."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from .sinkhorn_transport import Sinkhorn,solve
ROOT=Path(__file__).resolve().parents[2]


def problem(n,seed,kind):
    rng=np.random.default_rng(seed);x=rng.uniform(0,1,(n,2));y=rng.uniform(0,1,(n,2))
    cost=np.sum((x[:,None,:]-y[None,:,:])**2,axis=2)
    if kind=='image':
        files=['personal_deblurrer/source_assets/v3_skimage/camera.png',
               'paper/fast_meyer_bregman/assets/barbara_512.tif']
        masses=[]
        for points,file in zip([x,y],files):
            im=np.asarray(Image.open(ROOT/file).convert('L'),dtype=float)/255
            ij=np.minimum((points*np.array(im.shape)).astype(int),np.array(im.shape)-1)
            masses.append(.01+im[ij[:,0],ij[:,1]])
        p,q=masses
    elif kind=='mixture':
        def bumps(points,centers,width):
            return .001+sum(np.exp(-np.sum((points-c)**2,axis=1)/(2*width**2)) for c in centers)
        p=bumps(x,[[.2,.25],[.25,.75]],.10)
        q=bumps(y,[[.7,.4],[.75,.65]],.13)
    else:raise ValueError(kind)
    return cost,p/p.sum(),q/q.sum()


def run(args):
    data={'configuration':vars(args),'protocol':'dense unstructured costs; setup excluded from solver '
        'time and separately reported; all acquisition/actions/checks/rejections/settling timed; '
        'same 1e-8 L1 marginal target; no epsilon continuation; sampled discovery is not a certificate',
        'runs':[]}
    for n in map(int,args.sizes.split(',')):
      for seed in range(args.seeds):
       for kind in args.kinds.split(','):
        for eps in map(float,args.epsilons.split(',')):
            start=time.perf_counter();C,p,q=problem(n,seed,kind);m=Sinkhorn(C,p,q,eps)
            setup=time.perf_counter()-start
            methods=['ordinary','fixed','discovered','anderson']
            for method in methods: solve(m,method,target=0,budget=48)
            reference,rr=solve(m,'ordinary',target=1e-11,budget=args.budget*2)
            refplan=m.plan(reference)
            refobj=float(np.sum(refplan*C)+eps*np.sum(refplan*(np.log(np.maximum(refplan,1e-300))-1)))
            rng=np.random.default_rng(2026)
            for repeat in range(args.repeats):
                batch=[]
                for method in rng.permutation(methods):
                    z,row=solve(m,str(method),target=args.target,budget=args.budget)
                    plan=m.plan(z)
                    objective=float(np.sum(plan*C)+eps*np.sum(plan*(np.log(np.maximum(plan,1e-300))-1)))
                    row.update(n=n,seed=seed,kind=kind,epsilon=eps,repeat=repeat,
                        setup_seconds=setup,reference_status=rr['status'],
                        plan_l1_to_reference=float(np.sum(np.abs(plan-refplan))),
                        objective=objective,reference_objective=refobj,
                        row_l1=float(np.sum(np.abs(plan.sum(axis=1)-p))),
                        column_l1=float(np.sum(np.abs(plan.sum(axis=0)-q))))
                    data['runs'].append(row);batch.append((method,row['status'],row['work'],round(row['seconds'],4),row['accepted']))
                Path(args.out).write_text(json.dumps(data,indent=2))
                print(n,seed,kind,eps,repeat,batch,flush=True)
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--sizes',default='128,512');p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--kinds',default='image,mixture');p.add_argument('--epsilons',default='.003,.01,.03')
    p.add_argument('--repeats',type=int,default=7);p.add_argument('--target',type=float,default=1e-8)
    p.add_argument('--budget',type=int,default=20000)
    run(p.parse_args())
