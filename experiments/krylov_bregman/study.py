"""Cost-counted pilot, with no tuning on outcomes and no hidden acceptance gate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import time
import numpy as np
from .core import Anderson, finite_flow, local_arnoldi, shadow_diagnostic, metric_norm
from .problems import FAMILIES

METHODS = ['ordinary','flow_euclidean','flow_mirror','stationary_krylov','anderson']


def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [json_safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def solve(model, method, budget=4096, target=1e-6, depth=4, horizon=16,
          prefix=4, check_every=16):
    z=model.initial.copy()
    gap0=model.gap(z)
    calls=0;actions=0;updates=0;macros=0;checks=0
    next_check=check_every
    aa=Anderson(depth)
    rows=[{'work':0,'gap_ratio':1.,'seconds':0.}]
    status='budget'
    started=time.perf_counter()
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            while calls+actions < budget:
                work=calls+actions
                if (method.startswith('flow') or method=='stationary_krylov') and work>=prefix and work+depth+2<=budget:
                    basis,_=local_arnoldi(model,z,depth,method!='flow_euclidean')
                    calls+=1;actions+=basis.actions;macros+=1
                    if method=='stationary_krylov':
                        candidate=z.copy()
                        if basis.actions:
                            rhs=np.zeros(basis.actions);rhs[0]=basis.beta
                            coefficients=np.linalg.solve(np.eye(basis.actions)-basis.H,rhs)
                            candidate=z+basis.Q@coefficients
                    else:
                        candidate=z+basis.displacement(horizon)
                    z=model.step(candidate);calls+=1
                else:
                    nxt=model.step(z);calls+=1
                    z=aa.advance(z,nxt) if method=='anderson' else nxt
                updates+=1
                if not np.all(np.isfinite(z)):
                    raise FloatingPointError('nonfinite state')
                work=calls+actions
                if work>=next_check or work>=budget:
                    ratio=model.gap(z)/max(gap0,np.finfo(float).tiny);checks+=1
                    elapsed=time.perf_counter()-started
                    rows.append({'work':work,'gap_ratio':float(ratio),'seconds':elapsed})
                    next_check=(work//check_every+1)*check_every
                    if not np.isfinite(ratio) or ratio>1e12:
                        status='diverged';break
                    if ratio<=target:
                        status='target';break
    except (FloatingPointError,np.linalg.LinAlgError) as error:
        status='invalid: '+str(error)
    elapsed=time.perf_counter()-started
    final_gap=model.gap(z) if np.all(np.isfinite(z)) else float('inf')
    final_ratio=final_gap/max(gap0,np.finfo(float).tiny)
    # Post-run residual is explicitly a separate diagnostic, not an uncharged
    # acceptance or stopping evaluation.
    try:
        residual=float(np.linalg.norm(model.step(z)-z))
    except (FloatingPointError,ValueError):
        residual=float('inf')
    return {'method':method,'status':status,'work':calls+actions,'map_calls':calls,
            'tangent_actions':actions,'objective_checks':checks,'updates':updates,
            'macros':macros,'seconds':elapsed,'final_gap_ratio':float(final_ratio),
            'postrun_residual_norm':residual,'postrun_diagnostic_map_calls':1,'trace':rows}


def finite_screen(model):
    z=model.initial.copy();out=[]
    for prefix in [4,32,128]:
        z=model.initial.copy()
        for _ in range(prefix):z=model.step(z)
        for depth in [2,4,8]:
            for horizon in [4,16,64]:
                for natural in [False,True]:
                    predicted,basis,lin=finite_flow(model,z,depth,horizon,natural)
                    # Separate error of the full frozen derivative from its
                    # Krylov compression; all these actions are diagnostic.
                    affine=np.zeros_like(z);ordinary=z.copy();r=lin.next_state-z
                    for _ in range(horizon):
                        affine=r+lin.action(affine)
                        ordinary=model.step(ordinary)
                    norm=lambda v:metric_norm(v,basis.metric)
                    denominator=max(norm(ordinary-z),1e-30)
                    out.append({'prefix':prefix,'depth':depth,'horizon':horizon,
                                'metric':'mirror' if natural else 'euclidean',
                                'rank':basis.actions,
                                'relative_total_error':norm(predicted-ordinary)/denominator,
                                'relative_frozen_error':norm(z+affine-ordinary)/denominator,
                                'relative_krylov_error':norm(predicted-z-affine)/denominator,
                                'predicted_gap_ratio':model.gap(predicted)/max(model.gap(z),1e-30),
                                'reference_gap_ratio':model.gap(ordinary)/max(model.gap(z),1e-30),
                                'diagnostic_map_calls':horizon,'diagnostic_action_calls':horizon})
    return out


def run(args):
    families={k:v for k,v in FAMILIES.items() if k in args.families.split(',')}
    if not families:raise ValueError('no recognized families')
    def make(cls,seed):
        opts={'n':args.n,'seed':seed,'condition':args.condition}
        if cls.name in ['admm_lasso','admm_shadow']:opts['rho']=args.admm_rho
        return cls(**opts)
    configuration={'protocol':'2026-09-29-v2','n':args.n,'seeds':args.seeds,'repeats':args.repeats,
                   'budget':args.budget,'target':args.target,'depth':4,'horizon':16,
                   'condition':args.condition,'admm_rho':args.admm_rho,
                   'prefix':4,'check_every':16,'families':list(families),
                   'methods':METHODS,'python':platform.python_version(),
                   'numpy':np.__version__,'machine':platform.machine(),
                   'note':'Pilot; equal work counts one map or one analytic JVP. '
                          'Wall time includes acquisition, metrics, Arnoldi, reconstruction, '
                          'ordinary settling and spaced objective checks; setup excluded. '
                          'No guards, refreshes, adaptive horizons, or outcome-based tuning. '
                          'Anderson is an unguarded regularized type-II comparator.'}
    output={'configuration':configuration,'runs':[],'finite_screens':{}}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.config.json').write_text(json.dumps(configuration,indent=2))
    for family,cls in families.items():
        for seed in range(args.seeds):
            start=time.perf_counter();model=make(cls,seed)
            setup_seconds=time.perf_counter()-start
            # Warm all methods before randomized, interleaved repeated timings.
            for method in METHODS:
                solve(model,method,budget=32,target=0.)
            records=defaultdict(list)
            for repeat in range(args.repeats):
                order=list(METHODS);np.random.default_rng(seed+910*repeat).shuffle(order)
                for method in order:
                    result=solve(model,method,budget=args.budget,target=args.target)
                    records[method].append(result)
            for method in METHODS:
                rows=records[method]
                # Numeric outcome must be deterministic across timed repeats.
                assert all((r['work'],r['status'])==(rows[0]['work'],rows[0]['status']) for r in rows)
                record=rows[0].copy()
                record.update({'family':family,'seed':seed,'setup_seconds':setup_seconds,
                               'seconds_median':float(np.median([r['seconds'] for r in rows])),
                               'seconds_all':[r['seconds'] for r in rows]})
                output['runs'].append(record)
                print(f"{family} seed={seed} {method}: {record['status']} work={record['work']} "
                      f"gap={record['final_gap_ratio']:.3g} time={record['seconds_median']:.4f}",flush=True)
            out.write_text(json.dumps(json_safe(output),indent=2,allow_nan=False))
        if not args.skip_finite:
            output['finite_screens'][family]=finite_screen(make(cls,0))
            out.write_text(json.dumps(json_safe(output),indent=2,allow_nan=False))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=96)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--budget',type=int,default=4096)
    p.add_argument('--target',type=float,default=1e-6)
    p.add_argument('--families',default=','.join(FAMILIES))
    p.add_argument('--condition',type=float,default=1000.)
    p.add_argument('--admm-rho',type=float,default=.1)
    p.add_argument('--skip-finite',action='store_true')
    p.add_argument('--out',required=True)
    run(p.parse_args())
