"""Finite-pass transport with model-supplied nonlinear feature remainders.

Reuses the existing Arnoldi engine and doubling recurrence. A model provides
linearize(state), feature_relation(state, basis), output(state), output_gain,
and optionally remainder_gain. The feature relation supplies the squared
norm of its nonlinear remainder at every predicted input. The adapter must
prove nonexpansivity of its map and the remainder transfer norm bound.

This module does not infer those mathematical assumptions from an API shape.
The bound is an exact-arithmetic statement evaluated in float64, without
outward-rounded interval arithmetic. The running sum is relative to the same
fixed map from the same initial state, not to an independent adaptive policy.
"""
import numpy as np
from experiments.krylov_bregman.discovery import arnoldi_stream
from experiments.krylov_bregman.certified_piece import reduced_path


def discover_certified(model,z,maximum_horizon=64,depths=(2,4),tolerance=.02):
    lin=model.linearize(z);checks=0;actions=0
    for basis,closed in arnoldi_stream(lin.action,lin.next_state-z,max(depths)):
        actions=basis.actions
        if actions not in depths and not closed:continue
        disk=model.feature_relation(z,basis.Q)
        source=np.zeros(actions);source[0]=basis.beta
        C=reduced_path(basis.H,source,maximum_horizon+1)
        gram=basis.remainder.T@basis.remainder
        compression=np.sqrt(np.maximum(0,np.sum(C[:,:-1]*(gram@C[:,:-1]),axis=0)))
        best=None;total=0.
        for lo in range(0,maximum_horizon,8):
            hi=min(lo+8,maximum_horizon)
            curvature=getattr(model,"remainder_gain",1.)*np.sqrt(np.maximum(disk.squared_remainder_norms(C[:,lo:hi]),0))
            bounds=total+np.cumsum(curvature+compression[lo:hi]);total=bounds[-1]
            checks+=hi-lo
            for j in range(lo,hi):
                bound=float(bounds[j-lo]);relative=bound/max(np.linalg.norm(C[:,j+1]),1e-300)
                if j+1>=1.5*(1+actions) and relative<=tolerance:
                    best=(j+1,bound,relative)
            # Conservative early exit, may miss a later useful horizon.
            if total>tolerance*max(np.linalg.norm(C[:,hi]),1e-300):break
        if best:
            h,b,r=best
            return z+basis.Q@C[:,h],dict(accepted=True,horizon=h,depth=actions,
                       bound=b,relative_bound=r,checks=checks,map_calls=1,tangent_actions=actions)
        if closed:break
    return lin.next_state,dict(accepted=False,horizon=1,depth=actions,bound=0.,
                    relative_bound=0.,checks=checks,map_calls=1,tangent_actions=actions)


def run_certified(model,passes,tolerance=.02,warmup=8,cooldown=32,depths=(2,4),
                  stop=None,check_every=64,accelerated=True):
    """Bounded finite trajectory or solve with a charged common stopping test.

    When stop is supplied, every method visits exactly the same iteration
    checkpoints. An admitted jump never skips a stopping checkpoint.
    """
    z=model.initial.copy();count=0;next_attempt=warmup;events=[]
    stats=dict(map_calls=0,tangent_actions=0,accepted=0,rejected=0,checks=0,bound=0.,
               stopping_checks=0,converged=False)
    while count<passes:
        remaining=passes-count
        if stop is not None:remaining=min(remaining,check_every-count%check_every)
        if accelerated and count>=next_attempt and remaining>=8:
            candidate,event=discover_certified(model,z,min(64,remaining),depths=depths,tolerance=tolerance)
            for key in ('map_calls','tangent_actions','checks'):stats[key]+=event[key]
            z=candidate;count+=event['horizon']
            stats['accepted' if event['accepted'] else 'rejected']+=1
            stats['bound']+=event['bound']
            events.append(dict(at=count-event['horizon'],**event))
            next_attempt=count if event['accepted'] else count+cooldown
        else:
            z=model.step(z);count+=1;stats['map_calls']+=1
        if stop is not None and (count%check_every==0 or count==passes):
            stats['stopping_checks']+=1
            if stop(model.output(z)):
                stats['converged']=True
                break
    stats.update(passes=count,events=events,output_bound=model.output_gain*stats['bound'])
    return model.output(z),stats
