"""Event-aware transport inside SPORCO's actual adaptive BPDN update schedule.

A shortcut may not cross a rho-update or stopping-check iteration. It leaves
that iteration for an ordinary native pass, reconstructing all variables
needed by SPORCO's residual/update methods. Frozen-rho block bounds are valid;
a global bound against a different run's adaptive rho decisions is NOT claimed.
The original duality gap determines successful complete solves.
"""
import numpy as np
from sporco.admm.bpdn import BPDN
from .bpdn_adapter import SparseCoding
from .certified_transport import discover_certified
from .study_bpdn import gap


def solve(D,S,lam,maxiter=4096,target=1e-5,check_every=64,accelerated=True,
          tolerance=.02,depths=(2,4),warmup=16,cooldown=32,finite=False):
    opt=BPDN.Options(dict(Verbose=False,FastSolve=True,MaxMainIter=maxiter,
                         RelStopTol=0.,AbsStopTol=0.,ReturnX=False))
    solver=BPDN(D,S,lam,opt);count=0;next_attempt=warmup
    target_abs=target*.5*float(np.sum(S*S));period=opt['AutoRho','Period']
    stats=dict(accepted=0,rejected=0,map_calls=0,tangent_actions=0,checks=0,
               stopping_checks=0,converged=False,events=[],rho_events=[])
    while count<maxiter:
        next_rho=((count//period)+1)*period
        next_check=((count//check_every)+1)*check_every
        boundary=min(next_rho,next_check,maxiter)
        horizon=boundary-count-1
        if accelerated and count>=next_attempt and horizon>=5:
            model=SparseCoding(solver,allow_adaptive=True)
            z=(solver.Y+solver.U).ravel()
            candidate,event=discover_certified(model,z,horizon,depths=depths,tolerance=tolerance)
            for key in ('map_calls','tangent_actions','checks'):stats[key]+=event[key]
            if event['accepted']:
                solver.Y=model.output(candidate);solver.U=candidate.reshape(solver.Y.shape)-solver.Y
                count+=event['horizon'];stats['accepted']+=1
                stats['events'].append(dict(at=count-event['horizon'],rho=float(solver.rho),**event))
                next_attempt=count
                continue
            stats['rejected']+=1;next_attempt=count+cooldown
            # Rejected linearization produced v_next but not the full native
            # X needed for every later control path. Charge the native retry.
        solver.k=count
        solver.Yprev=solver.Y.copy()
        solver.xstep();solver.relax_AX();solver.ystep();solver.ustep()
        stats['map_calls']+=1;count+=1
        # Match upstream residual computations even when not used by an event.
        r,s,epri,edua=solver.compute_residuals()
        before=float(solver.rho);solver.update_rho(count-1,r,s)
        if float(solver.rho)!=before:stats['rho_events'].append(dict(at=count,old=before,new=float(solver.rho)))
        if count%check_every==0 or count==maxiter:
            stats['stopping_checks']+=1;objective,duality=gap(D,S,solver.Y,lam)
            if not finite and duality<=target_abs:
                stats['converged']=True;break
    stats.update(passes=count,target_gap=target_abs,final_gap=duality,objective=objective)
    return solver.Y,stats
