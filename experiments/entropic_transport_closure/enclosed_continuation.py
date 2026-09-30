"""Finite chart continuation with a sound nonlinear gradient enclosure.

Exact-arithmetic inequalities, evaluated in ordinary float64 (not interval
arithmetic). No unknown optimum or future kernel evaluation informs a rollout.
The mixture-inspired merge preserves inclusion. No probability is attached to
approximation uncertainty. All source contributions coexist with weights a.
"""
import time
import numpy as np
from .context_descent import evaluate, evaluate_factored, objective_difference, conditional_action


def phi2(x):
    if abs(x)<1e-4:
        return .5+x/6+x*x/24+x*x*x/120+x**4/720
    return float((np.expm1(x)-x)/(x*x))


def reduce_generators(G,budget):
    """Keep large generators; box the rest, preserving their Minkowski sum."""
    r,p=G.shape
    if budget<r: raise ValueError('budget must be at least the ambient dimension')
    if p<=budget: return G.copy()
    order=np.argsort(-np.sum(G*G,axis=0),kind='stable')
    keep=order[:budget-r]; drop=order[budget-r:]
    return np.column_stack((G[:,keep],np.diag(np.abs(G[:,drop]).sum(axis=1))))


def merge_zonotopes(c1,G1,c2,G2,budget):
    p=max(G1.shape[1],G2.shape[1]); r=len(c1)
    A=np.zeros((r,p)); B=A.copy(); A[:,:G1.shape[1]]=G1; B[:,:G2.shape[1]]=G2
    G=.5*np.column_stack((A+B,A-B,c1-c2))
    return (c1+c2)/2,reduce_generators(G,budget)


def merged_cloud(points,budget):
    """Balanced adjacent merges in the first chart coordinate; no fitted data.
    This uses the paper's enclosing operation, NOT its all-pairs greedy search.
    """
    r=points.shape[1]
    items=[(p.copy(),np.empty((r,0))) for p in points[np.argsort(points[:,0])]]
    while len(items)>1:
        out=[]
        for j in range(0,len(items)-1,2):
            out.append(merge_zonotopes(*items[j],*items[j+1],budget))
        if len(items)%2: out.append(items[-1])
        items=out
    return items[0]


def hessian_action(state,a,b,w):
    root=np.sqrt(b); v=w/root; pv=conditional_action(state,v)
    if 'P' in state: term=state['P'].T@(a*pv)
    else: term=state['v']*(state['K'].T@(a*pv/state['kv']))
    result=(state['c']*v-term)/root
    return result-root*(root@result)


def prepare(state,a,b,rank=4,enclosure='box',budget=None,candidates=None):
    root=np.sqrt(b); basis=[]; actions=[]; w=state['g'].copy()
    def orthogonalize(w):
        w=w.copy()
        for _ in range(2):
            w-=root*(root@w)
            for u in basis: w-=u*(u@w)
        return w
    if candidates is None:
        for _ in range(min(rank,len(b)-1)):
            w=orthogonalize(w); norm=np.linalg.norm(w)
            if norm<1e-14: break
            u=w/norm; hu=hessian_action(state,a,b,u)
            basis.append(u); actions.append(hu); w=hu.copy()
        if not basis: return None
        U=np.column_stack(basis); V=U/root[:,None]; HU=np.column_stack(actions)
        A=U.T@HU; A=(A+A.T)/2; products=2*len(basis)
    else:
        for candidate in [w]+list(candidates):
            if len(basis)>=min(rank,len(b)-1): break
            norm0=np.linalg.norm(candidate)
            if norm0==0: continue
            v=orthogonalize(candidate/norm0); norm=np.linalg.norm(v)
            if norm<1e-8: continue
            basis.append(v/norm)
        if not basis: return None
        U=np.column_stack(basis); V=U/root[:,None]
        if 'P' in state: PV=state['P']@V
        else: PV=(state['K']@(state['v'][:,None]*V))/state['kv'][:,None]
        A=V.T@(state['c'][:,None]*V)-PV.T@(a[:,None]*PV)
        A=(A+A.T)/2; products=len(basis)
    eigen=np.linalg.eigvalsh(A)
    if eigen[0]<=0: return None
    r=len(basis)
    if enclosure=='box':
        G=np.diag(np.ptp(V,axis=0)/2)
    elif enclosure=='merged':
        _,G=merged_cloud(V,2*r if budget is None else budget)
    elif enclosure=='exact': G=None
    else: raise ValueError(enclosure)
    return dict(U=U,V=V,A=A,g=U.T@state['g'],G=G,enclosure=enclosure,
                largest_eigenvalue=float(eigen[-1]),rank=r,acquisition_products=products)


def width(model,d):
    if model['G'] is None: return float(np.ptp(model['V']@d))
    return float(2*np.abs(model['G'].T@d).sum())


def gradient_support(model,s,d):
    """Upper bound on true projected gradient(s).d."""
    R=float(np.ptp(model['V']@s))
    energy=max(0.,float(s@model['A']@s))
    center=model['g']+model['A']@s
    uncertainty=energy*phi2(R)*width(model,d)
    return float(center@d+uncertainty),uncertainty


def increment_bound(model,s,d):
    """Upper bound on F(y+V(s+d))-F(y+Vs)."""
    support,uncertainty=gradient_support(model,s,d)
    R=float(np.ptp(model['V']@s)); D=float(np.ptp(model['V']@d))
    quadratic=max(0.,float(d@model['A']@d))
    curvature=float(np.exp(R)*phi2(D)*quadratic)
    return support+curvature,support,uncertainty,curvature


def rollout(model,max_steps=64,max_radius=20.,relative_stop=1e-4):
    s=np.zeros(model['rank']); scale=1/model['largest_eigenvalue']
    start_norm=np.linalg.norm(model['g']); path=[]; reason='step ceiling'
    for _ in range(max_steps):
        g=model['g']+model['A']@s
        if np.linalg.norm(g)<=relative_stop*start_norm:
            reason='model stationary'; break
        direction=-scale*g
        support,_=gradient_support(model,s,direction)
        if support>=-.05*abs(float(g@direction)):
            reason='uncertainty blocks direction'; break
        accepted=False
        for j in range(50):
            d=(.5**j)*direction
            if np.ptp(model['V']@(s+d))>max_radius: continue
            if np.linalg.norm(d)<1e-6*max(np.linalg.norm(s),start_norm/model['largest_eigenvalue']):
                break
            bound,first,error,curve=increment_bound(model,s,d)
            if bound<.01*first:
                accepted=True; break
        if not accepted:
            reason='radius or finite step bound'; break
        actual_width=float(np.ptp(model['V']@d)); enclosed=width(model,d)
        path.append(dict(s=s.tolist(),d=d.tolist(),bound=bound,support=first,
                         uncertainty=error,curvature=curve,
                         compression_width_ratio=enclosed/max(actual_width,1e-300)))
        s+=d
        Ad=model['A']@d
        if d@Ad>0: scale=float(d@d/(d@Ad))
    return dict(coords=s,path=path,reason=reason,steps=len(path),
                objective_bound=sum(x['bound'] for x in path))


def solve(logK,a,b,enclosure='box',rank=4,tolerance=1e-9,max_evaluations=4000,
          warmup=8,max_inner=64,min_inner=2,cooldown=4,chart='snapshot'):
    start=time.perf_counter(); K=np.exp(logK); root=np.sqrt(b)
    state=evaluate_factored(K,logK,a,b,np.zeros(len(b)))
    evaluations=1; acquisitions=0; acquisition_products=0; trials=0; scalar=1.
    blocks=[]; history=[]; attempts=[]; fallback_steps=0; failure=None; wait=warmup
    directions=[]; acquisition_seconds=0.; rollout_seconds=0.
    while state['residual']>tolerance and evaluations<max_evaluations:
        old=state; proposal=None; model=None
        if wait<=0:
            acquisitions+=1
            ta=time.perf_counter()
            model=prepare(state,a,b,rank,enclosure,candidates=directions if chart=='snapshot' else None)
            acquisition_seconds+=time.perf_counter()-ta
            acquisition_products+=2*rank if model is None else model['acquisition_products']
            if model is not None:
                tr=time.perf_counter()
                proposal=rollout(model,max_steps=max_inner)
                rollout_seconds+=time.perf_counter()-tr
                attempts.append(dict(steps=proposal['steps'],reason=proposal['reason'],
                                     bound=proposal['objective_bound']))
        if proposal is not None and proposal['steps']>=min_inner:
            z=state['z']+model['U']@proposal['coords']
            new=evaluate_factored(K,logK,a,b,z); evaluations+=1
            # This evaluation is only for reanchoring and stopping, not to
            # authorize rollout steps retrospectively. Independent audit below.
            blocks.append(dict(anchor_y=state['y'].tolist(),V=model['V'].tolist(),
                               path=proposal['path'],reason=proposal['reason'],
                               endpoint_residual=new['residual']))
            history.append(dict(kind='enclosed',steps=proposal['steps'],
                                residual=new['residual'],bound=proposal['objective_bound']))
            wait=0
        else:
            if model is not None: wait=cooldown
            direction=-scalar*state['g']; slope=float(state['g']@direction)
            alpha=min(1.,10/max(float(np.ptp(direction/root)),1e-300))
            for j in range(60):
                diff=objective_difference(state,alpha*direction/root,a,b); trials+=1
                if diff<=1e-4*alpha*slope: break
                alpha*=.5
            else: failure='fallback line search exhausted'; break
            new=evaluate_factored(K,logK,a,b,state['z']+alpha*direction); evaluations+=1
            history.append(dict(kind='scalar',steps=1,residual=new['residual'],difference=diff))
            fallback_steps+=1; wait-=1
        s=new['z']-old['z']; t=new['g']-old['g']
        directions=[s.copy(),t.copy()]+directions[:2*rank-2]
        if s@t>1e-12*np.linalg.norm(s)*np.linalg.norm(t) and t@t>0:
            scalar=float(s@t/(t@t))
        state=new
    certificate=evaluate(logK,a,b,state['z']); evaluations+=1
    return dict(enclosure=enclosure,rank=rank,chart=chart,acquisition_seconds=acquisition_seconds,rollout_seconds=rollout_seconds,converged=certificate['residual']<=tolerance,
                residual=certificate['residual'],seconds=time.perf_counter()-start,
                evaluations=evaluations,acquisitions=acquisitions,
                acquisition_products=acquisition_products,trials=trials,
                kernel_vector_products=2*(evaluations-1)+acquisition_products+trials,
                products_exclude_final_log_certificate=True,
                fallback_steps=fallback_steps,blocks=blocks,history=history,
                attempts=attempts,failure=failure,y=state['y'].tolist())


def audit(logK,a,b,answer):
    """Offline scoring ONLY. Every admitted internal step independently checked.
    Not included in solver timing and never used to build or accept a proposal.
    """
    max_violation=-np.inf; positive=0; checked=0; max_support=-np.inf
    worst_relative=-np.inf
    for block in answer['blocks']:
        anchor=np.array(block['anchor_y']); V=np.array(block['V'])
        for item in block['path']:
            s=np.array(item['s']); d=np.array(item['d'])
            at=evaluate(logK,a,b,np.sqrt(b)*(anchor+V@s))
            change=objective_difference(at,V@d,a,b)
            directional=float((at['c']-b)@(V@d))
            max_support=max(max_support,directional-item['support'])
            violation=change-item['bound']; max_violation=max(max_violation,violation)
            worst_relative=max(worst_relative,violation/max(abs(item['bound']),1e-300))
            positive+=int(change>1e-24); checked+=1
    return dict(checked_steps=checked,positive_changes_above_1e_minus24=positive,
                max_increment_bound_violation=None if not checked else max_violation,
                max_relative_increment_bound_violation=None if not checked else worst_relative,
                max_directional_support_violation=None if not checked else max_support)
