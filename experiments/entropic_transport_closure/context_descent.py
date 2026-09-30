"""A measured, finite higher-state descent probe.

The lifted conditional state is exact. Its inexpensive response-memory control
is limited-memory BFGS, explicitly a known baseline, not a new closure theorem.
No optimum, spectrum, anchor fit or future trajectory is used by the algorithm.
"""
import time
import numpy as np
from scipy.special import logsumexp


def evaluate(logK, a, b, z):
    root=np.sqrt(b); z=z-root*(root@z); y=z/root
    logits=logK+y[None,:]
    normalizer=logsumexp(logits,axis=1)
    P=np.exp(logits-normalizer[:,None]); c=a@P
    g=(c-b)/root; g-=root*(root@g)
    return dict(z=z,y=y,P=P,c=c,g=g,
                value=float(a@normalizer-b@y),
                residual=float(np.max(np.abs(c/b-1))))



def evaluate_factored(K,logK,a,b,z):
    """Implicit conditional state P=diag(1/Kv) K diag(v).
    Materialize no dynamic n-by-n table. Fall back to log arithmetic when
    ordinary scaling leaves the safe finite range.
    """
    root=np.sqrt(b); z=z-root*(root@z); y=z/root
    v=np.exp(y-y.max()); kv=K@v
    if np.any(kv<1e-280) or np.any(v<1e-280):
        return evaluate(logK,a,b,z)
    u=a/kv; c=v*(K.T@u)
    if np.any(c<=0) or not np.all(np.isfinite(c)):
        return evaluate(logK,a,b,z)
    g=(c-b)/root; g-=root*(root@g)
    return dict(z=z,y=y,K=K,v=v,kv=kv,c=c,g=g,
                value=float(a@np.log(kv)+y.max()-b@y),
                residual=float(np.max(np.abs(c/b-1))))


def conditional_action(state,v):
    if 'P' in state: return state['P']@v
    return (state['K']@(state['v']*v))/state['kv']


def objective_difference(state, displacement, a, b):
    """Exact exponential retilting identity; stable for small displacements."""
    h=np.asarray(displacement); h=h-(float(h.max())+float(h.min()))/2
    if np.ptp(h)<1:
        rows=np.log1p(conditional_action(state,np.expm1(h)))
    else:
        # P can underflow, so large steps are capped by the caller at osc=10.
        # Positive probabilities lost below float range cannot reappear at
        # material precision under that cap.
        shift=float(h.max()); rows=np.log(conditional_action(state,np.exp(h-shift)))+shift
    return float(a@rows-b@h)


def inverse_action(v, pairs):
    """Standard two-loop inverse BFGS, positive curvature pairs only."""
    q=v.copy(); alpha=[]
    for s,t in reversed(pairs):
        al=float(s@q/(s@t)); alpha.append(al); q-=al*t
    scale=float(pairs[-1][0]@pairs[-1][1]/(pairs[-1][1]@pairs[-1][1])) if pairs else 1.
    q*=scale
    for (s,t),al in zip(pairs,reversed(alpha)):
        q+=s*(al-float(t@q/(s@t)))
    return q


def solve(logK,a,b,method='adaptive',memory=8,tolerance=1e-9,max_steps=20000,
          defect_threshold=.1,y0=None,engine="factored"):
    if method not in ('ordinary','gradient','scalar','lbfgs','adaptive','frozen'):
        raise ValueError(method)
    start=time.perf_counter(); root=np.sqrt(b)
    K=np.exp(logK) if engine=='factored' else None
    def ev(z):
        return evaluate_factored(K,logK,a,b,z) if engine=='factored' else evaluate(logK,a,b,z)
    state=ev(np.zeros(len(b)) if y0 is None else root*y0)
    pairs=[]; history=[]; evaluations=1; trials=0; admitted=0; omitted=0
    scalar=1.; failure=None
    for step in range(max_steps):
        if state['residual']<=tolerance: break
        g=state['g']
        if method=='ordinary':
            direction=root*np.log(b/state['c'])
        elif method=='gradient': direction=-g
        elif method=='scalar': direction=-scalar*g
        else: direction=-inverse_action(g,pairs)
        direction-=root*(root@direction)
        slope=float(g@direction)
        if slope>=0 or not np.isfinite(slope):
            failure='non-descent direction'; break
        alpha=min(1.,10/max(float(np.ptp(direction/root)),1e-300)) if method!='ordinary' else 1.
        for backtrack in range(60):
            h=alpha*direction/root
            diff=objective_difference(state,h,a,b); trials+=1
            if diff<=1e-4*alpha*slope: break
            alpha*=.5
        else:
            failure='line search exhausted'; break
        new=ev(state['z']+alpha*direction); evaluations+=1
        s=new['z']-state['z']; t=new['g']-g
        defect=float(np.linalg.norm(inverse_action(t,pairs)-s)/max(np.linalg.norm(s),1e-300))
        curvature=float(s@t)
        valid=curvature>1e-12*np.linalg.norm(s)*np.linalg.norm(t) and curvature>0
        added=False
        if valid:
            scalar=float(curvature/(t@t))
            use=method=='lbfgs' or (method=='adaptive' and (not pairs or defect>defect_threshold)) or (method=='frozen' and len(pairs)<memory)
            if use:
                pairs.append((s.copy(),t.copy())); pairs=pairs[-memory:]; admitted+=1; added=True
            else: omitted+=1
        history.append(dict(residual=new['residual'],objective_difference=diff,
                            directional_derivative=slope,alpha=alpha,backtracks=backtrack,
                            memory=len(pairs),secant_defect=defect,pair_added=added))
        state=new
    certificate=evaluate(logK,a,b,state['z']); evaluations+=1
    return dict(method=method,engine=engine,converged=certificate['residual']<=tolerance,residual=certificate['residual'],
                seconds=time.perf_counter()-start,evaluations=evaluations,trials=trials,
                accepted_steps=len(history),pairs_admitted=admitted,pairs_omitted=omitted,
                memory_limit=memory,failure=failure,history=history,y=state['y'].tolist())


def ordinary_blas(logK,a,b,tolerance=1e-9,max_steps=20000):
    """Lean ordinary comparator, including kernel materialization and final
    independent log-domain certificate. Applicable to these finite-range cases.
    Returns an explicit failure if positive scaling denominators underflow.
    """
    start=time.perf_counter(); K=np.exp(logK); y=np.zeros(len(b)); failure=None
    for step in range(max_steps+1):
        v=np.exp(y-y.max()); kv=K@v
        if np.any(kv<=0): failure='row denominator underflow'; break
        u=a/kv; kt=K.T@u; c=v*kt
        if np.any(c<=0) or not np.all(np.isfinite(c)):
            failure='scaling range failure'; break
        residual=float(np.max(np.abs(c/b-1)))
        if residual<=tolerance or step==max_steps: break
        y=np.log(b)-np.log(kt); y-=y.mean()
    final=evaluate(logK,a,b,np.sqrt(b)*y)
    return dict(method='ordinary_blas',converged=final['residual']<=tolerance,
                residual=final['residual'],seconds=time.perf_counter()-start,
                evaluations=step+2,accepted_steps=step,failure=failure,
                history=[],y=final['y'].tolist())
