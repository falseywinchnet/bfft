"""Transfer test: existing polynomial transport on entropic optimal transport.

A full Sinkhorn sweep is the map in gauge-fixed log column scalings. This is
not the separable rational control and no reciprocal closure is assumed.
"""
import numpy as np
from scipy.special import logsumexp
from .core import Linearization, identity, finite_flow, Anderson
from .discovery import discover


def gauge(y):
    return y - np.mean(y)


class Sinkhorn:
    def __init__(self, cost, p, q, epsilon):
        self.cost=np.asarray(cost,dtype=float)
        self.p=np.asarray(p,dtype=float);self.q=np.asarray(q,dtype=float)
        if epsilon<=0 or np.any(self.p<=0) or np.any(self.q<=0):
            raise ValueError('positive regularization and marginals required')
        if self.cost.shape!=(len(p),len(q)) or not np.isclose(self.p.sum(),1) or not np.isclose(self.q.sum(),1):
            raise ValueError('normalized matching marginals required')
        self.epsilon=epsilon;self.logK=-self.cost/epsilon
        self.K=np.exp(self.logK); self.KT=np.ascontiguousarray(self.K.T)
        self.logp=np.log(self.p);self.logq=np.log(self.q)
        self.initial=np.zeros(len(q));self.reset_counts()

    def reset_counts(self):
        self.calls=0;self.actions=0;self.checks=0;self.fallbacks=0

    def factors(self,y):
        if not np.all(np.isfinite(y)): raise FloatingPointError('nonfinite dual state')
        # Ordinary stabilized matrix scaling; avoid all-pairs exponentials
        # unless extreme dynamic range makes these matvecs unsafe.
        v=np.exp(y-np.max(y));kv=self.K@v
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            u=self.p/kv
        if np.all(v>0) and np.all(kv>0) and np.all(np.isfinite(u)) and np.max(u)>0:
            u=u/np.max(u);ktu=self.KT@u
            if np.all(ktu>0):
                return v,kv,u,ktu
        return None

    def log_step(self,y):
        a=self.logp-logsumexp(self.logK+y[None,:],axis=1)
        b=self.logq-logsumexp(self.logK+a[:,None],axis=0)
        return gauge(b)

    def step(self,y):
        self.calls+=1
        f=self.factors(y)
        if f is None:
            self.fallbacks+=1;return self.log_step(y)
        return gauge(self.logq-np.log(f[3]))

    def linearize(self,y):
        self.calls+=1;f=self.factors(y)
        if f is None:
            self.fallbacks+=1
            lr=logsumexp(self.logK+y[None,:],axis=1)
            a=self.logp-lr
            lc=logsumexp(self.logK+a[:,None],axis=0)
            P=np.exp(self.logK+y[None,:]-lr[:,None])
            Q=np.exp(self.logK+a[:,None]-lc[None,:]).T.copy()
            nxt=gauge(self.logq-lc)
            def action(h):
                self.actions+=1;return gauge(Q@(P@h))
        else:
            v,kv,u,ktu=f;nxt=gauge(self.logq-np.log(ktu))
            def action(h):
                self.actions+=1
                return gauge((self.KT@(u*((self.K@(v*h))/kv)))/ktu)
        return Linearization(nxt,action,identity)

    def error(self,y):
        """L1 column marginal error of the exactly row-normalized plan."""
        self.checks+=1
        f=self.factors(y)
        if f is None:
            self.fallbacks+=1
            a=self.logp-logsumexp(self.logK+y[None,:],axis=1)
            column=np.exp(y+logsumexp(self.logK+a[:,None],axis=0))
        else:
            v,kv,u,ktu=f
            column=v*ktu
            column=column/column.sum() # removes only common u normalization
        return float(np.sum(np.abs(column-self.q)))

    def plan(self,y):
        a=self.logp-logsumexp(self.logK+y[None,:],axis=1)
        return np.exp(self.logK+a[:,None]+y[None,:])


def solve(model,method,target=1e-8,budget=20000,check_every=8):
    """Same target and work-indexed checks; entire loop is timed by caller.

    Existing parameters: fixed depth4/horizon16; discovered defaults; rejection
    cooldown32 as in the previous complete Meyer study. No optimum is queried.
    """
    import time
    model.reset_counts();z=model.initial.copy();aa=Anderson(depth=4)
    accepted=rejected=settles=0;cooldown=0;next_check=0;trace=[];choices=[]
    start=time.perf_counter();error=model.error(z);status='budget'
    while model.calls+model.actions<budget:
        try:
            if method=='ordinary' or model.calls+model.actions<4 or cooldown:
                z=model.step(z);cooldown=max(0,cooldown-1)
            elif method=='anderson':
                z=gauge(aa.advance(z,model.step(z)))
            elif method=='fixed':
                candidate,_,_=finite_flow(model,z,depth=4,horizon=16,natural_metric=False)
                z=model.step(candidate);settles+=1
            elif method=='discovered':
                d=discover(model,z)
                if d.accepted:
                    z=model.step(d.candidate);settles+=1;accepted+=1
                    choices.append([d.depth,d.horizon])
                else:
                    z=d.candidate;rejected+=1;cooldown=32
            else: raise ValueError(method)
            work=model.calls+model.actions
            if work>=next_check:
                error=model.error(z);next_check=(work//check_every+1)*check_every
                trace.append({'work':work,'error':error,'seconds':time.perf_counter()-start})
                if error<=target: status='target';break
                if not np.isfinite(error):status='invalid';break
        except (FloatingPointError,np.linalg.LinAlgError,OverflowError):
            status='invalid';break
    seconds=time.perf_counter()-start
    return z,dict(method=method,status=status,seconds=seconds,calls=model.calls,
        actions=model.actions,checks=model.checks,fallbacks=model.fallbacks,
        work=model.calls+model.actions,error=error,accepted=accepted,rejected=rejected,
        settling_calls=settles,choices=choices,trace=trace)
