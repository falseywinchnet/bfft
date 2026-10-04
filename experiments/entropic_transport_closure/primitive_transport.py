"""Carry fixed kernel responses; rebuild the two nonlinear normalizations.

This is inexact ordinary Sinkhorn with explicit per-query enclosure, not a
potential optimizer. Neither a Jacobian nor the evolving potential is fitted.
All coordinate vectors remain full dimensional. Exact-arithmetic bounds are
evaluated in float64, without an outward-rounding claim.
"""
import time
import numpy as np

class ResponseCarrier:
    def __init__(self,K,budget=64,transport=False):
        self.K=K;self.budget=budget;self.anchor=None;self.transport=transport
        self.Q=np.empty((K.shape[1],0));self.Y=np.empty((K.shape[0],0))
        self.products=0;self.reuses=0;self.resets=0

    def rebase(self,x,out):
        scale=np.sqrt(len(x))
        basis=[np.ones(len(x))/scale];responses=[np.ones(len(out))/scale]
        if self.anchor is not None and self.transport:
            # A fixed physical K-response survives diagonal frame changes.
            # Apply each orthogonalization operation to input AND response.
            inputs=(self.anchor/x)[:,None]*self.Q
            outputs=(self.base/out)[:,None]*self.Y
            for j in range(inputs.shape[1]-1,-1,-1):
                q=inputs[:,j].copy();y=outputs[:,j].copy()
                norm=np.linalg.norm(q);q/=norm;y/=norm
                for _ in range(2):
                    B=np.column_stack(basis);Y=np.column_stack(responses)
                    c=B.T@q;q-=B@c;y-=Y@c
                norm=np.linalg.norm(q)
                if norm<1e-7:continue
                basis.append(q/norm);responses.append(y/norm)
                if len(basis)>=self.budget:break
        self.anchor=x.copy();self.base=out.copy()
        self.Q=np.column_stack(basis);self.Y=np.column_stack(responses)

    def apply(self,x,relative_budget):
        if self.anchor is None:
            out=self.K@x;self.products+=1;self.rebase(x,out)
            return out,0.,True
        logratio=np.log(x)-np.log(self.anchor)
        if np.ptp(logratio)>8 or (not self.transport and self.Q.shape[1]>=self.budget):
            out=self.K@x;self.products+=1;self.rebase(x,out);self.resets+=1
            return out,0.,True
        scale=float(np.max(logratio));v=np.exp(logratio-scale)
        coefficients=self.Q.T@v;r=v-self.Q@coefficients
        correction=self.Q.T@r;coefficients+=correction;r-=self.Q@correction
        predicted=self.Y@coefficients
        # Positivity transports relative input error through K exactly:
        # |v-vhat| <= eta*v => |P v-P vhat| <= eta*P v.
        eta=float(np.max(np.abs(r)/v))
        if eta<=relative_budget and np.all(predicted>0):
            self.reuses+=1
            return self.base*np.exp(scale)*predicted,float(np.log1p(eta)-np.log1p(-eta)),False
        size=np.linalg.norm(r)
        if size<=1e-13*np.linalg.norm(v):
            out=self.K@x;self.products+=1
            return out,0.,True
        # Measure the missing primitive response directly. In this balanced
        # frame v is in [exp(-8),1], so its positive output has a known floor.
        # Differencing two almost equal full outputs would destroy this image.
        q=r/size;y=(self.K@(self.anchor*q))/self.base;self.products+=1
        if self.Q.shape[1]>=self.budget:
            keep=[0]+list(range(2,self.Q.shape[1]))
            self.Q=self.Q[:,keep];self.Y=self.Y[:,keep]
        self.Q=np.column_stack((self.Q,q));self.Y=np.column_stack((self.Y,y))
        out=self.base*np.exp(scale)*(predicted+size*y)
        if np.any(out<=0) or not np.all(np.isfinite(out)):
            out=self.K@x;self.products+=1;self.rebase(x,out);self.resets+=1
        return out,0.,True


def finite_run(K,a,b,passes=256,step_budget=1e-8,rank=64,carried=True,transport=False):
    start=time.perf_counter();left=ResponseCarrier(K,rank,transport);right=ResponseCarrier(K.T,rank,transport)
    v=np.ones(len(b));bound=0.;trace=[]
    for k in range(passes):
        if carried: kv,e1,p1=left.apply(v,step_budget)
        else: kv=K@v;e1=0.;left.products+=1
        u=a/kv;u/=np.max(u)
        if carried: ktu,e2,p2=right.apply(u,step_budget)
        else: ktu=K.T@u;e2=0.;right.products+=1
        v=b/ktu;v/=np.max(v)
        bound+=e1+e2
        trace.append(dict(pass_index=k+1,trajectory_bound=bound,
            products=left.products+right.products,reuses=left.reuses+right.reuses))
    seconds=time.perf_counter()-start
    return dict(y=np.log(v),seconds=seconds,products=left.products+right.products,
        reuses=left.reuses+right.reuses,resets=left.resets+right.resets,
        ranks=[left.Q.shape[1],right.Q.shape[1]],trajectory_bound=bound,trace=trace)
