"""Strong-frame probe: carry response pairs with their numerical uncertainty.

No optimizer. Full ordinary reciprocal requests, with measured missing K actions.
Roundoff budgets use conservative floating-operation estimates, not intervals.
"""
import time
import numpy as np
from .primitive_transport import ResponseCarrier

class GuardedCarrier:
    def __init__(self,K,budget=32,transport=True,audit=False):
        self.K=K;self.budget=budget;self.transport=transport;self.anchor=None
        self.gamma=64*max(K.shape)*np.finfo(float).eps
        self.products=0;self.reuses=0;self.resets=0;self.dropped=0
        self.events=[];self.audit=audit;self.audit_max=0.;self.max_amplification=1.
        self.Q=np.empty((K.shape[1],0));self.Y=np.empty((K.shape[0],0));self.E=self.Y.copy()

    def rebase(self,x,out):
        n=len(x);q0=np.ones(n)/np.sqrt(n);y0=np.ones(len(out))/np.sqrt(n)
        basis=[q0];images=[y0];errors=[self.gamma*np.abs(y0)]
        before=self.Q.shape[1];dropped=0;maxamp=1.;old_error=0.
        if self.anchor is not None and self.transport:
            ins=(self.anchor/x)[:,None]*self.Q
            outs=(self.base/out)[:,None]*self.Y
            errs=(self.base/out)[:,None]*self.E
            for j in range(before-1,-1,-1):
                q=ins[:,j].copy();y=outs[:,j].copy()
                e=errs[:,j]+self.gamma*(np.abs(y)+np.max(np.abs(q)))
                norm=np.linalg.norm(q);q/=norm;y/=norm;e/=norm
                for _ in range(2):
                    B=np.column_stack(basis);Y=np.column_stack(images);E=np.column_stack(errors)
                    c=B.T@q
                    e+=E@np.abs(c)+self.gamma*(np.abs(y)+np.abs(Y)@np.abs(c)+np.max(np.abs(q))+np.max(np.abs(B)@np.abs(c)))
                    q-=B@c;y-=Y@c
                norm=np.linalg.norm(q);maxamp=max(maxamp,1/max(norm,1e-300))
                if norm<1e-7 or np.max(e)/max(norm,1e-300)>1e-9:
                    dropped+=1;continue
                basis.append(q/norm);images.append(y/norm);errors.append(e/norm)
                if len(basis)>=self.budget:break
        self.anchor=x.copy();self.base=out.copy()
        self.Q=np.column_stack(basis);self.Y=np.column_stack(images);self.E=np.column_stack(errors)
        self.dropped+=dropped;self.max_amplification=max(self.max_amplification,maxamp)
        event=dict(old_rank=before,new_rank=len(basis),discarded=dropped,max_amplification=maxamp,
                   retained_uncertainty=float(np.max(self.E)))
        if self.audit:
            P=self.K*x[None,:]/out[:,None]
            event['relation_error']=float(np.max(np.abs(P@self.Q-self.Y)))
        self.events.append(event)

    def enclosure(self,pred,eta,delta):
        if eta>=1:return None
        low=(pred-delta)/(1+eta);high=(pred+delta)/(1-eta)
        if np.any(low<=0) or np.any(pred<=0):return None
        rel=float(np.max(np.maximum(pred-low,high-pred)/low))
        bound=float(np.max(np.log(high/pred))-np.min(np.log(low/pred)))
        return rel,bound

    def apply(self,x,tolerance):
        if self.anchor is None:
            out=self.K@x;self.products+=1;self.rebase(x,out);return out,0.
        logratio=np.log(x)-np.log(self.anchor)
        if np.ptp(logratio)>8 or (not self.transport and self.Q.shape[1]>=self.budget):
            out=self.K@x;self.products+=1;self.rebase(x,out);self.resets+=1;return out,0.
        scale=float(logratio.max());v=np.exp(logratio-scale)
        c=self.Q.T@v;r=v-self.Q@c;c2=self.Q.T@r;c+=c2;r-=self.Q@c2
        pred=self.Y@c;eta=float(np.max(np.abs(r)/v))
        delta=self.E@np.abs(c)+self.gamma*(np.abs(self.Y)@np.abs(c)+np.max(np.abs(v)))
        check=self.enclosure(pred,eta,delta)
        if check is not None and check[0]<=tolerance:
            self.reuses+=1;out=self.base*np.exp(scale)*pred;bound=check[1]
        else:
            size=np.linalg.norm(r)
            if size<1e-13*np.linalg.norm(v):
                out=self.K@x;self.products+=1;return out,0.
            q=r/size;y=(self.K@(self.anchor*q))/self.base;self.products+=1
            e=np.full(len(y),self.gamma*np.max(np.abs(q)))
            if hasattr(self,'remember'):self.remember(self.anchor*q,self.base*y,self.base*e)
            assembled=pred+size*y
            error=delta+size*e
            check=self.enclosure(assembled,0.,error)
            if check is None or check[0]>tolerance:
                out=self.K@x;self.products+=1;self.rebase(x,out);self.resets+=1;return out,0.
            if self.Q.shape[1]>=self.budget:
                keep=[0]+list(range(2,self.Q.shape[1]))
                self.Q=self.Q[:,keep];self.Y=self.Y[:,keep];self.E=self.E[:,keep]
            self.Q=np.column_stack((self.Q,q));self.Y=np.column_stack((self.Y,y));self.E=np.column_stack((self.E,e))
            out=self.base*np.exp(scale)*assembled;bound=check[1]
        if self.audit:
            exact=self.K@x
            self.audit_max=max(self.audit_max,float(np.ptp(np.log(out/exact)))-bound)
        return out,bound


class SpectralCarrier(GuardedCarrier):
    def rebase(self,x,out):
        q0=np.ones(len(x))/np.sqrt(len(x));y0=np.ones(len(out))/np.sqrt(len(x))
        basis=[q0];images=[y0];errors=[self.gamma*np.abs(y0)]
        before=self.Q.shape[1];dropped=0;maxamp=1.
        if self.anchor is not None:
            U=(self.anchor/x)[:,None]*self.Q;Y=(self.base/out)[:,None]*self.Y
            E=(self.base/out)[:,None]*self.E+self.gamma*(np.abs(Y)+np.max(np.abs(U),axis=0)[None,:])
            scales=np.linalg.norm(U,axis=0);U/=scales;Y/=scales;E/=scales
            c=q0@U;U-=q0[:,None]*c;Y-=y0[:,None]*c
            E+=self.gamma*(np.abs(Y)+np.abs(c)[None,:]+np.max(np.abs(U),axis=0)[None,:])
            L,values,Vt=np.linalg.svd(U,full_matrices=False)
            for j,value in enumerate(values):
                maxamp=max(maxamp,1/max(value,1e-300))
                if value<1e-7:dropped+=1;continue
                coeff=Vt[j]/value;y=Y@coeff
                e=E@np.abs(coeff)+self.gamma*(np.abs(Y)@np.abs(coeff)+np.max(np.abs(L[:,j])))
                if np.max(e)>1e-9:dropped+=1;continue
                basis.append(L[:,j]);images.append(y);errors.append(e)
                if len(basis)>=self.budget:break
        self.anchor=x.copy();self.base=out.copy()
        self.Q=np.column_stack(basis);self.Y=np.column_stack(images);self.E=np.column_stack(errors)
        self.dropped+=dropped;self.max_amplification=max(self.max_amplification,maxamp)
        event=dict(old_rank=before,new_rank=len(basis),discarded=dropped,max_amplification=maxamp,
                   retained_uncertainty=float(np.max(self.E)))
        if self.audit:
            P=self.K*x[None,:]/out[:,None]
            event['relation_error']=float(np.max(np.abs(P@self.Q-self.Y)))
        self.events.append(event)

class AtomicCarrier(GuardedCarrier):
    """Never feed a transformed/orthogonalized image back into its next frame.
    Retain original measured physical input-output pairs and re-express them.
    """
    def __init__(self,*args,**kwargs):
        self.atoms=[]
        super().__init__(*args,**kwargs)

    def remember(self,x,y,e):
        scale=np.max(np.abs(x))
        self.atoms.append((x/scale,y/scale,e/scale))
        self.atoms=self.atoms[-self.budget:]

    def rebase(self,x,out):
        q0=np.ones(len(x))/np.sqrt(len(x));y0=np.ones(len(out))/np.sqrt(len(x))
        basis=[q0];images=[y0];errors=[self.gamma*np.abs(y0)]
        before=len(self.atoms);dropped=0;maxamp=1.
        if self.atoms:
            U=np.column_stack([v[0]/x for v in self.atoms])
            Y=np.column_stack([v[1]/out for v in self.atoms])
            E=np.column_stack([v[2]/out for v in self.atoms])
            E+=self.gamma*(np.abs(Y)+np.max(np.abs(U),axis=0)[None,:])
            scales=np.linalg.norm(U,axis=0);U/=scales;Y/=scales;E/=scales
            c=q0@U;U-=q0[:,None]*c;Y-=y0[:,None]*c
            E+=self.gamma*(np.abs(Y)+np.abs(c)[None,:]+np.max(np.abs(U),axis=0)[None,:])
            L,values,Vt=np.linalg.svd(U,full_matrices=False)
            for j,value in enumerate(values):
                maxamp=max(maxamp,1/max(value,1e-300))
                if value<1e-7:dropped+=1;continue
                coeff=Vt[j]/value;y=Y@coeff
                e=E@np.abs(coeff)+self.gamma*(np.abs(Y)@np.abs(coeff)+np.max(np.abs(L[:,j])))
                if np.max(e)>1e-9:dropped+=1;continue
                basis.append(L[:,j]);images.append(y);errors.append(e)
                if len(basis)>=self.budget:break
        self.anchor=x.copy();self.base=out.copy()
        self.Q=np.column_stack(basis);self.Y=np.column_stack(images);self.E=np.column_stack(errors)
        self.dropped+=dropped;self.max_amplification=max(self.max_amplification,maxamp)
        event=dict(old_rank=before,new_rank=len(basis),discarded=dropped,max_amplification=maxamp,
                   retained_uncertainty=float(np.max(self.E)))
        if self.audit:
            P=self.K*x[None,:]/out[:,None]
            event['relation_error']=float(np.max(np.abs(P@self.Q-self.Y)))
        self.events.append(event)
        self.remember(x,out,self.gamma*np.abs(out))

def run(K,a,b,passes=256,rank=32,tolerance=1e-8,method='guarded',audit=False):
    start=time.perf_counter()
    if method in ['guarded','reset','spectral','atomic']:
        cls=AtomicCarrier if method=='atomic' else (SpectralCarrier if method=='spectral' else GuardedCarrier)
        left=cls(K,rank,method!='reset',audit)
        right=cls(K.T,rank,method!='reset',audit)
    else:
        left=ResponseCarrier(K,rank,method=='legacy');right=ResponseCarrier(K.T,rank,method=='legacy')
    v=np.ones(len(b));bound=0.
    for _ in range(passes):
        if method=='ordinary':kv=K@v;left.products+=1;e1=0.
        else:
            result=left.apply(v,tolerance);kv,e1=result[:2]
        u=a/kv;u/=u.max()
        if method=='ordinary':ktu=K.T@u;right.products+=1;e2=0.
        else:
            result=right.apply(u,tolerance);ktu,e2=result[:2]
        v=b/ktu;v/=v.max();bound+=e1+e2
    result=dict(y=np.log(v),seconds=time.perf_counter()-start,products=left.products+right.products,
        reuses=left.reuses+right.reuses,resets=left.resets+right.resets,bound=bound)
    if method in ['guarded','reset','spectral','atomic']:
        result.update(dropped=left.dropped+right.dropped,events=left.events+right.events,
                      audit_enabled=audit,audit_violation=max(left.audit_max,right.audit_max) if audit else None)
    return result
