"""Exact four-state realization of the finite midpoint-current prior.

Current/future marginal queries use the sufficient filtering state. Historical
smoothing and full coefficient exports lazily replay the retained dense oracle.
This changes elimination order, not the prior or observation likelihood.
"""
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp
from .relational_reference import LENGTHS


@lru_cache(maxsize=32)
def transitions(lengths, end, cells):
    h=end/cells
    a=np.sqrt(3)*h/np.asarray(lengths)
    A=np.exp(-a)[:,None,None]*np.array([[1+a,a],[-a,1-a]]).transpose(2,0,1)
    F=np.broadcast_to(np.eye(4),(len(lengths),4,4)).copy()
    F[:,0,1:3]=h
    F[:,2:,2:]=A
    Q=np.zeros_like(F)
    Q[:,2:,2:]=4*(np.eye(2)-A @ A.transpose(0,2,1))
    powers=[np.broadcast_to(np.eye(4),F.shape).copy()]; noises=[np.zeros_like(F)]
    for _ in range(cells):
        powers.append(F @ powers[-1])
        noises.append(F @ noises[-1] @ F.transpose(0,2,1)+Q)
    powers,noises=np.array(powers),np.array(noises)
    powers.flags.writeable=noises.flags.writeable=False
    return powers,noises


class RelationalKalman:
    def __init__(self,observation,sigma=.35,end=8.,cells=64,lengths=LENGTHS,
                 coordinates='zak',sever_at=None):
        self.anchor=np.asarray(observation,float).copy()
        self.lengths=tuple(lengths)
        if (self.anchor.shape!=(3,) or not np.isfinite(self.anchor).all() or
            not np.isfinite(sigma) or sigma<=0 or not np.isfinite(end) or end<=0 or
            not isinstance(cells,int) or cells<=0 or not self.lengths or
            np.any(~np.isfinite(self.lengths)) or min(self.lengths)<=0 or
            coordinates not in ('zak','current')):
            raise ValueError('Invalid observation, interval, noise, or prior')
        if coordinates=='zak' and cells%8:
            raise ValueError('Lattice must factor coefficient dimension')
        self.sigma,self.end,self.cells=sigma,end,cells
        self.coordinates,self.sever_at=coordinates,sever_at
        self.h=end/cells;self.cell=0;self.last_time=0.
        self._m=np.zeros((len(self.lengths),4,3))
        self._p=np.broadcast_to(np.diag([sigma*sigma,4.,4.,4.]),(len(self.lengths),4,4)).copy()
        self.logs=np.full(len(self.lengths),-np.log(len(self.lengths)))
        self._powers,self._noises=transitions(self.lengths,end,cells)
        self._split=cells if sever_at is None else int(np.count_nonzero((np.arange(cells)+.5)*self.h<sever_at))
        self._history=[];self._reference=None
        self._refresh_weights()

    def _refresh_weights(self):
        self._evidence=float(logsumexp(self.logs))
        self._weights=np.exp(self.logs-self._evidence)

    def _advance(self,m,p,start,target):
        def jump(m,p,n):
            f=self._powers[n]
            return f @ m,f @ p @ f.transpose(0,2,1)+self._noises[n]
        if start<self._split<=target:
            m,p=jump(m,p,self._split-1-start)
            f=np.zeros((4,4));f[0]=[1,self.h,self.h,0]
            m=f @ m;p=f @ p @ f.T+np.diag([0,4.,4.,4.])
            start=self._split
        return jump(m,p,target-start)

    def _at(self,time):
        k=min(int(time/self.h),self.cells-1)
        m,p=self._advance(self._m,self._p,self.cell,k) if k!=self.cell else (self._m,self._p)
        tau=time-k*self.h
        b=np.array([1.,tau,tau,0.])
        return m,p,b,k

    def update(self,time,observation,beta_steps=1):
        y=np.asarray(observation,float)
        if not self.last_time<time<=self.end or y.shape!=(3,) or not np.isfinite(y).all():
            raise ValueError('Require new increasing observation within interval')
        if not isinstance(beta_steps,int) or beta_steps<1:
            raise ValueError('Positive integer refinement count required')
        m,p,b,k=self._at(time)
        pb=p @ b;s=pb @ b+self.sigma**2
        innovation=y-self.anchor-b @ m
        self.logs+=-.5*(3*np.log(2*np.pi*s)+np.sum(innovation**2,axis=1)/s)
        self._m=m+(pb/s[:,None])[:,:,None]*innovation[:,None,:]
        p=p-pb[:,:,None]*pb[:,None,:]/s[:,None,None]
        self._p=(p+p.transpose(0,2,1))*.5
        self.cell=k;self.last_time=float(time)
        self._history.append((float(time),y.copy(),beta_steps));self._reference=None
        self._refresh_weights()

    def _dense(self):
        if self._reference is None:
            from .relational_reference import RelationalKalman as Reference
            r=Reference(self.anchor,self.sigma,self.end,self.cells,self.lengths,self.coordinates,self.sever_at)
            for t,y,beta in self._history:r.update(t,y,beta)
            self._reference=r
        return self._reference

    @property
    def means(self):return self._dense().means
    @property
    def covs(self):return self._dense().covs
    @property
    def u(self):return self._dense().u

    def position(self,time):
        if not 0<=time<=self.end:raise ValueError('Query outside represented interval')
        if time<self.last_time:return self._dense().forecast([time])['mean'][0]
        m,_,b,_=self._at(time)
        return self._weights @ (b @ m)+self.anchor

    def forecast(self,queries):
        q=np.asarray(queries,float)
        if q.ndim!=1 or np.any(~np.isfinite(q)) or np.any(q<0) or np.any(q>self.end):
            raise ValueError('Query outside represented interval')
        if np.any(q<self.last_time):return self._dense().forecast(q)
        means=np.empty((len(self.lengths),len(q),3));variances=np.empty((len(self.lengths),len(q)))
        for j,t in enumerate(q):
            m,p,b,_=self._at(float(t))
            means[:,j]=b @ m+self.anchor;variances[:,j]=(p @ b) @ b
        mean=np.einsum('k,ktd->td',self._weights,means);delta=means-mean
        covariance=(np.einsum('k,kt,ij->tij',self._weights,variances,np.eye(3))+
                    np.einsum('k,kti,ktj->tij',self._weights,delta,delta))
        return dict(mean=mean,covariance=covariance,weights=self._weights.copy(),
                    component_mean=means,component_variance=variances,log_evidence=self._evidence)
