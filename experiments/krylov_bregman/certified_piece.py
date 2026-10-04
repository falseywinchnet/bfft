"""Certified discovery on convex Huber mirror descent.

Whitened dual coordinates t=G^(-1/2)y=G^(1/2)x turn a
quadratic mirror into Euclidean norm without changing the primal objective.
B=A G^(-1/2). F(t)=t-alpha B^T clip(Bt-b,-delta,delta).
The global nonexpansive bound holds for alpha <= 2/||B||^2.
The current affine piece is checked on EVERY predicted input via cached BQ.
This is a class-specific certificate, not a certificate for Meyer disks.
"""
from dataclasses import dataclass
import numpy as np
from .core import Linearization
from .discovery import arnoldi_stream


class HuberMirror:
    def __init__(self,B,b,delta=1.,alpha=None,mirror_diagonal=None):
        self.B=np.asarray(B,dtype=float);self.b=np.asarray(b,dtype=float)
        self.delta=float(delta)
        L=float(np.linalg.norm(self.B,2)**2)
        self.alpha=.9/L if alpha is None else float(alpha)
        if not 0 < self.alpha <= 2/L:raise ValueError('nonexpansive step required')
        self.initial=np.zeros(self.B.shape[1])
        self.g=np.ones(len(self.initial)) if mirror_diagonal is None else np.asarray(mirror_diagonal)
    def primal(self,t):return t/np.sqrt(self.g)
    def step(self,t):
        return t-self.alpha*(self.B.T@np.clip(self.B@t-self.b,-self.delta,self.delta))
    def gap(self,t):
        r=np.abs(self.B@t-self.b)
        # The source-controlled noiseless study has optimum exactly zero.
        return float(np.where(r<=self.delta,.5*r*r,self.delta*(r-.5*self.delta)).sum())
    def linearize(self,t):
        residual=self.B@t-self.b; mask=np.abs(residual)<self.delta
        return Linearization(t-self.alpha*(self.B.T@np.clip(residual,-self.delta,self.delta)),
                             lambda v:v-self.alpha*(self.B.T@(mask*(self.B@v))))


@dataclass
class Certified:
    state:np.ndarray
    horizon:int
    depth:int
    calls:int
    actions:int
    accepted:bool
    error_bound:float
    relative_bound:float
    branch_checks:int
    reason:str


def reduced_path(H,source,count):
    """All c_j of c_(j+1)=source+H c_j by block doubling, O(k^2*count).

    No eigenvalue decomposition or inverse; valid at the unit eigenvalue.
    """
    C=np.zeros((len(source),1));power=H.copy();offset=source.copy()
    while C.shape[1]<count:
        C=np.column_stack((C,offset[:,None]+power@C))
        offset=offset+power@offset;power=power@power
    return C[:,:count]


def discover_piece(model,t,depths=(2,4,8),maximum_horizon=256,tolerance=.02,
                   minimum_work_gain=1.5):
    residual=model.B@t-model.b;mask=np.abs(residual)<model.delta
    next_state=t-model.alpha*(model.B.T@np.clip(residual,-model.delta,model.delta))
    features=[]
    def action(q):
        Bq=model.B@q;features.append(Bq)
        return q-model.alpha*(model.B.T@(mask*Bq))
    checks=0;actions=0
    for basis,closed in arnoldi_stream(action,next_state-t,max(depths)):
        actions=basis.actions
        if actions not in depths and not closed:continue
        BQ=np.column_stack(features)
        # Exact constant-drift region: one-dimensional affine closure.
        # Roundoff-sized compression is still included in the bound below.
        if not mask.any() and actions==1:
            drift=BQ[:,0]*basis.beta
            toward=((residual>model.delta)&(drift<0))|((residual<-model.delta)&(drift>0))
            times=(np.sign(residual[toward])*model.delta-residual[toward])/drift[toward]
            first=float(np.min(times)) if len(times) else float('inf')
            m=maximum_horizon if first>=maximum_horizon else max(1,int(np.floor(max(0.,first)))+1)
            # Check last INPUT, not output; the output may cross the boundary.
            last=residual+(m-1)*drift
            valid=np.all(np.where(residual>=model.delta,last>=model.delta,last<=-model.delta))
            if not valid:m=max(1,m-1)
            # With exact identity Jacobian, the physical drift is exact;
            # compute m*r directly, avoiding accumulated projected roundoff.
            if m>=minimum_work_gain*(1+actions):
                return Certified(t+m*(next_state-t),m,actions,1,actions,True,0.,0.,1,'certified drift event')
            break
        source=np.zeros(actions);source[0]=basis.beta
        coordinates=reduced_path(basis.H,source,maximum_horizon+1)
        gram=basis.remainder.T@basis.remainder
        defects=np.sqrt(np.maximum(0.,np.sum(coordinates[:,:-1]*(gram@coordinates[:,:-1]),axis=0)))
        bounds=np.cumsum(defects)
        relative=bounds/np.maximum(np.linalg.norm(coordinates[:,1:],axis=0),1e-30)
        positive=residual>=model.delta;negative=residual<=-model.delta
        stop=maximum_horizon;lo=0;block=16
        while lo<maximum_horizon:
            hi=min(maximum_horizon,lo+block)
            margins=residual[:,None]+BQ@coordinates[:,lo:hi];checks+=hi-lo
            valid=(np.all(margins[positive]>=model.delta,axis=0) &
                   np.all(margins[negative]<=-model.delta,axis=0) &
                   np.all(np.abs(margins[mask])<=model.delta,axis=0))
            bad=np.flatnonzero(~valid)
            if len(bad):stop=lo+int(bad[0]);break
            lo=hi;block*=2
        # An invalid input j permits only horizons <=j.
        ms=np.arange(1,stop+1)
        admitted=ms[(ms>=minimum_work_gain*(1+actions)) & (relative[:stop]<=tolerance)]
        best=None
        if len(admitted):
            m=int(admitted[-1]);best=(m,coordinates[:,m],float(bounds[m-1]),float(relative[m-1]))
        if best is not None:
            m,c,bound,relative=best
            return Certified(t+basis.Q@c,m,actions,1,actions,True,bound,relative,checks,'certified affine piece')
        if closed:break
    return Certified(next_state,1,actions,1,actions,False,0.,0.,checks,'no profitable certified horizon')
