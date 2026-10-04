"""Finite coupled increment transport learned from ordinary full-state history.

No Jacobian, objective optimum, or rational scalar chart is supplied. This is
finite-horizon DMD/observable recurrence identification, not a new name for a
fixed-point inverse. Live probes screen, but do not globally certify, reuse.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class IncrementChart:
    anchor:np.ndarray
    Q:np.ndarray
    T:np.ndarray
    last:np.ndarray
    rank:int
    fit_defect:float
    clipping:float

    @classmethod
    def fit(cls,states,maximum_rank=4,contractive=False):
        D=np.diff(np.column_stack(states),axis=1);X=D[:,:-1];Y=D[:,1:]
        U,s,Vt=np.linalg.svd(X,full_matrices=False)
        rank=min(maximum_rank,int(np.sum(s>max(s[0]*1e-10,1e-30))))
        if rank==0:
            return cls(states[-1].copy(),U[:,:0],np.empty((0,0)),np.empty(0),0,0.,0.)
        Q=U[:,:rank]
        T=(Q.T@Y)@(Vt[:rank].T/s[:rank])
        original=T.copy()
        if contractive:
            left,values,right=np.linalg.svd(T,full_matrices=False)
            T=(left*np.minimum(values,1.))@right
        defect=np.linalg.norm(Y-Q@T@(Q.T@X))/max(np.linalg.norm(Y),1e-30)
        return cls(states[-1].copy(),Q,T,Q.T@D[:,-1],rank,float(defect),float(np.linalg.norm(T-original)))

    def candidate(self,horizon):
        c=np.zeros(self.rank);d=self.last.copy()
        for _ in range(horizon):
            d=self.T@d;c+=d
        return self.anchor+self.Q@c

    def propose(self,step,horizons=(64,32,16),tolerance=.05):
        cache={};trials=[]
        for m in horizons:
            candidate=self.candidate(m);scale=max(np.linalg.norm(candidate-self.anchor),1e-30)
            score=0.
            for j in sorted(set([0,m//2,m-1])):
                if j not in cache:
                    here=self.candidate(j);nxt=self.candidate(j+1)
                    cache[j]=float(np.linalg.norm(step(here)-nxt))
                score=max(score,m*cache[j]/scale)
                if not np.isfinite(score) or score>tolerance:break
            passed=bool(np.isfinite(score) and score<=tolerance)
            trials.append(dict(horizon=m,score=score,accepted=passed))
            if passed:return candidate,m,len(cache),trials
        return self.anchor,0,len(cache),trials
