"""Compiled double-precision implementation of the same four-state algebra."""
import ctypes
from pathlib import Path
import numpy as np
from .relational_markov import RelationalKalman as PythonMarkov

_lib=None

def library():
    global _lib
    if _lib is None:
        lib=ctypes.CDLL(str(Path(__file__).parent/'native/libmarkov.so'))
        a=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        i=ctypes.c_int;d=ctypes.c_double
        lib.markov_update.argtypes=[a,a,a,a,a,a,i,i,i,i,d,d,a,d]
        lib.markov_update.restype=d
        lib.markov_query.argtypes=[a,a,a,a,a,i,i,i,i,d,a,i,a,a,a,a,a]
        lib.markov_query.restype=None
        _lib=lib
    return _lib

class RelationalKalman(PythonMarkov):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._lib=library();self._position_cache=None

    def update(self,time,observation,beta_steps=1):
        y=np.asarray(observation,float)
        if not self.last_time<time<=self.end or y.shape!=(3,) or not np.isfinite(y).all():
            raise ValueError('Require new increasing observation within interval')
        if not isinstance(beta_steps,int) or beta_steps<1:
            raise ValueError('Positive integer refinement count required')
        k=min(int(time/self.h),self.cells-1)
        self._evidence=self._lib.markov_update(self._m,self._p,self.logs,self._weights,
            self._powers,self._noises,len(self.lengths),self.cell,k,self._split,
            self.h,time-k*self.h,y-self.anchor,self.sigma**2)
        self.cell=k;self.last_time=float(time)
        self._history.append((float(time),y.copy(),beta_steps));self._reference=None;self._position_cache=None

    def forecast(self,queries):
        q=np.ascontiguousarray(queries,dtype=float)
        if q.ndim!=1 or np.any(~np.isfinite(q)) or np.any(q<0) or np.any(q>self.end):
            raise ValueError('Query outside represented interval')
        if np.any(q<self.last_time):return self._dense().forecast(q)
        K,T=len(self.lengths),len(q)
        means=np.empty((K,T,3));variances=np.empty((K,T));mean=np.empty((T,3));cov=np.empty((T,3,3))
        self._lib.markov_query(self._m,self._p,self._weights,self._powers,self._noises,K,
            self.cell,self._split,self.cells,self.h,q,T,self.anchor,means,variances,mean,cov)
        return dict(mean=mean,covariance=cov,weights=self._weights.copy(),component_mean=means,
                    component_variance=variances,log_evidence=self._evidence)
