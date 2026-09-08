"""Cost-only simple CV controls, not candidates in the acquisition study."""
import ctypes,time,json
from pathlib import Path
import numpy as np
from .baselines import ConventionalFilter
from .study import save

class SimpleCV:
    def __init__(self,y,sigma=.35,native=False):
        self.m=np.zeros((2,3));self.m[0]=y;self.p=np.diag([sigma*sigma,8.])
        self.last_time=0.;self.sigma=sigma;self.native=native
        if native:
            self.lib=ctypes.CDLL(str(Path(__file__).parent/'native/libcv_control.so'))
            a=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');d=ctypes.c_double;i=ctypes.c_int
            self.lib.cv_update.argtypes=[a,a,d,a,d];self.lib.cv_update.restype=None
            self.lib.cv_query.argtypes=[a,a,a,i,a,a];self.lib.cv_query.restype=None
    def update(self,t,y):
        y=np.asarray(y,float)
        if not self.last_time<t or y.shape!=(3,) or not np.isfinite(y).all():raise ValueError('Invalid observation')
        dt=t-self.last_time
        if self.native:self.lib.cv_update(self.m,self.p,dt,y,self.sigma**2)
        else:
            f=np.array([[1.,dt],[0,1.]])
            self.m=f @ self.m
            self.p=f @ self.p @ f.T+np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
            b=self.p[:,0].copy();s=b[0]+self.sigma**2
            self.m+=(b/s)[:,None]*(y-self.m[0]);self.p-=np.outer(b,b)/s
        self.last_time=t
    def forecast(self,queries):
        dt=np.ascontiguousarray(queries,dtype=float)-self.last_time
        if np.any(~np.isfinite(dt)) or np.any(dt<0):raise ValueError('Require future query')
        mean=np.empty((len(dt),3));cov=np.empty((len(dt),3,3))
        if self.native:self.lib.cv_query(self.m,self.p,dt,len(dt),mean,cov)
        else:
            mean=self.m[0]+dt[:,None]*self.m[1]
            cov=(self.p[0,0]+2*dt*self.p[0,1]+dt**2*self.p[1,1]+dt**3/3)[:,None,None]*np.eye(3)
        return dict(mean=mean,covariance=cov)

def run(out):
    rng=np.random.default_rng(60021);ts=np.r_[0,np.cumsum(rng.uniform(.04,.06,160))];ys=rng.normal(size=(161,3))
    a,b=SimpleCV(ys[0]),SimpleCV(ys[0],native=True)
    for t,y in zip(ts[1:],ys[1:]):
        a.update(t,y);b.update(t,y)
        for key,v in a.forecast([t,t+2]).items():np.testing.assert_allclose(v,b.forecast([t,t+2])[key],atol=2e-12)
    results=[]
    for native in (False,True):
        samples=[]
        for repeat in range(32):
            m=SimpleCV(ys[0],native=native);costs=np.zeros(2)
            for t,y in zip(ts[1:],ys[1:]):
                start=time.perf_counter_ns();m.update(t,y);costs[0]+=time.perf_counter_ns()-start
                start=time.perf_counter_ns();m.forecast([t,t+2]);costs[1]+=time.perf_counter_ns()-start
            if repeat:samples.append(costs/160/1000)
        results.append(dict(backend='simple_cv_native' if native else 'simple_cv_python',median_us=np.median(samples,axis=0),p10_us=np.percentile(samples,10,axis=0),p90_us=np.percentile(samples,90,axis=0)))
    save(out,dict(scope='Cost control only; 2-state CV per axis, shared isotropic covariance, fixed q=1, no length mixture. No accuracy comparison.',results=results,equivalence='160 irregular updates and 320 queries at 2e-12 absolute tolerance'))
    print(json.dumps(results,default=lambda x:x.tolist(),indent=2))

if __name__=='__main__':
    import sys
    run(sys.argv[1])
