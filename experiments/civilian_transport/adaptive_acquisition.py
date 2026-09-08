"""Semiperiodic acquisition with a causal boundary-triggered rate increase.

Only the acquisition cadence changes. The boundary is not a physical barrier or
an input to the motion prior. The sampler sees an estimated side of the plane,
never the clean trajectory or its true crossing time.
"""
import numpy as np
from .data import make_case


class BoundarySampler:
    def __init__(self, low_rate=2., high_rate=16., ramp_seconds=1., jitter=.2,
                 seed=0, constant_rate=None):
        if low_rate<=0 or high_rate<low_rate or ramp_seconds<=0 or not 0<=jitter<1:
            raise ValueError('Invalid cadence or ramp')
        if constant_rate is not None and constant_rate<=0:
            raise ValueError('Invalid constant cadence')
        self.low_rate=float(low_rate);self.high_rate=float(high_rate)
        self.ramp_seconds=float(ramp_seconds);self.jitter=float(jitter)
        self.constant_rate=constant_rate
        self.rng=np.random.default_rng(seed)
        self.trigger_time=None

    def observe_side(self,time,signed_estimated_position):
        if self.trigger_time is None and signed_estimated_position>=0:
            self.trigger_time=float(time)

    def rate(self,time):
        if self.constant_rate is not None:
            return float(self.constant_rate)
        elapsed=0. if self.trigger_time is None else max(0.,time-self.trigger_time)
        return self.low_rate+(self.high_rate-self.low_rate)*(-np.expm1(-elapsed/self.ramp_seconds))

    def integrated_rate(self,start,end):
        if end<start:
            raise ValueError('Backward scheduling interval')
        if self.constant_rate is not None:
            return float(self.constant_rate)*(end-start)
        value=self.low_rate*(end-start)
        if self.trigger_time is None or end<=self.trigger_time:
            return value
        a=max(start-self.trigger_time,0.);b=end-self.trigger_time
        # Integral of 1-exp(-elapsed/tau), including a possible crossing of trigger.
        value+=(self.high_rate-self.low_rate)*(b-a+self.ramp_seconds*(np.exp(-b/self.ramp_seconds)-np.exp(-a/self.ramp_seconds)))
        return float(value)

    def next_time(self,current_time):
        phase=float(self.rng.uniform(1-self.jitter,1+self.jitter))
        if self.constant_rate is not None:
            return current_time+phase/self.constant_rate
        if self.trigger_time is None or self.high_rate==self.low_rate:
            return current_time+phase/self.low_rate
        lo=current_time;hi=current_time+phase/self.low_rate
        for _ in range(44):
            mid=.5*(lo+hi)
            if self.integrated_rate(current_time,mid)<phase:
                lo=mid
            else:
                hi=mid
        return .5*(lo+hi)


class BoundaryFixture:
    """Fixed fine-grid truth, independent of every acquisition policy.

    The crossing axis progresses through x=0 near 6 s. Lateral components come
    from the existing independently generated motion families. This makes them
    boundary-crossing derivatives of those fixtures, not unchanged copies of
    the previous benchmark. The boundary causes no change in physical motion.
    """
    def __init__(self,family,seed):
        self.times=np.linspace(0,14,1401)
        case=make_case(family,seed,samples=len(self.times),observation_times=self.times)
        rng=np.random.default_rng(seed+2700000)
        phase=rng.uniform(-np.pi,np.pi)
        truth=case['truth'].copy()
        truth[:,0]=2.*(self.times-6.)+.3*np.sin(.7*self.times+phase)
        self.truth=truth
        j=int(np.flatnonzero(truth[:,0]>=0)[0])
        self.crossing_time=float(np.interp(0.,truth[j-1:j+1,0],self.times[j-1:j+1]))

    def position(self,times):
        times=np.asarray(times,float)
        if np.any(times<0) or np.any(times>14):
            raise ValueError('Query outside fixed truth fixture')
        return np.stack([np.interp(times,self.times,self.truth[:,d]) for d in range(3)],axis=-1)
