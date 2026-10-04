"""Curved-disk discovery for the ORIGINAL coupled Meyer recurrence.

The quotient (s=u+w, t_u, t_w) retains every future-driving variable. It
forgets only the current primal difference; one actual next pass recovers
both primals. Whitened coordinates use ||s||²+a||t_u||²+b||t_w||²,
a=eta_u/c_u, b=eta_w/c_w. The quotient map is nonexpansive.
No fixed exterior disk mask is treated as an affine map.
"""
from dataclasses import dataclass
import numpy as np
from .core import Linearization,arnoldi
from .certified_piece import reduced_path
from .discovery import arnoldi_stream
from experiments.meyer_transport_audit.model import ReducedMeyerMap,State,project_disk


class MeyerQuotient:
    def __init__(self,image,lam=.05,mu=40.):
        self.full=ReducedMeyerMap(image,lam,mu)
        self.a=self.full.etau/self.full.cu;self.b=self.full.etaw/self.full.cw
        self.weights=np.array([1.,np.sqrt(self.a),np.sqrt(self.a),np.sqrt(self.b),np.sqrt(self.b)])
        self.shape=self.full.shape;self.n=self.full.count
        self.initial=self.encode(self.full.initial())
        self.curvature_gain=self._curvature_gain()
    def encode(self,state):
        return (np.stack([state.u+state.w,state.tux,state.tuy,state.twx,state.twy])*self.weights[:,None,None]).ravel()
    def representative(self,z):
        s,ux,uy,wx,wy=np.asarray(z).reshape(5,*self.shape)/self.weights[:,None,None]
        return State(s,np.zeros_like(s),ux,uy,wx,wy)
    def next_full(self,z):return self.full.step(self.representative(z))
    def step(self,z):return self.encode(self.next_full(z))
    def linearize(self,z):
        anchor=self.representative(z)
        return Linearization(self.encode(self.full.step(anchor)),
                             lambda h:self.encode(self.full.tangent(anchor,self.representative(h))))
    def _curvature_gain(self):
        # At each frequency, transverse projection remainders have gain one.
        # Longitudinal inputs reduce to a real 3x2 matrix after a unitary phase
        # change. Its Gram has entries [[1-d,g],[g,1]].
        l=self.full.symbol;a=self.a;b=self.b
        d=4*a*l/((1+a*l)**2*(1+b*l))
        g=2*np.sqrt(a*b)*l/((1+a*l)*(1+b*l))
        return float(np.sqrt(np.max(1-d/2+np.sqrt((d/2)**2+g*g))))


@dataclass
class DiskRelation:
    radius:float
    rho:np.ndarray
    normal:np.ndarray
    tangent:np.ndarray
    inside:np.ndarray

    @classmethod
    def acquire(cls,tx,ty,Qx,Qy,radius):
        rho=np.hypot(tx,ty).ravel();safe=np.maximum(rho,1e-300)
        nx=tx.ravel()/safe;ny=ty.ravel()/safe
        # At the origin choose a frame; the projection remainder norm is
        # rotation invariant and the inside formula uses only the new radius.
        nx=np.where(rho>0,nx,1.);ny=np.where(rho>0,ny,0.)
        normal=nx[:,None]*Qx+ny[:,None]*Qy
        tangent=-ny[:,None]*Qx+nx[:,None]*Qy
        return cls(float(radius),rho,normal,tangent,rho<=radius)

    def squared_remainder_norms(self,C):
        """Exact pointwise norms of P(t+Qc)-P(t)-DP(t)Qc, summed over pixels.

        The code's derivative at the boundary is identity, matching the full
        map tangent. Formula remains valid across both inward/outward events.
        Radial exterior motion has zero remainder until it crosses the disk.
        """
        hn=self.normal@C;ht=self.tangent@C
        x=self.rho[:,None]+hn;length=np.hypot(x,ht)
        result=np.zeros(C.shape[1]);inside=self.inside
        if inside.any():
            result+=np.sum(np.maximum(length[inside]-self.radius,0.)**2,axis=0)
        outside=~inside
        if outside.any():
            xx=x[outside];yy=ht[outside];rr=length[outside]
            factor=np.minimum(1.,self.radius/np.maximum(rr,1e-300))
            qn=factor*xx-self.radius
            qt=(factor-self.radius/self.rho[outside,None])*yy
            result+=np.sum(qn*qn+qt*qt,axis=0)
        return result


@dataclass
class CurvedChart:
    model:MeyerQuotient
    anchor:np.ndarray
    basis:object
    disks:tuple

    @classmethod
    def acquire(cls,model,z,depth=4):
        lin=model.linearize(z);basis=arnoldi(lin.action,lin.next_state-z,depth)
        return cls.from_basis(model,z,basis)

    @classmethod
    def from_basis(cls,model,z,basis):
        Q=basis.Q.reshape(5,model.n,basis.actions)/model.weights[:,None,None]
        state=model.representative(z)
        du=DiskRelation.acquire(state.tux,state.tuy,Q[1],Q[2],model.full.ru)
        dw=DiskRelation.acquire(state.twx,state.twy,Q[3],Q[4],model.full.rw)
        return cls(model,z.copy(),basis,(du,dw))

    def scan(self,maximum_horizon=64,chunk=16,stop_tolerance=None):
        k=self.basis.actions
        source=np.zeros(k)
        if k:source[0]=self.basis.beta
        C=reduced_path(self.basis.H,source,maximum_horizon+1)
        gram=self.basis.remainder.T@self.basis.remainder
        compression=np.sqrt(np.maximum(0.,np.sum(C[:,:-1]*(gram@C[:,:-1]),axis=0)))
        curvature=np.zeros(maximum_horizon)
        scanned=maximum_horizon
        for lo in range(0,maximum_horizon,chunk):
            hi=min(maximum_horizon,lo+chunk)
            qu=self.disks[0].squared_remainder_norms(C[:,lo:hi])
            qw=self.disks[1].squared_remainder_norms(C[:,lo:hi])
            curvature[lo:hi]=self.model.curvature_gain*np.sqrt(self.model.a*qu+self.model.b*qw)
            if stop_tolerance is not None:
                cumulative=float(np.sum(curvature[:hi]+compression[:hi]))
                relative=cumulative/max(np.linalg.norm(C[:,hi]),1e-300)
                if relative>stop_tolerance:
                    scanned=hi;break
        C=C[:,:scanned+1];curvature=curvature[:scanned];compression=compression[:scanned]
        bounds=np.cumsum(curvature+compression)
        displacement=np.linalg.norm(C[:,1:],axis=0)
        return {'coordinates':C,'curvature':curvature,'compression':compression,
                'bound':bounds,'relative_bound':bounds/np.maximum(displacement,1e-300),
                'displacement':displacement}

    def candidate(self,scan,horizon):
        return self.anchor+self.basis.Q@scan['coordinates'][:,horizon]


@dataclass
class CurvedDiscovery:
    candidate:np.ndarray
    horizon:int
    depth:int
    accepted:bool
    bound:float
    relative_bound:float
    map_calls:int
    tangent_actions:int
    geometry_points:int


def discover_curved(model,z,tolerance=.1,depths=(2,4,8),maximum_horizon=64):
    """Conservative, cost-counted discovery. No future trajectory or gap oracle.

    Stop a depth's horizon scan after the first chunk exceeding tolerance;
    this need not find the globally longest admissible horizon. Rejected
    geometry evaluations are timed and retained in geometry_points.
    """
    lin=model.linearize(z);points=0;actions=0
    for basis,closed in arnoldi_stream(lin.action,lin.next_state-z,max(depths)):
        actions=basis.actions
        if actions not in depths and not closed:continue
        chart=CurvedChart.from_basis(model,z,basis)
        scan=chart.scan(maximum_horizon,chunk=8,stop_tolerance=tolerance)
        points+=len(scan['bound'])
        ms=np.arange(1,len(scan['bound'])+1)
        allowed=ms[(scan['relative_bound']<=tolerance)&(ms>=1.5*(1+actions+1))]
        if len(allowed):
            m=int(allowed[-1])
            return CurvedDiscovery(chart.candidate(scan,m),m,actions,True,
                                   float(scan['bound'][m-1]),float(scan['relative_bound'][m-1]),
                                   1,actions,points)
        if closed:break
    return CurvedDiscovery(lin.next_state,1,actions,False,0.,0.,1,actions,points)
