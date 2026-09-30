import unittest
import numpy as np
from .curved_meyer import MeyerQuotient,DiskRelation,CurvedChart,discover_curved
from experiments.meyer_transport_audit.model import State,project_disk,project_disk_derivative,grad,div,solve_screened


class CurvedMeyerTests(unittest.TestCase):
    def test_quotient_retains_all_future_outputs(self):
        rng=np.random.default_rng(51);q=MeyerQuotient(rng.normal(size=(9,10))*30+100)
        full=q.full.initial();z=q.initial.copy()
        for _ in range(80):
            full=q.full.step(full);recovered=q.next_full(z);z=q.step(z)
            np.testing.assert_allclose(q.full.pack(full),q.full.pack(recovered),atol=2e-12)
        arbitrary=rng.normal(size=q.shape)
        changed=State(full.u+arbitrary,full.w-arbitrary,full.tux,full.tuy,full.twx,full.twy)
        np.testing.assert_allclose(q.full.pack(q.full.step(full)),q.full.pack(q.full.step(changed)),atol=1e-13)

    def test_global_nonexpansivity_not_only_on_proximal_graph(self):
        rng=np.random.default_rng(52)
        for shape in [(1,9),(8,9)]:
            q=MeyerQuotient(rng.normal(size=shape)*20)
            for scale in [.01,1,10,100,1000]:
                for _ in range(15):
                    x,y=rng.normal(size=(2,len(q.initial)))*scale
                    fx,fy=q.step(x),q.step(y)
                    self.assertLessEqual(np.linalg.norm(fx-fy),np.linalg.norm(x-y)+1e-11)
                    # Stronger: quotient is 2/3-averaged in these coordinates.
                    left=np.sum((fx-fy)**2)+.5*np.sum(((x-fx)-(y-fy))**2)
                    self.assertLessEqual(left,np.sum((x-y)**2)*(1+1e-13)+1e-11)

    def test_quotient_tangent_matches_independent_differences(self):
        rng=np.random.default_rng(53);q=MeyerQuotient(rng.normal(size=(8,9))*30)
        z=q.initial;h=rng.normal(size=len(z));h/=np.linalg.norm(h)
        lin=q.linearize(z);eps=1e-5
        np.testing.assert_allclose(lin.action(h),(q.step(z+eps*h)-q.step(z-eps*h))/(2*eps),atol=1e-9)

    def test_disk_relations_exact_across_radial_and_mask_events(self):
        rng=np.random.default_rng(54);n=20;k=4;m=50;radius=2.
        t=rng.normal(size=(2,n))*3;t[:,0]=0;t[:,1]=[radius,0]
        Q=rng.normal(size=(2,n,k));C=rng.normal(size=(k,m))*3
        relation=DiskRelation.acquire(t[0],t[1],Q[0],Q[1],radius)
        expected=[];p=project_disk(*t,radius)
        for c in C.T:
            h=np.einsum('ink,k->in',Q,c);pn=project_disk(*(t+h),radius)
            dp=project_disk_derivative(*t,*h,radius)
            expected.append(sum(np.sum((a-b-d)**2) for a,b,d in zip(pn,p,dp)))
        np.testing.assert_allclose(relation.squared_remainder_norms(C),expected,atol=1e-11,rtol=1e-13)

    def test_radial_exterior_transport_is_exact_but_turning_is_curved(self):
        relation=DiskRelation.acquire(np.array([3.]),np.array([0.]),np.array([[1.,0.]]),np.array([[0.,1.]]),2.)
        C=np.array([[1.,-0.5,-2.,0.],[0.,0.,0.,1.]])
        values=relation.squared_remainder_norms(C)
        np.testing.assert_allclose(values[:2],0.,atol=1e-15)
        self.assertGreater(values[2],.9);self.assertGreater(values[3],.01)

    def test_linear_curvature_response_and_fourier_gain(self):
        rng=np.random.default_rng(55);q=MeyerQuotient(rng.normal(size=(10,11))*40)
        z=q.initial;lin=q.linearize(z);state=q.representative(z)
        for _ in range(20):
            h=rng.normal(size=len(z))*30;direction=q.representative(h)
            remainders=[]
            for tx,ty,hx,hy,r in [(state.tux,state.tuy,direction.tux,direction.tuy,q.full.ru),
                                   (state.twx,state.twy,direction.twx,direction.twy,q.full.rw)]:
                p=project_disk(tx,ty,r);pn=project_disk(tx+hx,ty+hy,r)
                dp=project_disk_derivative(tx,ty,hx,hy,r)
                remainders.append(tuple(a-b-d for a,b,d in zip(pn,p,dp)))
            qu,qw=remainders
            du=solve_screened(2*q.full.etau*div(*qu),q.full.cu,q.full.etau,q.full.symbol)
            dw=solve_screened(-q.full.cw*du+2*q.full.etaw*div(*qw),q.full.cw,q.full.etaw,q.full.symbol)
            gu=grad(du);gw=grad(dw)
            expected=q.encode(State(du,dw,gu[0]+qu[0],gu[1]+qu[1],gw[0]+qw[0],gw[1]+qw[1]))
            actual=q.step(z+h)-lin.next_state-lin.action(h)
            np.testing.assert_allclose(actual,expected,atol=1e-12)
            bound=q.curvature_gain*np.sqrt(q.a*sum(np.sum(v*v) for v in qu)+q.b*sum(np.sum(v*v) for v in qw))
            self.assertLessEqual(np.linalg.norm(actual),bound+1e-11)

    def test_fourier_gain_is_attained_by_a_longitudinal_mode(self):
        q=MeyerQuotient(np.zeros((16,17)));a,b=q.a,q.b;l=q.full.symbol
        dd=4*a*l/((1+a*l)**2*(1+b*l));gg=2*np.sqrt(a*b)*l/((1+a*l)*(1+b*l))
        gain2=1-dd/2+np.sqrt((dd/2)**2+gg*gg)
        iy,ix=np.unravel_index(np.argmax(gain2),q.shape)
        y,x=np.mgrid[:q.shape[0],:q.shape[1]]
        phi=np.cos(2*np.pi*(iy*y/q.shape[0]+ix*x/q.shape[1]))
        g=grad(phi);norm=np.sqrt(sum(np.sum(t*t) for t in g));g=tuple(t/norm for t in g)
        _,V=np.linalg.eigh(np.array([[1-dd[iy,ix],gg[iy,ix]],[gg[iy,ix],1.]]))
        u,w=V[:,-1];qu=tuple(u*t/np.sqrt(a) for t in g);qw=tuple(w*t/np.sqrt(b) for t in g)
        du=solve_screened(2*a*div(*qu),1.,a,l)
        dw=solve_screened(-du+2*b*div(*qw),1.,b,l)
        gu,gw=grad(du),grad(dw)
        response=q.encode(State(du,dw,gu[0]+qu[0],gu[1]+qu[1],gw[0]+qw[0],gw[1]+qw[1]))
        self.assertAlmostEqual(np.linalg.norm(response),q.curvature_gain,places=12)

    def test_whole_path_certificate_bounds_unmodified_meyer(self):
        rng=np.random.default_rng(56);q=MeyerQuotient(rng.normal(size=(8,9))*40+100)
        z=q.initial
        for _ in range(20):z=q.step(z)
        chart=CurvedChart.acquire(q,z,4);scan=chart.scan(40,chunk=7);actual=z.copy()
        for m in range(1,41):
            actual=q.step(actual)
            self.assertLessEqual(np.linalg.norm(actual-chart.candidate(scan,m)),scan['bound'][m-1]+2e-10)
        again=chart.scan(40,chunk=16)
        np.testing.assert_allclose(scan['bound'],again['bound'],atol=1e-10)

    def test_discovered_horizon_is_certified_without_reference_access(self):
        y,x=np.mgrid[:12,:16]
        q=MeyerQuotient(100+20*np.sin(2*np.pi*x/16))
        z=q.initial.copy()
        for _ in range(128):z=q.step(z)
        d=discover_curved(q,z)
        self.assertTrue(d.accepted)
        actual=z.copy()
        for _ in range(d.horizon):actual=q.step(actual)
        self.assertLessEqual(np.linalg.norm(actual-d.candidate),d.bound+1e-10)
        self.assertLessEqual(d.relative_bound,.1)
        self.assertEqual(d.map_calls,1)
        self.assertGreater(d.horizon,d.map_calls+d.tangent_actions)
