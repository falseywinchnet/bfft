"""Exact within-pass identities and phase-resolved transport measurements.

No accelerated updates: all trajectories use the authoritative six-field map.
Additional transforms are diagnostic work, never charged as free solver work.
"""
from pathlib import Path
import sys,json,argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import (
    ReducedMeyerMap,grad,div,project_disk,project_disk_derivative,State,rms)
from experiments.meyer_transport_audit.terminal_geometry import measure

def vnorm(v):return float(np.sqrt(np.mean(v[0]**2+v[1]**2)))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def lap(x):return -div(*grad(x))

def longitudinal(m,v):
    spec=np.fft.fft2(-div(*v))
    phi=np.fft.ifft2(np.divide(spec,m.symbol,out=np.zeros_like(spec),where=m.symbol>0)).real
    return grad(phi)

def branch(m,x,xn,t,tn,a):
    dx=grad(x);dxn=grad(xn);b=sub(t,dx);p=project_disk(*t,a)
    d=sub(t,p);db=sub(p,b);h=sub(tn,t)
    pn=project_disk(*tn,a)
    jp=project_disk_derivative(*t,*h,a)
    rem=sub(sub(pn,p),jp)
    # Exact Hodge split of the new retained field.
    pl=longitudinal(m,p);pt=sub(p,pl)
    dbl=longitudinal(m,db);dbt=sub(db,dbl)
    dt=sub(d,longitudinal(m,d))
    jt=project_disk_derivative(*tn,*pt,a)
    mag=np.hypot(*t);magn=np.hypot(*tn)
    crossed=(mag>a)!=(magn>a)
    stable_exterior=(mag>a)&(magn>a)
    rem2=rem[0]**2+rem[1]**2
    denom=float(np.sum(rem2))
    gxinc=sub(dxn,dx)
    cancellation=vnorm(h)/max(vnorm(gxinc)+vnorm(db),1e-300)
    result={
        'memory_change_rms':vnorm(db),'gradient_change_rms':vnorm(gxinc),
        't_change_rms':vnorm(h),'addition_cancellation_ratio':cancellation,
        'shrink_gradient_mismatch_rms':vnorm(sub(dx,d)),
        'shrink_identity_error':vnorm(sub(sub(dx,d),db)),
        'update_identity_error':vnorm(sub(h,add(gxinc,db))),
        'prox_complementarity_error':float(np.max(np.abs(np.hypot(*d)-(p[0]*d[0]+p[1]*d[1])/a))),
        'projected_capacity_excess':max(0.,float(np.max(np.hypot(*p)))-a),
        'incoming_capacity_excess':max(0.,float(np.max(np.hypot(*b)))-a),
        'active_fraction':float(np.mean(mag>a)),
        'crossing_fraction':float(np.mean(crossed)),
        'projection_remainder_rms':vnorm(rem),
        'remainder_energy_crossings':float(np.sum(rem2[crossed]))/denom if denom>1e-25 else None,
        'remainder_energy_stable_exterior':float(np.sum(rem2[stable_exterior]))/denom if denom>1e-25 else None,
        'retained_transverse_energy_fraction':vnorm(pt)**2/max(vnorm(p)**2,1e-300),
        'mismatch_transverse_energy_fraction':vnorm(dbt)**2/max(vnorm(db)**2,1e-300),
        'nonintegrability_identity_error':vnorm(add(dt,dbt)),
        'transverse_divergence_rms':rms(div(*pt)),
        'transverse_next_reflection_response_rms':rms(2*div(*jt)),
    }
    return result,b,p,db,rem

def trace(m,z):
    zn=m.step(z);du=zn.u-z.u;dw=zn.w-z.w
    bu,b0u,pu,dbu,eu=branch(m,z.u,zn.u,(z.tux,z.tuy),(zn.tux,zn.tuy),m.ru)
    bw,b0w,pw,dbw,ew=branch(m,z.w,zn.w,(z.twx,z.twy),(zn.twx,zn.twy),m.rw)
    q0=-m.etau*div(*b0u);q=-m.etau*div(*pu)
    v0=-m.etaw/m.cw*div(*b0w);v=-m.etaw/m.cw*div(*pw)
    pred_u=du+m.etau/m.cu*lap(du)
    rhs_u=z.w-(2*q-q0)/m.cu
    pred_w=dw+m.etaw/m.cw*lap(dw)
    rhs_w=m.image-z.u-z.w-du-(2*v-v0)
    nextpu=project_disk(zn.tux,zn.tuy,m.ru)
    nextpw=project_disk(zn.twx,zn.twy,m.rw)
    def gap(x,p,r):
        return measure(m.image,x,m.etau*p[0],m.etau*p[1],
                       m.etaw/m.cw*r[0],m.etaw/m.cw*r[1],m.lam,m.mu)
    phases={'incoming':gap(z.u,b0u,b0w),
            'after_projection':gap(z.u,pu,pw),
            'after_global_solves':gap(zn.u,pu,pw),
            'next_projection':gap(zn.u,nextpu,nextpw)}
    # Full-map nonlinear remainder is exactly driven by local projection remainders.
    h=m.subtract(zn,z);ah=m.tangent(z,h);znn=m.step(zn)
    actual=m.subtract(m.subtract(znn,zn),ah)
    def solve(rhs,c,eta):
        return np.fft.ifft2(np.fft.fft2(rhs)/(c+eta*m.symbol)).real
    ru=solve(2*m.etau*div(*eu),m.cu,m.etau)
    rw=solve(-m.cw*ru+2*m.etaw*div(*ew),m.cw,m.etaw)
    rtu=add(grad(ru),eu);rtw=add(grad(rw),ew)
    predicted=State(ru,rw,*rtu,*rtw)
    # Proximal complementarity bounds in terms of actual within-pass mismatches.
    cartoon_bound=2*float(np.mean(np.hypot(*dbu)))
    tq=add(grad(q-m.lam*z.w),tuple(m.lam*x for x in dbw))
    texture_bound=2*m.mu*float(np.mean(np.hypot(*tq)))
    return {'u_branch':bu,'w_branch':bw,'phases':phases,
            'u_balance_identity_rms':rms(pred_u-rhs_u),
            'w_balance_identity_rms':rms(pred_w-rhs_w),
            'full_remainder_identity_rms':rms(m.pack(m.subtract(actual,predicted))),
            'full_nonlinear_remainder_rms':rms(m.pack(actual)),
            'cartoon_complementarity_bound':cartoon_bound,
            'texture_complementarity_bound':texture_bound,
            'dual_primal_balance_rms':rms(q-m.lam*z.w)}

def run(n):
    from experiments.benchmark_meyer_flow_jump import benchmark_scene
    y,x=np.mgrid[:n,:n];rng=np.random.default_rng(983)
    scenes={'ramp':255*x/n,'edge':50.+150.*(x>n//2),
            'carrier':100+2*np.sin(2*np.pi*(x+2*y)/n),
            'crossing':benchmark_scene(n),'noise':rng.uniform(0,255,(n,n))}
    out={}
    for lam,mu in [(.02,20),(.05,40),(.1,80)]:
        for name,f in scenes.items():
            m=ReducedMeyerMap(f,lam,mu);z=m.initial();rows=[]
            for k in range(1,129):
                if k>1:z=m.step(z)
                if k in [1,2,4,8,16,32,64,128]:rows.append({'pass':k,**trace(m,z)})
            out[f'{name}_lambda{lam}_mu{mu}']=rows
            print(name,lam,'max identity',max(r['full_remainder_identity_rms'] for r in rows),flush=True)
    return {'size':n,'note':'Unmodified ordinary trajectories; phase and Hodge transforms are diagnostic work.',
            'scenes':out}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=64)
    p.add_argument('--out',required=True);a=p.parse_args()
    Path(a.out).write_text(json.dumps(run(a.size),indent=2))
