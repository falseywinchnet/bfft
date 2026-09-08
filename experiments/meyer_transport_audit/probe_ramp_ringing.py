"""Dense local derivative used only to diagnose repeated refresh feedback."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap, State, grad, project_disk_derivative, rms
from experiments.meyer_transport_audit.intermediate_chart import pack, unpack
from experiments.meyer_transport_audit.diagnose_ramp_refresh import refresh_endpoint


def run(n):
    result={}
    for lam,mu in [(.02,20),(.05,40)]:
        m=ReducedMeyerMap(255.*np.arange(n)[None,:]/n,lam,mu)
        case={}
        for name,branches in [('ordinary',''),('refresh','uw'),('u_refresh','u'),('w_refresh','w')]:
            def step(z):return refresh_endpoint(m,m.step(z),branches)
            def tangent(z,d,base):
                h=m.tangent(z,d);fields=list(h.fields())
                for b,x,tx,ty,hx,hy,radius,index in [('u',h.u,base.tux,base.tuy,h.tux,h.tuy,m.ru,2),('w',h.w,base.twx,base.twy,h.twx,h.twy,m.rw,4)]:
                    if b in branches:
                        dq=project_disk_derivative(tx,ty,hx,hy,radius);dx=grad(x)
                        fields[index],fields[index+1]=dx[0]+dq[0],dx[1]+dq[1]
                return State(*fields)
            z=m.initial()
            checkpoints=[]
            for k in range(2,513):
                z=step(z)
                if k in [64,128,256,384,512]:
                    zn=step(z);znn=step(zn)
                    h=pack(zn)-pack(z);hn=pack(znn)-pack(zn)
                    cosine=float(h@hn/max(np.linalg.norm(h)*np.linalg.norm(hn),1e-300))
                    checkpoints.append({'pass':k,'successive_increment_cosine':cosine,'increment_norm_ratio':float(np.linalg.norm(hn)/np.linalg.norm(h)),
                                        'signed_rayleigh':float(h@hn/(h@h))})
                if k==256:z256=z
            z=z256;base=m.step(z)
            A=np.column_stack([pack(tangent(z,unpack(m,e),base)) for e in np.eye(4*n)])
            eig=np.linalg.eigvals(A)
            nonunit=eig[np.abs(eig-1)>1e-7]
            ordered=nonunit[np.argsort(-np.abs(nonunit))]
            negative=eig[(eig.real<0)&(np.abs(eig.imag)<1e-9)]
            x=pack(z);h=pack(step(z))-x;moving=z;error=0.
            for _ in range(32):
                x=x+h;h=A@h;moving=step(moving)
                error=max(error,rms(x-pack(moving)))
            case[name]={'checkpoints':checkpoints,'largest_nonunit_modes':[{'real':float(e.real),'imag':float(e.imag),'modulus':float(abs(e))} for e in ordered[:8]],
                        'most_negative_eigenvalue':float(np.min(negative.real)) if len(negative) else None,
                        'frozen_derivative_orbit_error_rms_32_steps':error}
            print(lam,name,case[name]['most_negative_eigenvalue'],checkpoints[-2],flush=True)
        result[f'lambda{lam}_mu{mu}']=case
    return {'size':n,'note':'Dense numerical local-spectrum diagnosis; no new runtime solver. Derivative validity checked against 32 subsequent actual steps.','cases':result}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=128);p.add_argument('--out',required=True)
    a=p.parse_args();result=run(a.size);Path(a.out).write_text(json.dumps(result,indent=2))
