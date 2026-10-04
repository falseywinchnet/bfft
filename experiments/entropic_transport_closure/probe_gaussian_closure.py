"""Continuous centered Gaussian controls; not Gaussian point-cloud proxies.

Kernel exp(-|x-y|^2/(2t)), Gaussian covariance marginals A,B. The v scaling
is exp(-y^T V y/2); p=I/t+V is conditional precision. All matrix operations
are in physical dimension d, not the discretized support size.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def sqrt_spd(A):
    w,U=np.linalg.eigh(A)
    return (U*np.sqrt(w))@U.T


def step(A,B,t,p):
    return np.linalg.inv(B)+np.linalg.inv(t*t*np.linalg.inv(A)+np.linalg.inv(p))


def lift(A,B,t):
    I=np.eye(len(A)); T=t*t*np.linalg.inv(A); b=np.linalg.inv(B)
    return np.block([[I+b@T,b],[T,I]])


def jump(A,B,t,p,H):
    L=lift(A,B,t); L/=np.linalg.norm(L)
    P=np.eye(len(L)); count=0
    while H:
        if H&1:
            P=L@P; P/=np.linalg.norm(P); count+=1
        L=L@L; L/=np.linalg.norm(L); H//=2; count+=1
    d=len(A); num=P[:d,:d]@p+P[:d,d:]; den=P[d:,:d]@p+P[d:,d:]
    out=np.linalg.solve(den.T,num.T).T
    return out,float(np.linalg.cond(den)),count


def fixed_precision(A,B,t):
    root=sqrt_spd(A)
    f=sqrt_spd(root@B@root+t*t/4*np.eye(len(A)))-t/2*np.eye(len(A))
    return root@np.linalg.solve(f,root)/t


def coupling(A,p,t):
    # After the row projection, X has covariance A and Y|X has
    # mean p^-1 X/t and covariance p^-1.
    invp=np.linalg.inv(p); C=A@invp/t
    B=invp+invp@A@invp/(t*t)
    return C,B


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); args=ap.parse_args()
    angle=.47; R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    cases=[('isotropic',np.eye(2),2*np.eye(2)),
           ('anisotropic_commuting',np.diag([.5,2.]),np.diag([1.,3.])),
           ('anisotropic_rotated',np.diag([.5,2.]),R@np.diag([1.,3.])@R.T)]
    rows=[]
    for name,A,B in cases:
        for t in [1.,.1,.01,.001]:
            pstar=fixed_precision(A,B,t); C,achieved=coupling(A,pstar,t)
            invroot=np.linalg.inv(sqrt_spd(B))
            slow=np.linalg.eigvalsh(invroot@C.T@np.linalg.solve(A,C)@invroot)
            p=np.eye(2)/t; target={1,16,64,256,2048}; checks=[]
            for k in range(1,2049):
                p=step(A,B,t,p)
                if k in target:
                    try:
                        powered,cond,count=jump(A,B,t,np.eye(2)/t,k)
                        err=float(np.linalg.norm(powered-p)/np.linalg.norm(p))
                        failure=None
                    except np.linalg.LinAlgError as exc:
                        err=cond=None; count=None; failure=str(exc)
                    checks.append(dict(horizon=k,jump_relative_error=err,denominator_condition=cond,
                                       block_multiplications=count,failure=failure,
                                       ordinary_relative_error_to_fixed=float(np.linalg.norm(p-pstar)/np.linalg.norm(pstar))))
            rows.append(dict(case=name,t=t,dimension=2,precision_parameter_count=1 if name=='isotropic' else 3,
                             degree1_eigenvalues=slow.tolist(),degree1_gap_between_directions=float(np.ptp(slow)),
                             closed_form_marginal_relative_error=float(np.linalg.norm(achieved-B)/np.linalg.norm(B)),
                             fixed_point_relative_defect=float(np.linalg.norm(step(A,B,t,pstar)-pstar)/np.linalg.norm(pstar)),
                             horizons=checks))
    Path(args.out).write_text(json.dumps(dict(scope='exact continuous Gaussian covariance controls at fixed regularization; no annealing and no point-cloud approximation',cases=rows),indent=2,allow_nan=False))
    print('saved',args.out,flush=True)


if __name__=='__main__': main()
