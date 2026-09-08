"""Branch and stage interventions for the periodic-ramp refresh regression."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap, State, grad, div, project_disk, rms
from experiments.meyer_transport_audit.defect_relaxation import DefectRelaxation
from experiments.meyer_transport_audit.certificate import certificate, tv


def refresh_endpoint(m,z,branches):
    fields = list(z.fields())
    for branch,x,tx,ty,radius,index in [('u',z.u,z.tux,z.tuy,m.ru,2),('w',z.w,z.twx,z.twy,m.rw,4)]:
        if branch in branches:
            p = project_disk(tx,ty,radius)
            g = grad(x)
            fields[index],fields[index+1] = g[0]+p[0],g[1]+p[1]
    return State(*fields)


def terms(m,z,incoming=False):
    if incoming:
        gu,gw = grad(z.u),grad(z.w)
        pu=(z.tux-gu[0],z.tuy-gu[1]);pw=(z.twx-gw[0],z.twy-gw[1])
    else:
        pu=project_disk(z.tux,z.tuy,m.ru);pw=project_disk(z.twx,z.twy,m.rw)
    q=-m.etau*div(*pu)
    v=-(m.etaw/m.cw)*div(*pw)
    g=tuple(m.etaw/m.cw*a for a in pw)
    du,dq=grad(z.u),grad(q)
    a=(tv(z.u)-m.etau*sum(float(np.sum(x*y)) for x,y in zip(du,pu)))/m.count
    b=(m.mu*tv(q)-sum(float(np.sum(x*y)) for x,y in zip(dq,g)))/m.count
    c=.5*m.lam*float(np.mean((m.image-z.u-v-q/m.lam)**2))
    primal=(tv(z.u)+.5*m.lam*np.sum((m.image-z.u-v)**2))/m.count
    return {'cartoon':a,'texture':b,'balance':c,'gap':a+b+c,'primal':float(primal),
            'dual':float(primal-a-b-c)}


def trace_branch(m,z,ordinary,branch):
    x,t,gx,r = (z.u,(z.tux,z.tuy),ordinary.u,m.ru) if branch=='u' else (z.w,(z.twx,z.twy),ordinary.w,m.rw)
    dx,dnew=grad(x),grad(gx)
    b=tuple(ti-di for ti,di in zip(t,dx))
    p=project_disk(*t,r)
    q=project_disk(*(gi+pi for gi,pi in zip(dnew,p)),r)
    beta=tuple(pi-bi for pi,bi in zip(p,b));e=tuple(qi-pi for qi,pi in zip(q,p))
    dot=beta[0]*e[0]+beta[1]*e[1]
    energy=e[0]**2+e[1]**2
    return {'beta_rms':rms(np.hypot(*beta)),'extra_rms':rms(np.hypot(*e)),
            'opposing_energy_share':float(np.sum(energy[dot<0])/max(np.sum(energy),1e-300)),
            'interior_count':int(np.sum(np.hypot(*t)<r-1e-9))}


def run(n,steps):
    f=255.*np.arange(n)[None,:]/n
    report={'size':n,'steps':steps,'source':'Periodic sawtooth, f_i=255*i/n; one row is invariant under the repeated-row 2-D map.','cases':{}}
    for lam,mu in [(.02,20),(.05,40),(.1,80)]:
        m=ReducedMeyerMap(f,lam,mu)
        ref=m.initial()
        for _ in range(4095):ref=m.step(ref)
        rc=certificate(m,ref)
        opt=(rc['primal_upper']+rc['dual_lower'])/(2*n)
        trajectories={};states={}
        for method in ['ordinary','refresh','u_refresh','w_refresh','aligned_refresh']:
            op=DefectRelaxation(m,'aligned_refresh' if method=='aligned_refresh' else 'ordinary')
            z=m.initial();rows=[];kept={}
            for k in range(2,steps+1):
                old=z
                ordinary=m.step(old)
                if method=='aligned_refresh':z=op.step(old)
                elif method=='ordinary':z=ordinary
                else:z=refresh_endpoint(m,ordinary, 'uw' if method=='refresh' else method[0])
                cur,inc=terms(m,z),terms(m,z,True)
                assert abs(cur['gap']-certificate(m,z)['gap_per_pixel'])<1e-9
                row={'pass':k,'projected':cur,'incoming':inc,'primal_excess':cur['primal']-opt,
                     'dual_deficit':opt-cur['dual'],'u_reference_distance_rms':rms(z.u-ref.u),'w_reference_distance_rms':rms(z.w-ref.w),
                     'u':trace_branch(m,old,ordinary,'u'),'w':trace_branch(m,old,ordinary,'w')}
                rows.append(row)
                if k in [32,48,64,80,88,96,128]:kept[k]=z
            trajectories[method]=rows;states[method]=kept
        interventions=[]
        for k,z in states['ordinary'].items():
            for branch in ['none','u','w','uw']:
                candidate=z if branch=='none' else refresh_endpoint(m,z,branch)
                records=[{'horizon':0,**terms(m,candidate),'u_reference_distance_rms':rms(candidate.u-ref.u),'w_reference_distance_rms':rms(candidate.w-ref.w)}]
                for j in range(1,129):
                    candidate=m.step(candidate)
                    if j in [1,8,32,128]:records.append({'horizon':j,**terms(m,candidate),'u_reference_distance_rms':rms(candidate.u-ref.u),'w_reference_distance_rms':rms(candidate.w-ref.w)})
                interventions.append({'at_pass':k,'branch':branch,'records':records})
        stop_interventions=[]
        for k in [64,128]:
            seed=states['refresh'][k]
            for branches in ['', 'u','w','uw']:
                candidate=seed;records=[]
                def advance(x):return refresh_endpoint(m,m.step(x),branches)
                for j in range(1,129):
                    candidate=advance(candidate)
                    if j in [1,8,32,64,128]:
                        cur,inc=terms(m,candidate),terms(m,candidate,True)
                        zn=advance(candidate);znn=advance(zn)
                        h=m.pack(zn)-m.pack(candidate);hn=m.pack(znn)-m.pack(zn)
                        records.append({'horizon':j,'projected_gap':cur['gap'],
                            'best_gap':min(cur['primal'],inc['primal'])-max(cur['dual'],inc['dual']),
                            'increment_cosine':float(h@hn/max(np.linalg.norm(h)*np.linalg.norm(hn),1e-300))})
                stop_interventions.append({'at_pass':k,'continued_refresh_branches':branches,'records':records})
        report['cases'][f'lambda{lam}_mu{mu}']={'reference_gap':rc['gap_per_pixel'],'optimum_midpoint':opt,
                                               'trajectories':trajectories,'interventions':interventions,
                                               'stop_interventions':stop_interventions}
        print(lam,mu,{method:{str(t):next((r['pass'] for r in rows if r['projected']['gap']<=t),None) for t in [.01,.001,.0001]} for method,rows in trajectories.items()},flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=128);p.add_argument('--steps',type=int,default=384);p.add_argument('--out',required=True)
    a=p.parse_args();result=run(a.size,a.steps);Path(a.out).write_text(json.dumps(result,indent=2))
