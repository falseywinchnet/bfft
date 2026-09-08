"""Small 1-D exact-affine-chart probe, not a production acceleration.

The y fields vanish identically. Disk projection reduces to scalar clipping,
so a fixed branch/sign pattern makes the complete recurrence affine.
We solve its singular linear fixed-point equations by dense least squares,
then test whether the candidate actually belongs to that pattern. No factors.
"""
from pathlib import Path
import sys,json,argparse,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap,State,rms
from experiments.meyer_transport_audit.certificate import certificate

def pack(z):return np.concatenate([x.ravel() for x in [z.u,z.w,z.tux,z.twx]])
def unpack(m,v):
    u,w,tu,tw=np.asarray(v).reshape(4,*m.shape);zero=np.zeros(m.shape)
    return State(u,w,tu,zero.copy(),tw,zero.copy())
def labels(t,a):return np.where(t>a,1,np.where(t < -a,-1,0))

def probe(m,z):
    start=time.perf_counter();n=4*m.count
    A=np.empty((n,n));unit=np.zeros(n)
    for i in range(n):
        unit[i]=1.;A[:,i]=pack(m.tangent(z,unpack(m,unit)));unit[i]=0.
    r=pack(m.subtract(m.step(z),z))
    h,_,rank,singular=np.linalg.lstsq(np.eye(n)-A,r,rcond=None)
    obstruction=r-(np.eye(n)-A)@h
    candidate=unpack(m,pack(z)+h)
    changed=sum(int(np.sum(labels(a,c)!=labels(b,c))) for a,b,c in
                [(z.tux,candidate.tux,m.ru),(z.twx,candidate.twx,m.rw)])
    eig=np.linalg.eigvals(A)
    distance=np.abs(eig-1);nonunit=eig[distance>1e-8]
    residual=m.residual_norm(candidate)
    # A nonzero left-null obstruction forces motion while this chart persists.
    initial_labels=(labels(z.tux,m.ru),labels(z.twx,m.rw))
    forward=z;first_event=None
    for k in range(1,2049):
        forward=m.step(forward)
        if (np.any(labels(forward.tux,m.ru)!=initial_labels[0]) or
                np.any(labels(forward.twx,m.rw)!=initial_labels[1])):
            first_event=k;break
    result={'unknowns':n,'rank':int(rank),'nullity':n-int(rank),
            'linear_equation_residual_rms':rms((np.eye(n)-A)@h-r),
            'left_null_obstruction_check_rms':rms((np.eye(n)-A).T@obstruction),
            'first_actual_branch_event_within_2048_steps':first_event,
            'changed_clip_branches':changed,'clip_sites':2*m.count,
            'candidate_fixed_point_residual_rms':residual,
            'candidate_gap':certificate(m,candidate),
            'incoming_gap':certificate(m,z),
            'largest_nonunit_eigenvalue_modulus':float(np.max(np.abs(nonunit))) if len(nonunit) else None,
            'complex_eigenvalue_count':int(np.sum(np.abs(eig.imag)>1e-8)),
            'seconds_including_dense_diagnostics':time.perf_counter()-start}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=64);p.add_argument('--out',required=True);a=p.parse_args()
    x=np.arange(a.size)[None,:];out={}
    for name,f in {'ramp':255*x/a.size,'edge':50+150.*(x>a.size//2),
                   'carrier':100+2*np.sin(2*np.pi*x/a.size)}.items():
        m=ReducedMeyerMap(f,.05,40);z=m.initial();rows=[]
        for k in range(1,129):
            if k>1:z=m.step(z)
            if k in [4,16,64,128]:
                row={'pass':k,**probe(m,z)};rows.append(row)
                print(name,k,'branches',row['changed_clip_branches'],'residual',row['candidate_fixed_point_residual_rms'],flush=True)
        out[name]=rows
    Path(a.out).write_text(json.dumps({'size':a.size,'note':'Dense 1-D structural diagnostic; no speed claim.', 'scenes':out},indent=2))
