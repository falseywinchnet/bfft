"""Globally compatible quintic proposals admitted through CONV current laws.

The six-sample endpoint jets are polynomial exact through degree five.
The interior spline minimizes integrated squared third derivative with those
jets and the observed nodal values fixed. No truth or task label enters it.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize, linprog
from experiments.convstar import (raw_current_jet_bank, ordered_sign_ledger,
                                   project_signed_fibres, synthesize_polyphase)
from experiments.conv_functional_current import (admit_cell, minimum, basis,
                                                SLOPE_GRAM, POWER, lanczos)
from experiments.conv_projection_metric_certificate import matrices

VALUE_GRAM=np.array([[float(v) for v in row] for row in matrices()[0]])


def profile_range(c):
    roots=np.polynomial.polynomial.polyroots(POWER@c)
    points=np.array([0.,1.]+[float(r.real) for r in roots
                            if abs(r.imag)<1e-8 and 0<r.real<1])
    b=basis(points,5)
    controls=np.r_[0.,np.cumsum(c)]
    values=b@controls
    return float(values.min()),float(values.max())


def support_range_admission(y,c,signs,envelope=False):
    """Maximal feasible current ray under continuous local support ranges.

    Both endpoints obey the same continuous sign law. Range is tested at every
    polynomial extremum. 48 bisections bound the ray error by 2^-48.
    """
    result=c.copy()
    delta=np.diff(y,axis=0)
    baseline=(admitted(y,proposal='compact')[0] if envelope else
              project_signed_fibres(np.repeat(delta[:,None,:]/5,5,axis=1),signs,delta))
    for i in range(len(delta)):
        for ch in range(y.shape[1]):
            support=y[max(0,i-2):min(len(y),i+4),ch]-y[i,ch]
            lo,hi=float(support.min()),float(support.max())
            if envelope:
                base_lo,base_hi=profile_range(baseline[i,:,ch])
                lo=min(lo,base_lo);hi=max(hi,base_hi)
            v=c[i,:,ch]
            low,high=profile_range(v)
            tol=2e-13*max(1,hi-lo)
            if low>=lo-tol and high<=hi+tol:
                continue
            b=baseline[i,:,ch]
            left,right=0.,1.
            for _ in range(48):
                t=(left+right)/2
                low,high=profile_range(b+t*(v-b))
                if low>=lo-tol and high<=hi+tol:
                    left=t
                else:
                    right=t
            result[i,:,ch]=b+left*(v-b)
    return result


def compatible_proposal(source):
    y=np.asarray(source,dtype=float)
    if y.ndim==1:
        y=y[:,None]
    n=len(y)
    if n<6:
        return raw_current_jet_bank(y)
    # Solve the monomial moment identities once, rather than fitting a model
    # to the evaluation targets. Reflection supplies the right endpoint.
    v=np.vander(np.arange(6,dtype=float),6,increasing=True)
    inv=np.linalg.inv(v)
    left=[inv[1]@y[:6],2*inv[2]@y[:6]]
    right=[-inv[1]@y[-6:][::-1],2*inv[2]@y[-6:][::-1]]
    spl=make_interp_spline(np.arange(n),y,k=5,
                          bc_type=([(1,left[0]),(2,left[1])],
                                   [(1,right[0]),(2,right[1])]))
    m=spl(np.arange(n),1)
    q=spl(np.arange(n),2)
    delta=np.diff(y,axis=0)
    a=np.stack((m[:-1]/5,m[:-1]/5+q[:-1]/20,
                delta-.4*(m[:-1]+m[1:])+(q[1:]-q[:-1])/20,
                m[1:]/5-q[1:]/20,m[1:]/5),axis=1)
    return a,delta


def admitted(source, proposal='compatible', metric='slope', functional=True,
             ledger='compact', bounded=False,envelope=False):
    y=np.asarray(source,dtype=float)
    if y.ndim==1:
        y=y[:,None]
    compact,delta=raw_current_jet_bank(y)
    a=compatible_proposal(y)[0] if proposal=='compatible' else compact
    signs=ordered_sign_ledger(compact if ledger=='compact' else a,delta)
    c=project_signed_fibres(a,signs,delta)
    w=SLOPE_GRAM if metric=='slope' else VALUE_GRAM
    if functional:
        for i in range(len(delta)):
            for channel in range(y.shape[1]):
                s=signs[i,:,channel]
                if s[0]!=0 and np.all(s==s[0]):
                    c[i,:,channel]=admit_cell(a[i,:,channel],delta[i,channel],s[0],w,True)
    if bounded or envelope:
        c=support_range_admission(y,c,signs,envelope=envelope)
    return c,a,signs


def joint_admitted(source, continuity=1):
    """Joint W1 projection with shared derivative traces, numerical oracle.

    Uniform cells use continuous positivity; mixed cells retain the existing
    coefficient ledger. A feasibility failure is a reported outcome.
    continuity=1 shares U'; continuity=2 also shares U''.
    """
    y=np.asarray(source,dtype=float)
    c,a,signs=admitted(y)
    count=len(y)-1
    a=a[:,:,0]
    scale=max(np.max(abs(a)),1e-30)
    target=a.reshape(-1)/scale
    w=np.kron(np.eye(count),SLOPE_GRAM)
    equal=[]
    rhs=[]
    for i in range(count):
        row=np.zeros(5*count);row[5*i:5*i+5]=1
        equal.append(row);rhs.append((y[i+1]-y[i])/scale)
    for i in range(count-1):
        row=np.zeros(5*count);row[5*i+4]=1;row[5*(i+1)]=-1
        equal.append(row);rhs.append(0)
        if continuity>=2:
            row=np.zeros(5*count)
            row[5*i+4]=1;row[5*i+3]=-1
            row[5*(i+1)+1]=-1;row[5*(i+1)]=1
            equal.append(row);rhs.append(0)
    eq=np.array(equal);rhs=np.array(rhs)
    rows=[];uniform=[]
    for i in range(count):
        s=signs[i,:,0]
        if s[0]!=0 and np.all(s==s[0]):
            uniform.append((i,s[0]))
            local=s[0]*basis(np.linspace(0,1,9))
        else:
            local=np.diag(s)
        for b in local:
            row=np.zeros(5*count);row[5*i:5*i+5]=b;rows.append(row)
    rows=np.array(rows)
    if (np.min(rows@target)>=-1e-12 and
        all(minimum(s*target[5*i:5*i+5])[0]>=-1e-12 for i,s in uniform)):
        return a[:,:,None]
    feasible=linprog(np.zeros(5*count),A_ub=-rows,b_ub=np.zeros(len(rows)),
                     A_eq=eq,b_eq=rhs,bounds=[(None,None)]*(5*count),method='highs')
    if not feasible.success:
        raise ValueError('shared traces incompatible with the fixed mixed-cell ledger')
    z=feasible.x
    for _ in range(50):
        result=minimize(lambda x:.5*(x-target)@w@(x-target),z,
                        jac=lambda x:w@(x-target),method='SLSQP',
                        constraints=[{'type':'eq','fun':lambda x:eq@x-rhs,'jac':lambda x:eq},
                                     {'type':'ineq','fun':lambda x:rows@x,'jac':lambda x:rows}],
                        options={'ftol':1e-11,'maxiter':500})
        if not result.success:
            raise RuntimeError(result.message)
        z=result.x
        extra=[]
        for i,s in uniform:
            low,u=minimum(s*z[5*i:5*i+5])
            if low < -1e-9:
                row=np.zeros(5*count);row[5*i:5*i+5]=s*basis(u);extra.append(row)
        if not extra:
            return (z.reshape(count,5,1)*scale)
        rows=np.vstack((rows,extra))
    raise RuntimeError('joint continuous cone did not converge')


def cases(n,validation=False):
    phases=(.07,.27,.63,.91) if validation else (.1,.35,.5,.8)
    freqs=(.045,.13,.285,.37,.46) if validation else (.08,.22,.4)
    widths=(.22,.5,1.,2.2,4.) if validation else (.35,.7,1.5,3.)
    for p in phases:
        center=(n-1)/2+p
        for width in widths:
            yield 'sigmoid',f'w={width},p={p}',lambda z,c=center,w=width:.5*(1+np.tanh((z-c)/w))
        for eps in (0,.001,.02):
            yield 'shoulder',f'e={eps},p={p}',lambda z,c=center,e=eps:(z-c)**3/3+e*(z-c)
        for freq in freqs:
            yield 'sine',f'f={freq},p={p}',lambda z,f=freq,p=p:np.sin(2*np.pi*(f*z+p))
        yield 'step',f'p={p}',lambda z,c=center:(z>=c).astype(float)
        yield 'box',f'p={p}',lambda z,c=center:((z>=c-3)&(z<c+3)).astype(float)
        yield 'chirp',f'p={p}',lambda z,p=p:np.sin(2*np.pi*(.03*z+.19*z*z/(n-1)+p))
        yield 'mixed',f'p={p}',lambda z,c=center,p=p:.5*(1+np.tanh((z-c)/.65))+.07*np.sin(2*np.pi*(.3*z+p))


def variants(y, joint=False):
    raw,delta=raw_current_jet_bank(y[:,None])
    signs=ordered_sign_ledger(raw,delta)
    output={'CONV':project_signed_fibres(raw,signs,delta)}
    output['compact functional']=admitted(y,proposal='compact')[0]
    output['compatible raw']=compatible_proposal(y)[0]
    output['compatible coefficient']=admitted(y,functional=False)[0]
    output['compatible functional']=admitted(y)[0]
    output['compatible bounded']=admitted(y,bounded=True)[0]
    output['compatible envelope']=admitted(y,envelope=True)[0]
    output['compatible value']=admitted(y,metric='value')[0]
    output['compatible own ledger']=admitted(y,ledger='proposal')[0]
    if joint:
        for k in (1,2):
            try:
                output[f'compatible C{k}']=joint_admitted(y,k)
            except (ValueError,RuntimeError) as exc:
                output[f'compatible C{k}']=str(exc)
    return output


def run(out,validation=False,joint=False):
    n,scale=(33,32) if validation else (25,32)
    x=np.linspace(0,n-1,(n-1)*scale+1)
    nodes=np.arange(n,dtype=float)
    records=[];plots={}
    for family,label,f in cases(n,validation):
        y=f(nodes);truth=f(x)
        start=time.perf_counter()
        currents=variants(y,joint)
        mask=(x>4)&(x<n-5)
        if family in ('shoulder','sigmoid','step'):
            mask&=(x>(n-1)/2-3)&(x<(n-1)/2+4)
        norm=max(np.ptp(truth[mask])**2,1e-25)
        estimates={}
        for name,c in currents.items():
            if isinstance(c,str):
                records.append({'family':family,'case':label,'method':name,'error':c})
                continue
            estimates[name]=synthesize_polyphase(y[:,None],c,scale)[:,0]
        estimates['Lanczos-3']=lanczos(y,x)
        for name,z in estimates.items():
            dz=np.diff(z)
            sig=np.sign(dz[abs(dz)>1e-9*max(1,np.max(abs(dz)))])
            # Use the exact secant ledger used by the operator. Deleting tiny
            # but nonzero source secants would audit a different sign ledger.
            dy=np.diff(y);coarse=np.sign(dy[dy!=0])
            row={'family':family,'case':label,'method':name,
                 'nmse':float(np.mean((z[mask]-truth[mask])**2)/norm),
                 'full_nmse':float(np.mean((z-truth)**2)/max(np.ptp(truth)**2,1e-25)),
                 'excursion':float(max(0,z.max()-y.max(),y.min()-z.min())),
                 'extra_turns':int(np.sum(sig[1:]!=sig[:-1])-np.sum(coarse[1:]!=coarse[:-1]))}
            if name in currents:
                c=currents[name]
                row['slope_jump']=float(np.max(abs(5*(c[:-1,4,0]-c[1:,0,0]))))
            if family not in ('step','box'):
                mid=.5*(x[:-1]+x[1:])
                slope_mask=mask[:-1]&mask[1:]
                h=1e-5
                ts=(f(mid+h)-f(mid-h))/(2*h)
                if name in currents:
                    ds=(5*np.einsum('ij,pj->ip',currents[name][:,:,0],
                                     basis((np.arange(scale)+.5)/scale))).reshape(-1)
                else:
                    ds=(lanczos(y,mid+h)-lanczos(y,mid-h))/(2*h)
                row['slope_nmse']=float(np.mean((ds[slope_mask]-ts[slope_mask])**2)/norm)
            records.append(row)
        if label.endswith('p=0.5') and family in ('sigmoid','sine','step'):
            plots[family+' '+label]={'truth':truth.tolist(),'source':y.tolist(),
                                      'estimates':{k:v.tolist() for k,v in estimates.items()}}
        print(f'{family} {label}: {time.perf_counter()-start:.2f}s',flush=True)
    methods=sorted(set(r['method'] for r in records))
    summary={}
    for family in sorted(set(r['family'] for r in records)):
        summary[family]={}
        for method in methods:
            rs=[r for r in records if r['family']==family and r['method']==method]
            good=[r for r in rs if 'error' not in r]
            summary[family][method]={'failures':len(rs)-len(good),
                'mean_nmse':float(np.mean([r['nmse'] for r in good])) if good else None,
                'max_extra_turns':max([r['extra_turns'] for r in good],default=None),
                'max_excursion':max([r['excursion'] for r in good],default=None)}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'n':n,'scale':scale,'validation':validation,
                               'summary':summary,'records':records,'x':x.tolist(),'plots':plots},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def joint_probe(out):
    n=17;scale=32;x=np.linspace(0,n-1,(n-1)*scale+1)
    selected=[('sigmoid',lambda z:.5*(1+np.tanh((z-8.3)/.9))),
              ('sine',lambda z:np.sin(2*np.pi*(.3*z+.17))),
              ('shoulder',lambda z:(z-8.5)**3/3+.001*(z-8.5)),
              ('step',lambda z:(z>=8.3).astype(float)),
              ('box',lambda z:((z>=5.3)&(z<11.3)).astype(float))]
    rows=[]
    for family,f in selected:
        y=f(np.arange(n));truth=f(x)
        mask=(x>4)&(x<12)
        for continuity in (0,1,2):
            start=time.perf_counter()
            try:
                c=admitted(y)[0] if continuity==0 else joint_admitted(y,continuity)
                z=synthesize_polyphase(y[:,None],c,scale)[:,0]
                row={'family':family,'continuity':continuity,
                     'mse':float(np.mean((z[mask]-truth[mask])**2)),
                     'slope_jump':float(np.max(abs(5*(c[:-1,4,0]-c[1:,0,0])))),
                     'curvature_jump':float(np.max(abs(20*(c[:-1,4,0]-c[:-1,3,0]-c[1:,1,0]+c[1:,0,0]))))}
            except (RuntimeError,ValueError) as e:
                row={'family':family,'continuity':continuity,'error':str(e)}
            row['seconds']=time.perf_counter()-start
            rows.append(row);print(json.dumps(row),flush=True)
    Path(out).write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='/tmp/conv_compatible_current.json')
    p.add_argument('--validation',action='store_true');p.add_argument('--joint',action='store_true')
    p.add_argument('--joint-probe',action='store_true')
    a=p.parse_args()
    if a.joint_probe:
        joint_probe(a.out)
    else:
        run(a.out,a.validation,a.joint)
