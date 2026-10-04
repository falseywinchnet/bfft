"""Finite margin-limited sharpening of an already admitted joint atlas.
No solve or optimization. Diagnostic; not promoted to the image engine.
"""
import json,math,time,platform,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from experiments.conv_warp.joint_reference import (finite_joint_control_nets,
 _global_lattice_from_atlas,_topological_control_groups,evaluate_joint_atlas,
 _shared_support_bounds,_gradient_cone_constraints)


def constraints(source):
    h,w=source.shape;gw=5*(w-1)+1;gh=5*(h-1)+1
    lo,hi=_shared_support_bounds(source[...,None]);lo=lo[...,0].ravel();hi=hi[...,0].ravel()
    rows=list(range(gw*gh))+list(range(gw*gh,2*gw*gh));cols=list(range(gw*gh))*2
    data=[1.]*(gw*gh)+[-1.]*(gw*gh);lower=list(lo)+list(-hi)
    for y in range(h-1):
        for x in range(w-1):
            for coeff in _gradient_cone_constraints(source[...,None],y,x,0,support=True):
                yy,xx=np.nonzero(coeff);idx=(5*y+yy)*gw+5*x+xx;r=len(lower)
                rows.extend([r]*len(idx));cols.extend(idx);data.extend(coeff[yy,xx]);lower.append(0.)
    return sparse.csr_matrix((data,(rows,cols)),shape=(len(lower),gw*gh)),np.array(lower)


def proposal(P,amount=1.):
    # Shared control-lattice inverse diffusion. The factor 1/4 is a declared
    # one-step proposal scale, not an admissibility tolerance.
    D=np.zeros_like(P);D[1:-1,1:-1]=P[1:-1,1:-1]-.25*(P[1:-1,:-2]+P[1:-1,2:]+P[:-2,1:-1]+P[2:,1:-1])
    D[0,1:-1]=.5*P[0,1:-1]-.25*(P[0,:-2]+P[0,2:]);D[-1,1:-1]=.5*P[-1,1:-1]-.25*(P[-1,:-2]+P[-1,2:])
    D[1:-1,0]=.5*P[1:-1,0]-.25*(P[:-2,0]+P[2:,0]);D[1:-1,-1]=.5*P[1:-1,-1]-.25*(P[:-2,-1]+P[2:,-1])
    D[::5,::5]=0
    return amount*D


def concentration(P,source,amount=1.):
    # A C2 tensor-quintic corner profile, weighted by local source curvature.
    # Weights are shared nodal values interpolated onto the control lattice.
    h,w=source.shape
    gx=np.gradient(source,axis=1);gy=np.gradient(source,axis=0)
    kx=np.zeros_like(source);ky=np.zeros_like(source)
    kx[:,1:-1]=np.diff(source,n=2,axis=1);ky[1:-1]=np.diff(source,n=2,axis=0)
    kx[:,0]=kx[:,1];kx[:,-1]=kx[:,-2];ky[0]=ky[1];ky[-1]=ky[-2]
    K=kx*kx+ky*ky;G=gx*gx+gy*gy
    W=np.divide(K,K+G,out=np.zeros_like(K),where=K+G>1e-28)
    D=np.zeros_like(P)
    for y in range(h-1):
      for x in range(w-1):
        for j in range(6):
          for i in range(6):
            u=i/5;v=j/5
            weight=(1-u)*(1-v)*W[y,x]+u*(1-v)*W[y,x+1]+(1-u)*v*W[y+1,x]+u*v*W[y+1,x+1]
            target=source[y+(j>=3),x+(i>=3)]
            D[5*y+j,5*x+i]=weight*(target-P[5*y+j,5*x+i])
    D[::5,::5]=0
    return amount*D


def admit(P,D,A,b):
    # Small roundoff in the baseline is recorded; no baseline repair.
    flat=P.ravel();d=D.ravel();m=A@flat-b
    if m.min() < -2e-10:raise ValueError('Input is not an admitted baseline')
    h=(P.shape[0]-1)//5+1;w=(P.shape[1]-1)//5+1
    groups,owner=_topological_control_groups(h,w);active=owner>=0
    group=sparse.coo_matrix((np.ones(active.sum()),(np.flatnonzero(active),owner[active])),shape=(len(d),len(groups))).tocsr()
    response=A.multiply(d)@group;harm=response.copy();harm.data=np.maximum(-harm.data,0);harm.eliminate_zeros()
    H=np.asarray(harm.sum(axis=1)).ravel();ratio=np.ones(len(H));nz=H>1e-14
    ratio[nz]=np.minimum(1,np.maximum(m[nz],0)/H[nz]);caps=np.ones(len(groups));coo=harm.tocoo();np.minimum.at(caps,coo.col,ratio[coo.row])
    alpha=np.zeros_like(d);alpha[active]=caps[owner[active]];z=flat+alpha*d
    remaining=flat+d-z;motion=A@remaining;slack=A@z-b;negative=motion < -1e-14
    tau=min(1,float(np.min(np.maximum(slack[negative],0)/(-motion[negative])))) if negative.any() else 1.
    z+=tau*remaining
    return z.reshape(P.shape),{'baselineMargin':float(m.min()),'finalMargin':float((A@z-b).min()),'completion':tau,'meanCapacity':float(caps.mean()),'relativeMotion':float(np.linalg.norm(z-flat)/max(np.linalg.norm(d),1e-30))}


def atlas(P):
    h=(P.shape[0]-1)//5;w=(P.shape[1]-1)//5
    return np.array([[P[y*5:y*5+6,x*5:x*5+6] for x in range(w)] for y in range(h)])


def derivative_rows(P,orders=((1,0),(0,1))):
    """Unit-cell physical derivative Bernstein rows; no sampling certificate."""
    gh,gw=P.shape;rows=[];cols=[];data=[];r=0
    for y in range((gh-1)//5):
      for x in range((gw-1)//5):
       for a,b in orders:
        factor=math.factorial(5)/math.factorial(5-a)*math.factorial(5)/math.factorial(5-b)
        for j in range(6-b):
         for i in range(6-a):
          for v in range(b+1):
           for u in range(a+1):
            rows.append(r);cols.append((5*y+j+v)*gw+5*x+i+u)
            data.append(factor*(-1)**(a+b-u-v)*math.comb(a,u)*math.comb(b,v))
          r+=1
    return sparse.csr_matrix((data,(rows,cols)),shape=(r,gh*gw))


def half_controls(degree):
    """Exact de Casteljau coefficient map to two half-intervals."""
    left=np.zeros((degree+1,degree+1));right=np.zeros_like(left)
    for i in range(degree+1):
        for k in range(i+1):left[i,k]=math.comb(i,k)/2**i
        for k in range(i,degree+1):right[i,k]=math.comb(degree-i,k-i)/2**(degree-i)
    return sparse.csr_matrix(np.vstack([left,right]))


def refined_derivative_rows(P):
    cells=((P.shape[0]-1)//5)*((P.shape[1]-1)//5);parts=[]
    for a,b in ((1,0),(0,1)):
        transform=sparse.kron(half_controls(5-b),half_controls(5-a),format='csr')
        parts.append(sparse.kron(sparse.eye(cells),transform,format='csr')@derivative_rows(P,((a,b),)))
    return sparse.vstack(parts,format='csr')


def run(out,kind='concentration',derivative_cap=None,refine_derivative=False):
    n=9;yy,xx=np.mgrid[:n,:n];rng=np.random.default_rng(731)
    cases={'constant':np.full((n,n),.4),'affine':.15+.04*xx+.035*yy,
      'step':(xx>=4).astype(float),'thin-line':np.maximum(0,np.minimum(xx+.5,5.085)-np.maximum(xx-.5,4.085)),
      'diagonal':((xx+.7*yy)>=6.3).astype(float),
      'crossing':np.maximum((np.abs(xx-4)<.6).astype(float),(np.abs(yy-4)<.6).astype(float)),
      'smooth-wave':.5+.3*np.sin(.7*xx+.4*yy),'noise':rng.uniform(.1,.9,(n,n))}
    rows=[]
    for name,source in cases.items():
        start=time.perf_counter();control,base_diag=finite_joint_control_nets(source)
        P=_global_lattice_from_atlas(control[...,None])[...,0];A,b=constraints(source)
        derivative=derivative_rows(P);baseline_derivative=float(np.max(abs(derivative@P.ravel())))
        if refine_derivative:derivative=refined_derivative_rows(P)
        if derivative_cap is not None:
            bound=derivative_cap*baseline_derivative
            A=sparse.vstack([A,derivative,-derivative],format='csr');b=np.concatenate([b,np.full(2*derivative.shape[0],-bound)])
        q=np.linspace(0,n-1,257);Y,X=np.meshgrid(q,q,indexing='ij');base=evaluate_joint_atlas(atlas(P),X,Y)
        base_gradient=max(np.max(abs(np.diff(base,axis=0))),np.max(abs(np.diff(base,axis=1))))
        for strength in [1.,4.,16.]:
            D=concentration(P,source,strength) if kind=='concentration' else proposal(P,strength)
            Q,diag=admit(P,D,A,b);image=evaluate_joint_atlas(atlas(Q),X,Y)
            mass0=float(atlas(P).sum()/36);mass1=float(atlas(Q).sum()/36)
            row={'case':name,'strength':strength,**diag,'maxChange':float(np.max(abs(image-base))),
                 'sourceRecovery':float(np.max(abs(Q[::5,::5]-source))),
                 'globalRangeExcursion':float(max(0,image.max()-source.max(),source.min()-image.min())),
                 'fieldMassChange':mass1-mass0,'peakGradientRatio':float(max(np.max(abs(np.diff(image,axis=0))),np.max(abs(np.diff(image,axis=1))))/base_gradient) if base_gradient>1e-12 else None,
                 'derivativeControlRatio':float(np.max(abs(derivative@Q.ravel()))/baseline_derivative) if baseline_derivative>1e-12 else None,
                 'seconds':time.perf_counter()-start}
            if name in ['step','thin-line']:
                # Middle source row, positive exact quadrature over target pixels.
                nodes,weights=np.polynomial.legendre.leggauss(4)
                vals=[];basevals=[]
                for i in range(1,n-1):
                    x=np.concatenate([i-.25+.25*nodes,i+.25+.25*nodes]);weight=np.tile(weights*.25,2)
                    vals.append(float(evaluate_joint_atlas(atlas(Q),x,np.full_like(x,4))@weight));basevals.append(float(evaluate_joint_atlas(atlas(P),x,np.full_like(x,4))@weight))
                row['areaPeakBefore']=max(basevals);row['areaPeakAfter']=max(vals)
            if diag['finalMargin'] < -2e-10 or row['sourceRecovery']>1e-12 or row['globalRangeExcursion']>2e-10:raise AssertionError(row)
            if name in ['constant','affine'] and row['maxChange']>2e-7:raise AssertionError(row)
            rows.append(row);print(json.dumps(row),flush=True)
    result={'scope':'Canonical Float64 joint admission around native proposal. Finite grouped-margin limiter plus remaining-ray cap. Range and original support-current constraints. Optional first-derivative Bernstein cap; no cell-mass constraints. Strength is a diagnostic proposal parameter. No production change.','proposal':kind,'derivativeCap':derivative_cap,'refinedDerivativeCertificate':refine_derivative,'studySha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runtime':platform.python_version(),'rows':rows}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('out');parser.add_argument('--proposal',choices=['concentration','inverse-diffusion'],default='concentration');parser.add_argument('--derivative-cap',type=float);parser.add_argument('--refine-derivative',action='store_true')
 args=parser.parse_args()
 if args.derivative_cap is not None and args.derivative_cap<1:parser.error('Derivative cap must admit the baseline (at least 1).')
 run(args.out,args.proposal,args.derivative_cap,args.refine_derivative)
