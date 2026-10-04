"""Ablation of the joint atlas's added support-range policy. No sharpening."""
import json,math,time,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from experiments.conv_synthetic_evidence import _scene,_two_dimensional_cases
from experiments.conv_warp.joint_reference import (canonical_sampled_control_net,_global_lattice_from_atlas,
 _q1_global_lattice,_shared_support_bounds,_gradient_cone_constraints,_finite_capacity_admission)
from experiments.conv_warp.bounded_sharpening.study import admit,half_controls


def patches(P):
 n=(P.shape[0]-1)//5
 return np.array([[P[y*5:y*5+6,x*5:x*5+6] for x in range(n)] for y in range(n)])


def evaluation(P,side=129):
 n=(P.shape[0]-1)//5;x=np.linspace(0,n,side);cell=np.minimum(x.astype(int),n-1);u=x-cell
 W=np.zeros((side,6*n))
 for k in range(6):W[np.arange(side),6*cell+k]=math.comb(5,k)*u**k*(1-u)**(5-k)
 C=patches(P).transpose(0,2,1,3).reshape(n*6,n*6)
 return W@C@W.T


def constraints(source):
 n=source.shape[0]-1;gw=5*n+1;listed=[];rows=[];cols=[];data=[];lo=[];hi=[]
 for y in range(n):
  for x in range(n):
   support=source[max(0,y-2):y+4,max(0,x-2):x+4];lo.append(support.min());hi.append(support.max())
   for C in _gradient_cone_constraints(source[...,None],y,x,0,support=True):
    yy,xx=np.nonzero(C);ix=(5*y+yy)*gw+5*x+xx;r=len(listed);v=C[yy,xx]
    listed.append((ix,v));rows.extend([r]*len(ix));cols.extend(ix);data.extend(v)
 A=sparse.csr_matrix((data,(rows,cols)),shape=(len(listed),gw*gw))
 # Extraction into cell-local Bernstein nets, then exact half-cell subdivision.
 rr=[];cc=[]
 for y in range(n):
  for x in range(n):
   for j in range(6):
    for i in range(6):rr.append(((y*n+x)*6+j)*6+i);cc.append((5*y+j)*gw+5*x+i)
 extraction=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(36*n*n,gw*gw))
 T=sparse.kron(half_controls(5),half_controls(5),format='csr')
 refined=sparse.kron(sparse.eye(n*n),T,format='csr')@extraction
 lower=np.repeat(lo,144);upper=np.repeat(hi,144)
 return listed,A,refined,lower,upper,np.array(lo).reshape(n,n),np.array(hi).reshape(n,n)


def features():
 for n in (17,33):
  for i,p in enumerate(_two_dimensional_cases(n)):
   yield f"{p['kind']}-{i}-{n}",n,'smooth',p
  for angle in (0.,22.5,45.):
   for kind in ('step','thin-bar'):
    yield f'{kind}-{angle}-{n}',n,'sharp',dict(kind=kind,angle=angle,offset=.23/(n-1),phase=0.)


def analytic(p,side,n):
 if p['kind'] in ('edge','carrier','curved','crossing'):
  return _scene(p['kind'],side,**{k:v for k,v in p.items() if k!='kind'})
 y,x=np.mgrid[:side,:side]/(side-1);a=math.radians(p['angle']);t=(x-.5)*math.cos(a)+(y-.5)*math.sin(a)-p['offset']
 value=(t>=0).astype(float) if p['kind']=='step' else (abs(t)<.7/(n-1)).astype(float)
 return p.get('low',0.)+(p.get('high',1.)-p.get('low',0.))*value


def run(out,low_contrast=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);records=[];images={};start=time.perf_counter()
 cases=list(features())
 if low_contrast:
  cases=[(name+f'-contrast{high-low}',n,family,{**p,'low':low,'high':high}) for name,n,family,p in cases if family=='sharp' for low,high in ((.3,.7),(.48,.52))]
 for name,n,family,p in cases:
  source=analytic(p,n,n);truth=analytic(p,129,n)
  raw=_global_lattice_from_atlas(canonical_sampled_control_net(source[...,None]))[...,0];raw[::5,::5]=source
  base=_q1_global_lattice(source)[...,0];lower,upper=_shared_support_bounds(source);lower=lower[...,0];upper=upper[...,0]
  listed,A,R,rl,rh,cl,ch=constraints(source)
  clipped=np.clip(raw,lower,upper)
  nominal,_=_finite_capacity_admission(base,clipped,listed)
  current,cd=_finite_capacity_admission(base,raw,listed)
  physical,_=_finite_capacity_admission(base,np.clip(raw,0,1),listed)
  AR=sparse.vstack([A,R,-R],format='csr');b=np.r_[np.zeros(A.shape[0]),rl,-rh]
  # Preserve any candidate already passing every refined constraint.
  if np.min(AR@raw.ravel()-b)>=-2e-11:refined=raw.copy()
  else:refined,_=admit(base,raw-base,AR,b)
  nets={'raw':raw,'clipped':clipped,'nominal':nominal,'current-only':current,'physical-box':physical,'refined-support':refined}
  q=np.minimum((np.linspace(0,n-1,129)).astype(int),n-2);L=cl[q[:,None],q[None,:]];H=ch[q[:,None],q[None,:]]
  impossible=truth-np.clip(truth,L,H)
  common={'case':name,'sourceSide':n,'family':family,'parameters':p,'truthSupportViolationMax':float(np.max(abs(impossible))),
   'truthSupportViolationMseFloor':float(np.mean(impossible**2)),
   'clippedControlFraction':float(np.mean(raw!=clipped)),'currentOnlyChangeMax':float(np.max(abs(current-raw))),
   'nominalVsClippedMax':float(np.max(abs(nominal-clipped)))}
  mask=np.ones((129,129),bool);border=2*128//(n-1);mask[:border]=mask[-border:]=False;mask[:,:border]=mask[:,-border:]=False
  for method,P in nets.items():
   image=evaluation(P);delta=image-truth
   row={**common,'method':method,'mse':float(np.mean(delta**2)),'interiorMse':float(np.mean(delta[mask]**2)),
    'physicalExcursion':float(max(0,-image.min(),image.max()-1)),
    'sampleRangeExcursion':float(max(0,source.min()-image.min(),image.max()-source.max())),
    'localSupportExcursion':float(np.max(abs(image-np.clip(image,L,H)))),
    'cardinality':float(np.max(abs(P[::5,::5]-source))),
    'mass':float(patches(P).sum()/36/(n-1)**2),
    'currentMargin':float(np.min(A@P.ravel())) if A.shape[0] else 0.,
    'refinedRangeMargin':float(min(np.min(R@P.ravel()-rl),np.min(rh-R@P.ravel())))}
   if row['cardinality']>1e-12:raise AssertionError(row)
   if method not in ('raw','clipped') and row['currentMargin']< -2e-9:raise AssertionError(row)
   if method in ('nominal','refined-support') and row['localSupportExcursion']>2e-9:raise AssertionError(row)
   if method=='physical-box' and row['physicalExcursion']>2e-9:raise AssertionError(row)
   records.append(row)
   if n==17 and ((p['kind']=='carrier' and p['angle']==37.5 and p['phase']>0) or (p['kind']=='curved' and p['offset']>0) or (family=='sharp' and p['angle']==22.5)):
    images[f'{name}/{method}']=image;images[f'{name}/truth']=truth
  print(json.dumps({'case':name,'seconds':time.perf_counter()-start,'floor':common['truthSupportViolationMseFloor']}),flush=True)
 result={'scope':f'{len(cases)} declared cases; source sides 17 and 33; target 129. Nodal source data. No concentration or changes to production.','lowContrast':low_contrast,'methods':list(nets),'sourceSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':records}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'images.npz',**images)

if __name__=='__main__':
 import sys
 run(sys.argv[1],low_contrast='--low-contrast' in sys.argv)
