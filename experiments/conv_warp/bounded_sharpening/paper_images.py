"""Declared paper-plate fields, finite concentration, and exact atlas pixels."""
import json, math, hashlib, platform, time
from pathlib import Path
import numpy as np
from scipy import sparse
from PIL import Image
from .study import (finite_joint_control_nets,_global_lattice_from_atlas,constraints,
 derivative_rows,refined_derivative_rows,concentration,admit,atlas)
from experiments.conv_synthetic_evidence import _scene
from experiments.conv_distilled_core import distilled_conv_synthesis

CASES={
 'edge':dict(angle=37.5,phase=0.,offset=.25/16),
 'carrier':dict(angle=37.5,phase=math.pi/2,offset=0.),
 'curved':dict(angle=0.,phase=0.,offset=.25/16),
 'crossing':dict(angle=0.,phase=math.pi/2,offset=0.)}


def truth(kind,x,y):
 p=CASES[kind];angle=math.radians(p['angle']);normal=(x-.5)*math.cos(angle)+(y-.5)*math.sin(angle)
 if kind=='edge':return .5+.43*np.tanh((normal-p['offset'])/.035)
 if kind=='carrier':return .5+.34*np.sin(math.pi*x)**2*np.sin(math.pi*y)**2*np.sin(2*math.pi*3.25*(x*math.cos(angle)+y*math.sin(angle))+p['phase'])
 if kind=='curved':return .5+.42*np.tanh((np.hypot(x-.47,y-.53)-.27-p['offset'])/.032)
 return .5+.19*(np.sin(2*math.pi*(2.75*x+.9*y)+p['phase'])+np.sin(2*math.pi*(-.9*x+2.75*y)-p['phase']))


def basis(x,n=17):
 x=np.asarray(x);cell=np.minimum(np.floor(x).astype(int),n-2);u=x-cell
 W=np.zeros((x.size,(n-1)*6))
 for k in range(6):W[np.arange(x.size),6*cell+k]=math.comb(5,k)*u**k*(1-u)**(5-k)
 return W


def flat_controls(P):
 C=atlas(P);return C.transpose(0,2,1,3).reshape(P.shape[0]//5*6,P.shape[1]//5*6)


def point(P,side):
 W=basis(np.linspace(0,16,side));return W@flat_controls(P)@W.T


def quadrature(side,order=8):
 centres=np.linspace(0,16,side);edges=np.r_[0,(centres[:-1]+centres[1:])/2,16]
 nodes,weights=np.polynomial.legendre.leggauss(order);xx=[];ww=[];owner=[]
 for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
  knots=np.r_[a,np.arange(math.floor(a)+1,math.ceil(b)),b]
  for l,r in zip(knots[:-1],knots[1:]):
   xx.extend((l+r)/2+(r-l)/2*nodes);ww.extend((r-l)/2/(b-a)*weights);owner.extend([i]*order)
 return np.array(xx),np.array(ww),np.array(owner),np.diff(edges)


def area_weights(side,fn,order=4):
 x,w,owner,_=quadrature(side,order);W=fn(x);result=np.zeros((side,W.shape[1]));np.add.at(result,owner,w[:,None]*W);return result


def area(P,side):
 W=area_weights(side,basis);return W@flat_controls(P)@W.T


def lanczos_weights(x):
 d=np.asarray(x)[:,None]-np.arange(17)[None,:];W=np.where(abs(d)<3,np.sinc(d)*np.sinc(d/3),0.);return W/W.sum(axis=1)[:,None]


def true_area(kind,side,order=8):
 x,w,owner,_=quadrature(side,order)
 # Tensor quadrature summed through a sparse positive aggregation matrix.
 M=sparse.csr_matrix((w,(owner,np.arange(len(x)))),shape=(side,len(x)))
 return np.asarray((M@truth(kind,x[None,:]/16,x[:,None]/16))@M.T)


def metrics(image,reference,source):
 error=image-reference;mask=np.ones(image.shape,bool);mask[:4]=False;mask[-4:]=False;mask[:,:4]=False;mask[:,-4:]=False
 if image.shape==(33,33):mask[::2,::2]=False
 tv=lambda a:float(np.sum(abs(np.diff(a,axis=0)))+np.sum(abs(np.diff(a,axis=1))))
 return {'mse':float(np.mean(error**2)),'interiorHeldOutMse':float(np.mean(error[mask]**2)),
 'maxError':float(np.max(abs(error))),'sourceRangeExcursion':float(max(0,image.max()-source.max(),source.min()-image.min())),
 'tvRatio':tv(image)/tv(reference)}


def normal_metrics(kind,P):
 if kind not in ('edge','curved'):return {}
 t=np.linspace(-.2,.2,1601)
 if kind=='edge':
  angle=math.radians(CASES[kind]['angle']);q=t+CASES[kind]['offset'];x=.5+q*math.cos(angle);y=.5+q*math.sin(angle);lo,hi=.07,.93
 else:x=.47+.27+CASES[kind]['offset']+t;y=np.full_like(x,.53);lo,hi=.08,.92
 wx=basis(x*16);wy=basis(y*16);v=np.einsum('ni,ij,nj->n',wy,flat_controls(P),wx,optimize=True)
 def crossing(level):
  ids=np.where((v[:-1]<=level)&(v[1:]>level))[0]
  if len(ids)!=1:return None
  i=ids[0];return float(t[i]+(t[i+1]-t[i])*(level-v[i])/(v[i+1]-v[i]))
 left=crossing(lo+.1*(hi-lo));right=crossing(lo+.9*(hi-lo));mid=crossing(.5)
 return {'normalWidth10to90':None if left is None or right is None else right-left,'normalHalfLevelShift':mid,
 'normalReverseVariation':float(np.maximum(-np.diff(v),0).sum()),'normalPeakSlope':float(np.max(np.diff(v)/np.diff(t)))}


def save_image(path,a):
 Image.fromarray(np.uint8(np.round(np.clip(a,0,1)*255))).save(path)


def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);records=[];allimages={}
 settings=[('cap1',1.,1.),('cap2',2.,1.),('cap4',4.,1.),('cap2-strong',2.,4.)]
 for kind in CASES:
  start=time.perf_counter();source=_scene(kind,17,**CASES[kind]);base,_=finite_joint_control_nets(source)
  P=_global_lattice_from_atlas(base[...,None])[...,0];A,b=constraints(source);D=refined_derivative_rows(P)
  bound0=float(np.max(abs(derivative_rows(P)@P.ravel())));nets={'joint-base':(P,{})}
  for name,cap,strength in settings:
   AA=sparse.vstack([A,D,-D],format='csr');bb=np.r_[b,np.full(2*D.shape[0],-cap*bound0)]
   Q,diag=admit(P,concentration(P,source,strength),AA,bb)
   diag.update({'derivativeCertificateRatio':float(np.max(abs(D@Q.ravel()))/bound0),'cap':cap,'strength':strength})
   if diag['finalMargin']< -2e-10:raise AssertionError(diag)
   nets[name]=(Q,diag)
  truths={side:_scene(kind,side,**CASES[kind]) for side in (33,513)};ta={side:true_area(kind,side) for side in (33,65)}
  refine_error=float(np.max(abs(ta[33]-true_area(kind,33,12))))
  allimages[kind]={'truth':truths[513],'truth-area':ta[65]};save_image(out/f'{kind}-truth.png',truths[513]);save_image(out/f'{kind}-truth-area.png',ta[65])
  save_image(out/f'{kind}-source.png',source)
  for name,(Q,diag) in nets.items():
   display=point(Q,513);small=point(Q,33);a33=area(Q,33);a65=area(Q,65)
   mass=float(flat_controls(Q).sum()/36);basemass=float(flat_controls(P).sum()/36)
   row={'field':kind,'method':name,**diag,'point':metrics(small,truths[33],source),'area':metrics(a33,ta[33],source),
    'sourceRecovery':float(np.max(abs(Q[::5,::5]-source))),'atlasMassChange':mass-basemass,
    'atlasMeanChange':(mass-basemass)/256,'truthQuadratureRefinement':refine_error,**normal_metrics(kind,Q)}
   if row['sourceRecovery']>1e-12:raise AssertionError(row)
   records.append(row);allimages[kind][name]=display;allimages[kind][name+'-area']=a65
   save_image(out/f'{kind}-{name}.png',display);save_image(out/f'{kind}-{name}-area.png',a65)
   # Fixed error scale, neutral gray = zero; red positive, blue negative.
   e=np.clip((display-truths[513])/.10,-1,1);rgb=np.ones(e.shape+(3,))*.95
   rgb[:,:,0]-=.75*np.maximum(-e,0);rgb[:,:,2]-=.75*np.maximum(e,0);rgb[:,:,1]-=.75*abs(e)
   Image.fromarray(np.uint8(rgb*255)).save(out/f'{kind}-{name}-error.png')
  for name in ('paper-convstar','lanczos3'):
   if name=='paper-convstar':
    display=np.asarray(distilled_conv_synthesis(source,(513,513)));small=np.asarray(distilled_conv_synthesis(source,(33,33)))
   else:
    w=lanczos_weights(np.linspace(0,16,513));display=w@source@w.T;w=lanczos_weights(np.linspace(0,16,33));small=w@source@w.T
   row={'field':kind,'method':name,'point':metrics(small,truths[33],source)}
   if name=='lanczos3':
    for side in (33,65):
     w=area_weights(side,lanczos_weights,8);v=w@source@w.T
     if side==33:
      row['area']=metrics(v,ta[33],source);w12=area_weights(side,lanczos_weights,12);row['quadratureRefinement']=float(np.max(abs(v-w12@source@w12.T)))
     else:save_image(out/f'{kind}-{name}-area.png',v);allimages[kind][name+'-area']=v
   records.append(row);save_image(out/f'{kind}-{name}.png',display);allimages[kind][name]=display
  print(json.dumps({'field':kind,'seconds':time.perf_counter()-start,'rows':records[-7:]}),flush=True)
 packet={'source':'Paper technical_synthetics plate, exact declared four fields and 17x17 nodal samples','displaySide':513,'pointScoreSide':33,'areaScoreSide':33,'areaDisplaySide':65,'sourceParameters':CASES,'runtime':platform.python_version(),'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('study.py'),Path('experiments/conv_synthetic_evidence.py'),Path('experiments/conv_warp/joint_reference.py')]},'rows':records}
 (out/'results.json').write_text(json.dumps(packet,indent=2)+'\n')
 np.savez_compressed(out/'images.npz',**{f'{k}/{m}':v for k,images in allimages.items() for m,v in images.items()})
 return packet

if __name__=='__main__':
 import sys
 run(sys.argv[1])
