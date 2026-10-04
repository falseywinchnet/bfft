"""Isolated degree-six blend retention over the declared paper family.

Float64 factor reference; unchanged source analysis. No range/current
postprocessing, sharpening, or new adaptive analysis is part of either method.
"""
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from experiments.conv_distilled_core import nodal_current_geometry, distilled_conv_synthesis
from experiments.conv_synthetic_evidence import _scene, _two_dimensional_cases
from experiments.conv_warp.joint_reference import line_control_bank
from .faithful_blend import compile_blend


def bern(t, degree):
    t = np.asarray(t)
    return np.array([math.comb(degree,k)*t**k*(1-t)**(degree-k)
                     for k in range(degree+1)]).T


def line_eval(lines, x):
    p = line_control_bank(lines, admitted=True)
    cell = np.minimum(x.astype(int), len(lines)-2)
    return np.einsum('nkl,nk->nl', p[cell], bern(x-cell,5))


def orders(source, x):
    horizontal = line_eval(source.T, x).T
    forward = line_eval(horizontal, x)
    vertical = line_eval(source, x)
    reverse = line_eval(vertical.T, x).T
    return forward, reverse


def weights(x, cells, degree):
    cell = np.minimum(x.astype(int), cells-1)
    w = np.zeros((len(x), cells*(degree+1)))
    for k in range(degree+1):
        w[np.arange(len(x)), cell*(degree+1)+k] = bern(x-cell,degree)[:,k]
    return w


def beta_at(eta, x):
    cell = np.minimum(x.astype(int),len(eta)-2); t = x-cell
    return sum(eta[cell[:,None]+s,cell[None,:]+r]
               *((t if s else 1-t)[:,None])*((t if r else 1-t)[None,:])
               for s in range(2) for r in range(2))


def extract(sample, cells):
    return np.array([[sample[5*y:5*y+6,5*x:5*x+6]
                      for x in range(cells)] for y in range(cells)])


def collocate(sample, cells):
    inverse = np.linalg.inv(bern(np.linspace(0,1,6),5))
    return np.einsum('aj,xyjk,bk->xyab',inverse,extract(sample,cells),inverse)


def compile_all(a,b,eta):
    n=len(a); out=np.zeros((n,n,7,7))
    for s in range(2):
        for r in range(2):
            wy=(np.arange(1,7) if s else np.arange(6,0,-1))/6
            wx=(np.arange(1,7) if r else np.arange(6,0,-1))/6
            blend=eta[s:s+n,r:r+n,None,None]
            out[:,:,s:s+6,r:r+6] += ((1-blend)*a+blend*b)*wy[:,None]*wx[None,:]
    return out


def flatten(p):
    return p.transpose(0,2,1,3).reshape(len(p)*p.shape[-1],-1)


def evaluate(p,x):
    w=weights(x,len(p),p.shape[-1]-1)
    return w@flatten(p)@w.T


def quadrature(cells,side,order):
    axis=np.linspace(0,cells,side);edges=np.r_[0,(axis[:-1]+axis[1:])/2,cells]
    nodes,w=np.polynomial.legendre.leggauss(order); xx=[]; ww=[]; owner=[]
    for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        knots=np.r_[a,np.arange(math.floor(a)+1,math.ceil(b)),b]
        for l,r in zip(knots[:-1],knots[1:]):
            xx.extend((l+r)/2+(r-l)/2*nodes)
            ww.extend((r-l)/2/(b-a)*w);owner.extend([i]*order)
    return np.array(xx),np.array(ww),np.array(owner)


def area(p,side,order=4):
    x,w,owner=quadrature(len(p),side,order)
    basis=weights(x,len(p),p.shape[-1]-1)
    W=np.zeros((side,basis.shape[1]));np.add.at(W,owner,w[:,None]*basis)
    return W@flatten(p)@W.T


def truth_xy(p,x,y):
    theta=math.radians(p['angle']);normal=(x-.5)*math.cos(theta)+(y-.5)*math.sin(theta)
    kind=p['kind']; phase=p['phase'];offset=p['offset']
    if kind=='edge':return .5+.43*np.tanh((normal-offset)/.035)
    if kind=='carrier':return .5+.34*np.sin(math.pi*x)**2*np.sin(math.pi*y)**2*np.sin(2*math.pi*3.25*(x*math.cos(theta)+y*math.sin(theta))+phase)
    if kind=='curved':return .5+.42*np.tanh((np.hypot(x-.47,y-.53)-.27-offset)/.032)
    return .5+.19*(np.sin(2*math.pi*(2.75*x+.9*y)+phase)+np.sin(2*math.pi*(-.9*x+2.75*y)-phase))


def truth_area(p,cells,side,order):
    x,w,owner=quadrature(cells,side,order)
    from scipy import sparse
    W=sparse.csr_matrix((w,(owner,np.arange(len(x)))),shape=(side,len(x)))
    return np.asarray((W@truth_xy(p,x[None,:]/cells,x[:,None]/cells))@W.T)


def errors(a,b):
    d=a-b
    return dict(mse=float(np.mean(d*d)),rms=float(np.sqrt(np.mean(d*d))),max=float(np.max(abs(d))))


def continuous_errors(old,new,p,order):
    n=len(old);nodes,w=np.polynomial.legendre.leggauss(order);t=(nodes+1)/2;w=w/2
    x=(np.arange(n)[:,None]+t[None,:])/n
    truth=truth_xy(p,x[None,None,:,:],x[:,:,None,None]).transpose(0,2,1,3)
    b5,b6=bern(t,5),bern(t,6)
    a=np.einsum('aj,xyjk,bk->xyab',b5,old,b5,optimize=True)
    b=np.einsum('aj,xyjk,bk->xyab',b6,new,b6,optimize=True)
    def mse(e):return float(np.einsum('xyab,a,b->',e*e,w,w)/(n*n))
    return dict(quinticMse=mse(a-truth),retainedMse=mse(b-truth),correctionMse=mse(b-a),maxSampledCorrection=float(np.max(abs(b-a))))


def run(out,point_side=129):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];images={}
    for n in (17,33):
        for index,p in enumerate(_two_dimensional_cases(n)):
            started=time.perf_counter();cells=n-1;name=f"{p['kind']}-{index}-{n}"
            source=_scene(p['kind'],n,**{k:v for k,v in p.items() if k!='kind'})
            eta=nodal_current_geometry(source)[3];xs=np.linspace(0,cells,5*cells+1)
            f,g=orders(source,xs);beta=beta_at(eta,xs)
            old=collocate((1-beta)*f+beta*g,cells)
            a,b=collocate(f,cells),collocate(g,cells);new=compile_all(a,b,eta)
            np.testing.assert_allclose(new[0,0],compile_blend(a[0,0],b[0,0],eta[:2,:2]),atol=1e-14,rtol=0)
            x=np.linspace(0,cells,point_side);f,g=orders(source,x);beta=beta_at(eta,x)
            direct=(1-beta)*f+beta*g;old_image=evaluate(old,x);new_image=evaluate(new,x)
            np.testing.assert_allclose(new_image,(1-beta)*evaluate(a,x)+beta*evaluate(b,x),atol=5e-14,rtol=0)
            truth=_scene(p['kind'],point_side,**{k:v for k,v in p.items() if k!='kind'})
            ta=truth_area(p,cells,65,8);ta12=truth_area(p,cells,65,12)
            old_area,new_area=area(old,65),area(new,65)
            selected=n==17 and ((p['angle']==37.5 and (p['kind']=='edge' and p['offset']>0 or p['kind']=='carrier' and p['phase']>0)) or p['kind']=='curved' and p['offset']>0 or p['kind']=='crossing' and p['phase']>0)
            row=dict(case=name,sourceSide=n,parameters=p,
                     quinticTruth=errors(old_image,truth),retainedTruth=errors(new_image,truth),
                     directTruth=errors(direct,truth),quinticFactor=errors(old_image,direct),retainedFactor=errors(new_image,direct),
                     correction=errors(new_image,old_image),quinticAreaTruth=errors(old_area,ta),retainedAreaTruth=errors(new_area,ta),
                     areaCorrection=errors(new_area,old_area),truthAreaRefinement=errors(ta,ta12)['max'],
                     collocationAgreement=errors(evaluate(old,xs),evaluate(new,xs))['max'],
                     retainedAreaQuadratureCheck=errors(new_area,area(new,65,8))['max'])
            row['continuous']=continuous_errors(old,new,p,32)
            row['continuousRefinement16']=continuous_errors(old,new,p,16)
            if selected:
                native=np.asarray(distilled_conv_synthesis(source,(point_side,point_side)),dtype=float)
                row['nativeFloat32VsDirect']=errors(native,direct)
                xx=np.linspace(0,cells,513)
                for method,img in [('quintic',evaluate(old,xx)),('retained',evaluate(new,xx)),('truth',truth_xy(p,xx[None,:]/cells,xx[:,None]/cells))]:images[p['kind']+'/'+method]=img
                images[p['kind']+'/correction']=images[p['kind']+'/retained']-images[p['kind']+'/quintic']
            row['seconds']=time.perf_counter()-started;rows.append(row)
            print(json.dumps(dict(case=name,truthRatio=row['retainedTruth']['mse']/row['quinticTruth']['mse'],factorRatio=row['retainedFactor']['mse']/row['quinticFactor']['mse'],maxCorrection=row['correction']['max'],seconds=row['seconds'])),flush=True)
    paths=[Path(__file__),Path(__file__).with_name('faithful_blend.py'),Path('experiments/conv_warp/joint_reference.py'),Path('experiments/convstar.py'),Path('experiments/conv_synthetic_evidence.py')]
    packet=dict(scope=f'56 declared smooth paper cases; source 17/33; point {point_side}; area 65. Float64. No clipping or joint-current admission on either side. No new local analysis.',pointSide=point_side,runtime=platform.python_version(),numpy=np.__version__,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},rows=rows)
    (out/'results.json').write_text(json.dumps(packet,indent=2)+'\n');np.savez_compressed(out/'images.npz',**images)


if __name__=='__main__':
    import sys
    run(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 129)
