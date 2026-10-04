"""Finite visible-area construction and independent convex-difference oracle.

The input is a complete tile's <=4 supplied primitives, not inferred image data.
No primitive may be dropped on overflow. Depth and color are affine on the tile.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def clip(poly, plane):
    if not len(poly):return []
    values=[np.dot(plane[:2],p)+plane[2] for p in poly];out=[]
    for i,p in enumerate(poly):
        j=(i+1)%len(poly);q=poly[j];a,b=values[i],values[j]
        if a>=0:out.append(p)
        if (a<0)!=(b<0):out.append(p+(q-p)*a/(a-b))
    return out


def moments(poly):
    if len(poly)<3:return np.zeros(3)
    p=np.asarray(poly);q=np.roll(p,-1,axis=0);cross=p[:,0]*q[:,1]-p[:,1]*q[:,0]
    return np.array([cross.sum()/2,((p[:,0]+q[:,0])*cross).sum()/6,((p[:,1]+q[:,1])*cross).sum()/6])


def planes(tri):
    vertices=tri[:6].reshape(3,2);out=[]
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        e=b-a;n=np.array([-e[1],e[0]]);out.append(np.r_[n,-n@a])
    return out


def depth(tri):return tri[[6,7,8]]


def region(poly, constraints):
    for plane in constraints:
        poly=clip(poly,plane)
        if not poly:break
    return poly


def subtract(poly,constraints):
    """Disjoint outside pieces, no inclusion-exclusion in this oracle."""
    parts=[];inside=poly
    for plane in constraints:
        if np.all(np.asarray(plane)==0):continue
        outside=clip(inside,-np.asarray(plane))
        if abs(moments(outside)[0])>1e-14:parts.append(outside)
        inside=clip(inside,plane)
        if not inside:break
    return parts


def visible_moments(triangles,x,y,oracle=True):
    if len(triangles)>4:raise ValueError('capacity 4 exceeded; no truncation allowed')
    triangles=[np.asarray(t,dtype=np.float64) for t in triangles]
    square=[np.array([x,y]),np.array([x+1,y]),np.array([x+1,y+1]),np.array([x,y+1])]
    answer=[]
    for i,tri in enumerate(triangles):
        base=region(square,planes(tri));occluders=[]
        for j,other in enumerate(triangles):
            if i==j:continue
            d=depth(tri)-depth(other)
            if np.all(d==0) and j>i:continue  # Exact coplanar tie: lower ID wins.
            occluders.append(planes(other)+[d])
        if oracle:
            pieces=[base] if base else []
            for constraints in occluders:
                pieces=[part for piece in pieces for part in subtract(piece,constraints)]
            total=sum((moments(p) for p in pieces),np.zeros(3))
        else:
            total=np.zeros(3)
            for bits in itertools.product((0,1),repeat=len(occluders)):
                constraints=[p for bit,ps in zip(bits,occluders) if bit for p in ps]
                total+=(-1)**sum(bits)*moments(region(base,constraints))
        answer.append(total)
    return np.asarray(answer)


def color(tri,m):
    tri=np.asarray(tri,dtype=np.float64)
    return tri[9:12]*m[0]+tri[12:15]*m[1]+tri[16:19]*m[2]


def primitive(vertices,z=(0,0,.5),rgb=(1,1,1),dx=(0,0,0),dy=(0,0,0)):
    result=np.zeros(20,dtype=np.float32);result[:6]=np.array(vertices).ravel();result[6:9]=z;result[9:12]=rgb;result[12:15]=dx;result[16:19]=dy
    if np.cross(np.array(vertices[1])-vertices[0],np.array(vertices[2])-vertices[0])<0:
        result[2:6]=result[[4,5,2,3]]
    return result


def scenes():
    a=[[3.2,4.1],[29.1,7.2],[9.3,29.2]];b=[[5.3,1.2],[30.1,27.2],[1.2,24.1]]
    rgb1=(.9,.2,.1);rgb2=(.1,.6,.9)
    return {
      'single':[primitive(a,rgb=rgb1)],
      'duplicate':[primitive(a,z=(0,0,.3),rgb=rgb1),primitive(a,z=(0,0,.7),rgb=rgb2)],
      'overlap':[primitive(a,z=(0,0,.3),rgb=rgb1),primitive(b,z=(0,0,.7),rgb=rgb2)],
      'depth_crossing':[primitive(a,z=(.015,0,.2),rgb=rgb1),primitive(b,z=(-.015,0,.65),rgb=rgb2)],
      'shared_edges':[primitive([[3.2,3.2],[28.1,3.2],[28.1,28.1]],rgb=rgb1),primitive([[3.2,3.2],[28.1,28.1],[3.2,28.1]],rgb=rgb1)],
      'thin_crossing':[primitive([[1.2,15.1],[30.2,15.18],[1.2,15.26]],z=(0,0,.3),rgb=rgb1),primitive([[15.1,1.1],[15.27,1.1],[15.19,30.1]],z=(0,0,.7),rgb=rgb2)],
      'affine_shading':[primitive(a,z=(.015,0,.2),rgb=(.1,.2,.1),dx=(.02,0,.01)),primitive(b,z=(-.015,0,.65),rgb=(.2,.1,.2),dy=(0,.02,.015))],
      'four_layers':[primitive(a,z=(.015,0,.2),rgb=rgb1),primitive(b,z=(-.015,0,.65),rgb=rgb2),primitive([[1.1,1.1],[29.2,1.1],[29.2,29.2]],z=(0,.02,.15),rgb=(.1,.9,.3)),primitive([[2.2,29.2],[29.1,2.2],[29.1,29.2]],z=(0,-.02,.8),rgb=(.7,.1,.8))]
    }


def generate(out,random_cases=0):
    out.mkdir(parents=True,exist_ok=True);catalog=scenes();size=32
    rng=np.random.default_rng(20260923)
    for case in range(random_cases):
        ts=[]
        for i in range(1+case%4):
            v=rng.uniform(-8,40,(3,2))
            if np.cross(v[1]-v[0],v[2]-v[0])<0:v=v[::-1]
            ts.append(primitive(v,z=(*rng.uniform(-.03,.03,2),rng.uniform(.1,.9)),rgb=rng.uniform(.2,.7,3),dx=rng.uniform(-.003,.003,3),dy=rng.uniform(-.003,.003,3)))
        catalog[f'random_{case:02d}']=ts
    refs=[];names=[]
    with (out/'visibility.bin').open('wb') as f:
        np.array([size,size,len(catalog)],'<u4').tofile(f)
        for name,triangles in catalog.items():
            np.array([len(triangles)],'<u4').tofile(f)
            packed=np.zeros((4,20),'<f4');packed[:len(triangles)]=triangles;packed.tofile(f)
            image=np.zeros((size,size,3))
            for y in range(size):
                for x in range(size):
                    ms=visible_moments(triangles,x,y)
                    image[y,x]=sum((color(t,m) for t,m in zip(triangles,ms)),np.zeros(3))
            refs.append(image);names.append(name);print('Oracle',name,flush=True)
    np.save(out/'visibility_reference.npy',refs);(out/'visibility_cases.json').write_text(json.dumps(names)+'\n')


def analyze(out):
    ref=np.load(out/'visibility_reference.npy');names=json.loads((out/'visibility_cases.json').read_text());records=[]
    for mode in ('strict','fast','sparse'):
        a=np.fromfile(out/f'visibility_{mode}.bin',np.float32).reshape(len(names),32,32,4)[...,:3]
        for i,name in enumerate(names):
            records.append(dict(scene=name,mode=mode,mse=float(np.mean((a[i]-ref[i])**2)),max_error=float(np.max(abs(a[i]-ref[i]))),finite=bool(np.isfinite(a[i]).all())))
    (out/'visibility_quality.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
    assert all(r['finite'] and r['max_error']<1e-4 for r in records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','analyze']);p.add_argument('--out',type=Path,required=True);p.add_argument('--random-cases',type=int,default=0);a=p.parse_args();generate(a.out,a.random_cases) if a.action=='generate' else analyze(a.out)
