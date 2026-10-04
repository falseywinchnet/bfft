"""Spherical display of an image-only overlapping collection."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .core import map_points,samples
from .run import image,save
from .fusion import fuse,linearize,encode


def camera_chain(pairs,anchor,rotation=False):
    # 28mm 35mm-equivalent EXIF focal length, portrait 3:4 sensor aspect.
    f=28/np.hypot(36,24)*1500
    k=np.array([[f/899,0,.5],[0,f/1199,.5],[0,0,1.]])
    chain=[np.eye(3)]
    for p in pairs:
        h=np.asarray(p['matrices']['final']);m=np.linalg.inv(k)@h@k
        if rotation:
            seed=p.get('seed',{})
            if len(seed.get('reference_points',[]))>=8:
                a=np.c_[seed['reference_points'],np.ones(len(seed['reference_points']))]@np.linalg.inv(k).T
                b=np.c_[seed['moving_points'],np.ones(len(seed['moving_points']))]@np.linalg.inv(k).T
                a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
                weight=np.ones(len(a))
                for _ in range(6):
                    u,_,v=np.linalg.svd((b*weight[:,None]).T@a)
                    m=u@np.diag([1,1,np.linalg.det(u@v)])@v
                    error=np.linalg.norm(a@m.T-b,axis=1)
                    weight=1/(1+(error/.025)**2)
            else:
                u,_,v=np.linalg.svd(m);m=u@np.diag([1,1,np.linalg.det(u@v)])@v
        chain.append(m@chain[-1])
    inverse=np.linalg.inv(chain[anchor]);return k,[m@inverse for m in chain]


def world_rays(matrix,pixels,k):
    rays=np.c_[pixels,np.ones(len(pixels))]@np.linalg.inv(k@matrix).T
    return rays/np.linalg.norm(rays,axis=-1,keepdims=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--width',type=int,default=1400);p.add_argument('--sampler',choices=['linear','conv'],default='linear');p.add_argument('--rotation',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();registration=json.loads(args.registration.read_text());k,matrices=camera_chain(registration['pairs'],5,args.rotation)
    edge=np.linspace(0,1,40);boundary=np.concatenate([np.c_[edge,edge*0],np.c_[edge,edge*0+1],np.c_[edge*0,edge],np.c_[edge*0+1,edge]])
    bounds=[]
    for m in matrices:
        rays=world_rays(m,boundary,k);bounds.append(np.c_[np.arctan2(rays[:,0],rays[:,2]),np.arctan2(rays[:,1],np.hypot(rays[:,0],rays[:,2]))])
    b=np.concatenate(bounds);lo=b.min(0);hi=b.max(0)
    height=round(args.width*(hi[1]-lo[1])/(hi[0]-lo[0]));height=min(height,1600)
    pitch,yaw=np.meshgrid(np.linspace(lo[1],hi[1],height),np.linspace(lo[0],hi[0],args.width),indexing='ij');rays=np.stack([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)],-1)
    stack=[];weights=[];timings=[]
    for i,(name,m) in enumerate(zip(registration['files'],matrices)):
        t=time.perf_counter();rgb=image(args.data/name);xyz=rays@(k@m).T;points=xyz[...,:2]/np.where(np.abs(xyz[...,2:])>1e-12,xyz[...,2:],1e-12)
        valid=(xyz[...,2]>0)&np.all((points>=0)&(points<=1),axis=-1)
        values=np.zeros((height,args.width,3),np.float32);values[valid]=samples(rgb,points[valid],method=args.sampler);values=np.clip(values,0,1)
        edge_distance=np.min(np.concatenate((points,1-points),-1),-1)
        # Prefer central image support; soften borders without clamped extension.
        weight=np.clip(edge_distance/.12,0,1)*valid
        stack.append(values);weights.append(weight);save(np.where(valid[...,None],values,.025),args.out/f'source{i+1}.jpg')
        timings.append(time.perf_counter()-t);print(json.dumps({'rendered':name,'seconds':timings[-1],'covered':int(valid.sum())}),flush=True)
    stack=np.asarray(stack);weights=np.asarray(weights)
    # Solve all overlap log-gains together; anchor image 6 has zero log-gain.
    rows=[];targets=[];edges=[];lin=linearize(stack)
    for i in range(len(stack)):
        for j in range(i+1,len(stack)):
            mask=(weights[i]>.5)&(weights[j]>.5)&np.all((lin[i]>.025)&(lin[i]<.8)&(lin[j]>.025)&(lin[j]<.8),-1)
            if mask.sum()<200:continue
            ratio=np.median(np.log(lin[j][mask])-np.log(lin[i][mask]),axis=0)
            row=np.zeros(len(stack));row[i]=1;row[j]=-1;rows.append(row);targets.append(ratio);edges.append([i+1,j+1,int(mask.sum())])
    row=np.zeros(len(stack));row[5]=10;rows.append(row);targets.append(np.zeros(3))
    gain=np.exp(np.clip(np.linalg.lstsq(rows,targets,rcond=None)[0],-.3,.3));corrected=np.clip(encode(lin*gain[:,None,None,:]),0,1)
    fused,mean,fractions,raw,_=fuse(corrected,weights);covered=weights.sum(0)>0
    for name,v in [('fused',fused),('average',mean)]:save(np.where(covered[...,None],v,[.025,.032,.04]),args.out/f'{name}.png')
    # A coherent central-source selection avoids averaging incompatible depths.
    choice=weights.argmax(0);selected=corrected[choice,np.arange(height)[:,None],np.arange(args.width)[None,:]]
    save(np.where(covered[...,None],selected,[.025,.032,.04]),args.out/'selected.png')
    from .seams import seam_mosaic
    seamed,seam_labels=seam_mosaic(corrected,weights)
    save(np.where(covered[...,None],seamed,[.025,.032,.04]),args.out/'seamed.png')
    heat=np.clip(raw/.12,0,1);save(np.where(covered[...,None],np.stack((heat,heat**2,.15*(1-heat)),-1),[.025,.032,.04]),args.out/'disagreement.png')
    palette=np.random.default_rng(18).uniform(.15,.95,(len(stack),3));save(np.where(covered[...,None],palette[choice],[.025,.032,.04]),args.out/'sources.png')
    support=(weights>0).sum(0);save(np.repeat((support/max(support.max(),1))[...,None],3,axis=-1),args.out/'coverage.png')
    receipt=dict(seam_source_pixels=[int(np.sum((seam_labels==i)&covered)) for i in range(len(stack))],files=registration['files'],projection='spherical',rotation_only=args.rotation,sampler=args.sampler,anchor=6,shape=[height,args.width],bounds=[lo.tolist(),hi.tolist()],matrices=[m.tolist() for m in matrices],gains=gain.tolist(),exposure_edges=edges,seconds=time.perf_counter()-start,render_seconds=timings,coverage=int(covered.sum()),multiple_view_fraction=float(np.mean(support[covered]>1)),contributions=[float(fractions[i].sum()/covered.sum()) for i in range(len(stack))],mean_overlap_disagreement=float(np.mean(raw[support>1])),limitations='Uncalibrated scene without reference geometry. Spherical presentation of estimated projective or rotation transforms; parallax and occlusion remain. Point sampled working-resolution photos, not full-resolution or super-resolution reconstruction.')
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
