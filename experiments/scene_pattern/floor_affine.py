"""Targeted floor experiment: retain translation hypotheses, fit local affine.

The user-identified floor polygon is a diagnostic ROI, not a supplied match.
All geometric parameters come from the two retained source observations.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import ndimage,optimize
from PIL import Image,ImageDraw
from .average_alignment import atlas_sources,average
from .core import srgb_to_lab
from .run import HERE,save

POLYGON=[(423,267),(486,250),(507,262),(501,307),(477,324),(428,299)]


def affine_points(points,p,center):
    return points+np.asarray(p[:2])+(points-center)@(np.asarray(p[2:]).reshape(2,2)/50).T


def fit_floor(first,second,wa,wb,polygon=POLYGON,structural=False):
    shape=wa.shape;mask_im=Image.new('1',(shape[1],shape[0]));ImageDraw.Draw(mask_im).polygon(polygon,fill=1)
    mask=np.asarray(mask_im)&(wa>.2)&(wb>.2)
    yy,xx=np.nonzero(mask);points=np.c_[xx,yy].astype(float);center=points.mean(0)
    if len(points)<150:raise ValueError('Insufficient measured floor overlap')
    # Spatially interleaved four-pixel blocks: fit, hypothesis selection, test.
    split=((xx//4)+2*(yy//4))%3
    a=srgb_to_lab(first)[...,0];b=srgb_to_lab(second)[...,0]
    fa=np.stack([ndimage.gaussian_filter(a,.6)-ndimage.gaussian_filter(a,3),ndimage.gaussian_filter(a,1.5)],-1)
    fb=np.stack([ndimage.gaussian_filter(b,.6)-ndimage.gaussian_filter(b,3),ndimage.gaussian_filter(b,1.5)],-1)
    feature_weights=np.array([1.,.4])
    if structural:
        la=srgb_to_lab(first);lb=srgb_to_lab(second)
        fa=np.concatenate([fa,ndimage.gaussian_filter(la[...,0],5)[...,None],ndimage.gaussian_filter(la[...,1:],(2,2,0))],-1)
        fb=np.concatenate([fb,ndimage.gaussian_filter(lb[...,0],5)[...,None],ndimage.gaussian_filter(lb[...,1:],(2,2,0))],-1)
        feature_weights=np.array([.7,.7,1.,.6,.6])
    # Normalize each feature over the same initial observed overlap. The second
    # channel also has planar illumination removed over the selected floor.
    for f in [fa,fb]:
        for k in range(f.shape[-1]):
            values=f[yy,xx,k];design=np.c_[np.ones(len(xx)),xx-center[0],yy-center[1]]
            plane=np.linalg.lstsq(design,values,rcond=None)[0];gy,gx=np.indices(shape)
            if structural and k>0:plane=np.array([values.mean(),0,0])
            f[...,k]-=plane[0]+plane[1]*(gx-center[0])+plane[2]*(gy-center[1])
            f[...,k]/=max(float(np.std(f[yy,xx,k])),.004)
    target=fa[yy,xx]*feature_weights
    def residual(p):
        q=affine_points(points,p,center)
        v=np.stack([ndimage.map_coordinates(fb[...,k],[q[:,1],q[:,0]],order=1,mode='nearest') for k in range(fb.shape[-1])],-1)*feature_weights
        validity=ndimage.map_coordinates(wb,[q[:,1],q[:,0]],order=1,mode='constant',cval=0)
        return v-target,validity
    def score(p,part):
        r,v=residual(p);use=split==part
        return float(np.mean(np.minimum(r[use]**2,9)))+10*float(np.mean(v[use]<.05))
    candidates=[]
    radius=24 if structural else 12
    for dy in range(-radius,radius+1,2):
        for dx in range(-radius,radius+1,2):
            p=np.array([dx,dy,0,0,0,0.]);candidates.append((score(p,0),p))
    candidates.sort(key=lambda item:item[0]);seeds=[]
    for _,p in candidates:
        if all(np.linalg.norm(p[:2]-s[:2])>=4 for s in seeds):seeds.append(p)
        if len(seeds)>=8:break
    seeds.append(np.zeros(6));trials=[]
    def objective(p):
        r,v=residual(p);return np.r_[r[split==0].ravel(),np.maximum(.08-v[split==0],0)*5,p[2:]*.02]
    def jac(p):return np.column_stack([(objective(p+d)-objective(p-d))/.04 for d in np.eye(6)*.02])
    bounds=([-28,-28,-30,-30,-30,-30],[28,28,30,30,30,30]) if structural else ([-16,-16,-14,-14,-14,-14],[16,16,14,14,14,14])
    for seed in seeds:
        result=optimize.least_squares(objective,seed,jac=jac,bounds=bounds,loss='soft_l1',f_scale=.5,max_nfev=100)
        p=result.x;A=np.eye(2)+p[2:].reshape(2,2)/50
        valid=bool(np.linalg.det(A)>.6 and np.linalg.svd(A,compute_uv=False).min()>.65)
        trials.append(dict(seed=seed.tolist(),parameters=p.tolist(),selection_error=score(p,1),fit_error=score(p,0),valid=valid,determinant=float(np.linalg.det(A))))
    eligible=[r for r in trials if r['valid']]
    if not eligible:raise ValueError('No admissible affine hypothesis')
    best=min(eligible,key=lambda r:r['selection_error']);p=np.asarray(best['parameters'])
    return p,center,dict(polygon=polygon,point_count=len(points),center=center.tolist(),trials=trials,selected=best,before_selection=score(np.zeros(6),1),before_test=score(np.zeros(6),2),after_test=score(p,2),split='4-pixel blocks: fit, candidate selection, independent reporting test; all from same capture pair')


def floor_fields(shape,p,center,polygon=POLYGON):
    # Split the relative affine map symmetrically through its matrix logarithm.
    # This keeps the existing panorama's coordinate gauge instead of declaring
    # either source an exact sharp reference.
    from scipy.linalg import logm,expm
    h=np.eye(3);h[:2,:2]+=p[2:].reshape(2,2)/50;h[:2,2]=p[:2]-(h[:2,:2]-np.eye(2))@center
    generator=logm(h)
    if np.max(np.abs(generator.imag))>1e-8:raise ValueError('Affine map has no accepted real logarithm')
    half=expm(generator.real*.5);inverse=expm(-generator.real*.5)
    hh,ww=shape;yy,xx=np.indices(shape);points=np.stack((xx,yy),-1)
    im=Image.new('1',(ww,hh));ImageDraw.Draw(im).polygon(polygon,fill=1);mask=np.asarray(im)
    # Full correction inside the selected plane; fade outside its boundary.
    outside=ndimage.distance_transform_edt(~mask);taper=np.exp(-(outside/12)**2)
    fields=[]
    for m in [inverse,half]:fields.append(((points@m[:2,:2].T+m[:2,2])-points)*taper[...,None])
    return fields,mask,h


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('/tmp/floor_affine'));parser.add_argument('--preview',action='store_true');args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    base=json.loads((HERE/'out/v2/room/receipt.json').read_text())
    with np.load(HERE/'out/v2/room/alignment/fields.npz') as cache:
        sources=cache['original'];weights=cache['masks'];previous_fields=cache['fields'];previous_sources=cache['candidate']
    first,second=8,10;p,center,receipt=fit_floor(sources[first],sources[second],weights[first],weights[second])
    local,roi,h=floor_fields(weights.shape[1:],p,center);fields=previous_fields.copy()
    # Replace the former shrunk two-view field in this diagnostic plane only.
    outside=ndimage.distance_transform_edt(~roi);taper=np.exp(-(outside/12)**2)
    for i,f in zip([first,second],local):fields[i]=fields[i]*(1-taper[...,None])+f
    method='linear' if args.preview else 'conv'
    subset=dict(base)
    for key in ['files','matrices','gains']:subset[key]=[base[key][i] for i in [first,second]]
    rendered,valid_weights=atlas_sources(HERE/'data/desktop-scene',subset,700,method=method,fields=fields[[first,second]])
    candidate=previous_sources.copy();wb=weights.copy()
    candidate[[first,second]]=rendered;wb[[first,second]]=valid_weights
    common=np.minimum(weights,wb);covered=common.sum(0)>0
    for name,s in [('before',sources),('previous',previous_sources),('after',candidate)]:
        out=average(s,common);save(np.where(covered[...,None],out,[.025,.032,.04]),args.out/f'{name}.png')
        crop=out[235:345,410:560];save(crop,args.out/f'{name}-floor.png')
    checks=[]
    for i in [first,second]:
        dy,dx=np.gradient(fields[i],axis=(0,1));det=(1+dx[...,0])*(1+dy[...,1])-dx[...,1]*dy[...,0]
        checks.append(dict(view=i,min_jacobian=float(det.min()),max_displacement=float(np.linalg.norm(fields[i],axis=-1).max())))
    metrics=[]
    for name,s in [('before',sources),('previous',previous_sources),('after',candidate)]:
        a=srgb_to_lab(s[first])[...,0];b=srgb_to_lab(s[second])[...,0];a-=ndimage.gaussian_filter(a,3);b-=ndimage.gaussian_filter(b,3)
        valid=roi&(common[first]>.2)&(common[second]>.2)
        metrics.append(dict(name=name,contrast_error=float(np.mean(np.abs(a[valid]-b[valid]))),evaluated_pixels=int(valid.sum())))
    receipt.update(first=base['files'][first],second=base['files'][second],relative_affine=h.tolist(),fields=checks,metrics=metrics,sampler=method,seconds=time.perf_counter()-start,limitations='User-localized floor-plane ROI; two-view affine experiment, not full-scene automatic segmentation or verified depth. Test blocks share the same captures. No seam changes or sharpening applied.')
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2));np.savez_compressed(args.out/'fields.npz',fields=fields)
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
