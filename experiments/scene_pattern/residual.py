"""Bounded region-supported residual registration over the frozen fused atlas."""
import json,time,hashlib,argparse
from pathlib import Path
import numpy as np
from scipy import ndimage,optimize
from .core import analyze,grid,map_points,samples,warp,srgb_to_lab
from .fusion import HERE,fuse,linearize,encode
from .run import image,save


def feature(rgb):
    lab=srgb_to_lab(np.clip(rgb,0,1))
    # Local contrast removes smooth exposure differences without blurring output.
    return lab-ndimage.gaussian_filter(lab,(2,2,0))


def local_delays(reference,moving,valid,centers,radius=12):
    """Fit alternating pixels; accept only on held-out pixels in each region."""
    a,b=feature(reference),feature(moving);h,w=valid.shape
    records=[]
    for center in centers:
        x,y=np.rint(center*[w-1,h-1]).astype(int)
        if min(x,y,w-1-x,h-1-y)<radius+5:continue
        sl=np.s_[y-radius:y+radius+1,x-radius:x+radius+1]
        if not np.all(valid[y-radius-4:y+radius+5,x-radius-4:x+radius+5]):continue
        yy,xx=np.mgrid[y-radius:y+radius+1,x-radius:x+radius+1]
        train=((xx+yy)%2)==0;target=a[sl]
        if np.std(target[...,0])<.012:continue
        def residual(d):
            observed=np.stack([ndimage.map_coordinates(b[...,k],[yy+d[1],xx+d[0]],order=1,mode='nearest') for k in range(3)],-1)
            return (observed-target)*[1,1.5,1.5]
        initial=residual([0,0])
        def objective(d):return residual(d)[train].ravel()
        def jacobian(d):
            return np.column_stack([(objective(d+v)-objective(d-v))/.04 for v in np.eye(2)*.02])
        result=optimize.least_squares(objective,[0.,0.],jac=jacobian,bounds=(-2.5,2.5),loss='soft_l1',f_scale=.025,max_nfev=22)
        final=residual(result.x)
        before=float(np.mean(np.minimum(initial[~train]**2,.01)))
        after=float(np.mean(np.minimum(final[~train]**2,.01)))
        # Local proposals cannot earn credit by changing coverage or fitting only
        # the pixels used by the optimizer. Aperture ambiguity remains possible.
        accepted=after<before*.97 and np.linalg.norm(result.x)<2.4
        if accepted:records.append(dict(center=center.tolist(),delay=result.x.tolist(),before=before,after=after))
    return records


def field(records,shape):
    h,w=shape;y,x=np.indices(shape);numerator=np.zeros((h,w,2));denominator=np.zeros(shape)
    for r in records:
        cx,cy=np.asarray(r['center'])*[w-1,h-1]
        confidence=np.clip(1-r['after']/r['before'],0,1)
        weight=np.exp(-((x-cx)**2+(y-cy)**2)/(2*18**2))*confidence
        numerator+=weight[...,None]*r['delay'];denominator+=weight
    # Shrink unsupported estimates continuously toward the frozen homography.
    displacement=numerator/(denominator[...,None]+.15)
    edge=np.minimum.reduce([x,y,w-1-x,h-1-y])
    displacement*=np.clip(edge/15,0,1)[...,None]
    return displacement


def shifted_points(q,displacement):
    h,w=displacement.shape[:2]
    inside=np.all((q>=0)&(q<=1),axis=-1)
    d=samples(displacement,q)*inside[...,None]
    return q+d/[w-1,h-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=HERE/'out/v2/fusion');p.add_argument('--out',type=Path,default=Path('/tmp/scene_pattern_residual'));args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();base=json.loads((args.base/'receipt.json').read_text());matrices=np.asarray(base['matrices']);lo,hi=np.asarray(base['bounds']);shape=base['shape'];anchor=2
    photos=[image(HERE/f'data/graf/img{i}.ppm') for i in range(1,7)]
    scene=analyze(photos[anchor],side=400);ref=scene.rgb
    fields=[];proposals=[]
    for i in range(6):
        if i==anchor:fields.append(np.zeros((*ref.shape[:2],2)));proposals.append([]);continue
        moving,valid=warp(photos[i],matrices[i],ref.shape[:2],method='linear')
        records=local_delays(ref,moving,valid,scene.centers)
        f=field(records,ref.shape[:2]);fields.append(f);proposals.append(records)
        print(json.dumps({'photo':i+1,'accepted_regions':len(records),'max_delay_work_pixels':float(np.max(np.linalg.norm(f,axis=-1)))}),flush=True)
    y,x=np.meshgrid(np.linspace(lo[1],hi[1],shape[0]),np.linspace(lo[0],hi[0],shape[1]),indexing='ij');q=np.stack((x,y),-1)
    stacks=[[],[]];weights=[];metrics=[]
    for i in range(6):
        points=[map_points(matrices[i],q),map_points(matrices[i],shifted_points(q,fields[i]))]
        masks=[np.all((v>=0)&(v<=1),axis=-1) for v in points];valid=masks[0]&masks[1]
        distance=np.min(np.concatenate((points[0],1-points[0]),-1),-1)
        dy,dx=np.gradient(points[0]*[799,639],axis=(0,1));density=np.abs(dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0])
        weight=np.clip(distance/.045,0,1)*valid*np.clip(density,.08,4)**.5;weights.append(weight)
        for k in range(2):
            rgb=np.zeros((*shape,3),np.float32)
            if i==anchor and k==1:rgb=stacks[0][-1].copy()
            else:rgb[valid]=samples(photos[i],points[k][valid],method='conv')
            stacks[k].append(np.clip(rgb,0,1))
        print(json.dumps({'rendered_photo':i+1}),flush=True)
    weights=np.asarray(weights);outputs=[];disagreements=[]
    for k in range(2):
        stack=np.asarray(stacks[k]);corrected=np.clip(encode(linearize(stack)*np.asarray(base['gains'])[:,None,None,:]),0,1)
        fused,mean,fractions,raw,_=fuse(corrected,weights);outputs.append(fused);disagreements.append(raw)
        name=['before','after'][k];covered=weights.sum(0)>0
        save(np.where(covered[...,None],fused,[.025,.032,.04]),args.out/f'{name}.png')
        heat=np.clip(raw/.12,0,1);save(np.where(covered[...,None],np.stack((heat,heat**2,.15*(1-heat)),-1),[.025,.032,.04]),args.out/f'{name}-disagreement.png')
    # Fixed support, frozen exposures, identical fusion parameters and native
    # sampler: changes in these measurements reflect geometry alone.
    overlap=(weights>0).sum(0)>=2
    for i in range(6):
        if i==anchor:continue
        mask=(weights[i]>.2)&(weights[anchor]>.2)
        pair=[]
        for k in range(2):
            a=feature(stacks[k][anchor]);b=feature(stacks[k][i]);pair.append(float(np.mean(np.abs(a[mask]-b[mask]))))
        metrics.append(dict(photo=i+1,contrast_error_before=pair[0],contrast_error_after=pair[1],relative_change=pair[1]/pair[0]-1))
    # Homography references are opened only after all displacement fields freeze.
    from .run import reference_matrix
    evaluation=[];points=grid((32,40)).reshape(-1,2)
    for i in range(6):
        if i==anchor:continue
        truth=map_points(reference_matrix(HERE/'data/graf',3,i+1),points);valid=np.all((truth>.02)&(truth<.98),axis=-1)
        errors=[]
        for qq in [points,shifted_points(points,fields[i])]:
            d=np.linalg.norm((map_points(matrices[i],qq)-truth)*[799,639],axis=-1)[valid];errors.append(dict(median=float(np.median(d)),p95=float(np.quantile(d,.95))))
        evaluation.append(dict(photo=i+1,before=errors[0],after=errors[1]))
    field_checks=[]
    for i,f in enumerate(fields):
        dy,dx=np.gradient(f,axis=(0,1))
        det=(1+dx[...,0])*(1+dy[...,1])-dx[...,1]*dy[...,0]
        field_checks.append(dict(photo=i+1,min_jacobian=float(det.min()),max_jacobian=float(det.max())))
    detail=[]
    for output in outputs:
        l=srgb_to_lab(output)[...,0];dy,dx=np.gradient(l)
        detail.append(float(np.mean(np.hypot(dx,dy)[overlap])))
    receipt=dict(field_checks=field_checks,mean_fused_gradient_before=detail[0],mean_fused_gradient_after=detail[1],method='Region-centered local contrast delay, checkerboard pixel holdout, Gaussian support and shrinkage, native CONV final rendering',ground_truth_used_for_fit=False,seconds=time.perf_counter()-start,region_counts=[len(r) for r in proposals],proposals=proposals,metrics=metrics,evaluation=evaluation,mean_disagreement_before=float(np.mean(disagreements[0][overlap])),mean_disagreement_after=float(np.mean(disagreements[1][overlap])),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='Anchor-relative field; held-out pixels are spatially correlated, not independent validation scenes. No occlusion inference. Native point sampling does not integrate minification footprints.')
    np.savez_compressed(args.out/'fields.npz',fields=np.asarray(fields))
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='proposals'}),flush=True)

if __name__=='__main__':main()
