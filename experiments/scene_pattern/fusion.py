"""Fuse real photographs in photo 3's plane; no reference geometry in fitting."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from .core import analyze, appearance_score, fit_geometry, map_points, samples, srgb_to_lab
from .run import HERE, image, save


def anchor_matrices(matrices, anchor):
    inverse = np.linalg.inv(matrices[anchor])
    return [h @ inverse for h in matrices]


def atlas_grid(matrices, width):
    corners = np.array([[0,0],[1,0],[1,1],[0,1.]])
    bounds = []
    for h in matrices:
        inverse = np.linalg.inv(h)
        denominator = np.c_[corners,np.ones(4)] @ inverse[2]
        if np.min(denominator)*np.max(denominator) <= 0:
            raise ValueError('Source crosses the atlas projective horizon')
        bounds.append(map_points(inverse,corners))
    bounds = np.concatenate(bounds)
    lo, hi = bounds.min(0), bounds.max(0)
    # Normalized coordinates use the original 800:640 photograph aspect.
    height = round(width * (hi[1]-lo[1])/(hi[0]-lo[0]) * .8)
    if height > 3*width:
        raise ValueError('Unbounded-looking atlas; inspect geometry before rendering')
    y,x = np.meshgrid(np.linspace(lo[1],hi[1],height),np.linspace(lo[0],hi[0],width),indexing='ij')
    return np.stack((x,y),axis=-1), lo, hi


def linearize(rgb):
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def encode(rgb):
    return np.where(rgb<=.0031308,12.92*rgb,1.055*np.maximum(rgb,0)**(1/2.4)-.055)


def fuse(stack, weights):
    """Robust color consensus in Oklab, energy averaging in linear RGB.

Unsupported samples have zero weight even if their storage contains a color.
The robust center is a channel median; chromatic distances then suppress
inconsistent observations. This cannot recover detail absent from all views.
"""
    valid = weights>0
    lab = np.stack([srgb_to_lab(x) for x in stack])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        center = np.nanmedian(np.where(valid[...,None],lab,np.nan),axis=0)
    center = np.nan_to_num(center)
    residual = np.linalg.norm((lab-center)*[1,1.5,1.5],axis=-1)
    robust = weights / (1+(residual/.035)**4)
    total = robust.sum(0)
    fractions = robust/np.maximum(total,1e-20)
    fused = encode(np.sum(fractions[...,None]*linearize(stack),axis=0))
    mean = encode(np.sum(weights[...,None]*linearize(stack),axis=0)/np.maximum(weights.sum(0)[...,None],1e-20))
    disagreement = np.sqrt(np.sum(fractions*residual**2,axis=0))
    # Also retain unattenuated disagreement: robust rejection must not hide it.
    raw_disagreement = np.sqrt(np.sum(weights*residual**2,axis=0)/np.maximum(weights.sum(0),1e-20))
    return fused, mean, fractions, raw_disagreement, disagreement


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=HERE/'data/graf')
    parser.add_argument('--initial',type=Path,default=HERE/'out/v2/receipt.json')
    parser.add_argument('--out',type=Path,default=Path('/tmp/scene_pattern_fusion'))
    parser.add_argument('--width',type=int,default=960)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    initial=json.loads(args.initial.read_text())
    photos=[image(args.data/f'img{i}.ppm') for i in range(1,7)]
    for i in range(1,7):
        digest=hashlib.sha256((args.data/f'img{i}.ppm').read_bytes()).hexdigest()
        if initial['input_sha256'][f'img{i}.ppm']!=digest:
            raise ValueError('Input differs from registered collection')
    anchor=2
    matrices=anchor_matrices([np.asarray(h) for h in initial['global_matrices']],anchor)
    scenes=[analyze(x) for x in photos]
    refinements=[]
    for i,h in enumerate(matrices):
        if i==anchor: continue
        candidate,_=fit_geometry(scenes[anchor],scenes[i],h,side=256,method='conv',max_nfev=35)
        before=appearance_score(scenes[anchor],scenes[i],h)
        after=appearance_score(scenes[anchor],scenes[i],candidate)
        accepted=bool(np.isfinite(after) and after<before)
        if accepted: matrices[i]=candidate
        refinements.append(dict(photo=i+1,before=float(before),after=float(after),accepted=accepted))
        print(json.dumps({'anchor_refinement':refinements[-1]}),flush=True)
    geometry_seconds=time.perf_counter()-started
    q,lo,hi=atlas_grid(matrices,args.width)
    warped=[];weights=[];render=[]
    for i,(rgb,h) in enumerate(zip(photos,matrices)):
        t=time.perf_counter();p=map_points(h,q)
        valid=np.all((p>=0)&(p<=1),axis=-1)
        values=np.zeros((*q.shape[:2],3),dtype=np.float32)
        # Native CONV only where the photograph actually exists; no clamped borders.
        values[valid]=samples(rgb,p[valid],method='conv')
        values=np.clip(values,0,1)
        distance=np.min(np.concatenate((p,1-p),axis=-1),axis=-1)
        feather=np.clip(distance/.045,0,1)*valid
        # Image-space Jacobian measures local source pixels per atlas pixel.
        px=p*np.array([rgb.shape[1]-1,rgb.shape[0]-1])
        dy,dx=np.gradient(px,axis=(0,1))
        density=np.abs(dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0])
        quality=np.clip(density,.08,4)**.5
        weights.append(feather*quality)
        warped.append(values)
        save(np.where(valid[...,None],values,.035),args.out/f'source{i+1}.jpg')
        render.append(dict(photo=i+1,seconds=time.perf_counter()-t,covered_pixels=int(valid.sum())))
        print(json.dumps({'rendered':render[-1]}),flush=True)
    stack=np.asarray(warped);weight=np.asarray(weights)
    # Small per-channel exposure gains estimated against anchor on flat overlap.
    # The bounds deliberately prevent an overlap mismatch from recoloring a view.
    linear=linearize(stack);gains=[]
    for i in range(6):
        mask=(weight[i]>.3)&(weight[anchor]>.3)
        mask &= np.all((linear[i]>.03)&(linear[i]<.85)&(linear[anchor]>.03)&(linear[anchor]<.85),axis=-1)
        if i!=anchor and mask.sum()>100:
            gain=np.clip(np.median(linear[anchor][mask]/linear[i][mask],axis=0),.8,1.25)
        else: gain=np.ones(3)
        gains.append(gain.tolist());linear[i]*=gain
    corrected=np.clip(encode(linear),0,1)
    fused,mean,fractions,disagreement,robust_disagreement=fuse(corrected,weight)
    support=(weight>0).sum(0);covered=support>0
    background=np.array([.025,.032,.04])
    for name,value in [('fused',fused),('average',mean)]:
        save(np.where(covered[...,None],value,background),args.out/f'{name}.png')
    palette=np.array([[.025,.032,.04],[.20,.32,.6],[.1,.65,.8],[.2,.8,.65],[.65,.85,.3],[1,.65,.15],[1,.9,.7]])
    save(palette[support],args.out/'support.png')
    heat=np.clip(disagreement/.12,0,1)
    save(np.where(covered[...,None],np.stack((heat,heat**2,.15*(1-heat)),axis=-1),background),args.out/'disagreement.png')
    source_palette=np.array([[.94,.3,.3],[1,.65,.1],[.95,.9,.25],[.2,.8,.6],[.2,.6,1],[.7,.4,.95]])
    save(np.where(covered[...,None],source_palette[fractions.argmax(0)],background),args.out/'dominant.png')
    np.savez_compressed(args.out/'atlas.npz',matrices=matrices,lo=lo,hi=hi,support=support,weights=fractions.astype(np.float32),disagreement=disagreement.astype(np.float32))
    # Evaluation enters only after geometry, rendering and fusion have frozen.
    from .run import reference_matrix,error
    evaluation=[dict(photo=i+1,**error(h,reference_matrix(args.data,3,i+1),args.data,i+1)) for i,h in enumerate(matrices)]
    receipt=dict(anchor_photo=3,initial_receipt_sha256=hashlib.sha256(args.initial.read_bytes()).hexdigest(),
        ground_truth_used_for_fit=False,method='Anchor-frame photometric refinement; native CONV source evaluation; bounded linear-RGB exposure correction; Oklab robust consensus and linear-light fusion',
        limitations='Planar scene model; no depth, occlusion reasoning, seam optimization, super-resolution, or real-time performance claim. Disagreement is measured, not certified geometric confidence.',
        shape=list(q.shape[:2]),bounds=[lo.tolist(),hi.tolist()],matrices=[h.tolist() for h in matrices],
        refinements=refinements,gains=gains,evaluation=evaluation,render=render,
        geometry_seconds=geometry_seconds,total_seconds=time.perf_counter()-started,
        covered_pixels=int(covered.sum()),multiple_view_fraction=float(np.mean(support[covered]>=2)),
        source_contribution=[float(fractions[i].sum()/covered.sum()) for i in range(6)],
        overlap_disagreement_median=float(np.median(disagreement[support>=2])),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'core.py']})
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'complete':receipt}),flush=True)

if __name__=='__main__':main()
