"""Image-only regional registration. Reference homographies never enter this module.

Matrices map normalized reference image coordinates to moving-image coordinates.
Segmentation is the forest's native causal-density implementation. CONV uses
its existing C two-order arbitrary-coordinate evaluator; no substitute kernel.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import numpy as np
from scipy import ndimage, signal, optimize
from scipy.spatial.distance import cdist
from skimage.registration import phase_cross_correlation
from skimage.transform import ProjectiveTransform
from skimage.measure import ransac

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / 'viewer', ROOT / 'experiments'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from bfft.effects import meyer_channels, srgb_to_lab, lab_to_srgb
from port_needed.pipeline import SegmentingConfig, build_segmenting_representation
from standalone_conv_resize_demo.backend import conv_evaluate_profile_2d

@dataclass
class Scene:
    rgb: np.ndarray
    lab: np.ndarray
    cartoon: np.ndarray
    texture: np.ndarray
    channels: np.ndarray
    labels: dict
    centers: np.ndarray
    descriptor: np.ndarray
    receipt: dict


def resize(image, shape):
    """Antialiased pyramid restriction, never a source of reference geometry."""
    from skimage.transform import resize as skresize
    return skresize(image, shape, anti_aliasing=True, preserve_range=True).astype(np.float32)


def texture_coordinates(lab, cartoon, texture):
    """L texture, radial chroma change, and two circular hue chord coordinates.

Hue evidence vanishes at achromatic endpoints. The signed Meyer remainder is
retained separately; no clipping of negative texture precedes this calculation.
"""
    ab, cab = lab[..., 1:3], cartoon[..., 1:3]
    c = np.linalg.norm(ab, axis=-1)
    cc = np.linalg.norm(cab, axis=-1)
    direction = ab / np.maximum(c[..., None], 1e-7)
    cdir = cab / np.maximum(cc[..., None], 1e-7)
    hue = np.minimum(c, cc)[..., None] * (direction - cdir)
    return np.concatenate((texture[..., :1], (c-cc)[..., None], hue), axis=-1)


def region_centers(labels, maximum=220):
    h, w = labels.shape
    ids, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    inverse=inverse.ravel()
    y, x = np.indices((h,w))
    cx = np.bincount(inverse, weights=x.ravel()) / counts
    cy = np.bincount(inverse, weights=y.ravel()) / counts
    rows = np.argsort(counts)[::-1]
    rows = rows[(counts[rows] >= 4) & (cx[rows] > 5) & (cx[rows] < w-6) & (cy[rows] > 5) & (cy[rows] < h-6)][:maximum]
    return np.column_stack((cx[rows]/(w-1),cy[rows]/(h-1)))


def samples(image, points, method='linear'):
    p = np.asarray(points)
    h,w = image.shape[:2]
    x,y = np.clip(p[...,0],0,1)*(w-1), np.clip(p[...,1],0,1)*(h-1)
    if method == 'conv':
        return conv_evaluate_profile_2d(image,x.ravel(),y.ravel()).reshape(p.shape[:-1]+image.shape[2:])
    if image.ndim == 2:
        return ndimage.map_coordinates(image,[y,x],order=1,mode='nearest')
    return np.stack([ndimage.map_coordinates(image[...,k],[y,x],order=1,mode='nearest') for k in range(image.shape[2])],axis=-1)


def descriptors(lab, cartoon, channels, points):
    # Broad color context at two scales; texture energy, without hue angle seams.
    h,w = lab.shape[:2]
    offsets = np.array([(x,y) for y in (-1,0,1) for x in (-1,0,1)])
    pieces=[]
    for radius in (5,12):
        p=points[:,None,:]+offsets[None,:,:]*[radius/(w-1),radius/(h-1)]
        patch=samples(cartoon,p)*[1,2,2]
        pieces.append(patch.reshape(len(points),-1))
    energy=ndimage.gaussian_filter(channels**2,(3,3,0))**.5
    pieces.append(samples(energy,points)*2)
    return np.concatenate(pieces,axis=1)


def analyze(rgb, side=256):
    start=time.perf_counter()
    h,w=rgb.shape[:2]
    shape=(round(h*side/max(h,w)),round(w*side/max(h,w)))
    rgb=resize(rgb,shape)
    lab=srgb_to_lab(rgb)
    split=meyer_channels(rgb,space='oklab',method='fast',threads=4)
    cartoon=split.offset+split.cartoon/split.scale
    texture=split.texture/split.scale
    residual=split.residual/split.scale
    channels=texture_coordinates(lab,cartoon,texture)
    split_time=time.perf_counter()
    cfg=SegmentingConfig(allocation_method='causal_density',allocation_max_side=side,
        tgfd_sweeps=1,flow_sweeps=8,safety_cells=4096,characteristic_passes=1,
        characteristic_trust_fraction=.5,characteristic_core_radius=3.,soft_support_passes=0,ridge_count=1,threads=4)
    labels={}; timings={}; points=[]
    for name,plane in zip(('lightness','chroma','hue_a','hue_b'),np.moveaxis(channels,-1,0)):
        t=time.perf_counter()
        # Encode a signed scalar as neutral Oklab lightness for the unchanged
        # forest API, which converts its RGB input back to Oklab internally.
        extent=max(float(np.quantile(np.abs(plane),.995)),1e-7)
        scalar=np.clip(.5+.42*plane/extent,.02,.98)
        neutral=np.zeros((*shape,3));neutral[...,0]=scalar
        carrier=np.clip(lab_to_srgb(neutral),0,1).astype(np.float32)
        result=build_segmenting_representation(carrier,cfg)
        labels[name]=result['labels'].astype(np.int32)
        points.append(region_centers(labels[name]))
        timings[name]=time.perf_counter()-t
    # Cartoon supports remain the coarse geometric context, as in the forest.
    t=time.perf_counter()
    result=build_segmenting_representation(np.clip(lab_to_srgb(cartoon),0,1).astype(np.float32),cfg)
    labels['cartoon']=result['labels'].astype(np.int32)
    points.append(region_centers(labels['cartoon']))
    timings['cartoon']=time.perf_counter()-t
    centers=np.concatenate(points)
    # Deterministic cell deduplication prevents four channels overcounting a site.
    _,keep=np.unique(np.rint(centers*[shape[1]/3,shape[0]/3]).astype(int),axis=0,return_index=True)
    centers=centers[np.sort(keep)]
    desc=descriptors(lab,cartoon,channels,centers)
    return Scene(rgb,lab.astype(np.float32),cartoon.astype(np.float32),texture.astype(np.float32),channels.astype(np.float32),labels,centers,desc,{
        'shape':list(shape),'meyer_seconds':split_time-start,'segmentation_seconds':timings,
        'total_seconds':time.perf_counter()-start,'regions':{k:int(len(np.unique(v))) for k,v in labels.items()},
        'centers':len(centers),'closure_max_abs':float(np.max(np.abs(lab-cartoon-texture-residual))),
        'unresolved_residual_rms':float(np.sqrt(np.mean(residual**2))),
        'meyer_method':'fast: three finite-flow jumps; Oklab L,a,b',
        'segmentation':'forest causal-density transport, five independent maps',
    })


def map_points(matrix, points):
    q=np.concatenate((points,np.ones((*points.shape[:-1],1))),axis=-1)@matrix.T
    return q[...,:2]/q[...,2:3]


def grid(shape):
    y,x=np.meshgrid(np.linspace(0,1,shape[0]),np.linspace(0,1,shape[1]),indexing='ij')
    return np.stack((x,y),axis=-1)


def warp(image,matrix,shape=None,method='conv'):
    shape=shape or image.shape[:2]
    p=map_points(matrix,grid(shape))
    valid=np.all((p>=0)&(p<=1),axis=-1)
    return samples(image,p,method),valid


def region_seed(a,b):
    distances=cdist(a.descriptor,b.descriptor,'sqeuclidean')
    js=np.argmin(distances,axis=1)
    back=np.argmin(distances,axis=0)
    ii=np.arange(len(js));mutual=back[js]==ii
    candidates=ii[mutual]
    candidates=candidates[np.argsort(distances[candidates,js[candidates]])][:180]
    pa,pb=a.centers[candidates],b.centers[js[candidates]]
    receipt={'mutual_matches':len(pa),'inliers':0}
    if len(pa)<8:
        return np.eye(3),receipt
    try:
        model,inliers=ransac((pa,pb),ProjectiveTransform,min_samples=4,residual_threshold=.028,max_trials=1800,random_state=17)
    except TypeError:
        model,inliers=ransac((pa,pb),ProjectiveTransform,min_samples=4,residual_threshold=.028,max_trials=1800,rng=np.random.default_rng(17))
    if model is None or not np.all(np.isfinite(model.params)):
        return np.eye(3),receipt
    receipt['inliers']=int(inliers.sum())
    receipt['reference_points']=pa[inliers].tolist();receipt['moving_points']=pb[inliers].tolist()
    return model.params,receipt


def matrix_params(p):
    return np.array([[1+p[0],p[1],p[2]],[p[3],1+p[4],p[5]],[p[6],p[7],1.]])


def params_matrix(h):
    h=h/h[2,2]
    return np.array([h[0,0]-1,h[0,1],h[0,2],h[1,0],h[1,1]-1,h[1,2],h[2,0],h[2,1]])


def fit_geometry(a,b,initial,side=80,method='linear',max_nfev=55):
    """Robust direct geometric refinement, with a fixed reference sample set.

The optimizer sees appearance only. Outside-overlap residuals are penalized,
not silently removed; a transform cannot win by shrinking valid support.
"""
    shape=(round(a.lab.shape[0]*side/a.lab.shape[1]),side)
    aa=resize(a.lab,shape); bb=resize(b.lab,shape)
    # Sample one point per coarse cell, plus the actual transported region sites.
    q=grid((min(shape[0],26),min(shape[1],34))).reshape(-1,2)
    q=np.concatenate((q,a.centers[::max(1,len(a.centers)//250)]))
    q=q[np.all((q>.04)&(q<.96),axis=1)]
    target=samples(aa,q)
    weights=np.array([1.,2.,2.])
    def residual(p):
        mapped=map_points(matrix_params(p),q)
        observed=samples(bb,mapped,method)
        r=(observed-target)*weights
        outside=np.maximum(-mapped,0)+np.maximum(mapped-1,0)
        return np.concatenate((r.ravel(),(outside*.4).ravel()))
    p=params_matrix(initial)
    def jacobian(p):
        step=2e-4 if method=='conv' else 1e-4
        base=residual(p)
        return np.column_stack([(residual(p+np.eye(8)[j]*step)-base)/step for j in range(8)])
    result=optimize.least_squares(residual,p,jac=jacobian,loss='soft_l1',f_scale=.04,
        max_nfev=max_nfev,diff_step=1e-3 if method=='conv' else 1e-4,
        xtol=1e-5,ftol=1e-5,gtol=1e-6)
    h=matrix_params(result.x)
    return h,{'nfev':result.nfev,'cost':float(result.cost),'success':bool(result.success),'side':side,'sampler':method}


def appearance_score(a,b,h):
    q=grid((48,60)).reshape(-1,2)
    p=map_points(h,q)
    valid=np.all((p>.02)&(p<.98),axis=1)
    if valid.mean()<.45 or np.linalg.det(h)<=0:
        return 10.
    # Reject maps with a pole or severe collapse within the reference frame.
    den=np.c_[q,np.ones(len(q))]@h[2]
    if den.min()<=.15:
        return 10.
    error=np.linalg.norm((samples(a.lab,q[valid])-samples(b.lab,p[valid]))*[1,2,2],axis=1)
    return float(np.mean(np.minimum(error,.3))+.035*(1-valid.mean()))


def ringing(image):
    """Deliberately sharp, finite-support band-pass with visible sidelobes."""
    taps=signal.firwin(25,[.07,.48],pass_zero=False,window='boxcar')
    x=ndimage.convolve1d(image,taps,axis=1,mode='reflect')
    y=ndimage.convolve1d(image,taps,axis=0,mode='reflect')
    return x+y


def fractional_delay(reference,moving,mode='ringing'):
    """Return the x,y shift to apply to moving, in pixels (not exact under noise)."""
    if mode=='ringing':
        reference,moving=ringing(reference),ringing(moving)
    window=np.outer(np.hanning(reference.shape[0]),np.hanning(reference.shape[1]))
    shift,error,_=phase_cross_correlation(reference*window,moving*window,upsample_factor=40,normalization=None)
    return np.array([shift[1],shift[0]]),float(error)


def refine_delays(a,b,h,mode,warped_texture=None):
    warped,valid=warped_texture if warped_texture is not None else warp(b.texture[...,0],h,a.texture.shape[:2],method='conv')
    reference=a.texture[...,0];H,W=reference.shape
    # The same regional supports and patches are used in both ablation arms.
    controls=[];targets=[];delays=[]
    for cx,cy in a.centers[::max(1,len(a.centers)//100)]:
        x,y=int(round(cx*(W-1))),int(round(cy*(H-1)))
        radius=18
        if min(x,y,W-1-x,H-1-y)<radius+2:
            continue
        sl=np.s_[y-radius:y+radius+1,x-radius:x+radius+1]
        if not np.all(valid[sl]) or np.std(reference[sl])<.001:
            continue
        shift,error=fractional_delay(reference[sl],warped[sl],mode)
        if np.linalg.norm(shift)>4 or error>.9:
            continue
        q=np.array([cx,cy]);source=map_points(h,q-shift/[W-1,H-1])
        controls.append(q);targets.append(source);delays.append(shift.tolist())
    receipt={'mode':mode,'patches':len(controls),'delays_px':delays,'accepted':False}
    if len(controls)<8:
        return h,receipt
    model=ProjectiveTransform();ok=model.estimate(np.array(controls),np.array(targets))
    if not ok:return h,receipt
    before=appearance_score(a,b,h);after=appearance_score(a,b,model.params)
    receipt.update(score_before=before,score_after=after)
    if after<before:
        receipt['accepted']=True;return model.params,receipt
    return h,receipt


def refine_radial(a,b,h,warped_texture):
    from .radial import unwind,trace_similarity
    moving,valid=warped_texture
    height,width=moving.shape
    records=[]
    # Traces about several existing region centers; no detector is introduced.
    for point in a.centers[::max(1,len(a.centers)//24)]:
        center=point*[width-1,height-1]
        radius=min(45.,center[0]-2,center[1]-2,width-3-center[0],height-3-center[1])
        if radius<22:continue
        ta,mask,meta=unwind(a.texture[...,0],center,r_min=5,r_max=radius)
        tb,_,_=unwind(moving,center,r_min=5,r_max=radius)
        vm,_,_=unwind(valid.astype(float),center,r_min=5,r_max=radius)
        if vm.min()<.999:continue
        result=trace_similarity(ta,tb,meta['log_radius_step'],max_radial_shift=4)
        result['center_normalized']=point.tolist();records.append(result)
    good=[r for r in records if r['correlation']>.65 and not r['radial_search_boundary'] and abs(r['angle_degrees'])<8]
    receipt={'traces':records,'reliable_traces':len(good),'accepted':False}
    if len(good)<3:return h,receipt
    angle=float(np.median([r['angle_degrees'] for r in good]));scale=float(np.median([r['scale'] for r in good]))
    spread=float(np.median(np.abs(np.array([r['angle_degrees'] for r in good])-angle)))
    receipt.update(angle_degrees=angle,scale=scale,angular_mad=spread)
    if spread>2 or abs(np.log(scale))>.08:return h,receipt
    # Each trace rotates/scales about its own site. Fit the resulting local
    # offset correspondences rather than rotating the whole frame about zero.
    source=[];target=[];theta=np.deg2rad(angle)
    A=scale*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    for r in good:
        c=np.array(r['center_normalized'])
        for offset in ((-12,0),(12,0),(0,-12),(0,12)):
            d=np.array(offset)/[width-1,height-1]
            transformed=(A@np.array(offset))/[width-1,height-1]
            source.append(c+d);target.append(map_points(h,c+transformed))
    model=ProjectiveTransform();model.estimate(np.array(source),np.array(target))
    receipt['score_before']=appearance_score(a,b,h);receipt['score_after']=appearance_score(a,b,model.params)
    if receipt['score_after']<receipt['score_before']:
        receipt['accepted']=True;return model.params,receipt
    return h,receipt


def register(a,b):
    start=time.perf_counter();seed,seed_receipt=region_seed(a,b)
    candidates=[('region',seed),('identity',np.eye(3))]
    # Finite coarse rotation/scale bank. Translation is estimated per hypothesis.
    aa=resize(a.cartoon[...,0],(64,80));bb=resize(b.cartoon[...,0],(64,80))
    for angle in (-20,0,20):
        for scale in (.8,1.,1.2):
            theta=np.deg2rad(angle);A=scale*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            h=np.eye(3);h[:2,:2]=A;h[:2,2]=.5-A@np.array([.5,.5])
            moved,_=warp(bb,h,aa.shape,method='linear')
            shift,_=fractional_delay(aa,moved,'phase')
            h[:2,2]-=A@(shift/[79,63])
            candidates.append((f'coarse_{angle}_{scale}',h))
    candidates.sort(key=lambda pair:appearance_score(a,b,pair[1]))
    trials=[]
    for name,h in candidates[:4]:
        h,rec=fit_geometry(a,b,h,side=80,max_nfev=65)
        trials.append((appearance_score(a,b,h),name,h,rec))
    trials.sort(key=lambda item:item[0]);_,name,h,_=trials[0]
    stages=[]
    for side in (160,256):
        proposed,rec=fit_geometry(a,b,h,side=side,max_nfev=50)
        if appearance_score(a,b,proposed)<appearance_score(a,b,h):h=proposed
        stages.append(rec)
    base=h.copy()
    print('  geometry fitted; evaluating native CONV texture',flush=True)
    warped_texture=warp(b.texture[...,0],base,a.texture.shape[:2],method='conv')
    phase,phase_receipt=refine_delays(a,b,base,'phase',warped_texture)
    ring,ring_receipt=refine_delays(a,b,base,'ringing',warped_texture)
    radial,radial_receipt=refine_radial(a,b,base,warped_texture)
    polish_seed=min((ring,radial),key=lambda candidate:appearance_score(a,b,candidate))
    # CONV geometry polishing is shared by both arms; its effect is reported.
    h,conv_receipt=fit_geometry(a,b,polish_seed,side=256,method='conv',max_nfev=24)
    if appearance_score(a,b,h)>appearance_score(a,b,polish_seed):h=polish_seed
    return h,{'seconds':time.perf_counter()-start,'seed':seed_receipt,'winning_seed':name,
        'coarse_trials':[{'seed':n,'score':s,'fit':r} for s,n,_,r in trials],
        'stages':stages,'phase':phase_receipt,'ringing':ring_receipt,'radial':radial_receipt,'conv_refinement':conv_receipt,
        'matrices':{'regional_coarse':seed.tolist(),'geometric':base.tolist(),'phase':phase.tolist(),'ringing':ring.tolist(),'radial':radial.tolist(),'final':h.tolist()},
        'final_appearance_score':appearance_score(a,b,h)}
