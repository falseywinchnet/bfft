"""Laptop-only refinement using the shared distortion model and acceptance gates."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import ndimage
from .run import HERE,save
from .average_alignment import atlas_sources,average
from .structure_alignment import polygon_mask,REGIONS,jacobian_min
from .floor_affine import POLYGON as FLOOR
from .joint_distortion import fit_joint
from .registration_audit import contrast_error

def domain(shape):
    mask=polygon_mask(shape,REGIONS[1]['polygon']);yy,xx=np.indices(shape);mid=float(np.median(yy[mask]))
    regions=[mask&(yy<=mid),mask&(yy>mid)]
    allowed=(ndimage.distance_transform_edt(~mask)<=32)&~polygon_mask(shape,FLOOR)
    return mask,regions,allowed

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('/tmp/laptop_alignment'));ap.add_argument('--native',action='store_true');ap.add_argument('--fitted',type=Path);ap.add_argument('--partial-view',type=int);ap.add_argument('--absolute',action='store_true');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    base=json.loads((HERE/'out/v2/room/receipt.json').read_text());initial=np.load(HERE/'out/v2/room/alignment/joint/fields.npz')['fields'];mask,regions,allowed=domain(initial.shape[1:3]);reference=REGIONS[1]['anchor'];method='conv' if args.native else 'linear'
    before,wa=atlas_sources(HERE/'data/desktop-scene',base,700,method=method,fields=initial)
    if args.fitted:
        records=json.loads((args.fitted/'receipt.json').read_text())['views'];fields=np.load(args.fitted/'fields.npz')['fields']
    else:
        views=[i for i in range(len(before)) if i!=reference and np.sum(mask&(wa[i]>.2)&(wa[reference]>.2))>=300]
        if args.partial_view is not None:views=[args.partial_view]
        original,original_weights=atlas_sources(HERE/'data/desktop-scene',base,700,method='linear',fields=np.zeros_like(initial)) if args.absolute else (None,None)
        fields,records,_=fit_joint(before,wa,initial,base=base,regions=regions,reference=reference,views=views,allowed=allowed,allow_visibility_change=args.partial_view is not None,fit_sources=original,fit_weights=original_weights)
    after,wb=atlas_sources(HERE/'data/desktop-scene',base,700,method=method,fields=fields);common=np.minimum(wa,wb);covered=common.sum(0)>0
    for name,s in [('before',before),('after',after)]:
        out=average(s,common);save(np.where(covered[...,None],out,[.025,.032,.04]),args.out/f'{name}.png');save(out[209:308,281:438],args.out/f'laptop-{name}.png')
    metrics=[]
    for r in records:
        i=r['view'];valid=mask&(common[reference]>.2)&(common[i]>.2)
        metrics.append(dict(view=i,applied=r['applied'],pixels=int(valid.sum()),before=contrast_error(before[reference],before[i],valid),after=contrast_error(after[reference],after[i],valid)))
    unchanged=float(np.max(np.abs((fields-initial)[:,~allowed])))
    receipt=dict(views=records,metrics=metrics,reference=reference,sampler=method,seconds=time.perf_counter()-start,minimum_jacobian=min(jacobian_min(f) for f in fields),maximum_change_outside_laptop_domain=unchanged,limitations='Existing manually selected laptop region split into upper and lower validation regions. A 32-pixel continuation neighborhood excludes the protected floor region. Parameters and models are estimated; reference is a coordinate gauge. Ordinary averaging and all captures are retained. Region scores measure image consistency, not independent scene geometry.')
    assert unchanged==0
    np.savez_compressed(args.out/'fields.npz',fields=fields);(args.out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
