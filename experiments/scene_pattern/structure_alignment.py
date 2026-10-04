"""User-localized multi-view affine structures, composed with prior floor fields."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image,ImageDraw
from .run import HERE,save
from .average_alignment import atlas_sources,average
from .floor_affine import fit_floor,affine_points,POLYGON as FLOOR_POLYGON
from .core import srgb_to_lab

REGIONS=[dict(name='bronze',anchor=2,polygon=[(196,156),(215,148),(251,272),(229,285)],crop=[180,132,274,297]),
 dict(name='laptop',anchor=5,polygon=[(302,227),(406,224),(415,292),(302,297)],crop=[285,210,433,307]),
 dict(name='doorway',anchor=7,polygon=[(398,49),(427,50),(435,96),(397,98)],crop=[382,34,450,110])]

def polygon_mask(shape,polygon):
    im=Image.new('1',(shape[1],shape[0]));ImageDraw.Draw(im).polygon([tuple(point) for point in polygon],fill=1);return np.asarray(im)

def compose_field(previous,delta):
    """W_old(W_increment(p))-p, not the sum at the wrong coordinates."""
    y,x=np.indices(previous.shape[:2]);q=np.stack((x,y),-1)+delta
    return delta+np.stack([ndimage.map_coordinates(previous[...,k],[q[...,1],q[...,0]],order=1,mode='nearest') for k in range(2)],-1)

def jacobian_min(field):
    dy,dx=np.gradient(field,axis=(0,1));return float(((1+dx[...,0])*(1+dy[...,1])-dx[...,1]*dy[...,0]).min())

def fit_regions(sources,weights,initial):
    fields=initial.copy();records=[];shape=weights.shape[1:];yy,xx=np.indices(shape);points=np.stack((xx,yy),-1)
    for region in REGIONS:
        roi=polygon_mask(shape,region['polygon']);anchor=region['anchor'];record=dict(region,views=[])
        for i in range(len(sources)):
            overlap=roi&(weights[anchor]>.2)&(weights[i]>.2)
            if i==anchor or overlap.sum()<180 or (region['name']=='bronze' and i==0):continue
            print(json.dumps(dict(fitting=region['name'],anchor=anchor,view=i,pixels=int(overlap.sum()))),flush=True)
            p,c,r=fit_floor(sources[anchor],sources[i],weights[anchor],weights[i],region['polygon'],structural=region['name'] in ['bronze','laptop'])
            accepted=r['selected']['selection_error']<r['before_selection']*.95
            raw=affine_points(points,p,c)-points
            # Broad transition limits deformation outside the measured object.
            distance=ndimage.distance_transform_edt(~roi)
            for transition in [24,36,48,64]:
                taper=np.exp(-(distance/transition)**2)
                delta=raw*taper[...,None]
                if i in [8,10]:
                    floor=polygon_mask(shape,FLOOR_POLYGON)
                    clearance=ndimage.distance_transform_edt(~floor)
                    delta*= (1-np.exp(-(clearance/12)**2))[...,None]
                proposed=compose_field(fields[i],delta);j=jacobian_min(proposed)
                if j>.25:break
            accepted=accepted and j>.25
            if accepted:fields[i]=proposed
            record['views'].append(dict(view=i,accepted=bool(accepted),minimum_jacobian=j,transition_pixels=transition,fit=r))
        records.append(record)
    return fields,records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('/tmp/structure_alignment'));ap.add_argument('--preview',action='store_true');ap.add_argument('--fitted',type=Path);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    base=json.loads((HERE/'out/v2/room/receipt.json').read_text());initial=np.load(HERE/'out/v2/room/alignment/floor/fields.npz')['fields']
    method='linear' if args.preview else 'conv'
    before,wa=atlas_sources(HERE/'data/desktop-scene',base,700,method=method,fields=initial)
    if args.fitted:
        fields=np.load(args.fitted/'fields.npz')['fields'];records=json.loads((args.fitted/'receipt.json').read_text())['regions']
    else:fields,records=fit_regions(before,wa,initial)
    after,wb=atlas_sources(HERE/'data/desktop-scene',base,700,method=method,fields=fields)
    common=np.minimum(wa,wb);covered=common.sum(0)>0
    for name,s in [('before',before),('after',after)]:
        out=average(s,common);save(np.where(covered[...,None],out,[.025,.032,.04]),args.out/(name+'.png'))
        for region in REGIONS:
            x0,y0,x1,y1=region['crop'];save(out[y0:y1,x0:x1],args.out/(region['name']+'-'+name+'.png'))
    for region in records:
        roi=polygon_mask(common.shape[1:],region['polygon']);anchor=region['anchor'];metrics=[]
        for view in region['views']:
            i=view['view'];valid=roi&(common[i]>.2)&(common[anchor]>.2);errors=[]
            for s in [before,after]:
                a=srgb_to_lab(s[anchor])[...,0];b=srgb_to_lab(s[i])[...,0]
                a-=ndimage.gaussian_filter(a,3);b-=ndimage.gaussian_filter(b,3)
                errors.append(float(np.mean(np.abs(a[valid]-b[valid]))))
            metrics.append(dict(view=i,accepted=view['accepted'],pixels=int(valid.sum()),before=errors[0],after=errors[1]))
        region['metrics']=metrics
    receipt=dict(regions=records,files=base['files'],fitting_sampler='linear' if args.fitted or args.preview else method,sampler=method,seconds=time.perf_counter()-start,minimum_jacobian=min(jacobian_min(f) for f in fields),limitations='Manually localized objects and reference-view choices. Affine fits estimated from images; references define coordinates, not ground truth. Shared-capture reporting blocks are not independent data. Unaccepted and insufficient-overlap views retain previous correction. Ordinary averaging, frozen gains, no seam or sharpening changes.')
    np.savez_compressed(args.out/'fields.npz',fields=fields);(args.out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='regions'}),flush=True)
if __name__=='__main__':main()
