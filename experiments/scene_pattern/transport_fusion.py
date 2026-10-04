"""Equal-support ordinary-average rendering in each source camera's image chart.

Only closed, synchronized regional transports supply other observations. Each
panorama retains its own viewpoint; this does not flatten capture origins into
a global spherical camera. Unsupported areas remain visibly single-source.
"""
import argparse,json,time,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy import ndimage
from .core import samples
from .run import image,save
from bfft.effects import _srgb_decode,_srgb_encode

@lru_cache(maxsize=32)
def linear_image(path):return _srgb_decode(image(Path(path))).astype(np.float32)


def view_maps(evidence,anchor):
    views={}
    for track in evidence['tracks']:
        if 'render_anchor' in track and track['render_anchor']!=anchor:continue
        obs={o['capture']:o for o in track['observations']}
        if anchor not in obs:continue
        for link in track.get('transport_links',[]):
            a,b=link['a'][0],link['b'][0]
            if anchor not in (a,b):continue
            moving=b if a==anchor else a;matrix=np.array(link['affine'])
            if anchor==b:matrix=np.linalg.inv(matrix)
            views.setdefault(moving,[]).append(dict(radius=track.get('support_radius',9.),window_power=track.get('window_power',2),reference=np.array(obs[anchor]['xy']),moving=np.array(obs[moving]['xy']),
                reference_centroid=np.array(obs[anchor]['centroid']),moving_centroid=np.array(obs[moving]['centroid']),affine=matrix,track=track['id']))
    return views


def render(evidence,data,out,anchor,width=1536):
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();records=evidence['records'];h,w=evidence['cameras'][anchor]['shape'];oh=max(1,round(h*width/w));shape=(oh,width);yy,xx=np.indices(shape);grid=np.stack([xx/(width-1)*(w-1),yy/(oh-1)*(h-1)],axis=-1)
    ref=samples(linear_image(str(data/records[anchor]['name'])),grid/[w-1,h-1]);sums=[ref.astype(float).copy(),ref.astype(float).copy()];squares=[ref.astype(float)**2,ref.astype(float)**2];weights=np.ones(shape);counts=np.ones(shape,np.uint8);participation=[];source_masks=hashlib.sha256()
    for moving,maps in sorted(view_maps(evidence,anchor).items()):
        numerator=np.zeros((*shape,2));baseline=np.zeros_like(numerator);weight=np.zeros(shape)
        for m in maps:
            center=m['reference'];radius=m['radius'];lo=np.maximum(np.floor((center-radius)/[w-1,h-1]*[width-1,oh-1]).astype(int),0);hi=np.minimum(np.ceil((center+radius)/[w-1,h-1]*[width-1,oh-1]).astype(int)+1,[width,oh]);x0,y0=lo;x1,y1=hi
            if x1<=x0 or y1<=y0:continue
            g=grid[y0:y1,x0:x1];d=g-center;r2=np.sum(d*d,axis=-1)/radius**2;win=np.maximum(0,1-r2)**m['window_power']
            coords=m['moving']+d@m['affine'].T;old=m['moving_centroid']+(g-m['reference_centroid'])
            numerator[y0:y1,x0:x1]+=coords*win[...,None];baseline[y0:y1,x0:x1]+=old*win[...,None];weight[y0:y1,x0:x1]+=win
        active=weight>.02
        if not active.any():continue
        before=baseline[active]/weight[active,None];after=numerator[active]/weight[active,None];mh,mw=evidence['cameras'][moving]['shape'];limit=np.array([mw-1,mh-1]);valid=np.all((before>=0)&(before<=limit)&(after>=0)&(after<=limit),axis=1)
        src=linear_image(str(data/records[moving]['name']));colors=[samples(src,p/limit) for p in (before,after)];valid&=np.all([np.any(c>.002,axis=-1) for c in colors],axis=0);wt=np.minimum(1,weight[active])*valid
        source_masks.update(wt.tobytes());counts[active]+=(wt>.25).astype(np.uint8);weights[active]+=wt
        for k in (0,1):sums[k][active]+=colors[k]*wt[:,None];squares[k][active]+=colors[k]**2*wt[:,None]
        participation.append(dict(capture=moving,regions=len(maps),pixels_above_quarter_weight=int((wt>.25).sum()),effective_pixels=float(wt.sum())))
    mean=[s/weights[...,None] for s in sums];variance=[np.maximum(0,sq/weights[...,None]-m*m).mean(-1) for sq,m in zip(squares,mean)];support=counts>=2;multi=counts>=3
    prefix=f'{anchor:03d}';save(_srgb_encode(ref),out/f'{prefix}-reference.png');save(_srgb_encode(mean[0]),out/f'{prefix}-before.png');save(_srgb_encode(mean[1]),out/f'{prefix}-after.png')
    color=np.zeros((*shape,3));color[counts==1]=[.08,.12,.16];color[counts==2]=[.13,.5,.7];color[counts>=3]=[.22,.84,.61];save(color,out/f'{prefix}-support.png');save(_srgb_encode(mean[1])*.55+color*.45,out/f'{prefix}-overlay.jpg')
    # Dispersion uses identical source weights, visibility intersection and
    # ordinary averaging in both images, at twice the fitting resolution.
    before=float(np.mean(variance[0][support])) if support.any() else None;after=float(np.mean(variance[1][support])) if support.any() else None
    crop_width=min(width,512);crop_height=min(oh,320)
    score=ndimage.uniform_filter((variance[0]-variance[1])*support,size=(crop_height,crop_width),mode='constant')
    valid_score=score[crop_height//2:oh-(crop_height-1)//2,crop_width//2:width-(crop_width-1)//2]
    cy,cx=np.unravel_index(np.argmax(valid_score),valid_score.shape);suggested=[int(cx),int(cy),crop_width,crop_height]
    result=dict(suggested_crop=suggested,any_support_fraction=float((weights>1.000001).mean()),anchor=anchor,name=records[anchor]['name'],shape=list(shape),participation=participation,regions=sum(p['regions'] for p in participation),
        overlap_fraction=float(support.mean()),three_view_fraction=float(multi.mean()),maximum_sources=int(counts.max()),before_dispersion=before,after_dispersion=after,
        dispersion_reduction=(1-after/before) if before and after is not None else None,same_support_sha256=source_masks.hexdigest(),seconds=time.perf_counter()-start)
    (out/f'{prefix}.json').write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--width',type=int,default=1536);p.add_argument('--limit',type=int,default=0);args=p.parse_args();e=json.loads(args.evidence.read_text());anchors=[i for i,r in enumerate(e['records']) if r['panoramic']];anchors=sorted(anchors,key=lambda a:-sum(len(m) for m in view_maps(e,a).values()))
    if args.limit:anchors=anchors[:args.limit]
    rows=[]
    for a in anchors:
        row=render(e,args.data,args.out,a,args.width);rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='participation'}),flush=True)
    packet=dict(views=rows,records=e['records'],policy=dict(radius=9,width=args.width,averaging='linear RGB ordinary weighted mean; fixed weights in before/after',before='centroid translation control',after='synchronized regional affine maps',sampler='linear preview',unsupported='unchanged reference observation'),
        evidence_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='Each output is a source-camera view with fusion only inside measured local support. It is not a globally reconstructed surface, a free virtual camera, super-resolution or a complete 120-image scene fusion.')
    (args.out/'fusion.json').write_text(json.dumps(packet,indent=2))
if __name__=='__main__':main()
