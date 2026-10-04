"""Expand automatic panorama evidence with bidirectionally checked texture flow."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage.registration import optical_flow_tvl1
from .core import samples,resize,srgb_to_lab
from .sweep_mesh import evaluate,solve
from .sweep_fusion import raster_map,render,attach


def carrier(rgb):
    l=srgb_to_lab(rgb)[...,0];high=l-ndimage.gaussian_filter(l,4)
    contrast=np.sqrt(ndimage.gaussian_filter(high**2,4)+.001)
    return np.clip(high/contrast,-2,2).astype(np.float32)


def track(a,b,wa,wb,qa,inverseb):
    common=ndimage.binary_erosion((wa>.3)&(wb>.3),iterations=4);height,width=wa.shape
    if common.sum()<400:return np.empty(0,dtype=int),np.empty((0,2)),dict(accepted=0,reason='little_common_support')
    aa=carrier(a);bb=carrier(b);aa=np.where(common,aa,0);bb=np.where(common,bb,0)
    f=optical_flow_tvl1(aa,bb,attachment=12,tightness=.3,num_warp=7,num_iter=10,prefilter=True)
    back=optical_flow_tvl1(bb,aa,attachment=12,tightness=.3,num_warp=7,num_iter=10,prefilter=True)
    field=np.moveaxis(f[[1,0]],0,-1);reverse=np.moveaxis(back[[1,0]],0,-1)
    q=qa/[width-1,height-1];delta=samples(field,q);qb=qa+delta
    cycle=np.linalg.norm(delta+samples(reverse,qb/[width-1,height-1]),axis=-1)
    yy,xx=np.indices((height,width));p=np.stack([xx,yy],-1)+field
    moved=samples(bb,p/[width-1,height-1]);cross=ndimage.gaussian_filter(aa*moved,2)
    correlation=cross/np.sqrt(ndimage.gaussian_filter(aa**2,2)*ndimage.gaussian_filter(moved**2,2)+1e-8)
    corr=samples(correlation,q);valid=(samples(common.astype(float),q)>.99)&(samples(common.astype(float),qb/[width-1,height-1])>.99)&(cycle<1.2)&(corr>.65)&(np.linalg.norm(delta,axis=1)<60)
    pb=samples(inverseb,qb/[width-1,height-1]);valid&=np.all(np.isfinite(pb),axis=1)
    ids=np.flatnonzero(valid)
    return ids,pb[valid],dict(accepted=len(ids),median_correlation=float(np.median(corr[valid])) if len(ids) else None,median_cycle=float(np.median(cycle[valid])) if len(ids) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--attach',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    g=json.loads(args.graph.read_text());r=json.loads((args.scene/'receipt.json').read_text());records=g['records'];features=[np.load(args.features/(Path(row['name']).stem+'.npz')) for row in records];shapes=[f['shape'] for f in features]
    with np.load(args.scene/'panorama-meshes.npz') as f:meshes={int(k):f[k] for k in f.files}
    period=np.array(r['period']['period']);anchor=r['anchor'];lo=np.array(r['render']['origin']);scale=r['render']['scale']/2;shape=[int(np.ceil(x/2)) for x in r['render']['shape']];images={};weights={};inverse={};sites={}
    for i,m in meshes.items():
        with np.load(args.scene/f'layer-{i:03}.npz') as f:images[i]=resize(f['rgb'],shape);weights[i]=resize(f['weight'],shape)
        sheared=np.stack([m[...,0],m[...,1]-m[...,0]*period[1]/period[0]],-1);imap=np.full((*shape,2),np.nan,np.float32)
        for turn in range(int(np.floor(sheared[...,0].min()/period[0]))-1,int(np.ceil(sheared[...,0].max()/period[0]))+2):
            proposal=raster_map(sheared-[turn*period[0],0],shapes[i],lo,scale,shape);valid=np.isfinite(proposal[...,0]);imap[valid]=proposal[valid]
        inverse[i]=imap
        valid=np.isfinite(imap[...,0]);normalized=imap[valid]/np.array([shapes[i][1]-1,shapes[i][0]-1]);values=np.zeros((*shape,3),np.float32);values[valid]=samples(features[i]['rgb'],normalized);images[i]=values
        weight=np.zeros(shape,np.float32);weight[valid]=np.clip(np.min(np.c_[normalized,1-normalized],axis=1)/.06,0,1);weights[i]=weight
        points=features[i]['points'];_,keep=np.unique(np.rint(points),axis=0,return_index=True);points=points[keep]
        world=evaluate(m,points,shapes[i]);world[:,1]-=world[:,0]*period[1]/period[0];world[:,0]%=period[0];sites[i]=(points,(world-lo)*scale)
    edges=[e for e in g['edges'] if e['accepted']];augmented=[];audit=[]
    for e in edges:
        if e['stage']!=1 or e['a'] not in meshes or e['b'] not in meshes:augmented.append(e);continue
        a,b=e['a'],e['b'];pa,qa=sites[a];ids,pb,receipt=track(images[a],images[b],weights[a],weights[b],qa,inverse[b]);receipt.update(a=a,b=b);audit.append(receipt)
        if len(ids)>=16:
            updated=dict(e,a_points=e['a_points']+pa[ids].tolist(),b_points=e['b_points']+pb.tolist(),inliers=e['inliers']+len(ids),dense_verified=len(ids));augmented.append(updated)
        else:augmented.append(e)
        print(json.dumps({'dense_edge':len(audit),**receipt}),flush=True)
    panoedges=[e for e in augmented if e['stage']==1];pano,diag=solve(shapes,panoedges,meshes,{anchor},period=period)
    np.savez_compressed(args.out/'panorama-meshes.npz',**{str(i):m for i,m in pano.items()})
    if args.attach:
        base,unresolved=attach(shapes,pano,augmented,period=period);final,finaldiag=solve(shapes,augmented,base,set(pano),period=period)
    else:final=pano;finaldiag=diag;unresolved=sorted(i for i,row in enumerate(records) if row['panoramic'] and i not in pano)
    rendering=render(args.data,records,shapes,final,augmented,finaldiag,args.out,period=period)
    r.update(stage1=diag,stage2=finaldiag,stage1_images=sorted(pano),stage2_images=sorted(set(final)-set(pano)),render=rendering,unresolved=unresolved,panoramas_only=not args.attach,
        dense_audit=audit,seconds=time.perf_counter()-start,dense_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez_compressed(args.out/'meshes.npz',**{str(i):m for i,m in final.items()});(args.out/'receipt.json').write_text(json.dumps(r,indent=2));g['edges']=augmented;(args.out/'dense-graph.json').write_text(json.dumps(g,indent=2))
    print(json.dumps({'finished':len(final),'unresolved':unresolved,'seconds':r['seconds']}),flush=True)
if __name__=='__main__':main()
