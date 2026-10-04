"""Grow image-verified regional maps, then check continuous three-view loops."""
import argparse,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from .capture_transport import feature
from .region_transport import fit_affine_regions

DENSE_POLICY=dict(grid_step=8,margin=14,minimum_seed_maps=24,propagation_rounds=3,maximum_seed_distance=24,
    interpolation_radius=6.,cycle_pixels=1.5,cycle_jacobian=.45)


def lattice(shape):
    y=np.arange(14,shape[0]-14,8);x=np.arange(14,shape[1]-14,8);yy,xx=np.meshgrid(y,x,indexing='ij');return np.stack([xx,yy],axis=-1)


def from_prior(a,b,pa,pb,initial):
    f=fit_affine_regions(a,b,pa,pb,initial,iterations=14);r=fit_affine_regions(b,a,pb,pa,np.linalg.inv(initial),iterations=14)
    closure=np.linalg.norm(r['center']+np.einsum('nij,nj->ni',r['affine'],f['center']-pb)-pa,axis=1);jac=np.linalg.norm(r['affine']@f['affine']-np.eye(2),axis=(1,2))
    valid=f['admissible']&r['admissible']&(f['train_correlation']>=.78)&(f['validation_correlation']>=.73)&(r['validation_correlation']>=.73)&(closure<=1.5)&(jac<.6)&(f['information_ratio']>=.012)
    return dict(target=f['center'],affine=f['affine'],valid=valid,correlation=f['validation_correlation'],training=f['train_correlation'],prior=pb,prior_affine=initial,reverse_source=pb,reverse_target=r['center'],reverse_affine=r['affine'])


class Field:
    def __init__(self,p,q,a,quality=None):
        self.p=np.asarray(p);self.q=np.asarray(q);self.a=np.asarray(a);self.tree=cKDTree(self.p) if len(p) else None;self.quality=np.ones(len(p)) if quality is None else np.asarray(quality)
    def at(self,p,radius=6.):
        p=np.asarray(p);n=len(p)
        if self.tree is None:return np.zeros((n,2)),np.zeros((n,2,2)),np.zeros(n,bool)
        d,k=self.tree.query(p,k=min(3,len(self.p)));d=np.reshape(d,(n,-1));k=np.reshape(k,(n,-1));pred=self.q[k]+np.einsum('nkij,nkj->nki',self.a[k],p[:,None,:]-self.p[k]);coherent=np.linalg.norm(pred-pred[:,:1],axis=-1)<=1.5;wt=(d<=radius)*coherent/(d*d+.25);total=wt.sum(1);q=np.sum(pred*wt[...,None],axis=1)/np.maximum(total[:,None],1e-9);a=np.sum(self.a[k]*wt[...,None,None],axis=1)/np.maximum(total[:,None,None],1e-9)
        return q,a,total>0


def grow(spec):
    a,b,meta,root,features,out=spec;start=time.perf_counter();file=Path(root)/f'{a:03d}-{b:03d}.npz';fa=feature(str(Path(features)/(Path(meta['records'][a]['name']).stem+'.npz')));fb=feature(str(Path(features)/(Path(meta['records'][b]['name']).stem+'.npz')))
    with np.load(file) as f:
        good=f['well_localized'];sp=f['source'][good];sq=f['center'][good];sa=f['affine'][good];quality=f['validation_correlation'][good]
    grid=lattice(fa['shape']);p=grid.reshape(-1,2);n=len(p);accepted=np.zeros(n,bool);result={};progress=[]
    for round_index in range(3):
        if not len(sp):break
        todo=np.flatnonzero(~accepted);d,k=cKDTree(sp).query(p[todo]);chosen=todo[d<=24]
        if not len(chosen):break
        k=k[d<=24];initial=sa[k];pb=sq[k]+np.einsum('nij,nj->ni',initial,p[chosen]-sp[k]);r=from_prior(fa['lab'],fb['lab'],p[chosen],pb,initial)
        # Fixed training score selects the proposal. Pixel validation and reverse
        # transport determine whether it enters the next propagation frontier.
        keep=chosen[r['valid']]
        if not result:
            for key,value in r.items():result[key]=np.zeros((n,*value.shape[1:]),value.dtype)
        for key,value in r.items():result[key][keep]=value[r['valid']]
        accepted[keep]=True;progress.append(dict(round=round_index,attempted=len(chosen),accepted=len(keep)))
        if not len(keep):break
        sp=np.r_[sp,p[keep]];sq=np.r_[sq,r['target'][r['valid']]];sa=np.r_[sa,r['affine'][r['valid']]];quality=np.r_[quality,r['correlation'][r['valid']]]
    dest=Path(out)/f'{a:03d}-{b:03d}.npz';np.savez_compressed(dest,grid=grid,**result)
    # The opposite field uses independently fitted reverse patches, not an
    # algebraic inversion of the accepted forward result.
    reverse_grid=lattice(fb['shape']);rp=reverse_grid.reshape(-1,2);rev=Field(result['reverse_source'][accepted],result['reverse_target'][accepted],result['reverse_affine'][accepted]);rq,ra,rv=rev.at(rp)
    np.savez_compressed(Path(out)/f'{b:03d}-{a:03d}.npz',grid=reverse_grid,target=rq,affine=ra,valid=rv,prior=rq.copy(),prior_affine=np.tile(np.eye(2),(len(rp),1,1)),correlation_lower_bound=np.full(len(rp),.73))
    row=dict(a=a,b=b,forward=int(accepted.sum()),reverse=int(rv.sum()),grid_points=n,rounds=progress,seconds=time.perf_counter()-start);(dest.with_suffix('.json')).write_text(json.dumps(row));return row


def check_cycles(folder,meta):
    fields={};grids={};rows={};support={};compatible={}
    for row in meta['pairs']:
        for a,b in ((row['a'],row['b']),(row['b'],row['a'])):
            file=folder/f'{a:03d}-{b:03d}.npz'
            if not file.exists():continue
            with np.load(file) as f:record=dict(f)
            grid=record['grid'];p=grid.reshape(-1,2);valid=record['valid'];fields[a,b]=Field(p[valid],record['target'][valid],record['affine'][valid]);grids[a]=grid;rows[a,b]=record;support[a,b]=np.zeros(len(p),np.uint16);compatible[a,b]=np.zeros((len(p),len(meta['records'])),bool)
    receipts=[]
    for a in sorted(grids):
        neighbors=sorted(b for x,b in fields if x==a);p=grids[a].reshape(-1,2)
        for j,b in enumerate(neighbors):
            ab=rows[a,b];q=ab['target'];jab=ab['affine']
            for c in neighbors[j+1:]:
                if (b,c) not in fields:continue
                ac=rows[a,c];active=ab['valid']&ac['valid'];ids=np.flatnonzero(active)
                if not len(ids):continue
                bc,jbc,v=fields[b,c].at(q[ids]);scale=np.maximum(np.sqrt(np.abs(np.linalg.det(ac['affine'][ids]))),.25);closure=np.linalg.norm(bc-ac['target'][ids],axis=1)/scale;jac=np.linalg.norm(jbc@jab[ids]-ac['affine'][ids],axis=(1,2))/np.maximum(np.linalg.norm(ac['affine'][ids],axis=(1,2)),.25)
                good=v&(closure<=1.5)&(jac<=.45);support[a,b][ids[good]]+=1;support[a,c][ids[good]]+=1;compatible[a,b][ids[good],c]=True;compatible[a,c][ids[good],b]=True
                if good.any():receipts.append(dict(a=a,b=b,c=c,points=int(good.sum()),median_closure=float(np.median(closure[good]))))
    for (a,b),row in rows.items():row['third_view_support']=support[a,b];row['compatible_cameras']=compatible[a,b];np.savez_compressed(folder/f'{a:03d}-{b:03d}.npz',**row)
    return dict(triangles=receipts,validated_directed_sites=int(sum((s>0).sum() for s in support.values())),total_directed_sites=int(sum(r['valid'].sum() for r in rows.values())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--transport',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--probe',type=int,default=0);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);meta=json.loads((args.transport/'transport.json').read_text());pairs=[r for r in meta['pairs'] if r['localized']>=24]
    if args.probe:pairs=pairs[:args.probe]
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(grow,[(r['a'],r['b'],meta,str(args.transport),str(args.features),str(args.out)) for r in pairs]):rows.append(row);print(json.dumps(row),flush=True)
    cycles=check_cycles(args.out,dict(pairs=rows,records=meta['records']));packet=dict(policy=DENSE_POLICY,records=meta['records'],cameras=meta['cameras'],pairs=rows,cycles=cycles,seconds=time.perf_counter()-start,source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('dense_transport.py','region_transport.py')});(args.out/'dense.json').write_text(json.dumps(packet,indent=2));print(json.dumps(dict(pairs=len(rows),validated_directed_sites=cycles['validated_directed_sites'],total_directed_sites=cycles['total_directed_sites'],seconds=packet['seconds'])),flush=True)
if __name__=='__main__':main()
