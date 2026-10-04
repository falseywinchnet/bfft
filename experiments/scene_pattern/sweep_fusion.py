"""Two-stage mesh fusion and source-attributed information promotion."""
import argparse,json,time,shutil,hashlib,platform
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.transform import SimilarityTransform
from .sweep_mesh import initialize,solve,evaluate,lattice,panorama_initialization,periodic_panorama_initialization,unwrap_points
from .sweep_graph import components
from .core import samples
from .fusion import linearize,encode
from .run import image,save


def attach(shapes,panos,edges,period=None):
    base={i:m.copy() for i,m in panos.items()};pending=set(range(len(shapes)))-set(base)
    while pending:
        proposals=[]
        for i in sorted(pending):
            local=[];world=[]
            for e in edges:
                if e['a']==i and e['b'] in base:
                    local.extend(e['a_points']);world.extend(evaluate(base[e['b']],e['b_points'],shapes[e['b']]))
                elif e['b']==i and e['a'] in base:
                    local.extend(e['b_points']);world.extend(evaluate(base[e['a']],e['a_points'],shapes[e['a']]))
            if len(local)>=10:
                m=SimilarityTransform()
                if m.estimate(np.array(local),unwrap_points(world,period)) and .12<m.scale<8:
                    proposals.append((i,m,len(local)))
        if not proposals:break
        # All candidates in one frontier see the same previously attached set.
        for i,m,n in proposals:
            p,_=lattice(shapes[i]);base[i]=m(p.reshape(-1,2)).reshape(p.shape);pending.remove(i)
    return base,sorted(pending)


def raster_map(mesh,source_shape,lo,scale,canvas_shape):
    """Inverse piecewise-affine sampling map, no clamped edge extension."""
    height,width=canvas_shape;xy=np.full((height,width,2),np.nan,np.float32)
    src,_=lattice(source_shape);world=(mesh-lo)*scale
    for y in range(world.shape[0]-1):
        for x in range(world.shape[1]-1):
            for offsets in (((0,0),(0,1),(1,0)),((1,1),(1,0),(0,1))):
                a=np.array([world[y+dy,x+dx] for dy,dx in offsets]);b=np.array([src[y+dy,x+dx] for dy,dx in offsets])
                lower=np.maximum(np.floor(a.min(0)).astype(int),[0,0]);upper=np.minimum(np.ceil(a.max(0)).astype(int),[width-1,height-1])
                if np.any(upper<lower):continue
                mat=np.stack([a[1]-a[0],a[2]-a[0]],1)
                if np.linalg.det(mat)<=1e-8:continue
                yy,xx=np.meshgrid(np.arange(lower[1],upper[1]+1),np.arange(lower[0],upper[0]+1),indexing='ij')
                q=np.stack([xx,yy],-1);uv=(q-a[0])@np.linalg.inv(mat).T
                valid=(uv[...,0]>=-1e-5)&(uv[...,1]>=-1e-5)&(uv.sum(-1)<=1+1e-5)
                points=b[0]+uv[...,0,None]*(b[1]-b[0])+uv[...,1,None]*(b[2]-b[0])
                patch=xy[lower[1]:upper[1]+1,lower[0]:upper[0]+1];patch[valid]=points[valid]
    return xy


def render(data,records,shapes,meshes,edges,diagnostics,out,width=1600,sampler='linear',period=None):
    out.mkdir(parents=True,exist_ok=True)
    if period is not None:
        meshes={i:np.stack([m[...,0],m[...,1]-m[...,0]*period[1]/period[0]],-1) for i,m in meshes.items()}
    allpoints=np.concatenate([m.reshape(-1,2) for m in meshes.values()]);lo=allpoints.min(0)-2;hi=allpoints.max(0)+2
    if period is not None:lo[0]=0;hi[0]=period[0]
    scale=min(width/(hi[0]-lo[0]),1000/(hi[1]-lo[1]));w=int(np.ceil((hi[0]-lo[0])*scale));h=int(np.ceil((hi[1]-lo[1])*scale))
    summed=np.zeros((h,w,3),np.float64);squared=np.zeros_like(summed);mass=np.zeros((h,w),np.float64)
    panosum=np.zeros_like(summed);panomass=np.zeros_like(mass);support=np.zeros((h,w),np.uint16)
    ids=sorted(meshes);layers=[];qualities=[];perimage=[]
    residual={}
    for r in diagnostics['residuals']:
        for i in (r['a'],r['b']):residual.setdefault(i,[]).append(r['median'])
    for i in ids:
        start=time.perf_counter();rgb=image(data/records[i]['name']);xy=raster_map(meshes[i],shapes[i],lo,scale,(h,w))
        if period is not None:
            for turn in range(int(np.floor(meshes[i][...,0].min()/period[0]))-1,int(np.ceil(meshes[i][...,0].max()/period[0]))+2):
                if turn==0:continue
                extra=raster_map(meshes[i]-[turn*period[0],0],shapes[i],lo,scale,(h,w));available=np.isfinite(extra[...,0]);xy[available]=extra[available]
        valid=np.isfinite(xy[...,0]);points=xy[valid]/np.array([shapes[i][1]-1,shapes[i][0]-1]);values=np.zeros((h,w,3),np.float32)
        values[valid]=np.clip(samples(rgb,points,method=sampler),0,1);lin=linearize(values).astype(np.float32)
        border=np.min(np.c_[points,1-points],axis=1);weight=np.zeros((h,w),np.float32);weight[valid]=np.clip(border/.06,0,1)
        # Registration support is measured at actual cross-image region ties.
        controls=[]
        for e in edges:
            if e['a']==i:controls.extend(e['a_points'])
            elif e['b']==i:controls.extend(e['b_points'])
        quality=np.zeros((h,w),np.float32)
        if controls:
            distance=cKDTree(controls).query(xy[valid])[0]
            uncertainty=float(np.median(residual.get(i,[20.])))
            quality[valid]=np.exp(-distance/64)/(1+(uncertainty/6)**2)
        summed+=lin*weight[...,None];squared+=lin*lin*weight[...,None];mass+=weight;support+=weight>.05
        if records[i]['panoramic']:panosum+=lin*weight[...,None];panomass+=weight
        # Lossless source layer cache keeps provenance and permits later views.
        np.savez_compressed(out/f'layer-{i:03}.npz',rgb=values,weight=weight,quality=quality)
        layers.append(i);qualities.append(float(np.mean(quality[valid])) if valid.any() else 0)
        perimage.append(dict(index=i,covered=int(valid.sum()),median_graph_residual=float(np.median(residual.get(i,[20.]))),mean_support_confidence=qualities[-1],seconds=time.perf_counter()-start))
        print(json.dumps({'rendered':records[i]['name'],**perimage[-1]}),flush=True)
    mean=summed/np.maximum(mass[...,None],1e-8);variance=np.maximum(squared/np.maximum(mass[...,None],1e-8)-mean**2,0);spread=np.sqrt(variance.mean(-1))
    pano=panosum/np.maximum(panomass[...,None],1e-8);numerator=np.zeros_like(summed);denominator=np.zeros_like(mass);best=np.full((h,w),-1.);chosen=np.full((h,w),-1,np.int16)
    for i in ids:
        with np.load(out/f'layer-{i:03}.npz') as f:rgb=f['rgb'];weight=f['weight'];quality=f['quality']
        lin=linearize(rgb);difference=np.sqrt(np.mean((lin-mean)**2,axis=-1));agreement=1/(1+(difference/np.maximum(.025,spread))**2)
        light=rgb.mean(-1);detail=np.abs(light-ndimage.gaussian_filter(light,1.2))
        # A bounded detail preference cannot turn texture strength into geometry.
        promotion=weight*quality*agreement*(1+np.minimum(detail/.04,1.))
        numerator+=lin*promotion[...,None];denominator+=promotion
        replace=promotion>best;chosen[replace & (weight>0)]=i;best=np.maximum(best,promotion)
    promoted=numerator/np.maximum(denominator[...,None],1e-8);bg=np.array([.025,.032,.04]);covered=mass>0
    save(np.where(covered[...,None],encode(mean),bg),out/'average.png');save(np.where(panomass[...,None]>0,encode(pano),bg),out/'panoramas.png')
    save(np.where(covered[...,None],encode(promoted),bg),out/'promoted.png');heat=np.clip(spread/.18,0,1);save(np.where(covered[...,None],np.stack([heat,heat**2,.15*(1-heat)],-1),bg),out/'disagreement.png')
    save(np.repeat((support/max(1,support.max()))[...,None],3,-1),out/'coverage.png')
    np.savez_compressed(out/'scene.npz',support=support,disagreement=spread,preferred_source=chosen,origin=lo,scale=scale)
    # Keep source images separate from the coordinate chart for a future
    # parallax/depth representation; the current viewer claims only 2-D meshes.
    return dict(shape=[h,w],origin=lo.tolist(),scale=scale,sampler=sampler,images=perimage,
        covered=int(covered.sum()),multiple_view_fraction=float(np.mean(support[covered]>1)),mean_disagreement=float(spread[support>1].mean()),
        promotion='bounded local detail preference times source support, graph consistency and color agreement; not super-resolution or depth',
        source_contributions={str(i):int(np.sum(chosen==i)) for i in ids})


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--sampler',choices=['linear','conv'],default='linear');p.add_argument('--panoramas-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    graph=json.loads(args.graph.read_text());records=graph['records'];shapes=[np.load(args.features/(Path(r['name']).stem+'.npz'))['shape'] for r in records];edges=[e for e in graph['edges'] if e['accepted']];panoedges=[e for e in edges if e['stage']==1]
    panoids={i for i,r in enumerate(records) if r['panoramic']};comps=components(len(records),panoedges);component=max(comps,key=lambda c:len(set(c)&panoids));panoids&=set(component)
    anchor=max(sorted(panoids),key=lambda i:sum(e['inliers'] for e in panoedges if i in (e['a'],e['b'])))
    initial,period,period_receipt=periodic_panorama_initialization(shapes,[e for e in panoedges if e['a'] in panoids and e['b'] in panoids],anchor)
    pano,pdiag=solve(shapes,panoedges,initial,{anchor},period=period)
    np.savez_compressed(args.out/'panorama-meshes.npz',**{str(i):m for i,m in pano.items()})
    (args.out/'panorama-solve.json').write_text(json.dumps(pdiag,indent=2))
    print(json.dumps({'stage':1,'panoramas':len(pano),'anchor':anchor,'minimum_area_ratio':pdiag['minimum_area_ratio']}),flush=True)
    if args.panoramas_only:
        final=pano;diag=pdiag;unresolved=sorted({i for i,r in enumerate(records) if r['panoramic']}-set(pano))
    else:
        second,unresolved=attach(shapes,pano,edges,period=period);final,diag=solve(shapes,edges,second,set(pano),period=period)
    # Stage 1 coordinate values must remain bit-for-bit fixed in stage 2.
    assert all(np.array_equal(pano[i],final[i]) for i in pano)
    np.savez_compressed(args.out/'meshes.npz',**{str(i):m for i,m in final.items()})
    rendering=render(args.data,records,shapes,final,edges,diag,args.out,sampler=args.sampler,period=period)
    receipt=dict(source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("sweep*.py")},runtime=dict(python=platform.python_version(),numpy=np.__version__),policy=graph['policy'],records=records,anchor=anchor,stage1_images=sorted(pano),stage2_images=sorted(set(final)-set(pano)),unresolved=unresolved,
        period=period_receipt,panoramas_only=args.panoramas_only,stage1=pdiag,stage2=diag,render=rendering,seconds=time.perf_counter()-start,manual_annotations=False,
        limitations='Automatic development prototype. Sparse region graph and piecewise affine meshes, not calibrated depth or complete visibility reconstruction. A connected graph is not proof of correct correspondence. All-source ordinary average is retained next to information-weighted fusion. No room repair fields or object annotations are used.')
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'finished':len(final),'unresolved':unresolved,'seconds':receipt['seconds']}),flush=True)
if __name__=='__main__':main()
