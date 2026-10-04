"""Diagnose ordinary-average mixing and synchronize local capture centers.

Reuses personal_deblurrer Fourier-circle measurements and zero-mean robust
multi-capture graph coordinates. No image is designated a sharp reference.
"""
import argparse,json,time,itertools
from pathlib import Path
import numpy as np
from scipy import ndimage,optimize
from personal_deblurrer.circles import prepare_phase_circle_spectrum,phase_circle_translation_from_spectra
from personal_deblurrer.multicapture_transport import _graph_coordinates
from personal_deblurrer.decomposition import estimate_centered_mixing_phase
from .core import samples,resize,srgb_to_lab
from .fusion import linearize,encode
from .run import image,save,HERE


def connected(count,edges):
    remaining=set(range(count));groups=[]
    while remaining:
        group={remaining.pop()};changed=True
        while changed:
            changed=False
            for a,b in edges:
                if (a in group) != (b in group):
                    new=b if a in group else a
                    if new in remaining:remaining.remove(new);group.add(new);changed=True
        groups.append(sorted(group))
    return groups


def synchronize(count,edges,values,weights):
    centers=np.zeros((count,2));residual=np.zeros((len(edges),2));components=[]
    for group in connected(count,edges):
        if len(group)<2:continue
        lookup={v:i for i,v in enumerate(group)};indices=[i for i,(a,b) in enumerate(edges) if a in lookup and b in lookup]
        local_edges=[(lookup[edges[i][0]],lookup[edges[i][1]]) for i in indices]
        x,r=_graph_coordinates(len(group),local_edges,np.asarray(values)[indices],np.asarray(weights)[indices])
        centers[group]=x;residual[indices]=r
        components.append(dict(vertices=group,edges=indices,cycle_rank=len(indices)-len(group)+1))
    return centers,residual,components


def center_edge(a,b):
    spectra=[prepare_phase_circle_spectrum(x) for x in (a,b)]
    shift,record=phase_circle_translation_from_spectra(*spectra)
    ambiguity=record['weighted_peak_ambiguity'];dispersion=record['translation_dispersion_pixels']
    if ambiguity>.85 or dispersion>2 or np.linalg.norm(shift)>8:return None
    # Fractional polish on separated fitting/validation pixels, initialized by
    # the established Fourier-circle center. A common blur levels bandwidth.
    aa=ndimage.gaussian_filter(a,.8);bb=ndimage.gaussian_filter(b,.8)
    aa=(aa-aa.mean())/max(aa.std(),.005);bb=(bb-bb.mean())/max(bb.std(),.005)
    yy,xx=np.mgrid[10:a.shape[0]-10,10:a.shape[1]-10];target=aa[10:-10,10:-10]
    train=(xx+2*yy)%3!=0
    def residual(d):return ndimage.map_coordinates(bb,[yy+d[1],xx+d[0]],order=1,mode='nearest')-target
    def objective(d):return residual(d)[train].ravel()
    def jac(d):return np.column_stack([(objective(d+v)-objective(d-v))/.04 for v in np.eye(2)*.02])
    result=optimize.least_squares(objective,shift,jac=jac,bounds=(shift-1.5,shift+1.5),loss='soft_l1',f_scale=.3,max_nfev=18)
    before=float(np.mean(np.minimum(residual([0,0])[~train]**2,4)))
    after=float(np.mean(np.minimum(residual(result.x)[~train]**2,4)))
    if after>before*1.02 or np.linalg.norm(result.x)>8:return None
    reliability=(1-ambiguity)**2*np.exp(-(dispersion/1.5)**2)
    return dict(shift=result.x.tolist(),weight=float(reliability),ambiguity=ambiguity,dispersion=dispersion,before=before,after=after)


def atlas_sources(data,base,width,method='linear',fields=None):
    shape=(round(base['shape'][0]*width/base['shape'][1]),width);h,w=shape
    yy,xx=np.indices(shape);lo,hi=np.asarray(base['bounds']);stack=[];weights=[]
    f=28/np.hypot(36,24)*1500;k=np.array([[f/899,0,.5],[0,f/1199,.5],[0,0,1.]])
    for i,(name,matrix) in enumerate(zip(base['files'],base['matrices'])):
        d=np.zeros((h,w,2)) if fields is None else fields[i]
        yaw=lo[0]+(xx+d[...,0])/(w-1)*(hi[0]-lo[0]);pitch=lo[1]+(yy+d[...,1])/(h-1)*(hi[1]-lo[1])
        rays=np.stack([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)],-1)
        xyz=rays@(k@np.asarray(matrix)).T;p=xyz[...,:2]/np.maximum(xyz[...,2:],1e-9)
        valid=(xyz[...,2]>0)&np.all((p>=0)&(p<=1),axis=-1)
        rgb=image(data/name);values=np.zeros((*shape,3),np.float32);values[valid]=samples(rgb,p[valid],method=method)
        weight=np.clip(np.min(np.concatenate((p,1-p),-1),-1)/.12,0,1)*valid
        stack.append(np.clip(values,0,1));weights.append(weight)
        if method=='conv':print(json.dumps({'native_source':name,'corrected':fields is not None}),flush=True)
    corrected=np.clip(encode(linearize(np.asarray(stack))*np.asarray(base['gains'])[:,None,None,:]),0,1)
    return corrected,np.asarray(weights)


def average(stack,weights):
    return encode(np.sum(linearize(stack)*weights[...,None],0)/np.maximum(weights.sum(0)[...,None],1e-20))


def diagnose(stack,weights,patch=64,stride=24):
    n,h,w,_=stack.shape;luma=np.einsum('nhwc,c->nhw',stack,[.2126,.7152,.0722]);avg=average(stack,weights)
    records=[];numerator=np.zeros((n,h,w,2));mass=np.zeros((n,h,w));yy,xx=np.indices((h,w));half=patch//2
    for cy in range(half,h-half,stride):
        for cx in range(half,w-half,stride):
            sl=np.s_[cy-half:cy+half,cx-half:cx+half]
            views=[i for i in range(n) if np.min(weights[i][sl])>.15 and np.std(luma[i][sl])>.015]
            if len(views)<2:continue
            patches=[luma[i][sl] for i in views];edges=[];values=[];reliabilities=[];details=[]
            for i,j in itertools.combinations(range(len(views)),2):
                edge=center_edge(patches[i],patches[j])
                if edge is None:continue
                edges.append((i,j));values.append(edge['shift']);reliabilities.append(edge['weight']);details.append(dict(first=views[i],second=views[j],**edge))
            if not edges:continue
            centers,residual,groups=synchronize(len(views),edges,values,reliabilities)
            closure=float(np.sqrt(np.average(np.sum(residual**2,axis=1),weights=reliabilities)))
            raw_cov=np.zeros((2,2));count=0;authority_by_view=np.zeros(len(views))
            for group in groups:
                indices=group['vertices'];x=centers[indices];raw_cov+=x.T@x;count+=len(x)
                # A two-view edge has no independent cycle. Preserve that limit.
                authority=(1 if group['cycle_rank'] else .25)*np.exp(-(closure/.65)**2)*float(np.mean(np.asarray(reliabilities)[group['edges']]))
                authority_by_view[indices]=authority
            raw_cov/=max(count,1)
            blind=estimate_centered_mixing_phase(avg[sl],radius=10)
            source_blind=[estimate_centered_mixing_phase(stack[i][sl],radius=10) for i in views]
            gaussian=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*(stride*.8)**2))
            for j,i in enumerate(views):
                a=authority_by_view[j];numerator[i]+=gaussian[...,None]*a*centers[j];mass[i]+=gaussian*a
            records.append(dict(center=[cx,cy],views=views,edges=details,centers=centers.tolist(),components=groups,closure_rms=closure,authority=authority_by_view.tolist(),registration_covariance=raw_cov.tolist(),registration_rms=float(np.sqrt(np.trace(raw_cov))),average_blind_covariance=blind.covariance.tolist(),source_blind_trace_median=float(np.median([np.trace(b.covariance) for b in source_blind])),average_blind_confidence=blind.confidence))
    fields=numerator/(mass[...,None]+.3)
    supported=(mass>.001)&(weights>.1)
    fields*=supported[...,None]
    # Maintain a local shared-frame gauge across supported views, rather than
    # anchoring every patch to one photograph or moving the average wholesale.
    mean=np.sum(fields,axis=0)/np.maximum(supported.sum(0)[...,None],1)
    fields=(fields-mean)*supported[...,None]
    norm=np.max(np.linalg.norm(fields,axis=-1),axis=0);fields*=np.minimum(1,6/np.maximum(norm,1e-9))[None,...,None]
    return avg,fields,records


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=HERE/'out/v2/room');p.add_argument('--out',type=Path,default=Path('/tmp/room_alignment'));p.add_argument('--width',type=int,default=700);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();base=json.loads((args.base/'receipt.json').read_text());data=HERE/'data/desktop-scene'
    stack,weights=atlas_sources(data,base,args.width);avg,fields,records=diagnose(stack,weights)
    print(json.dumps({'charts':len(records),'edges':sum(len(r['edges']) for r in records),'diagnostic_seconds':time.perf_counter()-start}),flush=True)
    # Freeze fields before independent native resampling of the original sources.
    original,wa=atlas_sources(data,base,args.width,method='conv');candidate,wb=atlas_sources(data,base,args.width,method='conv',fields=fields)
    fixed=np.minimum(wa,wb);covered=fixed.sum(0)>0;overlap=(fixed>.1).sum(0)>=2
    averages=[average(s,fixed) for s in [original,candidate]];metrics=[]
    for name,s,a in zip(['before','after'],[original,candidate],averages):
        save(np.where(covered[...,None],a,[.025,.032,.04]),args.out/f'{name}.png')
        lab=np.asarray([srgb_to_lab(v) for v in s]);center=np.sum(lab*fixed[...,None],0)/np.maximum(fixed.sum(0)[...,None],1e-20)
        error=np.sqrt(np.sum(np.sum((lab-center)**2,-1)*fixed,0)/np.maximum(fixed.sum(0),1e-20))
        dy,dx=np.gradient(srgb_to_lab(a)[...,0]);metrics.append(dict(name=name,mean_disagreement=float(np.mean(error[overlap])),mean_gradient=float(np.mean(np.hypot(dx,dy)[overlap]))))
        heat=np.clip(error/.12,0,1);save(np.where(covered[...,None],np.stack((heat,heat**2,.15*(1-heat)),-1),[.025,.032,.04]),args.out/f'{name}-disagreement.png')
    # Forward observation cross-prediction on all available source pairs.
    pairs=[]
    for i,j in itertools.combinations(range(len(original)),2):
        mask=(fixed[i]>.25)&(fixed[j]>.25)
        if mask.sum()<300:continue
        scores=[]
        for s in (original,candidate):
            f=[srgb_to_lab(s[v])[...,0] for v in (i,j)]
            f=[v-ndimage.gaussian_filter(v,2) for v in f]
            scores.append(float(np.mean(np.abs(f[0][mask]-f[1][mask]))))
        pairs.append(dict(first=i,second=j,pixels=int(mask.sum()),before=scores[0],after=scores[1]))
    checks=[]
    for i,f in enumerate(fields):
        dy,dx=np.gradient(f,axis=(0,1));det=(1+dx[...,0])*(1+dy[...,1])-dx[...,1]*dy[...,0]
        checks.append(dict(view=i,min_jacobian=float(det.min()),max_shift=float(np.max(np.linalg.norm(f,axis=-1)))))
    receipt=dict(method='Personal deblurrer local Fourier-circle centers, robust connected overlap graphs, component-wise zero-mean gauges, ordinary-average blind mixing diagnostic',files=base['files'],shape=list(avg.shape[:2]),chart_count=len(records),edge_count=sum(len(r['edges']) for r in records),records=records,metrics=metrics,pairs=pairs,field_checks=checks,total_seconds=time.perf_counter()-start,ground_truth_used=False,limitations='Local translation charts approximate complex spatial transport. Average-only phase covariance is not a unique registration estimate. A common translation and common blur remain unidentifiable. Spatially correlated holdouts and cycle closure do not certify geometry. Final checks use native source sampling, not independent scene truth.')
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2));np.savez_compressed(args.out/'fields.npz',fields=fields,masks=fixed,original=original,candidate=candidate)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['records','pairs','files']}),flush=True)
if __name__=='__main__':main()
