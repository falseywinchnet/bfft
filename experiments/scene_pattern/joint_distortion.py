"""Apply evidence-checked joint distortion models to the existing room candidate."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import ndimage
from .run import HERE,save
from .structure_alignment import polygon_mask,compose_field,jacobian_min,REGIONS
from .shade_diagnostic import POLYGON as SHADE
from .distortion_basis import fit_models,map_basis
from .registration_audit import contrast_error,nonregressing_regions
from .average_alignment import average,atlas_sources


def supported_extension(raw,support,width):
    """Extend measured boundary displacements, never the unbounded model itself."""
    distance,nearest=ndimage.distance_transform_edt(~support,return_indices=True)
    return raw[nearest[0],nearest[1]]*np.exp(-(distance/width)**2)[...,None]

def harmonic_extension(raw,support,width,allowed=None):
    """Screened harmonic continuation with observed displacements held fixed."""
    from scipy.sparse.linalg import LinearOperator,cg
    import inspect
    free=~support.copy()
    if allowed is not None:free &= allowed
    free[[0,-1]]=False;free[:,[0,-1]]=False
    n=int(free.sum());diagonal=4+1/width**2
    def neighbors(a):return np.roll(a,1,0)+np.roll(a,-1,0)+np.roll(a,1,1)+np.roll(a,-1,1)
    def matvec(z):
        a=np.zeros(support.shape);a[free]=z
        return (diagonal*a-neighbors(a))[free]
    operator=LinearOperator((n,n),matvec=matvec,dtype=np.float64)
    initial=supported_extension(raw,support,width);result=raw.copy();checks=[]
    tolerance={'rtol':1e-6} if 'rtol' in inspect.signature(cg).parameters else {'tol':1e-6}
    for k in range(2):
        fixed=np.where(support,raw[...,k],0);rhs=neighbors(fixed)[free]
        x,info=cg(operator,rhs,x0=initial[...,k][free],atol=0,maxiter=1500,**tolerance)
        relative=float(np.linalg.norm(matvec(x)-rhs)/max(np.linalg.norm(rhs),1e-20))
        if info!=0 or relative>1.1e-6:raise ValueError('Continuation did not converge')
        result[...,k][free]=x;checks.append(relative)
    return result,checks

def fit_joint(s,w,initial,saved=None,base=None,regions=None,reference=2,views=None,allowed=None,allow_visibility_change=False,fit_sources=None,fit_weights=None):
    absolute_fit=fit_sources is not None
    if absolute_fit and base is None:raise ValueError('Absolute fitting requires direct original-capture validation')
    if fit_sources is None:fit_sources,fit_weights=s,w
    if regions is None:regions=[polygon_mask(w.shape[1:],p) for p in [SHADE,REGIONS[0]['polygon']]]
    fields=initial.copy();yy,xx=np.indices(w.shape[1:]);points=np.stack((xx,yy),-1);selection=(xx//4+2*(yy//4))%3==1;records=[]
    for i in ([0,1,3] if views is None else views):
        print(json.dumps(dict(fitting_view=i)),flush=True)
        r=fit_models(s[reference],fit_sources[i],w[reference],fit_weights[i],regions,allow_visibility_change=allow_visibility_change) if saved is None else next(v for v in saved if v['view']==i);r['view']=i;r['applied']=False;r['fit_coordinates']='original_atlas' if absolute_fit else 'previously_warped_atlas'
        for trial in r['trials']:
            trial.pop('extension_rejected',None);trial.pop('contrast_selection',None)
        if r['status']=='accepted':
            support=np.logical_or.reduce(regions)&(w[reference]>.2)&(fit_weights[i]>.2)
            eligible=sorted([t for t in r['trials'] if t['accepted']],key=lambda t:t['selection_mean']*(1+.015*(len(t['parameters'])-6)))
            for t in eligible:
                # Evaluate the chosen basis only where it has observed support.
                raw=np.zeros_like(initial[i]);raw[support]=map_basis(points[support],np.asarray(t['parameters']),np.asarray(r['center']),t['model'])-points[support]
                if absolute_fit:raw[support]-=initial[i][support]
                for width in [24,48,96,160,256]:
                    delta,residuals=harmonic_extension(raw,support,width,allowed=allowed);proposed=initial[i]+delta if absolute_fit else compose_field(initial[i],delta);minimum=jacobian_min(proposed)
                    if minimum>.2:break
                if minimum<=.2:t['extension_rejected']=True;continue
                q=points+delta;moving=np.stack([ndimage.map_coordinates(s[i,...,k],[q[...,1],q[...,0]],order=1,mode='constant') for k in range(3)],-1)
                weight=ndimage.map_coordinates(w[i],[q[...,1],q[...,0]],order=1,mode='constant')
                if base is not None:
                    subset=dict(base)
                    for key in ['files','matrices','gains']:subset[key]=[base[key][i]]
                    direct,direct_weight=atlas_sources(HERE/'data/desktop-scene',subset,700,method='linear',fields=proposed[None])
                    moving=direct[0];weight=direct_weight[0]
                checks=[]
                for k in r['active_regions']:
                    mask=regions[k]&(w[reference]>.2)&(w[i]>.2)&(weight>.2)&selection
                    a=contrast_error(s[reference],s[i],mask);b=contrast_error(s[reference],moving,mask);checks.append(dict(region=k,before=a,after=b,pixels=int(mask.sum())))
                t['contrast_selection']=checks
                if not nonregressing_regions(checks):continue
                # One final holdout gate; failure retains the prior map rather
                # than searching more alternatives against the held-out data.
                holdout=(xx//4+2*(yy//4))%3==2;final_checks=[]
                for k in r['active_regions']:
                    mask=regions[k]&(w[reference]>.2)&(w[i]>.2)&(weight>.2)&holdout
                    a=contrast_error(s[reference],s[i],mask);b=contrast_error(s[reference],moving,mask)
                    final_checks.append(dict(region=k,before=a,after=b,pixels=int(mask.sum())))
                r['final_holdout']=final_checks
                if not nonregressing_regions(final_checks):
                    r['status']='retained_prior_after_holdout_regression';break
                fields[i]=proposed;r.update(applied=True,selected=t,transition=width,minimum_jacobian=minimum,continuation_residuals=residuals);break
        records.append(r);print(json.dumps(dict(view=i,status=r['status'],applied=r['applied'],model=r.get('selected',{}).get('model') if r.get('selected') else None)),flush=True)
    return fields,records,regions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('/tmp/joint_distortion'));ap.add_argument('--native',action='store_true');ap.add_argument('--fitted',type=Path);ap.add_argument('--hypotheses',type=Path);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    initial=np.load(HERE/'out/v2/room/alignment/structures/fields.npz')['fields'];base=json.loads((HERE/'out/v2/room/receipt.json').read_text())
    if args.fitted:
        r=json.loads((args.fitted/'receipt.json').read_text());records=r['views'];fields=np.load(args.fitted/'fields.npz')['fields'];regions=[polygon_mask(initial.shape[1:3],p) for p in [SHADE,REGIONS[0]['polygon']]]
    else:
        z=np.load(HERE/'out/shade-diagnostic/observations.npz');fields,records,regions=fit_joint(z['current'],z['weights'],initial,json.loads((args.hypotheses/'receipt.json').read_text())['views'] if args.hypotheses else None,base=base)
    method='conv' if args.native else 'linear'
    before,wa=atlas_sources(HERE/'data/desktop-scene',base,700,method=method,fields=initial)
    after,wb=atlas_sources(HERE/'data/desktop-scene',base,700,method=method,fields=fields);common=np.minimum(wa,wb);covered=common.sum(0)>0
    for name,s in [('before',before),('after',after)]:
        out=average(s,common);save(np.where(covered[...,None],out,[.025,.032,.04]),args.out/f'{name}.png');save(out[45:300,65:274],args.out/f'lamp-{name}.png')
    metrics=[]
    for r in records:
        i=r['view'];m=[]
        for k,region in enumerate(regions):
            valid=region&(common[2]>.2)&(common[i]>.2)
            m.append(dict(region=k,pixels=int(valid.sum()),before=contrast_error(before[2],before[i],valid),after=contrast_error(after[2],after[i],valid)))
        metrics.append(dict(view=i,applied=r['applied'],regions=m))
    receipt=dict(views=records,metrics=metrics,sampler=method,seconds=time.perf_counter()-start,minimum_jacobian=min(jacobian_min(f) for f in fields),limitations='Joint shade/shaft diagnostic regions and reference photo 3 remain manually selected. Distortion parameters and model choices are estimated, with region-balanced fitting and per-region selection checks. Unresolved sources remain in the ordinary average. No claim of recovered depth or global camera geometry.')
    np.savez_compressed(args.out/'fields.npz',fields=fields);(args.out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
