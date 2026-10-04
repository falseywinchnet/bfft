"""Revisable regional identities: annular retrieval, phase proposals, direct checks.

No camera poses, old tracks, panorama assignments or fused pixels are inputs.
An appearance match is a candidate; cyclic agreement is not certified identity.
"""
import numpy as np
from scipy import ndimage, signal
from .radial import unwind, _peak_offset
from .region_transport import fit_affine_regions, sample

POLICY = dict(version=1, sites_per_capture=96, radii=[16.,24.], annular_rows=16,
    angular_samples=48, retrieval_neighbors=6, phase_peaks=3, radial_shift=4,
    ring_radius=20, ring_search=6, ring_peaks=2, minimum_correlation=.78,
    minimum_validation=.73, cycle_pixels=1.5, cycle_jacobian=.45,
    identity='revisable alternatives plus unresolved probability; no greedy union')


def texture_coordinates(lab, texture):
    cartoon=lab-texture; ab=lab[...,1:]; cab=cartoon[...,1:]
    c=np.linalg.norm(ab,axis=-1); cc=np.linalg.norm(cab,axis=-1)
    hue=np.minimum(c,cc)[...,None]*(ab/np.maximum(c[...,None],1e-7)-cab/np.maximum(cc[...,None],1e-7))
    return np.concatenate((texture[...,:1],(c-cc)[...,None],hue),axis=-1)


def signature(trace):
    """Angular magnitude proposes rotation-invariant identity; phase is kept elsewhere."""
    a=np.asarray(trace,float); a=a-a.mean(axis=(0,1),keepdims=True)
    power=np.abs(np.fft.rfft(a,axis=1))[:,:7]
    # Keep radial structure and signed hue-coordinate channel identities.
    descriptor=np.sqrt(np.maximum(np.stack([x.mean(0) for x in np.array_split(power,4)]),0)).ravel()
    return descriptor/max(np.linalg.norm(descriptor),1e-12)


def trace_peaks(a,b,step,maximum=4,count=3):
    """Multiple finite-radius/periodic-angle correlation peaks, never radial wrap."""
    a=a-a.mean(axis=1,keepdims=True); b=b-b.mean(axis=1,keepdims=True)
    nr,na,_=a.shape; shifts=np.arange(-maximum,maximum+1); scores=[]
    for dr in shifts:
        aa=a[max(0,-dr):min(nr,nr-dr)]; bb=b[max(0,dr):min(nr,nr+dr)]
        cross=np.fft.ifft(np.conj(np.fft.fft(aa,axis=1))*np.fft.fft(bb,axis=1),axis=1).real
        scores.append(cross.sum(axis=(0,2))/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-12))
    scores=np.asarray(scores); work=scores.copy(); out=[]
    for _ in range(count):
        ir,it=np.unravel_index(np.argmax(work),work.shape)
        if not np.isfinite(work[ir,it]):break
        dr=float(shifts[ir]); dt=float(it)
        if 0<ir<len(shifts)-1:dr+=_peak_offset(scores[ir-1,it],scores[ir,it],scores[ir+1,it])
        dt+=_peak_offset(scores[ir,(it-1)%na],scores[ir,it],scores[ir,(it+1)%na])
        angle=dt*2*np.pi/na; scale=np.exp(dr*step)
        out.append(dict(scale=float(scale),angle=float(angle),correlation=float(scores[ir,it]),boundary=ir in (0,len(shifts)-1)))
        for r in range(max(0,ir-1),min(len(shifts),ir+2)):
            work[r,[(it+j)%na for j in range(-3,4)]]=-np.inf
    return out


def ringing(image):
    taps=signal.firwin(25,[.07,.48],pass_zero=False,window='boxcar')
    return ndimage.convolve1d(image,taps,axis=1,mode='reflect')+ndimage.convolve1d(image,taps,axis=0,mode='reflect')


def ring_delays(reference,moving,maximum=6,count=2):
    """Moving-content displacement, with competing finite-overlap peaks.

Ringing reshapes existing evidence; it cannot manufacture uniqueness or exact delay.
Subpixel offsets use a local quadratic peak approximation.
"""
    a=ringing(reference);b=ringing(moving); n=a.shape[0];m=a.shape[1]
    scores=np.empty((2*maximum+1,2*maximum+1))
    for y in range(-maximum,maximum+1):
        for x in range(-maximum,maximum+1):
            aa=a[max(0,-y):min(n,n-y),max(0,-x):min(m,m-x)]
            bb=b[max(0,y):min(n,n+y),max(0,x):min(m,m+x)]
            aa=aa-aa.mean();bb=bb-bb.mean()
            scores[y+maximum,x+maximum]=np.sum(aa*bb)/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-12)
    work=scores.copy();out=[]
    for _ in range(count):
        iy,ix=np.unravel_index(np.argmax(work),work.shape);dx=ix-maximum;dy=iy-maximum
        if not np.isfinite(work[iy,ix]):break
        if 0<ix<2*maximum:dx+=_peak_offset(scores[iy,ix-1],scores[iy,ix],scores[iy,ix+1])
        if 0<iy<2*maximum:dy+=_peak_offset(scores[iy-1,ix],scores[iy,ix],scores[iy+1,ix])
        out.append(dict(shift=[float(dx),float(dy)],correlation=float(scores[iy,ix])))
        work[max(0,iy-2):iy+3,max(0,ix-2):ix+3]=-np.inf
    return out


def verify(reference,moving,texture_a,texture_b,pa,pb,matrices,use_ringing=True):
    """All phase alternatives compete; direct Lab checks validate each independently."""
    pa=np.asarray(pa);pb=np.asarray(pb);starts=[];r=POLICY['ring_radius']
    yy,xx=np.mgrid[-r:r+1,-r:r+1];offset=np.stack((xx,yy),axis=-1)
    ra=sample(texture_a[...,None],pa+offset)[...,0]
    for A in matrices:
        # The zero-delay alternative is always present, allowing a ringing ablation.
        starts.append((A,pb,'phase'))
        if use_ringing:
            xy=pb+offset@A.T
            if np.any(xy<1) or np.any(xy>np.array(texture_b.shape[::-1])-2):continue
            rb=sample(texture_b[...,None],xy)[...,0]
            for peak in ring_delays(ra,rb):starts.append((A,pb+A@peak['shift'],'ringing'))
    A=np.array([x[0] for x in starts]);q=np.array([x[1] for x in starts]);p=np.tile(pa,(len(starts),1))
    f=fit_affine_regions(reference,moving,p,q,A)
    # Independent reverse fits begin at the candidate region center, not fitted q.
    reverse=fit_affine_regions(moving,reference,np.tile(pb,(len(starts),1)),p,np.linalg.inv(A))
    back=reverse['center']+np.einsum('nij,nj->ni',reverse['affine'],f['center']-pb)
    closure=np.linalg.norm(back-p,axis=1);jac=np.linalg.norm(reverse['affine']@f['affine']-np.eye(2),axis=(1,2))
    accepted=f['admissible']&reverse['admissible']&(f['train_correlation']>=.78)&(f['validation_correlation']>=.73)&(reverse['validation_correlation']>=.73)&(closure<=1.5)&(jac<.6)&(f['information_ratio']>=.012)
    out=[]
    for k in range(len(starts)):
        out.append(dict(p=pa.tolist(),site_q=pb.tolist(),q=f['center'][k].tolist(),A=f['affine'][k].tolist(),
            train=float(f['train_correlation'][k]),validation=float(f['validation_correlation'][k]),
            reverse_validation=float(reverse['validation_correlation'][k]),roundtrip=float(closure[k]),
            jacobian_roundtrip=float(jac[k]),accepted=bool(accepted[k]),initialization=starts[k][2]))
    return out


def rerank(edges):
    """Recompute alternative scores from image checks and distinct third captures.

All hypotheses survive. No transitive identity union or station assignment occurs.
Triangle votes are capped once per third capture; reciprocal edges are not votes.
"""
    from .transport_tracks import close_triangle
    adjacency={}; valid=[]
    for k,e in enumerate(edges):
        if not e['accepted']:continue
        row={**e,'a':tuple(e['a']),'b':tuple(e['b'])}
        for name in ('p','q','site_q','A'):row[name]=np.asarray(row[name],float)
        valid.append((k,row));adjacency.setdefault(row['a'],[]).append((k,row))
        inv=np.linalg.inv(row['A'])
        rev={**row,'a':row['b'],'b':row['a'],'p':row['site_q'],'site_q':row['p'],
             'q':row['p']+inv@(row['site_q']-row['q']),'A':inv}
        adjacency.setdefault(rev['a'],[]).append((k,rev))
    support={k:set() for k,_ in valid}
    for k,e in valid:
        for ki,ab in adjacency.get(e['a'],[]):
            third=ab['b'][0]
            if third in (e['a'][0],e['b'][0]):continue
            for kj,bc in adjacency.get(ab['b'],[]):
                if bc['b']!=e['b'] or len({k,ki,kj})<3:continue
                error,jac=close_triangle(ab,bc,e)
                if error<=1.5 and jac<=.45:support[k].add(third)
    groups={};out=[]
    for k,e in enumerate(edges):
        votes=sorted(support.get(k,set()));logit=8*(min(e['validation'],e['reverse_validation'])-.78)+.6*min(len(votes),4) if e['accepted'] else -30.
        row={**e,'third_capture_support':votes,'logit':float(logit)};out.append(row)
        groups.setdefault((tuple(e['a']),e['b'][0]),[]).append(k)
    for ids in groups.values():
        identities={}
        for k in ids:identities.setdefault(tuple(out[k]['b']),[]).append(k)
        keys=sorted(identities)
        # Multiple starts of one physical identity are not independent votes.
        values=np.array([0.]+[max(out[k]['logit'] for k in identities[key]) for key in keys])
        weights=np.exp(values-values.max());weights/=weights.sum()
        for index,key in enumerate(keys):
            ks=identities[key];v=np.array([out[k]['logit'] for k in ks]);v=np.exp(v-v.max());v/=v.sum()
            for j,k in enumerate(ks):
                out[k]['identity_weight']=float(weights[index+1])
                out[k]['relative_weight']=float(weights[index+1]*v[j])
                out[k]['unresolved_weight']=float(weights[0])
    return out
