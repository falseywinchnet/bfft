"""Finite annular traces: periodic angle, nonperiodic logarithmic radius.

For a centered similarity I_m(r,theta)=I_r(r/s,theta-alpha), an unwound
trace translates by (log(s),alpha). A wrong center or projective distortion
violates this model. Angular phase is retained; nothing is magnitude-only.
"""
import numpy as np
from scipy import ndimage


def unwind(image,center,r_min=6.,r_max=60.,radial_count=48,angular_count=128):
    if not 0<r_min<r_max or radial_count<3 or angular_count<4:
        raise ValueError('trace requires positive ordered radii and at least 3x4 samples')
    image=np.asarray(image,dtype=np.float64)
    u=np.linspace(np.log(r_min),np.log(r_max),radial_count)
    theta=np.arange(angular_count)*(2*np.pi/angular_count)
    x=center[0]+np.exp(u[:,None])*np.cos(theta)[None,:]
    y=center[1]+np.exp(u[:,None])*np.sin(theta)[None,:]
    valid=(x>=0)&(x<=image.shape[1]-1)&(y>=0)&(y<=image.shape[0]-1)
    if image.ndim==2:
        trace=ndimage.map_coordinates(image,[y,x],order=1,mode='constant',cval=0)
    else:
        trace=np.stack([ndimage.map_coordinates(image[...,k],[y,x],order=1,mode='constant',cval=0) for k in range(image.shape[2])],axis=-1)
    return trace,valid,{'log_radius_step':float(u[1]-u[0]),'angle_step':float(2*np.pi/angular_count)}


def _peak_offset(left,center,right):
    den=left-2*center+right
    return float(np.clip(.5*(left-right)/den,-.5,.5)) if den < -1e-12 else 0.


def trace_similarity(reference,moving,log_radius_step,max_radial_shift=10):
    """Return moving-content scale/angle; radius is never circularly wrapped.

Correlation sums over actual overlapping radial rows. Exclude the trivial
radial DC per row, preserve angular phase, and normalize each finite overlap.
"""
    a=np.asarray(reference,dtype=float);b=np.asarray(moving,dtype=float)
    if a.shape!=b.shape or a.ndim not in (2,3):raise ValueError('trace shapes must match')
    if a.ndim==2:a=a[...,None];b=b[...,None]
    a=a-a.mean(axis=1,keepdims=True);b=b-b.mean(axis=1,keepdims=True)
    nr,na,_=a.shape;maximum=min(int(max_radial_shift),nr//3)
    shifts=np.arange(-maximum,maximum+1);scores=[]
    for dr in shifts:
        # moving at radial index i+dr corresponds to reference index i.
        aa=a[max(0,-dr):min(nr,nr-dr)]
        bb=b[max(0,dr):min(nr,nr+dr)]
        cross=np.fft.ifft(np.conj(np.fft.fft(aa,axis=1))*np.fft.fft(bb,axis=1),axis=1).real
        norm=np.sqrt(np.sum(aa*aa)*np.sum(bb*bb))
        scores.append(np.sum(cross,axis=(0,2))/max(norm,1e-15))
    scores=np.array(scores);ir,it=np.unravel_index(np.argmax(scores),scores.shape)
    radial_offset=_peak_offset(scores[ir-1,it],scores[ir,it],scores[ir+1,it]) if 0<ir<len(shifts)-1 else 0.
    angle_offset=_peak_offset(scores[ir,(it-1)%na],scores[ir,it],scores[ir,(it+1)%na])
    dr=float(shifts[ir]+radial_offset);dt=float(it+angle_offset)
    if dt>na/2:dt-=na
    return {'scale':float(np.exp(dr*log_radius_step)),'angle_degrees':dt*360/na,
            'radial_shift':dr,'angular_shift':dt,'correlation':float(scores[ir,it]),
            'radial_search_boundary':bool(ir in (0,len(shifts)-1))}
