"""Relative camera hypotheses from regional centers AND image-map derivatives.

Central perspective/cylindrical camera models remain conditional on calibration.
A planar map can propose several poses; it never fixes the scene to one plane.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .multiview_geometry import bearings,essential_poses,rotation_fit,unit


def skew(t):
    x,y,z=t;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def ray_jets(points,camera):
    def ray(p):return bearings(p,camera['shape'],camera['focal'],camera['projection'],camera.get('focal_x'),camera.get('focal_y'),camera.get('principal_point'))
    p=np.asarray(points,float);eps=.001
    j=np.stack([(ray(p+np.eye(2)[k]*eps)-ray(p-np.eye(2)[k]*eps))/(2*eps) for k in range(2)],axis=-1)
    return ray(p),j


def measurements(rows,cameras):
    p=np.array([r['p'] for r in rows]);q=np.array([r['q'] for r in rows]);A=np.array([r['A'] for r in rows])
    a,ja=ray_jets(p,cameras[0]);b,jb=ray_jets(q,cameras[1]);db=jb@A
    center=np.einsum('ni,nj->nij',b,a)
    derivatives=np.einsum('nik,nj->nkij',db,a)+np.einsum('ni,njk->nkij',b,ja)
    return dict(a=a,b=b,ja=ja,jb=jb,db=db,linear=np.concatenate((center[:,None],derivatives),axis=1).reshape(len(rows),3,9),
        focal=float(np.sqrt(cameras[0]['focal']*cameras[1]['focal'])))


def essential_linear(m,ids):
    matrix=m['linear'][ids]*np.array([1.,12.,12.])[None,:,None]
    _,s,v=np.linalg.svd(matrix.reshape(-1,9),full_matrices=False)
    if len(s)<9 or s[-2]<s[0]*1e-7:raise ValueError('Essential linear system has unresolved degrees of freedom')
    u,s,v=np.linalg.svd(v[-1].reshape(3,3));return u@np.diag([(s[0]+s[1])/2]*2+[0])@v


def homography_linear(m,ids):
    rows=[]
    for i in ids:
        a,b,ja,db=m['a'][i],m['b'][i],m['ja'][i],m['db'][i]
        rows.append(np.kron(skew(b),a[None,:]))
        for k in range(2):rows.append(12*(np.kron(skew(db[:,k]),a[None,:])+np.kron(skew(b),ja[:,k][None,:])))
    _,s,v=np.linalg.svd(np.concatenate(rows),full_matrices=False)
    if len(s)<9 or s[-2]<s[0]*1e-7:raise ValueError('Planar map is unresolved')
    return v[-1].reshape(3,3)


def homography_poses(H):
    """SVD factorization H = R + t n^T, retaining both tilt and sign branches.

In singular-vector coordinates diag(s1,1,s3)-Ry(theta) must be rank one;
its 2x2 determinant gives cos(theta)=(s1*s3+1)/(s1+s3).
"""
    H=np.asarray(H,float)
    if np.linalg.det(H)<0:H=-H
    u,s,v=np.linalg.svd(H)
    if s[1]<1e-12:return []
    s=s/s[1];c=np.clip((s[0]*s[2]+1)/(s[0]+s[2]),-1,1);ss=np.sqrt(max(0.,1-c*c));out=[]
    for sign in (1.,-1.):
        rr=np.array([[c,0,sign*ss],[0,1,0],[-sign*ss,0,c]])
        tmat=np.diag(s)-rr;ut,st,vt=np.linalg.svd(tmat)
        if st[0]<1e-8:continue
        r=u@rr@v;t=u@ut[:,0];normal=st[0]*vt[0]@v
        for direction in (1.,-1.):out.append(dict(rotation=r,translation=t*direction,normal=normal*direction))
    return out


def epi_residual(m,E):
    na=m['a']@E.T;nb=m['b']@E
    ga=np.einsum('nik,ni->nk',m['ja'],nb);gb=np.einsum('nik,ni->nk',m['jb'],na)
    norm=np.sqrt(np.sum(ga*ga+gb*gb,axis=1));norm=np.maximum(norm,1e-10)
    value=np.einsum('nki,i->nk',m['linear'],E.ravel())
    return value*np.array([1.,3.,3.])[None,:]/norm[:,None]


def epi_error(m,E):
    r=epi_residual(m,E);return np.abs(r[:,0])+np.abs(r[:,1:]).sum(1)


def refine_essential(m,ids,E,maximum=35):
    poses=essential_poses(E,m['a'][ids],m['b'][ids])
    if not poses:return E
    p=poses[0];x=np.r_[Rotation.from_matrix(p['rotation']).as_rotvec(),p['translation']]
    subset={k:v[ids] if isinstance(v,np.ndarray) else v for k,v in m.items()}
    def matrix(v):return skew(unit(v[3:]))@Rotation.from_rotvec(v[:3]).as_matrix()
    opt=least_squares(lambda v:epi_residual(subset,matrix(v)).ravel(),x,loss='soft_l1',f_scale=1.,max_nfev=maximum)
    return matrix(opt.x)


def rotation_error(m,R):
    center=(m['a']@R.T-m['b'])*m['focal']
    derivative=(np.einsum('ij,njk->nik',R,m['ja'])-m['db'])*m['focal']*3
    return np.linalg.norm(center,axis=1)+np.linalg.norm(derivative,axis=1).sum(1)


def fit_rotation(m,ids):
    a=np.concatenate((m['a'][ids],m['ja'][ids,:,0]*12,m['ja'][ids,:,1]*12))
    b=np.concatenate((m['b'][ids],m['db'][ids,:,0]*12,m['db'][ids,:,1]*12))
    return rotation_fit(a,b)
