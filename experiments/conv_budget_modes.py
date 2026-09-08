"""Budgeted direct characteristic catalog with unchanged CONV fallback.

Finite proposals, scalar ordering and original support-cone certificates.
No optimization or linear-system solve. Experimental, scalar rectangular data.
"""
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter
from experiments.conv_joint_ridge_modes import direction_arc
from experiments.convstar import convstar_resize,refine_lines

def conv_resize(source,scale,method=None):
    return convstar_resize(source,scale)

EPS=np.finfo(float).eps


def profile(phase,values,flat=False):
    idx=np.argsort(phase,kind='stable');t=phase[idx];f=values[idx]
    peps=128*EPS*max(1,float(np.max(abs(t))))
    start=np.r_[0,np.flatnonzero(np.diff(t)>peps)+1]
    if len(start)<2:return None
    merge_error=0.
    if len(start)<len(t):
        count=np.diff(np.r_[start,len(t)])
        lo=np.minimum.reduceat(f,start);hi=np.maximum.reduceat(f,start)
        merge_error=float(np.max(hi-lo))
        if merge_error>2e-11:return None
        t=np.add.reduceat(t,start)/count;f=np.add.reduceat(f,start)/count
    if np.min(np.diff(f))<-2e-11:return None
    old=f;f=np.maximum.accumulate(f);adjustment=float(np.max(abs(f-old)))
    binary=np.all((f==f[0])|(f==f[-1]))
    if binary and len(f)>4:
        # Exact removal of redundant flat phase samples. Their values remain
        # represented by the same constant pieces; retain both run endpoints.
        keep=np.r_[True,(f[1:-1]!=f[:-2])|(f[1:-1]!=f[2:]),True]
        t=t[keep];f=f[keep]
    h=np.diff(t);d=np.diff(f)/h;m=np.zeros_like(f)
    if not flat:m[0]=d[0];m[-1]=d[-1]
    a=d[:-1];b=d[1:];positive=(a>0)&(b>0)
    harmonic=np.zeros_like(a);harmonic[positive]=2/(1/a[positive]+1/b[positive])
    m[1:-1]=np.minimum(harmonic,1.25*np.minimum(a,b))
    curvature=np.zeros_like(f)
    if len(t)>=5 and not np.all((f==f[0])|(f==f[-1])):
        # Five-point nonuniform Newton polynomial derivatives, evaluated
        # directly by differentiated Horner recurrences; no Vandermonde solve.
        indices=np.clip(np.arange(len(t))-2,0,len(t)-5)[:,None]+np.arange(5)
        nodes=t[indices]-t[:,None];dd=f[indices].copy()
        for degree in range(1,5):
            dd[:,degree:]=(dd[:,degree:]-dd[:,degree-1:-1])/(nodes[:,degree:]-nodes[:,:-degree])
        val=dd[:,4];first=np.zeros_like(val);second=np.zeros_like(val)
        for k in range(3,-1,-1):
            second=2*first-nodes[:,k]*second
            first=val-nodes[:,k]*first
            val=dd[:,k]-nodes[:,k]*val
        if flat:first[[0,-1]]=0;second[[0,-1]]=0
        dm=first-m;dc=second
        base=np.stack((m[:-1],m[:-1],5*d-2*m[:-1]-2*m[1:],m[1:],m[1:]),axis=1)
        zero=np.zeros_like(h)
        left=np.stack((dm[:-1],dm[:-1]+h*dc[:-1]/4,-2*dm[:-1]-h*dc[:-1]/4,zero,zero),axis=1)
        right=np.stack((zero,zero,-2*dm[1:]+h*dc[1:]/4,dm[1:]-h*dc[1:]/4,dm[1:]),axis=1)
        harm=np.maximum(-left,0)+np.maximum(-right,0)
        cap=np.minimum(1,np.divide(np.maximum(base,0),harm,out=np.ones_like(base),where=harm>0))
        alpha=np.ones_like(f)
        alpha[:-1]=np.minimum(alpha[:-1],np.min(np.where(left<0,cap,1),axis=1))
        alpha[1:]=np.minimum(alpha[1:],np.min(np.where(right<0,cap,1),axis=1))
        if np.min(base+left+right)>=0:alpha[:]=1
        m+=alpha*dm;curvature=alpha*dc
    delta=np.diff(f);l=h*m[:-1];r=h*m[1:];ql=h*h*curvature[:-1];qr=h*h*curvature[1:]
    power=np.stack((f[:-1],l,ql/2,10*delta-6*l-4*r-1.5*ql+.5*qr,
                    -15*delta+8*l+7*r+1.5*ql-qr,6*delta-3*l-3*r-.5*ql+.5*qr),axis=1)
    derivative=np.stack((m[:-1],m[:-1]+h*curvature[:-1]/4,
                         5*d-2*m[:-1]-2*m[1:]+h*(curvature[1:]-curvature[:-1])/4,
                         m[1:]-h*curvature[1:]/4,m[1:]),axis=1)
    result={'knots':t,'samples':f,'power':power,'upper':np.max(derivative,axis=1),
            'slope':m,'curvature':curvature,'minimum_derivative':float(derivative.min()),
            'merge_error':merge_error, 'order_adjustment':adjustment}
    if binary and np.all(m==0) and np.all(curvature==0):
        j=int(np.flatnonzero(np.diff(f)>0)[0]);result['step_interval']=(t[j],t[j+1],f[0],f[-1])
    return result


def sample_profile(model,t):
    if 'step_interval' in model:
        lo,hi,f0,f1=model['step_interval'];u=np.clip((t-lo)/(hi-lo),0,1)
        return f0+(f1-f0)*u*u*u*(10+u*(-15+6*u))
    knots=model['knots'];idx=np.clip(np.searchsorted(knots,t,side='right')-1,0,len(knots)-2)
    u=np.clip((t-knots[idx])/(knots[idx+1]-knots[idx]),0,1)
    if np.size(t)>4096:
        # Keep one output-sized accumulator instead of a Q-by-6 gather.
        coefficients=model['power'];result=coefficients[:,5][idx]
        for k in range(4,-1,-1):
            result*=u;result+=coefficients[:,k][idx]
        return result
    p=model['power'][idx]
    return p[...,0]+u*(p[...,1]+u*(p[...,2]+u*(p[...,3]+u*(p[...,4]+u*p[...,5]))))


def evaluate(model,x,y):
    if 'constant' in model:return np.full(np.broadcast_shapes(np.shape(x),np.shape(y)),model['constant'])
    if model['chart']=='ridge':t=model['direction'][0]*x+model['direction'][1]*y
    else:t=model['polarity']*((x-model['center'][0])**2+(y-model['center'][1])**2)
    return model['offset']+model['scale']*sample_profile(model,t)


def cone_bank(z):
    """Original pointed support cones from two angular frames, O(P) work.

    Every arc of width <= pi excludes at least one of two antipodal cuts.
    Its min/max in that frame are therefore its actual cone boundaries.
    Wider arcs generate the whole plane. Near-pi ambiguity conservatively
    uses the line unless a strictly interior generating vector is witnessed.
    """
    dx=np.diff(z,axis=1);dy=np.diff(z,axis=0)
    gx=np.stack((dx[:-1],dx[:-1],dx[1:],dx[1:]),axis=-1)
    gy=np.stack((dy[:,:-1],dy[:,1:],dy[:,:-1],dy[:,1:]),axis=-1)
    magnitude=np.hypot(gx,gy);active=magnitude>2048*EPS
    angles=np.mod(np.arctan2(gy,gx),2*np.pi)
    starts=[];widths=[]
    for shift in (0.,np.pi):
        angle=np.mod(angles-shift,2*np.pi)
        low=minimum_filter(np.min(np.where(active,angle,np.inf),axis=-1),size=5,mode='constant',cval=np.inf)
        high=maximum_filter(np.max(np.where(active,angle,-np.inf),axis=-1),size=5,mode='constant',cval=-np.inf)
        starts.append(low+shift);widths.append(high-low)
    choose=widths[1]<widths[0];width=np.where(choose,widths[1],widths[0]);start=np.where(choose,starts[1],starts[0])
    empty=~np.isfinite(start)
    if np.any((width>np.pi+4096*EPS)&~empty):
        # The original oracle treats pi +/- angular tolerance specially.
        # Two extra cuts resolve near-antipodal rays straddling both first cuts.
        for shift in (.5*np.pi,1.5*np.pi):
            angle=np.mod(angles-shift,2*np.pi)
            low=minimum_filter(np.min(np.where(active,angle,np.inf),axis=-1),size=5,mode='constant',cval=np.inf)
            high=maximum_filter(np.max(np.where(active,angle,-np.inf),axis=-1),size=5,mode='constant',cval=-np.inf)
            newwidth=high-low;better=newwidth<width
            start=np.where(better,low+shift,start);width=np.minimum(width,newwidth)
    start=np.where(empty,0,start);width=np.where(empty,2*np.pi,width)
    first=np.stack((-np.sin(start),np.cos(start)),axis=-1)
    second=np.stack((np.sin(start+width),-np.cos(start+width)),axis=-1)
    ray=np.stack((np.cos(start),np.sin(start)),axis=-1)
    angular=4096*EPS;whole=(width>np.pi+angular)|empty
    nearpi=abs(width-np.pi)<=angular
    if np.any(nearpi):
        # Unnormalized positive sum witnesses an interior generator. Guard
        # much more strongly than roundoff; ambiguous halfplanes become lines.
        from scipy.ndimage import uniform_filter
        ux=np.divide(gx,magnitude,out=np.zeros_like(gx),where=active).sum(-1)
        uy=np.divide(gy,magnitude,out=np.zeros_like(gy),where=active).sum(-1)
        sx=uniform_filter(ux,size=5,mode='constant')*25
        sy=uniform_filter(uy,size=5,mode='constant')*25
        interior=(first[...,0]*sx+first[...,1]*sy)>1e-9
        second=np.where((nearpi&interior)[...,None],0,second)
        second=np.where((nearpi&~interior)[...,None],-first,second)
    ray=np.where((width<=angular)[...,None],ray,0)
    result=np.stack((first,second,ray),axis=-2)
    return np.where(whole[...,None,None],0,result)


def certificate(model,z,xx,yy):
    """Same cones and 6x6 range; finite actual-current amplitude bound."""
    if model['chart']=='ridge':
        n=model['direction'];phase=n[0]*xx+n[1]*yy
    else:
        center=model['center'];s=model['polarity'];phase=s*((xx-center[0])**2+(yy-center[1])**2)
    corners=np.stack((phase[:-1,:-1],phase[:-1,1:],phase[1:,:-1],phase[1:,1:]),axis=-1)
    low=corners.min(-1);high=corners.max(-1)
    if model['chart']=='radial':
        cx=np.clip(center[0],xx[:-1,:-1],xx[:-1,:-1]+1)
        cy=np.clip(center[1],yy[:-1,:-1],yy[:-1,:-1]+1)
        nearest=(cx-center[0])**2+(cy-center[1])**2
        if s>0:low=nearest
        else:high=-nearest
    # Prefix counts certify exactly flat phase ranges, preserving zero-amplitude admission.
    knots=model['knots'];begin=np.maximum(0,np.searchsorted(knots,low,side='right')-1)
    end=np.minimum(len(knots)-1,np.searchsorted(knots,high,side='left'))
    prefix=np.r_[0,np.cumsum(np.diff(model['samples'])>0)]
    nonzero=prefix[end]>prefix[begin]
    normals=cone_bank(z)
    if model['chart']=='ridge':
        margin=np.min(normals@n,axis=-1)
    else:
        margin=np.full(low.shape,np.inf)
        for ox,oy in ((0,0),(1,0),(0,1),(1,1)):
            gx=2*s*(xx[:-1,:-1]+ox-center[0]);gy=2*s*(yy[:-1,:-1]+oy-center[1])
            margin=np.minimum(margin,np.min(normals[...,0]*gx[...,None]+normals[...,1]*gy[...,None],axis=-1))
    # Global upper bound is conservative and avoids a phase-piece scan per cell.
    current_margin=float(np.min(np.where(nonzero,np.minimum(margin,0)*np.max(model['upper']),0)))
    if current_margin < -2e-11:
        # Sparse-table range maximum: exactly the intersecting phase pieces.
        upper=model['upper'];table=[upper];stride=1
        while stride<len(upper):
            prev=table[-1];table.append(np.maximum(prev,np.r_[prev[stride:],np.full(stride,-np.inf)][:len(prev)]));stride*=2
        table=np.array(table);length=np.maximum(end-begin,1)
        level=np.floor(np.log2(length)).astype(int);span=1<<level
        maximum=np.maximum(table[level,np.minimum(begin,len(upper)-1)],table[level,np.maximum(end-span,0)])
        current_margin=float(np.min(np.where(nonzero,np.minimum(margin,0)*maximum,0)))
        if current_margin < -2e-11:return False,{'current_margin':current_margin}
    # Range extrema are exact by scalar monotonicity; use original support bounds.
    lower=minimum_filter(z,size=6,origin=-1,mode='nearest')[:-1,:-1]
    upper=maximum_filter(z,size=6,origin=-1,mode='nearest')[:-1,:-1]
    range_margin=float(min(np.min(sample_profile(model,low)-lower),np.min(upper-sample_profile(model,high))))
    return range_margin>=-2e-11,{'current_margin':current_margin,'range_margin':range_margin}


def candidates(z,xx,yy):
    """A fixed small catalog from boundary flux, adjacent order and repeated levels."""
    h,w=z.shape;values=z.ravel();points=np.stack((xx.ravel(),yy.ravel()),axis=-1)
    dx=np.diff(z,axis=1);dy=np.diff(z,axis=0);tol=64*EPS
    monotone_x=not (np.min(dx)<-tol and np.max(dx)>tol)
    monotone_y=not (np.min(dy)<-tol and np.max(dy)>tol)
    if monotone_x and monotone_y:
        order=np.argsort(values,kind='stable');dv=np.diff(values[order]);dp=np.diff(points[order],axis=0)
        directions=[]
        # For an exact two-level edge, only the two row frontiers matter.
        # Choose the shorter transverse axis, so its pair count is O(HW).
        if np.all((values==0)|(values==1)):
            transposed=h>w;field=z.T if transposed else z
            increasing=float(np.sum(np.diff(field,axis=1)))>=0
            if not increasing:field=field[:,::-1]
            nh,nw=field.shape;low=field==field.min();high=~low
            rows=np.arange(nh);left=np.max(np.where(low,np.arange(nw),-1),axis=1)
            right=np.min(np.where(high,np.arange(nw),nw),axis=1)
            lp=np.stack((left[left>=0],rows[left>=0]),axis=-1)
            rp=np.stack((right[right<nw],rows[right<nw]),axis=-1)
            constraints=(rp[:,None,:]-lp[None,:,:]).reshape(-1,2)
            arc=direction_arc(np.concatenate((constraints,[[1,0]])))
            if arc is not None:
                n=np.array([np.cos(arc[0]),np.sin(arc[0])])
                if not increasing:n[0]*=-1
                if transposed:n=n[::-1]
                directions.append(n)
        equal=dp[dv<=tol]
        if len(equal):
            divisor=np.gcd(abs(equal[:,0]),abs(equal[:,1]));equal=equal//divisor[:,None]
            flip=(equal[:,0]<0)|((equal[:,0]==0)&(equal[:,1]<0));equal[flip]*=-1
            code=equal[:,0]*(2*h-1)+equal[:,1]+h-1
            selected=int(np.argmax(np.bincount(code)))
            v=np.array([selected//(2*h-1),selected%(2*h-1)-(h-1)])
            directions.append(np.array([v[1],-v[0]],float))
        arc=direction_arc(dp[dv>tol])
        wy=np.ones(h);wy[[0,-1]]=.5;wx=np.ones(w);wx[[0,-1]]=.5
        flux=np.array([wy@(z[:,-1]-z[:,0]),wx@(z[-1]-z[0])])
        def quadrature(v):
            blocks=(len(v)-1)//4;total=0.
            if blocks:
                k=np.arange(blocks)[:,None]*4+np.arange(5)
                total=float(np.sum(v[k]@np.array([7,32,12,32,7]))*2/45)
            tail=v[4*blocks:]
            if len(tail)>1:total+=float(np.sum((tail[:-1]+tail[1:])/2))
            return total
        precise_flux=np.array([quadrature(z[:,-1]-z[:,0]),quadrature(z[-1]-z[0])])
        directions.append(precise_flux)
        if arc is not None:directions.append(np.array([np.cos(arc[0]),np.sin(arc[0])]))
        for n in directions:
            length=np.hypot(*n)
            if length==0:continue
            n=n/length
            if n@flux<0:n=-n
            yield {'chart':'ridge','direction':n},points@n,False
    # Extremum centers; coordinatewise radial order cheaply rejects textures.
    centers=[]
    for mask in (values<=tol,values>=1-tol):
        center=np.mean(points[mask],axis=0)
        if any(np.max(abs(center-c))<1e-12 for c in centers):continue
        centers.append(center)
        if center[0] in (0,w-1) and center[1] in (0,h-1):
            # A corner-centered radial phase has an exact diagonal reflection
            # on the inscribed square. Reject incompatible source witnesses
            # before allocating/sorting that phase.
            view=z[::(-1 if center[1]==h-1 else 1),::(-1 if center[0]==w-1 else 1)]
            size=min(h,w);square=view[:size,:size]
            if np.max(abs(square-square.T))>2e-11:continue
        for s in (1.,-1.):
            if np.min(dx*(xx[:,:-1]+.5-center[0])*s)<-tol:continue
            if np.min(dy*(yy[:-1,:]+.5-center[1])*s)<-tol:continue
            yield {'chart':'radial','center':center,'polarity':s},s*np.sum((points-center)**2,axis=1),True


def compile_mode(source,*,_validated=False):
    source=np.asarray(source,dtype=float)
    if not _validated and (source.ndim!=2 or min(source.shape)<5 or not np.isfinite(source).all()):
        raise ValueError('finite scalar rectangle with sides >= 5 required')
    scale=float(np.ptp(source));offset=float(np.min(source))
    if scale==0:return {'constant':offset},{'accepted':True,'chart':'constant'}
    z=(source-offset)/scale
    if max(np.max(abs(np.diff(z,n=3,axis=0))),np.max(abs(np.diff(z,n=3,axis=1))))<=128*EPS:
        return None,dict(accepted=False,attempts=0,reason='biquadratic source retained by CONV')
    yy,xx=np.mgrid[:source.shape[0],:source.shape[1]];attempts=0
    for geometry,phase,flat in candidates(z,xx,yy):
        attempts+=1;m=profile(phase,z.ravel(),flat)
        if m is None:continue
        m.update(geometry)
        passed,diag=certificate(m,z,xx,yy)
        if not passed:return None,dict(accepted=False,attempts=attempts,**diag)
        m.update(offset=offset,scale=scale)
        return m,dict(accepted=True,chart=m['chart'],attempts=attempts,**diag)
    return None,dict(accepted=False,attempts=attempts)


def resize(source,scale=3,diagnostics=False):
    source=np.asarray(source,dtype=float)
    if source.ndim not in (2,3) or min(source.shape[:2])<5 or not np.isfinite(source).all():
        raise ValueError('finite HxW or HxWxC array with spatial sides >= 5 required')
    if isinstance(scale,bool) or int(scale)!=scale or scale<1:raise ValueError('positive integer scale required')
    scale=int(scale)
    if source.ndim==3:
        result=convstar_resize(source,scale);diag=dict(accepted=False,reason='multichannel CONV path')
        return (result,diag) if diagnostics else result
    if scale==1:
        result=source.copy();diag=dict(accepted=False,reason='identity scale')
        return (result,diag) if diagnostics else result
    # Exactly one varying axis needs only one existing CONV factor.
    if np.array_equal(source,np.broadcast_to(source[:1],source.shape)):
        line=refine_lines(source[0,:,None],scale)[:,0]
        result=np.broadcast_to(line,((source.shape[0]-1)*scale+1,len(line))).copy()
        diag=dict(accepted=False,reason='exact axial CONV shortcut')
        return (result,diag) if diagnostics else result
    if np.array_equal(source,np.broadcast_to(source[:,:1],source.shape)):
        line=refine_lines(source[:,0,None],scale)[:,0]
        result=np.broadcast_to(line[:,None],(len(line),(source.shape[1]-1)*scale+1)).copy()
        diag=dict(accepted=False,reason='exact axial CONV shortcut')
        return (result,diag) if diagnostics else result
    model,diag=compile_mode(source,_validated=True)
    if model is None:z=conv_resize(source,scale,'CONV')
    else:
        h,w=np.shape(source);x=np.linspace(0,w-1,(w-1)*scale+1);y=np.linspace(0,h-1,(h-1)*scale+1)
        z=evaluate(model,x[None,:],y[:,None])
    return (z,diag) if diagnostics else z


# Explicit name for callers replacing the existing scalar 2-D entry point.
conv_budget_resize=resize
