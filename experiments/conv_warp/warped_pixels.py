"""Conservative warped-pixel analysis of a joint CONV Bernstein atlas."""

from __future__ import annotations

import math

import numpy as np

from .geometry import ProjectiveMap
from .joint_reference import evaluate_joint_atlas,finite_joint_control_nets
from .synthetic import endpoint_basin_bounds


Array=np.ndarray


def _clip_axis(polygon: Array,axis: int,bound: float,keep_greater: bool)->Array:
    value=np.asarray(polygon,dtype=np.float64)
    if not value.size:
        return value.reshape(0,2)
    output=[]
    previous=value[-1]
    previous_inside=(previous[axis]>=bound) if keep_greater else (previous[axis]<=bound)
    for current in value:
        current_inside=(current[axis]>=bound) if keep_greater else (current[axis]<=bound)
        if current_inside!=previous_inside:
            denominator=current[axis]-previous[axis]
            t=0.0 if denominator==0.0 else (bound-previous[axis])/denominator
            output.append(previous+t*(current-previous))
        if current_inside:
            output.append(current)
        previous=current;previous_inside=current_inside
    return np.asarray(output,dtype=np.float64).reshape(-1,2)


def _clip_rectangle(polygon: Array,x0: float,x1: float,y0: float,y1: float)->Array:
    value=_clip_axis(polygon,0,x0,True)
    value=_clip_axis(value,0,x1,False)
    value=_clip_axis(value,1,y0,True)
    return _clip_axis(value,1,y1,False)


def _signed_polygon_area(polygon: Array)->float:
    value=np.asarray(polygon,dtype=np.float64)
    if value.shape[0]<3:
        return 0.0
    return 0.5*float(np.sum(
        value[:,0]*np.roll(value[:,1],-1)-value[:,1]*np.roll(value[:,0],-1)
    ))


def _gauss_unit(order: int)->tuple[Array,Array]:
    node,weight=np.polynomial.legendre.leggauss(int(order))
    return .5*(node+1.0),.5*weight


def _evaluate_patch(control: Array,x: Array,y: Array)->Array:
    patch=np.asarray(control,dtype=np.float64)
    scalar=patch.ndim==2
    if scalar:
        patch=patch[...,None]
    qx,qy=np.broadcast_arrays(np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64))
    flat_x=qx.reshape(-1);flat_y=qy.reshape(-1)
    bx=np.stack([
        math.comb(5,k)*flat_x**k*(1.0-flat_x)**(5-k) for k in range(6)
    ],axis=1)
    by=np.stack([
        math.comb(5,k)*flat_y**k*(1.0-flat_y)**(5-k) for k in range(6)
    ],axis=1)
    value=np.einsum("ni,nj,ijc->nc",by,bx,patch,optimize=True)
    shaped=value.reshape(qx.shape+(patch.shape[-1],))
    return shaped[...,0] if scalar else shaped


def _integrate_patch_polygon(control: Array,polygon: Array,order: int=6)->Array:
    """Integrate one bidegree-(5,5) patch exactly over a convex polygon."""

    value=np.asarray(polygon,dtype=np.float64)
    patch=np.asarray(control,dtype=np.float64)
    channels=1 if patch.ndim==2 else patch.shape[-1]
    total=np.zeros(channels,dtype=np.float64)
    if value.shape[0]<3:
        return total[0] if patch.ndim==2 else total
    # Clipping preserves convexity.  A fan triangulation is therefore positive
    # and nonoverlapping regardless of the polygon's orientation.
    node,weight=_gauss_unit(order)
    anchor=value[0]
    for k in range(1,value.shape[0]-1):
        b=value[k];c=value[k+1]
        edge_b=b-anchor;edge_c=c-anchor
        determinant=abs(float(edge_b[0]*edge_c[1]-edge_b[1]*edge_c[0]))
        if determinant<=64.0*np.finfo(np.float64).eps:
            continue
        uu,vv=np.meshgrid(node,node,indexing="ij")
        point=(anchor+uu[...,None]*(b-anchor)
               +(1.0-uu)[...,None]*vv[...,None]*(c-anchor))
        sample=_evaluate_patch(patch,point[...,0],point[...,1])
        sample=sample[...,None] if patch.ndim==2 else sample
        quadrature=weight[:,None]*weight[None,:]*determinant*(1.0-uu)
        total+=np.einsum("ij,ijc->c",quadrature,sample,optimize=True)
    return total[0] if patch.ndim==2 else total


def exact_affine_warped_pixel_average(
    values: Array,
    transform: ProjectiveMap,
    target_shape: tuple[int,int],
    *,
    control: Array|None=None,
)->tuple[Array,dict[str,float|int]]:
    """Exactly average the admitted atlas over affine target-pixel basins.

    Each target rectangle maps to a source parallelogram.  Source knot lines
    split that parallelogram into convex polygons, each lying in one quintic
    patch.  Fan triangulation followed by a positive order-six Duffy--Gauss
    rule is exact for the resulting total-degree-ten polynomial: after the
    Duffy Jacobian, the separate degrees are at most eleven and ten.
    """

    if not transform.is_affine:
        raise ValueError("exact polynomial warped-pixel analysis requires an affine map")
    source=np.asarray(values,dtype=np.float64)
    scalar=source.ndim==2
    if source.ndim not in (2,3):
        raise ValueError("values must be HxW or HxWxC")
    if control is None:
        atlas,admission=finite_joint_control_nets(source)
    else:
        atlas=np.asarray(control,dtype=np.float64);admission={}
    target_height,target_width=map(int,target_shape)
    output_shape=(target_height,target_width)+(() if scalar else (source.shape[-1],))
    output=np.empty(output_shape,dtype=np.float64)
    xb=endpoint_basin_bounds(target_width);yb=endpoint_basin_bounds(target_height)
    maximum_area_error=0.0;outside_basins=0;pieces=0
    sx=float(source.shape[1]-1);sy=float(source.shape[0]-1)
    for row in range(target_height):
        for column in range(target_width):
            q=np.array(((xb[column],yb[row]),(xb[column+1],yb[row]),
                        (xb[column+1],yb[row+1]),(xb[column],yb[row+1])))
            px,py=transform.map(q[:,0],q[:,1])
            polygon=np.stack((sx*px,sy*py),axis=1)
            source_area=abs(_signed_polygon_area(polygon))
            if source_area<=64.0*np.finfo(np.float64).eps:
                raise ValueError("a target basin maps to zero source area")
            xmin,xmax=float(np.min(polygon[:,0])),float(np.max(polygon[:,0]))
            ymin,ymax=float(np.min(polygon[:,1])),float(np.max(polygon[:,1]))
            if xmin<0.0 or ymin<0.0 or xmax>sx or ymax>sy:
                outside_basins+=1
                output[row,column]=np.nan
                continue
            integral=np.zeros(1 if scalar else source.shape[-1],dtype=np.float64)
            covered=0.0
            for cy in range(max(0,int(math.floor(ymin))),min(source.shape[0]-2,int(math.floor(ymax)))+1):
                for cx in range(max(0,int(math.floor(xmin))),min(source.shape[1]-2,int(math.floor(xmax)))+1):
                    clipped=_clip_rectangle(polygon,cx,cx+1.0,cy,cy+1.0)
                    area=abs(_signed_polygon_area(clipped))
                    if area<=256.0*np.finfo(np.float64).eps:
                        continue
                    local=clipped-np.array((cx,cy),dtype=np.float64)
                    integral+=np.atleast_1d(_integrate_patch_polygon(atlas[cy,cx],local,6))
                    covered+=area;pieces+=1
            maximum_area_error=max(maximum_area_error,abs(covered-source_area))
            value=integral/source_area
            output[row,column]=value[0] if scalar else value
    diagnostic={
        **admission,"quadrature_order":6,"polygon_pieces":pieces,
        "outside_basins":outside_basins,"maximum_source_area_partition_error":maximum_area_error,
    }
    return (output if not scalar else output.reshape(target_height,target_width)),diagnostic


def _patch_value_bounds(control: Array,triangle: Array)->tuple[Array,Array,Array]:
    patch=np.asarray(control,dtype=np.float64)
    scalar=patch.ndim==2
    if scalar:
        patch=patch[...,None]
    centre=np.mean(triangle,axis=0)
    value=np.atleast_1d(_evaluate_patch(patch,np.array(centre[0]),np.array(centre[1])))
    dx=5.0*(patch[:,1:]-patch[:,:-1])
    dy=5.0*(patch[1:]-patch[:-1])
    lx=np.max(np.abs(dx),axis=(0,1));ly=np.max(np.abs(dy),axis=(0,1))
    radius=np.max(np.abs(triangle-centre),axis=0)
    oscillation=lx*radius[0]+ly*radius[1]
    lower=np.maximum(np.min(patch,axis=(0,1)),value-oscillation)
    upper=np.minimum(np.max(patch,axis=(0,1)),value+oscillation)
    return value,lower,upper


def _inverse_area_density(
    inverse: ProjectiveMap,points: Array,sx: float,sy: float
)->Array:
    normalized=np.asarray(points,dtype=np.float64)/np.array((sx,sy))
    jacobian=inverse.jacobian(normalized[:,0],normalized[:,1])
    # dp_normalized = dp_index/(sx*sy).
    return np.abs(np.linalg.det(jacobian))/(sx*sy)


def _interval_product_with_positive_weight(
    lower: Array,upper: Array,weight_lower: float,weight_upper: float
)->tuple[Array,Array]:
    products=np.stack((
        lower*weight_lower,lower*weight_upper,
        upper*weight_lower,upper*weight_upper,
    ))
    return np.min(products,axis=0),np.max(products,axis=0)


def _subdivide_triangle(triangle: Array)->list[Array]:
    a,b,c=np.asarray(triangle,dtype=np.float64)
    ab=.5*(a+b);bc=.5*(b+c);ca=.5*(c+a)
    return [np.array((a,ab,ca)),np.array((ab,b,bc)),
            np.array((ca,bc,c)),np.array((ab,bc,ca))]


def _binomial_tail_order_three(radius: float,first_order: int)->float:
    """Sum ``C(k+2,2) r^k`` from ``first_order`` to infinity."""

    r=float(radius);n=int(first_order)
    if not (0.0<=r<1.0):
        return float("inf")
    denominator=1.0-r
    return .5*r**n*(
        r*(1.0+r)/denominator**3
        +(2*n+3)*r/denominator**2
        +(n+1)*(n+2)/denominator
    )


def _certified_triangle_series(
    control: Array,
    triangle: Array,
    origin: Array,
    inverse: ProjectiveMap,
    sx: float,
    sy: float,
    reference: Array,
    budget: float,
    depth: int,
    maximum_depth: int,
    series_order: int=6,
)->tuple[Array,Array,int,int]:
    """Integrate a polynomial times the projective density with a tail proof."""

    tri=np.asarray(triangle,dtype=np.float64)
    edge_b=tri[1]-tri[0];edge_c=tri[2]-tri[0]
    determinant=abs(float(edge_b[0]*edge_c[1]-edge_b[1]*edge_c[0]))
    matrix=inverse.matrix
    normalized=(tri+origin)/np.array((sx,sy),dtype=np.float64)
    denominator=(matrix[2,0]*normalized[:,0]
                 +matrix[2,1]*normalized[:,1]+matrix[2,2])
    if np.min(denominator)*np.max(denominator)<=0.0:
        raise ValueError("a projective footprint intersects the inverse homography pole")
    centre=.5*(float(np.min(denominator))+float(np.max(denominator)))
    radius=float(np.max(np.abs(denominator/centre-1.0)))
    density0=abs(float(np.linalg.det(matrix)))/(abs(centre)**3*sx*sy)
    patch=np.asarray(control,dtype=np.float64)
    scalar=patch.ndim==2
    if scalar:
        patch=patch[...,None]
    residual_control=patch-reference
    maximum_residual=np.max(np.abs(residual_control),axis=(0,1))
    tail=_binomial_tail_order_three(radius,series_order+1)
    error=.5*determinant*density0*tail*maximum_residual
    if float(np.max(error))>budget and depth<maximum_depth:
        estimate=np.zeros_like(error);bound=np.zeros_like(error)
        leaves=0;deepest=depth
        for child in _subdivide_triangle(tri):
            value,uncertainty,count,reached=_certified_triangle_series(
                control,child,origin,inverse,sx,sy,reference,budget/4.0,
                depth+1,maximum_depth,series_order,
            )
            estimate+=value;bound+=uncertainty;leaves+=count;deepest=max(deepest,reached)
        return estimate,bound,leaves,deepest
    total_degree=10+series_order
    quadrature_order=(total_degree+3)//2
    node,weight=_gauss_unit(quadrature_order)
    uu,vv=np.meshgrid(node,node,indexing="ij")
    point=(tri[0]+uu[...,None]*edge_b+(1.0-uu)[...,None]*vv[...,None]*edge_c)
    global_point=point+origin
    normalized_point=global_point/np.array((sx,sy),dtype=np.float64)
    ell=(matrix[2,0]*normalized_point[...,0]
         +matrix[2,1]*normalized_point[...,1]+matrix[2,2])
    z=ell/centre-1.0
    polynomial=np.zeros_like(z);power=np.ones_like(z)
    for k in range(series_order+1):
        polynomial+=((-1.0)**k)*math.comb(k+2,2)*power
        power*=z
    sample=_evaluate_patch(control,point[...,0],point[...,1])
    sample=sample[...,None] if scalar else sample
    sample=sample-reference
    quadrature=(weight[:,None]*weight[None,:]*determinant*(1.0-uu)
                *density0*polynomial)
    estimate=np.einsum("ij,ijc->c",quadrature,sample,optimize=True)
    return estimate,error,1,depth


def _certified_triangle_residual(
    control: Array,
    triangle: Array,
    inverse: ProjectiveMap,
    sx: float,
    sy: float,
    origin: Array,
    reference: Array,
    budget: float,
    depth: int,
    maximum_depth: int,
)->tuple[Array,Array,int,int]:
    tri=np.asarray(triangle,dtype=np.float64)
    edge_b=tri[1]-tri[0];edge_c=tri[2]-tri[0]
    area=.5*abs(float(edge_b[0]*edge_c[1]-edge_b[1]*edge_c[0]))
    _,value_lower,value_upper=_patch_value_bounds(control,tri)
    density=_inverse_area_density(inverse,tri+origin,sx,sy)
    residual_lower=value_lower-reference;residual_upper=value_upper-reference
    product_lower,product_upper=_interval_product_with_positive_weight(
        residual_lower,residual_upper,float(np.min(density)),float(np.max(density))
    )
    lower=area*product_lower;upper=area*product_upper
    if float(np.max(upper-lower))<=budget or depth>=maximum_depth:
        return lower,upper,1,depth
    total_lower=np.zeros_like(lower);total_upper=np.zeros_like(upper)
    leaves=0;deepest=depth
    for child in _subdivide_triangle(tri):
        lo,hi,count,reached=_certified_triangle_residual(
            control,child,inverse,sx,sy,origin,reference,budget/4.0,
            depth+1,maximum_depth,
        )
        total_lower+=lo;total_upper+=hi;leaves+=count;deepest=max(deepest,reached)
    return total_lower,total_upper,leaves,deepest


def certified_projective_warped_pixel_average(
    values: Array,
    transform: ProjectiveMap,
    target_shape: tuple[int,int],
    *,
    tolerance: float=1e-5,
    maximum_depth: int=8,
    control: Array|None=None,
)->tuple[Array,dict[str,float|int]]:
    """Evaluate projective basin means with a rigorous interval enclosure.

    The mathematical operator is the exact target-basin integral.  Numerical
    evaluation works in source coordinates.  On every clipped triangle,
    Bernstein differences bound both partial derivatives of the polynomial
    atlas, while the inverse-homography area density has exact extrema because
    its denominator is linear and pole-free.  Recursive four-way subdivision
    therefore encloses, rather than estimates, every pixel integral.
    """

    if transform.is_affine:
        return exact_affine_warped_pixel_average(
            values,transform,target_shape,control=control
        )
    if tolerance<=0.0 or maximum_depth<0:
        raise ValueError("tolerance must be positive and maximum_depth nonnegative")
    source=np.asarray(values,dtype=np.float64);scalar=source.ndim==2
    if source.ndim not in (2,3):
        raise ValueError("values must be HxW or HxWxC")
    if control is None:
        atlas,admission=finite_joint_control_nets(source)
    else:
        atlas=np.asarray(control,dtype=np.float64);admission={}
    inverse=ProjectiveMap(np.linalg.inv(transform.matrix))
    height,width=map(int,target_shape)
    channels=1 if scalar else source.shape[-1]
    output=np.empty((height,width,channels),dtype=np.float64)
    error=np.empty_like(output)
    xb=endpoint_basin_bounds(width);yb=endpoint_basin_bounds(height)
    sx=float(source.shape[1]-1);sy=float(source.shape[0]-1)
    global_low=np.min(atlas,axis=tuple(range(atlas.ndim-1))) if not scalar else np.array([np.min(atlas)])
    global_high=np.max(atlas,axis=tuple(range(atlas.ndim-1))) if not scalar else np.array([np.max(atlas)])
    outside=0;leaf_count=0;deepest=0;maximum_area_error=0.0
    for row in range(height):
        for column in range(width):
            target_area=(xb[column+1]-xb[column])*(yb[row+1]-yb[row])
            q=np.array(((xb[column],yb[row]),(xb[column+1],yb[row]),
                        (xb[column+1],yb[row+1]),(xb[column],yb[row+1])))
            px,py=transform.map(q[:,0],q[:,1])
            polygon=np.stack((sx*px,sy*py),axis=1)
            xmin,xmax=float(np.min(polygon[:,0])),float(np.max(polygon[:,0]))
            ymin,ymax=float(np.min(polygon[:,1])),float(np.max(polygon[:,1]))
            if xmin<0.0 or ymin<0.0 or xmax>sx or ymax>sy:
                outside+=1;output[row,column]=np.nan;error[row,column]=np.nan;continue
            centre_q=np.array((.5*(xb[column]+xb[column+1]),.5*(yb[row]+yb[row+1])))
            centre_p=transform.map(centre_q[0],centre_q[1])
            reference=np.atleast_1d(evaluate_joint_atlas(
                atlas,np.array(centre_p[0]*sx),np.array(centre_p[1]*sy)
            ))
            source_area=abs(_signed_polygon_area(polygon));covered=0.0
            residual_integral=np.zeros(channels);error_integral=np.zeros(channels)
            initial_triangles=[]
            for cy in range(max(0,int(math.floor(ymin))),min(source.shape[0]-2,int(math.floor(ymax)))+1):
                for cx in range(max(0,int(math.floor(xmin))),min(source.shape[1]-2,int(math.floor(xmax)))+1):
                    clipped=_clip_rectangle(polygon,cx,cx+1.0,cy,cy+1.0)
                    area=abs(_signed_polygon_area(clipped))
                    if area<=256.0*np.finfo(np.float64).eps:
                        continue
                    covered+=area
                    local=clipped-np.array((cx,cy),dtype=np.float64)
                    for k in range(1,local.shape[0]-1):
                        triangle=np.array((local[0],local[k],local[k+1]))
                        tri_edge1=triangle[1]-triangle[0];tri_edge2=triangle[2]-triangle[0]
                        tri_area=.5*abs(float(tri_edge1[0]*tri_edge2[1]-tri_edge1[1]*tri_edge2[0]))
                        initial_triangles.append((cy,cx,triangle,tri_area))
            maximum_area_error=max(maximum_area_error,abs(covered-source_area))
            for cy,cx,triangle,tri_area in initial_triangles:
                # Allocate the requested pixel error in proportion to mapped
                # source area; the child budgets partition exactly by four.
                budget=tolerance*target_area*(tri_area/source_area)
                value,uncertainty,leaves,reached=_certified_triangle_series(
                    atlas[cy,cx],triangle,np.array((cx,cy),dtype=np.float64),
                    inverse,sx,sy,reference,budget,0,maximum_depth,
                )
                residual_integral+=value;error_integral+=uncertainty
                leaf_count+=leaves;deepest=max(deepest,reached)
            output[row,column]=np.clip(
                reference+residual_integral/target_area,global_low,global_high
            )
            error[row,column]=error_integral/target_area
    diagnostic={
        **admission,"interval_tolerance":float(tolerance),
        "maximum_depth_allowed":int(maximum_depth),"maximum_depth_reached":deepest,
        "leaf_triangles":leaf_count,"outside_basins":outside,
        "maximum_source_area_partition_error":maximum_area_error,
        "maximum_certified_error":float(np.nanmax(error)),
    }
    result=output[...,0] if scalar else output
    return result,diagnostic
