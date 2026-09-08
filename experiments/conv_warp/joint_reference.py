"""Joint Bernstein-current admission for direct CONV warps.

The canonical path constructs one shared bidegree-(5,5) potential atlas and
admits its support-current cones and support ranges by a finite analytic law.
Older cyclic projections remain below only as reference destinations used to
derive and falsify candidate admissions; the canonical constructor contains
no iterative solve.
"""

from __future__ import annotations

import math

import numpy as np

from experiments.conv_distilled_core import _variation_jet
from experiments.convstar import (
    ordered_sign_ledger,
    project_signed_fibres,
    raw_current_jet_bank,
)


Array = np.ndarray


def _directional_derivative_coefficients(dx: float,dy: float)->list[Array]:
    norm=float(np.hypot(dx,dy))
    if norm==0.0:
        return []
    dx,dy=dx/norm,dy/norm
    constraints=[]
    for j in range(6):
        for i in range(6):
            coefficient=np.zeros((6,6),dtype=np.float64)
            if dx:
                if i==0:
                    coefficient[j,0]-=5.0*dx;coefficient[j,1]+=5.0*dx
                elif i==5:
                    coefficient[j,4]-=5.0*dx;coefficient[j,5]+=5.0*dx
                else:
                    left=i/5.0;right=1.0-left
                    coefficient[j,i-1]-=5.0*dx*left
                    coefficient[j,i]+=5.0*dx*(left-right)
                    coefficient[j,i+1]+=5.0*dx*right
            if dy:
                if j==0:
                    coefficient[0,i]-=5.0*dy;coefficient[1,i]+=5.0*dy
                elif j==5:
                    coefficient[4,i]-=5.0*dy;coefficient[5,i]+=5.0*dy
                else:
                    top=j/5.0;bottom=1.0-top
                    coefficient[j-1,i]-=5.0*dy*top
                    coefficient[j,i]+=5.0*dy*(top-bottom)
                    coefficient[j+1,i]+=5.0*dy*bottom
            constraints.append(coefficient)
    return constraints


def _witnessed_direction(local_gradient: Array,scale: float)->tuple[float,float]|None:
    gradient=np.asarray(local_gradient,dtype=np.float64).reshape(-1,2)
    magnitude=np.linalg.norm(gradient,axis=1)
    active=magnitude>256.0*np.finfo(np.float64).eps*max(1.0,float(scale))
    if not np.any(active):
        return None
    witnessed=gradient[active];gram=witnessed@witnessed.T
    if not np.all(gram>=-512.0*np.finfo(np.float64).eps*np.max(magnitude)**2):
        return None
    total=np.sum(witnessed,axis=0)
    if np.linalg.norm(total)==0.0:
        return None
    return float(total[0]),float(total[1])


def _support_sum_directions(gx: Array,gy: Array)->Array:
    """Oriented current summed over the exact two-jet cell support.

    Cell ``(j,i)`` reads nodal coordinates ``i-2,...,i+3`` and
    ``j-2,...,j+3`` after clipping at the represented boundary.  Summation
    carries the orientation into adjacent plateaus without selecting an owner
    or introducing a support radius beyond the existing jet stencil.
    """

    horizontal=np.asarray(gx,dtype=np.float64)
    vertical=np.asarray(gy,dtype=np.float64)
    height,width,channels=horizontal.shape
    output=np.zeros((height-1,width-1,channels,2),dtype=np.float64)
    for y in range(height-1):
        top=max(0,y-2);bottom=min(height,y+4)
        for x in range(width-1):
            left=max(0,x-2);right=min(width,x+4)
            output[y,x,:,0]=np.sum(horizontal[top:bottom,left:right],axis=(0,1))
            output[y,x,:,1]=np.sum(vertical[top:bottom,left:right],axis=(0,1))
    return output


def _support_witnessed_directions(gx: Array,gy: Array,values: Array)->Array:
    """Return a support direction only when all measured currents corroborate it.

    Summation alone can merge crossing or reversing currents into a fictitious
    orientation.  Here the same declared six-sample support is used, but its
    active gradient vectors must have pairwise nonnegative inner products.
    Thus they occupy one closed Euclidean cone of aperture at most pi/2 and a
    single oriented current is genuinely witnessed by every contributor.
    """

    horizontal=np.asarray(gx,dtype=np.float64)
    vertical=np.asarray(gy,dtype=np.float64)
    field=np.asarray(values,dtype=np.float64)
    height,width,channels=horizontal.shape
    output=np.zeros((height-1,width-1,channels,2),dtype=np.float64)
    for y in range(height-1):
        top=max(0,y-2);bottom=min(height,y+4)
        for x in range(width-1):
            left=max(0,x-2);right=min(width,x+4)
            for channel in range(channels):
                gradient=np.stack((
                    horizontal[top:bottom,left:right,channel],
                    vertical[top:bottom,left:right,channel],
                ),axis=-1)
                scale=float(np.max(np.abs(field[top:bottom,left:right,channel])))
                direction=_witnessed_direction(gradient,scale)
                if direction is not None:
                    output[y,x,channel]=direction
    return output


def _support_phase_nodes(values: Array)->Array:
    """Build a nodal scalar phase from the declared cell-support means."""

    field=np.asarray(values,dtype=np.float64)
    height,width,channels=field.shape
    cell=np.empty((height-1,width-1,channels),dtype=np.float64)
    for y in range(height-1):
        top=max(0,y-2);bottom=min(height,y+4)
        for x in range(width-1):
            left=max(0,x-2);right=min(width,x+4)
            cell[y,x]=np.mean(field[top:bottom,left:right],axis=(0,1))
    node=np.zeros_like(field);count=np.zeros((height,width,1),dtype=np.float64)
    for dy in (0,1):
        for dx in (0,1):
            node[dy:dy+height-1,dx:dx+width-1]+=cell
            count[dy:dy+height-1,dx:dx+width-1]+=1.0
    return node/count


def _bernstein_product_2d(first: Array,second: Array)->Array:
    """Exact Bernstein coefficients of the product of two tensor polynomials."""

    a=np.asarray(first,dtype=np.float64);b=np.asarray(second,dtype=np.float64)
    my,mx=a.shape[0]-1,a.shape[1]-1
    py,px=b.shape[0]-1,b.shape[1]-1
    output=np.zeros((my+py+1,mx+px+1),dtype=np.float64)
    for j in range(my+1):
        for i in range(mx+1):
            for l in range(py+1):
                for k in range(px+1):
                    output[j+l,i+k]+=(
                        math.comb(my,j)*math.comb(py,l)/math.comb(my+py,j+l)
                        *math.comb(mx,i)*math.comb(px,k)/math.comb(mx+px,i+k)
                        *a[j,i]*b[l,k]
                    )
    return output


def _elevate_axis(coefficients: Array,axis: int,target_degree: int)->Array:
    value=np.asarray(coefficients,dtype=np.float64)
    while value.shape[axis]-1<target_degree:
        degree=value.shape[axis]-1
        moved=np.moveaxis(value,axis,0)
        elevated=np.empty((degree+2,)+moved.shape[1:],dtype=np.float64)
        elevated[0]=moved[0];elevated[-1]=moved[-1]
        for k in range(1,degree+1):
            alpha=k/(degree+1)
            elevated[k]=alpha*moved[k-1]+(1.0-alpha)*moved[k]
        value=np.moveaxis(elevated,0,axis)
    return value


def _phase_gradient_constraints(phase_corners: Array)->list[Array]:
    """Linear controls certifying ``grad(phi).grad(U) >= 0`` on one cell."""

    phase=np.asarray(phase_corners,dtype=np.float64)
    if phase.shape!=(2,2):
        raise ValueError("phase corners must have shape (2,2)")
    # du phi has degrees (1,0); dv phi has degrees (0,1).
    du_phi=np.array(((phase[0,1]-phase[0,0],),
                     (phase[1,1]-phase[1,0],)),dtype=np.float64)
    dv_phi=np.array(((phase[1,0]-phase[0,0],
                      phase[1,1]-phase[0,1]),),dtype=np.float64)
    if max(float(np.max(np.abs(du_phi))),float(np.max(np.abs(dv_phi))))<1e-15:
        return []
    maps=np.empty((7,7,6,6),dtype=np.float64)
    for control_y in range(6):
        for control_x in range(6):
            basis=np.zeros((6,6),dtype=np.float64);basis[control_y,control_x]=1.0
            du=5.0*(basis[:,1:]-basis[:,:-1])
            dv=5.0*(basis[1:]-basis[:-1])
            term_u=_elevate_axis(_bernstein_product_2d(du,du_phi),1,6)
            term_v=_elevate_axis(_bernstein_product_2d(dv,dv_phi),0,6)
            maps[:,:,control_y,control_x]=term_u+term_v
    return [maps[j,i] for j in range(7) for i in range(7)]


def _project_direction_to_q1_dual_cone(
    transported: Array,corners: Array
)->Array:
    """Closest direction whose current is nonnegative on the full Q1 cell."""

    s=np.asarray(transported,dtype=np.float64)
    if np.linalg.norm(s)==0.0:
        return np.zeros(2,dtype=np.float64)
    s=s/np.linalg.norm(s)
    p=np.asarray(corners,dtype=np.float64)
    gradients=np.array((
        (p[0,1]-p[0,0],p[1,0]-p[0,0]),
        (p[0,1]-p[0,0],p[1,1]-p[0,1]),
        (p[1,1]-p[1,0],p[1,0]-p[0,0]),
        (p[1,1]-p[1,0],p[1,1]-p[0,1]),
    ),dtype=np.float64)
    tolerance=512.0*np.finfo(np.float64).eps*max(1.0,float(np.max(np.abs(p))))
    def feasible(candidate: Array)->bool:
        return bool(np.all(gradients@candidate>=-tolerance))
    candidates=[s] if feasible(s) else []
    for gradient in gradients:
        norm2=float(gradient@gradient)
        if norm2==0.0:
            continue
        candidate=s-(s@gradient/norm2)*gradient
        if feasible(candidate):
            candidates.append(candidate)
    candidates.append(np.zeros(2,dtype=np.float64))
    return min(candidates,key=lambda value:float(np.sum((value-s)**2)))


def _completed_direction_constraints(corners: Array,transported: Array)->list[Array]:
    """Transported directional constraints with a constructive Q1 witness."""

    direction=_project_direction_to_q1_dual_cone(transported,corners)
    if np.linalg.norm(direction)>512.0*np.finfo(np.float64).eps:
        return _directional_derivative_coefficients(*direction)
    return []


def _q1_corner_gradients(corners: Array)->Array:
    p=np.asarray(corners,dtype=np.float64)
    return np.array((
        (p[0,1]-p[0,0],p[1,0]-p[0,0]),
        (p[0,1]-p[0,0],p[1,1]-p[0,1]),
        (p[1,1]-p[1,0],p[1,0]-p[0,0]),
        (p[1,1]-p[1,0],p[1,1]-p[0,1]),
    ),dtype=np.float64)


def _dual_generators_of_planar_cone(vectors: Array,scale: float)->list[Array]:
    """Return finite half-space normals defining a generated planar cone.

    The empty list denotes the whole plane (no directional information).
    A pointed wedge needs its two inward boundary normals; a ray additionally
    needs its forward normal; a line needs the two opposing transverse
    normals.  The construction depends only on the circular order of the
    measured currents and contains no angular bandwidth parameter.
    """

    value=np.asarray(vectors,dtype=np.float64).reshape(-1,2)
    magnitude=np.linalg.norm(value,axis=1)
    tolerance=2048.0*np.finfo(np.float64).eps*max(1.0,float(scale))
    value=value[magnitude>tolerance]
    if not value.size:
        return []
    angle=np.sort(np.mod(np.arctan2(value[:,1],value[:,0]),2.0*np.pi))
    gap=np.diff(np.r_[angle,angle[0]+2.0*np.pi])
    cut=int(np.argmax(gap));start=float(angle[(cut+1)%angle.size])
    width=float(2.0*np.pi-gap[cut])
    angular_tolerance=4096.0*np.finfo(np.float64).eps
    if width>np.pi+angular_tolerance:
        return []
    a=start;b=start+width
    first=np.array((-math.sin(a),math.cos(a)),dtype=np.float64)
    second=np.array((math.sin(b),-math.cos(b)),dtype=np.float64)
    if width<=angular_tolerance:
        ray=np.array((math.cos(a),math.sin(a)),dtype=np.float64)
        return [first,second,ray]
    if abs(width-np.pi)<=angular_tolerance:
        relative=np.mod(angle-start,2.0*np.pi)
        interior=np.any((relative>angular_tolerance)&(relative<np.pi-angular_tolerance))
        if interior:
            return [first]
        # Only the two antipodal rays are generated: the cone is a line.
        return [first,-first]
    return [first,second]


def _gradient_cone_constraints(
    values: Array,cell_y: int,cell_x: int,channel: int,*,support: bool
)->list[Array]:
    """Exact Bernstein constraints for a Q1-current cone witness."""

    source=np.asarray(values,dtype=np.float64)
    if support:
        y0=max(0,cell_y-2);y1=min(source.shape[0]-1,cell_y+3)
        x0=max(0,cell_x-2);x1=min(source.shape[1]-1,cell_x+3)
    else:
        y0,y1=cell_y,cell_y+1;x0,x1=cell_x,cell_x+1
    vectors=[]
    for y in range(y0,y1):
        for x in range(x0,x1):
            vectors.extend(_q1_corner_gradients(source[y:y+2,x:x+2,channel]))
    dual=_dual_generators_of_planar_cone(
        np.asarray(vectors),float(np.max(np.abs(source[y0:y1+1,x0:x1+1,channel])))
    )
    constraints=[]
    for normal in dual:
        constraints.extend(_directional_derivative_coefficients(*normal))
    return constraints


def _controls_from_currents(source: Array, current: Array) -> Array:
    controls = np.empty((source.shape[0]-1, 6, source.shape[1]), dtype=np.float64)
    controls[:, 0] = source[:-1]
    controls[:, 1:] = source[:-1, None, :] + np.cumsum(current, axis=1)
    # Preserve the cardinal endpoint exactly rather than relying on the last
    # floating sum of conservative currents.
    controls[:, -1] = source[1:]
    return controls


def line_control_bank(lines: Array, *, admitted: bool) -> Array:
    """Return every raw or admitted quintic Bernstein control vector."""

    source = np.asarray(lines, dtype=np.float64)
    raw, delta = raw_current_jet_bank(source)
    if admitted:
        signs = ordered_sign_ledger(raw, delta)
        current = project_signed_fibres(raw, signs, delta)
    else:
        current = raw
    return _controls_from_currents(source, current)


def raw_tensor_control_net(values: Array) -> Array:
    """Return commuting raw two-jet controls with shape ``Cy,Cx,6,6,C``."""

    field = np.asarray(values, dtype=np.float64)
    if field.ndim == 2:
        field = field[..., None]
    height, width, channels = field.shape
    horizontal = line_control_bank(
        np.moveaxis(field, 1, 0).reshape(width, height*channels), admitted=False
    )
    horizontal = horizontal.reshape(width-1, 6, height, channels).transpose(2,0,1,3)
    vertical_input = horizontal.transpose(0,1,2,3).reshape(height, -1)
    vertical = line_control_bank(vertical_input, admitted=False)
    return vertical.reshape(height-1, 6, width-1, 6, channels).transpose(0,2,1,3,4)


def canonical_sampled_control_net(values: Array) -> Array:
    """Construct the fixed collocation proposal for the joint CONV atlas.

    The proposal is the unique bidegree-(5,5) Bernstein patch interpolating
    complete two-order CONV synthesis on the 6-by-6 fifth-step tensor lattice.
    Native factor passes evaluate that lattice once globally; multiplication
    by the fixed Bernstein collocation inverse then forms every patch.
    """

    from experiments.conv_distilled_core import distilled_conv_synthesis

    field=np.asarray(values,dtype=np.float32)
    scalar=field.ndim==2
    if scalar:
        field=field[...,None]
    height,width,channels=field.shape
    phase=np.linspace(0.0,1.0,6)
    collocation=np.array([
        [math.comb(5,k)*u**k*(1.0-u)**(5-k) for k in range(6)]
        for u in phase
    ],dtype=np.float64)
    inverse=np.linalg.inv(collocation)
    # Every cell collocation node belongs to one global fifth-step lattice.
    # Evaluating it once avoids recomputing the complete source analysis for
    # every patch and gives incident cells bit-identical boundary samples.
    global_y=np.linspace(0.0,1.0,5*(height-1)+1)
    global_x=np.linspace(0.0,1.0,5*(width-1)+1)
    sampled_global=np.asarray(distilled_conv_synthesis(
        field,(global_y.size,global_x.size)
    ),dtype=np.float64)
    if sampled_global.ndim==2:
        sampled_global=sampled_global[...,None]
    output=np.empty((height-1,width-1,6,6,channels),dtype=np.float64)
    for y in range(height-1):
        for x in range(width-1):
            sampled=sampled_global[5*y:5*y+6,5*x:5*x+6]
            output[y,x]=np.einsum(
                "ai,ijc,bj->abc",inverse,sampled,inverse,optimize=True
            )
    return output[...,0] if scalar else output


def _global_lattice_from_atlas(atlas: Array) -> Array:
    """Fuse a C0 patch atlas into its single shared control lattice."""

    patch=np.asarray(atlas,dtype=np.float64)
    if patch.ndim==4:
        patch=patch[...,None]
    height_cells,width_cells=patch.shape[:2]
    value=np.zeros((5*height_cells+1,5*width_cells+1,patch.shape[-1]),dtype=np.float64)
    count=np.zeros(value.shape[:2]+(1,),dtype=np.float64)
    for cy in range(height_cells):
        for cx in range(width_cells):
            region=np.s_[5*cy:5*cy+6,5*cx:5*cx+6]
            value[region]+=patch[cy,cx]
            count[region]+=1.0
    return value/count


def _q1_global_lattice(values: Array) -> Array:
    """Degree-elevated bilinear atlas on the shared five-subcell lattice."""

    source=np.asarray(values,dtype=np.float64)
    if source.ndim==2:
        source=source[...,None]
    height,width,channels=source.shape
    lattice=np.empty((5*(height-1)+1,5*(width-1)+1,channels),dtype=np.float64)
    for gy in range(lattice.shape[0]):
        cy=min(gy//5,height-2);v=(gy-5*cy)/5.0
        for gx in range(lattice.shape[1]):
            cx=min(gx//5,width-2);u=(gx-5*cx)/5.0
            lattice[gy,gx]=(
                (1.0-u)*(1.0-v)*source[cy,cx]
                +u*(1.0-v)*source[cy,cx+1]
                +(1.0-u)*v*source[cy+1,cx]
                +u*v*source[cy+1,cx+1]
            )
    return lattice


def _shared_support_bounds(values: Array)->tuple[Array,Array]:
    """Intersect declared two-jet value ranges at every shared control."""

    source=np.asarray(values,dtype=np.float64)
    if source.ndim==2:
        source=source[...,None]
    height,width,channels=source.shape
    shape=(5*(height-1)+1,5*(width-1)+1,channels)
    lower=np.full(shape,-np.inf,dtype=np.float64)
    upper=np.full(shape,np.inf,dtype=np.float64)
    for cy in range(height-1):
        top=max(0,cy-2);bottom=min(height,cy+4)
        for cx in range(width-1):
            left=max(0,cx-2);right=min(width,cx+4)
            low=np.min(source[top:bottom,left:right],axis=(0,1))
            high=np.max(source[top:bottom,left:right],axis=(0,1))
            region=np.s_[5*cy:5*cy+6,5*cx:5*cx+6]
            lower[region]=np.maximum(lower[region],low)
            upper[region]=np.minimum(upper[region],high)
    if np.any(lower>upper+2048.0*np.finfo(np.float64).eps):
        raise RuntimeError("incident support ranges have empty intersection")
    return lower,upper


def _topological_control_groups(height: int,width: int)->tuple[list[Array],Array]:
    """Partition nonvertex controls into shared-edge and cell-interior entities."""

    lattice_width=5*(width-1)+1
    groups:list[Array]=[]
    owner=np.full((5*(height-1)+1,lattice_width),-1,dtype=np.int64)
    def add(coordinates:list[tuple[int,int]])->None:
        index=np.array([y*lattice_width+x for y,x in coordinates],dtype=np.int64)
        group=len(groups);groups.append(index)
        for y,x in coordinates:
            if owner[y,x]>=0:
                raise RuntimeError("control entity partition overlaps")
            owner[y,x]=group
    # Each complete grid edge is one shared entity.  Its four nonvertex
    # controls are consequently read identically by both incident cells.
    for y in range(height):
        gy=5*y
        for x in range(width-1):
            add([(gy,5*x+k) for k in range(1,5)])
    for x in range(width):
        gx=5*x
        for y in range(height-1):
            add([(5*y+k,gx) for k in range(1,5)])
    # Cell-interior controls do not occur in any neighbouring patch.
    for y in range(height-1):
        for x in range(width-1):
            add([(5*y+j,5*x+i) for j in range(1,5) for i in range(1,5)])
    return groups,owner.reshape(-1)


def _finite_capacity_admission(
    baseline: Array,candidate: Array,constraints: list[tuple[Array,Array]]
)->tuple[Array,dict[str,float|int]]:
    """Admit entity corrections by simultaneous exact margin capacities.

    For constraint ``a_k^T z >= 0``, write the candidate correction as a sum
    of disjoint topological-entity corrections ``d_g``.  If

        H_k = sum_g max(0, -a_k^T d_g),

    then assigning every harmful entity at most ``(a_k^T b)/H_k`` of its
    correction leaves the inequality nonnegative.  Taking the minimum of
    these capacities over incident constraints is finite, order independent,
    and preserves every shared control exactly once.  A final common ray step
    consumes any margin left by the simultaneous allocation.
    """

    base=np.asarray(baseline,dtype=np.float64)
    target=np.asarray(candidate,dtype=np.float64)
    if base.shape!=target.shape or base.ndim!=2:
        raise ValueError("capacity admission expects equal scalar lattices")
    height=(base.shape[0]-1)//5+1;width=(base.shape[1]-1)//5+1
    groups,owner=_topological_control_groups(height,width)
    flat_base=base.reshape(-1);flat_target=target.reshape(-1)
    delta=flat_target-flat_base
    capacities=np.ones(len(groups),dtype=np.float64)
    scale=max(1.0,float(np.max(np.abs(flat_base))),float(np.max(np.abs(flat_target))))
    epsilon=2048.0*np.finfo(np.float64).eps*scale
    if not constraints:
        return target.copy(),{
            "constraint_count":0,"active_constraints":0,
            "minimum_baseline_margin":0.0,"minimum_candidate_margin":0.0,
            "minimum_admitted_margin":0.0,"minimum_entity_capacity":1.0,
            "mean_entity_capacity":1.0,"completion_ray":1.0,
        }
    candidate_min=float("inf");baseline_min=float("inf")
    for index,coefficient in constraints:
        baseline_min=min(baseline_min,float(coefficient@flat_base[index]))
        candidate_min=min(candidate_min,float(coefficient@flat_target[index]))
    if candidate_min>=-epsilon:
        return target.copy(),{
            "constraint_count":len(constraints),"active_constraints":0,
            "minimum_baseline_margin":baseline_min,
            "minimum_candidate_margin":candidate_min,
            "minimum_admitted_margin":candidate_min,
            "minimum_entity_capacity":1.0,"mean_entity_capacity":1.0,
            "completion_ray":1.0,
        }
    active_constraints=0
    for index,coefficient in constraints:
        margin=float(coefficient@flat_base[index])
        proposed=float(coefficient@flat_target[index])
        group_ids=np.unique(owner[index])
        harmful:list[tuple[int,float]]=[]
        total=0.0
        for group in group_ids:
            if group<0:
                continue
            mask=owner[index]==group
            contribution=float(coefficient[mask]@delta[index[mask]])
            if contribution<0.0:
                loss=-contribution;harmful.append((int(group),loss));total+=loss
        if total<=epsilon:
            continue
        active_constraints+=int(proposed<0.0)
        ratio=float(np.clip(max(0.0,margin)/total,0.0,1.0))
        for group,_ in harmful:
            capacities[group]=min(capacities[group],ratio)
    alpha=np.zeros_like(delta)
    assigned=owner>=0
    alpha[assigned]=capacities[owner[assigned]]
    admitted=flat_base+alpha*delta
    # The remaining candidate direction is one affine ray.  Its exact first
    # intersection with the polytope is therefore the minimum scalar ratio.
    remaining=flat_target-admitted
    ray=1.0
    for index,coefficient in constraints:
        margin=float(coefficient@admitted[index])
        motion=float(coefficient@remaining[index])
        if motion<0.0:
            ray=min(ray,float(np.clip(max(0.0,margin)/(-motion),0.0,1.0)))
    admitted+=ray*remaining
    minimum=float("inf")
    for index,coefficient in constraints:
        minimum=min(minimum,float(coefficient@admitted[index]))
    return admitted.reshape(base.shape),{
        "constraint_count":len(constraints),
        "active_constraints":active_constraints,
        "minimum_baseline_margin":baseline_min,
        "minimum_candidate_margin":candidate_min,
        "minimum_admitted_margin":minimum,
        "minimum_entity_capacity":float(np.min(capacities)) if capacities.size else 1.0,
        "mean_entity_capacity":float(np.mean(capacities)) if capacities.size else 1.0,
        "completion_ray":ray,
    }


def admitted_boundary_banks(values: Array) -> tuple[Array, Array]:
    """Return shared horizontal and vertical admitted edge controls."""

    field = np.asarray(values, dtype=np.float64)
    if field.ndim == 2:
        field = field[..., None]
    height, width, channels = field.shape
    horizontal = line_control_bank(
        np.moveaxis(field, 1, 0).reshape(width, height*channels), admitted=True
    ).reshape(width-1, 6, height, channels).transpose(2,0,1,3)
    vertical = line_control_bank(
        field.reshape(height, width*channels), admitted=True
    ).reshape(height-1, 6, width, channels).transpose(0,2,1,3)
    return horizontal, vertical


def _consistent_sign(difference: Array, tolerance: float) -> int:
    value = np.asarray(difference, dtype=np.float64)
    sign = np.sign(value[np.abs(value) > tolerance])
    if not sign.size:
        return 0
    return int(sign[0]) if np.all(sign == sign[0]) else 0


def _reference_project_patch(
    raw: Array,
    boundary: Array,
    corners: Array,
    *,
    tolerance: float = 2e-12,
    maximum_sweeps: int = 20000,
    enforce_range: bool = False,
    direction: tuple[float,float] | None = None,
) -> tuple[Array, dict[str, float | int]]:
    """Dykstra projection onto fixed edges, range, and witnessed face cones."""

    value = np.asarray(raw, dtype=np.float64).copy()
    fixed = np.zeros((6,6), dtype=bool)
    fixed[0] = fixed[-1] = True
    fixed[:,0] = fixed[:,-1] = True
    low, high = float(np.min(corners)), float(np.max(corners))
    scale = max(1.0, abs(low), abs(high))
    sign_tolerance = 256.0*np.finfo(np.float64).eps*scale
    # A directional cone is witnessed only when the complete pair of shared
    # boundary curves already has that order.  Corner signs alone are
    # insufficient: two admitted transverse edge curves can cross between
    # their endpoints even when both corner secants agree.
    sx = _consistent_sign(np.concatenate((
        np.diff(boundary[0]), np.diff(boundary[-1]),
        boundary[:,5]-boundary[:,0],
    )), sign_tolerance)
    sy = _consistent_sign(np.concatenate((
        np.diff(boundary[:,0]), np.diff(boundary[:,-1]),
        boundary[5,:]-boundary[0,:],
    )), sign_tolerance)
    constraints: list[Array] = []
    def pair_constraint(a: tuple[int,int],b: tuple[int,int],orientation: int)->Array:
        coefficient=np.zeros((6,6),dtype=np.float64)
        coefficient[a]-=orientation;coefficient[b]+=orientation
        return coefficient
    if sx:
        constraints.extend(pair_constraint((y,x),(y,x+1),sx)
                           for y in range(6) for x in range(5))
    if sy:
        constraints.extend(pair_constraint((y,x),(y+1,x),sy)
                           for y in range(5) for x in range(6))

    directional_constraints: list[Array]=[]
    direction_boundary_conflict=0
    if direction is not None:
        dx,dy=map(float,direction)
        norm=float(np.hypot(dx,dy))
        if norm>0.0:
            dx,dy=dx/norm,dy/norm
            directional_constraints.extend(
                _directional_derivative_coefficients(dx,dy)
            )

    # A constraint containing no free interior coefficient is a statement
    # solely about the already fixed 1-D edge traces.  If it is violated, no
    # interior patch can repair the directional current; joint admission must
    # alter shared edge currents globally.  Record and omit the impossible
    # directional family rather than silently returning a nonfeasible patch.
    for coefficient in directional_constraints:
        if np.sum(np.where(fixed,0.0,coefficient)**2)==0.0 and float(
            np.sum(coefficient*boundary)
        ) < -sign_tolerance:
            direction_boundary_conflict=1
            directional_constraints=[]
            break
    constraints.extend(directional_constraints)

    # Dykstra corrections make the cyclic half-space projections converge to
    # the Euclidean projection, rather than merely to an arbitrary feasible
    # point.  Fixed edges and the range box are included as convex sets.
    value[fixed] = boundary[fixed]
    corrections = [np.zeros_like(value) for _ in range((1 if enforce_range else 0)+len(constraints))]
    previous = value.copy()
    residual = float("inf")
    for sweep in range(maximum_sweeps):
        offset=0
        if enforce_range:
            candidate = value + corrections[0]
            projected = candidate.copy()
            projected[~fixed] = np.clip(projected[~fixed], low, high)
            projected[fixed] = boundary[fixed]
            corrections[0] = candidate-projected
            value = projected
            offset=1

        for constraint_index,coefficient in enumerate(constraints):
            item=offset+constraint_index
            candidate = value + corrections[item]
            candidate[fixed] = boundary[fixed]
            projected = candidate.copy()
            violation = float(np.sum(coefficient*candidate))
            if violation < 0.0:
                free_coefficient=np.where(fixed,0.0,coefficient)
                denominator=float(np.sum(free_coefficient*free_coefficient))
                if denominator==0.0:
                    raise RuntimeError("witnessed boundary cone is infeasible")
                projected -= (violation/denominator)*free_coefficient
            projected[fixed] = boundary[fixed]
            corrections[item] = candidate-projected
            value = projected

        residual = float(np.max(np.abs(value-previous)))
        previous[...] = value
        if residual <= tolerance*scale:
            break
    else:
        raise RuntimeError("joint reference projection did not converge")

    face_violation = 0.0
    for coefficient in constraints:
        face_violation = max(face_violation, -float(np.sum(coefficient*value)))
    fixed_error = float(np.max(np.abs(value[fixed]-boundary[fixed])))
    range_violation = max(low-float(np.min(value)), float(np.max(value))-high, 0.0)
    return value, {
        "sweeps": sweep+1,
        "step_residual": residual,
        "fixed_error": fixed_error,
        "range_violation": range_violation,
        "face_violation": float(face_violation),
        "horizontal_sign": sx,
        "vertical_sign": sy,
        "constraint_count":len(constraints),
        "direction_boundary_conflict":direction_boundary_conflict,
    }


def joint_control_nets(
    values: Array, *, enforce_range: bool=False, proposal: str="raw_tensor"
) -> tuple[Array, dict[str, float | int]]:
    """Build the complete C0 joint-admitted bivariate control-net atlas."""

    field = np.asarray(values, dtype=np.float64)
    scalar = field.ndim == 2
    if scalar:
        field = field[..., None]
    if field.ndim != 3 or min(field.shape[:2]) < 5:
        raise ValueError("joint reference requires HxW or HxWxC, H,W >= 5")
    if proposal=="raw_tensor":
        raw = raw_tensor_control_net(field)
    elif proposal=="canonical_sampled":
        raw = canonical_sampled_control_net(field)
    else:
        raise ValueError("proposal must be 'raw_tensor' or 'canonical_sampled'")
    horizontal, vertical = admitted_boundary_banks(field)
    gx,gy=_variation_jet(field)
    output = raw.copy()
    maximum = {
        "sweeps": 0,
        "step_residual": 0.0,
        "fixed_error": 0.0,
        "range_violation": 0.0,
        "face_violation": 0.0,
        "direction_boundary_conflict":0,
    }
    for y in range(field.shape[0]-1):
        for x in range(field.shape[1]-1):
            for channel in range(field.shape[2]):
                boundary = output[y,x,:,:,channel].copy()
                boundary[0] = horizontal[y,x,:,channel]
                boundary[-1] = horizontal[y+1,x,:,channel]
                boundary[:,0] = vertical[y,x,:,channel]
                boundary[:,-1] = vertical[y,x+1,:,channel]
                corners = field[y:y+2,x:x+2,channel]
                local_gradient=np.stack((gx[y:y+2,x:x+2,channel],
                                         gy[y:y+2,x:x+2,channel]),axis=-1)
                direction=_witnessed_direction(local_gradient,np.max(np.abs(corners)))
                try:
                    admitted, diagnostic = _reference_project_patch(
                        output[y,x,:,:,channel], boundary, corners,
                        enforce_range=enforce_range,
                        direction=direction,
                    )
                except RuntimeError as error:
                    if direction is None or "boundary cone is infeasible" not in str(error):
                        raise
                    admitted, diagnostic = _reference_project_patch(
                        output[y,x,:,:,channel], boundary, corners,
                        enforce_range=enforce_range,
                        direction=None,
                    )
                    diagnostic["direction_boundary_conflict"]=1
                output[y,x,:,:,channel] = admitted
                for key in maximum:
                    maximum[key] = max(maximum[key], diagnostic[key])
    return (output[...,0] if scalar else output), maximum


def global_joint_control_nets(
    values: Array,
    *,
    maximum_sweeps: int=1000,
    tolerance: float=1e-9,
    exact_projection: bool=False,
    witness_mode: str="completed_direction",
    relaxation: float=1.0,
    symmetric_sweeps: bool=True,
    external_direction: Array | None=None,
) -> tuple[Array,dict[str,float|int]]:
    """Project one globally shared control lattice onto joint current cones.

    Only source nodes are fixed.  Edge controls are single global variables,
    so both incident patches receive the same readmitted trace.  Dykstra
    corrections are scalar because every half-space correction is parallel
    to its sparse normal; the reference therefore avoids a dense solver even
    though it remains iterative and noncanonical.
    """

    source=np.asarray(values,dtype=np.float64)
    scalar=source.ndim==2
    if scalar:
        source=source[...,None]
    if source.ndim!=3 or min(source.shape[:2])<5:
        raise ValueError("global joint reference requires HxW or HxWxC, H,W >= 5")
    initial=canonical_sampled_control_net(source)
    height,width,channels=source.shape
    global_shape=(5*(height-1)+1,5*(width-1)+1)
    atlas=np.empty_like(initial)
    gx,gy=_variation_jet(source)
    support_direction=_support_witnessed_directions(gx,gy,source)
    phase_nodes=_support_phase_nodes(source)
    if external_direction is not None:
        supplied=np.asarray(external_direction,dtype=np.float64)
        expected=(height-1,width-1,channels,2)
        if supplied.shape==expected[:-2]+(2,) and channels==1:
            supplied=supplied[...,None,:]
        if supplied.shape!=expected:
            raise ValueError(f"external_direction must have shape {expected}")
    else:
        supplied=None
    aggregate={"sweeps":0,"constraint_count":0,"witnessed_cells":0,
               "fixed_only_conflicts":0,"maximum_violation":0.0,
               "step_residual":0.0,"converged":1}
    for channel in range(channels):
        value=np.zeros(global_shape,dtype=np.float64)
        count=np.zeros(global_shape,dtype=np.float64)
        for cy in range(height-1):
            for cx in range(width-1):
                value[5*cy:5*cy+6,5*cx:5*cx+6]+=initial[cy,cx,:,:,channel]
                count[5*cy:5*cy+6,5*cx:5*cx+6]+=1.0
        value/=count
        fixed=np.zeros(global_shape,dtype=bool)
        fixed[::5,::5]=True
        value[::5,::5]=source[...,channel]
        flat=value.reshape(-1);fixed_flat=fixed.reshape(-1)
        sparse_constraints=[]
        witnessed_cells=0;conflicts=0
        for cy in range(height-1):
            for cx in range(width-1):
                phase_constraints=None
                if supplied is not None:
                    vector=supplied[cy,cx,channel]
                    direction=(float(vector[0]),float(vector[1])) if np.linalg.norm(vector)>(
                        512.0*np.finfo(np.float64).eps
                    ) else None
                elif witness_mode=="corner_cone":
                    local_gradient=np.stack((gx[cy:cy+2,cx:cx+2,channel],
                                             gy[cy:cy+2,cx:cx+2,channel]),axis=-1)
                    direction=_witnessed_direction(
                        local_gradient,np.max(np.abs(source[cy:cy+2,cx:cx+2,channel]))
                    )
                elif witness_mode=="support_sum":
                    vector=support_direction[cy,cx,channel]
                    scale=max(1.0,float(np.max(np.abs(source[cy:cy+2,cx:cx+2,channel]))))
                    direction=(float(vector[0]),float(vector[1])) if np.linalg.norm(vector)>(
                        512.0*np.finfo(np.float64).eps*scale
                    ) else None
                elif witness_mode=="phase_gradient":
                    direction=None
                    phase_constraints=_phase_gradient_constraints(
                        phase_nodes[cy:cy+2,cx:cx+2,channel]
                    )
                elif witness_mode=="completed_direction":
                    direction=None
                    phase_constraints=_completed_direction_constraints(
                        source[cy:cy+2,cx:cx+2,channel],
                        support_direction[cy,cx,channel],
                    )
                else:
                    raise ValueError(
                        "witness_mode must be 'corner_cone', 'support_sum', "
                        "'phase_gradient', or 'completed_direction'"
                    )
                if direction is None:
                    if phase_constraints is None:
                        continue
                    local_constraints=phase_constraints
                else:
                    local_constraints=_directional_derivative_coefficients(*direction)
                cell_constraints=[];cell_conflict=False
                for coefficient in local_constraints:
                    ly,lx=np.nonzero(coefficient)
                    index=(5*cy+ly)*global_shape[1]+5*cx+lx
                    coeff=coefficient[ly,lx]
                    free=~fixed_flat[index]
                    constant=float(np.sum(coeff[~free]*flat[index[~free]]))
                    if not np.any(free):
                        if constant < -1e-12:
                            cell_conflict=True;break
                        continue
                    cell_constraints.append((index[free],coeff[free],constant))
                if cell_conflict:
                    conflicts+=1;continue
                witnessed_cells+=1;sparse_constraints.extend(cell_constraints)
        correction=np.zeros(len(sparse_constraints),dtype=np.float64)
        scale=max(1.0,float(np.max(np.abs(flat))))
        residual=float("inf")
        if not (0.0<relaxation<2.0):
            raise ValueError("projection relaxation must lie in (0,2)")
        orders=(range(len(sparse_constraints)),range(len(sparse_constraints)-1,-1,-1)) \
            if symmetric_sweeps else (range(len(sparse_constraints)),)
        for sweep in range(maximum_sweeps):
            residual=0.0
            for order in orders:
                for k in order:
                    index,coefficient,constant=sparse_constraints[k]
                    previous=flat[index].copy()
                    candidate=(previous+correction[k]*coefficient
                               if exact_projection else previous)
                    signed=float(coefficient@candidate+constant)
                    denominator=float(coefficient@coefficient)
                    if signed<0.0:
                        factor=(signed/denominator)*(1.0 if exact_projection else relaxation)
                        updated=candidate-factor*coefficient
                        correction[k]=signed/denominator if exact_projection else 0.0
                    else:
                        updated=candidate;correction[k]=0.0
                    flat[index]=updated
                    residual=max(residual,float(np.max(np.abs(updated-previous))))
            flat[fixed_flat]=source[...,channel].reshape(-1)
            if residual<=tolerance*scale:
                break
        else:
            aggregate["converged"]=0
            sweep=maximum_sweeps-1
        maximum_violation=0.0
        for index,coefficient,constant in sparse_constraints:
            maximum_violation=max(maximum_violation,-float(coefficient@flat[index]+constant))
        value=flat.reshape(global_shape)
        for cy in range(height-1):
            for cx in range(width-1):
                atlas[cy,cx,:,:,channel]=value[5*cy:5*cy+6,5*cx:5*cx+6]
        aggregate["sweeps"]=max(aggregate["sweeps"],sweep+1)
        aggregate["constraint_count"]+=len(sparse_constraints)
        aggregate["witnessed_cells"]+=witnessed_cells
        aggregate["fixed_only_conflicts"]+=conflicts
        aggregate["maximum_violation"]=max(aggregate["maximum_violation"],maximum_violation)
        aggregate["step_residual"]=max(aggregate["step_residual"],residual)
    return (atlas[...,0] if scalar else atlas),aggregate


def _experimental_finite_joint_control_nets(
    values: Array,*,witness_mode: str="support_sum",
    enforce_support_range: bool=True,
)->tuple[Array,dict[str,float|int]]:
    """Construct a cardinal C0 joint atlas by finite entity admission.

    The safe state is the degree-elevated Q1 atlas.  Each cell admits the cone
    generated by Q1 currents over its declared two-jet support, which contains
    the baseline gradient constructively.  Shared-edge and cell-interior
    corrections toward the canonical sampled CONV atlas are then admitted by
    :func:`_finite_capacity_admission`.  When requested, intersected support
    ranges also bound every shared Bernstein control before current admission.
    """

    source=np.asarray(values,dtype=np.float64)
    scalar=source.ndim==2
    if scalar:
        source=source[...,None]
    if source.ndim!=3 or min(source.shape[:2])<5:
        raise ValueError("finite joint admission requires HxW or HxWxC, H,W >= 5")
    height,width,channels=source.shape
    candidate=_global_lattice_from_atlas(canonical_sampled_control_net(source))
    baseline=_q1_global_lattice(source)
    # Remove collocation roundoff at the cardinal source nodes.
    candidate[::5,::5]=source
    clipped_controls=0
    if enforce_support_range:
        lower,upper=_shared_support_bounds(source)
        clipped=np.clip(candidate,lower,upper)
        clipped_controls=int(np.sum(clipped!=candidate))
        candidate=clipped
    gx,gy=_variation_jet(source)
    if witness_mode=="support_sum":
        support_direction=_support_sum_directions(gx,gy)
    elif witness_mode=="support_cone":
        support_direction=_support_witnessed_directions(gx,gy,source)
    elif witness_mode in ("local_gradient_cone","support_gradient_cone"):
        support_direction=None
    else:
        raise ValueError(
            "witness_mode must be 'support_sum', 'support_cone', "
            "'local_gradient_cone', or 'support_gradient_cone'"
        )
    atlas=np.empty((height-1,width-1,6,6,channels),dtype=np.float64)
    aggregate:dict[str,float|int]={
        "constraint_count":0,"active_constraints":0,
        "minimum_baseline_margin":float("inf"),
        "minimum_candidate_margin":float("inf"),
        "minimum_admitted_margin":float("inf"),
        "minimum_entity_capacity":1.0,"mean_entity_capacity":0.0,
        "completion_ray":0.0,
        "support_range_clipped_controls":clipped_controls,
    }
    for channel in range(channels):
        constraints:list[tuple[Array,Array]]=[]
        for cy in range(height-1):
            for cx in range(width-1):
                if support_direction is None:
                    local=_gradient_cone_constraints(
                        source,cy,cx,channel,
                        support=witness_mode=="support_gradient_cone",
                    )
                else:
                    local=_completed_direction_constraints(
                        source[cy:cy+2,cx:cx+2,channel],
                        support_direction[cy,cx,channel],
                    )
                for coefficient in local:
                    ly,lx=np.nonzero(coefficient)
                    index=(5*cy+ly)*candidate.shape[1]+5*cx+lx
                    constraints.append((index.astype(np.int64),coefficient[ly,lx]))
        admitted,diagnostic=_finite_capacity_admission(
            baseline[...,channel],candidate[...,channel],constraints
        )
        for cy in range(height-1):
            for cx in range(width-1):
                atlas[cy,cx,:,:,channel]=admitted[
                    5*cy:5*cy+6,5*cx:5*cx+6
                ]
        aggregate["constraint_count"]+=int(diagnostic["constraint_count"])
        aggregate["active_constraints"]+=int(diagnostic["active_constraints"])
        for key in ("minimum_baseline_margin","minimum_candidate_margin",
                    "minimum_admitted_margin","minimum_entity_capacity"):
            aggregate[key]=min(float(aggregate[key]),float(diagnostic[key]))
        aggregate["mean_entity_capacity"]+=float(diagnostic["mean_entity_capacity"])/channels
        aggregate["completion_ray"]+=float(diagnostic["completion_ray"])/channels
    return (atlas[...,0] if scalar else atlas),aggregate


def finite_joint_control_nets(values: Array)->tuple[Array,dict[str,float|int]]:
    """Construct the single canonical joint CONV control atlas."""

    return _experimental_finite_joint_control_nets(
        values,witness_mode="support_gradient_cone",enforce_support_range=True
    )


def _bernstein5(u: float) -> Array:
    return np.array([
        math.comb(5,k)*u**k*(1.0-u)**(5-k) for k in range(6)
    ], dtype=np.float64)


def evaluate_joint_atlas(control: Array, x: Array, y: Array) -> Array:
    """Evaluate a joint control atlas at source-index coordinates."""

    atlas = np.asarray(control, dtype=np.float64)
    scalar = atlas.ndim == 4
    if scalar:
        atlas = atlas[...,None]
    query_x, query_y = np.broadcast_arrays(
        np.asarray(x,dtype=np.float64), np.asarray(y,dtype=np.float64)
    )
    output = np.empty(query_x.shape+(atlas.shape[-1],), dtype=np.float64)
    for index in np.ndindex(query_x.shape):
        qx = float(np.clip(query_x[index],0.0,atlas.shape[1]))
        qy = float(np.clip(query_y[index],0.0,atlas.shape[0]))
        ix = min(int(np.floor(qx)),atlas.shape[1]-1)
        iy = min(int(np.floor(qy)),atlas.shape[0]-1)
        bx = _bernstein5(qx-ix)
        by = _bernstein5(qy-iy)
        output[index] = np.einsum("i,j,ijc->c",by,bx,atlas[iy,ix],optimize=True)
    return output[...,0] if scalar else output


def direct_joint_warp(
    values: Array,
    transform: "object",
    target_shape: tuple[int,int],
    *,
    enforce_range: bool=False,
    proposal: str="raw_tensor",
) -> tuple[Array, dict[str,float|int]]:
    """Evaluate the reference joint atlas through one direct inverse map."""

    control, diagnostic = joint_control_nets(
        values,enforce_range=enforce_range,proposal=proposal
    )
    qy, qx = np.meshgrid(
        np.linspace(0.0,1.0,int(target_shape[0])),
        np.linspace(0.0,1.0,int(target_shape[1])),
        indexing="ij",
    )
    px, py = transform.map(qx,qy)
    source_shape = np.asarray(values).shape[:2]
    result = evaluate_joint_atlas(
        control, px*(source_shape[1]-1), py*(source_shape[0]-1)
    )
    return np.asarray(result,dtype=np.float32), diagnostic
