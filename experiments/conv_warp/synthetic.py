"""Procedural fields with direct and, for affine maps, exact raster truth.

The Fourier family is deliberately used as the first warp corpus.  Its value
at a transformed point is closed form, while its average over an affine image
of any rectangular target basin is also closed form.  Thus interpolation and
minification can be scored against the generating field rather than against
the output of a competing resampler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ProjectiveMap


Array = np.ndarray


def endpoint_basin_bounds(count: int) -> Array:
    """Voronoi basin boundaries for ``count`` endpoint-aligned nodes."""

    count = int(count)
    if count < 2:
        raise ValueError("at least two endpoint nodes are required")
    sites = np.linspace(0.0, 1.0, count)
    bounds = np.empty(count + 1, dtype=np.float64)
    bounds[0], bounds[-1] = 0.0, 1.0
    bounds[1:-1] = 0.5 * (sites[:-1] + sites[1:])
    return bounds


@dataclass(frozen=True)
class FourierField:
    """A finite real Fourier field on the plane.

    ``wavevectors`` are cycles per normalized source coordinate.  Amplitudes
    may be ``(K,)`` for a scalar field or ``(K,C)`` for a multichannel field.
    """

    wavevectors: Array
    amplitudes: Array
    phases: Array
    offset: Array | float = 0.5

    def __post_init__(self) -> None:
        wavevectors = np.asarray(self.wavevectors, dtype=np.float64)
        amplitudes = np.asarray(self.amplitudes, dtype=np.float64)
        phases = np.asarray(self.phases, dtype=np.float64)
        if wavevectors.ndim != 2 or wavevectors.shape[1] != 2:
            raise ValueError("wavevectors must have shape (K,2)")
        if amplitudes.ndim not in (1, 2) or amplitudes.shape[0] != wavevectors.shape[0]:
            raise ValueError("amplitudes must have shape (K,) or (K,C)")
        if phases.shape != (wavevectors.shape[0],):
            raise ValueError("phases must have shape (K,)")
        object.__setattr__(self, "wavevectors", wavevectors)
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "offset", np.asarray(self.offset, dtype=np.float64))

    @property
    def channels(self) -> int:
        return 1 if self.amplitudes.ndim == 1 else int(self.amplitudes.shape[1])

    def evaluate(self, x: Array, y: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        phase = 2.0 * np.pi * (
            self.wavevectors[:, 0, None] * x.reshape(1, -1)
            + self.wavevectors[:, 1, None] * y.reshape(1, -1)
        ) + self.phases[:, None]
        carrier = np.cos(phase)
        if self.amplitudes.ndim == 1:
            value = np.asarray(self.offset) + np.sum(
                self.amplitudes[:, None] * carrier, axis=0
            )
            return value.reshape(x.shape)
        value = np.asarray(self.offset) + np.einsum(
            "kc,kn->nc", self.amplitudes, carrier, optimize=True
        )
        return value.reshape(x.shape + (self.channels,))

    def sample_nodes(self, shape: tuple[int, int]) -> Array:
        y = np.linspace(0.0, 1.0, int(shape[0]))
        x = np.linspace(0.0, 1.0, int(shape[1]))
        yy, xx = np.meshgrid(y, x, indexing="ij")
        return self.evaluate(xx, yy)

    def warped_nodes(self, transform: ProjectiveMap, shape: tuple[int, int]) -> Array:
        y = np.linspace(0.0, 1.0, int(shape[0]))
        x = np.linspace(0.0, 1.0, int(shape[1]))
        yy, xx = np.meshgrid(y, x, indexing="ij")
        px, py = transform.map(xx, yy)
        return self.evaluate(px, py)

    def affine_basin_average(
        self,
        transform: ProjectiveMap,
        shape: tuple[int, int],
    ) -> Array:
        """Return the analytic average over every endpoint Voronoi basin."""

        linear, offset = transform.affine_parts()
        y_bounds = endpoint_basin_bounds(int(shape[0]))
        x_bounds = endpoint_basin_bounds(int(shape[1]))
        x_mid = 0.5 * (x_bounds[:-1] + x_bounds[1:])
        y_mid = 0.5 * (y_bounds[:-1] + y_bounds[1:])
        x_width = np.diff(x_bounds)
        y_width = np.diff(y_bounds)
        yy, xx = np.meshgrid(y_mid, x_mid, indexing="ij")
        wy, wx = np.meshgrid(y_width, x_width, indexing="ij")
        px = linear[0, 0] * xx + linear[0, 1] * yy + offset[0]
        py = linear[1, 0] * xx + linear[1, 1] * yy + offset[1]
        transformed_wave = self.wavevectors @ linear
        phase = 2.0 * np.pi * (
            self.wavevectors[:, 0, None] * px.reshape(1, -1)
            + self.wavevectors[:, 1, None] * py.reshape(1, -1)
        ) + self.phases[:, None]
        attenuation = (
            np.sinc(transformed_wave[:, 0, None] * wx.reshape(1, -1))
            * np.sinc(transformed_wave[:, 1, None] * wy.reshape(1, -1))
        )
        carrier = np.cos(phase) * attenuation
        if self.amplitudes.ndim == 1:
            value = np.asarray(self.offset) + np.sum(
                self.amplitudes[:, None] * carrier, axis=0
            )
            return value.reshape(tuple(map(int, shape)))
        value = np.asarray(self.offset) + np.einsum(
            "kc,kn->nc", self.amplitudes, carrier, optimize=True
        )
        return value.reshape(tuple(map(int, shape)) + (self.channels,))


def canonical_fourier_fields() -> dict[str, FourierField]:
    """Return deterministic directional and crossed-phase warp probes."""

    return {
        "oblique": FourierField(
            wavevectors=np.array(((5.0, 2.0), (9.0, 3.0), (2.0, -4.0))),
            amplitudes=np.array((0.22, 0.09, 0.07)),
            phases=np.array((0.13, 1.17, -0.41)),
        ),
        "crossed": FourierField(
            wavevectors=np.array(((6.0, 1.0), (-1.0, 6.0), (4.0, 4.0))),
            amplitudes=np.array((0.17, 0.17, 0.09)),
            phases=np.array((0.2, -0.6, 1.1)),
        ),
        "multichannel": FourierField(
            wavevectors=np.array(((4.0, 1.0), (1.0, 5.0), (5.0, -3.0))),
            amplitudes=np.array(((0.22, 0.05, 0.08),
                                 (0.04, 0.20, 0.07),
                                 (0.06, 0.04, 0.18))),
            phases=np.array((0.0, 0.7, -0.4)),
            offset=np.array((0.5, 0.5, 0.5)),
        ),
    }


def _polygon_halfplanes(vertices: Array) -> Array:
    polygon = np.asarray(vertices, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError("a polygon requires at least three planar vertices")
    x, y = polygon[:,0], polygon[:,1]
    signed_area = 0.5*np.sum(x*np.roll(y,-1)-y*np.roll(x,-1))
    if signed_area <= 0.0:
        raise ValueError("convex polygon vertices must be counterclockwise")
    edge = np.roll(polygon,-1,axis=0)-polygon
    # cross(edge,p-vertex) >= 0.
    return np.stack((
        -edge[:,1], edge[:,0],
        edge[:,1]*polygon[:,0]-edge[:,0]*polygon[:,1],
    ),axis=1)


def _clip_halfplane(polygon: Array, line: Array) -> Array:
    if polygon.shape[0] == 0:
        return polygon
    output=[]
    tolerance=128.0*np.finfo(np.float64).eps*max(1.0,float(np.max(np.abs(line))))
    for first,second in zip(polygon,np.roll(polygon,-1,axis=0)):
        fa=float(line[0]*first[0]+line[1]*first[1]+line[2])
        fb=float(line[0]*second[0]+line[1]*second[1]+line[2])
        inside_a=fa>=-tolerance; inside_b=fb>=-tolerance
        if inside_a:
            output.append(first)
        if inside_a != inside_b:
            denominator=fa-fb
            if abs(denominator)>tolerance:
                output.append(first+(fa/denominator)*(second-first))
    return np.asarray(output,dtype=np.float64).reshape(-1,2)


def _polygon_area(polygon: Array) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    return abs(0.5*float(np.sum(
        polygon[:,0]*np.roll(polygon[:,1],-1)
        -polygon[:,1]*np.roll(polygon[:,0],-1)
    )))


@dataclass(frozen=True)
class ConvexPolygonField:
    """A linear combination of convex indicators with exact projective coverage."""

    polygons: tuple[Array,...]
    amplitudes: Array
    offset: float=0.0

    def __post_init__(self) -> None:
        polygons=tuple(np.asarray(p,dtype=np.float64) for p in self.polygons)
        halfplanes=tuple(_polygon_halfplanes(p) for p in polygons)
        amplitudes=np.asarray(self.amplitudes,dtype=np.float64)
        if amplitudes.shape != (len(polygons),):
            raise ValueError("one amplitude is required per polygon")
        object.__setattr__(self,"polygons",polygons)
        object.__setattr__(self,"amplitudes",amplitudes)
        object.__setattr__(self,"_halfplanes",halfplanes)

    def evaluate(self,x: Array,y: Array) -> Array:
        x,y=np.broadcast_arrays(np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64))
        result=np.full(x.shape,float(self.offset),dtype=np.float64)
        for amplitude,lines in zip(self.amplitudes,self._halfplanes):
            inside=np.ones(x.shape,dtype=bool)
            for a,b,c in lines:
                inside &= a*x+b*y+c >= 0.0
            result += amplitude*inside
        return result

    def sample_nodes(self,shape: tuple[int,int]) -> Array:
        y=np.linspace(0.0,1.0,int(shape[0]));x=np.linspace(0.0,1.0,int(shape[1]))
        yy,xx=np.meshgrid(y,x,indexing="ij")
        return self.evaluate(xx,yy)

    def warped_nodes(self,transform: ProjectiveMap,shape: tuple[int,int]) -> Array:
        y=np.linspace(0.0,1.0,int(shape[0]));x=np.linspace(0.0,1.0,int(shape[1]))
        yy,xx=np.meshgrid(y,x,indexing="ij");px,py=transform.map(xx,yy)
        return self.evaluate(px,py)

    def warped_basin_average(
        self,transform: ProjectiveMap,shape: tuple[int,int]
    ) -> Array:
        """Exact polygon coverage of every target basin under a homography."""

        h=transform.matrix
        # The substitution of a source half-plane through a projective map is
        # linear only after fixing the denominator sign.  The domain corners
        # suffice because the denominator is affine.
        corners=np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float64)
        denominator=corners@h[2,:2]+h[2,2]
        if np.min(denominator)<=0.0:
            raise ValueError("projective coverage domain crosses or reverses the pole")
        transformed=[]
        for lines in self._halfplanes:
            transformed.append(np.stack([
                a*h[0]+b*h[1]+c*h[2] for a,b,c in lines
            ]))
        xb=endpoint_basin_bounds(int(shape[1]));yb=endpoint_basin_bounds(int(shape[0]))
        output=np.full(tuple(map(int,shape)),float(self.offset),dtype=np.float64)
        for row in range(int(shape[0])):
            for column in range(int(shape[1])):
                rectangle=np.array((
                    (xb[column],yb[row]),(xb[column+1],yb[row]),
                    (xb[column+1],yb[row+1]),(xb[column],yb[row+1]),
                ),dtype=np.float64)
                basin_area=(xb[column+1]-xb[column])*(yb[row+1]-yb[row])
                for amplitude,lines in zip(self.amplitudes,transformed):
                    clipped=rectangle
                    for line in lines:
                        clipped=_clip_halfplane(clipped,line)
                        if not clipped.size:
                            break
                    output[row,column] += amplitude*_polygon_area(clipped)/basin_area
        return output


@dataclass(frozen=True)
class HalfPlaneField:
    """One exact jump across a source-space projective line."""

    line: Array
    low: float=0.0
    high: float=1.0

    def __post_init__(self)->None:
        line=np.asarray(self.line,dtype=np.float64)
        if line.shape!=(3,) or np.linalg.norm(line[:2])==0.0:
            raise ValueError("half-plane line must be (a,b,c), (a,b) nonzero")
        object.__setattr__(self,"line",line/np.linalg.norm(line[:2]))

    def target_line(self,transform: ProjectiveMap)->Array:
        a,b,c=self.line
        return a*transform.matrix[0]+b*transform.matrix[1]+c*transform.matrix[2]

    def evaluate(self,x: Array,y: Array)->Array:
        x,y=np.broadcast_arrays(np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64))
        return np.where(self.line[0]*x+self.line[1]*y+self.line[2]>=0.0,
                        float(self.high),float(self.low))

    def sample_nodes(self,shape: tuple[int,int])->Array:
        y=np.linspace(0.0,1.0,int(shape[0]));x=np.linspace(0.0,1.0,int(shape[1]))
        yy,xx=np.meshgrid(y,x,indexing="ij");return self.evaluate(xx,yy)

    def warped_nodes(self,transform: ProjectiveMap,shape: tuple[int,int])->Array:
        y=np.linspace(0.0,1.0,int(shape[0]));x=np.linspace(0.0,1.0,int(shape[1]))
        yy,xx=np.meshgrid(y,x,indexing="ij");px,py=transform.map(xx,yy)
        return self.evaluate(px,py)

    def warped_basin_average(self,transform: ProjectiveMap,shape: tuple[int,int])->Array:
        h=transform.matrix
        corners=np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float64)
        denominator=corners@h[2,:2]+h[2,2]
        if np.min(denominator)<=0.0:
            raise ValueError("projective coverage domain crosses or reverses the pole")
        line=self.target_line(transform)
        xb=endpoint_basin_bounds(int(shape[1]));yb=endpoint_basin_bounds(int(shape[0]))
        output=np.empty(tuple(map(int,shape)),dtype=np.float64)
        for row in range(int(shape[0])):
            for column in range(int(shape[1])):
                rectangle=np.array(((xb[column],yb[row]),(xb[column+1],yb[row]),
                                    (xb[column+1],yb[row+1]),(xb[column],yb[row+1])))
                clipped=_clip_halfplane(rectangle,line)
                area=(xb[column+1]-xb[column])*(yb[row+1]-yb[row])
                fraction=_polygon_area(clipped)/area
                output[row,column]=self.low+(self.high-self.low)*fraction
        return output


def _oriented_rectangle(
    centre: tuple[float,float],length: float,width: float,angle_degrees: float
) -> Array:
    angle=np.deg2rad(angle_degrees)
    tangent=np.array((np.cos(angle),np.sin(angle)))
    normal=np.array((-np.sin(angle),np.cos(angle)))
    c=np.asarray(centre,dtype=np.float64)
    # Counterclockwise order.
    return np.stack((
        c-0.5*length*tangent-0.5*width*normal,
        c+0.5*length*tangent-0.5*width*normal,
        c+0.5*length*tangent+0.5*width*normal,
        c-0.5*length*tangent+0.5*width*normal,
    ))


def canonical_polygon_fields() -> dict[str,ConvexPolygonField]:
    return {
        "thin_oblique_bar":ConvexPolygonField(
            polygons=(_oriented_rectangle((0.5,0.5),1.4,0.065,27.0),),
            amplitudes=np.array((1.0,)),offset=0.0,
        ),
        "crossing_bars":ConvexPolygonField(
            polygons=(
                _oriented_rectangle((0.5,0.5),1.2,0.055,31.0),
                _oriented_rectangle((0.5,0.5),1.2,0.055,121.0),
            ), amplitudes=np.array((0.5,0.5)),offset=0.0,
        ),
        "rotated_box":ConvexPolygonField(
            polygons=(_oriented_rectangle((0.5,0.5),0.62,0.34,-19.0),),
            amplitudes=np.array((1.0,)),offset=0.0,
        ),
    }


def half_plane_at_angle(angle_degrees: float,offset: float=0.0)->HalfPlaneField:
    angle=np.deg2rad(float(angle_degrees))
    normal=np.array((np.cos(angle),np.sin(angle)),dtype=np.float64)
    point=np.array((0.5,0.5),dtype=np.float64)+offset*normal
    return HalfPlaneField(np.array((normal[0],normal[1],-normal@point)))


@dataclass(frozen=True)
class EllipseField:
    """An exact binary ellipse with a homogeneous conic representation."""

    centre: tuple[float,float]=(0.5,0.5)
    radius_x: float=0.29
    radius_y: float=0.21
    angle_degrees: float=0.0
    low: float=0.0
    high: float=1.0

    def source_conic(self)->Array:
        angle=np.deg2rad(float(self.angle_degrees));rotation=np.array(
            ((np.cos(angle),-np.sin(angle)),(np.sin(angle),np.cos(angle)))
        )
        precision=rotation@np.diag((1.0/self.radius_x**2,1.0/self.radius_y**2))@rotation.T
        centre=np.asarray(self.centre,dtype=np.float64)
        conic=np.empty((3,3),dtype=np.float64);conic[:2,:2]=precision
        conic[:2,2]=conic[2,:2]=-precision@centre
        conic[2,2]=float(centre@precision@centre-1.0)
        return conic

    def target_conic(self,transform: ProjectiveMap)->Array:
        return transform.matrix.T@self.source_conic()@transform.matrix

    def source_boundary_point(self,phase: float)->Array:
        angle=np.deg2rad(float(self.angle_degrees));rotation=np.array(
            ((np.cos(angle),-np.sin(angle)),(np.sin(angle),np.cos(angle)))
        )
        local=np.array((self.radius_x*np.cos(phase),self.radius_y*np.sin(phase)))
        return np.asarray(self.centre)+rotation@local

    def target_boundary_frame(
        self,transform: ProjectiveMap,phase: float
    )->tuple[Array,Array,Array]:
        source=np.r_[self.source_boundary_point(phase),1.0]
        target=np.linalg.solve(transform.matrix,source);point=target[:2]/target[2]
        homogeneous=np.r_[point,1.0]
        gradient=2.0*(self.target_conic(transform)@homogeneous)[:2]
        # The conic is negative inside.  The inward normal points against its
        # gradient and therefore orients a normal probe from low to high.
        inward=-gradient/np.linalg.norm(gradient);tangent=np.array((-inward[1],inward[0]))
        return point,inward,tangent

    def evaluate(self,x: Array,y: Array)->Array:
        x,y=np.broadcast_arrays(np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64))
        homogeneous=np.stack((x,y,np.ones_like(x)),axis=-1);conic=self.source_conic()
        implicit=np.einsum("...i,ij,...j->...",homogeneous,conic,homogeneous,optimize=True)
        return np.where(implicit<=0.0,float(self.high),float(self.low))

    def sample_nodes(self,shape: tuple[int,int])->Array:
        y=np.linspace(0.0,1.0,int(shape[0]));x=np.linspace(0.0,1.0,int(shape[1]))
        yy,xx=np.meshgrid(y,x,indexing="ij");return self.evaluate(xx,yy)

    def warped_nodes(self,transform: ProjectiveMap,shape: tuple[int,int])->Array:
        y=np.linspace(0.0,1.0,int(shape[0]));x=np.linspace(0.0,1.0,int(shape[1]))
        yy,xx=np.meshgrid(y,x,indexing="ij");px,py=transform.map(xx,yy)
        return self.evaluate(px,py)
