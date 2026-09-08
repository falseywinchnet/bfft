# Companion Owner Coordinate and Integrable Two-Current Law

This is a formal working derivation.  It does not modify the paper.

## 1. Domain of one owner chart

Let \((C,g)\) be an oriented two-dimensional Riemannian owner and let
\(\Gamma\subset C\) be its transverse zero-line.  Parameterize \(\Gamma\)
by Riemannian arclength,

\[
\gamma:\psi\longmapsto\Gamma,
\qquad |\dot\gamma|_g=1.
\]

For each \(\gamma(\psi)\), emit the unit-speed geodesic normal to \(\Gamma\):

\[
X(\phi,\psi)=\exp_{\gamma(\psi)}(\phi n(\psi)).
\]

This map is a diffeomorphism until either two normal geodesics meet or its
Jacobian vanishes.  Those events are respectively the cut locus and a
conjugate point.  They are chart boundaries, not locations at which two
coordinates should be averaged.

The first coordinate is the signed Eikonal distance

\[
\phi(X(\phi,\psi))=\phi,
\qquad |d\phi|_{g^{-1}}=1.
\]

The companion coordinate is therefore fixed:

\[
\boxed{\psi(X(\phi,q))=q.}
\]

It is the arclength label of the characteristic's footpoint on \(\Gamma\),
carried unchanged along the normal geodesic.  It is not another arrival time,
a fitted phase, or a second direction vote.

The only gauge is orientation.  Set \(\psi=0\) at the owner barycenter and
orient \((n,t)\) by the ambient image orientation.  Reversing both coordinate
signs changes no reconstructed scalar.

## 2. Ray-tube scale is not an optional parameter

Gauss' lemma makes the pullback metric orthogonal:

\[
X^*g=d\phi^2+h(\phi,\psi)^2d\psi^2,
\qquad h(0,\psi)=1.
\]

Here \(h\) is the separation of neighboring normal geodesics.  It is a Jacobi
field magnitude, not a user-selected anisotropy gain.  On a surface with
Gaussian curvature \(K\),

\[
\partial_{\phi\phi}h+Kh=0,
\qquad
\partial_\phi h(0,\psi)=-\kappa_g(\psi),
\]

where \(\kappa_g\) is the geodesic curvature of \(\Gamma\), with sign fixed
by the chosen normal.

Let \(*_g\) denote the Riemannian Hodge star.  Since

\[
*_g d\phi=h\,d\psi,
\]

the companion covector is

\[
\boxed{d\psi=a\,*_g d\phi,\qquad a={1\over h}.}
\]

The integrating factor \(a\) is uniquely determined along the Eikonal flow:

\[
d(a*_g d\phi)=0
\quad\Longleftrightarrow\quad
\operatorname{div}_g(a\operatorname{grad}_g\phi)=0,
\qquad a|_\Gamma=1.
\]

Equivalently,

\[
\partial_\phi a+a\,\Delta_g\phi=0,
\qquad
a(\phi,\psi)=
\exp\!\left[-\int_0^\phi
\Delta_g\phi(X(s,\psi))\,ds\right].
\]

This is a first-order characteristic transport, not an elliptic solve.  The
V3 predecessor forest can propagate it away from the zero-line.  The chart is
valid precisely while \(h>0\), equivalently while \(a\) is finite and the
footpoint is unique.

For a Euclidean circle of radius \(R\),

\[
X(\phi,\psi)=(R+\phi)
\left(\cos{\psi\over R},\sin{\psi\over R}\right),
\]

and therefore

\[
X^*g=d\phi^2+{(R+\phi)^2\over R^2}d\psi^2,
\quad
h={R+\phi\over R},
\quad
a={R\over R+\phi}.
\]

The symbolic certificate verifies the Eikonal identity, orthogonality, and
the density transport equation exactly.

## 3. An exact current has only one free interior component

Write the owner current in coordinate covectors:

\[
\alpha=df=j_\phi\,d\phi+j_\psi\,d\psi.
\]

Integrability is the exterior-derivative identity

\[
\boxed{\partial_\psi j_\phi=\partial_\phi j_\psi.}
\]

The coordinate component \(j_\psi\) is not the unit-tangent derivative.  If
\(j_t\) denotes the derivative along the unit tangent, then

\[
j_t={1\over h}j_\psi=a j_\psi,
\]

and the closure law becomes

\[
\partial_\psi j_\phi=\partial_\phi(hj_t).
\]

Dropping \(h\) would treat converging and diverging ray tubes as though they
had equal transverse mass.

Choose the normal current \(j_\phi\) as the component admitted by ordered
CONV transport, and retain the scalar trace

\[
b(\psi)=f(0,\psi)
\]

on the zero-line.  Then there is exactly one integrable two-current extension:

\[
\boxed{
F(\phi,\psi)=b(\psi)+\int_0^\phi j_\phi(s,\psi)\,ds,
}
\]

\[
\boxed{
j_\psi(\phi,\psi)=b'(\psi)+
\int_0^\phi\partial_\psi j_\phi(s,\psi)\,ds.
}
\]

Existence follows by construction.  For uniqueness, any other potential with
the same trace and normal derivative has zero normal derivative and vanishes
at \(\phi=0\), hence vanishes everywhere on every characteristic.

This produces an important prohibition:

> The two current components may not be admitted or projected independently.

Changing \(j_\psi\) after \(j_\phi\) and \(b\) have been fixed creates a
nonzero curl unless the change is constant along every normal ray.  The
companion current is derived state.

## 4. Exact discrete law

Let \(B_l=F_{0,l}\) be the zero-line trace and let \(U_{k,l}\) be admitted
normal current increments.  Define

\[
F_{k+1,l}=F_{k,l}+U_{k,l},
\qquad F_{0,l}=B_l,
\]

and derive the tangent increments

\[
V_{k,l}=F_{k,l+1}-F_{k,l}.
\]

Then every chart cell obeys the exact commuting-square identity

\[
\boxed{
U_{k,l+1}-U_{k,l}=V_{k+1,l}-V_{k,l}.
}
\]

This is discrete \(d^2F=0\).  It holds over integers or rationals without an
error tolerance and over floating point up to the arithmetic used to form the
shared potential.  No Hodge projection or Poisson solve is required.

For a tensor-product Bernstein realization, the same law says that the
normal and tangent Bezier edge increments must come from one shared control
net.  A common control net also retains nonnegative Bernstein synthesis and
its convex-hull bound.  Separate face projections do not.

## 5. Transport between scales

Let a monotone target-to-source chart map be

\[
R(\widehat\phi,\widehat\psi)
=(A(\widehat\phi),B(\widehat\psi)).
\]

Scalar transport is \(\widehat F=F\circ R\).  Its current is the pullback

\[
\boxed{
\widehat\alpha=R^*\alpha
=A'\,j_\phi(A,B)\,d\widehat\phi
+B'\,j_\psi(A,B)\,d\widehat\psi.
}
\]

Exterior differentiation commutes with pullback, so

\[
d\widehat\alpha=R^*(d\alpha)=0.
\]

Positive \(A'\) and \(B'\) preserve coordinate-current orientation.  Because
the scalar is composed with a coordinate map rather than filtered by signed
value weights, this transport itself introduces neither a new value nor a
circulation current.  Analysis basins use the Riemannian area element

\[
dA_g=h(\phi,\psi)\,d\phi\,d\psi;
\]

omitting \(h\) would fail conservation when the owner rays spread.

## 6. Consequence for the interpolation architecture

The canonical owner stack is now

\[
(C,M,\Gamma)
\longrightarrow
(\phi,\psi,h)
\longrightarrow
(b,j_\phi)
\longrightarrow
F
\longrightarrow
(j_\phi,j_\psi).
\]

The independent stored state is the zero-line scalar trace plus the admitted
normal current.  The companion coordinate, transverse current, ray-tube
density, and scale transport are forced.  A Bruun phase pair, local angle
vote, independently constrained tangent current, or post hoc Hodge solve is
not part of this construction.

The remaining implementation question is rasterization: how the V3 owner
forest samples \((b,j_\phi)\) on its irregular Fermi chart and how adjacent
owner traces meet at the cut locus.  It is no longer a question of what
\(\psi\) or the integrability law should be.

## 7. Direct forest realization

The continuous definition has a noniterative causal realization on the
existing first-arrival forest.

1. Intersect the owner with \(\Gamma\), order its zero-line seeds by
   Riemannian arclength, and assign those seed labels \(\psi\).
2. Run signed first arrival from the complete zero-line.  In addition to
   distance and predecessor, carry the seed footpoint label.  Thus every
   accepted node inherits \(\psi\) from its unique causal predecessor.
3. Propagate the ray-tube state by
   \[
   \partial_\phi\log a=-\Delta_g\phi,
   \qquad a|_\Gamma=1,
   \]
   or equivalently propagate the transverse Jacobi width \(h=1/a\).
4. If two different zero-line footpoints reach the same node with equal
   action, the node lies on a cut locus.  If \(h\) reaches zero, the chart has
   a conjugate point.  In either case terminate the chart and expose a trace;
   do not average the two labels.
5. Admit the sampled normal current along each inherited \(\psi\)-ray, retain
   the admitted zero-line trace, integrate once in \(\phi\), and derive the
   tangent current by commuting differences.

Every operation is an ordered front propagation, a one-dimensional current
admission, or a prefix integration.  No global optimizer is introduced.
