# Eikonal ownership as the missing CONV transport object

This is a working derivation, not paper text and not a finished interpolation
rule.  It records what the V3 segmenter establishes and the remaining
mathematical obligation for a general image resampler.

## 1. What V3 actually contributes

V3 does not extrapolate a locally estimated tangent.  It constructs two
coupled fields from the unchanged image:

\[
 Q(x)\in\operatorname{Sym}^{+}(2),\qquad
 \mu(x)=\frac{\sqrt{\det Q(x)}}{\pi}.
\]

The eigenvectors and eigenvalue ratio of \(Q\) describe local support shape;
\(\mu\) describes how many support germs that shape requires.  Germs compete
under the Riemannian action

\[
 d_M(s,x)=\inf_{\gamma(0)=s,\,\gamma(1)=x}
 \int_0^1\sqrt{\dot\gamma(t)^T M(\gamma(t))\dot\gamma(t)}\,dt,
\]

and the owner and arrival fields are

\[
 o(x)=\arg\min_a d_M(s_a,x),\qquad
 T(x)=\min_a d_M(s_a,x).
\]

The numerical front also returns an achieving predecessor forest.  Thus one
owner is not merely a label.  It is a region, an action coordinate, and a
unique causal transport path away from the cut locus.

For eigenvalues \(\lambda_n\geq\lambda_t>0\), the tensor support ellipse has
normal and tangent semispans

\[
 b=\lambda_n^{-1/2},\qquad a=\lambda_t^{-1/2},
\]

and area \(\pi/\sqrt{\det Q}\).  This explains the population law exactly:
one germ represents one such local area.  A coherent contour has small
\(\lambda_t\), hence a long tangent semispan and low germ density.  The same
eigenstructure makes normal travel expensive and tangential travel cheap in
\(d_M\), so first-arrival competition realizes the long support rather than
merely predicting it.

The curvature correction is equally structural.  If the director curvature
is \(\kappa\), a tangent chord of half-length \(a\) has leading sagitta

\[
 \frac{\kappa a^2}{2}.
\]

The straight support is valid only while this is at most \(b\).  V3 increases
population by

\[
 \rho_\kappa=
 \sqrt{\max\!\left(1,\frac{\kappa a^2}{2b}\right)}.
\]

Consequently a straight feature receives long cells, a curved feature is
subdivided exactly when the straight chart loses validity, and a crossing
raises both eigenvalues and therefore \(\det Q\).  Competing directions create
more, shorter owners rather than being averaged into one false direction.

## 2. Why the local CONV chord failed

The local chord used only \(Q(x)\)'s smallest eigenvector.  Its angular error
was integrated as literal transverse drift.  At shallow angles that drift
grew with chord length, so enlarging the chord worsened the result.

An eikonal owner makes the opposite construction.  Every infinitesimal
direction is charged by the spatially varying quadratic form, and the whole
path minimizes accumulated action.  Front collision, not an arbitrary box,
ends support.  Curvature changes population before it is allowed to make a
single chart invalid.  The needed support is therefore a basin and its causal
forest, not a ray.

## 3. The state that should be transported

Scalar values are not the complete state.  For each owner \(C_a\), retain:

1. the owner restriction \(C_a\), arrival field \(T_a\), and predecessor
   forest \(P_a\);
2. an unoriented wave covector \(k_a\) and a real phase pair
   \(z_a=(c_a,s_a)\);
3. the admitted ordered differential current carried by the existing CONV
   profile.

For a predecessor edge \(p\to x\), phase transport is the real rotation

\[
 z(x)=R(\delta\phi_{p x})z(p),\qquad
 \delta\phi_{p x}=\int_{\gamma_{p x}}k\cdot d\gamma,
\]

where

\[
 R(\theta)=
 \begin{bmatrix}\cos\theta&-\sin\theta\\
                 \sin\theta& \cos\theta\end{bmatrix}.
\]

This is the Bruun pair state already used elsewhere in the repository; no
complex arithmetic is required.  Because the owner predecessor graph is a
forest, the accumulated phase has one path and no cycle-consistency problem.
At a front collision the phases are not averaged.  The collision is the cut
locus between competing transport truths.

The differential current must be pushed forward by a positive, mass-one
operator on each characteristic interval.  If \(\nu\) is the signed current
measure and \(K_{p x}\) is the transport kernel, the required local law is

\[
 \nu_x=K_{p x}\nu_p,\qquad K_{p x}\geq0,\qquad K_{p x}\mathbf1=\mathbf1.
\]

The positive operator transports the positive and negative Jordan parts of
the current without manufacturing an alternating pair.  CONV synthesis then
integrates that transported current.  This is the appropriate place for the
existing ordered-current projection; it should not be replaced by a signed
value kernel.

## 4. Dimensionless metric obligation

The image-derived form and Cartesian sampling scale must be combined before
any square root or eigendecomposition.  If \(H\) maps index increments to
physical increments and \(J\) is the channel-summed gradient second moment,
an amplitude-normalized dimensionless form is

\[
 A=I+\frac{H^T J H}{E},
\]

with the quotient defined as zero where the local variation energy \(E\) is
zero.  Equivalently, the physical metric is

\[
 M=H^{-T}AH^{-1}=H^{-T}H^{-1}+\frac{J}{E}.
\]

This construction is coordinate-correct for anisotropic Cartesian spacing;
it never treats the sampling axes as though they lived in the tensor
eigenframe.  The exact support-scale law and the base population measure still
need to be fixed by the matched analysis/synthesis lattice, not by a tuned
image threshold.

## 5. What can and cannot yet be claimed

The V3 mechanism already establishes, both algebraically and in the focused
audit, that eikonal competition converts local anisotropy into long aligned
regions and terminates those regions at curvature and directional conflict.
It also supplies a deterministic causal forest on which a phase pair can be
transported.

It does **not** by itself prove that a new interpolator is cardinal,
range-preserving, or variation diminishing.  Those properties require one
final definition: the positive pushforward of CONV's admitted current from
source anchors onto the owner forest and the trace rule at owner collisions.
The correct next experiment is therefore not another chord.  It is a frozen
eikonal-owner ablation with the following invariant stack:

\[
 \text{CONV jets}
 \to Q,\mu
 \to (o,T,P)
 \to \text{real phase transport on }P
 \to \text{positive ordered-current pushforward}
 \to \text{unchanged CONV integration}.
\]

Lanczos remains only a destination reference for the recovered basin; it
does not supply a coefficient, phase, support radius, or decision rule.
