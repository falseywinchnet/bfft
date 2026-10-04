# Current excursion and geometric admission in CONV

14 September 2026.

Let a source interval have endpoint values \(y_0,y_1\), physical width
\(h>0\), and admitted quintic currents \(c_0,\ldots,c_4\). Write

\[
\delta=y_1-y_0=\sum_{k=0}^4c_k,\qquad
Q_0=y_0,\quad Q_j=y_0+\sum_{k<j}c_k.
\]

Its potential and physical derivative are

\[
U(t)=\sum_{j=0}^5Q_jB_j^5(t)
=y_0+\sum_{k=0}^4\tau_k(t)c_k,
\qquad
\frac{dU}{dx}=\frac5h\sum_{k=0}^4c_kB_k^4(t),
\quad t\in[0,1],
\]

where \(\tau_k=\sum_{j=k+1}^5B_j^5\). The ordered current ledger fixes the
permitted signs of the currents. The signed-fibre projection fixes their
sum to \(\delta\).

## 1. Excursion follows from opposing current

Define

\[
p=\sum_k(c_k)_+,\qquad n=\sum_k(-c_k)_+,\qquad
D=\min(p,n)=\frac{\|c\|_1-|\delta|}{2}.
\]

The quantity \(D\) measures the current which cancels in the endpoint
difference. It is determined by the admitted field.

**Proposition 1 (endpoint excursion).** Every quintic with these currents
satisfies

\[
\min(y_0,y_1)-\frac{15}{16}D
\ \leq U(t)\leq\
\max(y_0,y_1)+\frac{15}{16}D.
\tag{1}
\]

The constant \(15/16\) is sharp over quintic current vectors.

**Proof.** Suppose first that \(\delta\geq0\), so \(p=\delta+D\) and
\(n=D\). The Bernstein tails obey

\[
\tau_4=t^5\leq\tau_k\leq\tau_0=1-(1-t)^5.
\]

Separating positive and negative currents gives

\[
y_0+p\tau_4-n\tau_0\leq U(t)
\leq y_0+p\tau_0-n\tau_4.
\tag{2}
\]

Consequently

\[
U(t)-y_1\leq D[1-(1-t)^5-t^5]-\delta(1-t)^5,
\]

\[
y_0-U(t)\leq D[1-(1-t)^5-t^5]-\delta t^5.
\]

The bracket attains its maximum \(15/16\) at \(t=1/2\). Replacing
\(U\) by \(-U\) treats \(\delta<0\). Equality in (1) occurs for
\(y_0=y_1\) and \(c=(D,0,0,0,-D)\) at \(t=1/2\). ∎

For \(D>0\), retaining the endpoint difference gives the sharper uniform
bound

\[
\operatorname{exc}(U;[\min(y_0,y_1),\max(y_0,y_1)])
\leq E_5(|\delta|,D),
\]

\[
E_5(a,D)=D-
\frac{D(D+a)}{\bigl((D+a)^{1/4}+D^{1/4}\bigr)^4}
\leq\frac{15}{16}D.
\tag{3}
\]

Indeed, the maximum of
\(D-(D+a)(1-t)^5-Dt^5\) occurs at
\(t=(D+a)^{1/4}/((D+a)^{1/4}+D^{1/4})\). The lower excursion has the
reflected maximizer. Define \(E_5(a,0)=0\). The upper bound in (3) is
attained by \(c=(a+D,0,0,0,-D)\). A prescribed ledger can further restrict
the available current arrangements.

If all currents have one sign, \(D=0\), the derivative is one-signed and
the endpoint interval is preserved. A nonzero excursion therefore requires
opposing current. The flattened ordered ledger independently bounds the
number of derivative sign transitions:

\[
V(U')\leq\operatorname{var}(c^{\flat})
\leq\operatorname{var}(\delta^{\flat}).
\tag{4}
\]

Equations (1)–(4) control amplitude and variation through the same admitted
currents. They do not introduce a tunable excursion allowance.

The original source estimate
\(\|c\|_2\leq C_0h|y|_{1,h}\),
\(C_0=1+2\sqrt{185}/15\), also implies

\[
D\leq\frac{\sqrt5C_0h|y|_{1,h}-|\delta|}{2},\qquad
\|U_x\|_\infty\leq\frac5h\|c\|_\infty.
\tag{5}
\]

Thus the excursion is bounded by source variation even before its tighter
current-dependent value is evaluated. An upper derivative bound alone does
not establish (4).

## 2. Sample bounds and polynomial bounds

Consider the quadratic

\[
f(x)=M-\frac\kappa2(x-x_*)^2,\qquad \kappa>0,
\quad x_*=x_i+h/2.
\]

On a uniform source lattice, its largest samples are
\(M-\kappa h^2/8\). Its derivative changes sign once. Any reconstruction
bounded above by that sampled maximum has error at least
\(\kappa h^2/8\) at \(x_*\). The intersample extremum and the sampled
maximum are different mathematical objects. A support-sample bound can
exclude the former while respecting the latter.

There is a second restriction when a function bound is enforced by bounding
each Bernstein coefficient. For

\[
f(t)=4t(1-t),\qquad
Q=(0,4/5,6/5,6/5,4/5,0),
\]

the function already lies in \([0,1]\). Clipping the two \(6/5\) controls
to 1 changes the polynomial by

\[
\Delta f(t)=-\frac15[B_2^5(t)+B_3^5(t)],
\qquad \Delta f(1/2)=-\frac18.
\tag{6}
\]

Its interval integral changes by \(-1/15\). The source interval and its
polynomial certificate therefore impose separate restrictions.

For a particular univariate polynomial, the exact range consists of extrema
among its endpoints and the real roots of its quartic derivative in
\((0,1)\). This is a finite algebraic characterization. Bernstein
subdivision supplies sufficient enclosures without changing the polynomial;
coefficient clipping changes the polynomial. These operations have different
effects even when they are applied to the same proposed interval.

**Proposition 2 (a common coefficient clamp).** Let \(C\) be the scalar
clamp to one fixed interval containing \(y_0,y_1\). Replacing every
\(Q_j\) by \(C(Q_j)\) preserves endpoint mass and cannot increase the
sign variation of the current coefficients.

**Proof.** The map \(C\) is nondecreasing, so
\(C(Q_{j+1})-C(Q_j)\) has the original difference's sign or is zero.
The endpoint controls remain fixed. Removing zero entries cannot increase
sign variation. ∎

This operation can suppress an admitted extremum while retaining the
variation bound. Absence of new transitions does not require preservation
of their original amplitude. The result does not apply to different clamps
at different controls. For example, increasing controls \(3/5,7/10\)
clamped respectively to \([0,4/5]\) and \([0,1/2]\) become decreasing.
Intersected incident-cell intervals are control-dependent, so a common-clamp
argument does not certify their current signs. This example establishes the
logical limitation of that argument, without asserting that this exact pair
is produced by a particular source stencil.

## 3. What quintic collocation retains

For fixed active faces, each one-dimensional factor value is a quintic
polynomial of phase and a linear combination of source samples. In the
horizontal–vertical order, the second-factor active face is selected from
first-factor values and can change with horizontal phase. On a region where
that face remains fixed, substitution gives a polynomial of bidegree at
most \((5,5)\). Exchanging the axes gives the other order.

Let these two potentials be \(A,B\). With the bilinear order coordinate
\(\beta\), their blend is

\[
T=(1-\beta)A+\beta B=A+\beta(B-A).
\tag{7}
\]

On a region with fixed faces for both orders, (7) has bidegree at most
\((6,6)\). Face changes can require several such regions inside a source
cell. A global inverse-distance order coordinate need not be polynomial.

Let \(\mathcal I_{5,5}\) interpolate at the six nodes
\(0,1/5,\ldots,1\) on each axis. It is exact on
\(\mathbb P_5\otimes\mathbb P_5\). It is not generally exact on the
function class just described. The elementary identity

\[
u^6-\mathcal I_5(u^6)
=\prod_{r=0}^5(u-r/5)
\tag{8}
\]

exhibits the missing degree. Equality at every collocation node does not
imply equality between them. Equation (8) describes the representation
obstruction; it does not claim that \(u^6\) itself is an admitted source
field.

Shared collocation nodes prove cardinality and equality of neighboring
quintic boundary polynomials. They consequently prove a continuous atlas.
They do not transfer the original factor derivative ledger to that atlas.
Such a transfer requires either exact representation of the selected
factor field or a new derivative-sign certificate for its replacement.

There is also a differential contribution from the order coordinate:

\[
\nabla T=(1-\beta)\nabla A+\beta\nabla B
+(B-A)\nabla\beta.
\tag{9}
\]

Although (7) is a convex value blend, (9) is not simply a convex blend of
currents. Membership of both factor gradients in a cone does not by itself
put the final gradient in that cone. The original factorwise variation
theorem applies to each admitted factor operation; it does not remove the
last term of (9).

## 4. Directional cones and critical-point structure

Suppose a cell constraint is \(\nabla U\in K\), for a closed convex cone
\(K\). For every \(d\in K^*\),

\[
d^T\nabla U\geq0.
\tag{10}
\]

This proves monotonicity along paths whose tangent belongs to the dual cone.
It does not encode the ordering of all sign transitions along other paths.
If \(K=\mathbb R^2\), then \(K^*=\{0\}\), and (10) imposes no
directional restriction.

**Proposition 3 (bounded derivatives do not fix critical structure).**
Cardinality, continuous boundary matching, support-range bounds, a
whole-plane current cone, and any finite collection of strictly positive
upper derivative bounds can admit an interior extremum absent from a flat
cell.

**Proof.** On the unit cell take a constant baseline \(b\) and put

\[
U_\varepsilon(u,v)=b+
\varepsilon u^2(1-u)^2v^2(1-v)^2,\qquad\varepsilon>0.
\tag{11}
\]

The perturbation and its first normal derivative vanish on every edge. It
preserves the vertices and can be inserted without changing adjacent
patches or their existing traces. It has a strict interior maximum at
\((1/2,1/2)\), of height \(\varepsilon/256\).

The degree-five Bernstein coefficients of \(u^2(1-u)^2\) are
\((0,0,1/10,1/10,0,0)\). Thus the only nonzero tensor perturbation
controls equal \(\varepsilon/100\) and are interior controls. Every
support interval \([b,b+a]\), \(a>0\), accepts these controls when
\(0<\varepsilon\leq100a\). All derivatives of each fixed order scale
linearly with \(\varepsilon\); any finite list of strictly positive
upper bounds therefore admits a sufficiently small positive value. A
whole-plane cone accepts every gradient. ∎

This support situation is realizable: a flat cell can have a raised source
sample elsewhere in its declared support. The incident bilinear gradients
around an isolated raised sample include both signs of both coordinate
directions, generating the whole plane. The proposition describes what the
constraints permit, not a claim that a particular admission algorithm must
select (11).

The ordered ledger excludes newly unrecorded turns by sign order. An upper
bound on derivative magnitude excludes large derivatives. A cone excludes
certain directions. These are separate conditions; none of the latter two
recovers a missing sign-order condition in (11).

## 5. Geometry under a nonsingular warp

Let \(F\) be a smooth local diffeomorphism and \(V=U\circ F\). Then

\[
\nabla_qV=J_F^T\nabla_pU.
\tag{12}
\]

Invertibility of \(J_F\) implies that interior critical points correspond
exactly. At a critical point the terms containing \(\nabla U\) in the
second derivative vanish, giving

\[
H_qV=J_F^TH_pU J_F.
\tag{13}
\]

The Hessians have the same inertia. Nondegenerate maxima, minima and saddles
therefore retain their type under the coordinate map. A source extremum
does not need to be suppressed to make perspective transport well defined.

For a target path \(\gamma\),

\[
\frac{d}{ds}V(\gamma(s))
=(J_F\gamma')^T\nabla U.
\tag{14}
\]

Equation (10) applies when the transported tangent lies in the source dual
cone. This condition addresses line restrictions of the warped field.
It is distinct from the invariant critical-point statement (13).

The same identities describe a class of source-geometry modifications.
Let \(H:\Omega\to\Omega\) be a smooth diffeomorphism onto the source
domain which fixes every source node, and define \(\widetilde U=U\circ H\).
Then \(\widetilde U\) is cardinal wherever \(U\) is cardinal, has exactly
the same value range, and transports its smooth interior critical points
with their type. On an interval, an increasing \(H\) preserves the ordered
derivative sign transitions exactly. If

\[
0<m\leq H'(x)\leq L,
\]

then \(\widetilde U'=U'(H)H'\): local steepening is admitted up to the
factor \(L\), without admitting an extra derivative turn. In two dimensions
the corresponding upper bound is
\(\|\nabla\widetilde U\|\leq\|J_H\|\,\|\nabla U(H)\|\).

This construction shows that steepness and preservation of critical
structure can coexist through geometry. It preserves the original range,
so it cannot increase the original extrema's amplitudes. Moreover,
\(\int U\circ H=\int U(p)|\det J_{H^{-1}}(p)|\,dp\), which is not
generally the original integral. Exact preservation of moments or of a
quintic representation therefore imposes additional conditions. A polynomial
coordinate composition generally raises degree. Selecting \(H\) from source
data is a further construction; the identities establish the properties
of the class without selecting one member.

## 6. Pixel integration acts on the constructed potential

For a target pixel \(P\), define the area functional

\[
\mathcal A_{F,P}[U]=\frac1{|P|}\int_P U(F(q))\,dq.
\tag{15}
\]

For fixed geometry it is linear, positive and normalized on a fully covered
pixel. If \(\widehat U=U+E\), then

\[
\mathcal A_{F,P}[\widehat U]-\mathcal A_{F,P}[U]
=\mathcal A_{F,P}[E],\qquad
|\mathcal A_{F,P}[E]|\leq\|E\|_{L^\infty(F(P))}.
\tag{16}
\]

For partial coverage, zero extension gives the same norm bound and includes
zero in the range enclosure. Integration cannot restore source information
removed before (15). If the source error is one-signed over a footprint of
positive measure, its area error has that sign as well.

On a complete unwarped source cell of area \(|C|\), tensor-quintic
control changes satisfy the exact moment identity

\[
\int_C(\widehat U-U)
=\frac{|C|}{36}\sum_{i,j=0}^5(\widehat P_{ij}-P_{ij}).
\tag{17}
\]

Consequently range clipping can change both an extremum and its integrated
contribution. Positive averaging preserves value intervals and monotonicity
for a common translated positive averaging kernel acting on a monotone
one-dimensional field. Positivity alone does not establish a general
variation-diminishing or critical-point preservation theorem for warped
pixel arrays.

## 7. Consequences for a finite construction

The factor construction already supplies an algebraic excursion law through
\(D\), an absolute source bound through (5), and a variation law through
the ordered ledger. Tightening the sampled amplitude interval is not a
consequence of those laws. It adds a restriction which can remove admitted
extrema and alter their moments.

An exact compilation of the bilinearly blended factor potential must retain
its active-state regions and degree at most \((6,6)\) on each such region.
Algebraic substitution and Bernstein degree elevation preserve that
potential without refitting it. Replacing its representation by one quintic
per source cell requires a separate approximation and variation argument.
Retaining degree six alone is insufficient when an active face changes
inside the cell. The phase boundaries and their integration domains also
belong to the representation.

For a joint field with additional directional requirements, admission must
certify the gradient of the complete potential, including (9). A theorem
about its critical structure must also specify which source currents or
critical events it transports. Equations (10) and (11) identify exactly why
a direction cone and derivative ceiling cannot supply that theorem by
themselves.

Once the source potential has been specified and its required properties
proved, point synthesis and pixel integration are two functionals of that
same field. Equations (12)–(17) then describe their geometric and amplitude
behavior without an empirical sharpness rule.

## Sources within the construction

- [Original CONV paper](../../../output/pdf/eikonal_interpolation.tex):
  signed-fibre synthesis, factorwise variation, and absolute stability.
- [Joint warp appendix](../../../output/pdf/conv_warp_appendix.tex):
  fixed collocation, incident support intervals, gradient cones and admission.
- [CONV* warp addendum](../../../output/pdf/convstar_warp_addendum.tex):
  face-indexed factor banks, phase-dependent polynomial degree and source-state
  factorization.
- [Joint reference](../joint_reference.py):
  `canonical_sampled_control_net` constructs the fifth-step interpolant.

Equations (1)–(3), (6), (8), and (11) are direct finite-dimensional
derivations. No image census, fitted threshold, or numerical optimization is
used in their conclusions.
