# Boundary-motive screen: the m=2 fivefold cluster

This is the cheapest stable-reduction test suggested by the mixed-boundary
route.  It resolves the three zero branches and two pole branches hidden at
`t=0` in the symmetric `m=2` inverse slice.

For the perturbation `(a,b,c)=(0,epsilon,1)`, the first source coordinate is

\[
x_\epsilon(t)=
\frac{4t(t^2+\epsilon)}
{4t^4-(4\epsilon+3)t^2-\epsilon}.
\]

The three zero sections are

\[
t=0,\qquad t^2=-\epsilon,
\]

and the two small pole sections satisfy

\[
t^2=-\frac{\epsilon}{3}+O(\epsilon^2).
\]

Thus the markings are not sections before a quadratic base change.  Put

\[
\epsilon=u^2,\qquad t=uT.
\]

On the exceptional component their limiting positions are

\[
T=0,\quad T=\pm i,\quad T=\pm\frac{i}{\sqrt3}.
\]

All five are separated by this one blowup.  The stable central fiber has two
components and one edge:

- the bubble carries the five resolved markings and its node;
- the main component carries the zero at infinity, the two distant poles at
  `+/-sqrt(3)/2`, and its node.

If `w=1/T` is the bubble coordinate at its node, the chart relation is

\[
t w=u.
\]

Consequently the plumbing coordinate is `q=u`.  Its order is one on the
minimal semistable base, or `1/2` if recorded against the original parameter
`epsilon`.  Five colliding labels therefore do **not** produce order-five
plumbing and do **not** produce a four-edge chain.

There is a tempting integer three elsewhere.  On the unperturbed C-axis,

\[
c(t)=\frac{4t^2-1}{2t^3},\qquad
v=\frac1c=\frac{2t^3}{4t^2-1}=-2t^3+O(t^5).
\]

This is cubic ramification of a map from a fixed normalized curve to target
space.  It is not a node-smoothing parameter.  It yields
`log(v)=3 log(t)+constant+...`, but no extra edge and no stable-reduction
exponent gain.

The result is a clean negative screen for the visible `m=2` cluster.  A
depth-four quotient of the open-curve fundamental group can still contain the
formal tower `1, log(q), ..., log(q)^4/4!`, but it repeats the **same** single
logarithm.  The collision multiplicity has not supplied four independent
plumbing directions or a larger nome exponent.

## The known m=3 two-node boundary

The projective degeneration already derived in `experiments/genus2_collision`
provides a second cheap test:

\[
y^2=f_z(x),\qquad
f_z=(1-z)x^5+5x^2-\frac{15}{4}x+\frac92.
\]

At `z=0`,

\[
f_0=\frac{x+2}{4}(2x^2-2x+3)^2,
\]

so there are two nodes.  If `r` is either root of
`2r^2-2r+3=0` and `x=r+X`, the local equation begins

\[
y^2=A(r)X^2+B(r)z+\cdots,
\]

where, modulo the node equation,

\[
A=-5(x+2),\qquad B=\frac{5x-12}{4}.
\]

The resultants of the node polynomial with `A` and `B` are respectively
`375` and `243/16`, so neither coefficient vanishes at either node.  Each
plumbing coordinate is therefore a unit times `z`: the orders are `(1,1)`.
The dual graph has one genus-zero vertex and two loop edges, not a four-edge
chain.

The factor `z^2` in the discriminant is thus the contribution of two
transverse nodes, not an order-two plumbing parameter.

## Interpretation

The integer `ord_s(q_e)` is not invariant under an arbitrary base change
`s -> s^k`; it can be inflated at will.  Comparisons across `m` must use a
canonically primitive base parameter, or equivalently report thickness after
minimal semistable saturation.  On that normalization the two available
collision tests both have thickness one.

What can still survive at higher `m` is a **nested cluster tree**: groups of
marked places would need to collide at genuinely different valuations, so
successive blowups create several edges.  Merely increasing the number of
branches in one equal-rate cluster cannot produce the desired exponent gain.

## Exact m=4 and m=5 Newton screen

The inverse-coordinate polynomial `H_m(A,B,C,x)` gives an exact, inexpensive
screen at the natural projective line through the collision point

\[
(A,B,C)=(1,1/r,1/r),\qquad r\longrightarrow0.
\]

Put `x=rU` and clear the common pole by defining

\[
\mathcal H_m(r,U)=rH_m(1,1/r,1/r,rU).
\]

For `m=4`, its central polynomial is

\[
\mathcal H_4(0,U)=
-(U+1)^4(U^3-4U^2+10U-20).
\]

Centering with `V=U+1`, the lower Newton hull is

\[
(0,2)\longrightarrow(4,0)\longrightarrow(7,0).
\]

Thus four branches form one cluster of depth `1/2`.  With `r=u^2` and
`V=uW`, its face polynomial is

\[
35(W^4-35),
\]

which is squarefree.  One blowup separates the entire cluster, and its sole
node has `q=u`.

For `m=5`,

\[
\mathcal H_5(0,U)=
-(U+1)^5(U^4-5U^3+15U^2-35U+70),
\]

and the lower hull is

\[
(0,2)\longrightarrow(5,0)\longrightarrow(9,0).
\]

The five-branch cluster has depth `2/5`.  After `r=u^5`, put `V=u^2W`;
the face polynomial is

\[
-126(W^5+126),
\]

again squarefree.  The stable model has one edge with plumbing
`q=u^2`.  Resolving its thickness-two total-space singularity expands it to
two unit edges, still far short of a four-edge chain and still only one
cluster level.

The observed pattern is depth `2/m`, not growing depth.  After the minimal
root-separating base change its thickness is

\[
\frac{2}{\gcd(m,2)},
\]

so this alternates between one and two rather than increasing with `m`.
This formula is proved here only for the explicitly checked `m=4,5` cases;
the lower `m` outputs exhibit the same Newton-hull pattern.

This screen concerns the canonical projective collision-line branch divisor.
It does not logically exclude a specially tuned nonlinear arc, but such an
arc must exhibit a second Newton segment after primitive normalization.  A
mere high-order reparameterization does not count.

## General theorem for every m

The `2/m` pattern is not merely experimental.  It follows directly from the
resultant defining the inverse-coordinate polynomial.

Let `H_m(A,B,C,x)` be the degree `2m-1` inverse polynomial and set

\[
\mathcal H_m(r,U)=rH_m(1,1/r,1/r,rU).
\]

In the defining resultant, after the same substitution, the quadratic
equation for the eliminated coordinate becomes

\[
t^2-(1+U)t+r^2U^2=0.
\]

Call its roots `alpha,beta`.  Then

\[
\alpha+\beta=1+U,
\qquad
\alpha\beta=r^2U^2.
\]

For completeness, the second polynomial inside the resultant becomes

\[
\frac1r E_2(t),
\qquad
E_2(t)=t^{2m-1}-r^2U,t^{m-1}
R_m\!\left(t,\frac{r^2U^2}{t}\right).
\]

The defining chart factor for `H_m` is `x^(2m-1)`.  Since scaling the
second polynomial by `1/r` scales its resultant with the quadratic first
polynomial by `1/r^2`, one obtains

\[
\mathcal H_m(r,U)
=\frac{E_2(\alpha)E_2(\beta)}
{r^{2m}U^{2m-1}}.
\]

At the two roots, `r^2U^2/alpha=beta` and conversely, so

\[
E_2(\alpha)=\alpha^{m-1}
\left(\alpha^m-r^2U R_m(\alpha,\beta)\right),
\]

with the conjugate formula for `beta`.  Expanding their product gives (1).

If `R_m(t,p)` is Aldred's polynomial, direct evaluation of the resultant at
the two roots gives the exact identity

\[
\begin{aligned}
\mathcal H_m(r,U)
={}&r^{2m-2}U^{2m-1}\\
&-\alpha^mR_m(\beta,\alpha)
 -\beta^mR_m(\alpha,\beta)\\
&+r^2U R_m(\alpha,\beta)R_m(\beta,\alpha).
\end{aligned}
\tag{1}
\]

The right side is symmetric in `alpha,beta`.  It is consequently a
polynomial in

\[
\alpha+\beta=1+U,
\qquad
\alpha\beta=r^2U^2,
\]

and therefore contains only even powers of `r`.

Put

\[
V=1+U,
\qquad
A_m(V)=R_m(0,V).
\]

At `r=0`, the two roots specialize to `0,V`, and (1) becomes

\[
\boxed{\mathcal H_m(0,V-1)=-V^mA_m(V).}
\tag{2}
\]

The constant of `A_m` comes from the single `(i,j)=(0,0)` term in the
definition of `R_m`:

\[
a_m=A_m(0)
=(-1)^{m-1}\binom{2m-1}{m-1}\ne0.
\tag{3}
\]

Equation (2) proves that exactly `m` branches meet at `V=0`, while the other
`m-1` branches remain outside that cluster.  Since the full expression is a
polynomial in `r^2`, every coefficient below `V^m` has `r`-valuation at least
two.

At `V=0`, the last term of (1) gives

\[
\mathcal H_m(r,-1)=-a_m^2r^2+O(r^4)
\qquad(m\ge3),
\tag{4}
\]

whereas the coefficient of `V^m` at `r=0` is `-a_m`.  Equations (2)--(4)
force the complete lower Newton hull

\[
\boxed{
(0,2)\longrightarrow(m,0)\longrightarrow(2m-1,0).
}
\tag{5}
\]

The first edge has slope `-2/m`, proving that the unique nontrivial cluster
has depth

\[
\boxed{d_m=\frac2m.}
\]

Let `g=gcd(m,2)`.  The minimal extension making the clustered roots into
sections is

\[
r=u^{m/g},
\qquad
V=u^{2/g}W.
\]

Dividing by the common power of `u`, the Newton face is

\[
\boxed{-a_m(W^m+a_m).}
\tag{6}
\]

It is squarefree in characteristic zero because `a_m` is nonzero.  Thus no
second cluster level occurs.  The bubble and main charts meet with

\[
q=u^{2/g},
\]

so the stable thickness is

\[
\boxed{
r_m=\frac2{\gcd(m,2)}
=\begin{cases}
1,&m\text{ even},\\
2,&m\text{ odd}.
\end{cases}}
\tag{7}
\]

This proves for every `m>=3` that the natural projective collision line has
one cluster level, one stable edge, and at most two unit edges in its regular
resolution.  Increasing `m` increases cluster cardinality while decreasing
its unsaturated depth; it never produces the proposed growing plumbing
hierarchy.

The theorem closes the natural higher-`m` boundary-acceleration route.  It
does not quantify over arbitrary specially tuned nonlinear arcs, which are
different one-parameter families rather than further members of this natural
line.

## Reverse search for tuned arcs

The remaining loophole can be searched in reverse.  Start with a tropical
target arc

\[
A=a r^p,\qquad B=b r^q,\qquad C=c r^s,
\]

construct every lower Newton face of `H_m`, and retain faces with a repeated
nonzero root.  A repeated face is exactly the condition needed for another
cluster level after the first rescaling.

For `m=4,5`, enumerating `-3<=p,q,s<=3` produces ten distinct generic faces
in each case.  Only the already-known universal collision face is generically
repeated.  Testing every nonzero integer leading ratio `a,b,c` of height at
most three produces no additional repetition.  Extending the valuation box
through six leaves the same ten generic faces.

The symbolic face discriminants nevertheless reveal algebraic tunings.  The
simplest occurs for `m=4` on the primitive valuation ray

\[
(p,q,s)=(-1,-1,1).
\]

One outer face has factors

\[
Q(X)=X^3-4X^2+10X-20,
\]

\[
G_c(X)=(c+35)(X+1)^4-35.
\]

Their resultant is

\[
1715\left(
875c^3+92176c^2+3236800c+37888000
\right).
\]

Thus a cubic algebraic choice of `c` makes `Q` and `G_c` share a root and
creates a genuine second cluster level.  If `rho` is the shared root, then

\[
Q(\rho)=0,
\qquad
c=\frac{35}{(\rho+1)^4}-35.
\]

The first nonlinear jet can be tuned once more.  Writing the relevant jet
combination as `q`, elimination over `Q(rho)=0` gives

\[
\boxed{12950q^3-14343q^2+5292q-652=0.}
\]

This makes the inner quadratic face repeat and therefore forces a third
level.  Further levels can be manufactured by continuing to impose contact
with the discriminant.  That observation is double-edged: reverse design can
indeed create arbitrarily high contact, but after the first exceptional
intersection the subsequent depth is chosen jet by jet rather than forced by
the collision family.

There is also a striking rational singular point of the `m=4` discriminant:

\[
(A,B,C)=\left(-\frac72,1,-\frac{160}{7}\right),
\]

where

\[
H_4(x)=\frac{100}{7}
(9x^2-3x+2)^2
(27x^3+18x^2-3x-8).
\]

It gives two simultaneous nodes.  Every target direction is tangent to their
smoothing at first order; along the `A` direction both plumbing parameters
start at order two.  The edges are parallel rather than nested.

Normalizing the two nodes leaves the elliptic curve

\[
y^2=27x^3+18x^2-3x-8,
\qquad j=-\frac{224}{3}.
\]

Since this rational `j` is not integral, the elliptic normalization is not CM.
It therefore fails the arithmetic screen for a natural pi-period evaluation,
despite being the best low-height geometric exception found by the reverse
search.

The reverse search changes the final verdict slightly:

- nested collision scales can be engineered;
- the first extra level already costs cubic algebraic data;
- deeper levels are discriminant-contact conditions, not an intrinsic
  hierarchy growing with `m`;
- the clean rational exceptional fiber is non-CM.

So the loophole exists geometrically, but the first candidates do not yet
restore a credible pi engine.

## Reverse CM-normalization search

The next reverse filter starts from the arithmetic object wanted at the end,
not from another target arc.  A maximally nodal `m=5` inverse fiber would have

\[
H_5(A,B,C,x)=KQ_3(x)^2E_3(x),
\]

and normalizing its three nodes would leave the elliptic curve
`y^2=E_3(x)`.  This is the first place where a new CM elliptic period might
have appeared naturally.

Cancelling the absent powers `x^8,x^7,x^6,x^5` gives, on the chart where the
quadratic coefficient of `Q` is one,

\[
Q=x^3+x^2+tx+\frac{-5+12t-3t^2}{6},
\]

\[
E=x^3-2x^2+(3-2t)x
-2\left(2-3t+\frac{-5+12t-3t^2}{6}\right).
\]

Triangular elimination against the six nonzero coefficients of `H_5` leaves
one non-pole degree-ten candidate for `t`.  The final leading-coefficient
equation has a degree-nine remainder relatively prime to that candidate, so
the generic branch is empty.  Four factors divided out during the triangular
solve were checked separately over their exact number fields; after
saturating by `K != 0`, every one has Groebner basis `[1]`.

The most seductive exceptional chart has

\[
(A,B,C)=\left(\frac{\beta^2}{4},\beta,\frac{140}{\beta}\right).
\]

Its formal elliptic remainder is equianharmonic (`j=0`), but the actual
inverse polynomial is

\[
H_5=-\frac{9800}{\beta}.
\]

The degree-nine curve has disappeared completely.  This is a degree-drop
ghost, not a CM normalization.

There is also a useful dimension warning.  A monic
`Q_(m-2)^2 E_3` ansatz has two shape parameters after the missing high
coefficients are imposed.  Adding `A,B,C,K` gives six unknowns, while matching
`H_m` gives `m+1` equations.  The expected dimension is therefore `5-m`
before quotienting the scaling orbit: `m=4` is the last naturally permissive
case, `m=5` is isolated, and every higher `m` is overdetermined.  This is not
an impossibility theorem for all special higher-`m` fibers, but it reverses
the prior: higher `m` is less likely to provide a maximally nodal elliptic
normalization, not more likely.

The exact certificate is in `reverse_cm_search.py`.  Its default tests verify
the shape, the degree-drop ghost, and the coprime generic obstruction.  Running
the module itself also performs the slower number-field Groebner audit of all
exceptional strata.

The same elimination completes the algebraic `m=4` classification, where a
CM alternative was still dimensionally possible.  Write

\[
Q=x^2+x+t,
\qquad
E=x^3-2x^2+(3-2t)x+6t-4.
\]

The shape resultant is, up to a nonzero constant,

\[
t^4(t-2)^6(3t-2)^6(4t-1)^5
\bigl(8t^3-167t^2+180t-50\bigr)^4.
\]

The cases `t=0` and `t=2/3` are empty after saturation by `K != 0`, and
`t=1/4` is extraneous.  The final cubic is precisely the discriminant factor
of `E`, so those normalizations are singular rather than elliptic.  The only
nonsingular elliptic survivor is

\[
t=2,\qquad
(A,B,C,K)=\left(-\frac7{18},-\frac13,\frac{480}{7},\frac{300}{7}\right),
\]

\[
H_4=\frac{300}{7}(x^2+x+2)^2(x^3-2x^2-x+8),
\qquad
j=-\frac{224}{3}.
\]

This is the previously found rational two-node fiber in a rescaled source
coordinate.  Since a rational CM `j`-invariant is an integer, its nonintegral
value rules out CM.  Thus reverse search does not uncover a hidden CM
maximally nodal fiber even in the sole dimensionally favorable member.

Run the exact audit with:

```sh
python experiments/boundary_motive/test_m2_stable_boundary.py
python experiments/boundary_motive/test_m3_two_node_boundary.py
python experiments/boundary_motive/test_m45_cluster_tree.py
python experiments/boundary_motive/test_general_cluster_theorem.py
python experiments/boundary_motive/test_reverse_collision_search.py
python experiments/boundary_motive/test_reverse_cm_search.py
```
