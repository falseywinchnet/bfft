# Equilateral polygonal rep-tiles: a local obstruction for every n >= 5

Date: 2026-09-23

## Statement and scope

**Theorem. A simple equilateral polygon with at least five genuine vertices
cannot be dissected into finitely many smaller similar copies of itself.**

The copies may be reflected, rotated, and of different sizes. No
edge-to-edge hypothesis is required. In particular, the exact number of
equilateral hexagonal rep-tile shapes, up to similarity, is zero.

A polygon here has positive area and a Jordan boundary: nonadjacent edges
are disjoint, adjacent edges meet only at their common endpoint, and all
vertices are distinct. A genuine vertex has interior angle different from
pi. Straight marked subdivisions of fewer-sided polygons are excluded.

The argument below is self-contained. It does not use the published convex
reptile theorem or the pentagon-specific acute-angle and ear lemmas in
PROOF.md. No claim of literature novelty is made.

Normalize the polygon's side length to 1. All angles below are in radians.

## 1. Necessary conditions at a smallest-angle corner

### Exact dissection hypotheses and scale bound

Regard P and the tiles T_1,...,T_N as closed polygonal regions, with
N >= 2, positive tile areas, union exactly P, and pairwise disjoint
interiors. Every tile is a similarity image of P; its ratio r_i is
positive, and reflections are allowed. Polygonal boundaries have area
zero, so area additivity gives

    sum_i r_i^2 = 1.

Because N >= 2 and all ratios are positive, EVERY ratio satisfies r_i < 1.
Thus the proof does not require congruent tiles or an independently
assumed common scale.

Let alpha be the smallest interior angle of P. The angle sum gives
0 < alpha <= (n-2)pi/n < pi. Choose an outer vertex A_0 with angle alpha.

### Finite-sector fact

At any point X of P, a sufficiently small disk about X meets only tiles
that contain X. Indeed, each tile is compact, so a tile not containing X
has strictly positive distance from X; there are only finitely many tiles.
Shrink the disk further so that every incident tile has its exact local
polygonal form there: a full disk at an interior point, a half-disk at
the relative interior of a side, or an interior-angle sector at a vertex.
The same choice makes P locally its corresponding disk, half-disk, or
sector. This is possible because there are finitely many edges and
vertices, and each boundary is simple.

These incident sectors cover the ambient sector of P and have pairwise
disjoint angular interiors. Their angle measures therefore add exactly
to the ambient angle. This fact requires no edge-to-edge matching.

### A unique corner tile and its full incident sides

Apply the finite-sector fact at A_0. A tile interior or the relative
interior of a tile edge contributes 2pi or pi and cannot fit into alpha<pi.
Every incident tile must instead have a vertex at A_0, with angle at
least alpha. Their angles sum to alpha, so precisely one tile T contains
A_0, and its angle there is alpha.

Moreover, its closed local sector equals P's closed local sector. Because
alpha<pi, equality of these sectors forces equality of their two boundary
RAYS, not merely their supporting lines. Let u and v be the unit vectors
along the two outer sides issuing from A_0. The original sides are

    {A_0+s u : 0 <= s <= 1},
    {A_0+s v : 0 <= s <= 1}.

If T has ratio r, both of its true incident sides have length r, because
P is equilateral. Polygon sides are straight segments; local agreement
with a ray fixes the direction of the ENTIRE side. Their endpoints must
therefore be

    X = A_0+r u,   Y = A_0+r v.

Since 0<r<1, both full tile sides lie on the original sides, and X,Y lie
strictly inside those original sides. An endpoint or junction from a
different tile cannot shorten, bend, or redefine a true side of T.
No assumption about matching subdivisions or neighboring tile sizes is
being made here.

### Exact angle sums at the two endpoints

Let beta and gamma be the angles of T at X and Y. Apply the finite-sector
fact at X. Since X is in the relative interior of an outer side, P's
ambient angle is pi. Containment gives beta<=pi, and the genuine-vertex
hypothesis excludes beta=pi. Thus 0<beta<pi, leaving a positive angular
gap pi-beta.

Other incident tiles must cover that gap. Neither a full-disk contribution
2pi nor a straight-edge contribution pi can fit into a gap smaller than
pi. Every such tile therefore contributes a true vertex angle theta_j
of P, with theta_j>=alpha. At least one is present, and angle additivity
gives the exact relation

    pi = beta + sum_j theta_j,   with a nonempty sum.

Consequently beta+alpha<=pi. Also beta>=alpha, since it is itself an angle
of P. Applying the identical argument at Y proves

    alpha <= beta <= pi - alpha,
    alpha <= gamma <= pi - alpha.                         (1)

In particular alpha <= pi/2. If several vertices attain the minimum, the
corner tile may use any of them; the next lemma rules out (1) at every one.

### Scope audit of the corner argument

- Finiteness and compactness justify reducing coverage to tiles incident
  at the point; an infinite family would need a separate argument.
- Area additivity gives r<1 independently for each tile, even at unequal
  scales. Equal side lengths then place BOTH neighboring vertices at
  distance r along original sides of length 1.
- The full-side assertion follows from sector-ray equality and the
  straightness and prescribed length of a tile's true sides. It is not
  inferred from edge-to-edge matching.
- Genuine vertices ensure beta,gamma!=pi. This hypothesis is essential:
  the 2-by-1 rectangle with boundary vertices (0,0),(1,0),(2,0),(2,1),
  (1,1),(0,1) has six marked unit segments and a four-copy self-dissection,
  but two marked vertices are straight and the shape has only four true
  sides. At a relevant marked endpoint the residual gap can be zero.

## 2. Local geometric lemma

**Lemma.** Let a simple equilateral polygon have at least five genuine
vertices. At a smallest-angle vertex A, let its two adjacent interior
angles be beta and gamma. They cannot both be at most pi - alpha, where
alpha is the angle at A.

### Setup

Suppose the contrary. Orient the polygon counterclockwise, with successive
local vertices

    ..., E, F, A, B, C, ...

Thus alpha is at A, beta at B, and gamma at F. The five vertices E,F,A,B,C
are distinct, including for a pentagon. We know

    AB = BC = AF = FE = 1,
    alpha <= beta,gamma <= pi-alpha,
    0 < alpha <= pi/2.

Put

    t = (pi-alpha)/2,
    b = BF = 2 sin(alpha/2) = 2 cos(t).

Triangle ABF is isosceles, with base angles t.

### The outgoing edges must point beyond the base BF

We first prove beta > t and gamma > t.

If alpha > pi/3, then beta,gamma >= alpha > t, immediately.

If alpha <= pi/3, then b <= 1. Suppose beta < t. Ray BC starts strictly
inside triangle ABF at B and exits through AF. The distance from B to
every point of AF is at most 1, since both endpoints A,F are at distance
at most 1 from B and the closed unit disk centered at B is convex. Thus
the unit segment BC meets AF, contradicting simplicity. If beta = t,
ray BC follows BF and its unit segment contains F because BF <= 1, also
contradicting simplicity. Hence beta > t. The same argument gives gamma > t.

Define

    delta = beta-t,
    epsilon = gamma-t.

Then

    0 < delta,epsilon <= t,
    delta+epsilon <= 2t = pi-alpha < pi.                  (2)

### Orientation lemma: the forward rays meet opposite A

Choose coordinates preserving the counterclockwise boundary orientation:

    B = (0,0),  F = (b,0),  A = (b/2, sin(t)).

These coordinates are possible because A is convex, AB=AF=1, and
b=2 cos(t). Thus A lies in the upper open half-plane of the line BF.
The direction angles of BA and FA are t and pi-t, respectively.

At B the incoming direction AB is t+pi. Since beta < pi, the signed
counterclockwise boundary turn is pi-beta. The outgoing direction BC is
therefore t+pi+(pi-beta), or t-beta=-delta modulo 2pi. At F the outgoing
direction FA is pi-t, and the turn from EF to FA is pi-gamma. Hence the
incoming direction EF is gamma-t=epsilon, and the reversed direction FE
is pi+epsilon. The rays are consequently

    B + lambda (cos(delta), -sin(delta)),       lambda >= 0,
    F + mu     (-cos(epsilon), -sin(epsilon)), mu >= 0.

Since delta,epsilon > 0 and both are below pi/2, each ray, except its
initial point, lies in the lower open half-plane, opposite A. Their
intersection equations are

    lambda cos(delta) + mu cos(epsilon) = b,
    lambda sin(delta) - mu sin(epsilon) = 0.

By (2), sin(delta+epsilon)>0. The equations have the unique solution

    lambda = b sin(epsilon)/sin(delta+epsilon) > 0,
    mu     = b sin(delta)/sin(delta+epsilon) > 0.

Thus the intersection Z is on BOTH forward rays, strictly below BF;
neither initial point B nor F can be Z. Because the direction vectors
are unit vectors, the solution also proves directly (equivalently by
the sine rule in triangle BFZ) that

    BZ = b sin(epsilon)/sin(delta+epsilon),
    FZ = b sin(delta)/sin(delta+epsilon).                 (3)

This calculation uses only the convex local angles alpha,beta,gamma
forced by (1). Other vertices may be concave. If a tile was reflected,
its boundary is simply relabeled counterclockwise before this argument;
the angle bounds and equal side lengths are unchanged.

### The intersection lies on both closed unit edges

Both lengths are at most 1. For the first, epsilon <= t < pi/2 and
sin(delta) > 0 imply

    sin(delta+epsilon)/sin(epsilon)
      = sin(delta) cot(epsilon) + cos(delta)
      >= sin(delta) cot(t) + cos(delta)
      = sin(delta+t)/sin(t)
      = sin(beta)/sin(t)
      >= sin(alpha)/sin(t)
      = 2 cos(t)
      = b.

The last inequality uses beta in [alpha, pi-alpha] and alpha <= pi/2.
Equation (3) now gives BZ <= 1. Interchanging delta and epsilon, and using
the corresponding bound on gamma, gives FZ <= 1.

Since BC and FE both have length 1, we have 0 < BZ <= 1 and 0 < FZ <= 1.
The four possible cases are exhaustive:

1. BZ < 1 and FZ < 1: Z is in the relative interior of both edges. Their
   supporting lines are nonparallel because sin(delta+epsilon)>0, so this
   is a transverse intersection of nonadjacent edges.
2. BZ = 1 and FZ < 1: Z=C lies in the relative interior of FE. This places
   a polygon vertex on a nonincident edge, violating simplicity.
3. BZ < 1 and FZ = 1: Z=E lies in the relative interior of BC, likewise
   placing a polygon vertex on a nonincident edge.
4. BZ = 1 and FZ = 1: Z=C=E, contradicting distinct vertices when n>=5.

For an algebraic equality audit, the first inequality in the displayed
bound on BZ is an equality exactly when epsilon=t, because sin(delta)>0
and cotangent is strictly decreasing on (0,pi). The second is an equality
exactly when beta is alpha or pi-alpha. Therefore, under the standing
strict bounds (2),

    BZ=1 iff gamma=pi-alpha and beta in {alpha, pi-alpha},
    FZ=1 iff beta=pi-alpha and gamma in {alpha, pi-alpha}.

Both equalities occur precisely when beta=gamma=pi-alpha. Only BZ=1
occurs precisely when pi/3 < alpha < pi/2, beta=alpha, gamma=pi-alpha;
only FZ=1 has beta and gamma interchanged. In those single-equality
cases the strict lower bound alpha>pi/3 is required by beta>t or gamma>t.
At alpha=pi/2 the two endpoint angle values coincide and both lengths
equal 1. Every admissible equality case is thus included in cases 2-4.

This proves the lemma.

## 3. Nonexistence and classification

A proposed self-dissection forces a minimum-angle occurrence satisfying
(1), by Section 1. Section 2 shows that no such occurrence exists in any
simple equilateral polygon with at least five genuine vertices. Therefore
no such polygon admits the dissection.

In particular:

    number of equilateral hexagonal rep-tile shapes = 0;
    allowable rep-N orders for those hexagons = the empty set.

The full equilateral classification follows as well. An equilateral
triangle is the regular triangle, and a simple equilateral quadrilateral
is a rhombus. Each admits a subdivision into four congruent copies at
scale 1/2. Since every n >= 5 is excluded, the simple nondegenerate
equilateral polygonal rep-tiles are exactly the equilateral triangle and
the rhombi.

The nonexistence argument also covers unequal-sized smaller similar pieces:
only the corner tile's ratio r < 1 was used. The triangle and rhombus
constructions already supply congruent pieces.

## 4. Why the pentagon-specific route did not directly extend

There are concave equilateral hexagons with no angle below pi/3. An exact
example, in boundary order, is

    (1,0), (1/2,sqrt(3)/2), (-1/2,sqrt(3)/2),
    (-1,0), (-1/2,-sqrt(3)/2), (0,0).

It is the union of four consecutive unit equilateral triangles around the
origin. Its six interior angles are

    pi/3, 2pi/3, 2pi/3, 2pi/3, pi/3, 4pi/3.

Thus the earlier assertion that every concave equilateral pentagon has an
angle strictly below pi/3 cannot be substituted unchanged for hexagons.
The local lemma above avoids that assertion entirely.

## Proof audit

- No unproved interior-diagonal or ear assertion is used.
- The only required intersections involve the original four unit edges
  FE, FA, AB, BC; the rest of the boundary is irrelevant.
- The orientation lemma establishes both ray directions in counterclockwise
  coordinates and solves for strictly positive forward-ray parameters.
- The sine-rule denominators are positive by the same strict bounds.
- Equality in BZ <= 1 or FZ <= 1 is audited both algebraically and through
  four exhaustive geometric cases, including the three endpoint cases.
- At n=4 the vertices C and E are the same prescribed vertex, so the final
  contact is permitted. This is consistent with the rhombus constructions.
- Repeated minimum angles, reflected copies, arbitrary finite replication
  counts, non-edge-to-edge contacts, and different tile scales are covered.
- Genuine vertices are necessary: a subdivided rhombus boundary can have
  five or more marked equal segments while retaining its self-dissection.
- This is an analytical proof, not an inference from a finite search.
