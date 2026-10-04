# No nondegenerate equilateral pentagon is a rep-tile

A pentagon here is a simple planar polygon of positive area with five genuine
corners: its boundary is a Jordan curve, its vertices are distinct, and no
interior angle equals pi. In particular, nonadjacent edges are disjoint and
adjacent edges meet only at their common endpoint. Concave pentagons are
included. A rep-N dissection, N >= 2,
is a finite partition into mutually congruent smaller polygons similar to the
original; reflections and rotations are allowed. No edge-to-edge hypothesis
is imposed.

**Theorem. There are no equilateral pentagonal rep-tiles in this class.**

The convex case uses an established theorem. The concave case has the
self-contained proof below. This note makes no claim that the concave
argument is new in the literature.

Normalize every side length of the original pentagon to 1. Angles below are
interior angles unless stated otherwise.

## Lemma 1: angular span of a unit segment

Let O, X, Y be distinct points, with |XY| = 1 and |OX|, |OY| >= 1.
The smaller angle XOY is at most pi/3.

Write r = |OX| and s = |OY|. The cosine rule gives

    cos(angle XOY) = (r^2 + s^2 - 1)/(2rs) >= 1/2,

because

    r^2 + s^2 - rs = (r-s)^2 + rs >= 1.

The continuous argument change along XY therefore has absolute value at
most pi/3. In particular XY cannot pass through O.

## Lemma 2: a concave equilateral pentagon has an angle below pi/3

Suppose instead that all five angles are at least pi/3. Their sum is 3pi,
so every angle is also at most 5pi/3. For each vertex V, the diagonal joining
its neighbors has length

    2 sin(angle(V)/2) >= 1.

Thus every pair of distinct vertices is separated by at least 1.

Choose a reflex vertex O of angle rho > pi. Orient the polygon
counterclockwise and write its vertex order as O, X1, X2, X3, X4.
Follow the three-edge path X1-X2-X3-X4, continuously tracking its argument
about O. Its total argument change is rho.

For completeness, this identity does not assume O sees the entire polygon.
Take a sufficiently small circle about O, intersecting the incident edges at
U and V. The long boundary path U-X1-X2-X3-X4-V can be closed by the
clockwise circular arc inside the polygon from V to U, whose argument change
is -rho. This is the boundary of the polygon with a small corner sector
removed. It excludes O, so its winding number about O is zero. The radial
pieces contribute zero; hence the three-edge path contributes rho.

Lemma 1 bounds each of its three argument changes in absolute value by pi/3.
Their sum cannot exceed pi, contradicting rho > pi. Therefore some interior
angle is strictly below pi/3.

## Lemma 3: an elementary triangle obstruction

If X lies strictly inside triangle UVW and |XU| = |XV| = 1, it is impossible
to have both |WU| <= 1 and |WV| <= 1.

Translate X to the origin and write u, v, w for the other vertices. Then
|u| = |v| = 1, and the two asserted inequalities imply

    u dot w >= |w|^2/2 > 0,
    v dot w >= |w|^2/2 > 0.

Also w dot w > 0. Every point of triangle UVW therefore has strictly positive
dot product with w, whereas X has dot product zero. This is a contradiction.

## Lemma 4: explicit visibility of an empty pentagonal corner triangle

Let ABCDE be a simple pentagon, let A be convex, and let T be the closed
triangle ABE. If C and D are outside T, then the interior of T and the
relative interior of BE are contained in the interior of the polygon.
Consequently T is an ear, and cutting along BE leaves the simple polygon
BCDE. No general-position assumption about nonconsecutive vertices is used.

First, none of the remaining edges BC, CD, DE can enter the interior of T.
Indeed, if such an edge entered, its intersection with the closed convex
triangle T would be a segment of positive length containing interior points
of T. Its two endpoints would lie on the boundary of T: C and D are outside
T, and the only possible endpoints of the original edge in T are B and E.
Simplicity forbids contact with AB or AE except at the prescribed common
endpoints B or E. Thus both ends of the intersection segment would have to
lie on BE. The entire segment would then lie on the line BE and could not
contain an interior point of T. This contradiction excludes entry.

It follows that the interior of T is disjoint from the polygon boundary.
Because A is convex, some interior points of T sufficiently close to A are
inside the polygon. The interior of T is connected, so it lies entirely in
the interior component of the complement of the Jordan boundary.

Next, no point of the open segment BE can belong to the polygon boundary.
Suppose X were such a point. It is not a polygon vertex: C and D are outside T, and A is not
on BE. It would therefore be in the relative interior of an edge BC, CD,
or DE. Let S be that edge. By the preceding argument, the convex intersection
S intersect T contains no point of the interior of T. Suppose S is not
collinear with BE. Because X is in the relative interior of BE, choose a
small disk about X that misses AB and AE; within this disk, T is exactly
the closed half-disk on its interior side of the line BE. Because X is
also in the relative interior of S, the segment S extends in both directions
from X within the disk. Noncollinearity therefore makes S intersect T
contain a nontrivial subsegment whose points other than X are in the
interior of T, contradicting the preceding argument. Both relative-interior
conditions are essential to this local conclusion; contact with the
supporting line alone would not suffice. If S were collinear with BE,
the absence of its endpoints from the open segment BE would
force it to contain the whole segment BE. Edge BC would then contain the
nonincident vertex E, edge DE would contain B, and edge CD would contain
both. Each possibility violates simplicity. This also excludes tangencies
and overlaps, not just transverse crossings.

Finally, every point of the open segment BE is a limit of interior points
of T, hence belongs to the closed polygonal region. Since none is on its
boundary, every such point is strictly inside the polygon. This proves
visibility of BE and the ear assertion.

## Lemma 5: every angle below pi/3 is adjacent to every reflex vertex

Label the pentagon A, B, C, D, E in boundary order, with angle A = alpha < pi/3.
Then

    |BE| = 2 sin(alpha/2) < 1.

We verify the closed-triangle exclusion required by Lemma 4. If X belongs
to the closed triangle ABE, write X = a A + b B + e E with nonnegative
coefficients summing to 1. Then

    |X-B| <= a |A-B| + e |E-B| = a + e |E-B| < 1

unless X = A. The strict inequality follows from |E-B| < 1 and b+e > 0
whenever X != A. Therefore C, which is distinct from A and satisfies
|C-B| = 1, is outside the CLOSED triangle ABE. Interchanging B and E proves
the same for D, since |D-E| = 1. In particular neither C nor D can lie on
the base BE or on either other side.

Since A is convex, Lemma 4 now applies: triangle ABE is an ear, the open
segment BE is wholly inside the polygon, and removing the triangle leaves
the simple quadrilateral BCDE. The new angles at B or E may be straight;
the argument below only requires simplicity and the unchanged angles at
C and D.

The quadrilateral has consecutive side lengths 1, 1, 1, d, with d = |BE| < 1.
Its angles at C and D are unchanged by the ear removal.

If C were reflex, it would lie strictly inside triangle BDE. But

    |CB| = |CD| = 1,
    |EB| = d < 1,  |ED| = 1,

contradicting Lemma 3 with X=C, U=B, V=D, W=E.

If D were reflex, the same lemma applies with X=D, U=C, V=E, W=B.
Therefore neither C nor D is reflex. Every reflex vertex of the original
pentagon must be B or E, adjacent to A.

This argument does not assume the pentagon has exactly one reflex vertex.

## Concave nonexistence theorem

Let P be a concave equilateral pentagon and let alpha be its smallest
interior angle. Lemma 2 gives alpha < pi/3. Lemma 5 shows that every vertex
of angle alpha is adjacent to a reflex vertex. This holds even if the
minimum angle occurs more than once.

Suppose P has a rep-N dissection, N >= 2. Every tile has side length
r = 1/sqrt(N), with 0 < r < 1.

At a corner A of P with angle alpha, precisely one tile meets A. A tile
edge passing through A would require a pi-sector, and a tile interior
would require a 2pi-sector. Both are impossible. Each incident tile must
instead contribute one of P's interior angles, all of which are >= alpha;
their sum is alpha, so exactly one tile contributes exactly alpha.

That tile's two edges at A follow the two boundary edges of P. In the tile,
the alpha vertex is adjacent to a reflex vertex R, by Lemma 5. Hence R lies
on one of those boundary edges at distance r from A. Since r < 1, it is in
the relative interior of an original side.

Near R the original polygon occupies only a pi-sector. The tile occupies
a sector strictly larger than pi, since R is reflex. The tile cannot be
contained in P. This contradiction proves concave nonexistence.

In fact this concave proof also excludes a finite dissection into smaller
similar copies of differing sizes: only the corner tile's scale r < 1
was used, not equality of scales among tiles.

## Convex case and exact count

The established convex-reptile theorem states that every convex reptile
is a triangle or a trapezoid. In particular a genuine convex pentagon cannot
be a reptile, equilateral or otherwise.

Reference: Miklos Laczkovich, *Quadrilateral reptiles* (2022), Corollary 1.2,
combining the paper's main theorem with earlier results of U. Betke and
I. Osburg. The introduction also records the earlier result that convex
reptiles have at most four sides.

- https://arxiv.org/abs/2205.11068
- https://arxiv.org/html/2205.11068

Every nondegenerate simple equilateral pentagon is either convex or concave.
The two arguments exclude both cases. Therefore the exact number of such
rep-tile shapes, up to similarity, is **zero**, and no replication order
N >= 2 is possible.

The extension to different tile sizes above concerns the concave case only.
The cited convex theorem concerns congruent smaller copies and must not be
used to claim a classification of convex irreptiles.

## Definition boundary

Allowing a 180-degree marked vertex admits shapes with fewer than five
genuine sides, so that convention is outside this theorem. Self-intersecting
five-link chains likewise require a different specification of the region
to be tiled. Neither convention is used here.

## Proof audit

- No classification of concave pentagonal plane tilings is needed.
- No assumption that there is only one reflex vertex is needed.
- Tied minimum angles are covered by the universal statement in Lemma 5.
- Lemma 4 explicitly establishes diagonal visibility, excluding transverse
  crossings, contacts at vertices, and collinear overlaps. Lemma 5 supplies
  exclusion from the closed triangle, not just from its interior.
- Reflections, rotations, and non-edge-to-edge dissections are allowed.
- The winding argument in Lemma 2 does not assume a visible fan triangulation.
- The conclusion follows analytically; a finite numerical search is not used
  as an exhaustive classification.
- The convex result is an explicitly cited external theorem; the concave
  argument is supplied in full and is not claimed as a literature novelty.
