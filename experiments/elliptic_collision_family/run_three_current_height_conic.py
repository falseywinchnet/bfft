"""Universal three-current height conic and its first split surface.

Let three zero-sum support triples have the same quadratic norm ``m`` and
product currents ``K1,K2,K3``.  Put ``h1=1`` and intersect

    h3 = 1 + s*(h2-1)

with the conic on which the three squared heights are affine in the
currents.  If ``A=K3-K1`` and ``B=K2-K1``, the second height is

    h2 = 1 + 2*(B*s-A)/(A-B*s^2),

and

    lambda = (h2^2-1)/B.

After multiplying every support root and height by ``lambda``, all nine
points lie on

    y^2 = x^3 - m*lambda^2*x + lambda^2*(1-lambda*K1).

This removes the final height compatibility gate identically; the only
remaining enemy is independence of the six section representatives.

The first shell with three distinct absolute nondegenerate currents is
``m=637``.  Its self-collision lift ``m=3*637^2`` has a stronger property:
the residual degree-eight curve-discriminant current splits into eight
rational linear factors.  The exact binary factors are exposed below.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import prod


Q = Fraction


SHELL_637_TRIPLES = (
    (-23, -4, 27),
    (-21, -7, 28),
    (-17, -12, 29),
)

SELF_COLLISION_TRIPLES = (
    (-1249, 407, 842),
    (-1078, -49, 1127),
    (-913, -313, 1226),
)

# The first square pullback of the two-orbit m=91 shell occurs at scale 7.
# Dividing the four integral supports of norm 91*7^2 by 7 produces four
# genuinely distinct rational support-current orbits on the original small
# norm.  Their positive currents are 90, 63510/343, 87450/343 and 330.
RATIONAL_91_TRIPLES = (
    (Q(10), Q(-9), Q(-1)),
    (Q(73, 7), Q(-58, 7), Q(-15, 7)),
    (Q(75, 7), Q(-53, 7), Q(-22, 7)),
    (Q(11), Q(-6), Q(-5)),
)

# Four rational support orbits on the m=217 rank-five wall.  The first,
# second and fourth supports parametrize the height conic; the third is the
# additional height whose square cuts out the rank-two quartic below.
RANK_FIVE_217_TRIPLES = (
    (Q(-17), Q(8), Q(9)),
    (Q(-16), Q(3), Q(13)),
    (Q(-13), Q(-3), Q(16)),
    (Q(-9), Q(-8), Q(17)),
)

# Each pair (a,b) denotes the primitive binary current a*p-b*q.  Their
# roots b/a are precisely the eight singular members of the self-collision
# height conic.
SELF_COLLISION_DISCRIMINANT_FORMS = (
    (3, 14),
    (3, -8),
    (38, 13),
    (75, 143),
    (152, -91),
    (209, 243),
    (209, 175),
    (729, 1001),
)


def support_current(triple: tuple[Fraction, Fraction, Fraction]) -> Fraction:
    """Return the product current of a zero-sum support triple."""

    if sum(triple) != 0:
        raise ValueError("a support triple must sum to zero")
    return prod(triple)


def support_norm(triple: tuple[Fraction, Fraction, Fraction]) -> Fraction:
    """Return ``m`` from ``sum(root^2)=2*m``."""

    if sum(triple) != 0:
        raise ValueError("a support triple must sum to zero")
    return sum(root * root for root in triple) / 2


@dataclass(frozen=True)
class ThreeCurrentHeightFiber:
    chord_slope: Fraction
    norm: Fraction
    base_triples: tuple[
        tuple[Fraction, Fraction, Fraction],
        tuple[Fraction, Fraction, Fraction],
        tuple[Fraction, Fraction, Fraction],
    ]
    currents: tuple[Fraction, Fraction, Fraction]
    height_ratio: tuple[Fraction, Fraction, Fraction]
    dilation: Fraction
    a0: Fraction

    @property
    def short_model(self) -> tuple[Fraction, Fraction, Fraction]:
        return Q(0), -self.norm * self.dilation**2, self.dilation**2 * self.a0

    @property
    def triples(self) -> tuple[
        tuple[Fraction, Fraction, Fraction],
        tuple[Fraction, Fraction, Fraction],
        tuple[Fraction, Fraction, Fraction],
    ]:
        return tuple(
            tuple(self.dilation * root for root in triple)
            for triple in self.base_triples
        )

    @property
    def heights(self) -> tuple[Fraction, Fraction, Fraction]:
        return tuple(self.dilation * height for height in self.height_ratio)

    @property
    def points(self) -> tuple[tuple[Fraction, Fraction], ...]:
        return tuple(
            (root, height)
            for triple, height in zip(self.triples, self.heights)
            for root in triple
        )

    @property
    def discriminant(self) -> Fraction:
        _a2, a4, a6 = self.short_model
        return -16 * (4 * a4**3 + 27 * a6**2)

    def verify(self) -> bool:
        _a2, a4, a6 = self.short_model
        if len(set(self.currents)) != 3 or not self.dilation:
            return False
        for triple, current, height in zip(
            self.triples, self.currents, self.heights
        ):
            if sum(triple) != 0:
                return False
            if sum(root * root for root in triple) != 2 * self.norm * self.dilation**2:
                return False
            if support_current(triple) != self.dilation**3 * current:
                return False
            for root in triple:
                if height * height != root**3 + a4 * root + a6:
                    return False
        return True


def build_three_current_height_fiber(
    triples: tuple[tuple[int, int, int], ...],
    chord_slope: Fraction,
    base_height: Fraction = Q(1),
) -> ThreeCurrentHeightFiber:
    """Build the universal fiber from three equal-norm split supports.

    ``base_height`` is the provisional height above the first support.  The
    former normalized construction is the default ``base_height=1``.  A
    nonunit base height lets the conic pass exactly through a known integral
    rank-five wall before taking a transverse Farey or group-law step.
    """

    if len(triples) != 3:
        raise ValueError("exactly three support triples are required")
    base_triples = tuple(tuple(Q(root) for root in triple) for triple in triples)
    norms = tuple(support_norm(triple) for triple in base_triples)
    if len(set(norms)) != 1:
        raise ValueError("the support triples must have one common norm")
    currents = tuple(support_current(triple) for triple in base_triples)
    if len(set(currents)) != 3:
        raise ValueError("the support currents must be distinct")

    slope = Q(chord_slope)
    first_height = Q(base_height)
    first, second, third = currents
    a_difference = third - first
    b_difference = second - first
    denominator = a_difference - b_difference * slope * slope
    if not denominator:
        raise ValueError("the height chord is tangent at infinity")
    increment = (
        2
        * first_height
        * (b_difference * slope - a_difference)
        / denominator
    )
    h2 = first_height + increment
    h3 = first_height + slope * increment
    dilation = (h2 * h2 - first_height * first_height) / b_difference
    if not dilation:
        raise ValueError("the height-conic point is degenerate")
    if h3 * h3 - first_height * first_height != dilation * a_difference:
        raise ArithmeticError("height-conic parametrization failed")
    a0 = first_height * first_height - dilation * first
    fiber = ThreeCurrentHeightFiber(
        slope,
        norms[0],
        base_triples,
        currents,
        (first_height, h2, h3),
        dilation,
        a0,
    )
    if not fiber.verify():
        raise ArithmeticError("three-current height fiber verification failed")
    return fiber


def self_collision_binary_currents(p: int, q: int) -> tuple[int, ...]:
    """Evaluate the eight primitive discriminant currents at ``s=p/q``."""

    if not q:
        raise ValueError("the projective chord denominator cannot vanish")
    return tuple(a * p - b * q for a, b in SELF_COLLISION_DISCRIMINANT_FORMS)


def self_collision_discriminant_current(slope: Fraction) -> Fraction:
    """Return the factored residual discriminant current.

    For the self-collision fiber, direct expansion gives

        D(s)=27-54*K1*lambda+(27*K1^2-4*m^3)*lambda^2
            = product(a*s-b)/(435600*(57*s^2-91)^4).

    The curve discriminant itself is ``-16*lambda^4*D(s)``.
    """

    slope = Q(slope)
    denominator = 57 * slope * slope - 91
    if not denominator:
        raise ValueError("the discriminant current is at infinity")
    factors = (a * slope - b for a, b in SELF_COLLISION_DISCRIMINANT_FORMS)
    return prod(factors) / (435_600 * denominator**4)


def shortest_farey_neighbors() -> tuple[Fraction, ...]:
    """Return the shortest primitive neighbor of each singular branch."""

    neighbors = []
    for a, b in SELF_COLLISION_DISCRIMINANT_FORMS:
        candidates = []
        for sign in (1, -1):
            for q in range(1, abs(a) + 1):
                numerator = b * q + sign
                if numerator % a == 0:
                    candidates.append((q, Q(numerator // a, q)))
                    break
        neighbors.append(min(candidates)[1])
    return tuple(neighbors)


def four_height_quartic_coefficients() -> tuple[int, int, int, int, int]:
    """Return the genus-one gate for all four rational ``m=91`` supports.

    Parametrize the height conic for currents ``90,63510/343,330``.  The
    remaining current ``87450/343`` has a square height exactly when

        H^2 = 18496*s^4 - 128248*s^3 + 358401*s^2
              - 323449*s + 117649.

    The omitted denominator is ``(136*s^2-343)^2``, already a square.
    """

    return 18_496, -128_248, 358_401, -323_449, 117_649


def binary_quartic_invariants(
    coefficients: tuple[int, int, int, int, int]
) -> tuple[int, int]:
    """Return the classical ``(I,J)`` invariants of a binary quartic."""

    a, b, c, d, e = coefficients
    invariant_i = 12 * a * e - 3 * b * d + c * c
    invariant_j = (
        72 * a * c * e
        + 9 * b * c * d
        - 27 * a * d * d
        - 27 * b * b * e
        - 2 * c**3
    )
    return invariant_i, invariant_j


def four_height_jacobian_model() -> tuple[int, int, int, int, int]:
    """Return the exact-rank-one minimal Jacobian of the quartic."""

    invariants = binary_quartic_invariants(four_height_quartic_coefficients())
    if invariants != (30_118_645_593, -6_610_138_671_777_930):
        raise ArithmeticError("unexpected four-height quartic invariants")
    return 0, 0, 0, -10_039_548_531, 244_819_950_806_590


def first_four_height_slopes() -> tuple[Fraction, Fraction]:
    """Return the two presentations of the first nonboundary recurrence.

    Tangency at ``(0,343)`` through ``(1,-207)`` leaves
    ``s^2*(s-1)*(5*s-77)``.  Tangency at the other boundary leaves
    ``s*(s-1)^2*(139*s-392)``.  Both slopes give the same elliptic curve,
    with one support height sign reversed.
    """

    return Q(77, 5), Q(392, 139)


def rank_five_217_quartic_coefficients() -> tuple[int, int, int, int, int]:
    """Return the four-height gate through the subrecord rank-five wall.

    Use currents ``-1224,-624,+1224`` with base height 19.  Requiring the
    unused ``+624`` support height to be rational gives

        Y^2=625*s^4-7700*s^3+34016*s^2-31416*s+10404.

    The wall ``s=17/6`` and subrecord member ``s=3`` are rational points.
    """

    return 625, -7_700, 34_016, -31_416, 10_404


def rank_five_217_jacobian_model() -> tuple[int, int, int, int, int]:
    """Return the exact-rank-two minimal Jacobian of the m=217 gate."""

    invariants = binary_quartic_invariants(rank_five_217_quartic_coefficients())
    if invariants != (509_408_656, -22_046_274_731_392):
        raise ArithmeticError("unexpected m=217 quartic invariants")
    return 0, -1, 0, -10_612_680, 12_761_798_400


def rank_five_217_distinguished_slopes() -> tuple[Fraction, Fraction, Fraction]:
    """Return wall, subrecord rank-five, and best rank-six slopes."""

    return Q(17, 6), Q(3), Q(-2)
