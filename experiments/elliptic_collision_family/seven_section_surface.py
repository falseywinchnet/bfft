"""The normalized two-parameter surface behind the seven-section family.

The shape coordinates are ``c=s/r`` and the Euclidean parameter ``t``.
The third coordinate ``w`` is the ordinary Weierstrass dilation and therefore
does not change the isomorphism class over Q.  Keeping it explicit separates
geometric shape from arithmetic height.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

from elliptic_collision_family import CollisionFiber


Q = Fraction
Quadratic6 = tuple[Fraction, Fraction]


def surface_fiber(
    c: int | Fraction,
    t: int | Fraction,
    w: int | Fraction = 1,
) -> CollisionFiber:
    """Return the seven-section fiber at normalized coordinates ``(c,t,w)``.

    The parametrization is obtained by matching the support area to the
    Euclidean right triangle with slope parameter ``t``.
    """

    c, t, w = map(Q, (c, t, w))
    if c in (0, -1) or t in (0, 1, -1) or w == 0:
        raise ValueError("degenerate seven-section surface coordinate")
    transport = t * (1 - t * t)
    r = transport * c * (1 + c) * w * w
    s = c * r
    height_scale = transport * c * c * (1 + c) ** 2 * w**3
    return CollisionFiber(
        r=r,
        s=s,
        lower_height=height_scale * (1 - 2 * t - t * t),
        center_height=height_scale * (1 + t * t),
        upper_height=height_scale * (1 + 2 * t - t * t),
    )


def extra_section_r_minus_s_conic(
    c: int | Fraction,
    t: int | Fraction,
) -> Fraction:
    """Square carrier for the additional abscissa ``x=r-s``.

    If this value is ``z^2``, the normalized seven-section fiber acquires the
    rational point returned by :func:`extra_section_r_minus_s`.
    """

    c, t = map(Q, (c, t))
    return (c + 1) * (
        (c + 1) * (1 + t * t) ** 2 + 12 * (c - 1) * t * (1 - t * t)
    )


def extra_section_r_minus_s_parameter(
    t: int | Fraction,
    slope: int | Fraction,
) -> tuple[Fraction, Fraction]:
    """Parametrize the extra-section conic from its point at ``c=1``.

    The line is ``z=2*(1+t^2)+slope*(c-1)``.  The returned point is its
    second intersection with the conic.
    """

    t, slope = map(Q, (t, slope))
    linear = 1 + t * t
    transport = t * (1 - t * t)
    denominator = slope * slope - linear * linear - 12 * transport
    if denominator == 0:
        raise ValueError("the parametrizing line is tangent at infinity")
    offset = (
        4 * linear * linear + 24 * transport - 4 * linear * slope
    ) / denominator
    c = 1 + offset
    z = 2 * linear + slope * offset
    assert z * z == extra_section_r_minus_s_conic(c, t)
    return c, z


def extra_section_r_minus_s(
    c: int | Fraction,
    t: int | Fraction,
    z: int | Fraction,
    w: int | Fraction = 1,
) -> tuple[Fraction, Fraction]:
    """Return the extra rational point at ``x=r-s`` on the surface fiber."""

    c, t, z, w = map(Q, (c, t, z, w))
    if z * z != extra_section_r_minus_s_conic(c, t):
        raise ValueError("z is not on the extra-section conic")
    transport = t * (1 - t * t)
    r = transport * c * (1 + c) * w * w
    x = (1 - c) * r
    y = transport * c * c * (1 + c) * z * w**3 / 2
    fiber = surface_fiber(c, t, w)
    assert y * y == x**3 - fiber.m * x + fiber.center_height**2 / 4
    return x, y


def sixth_height_shell_forms(
    c: int | Fraction, t: int | Fraction
) -> tuple[Fraction, ...]:
    """Primitive factors of the extra section's first height shell.

    Write ``E`` for the section at ``x=r-s`` and use the rank-five basis
    ``P0, P_r, P_s, Q_-r, Q_-s``.  In order, these are the nonboundary
    x-coincidence divisors for

    ``E = +/- (P0+P_r), +/- (P0-P_r), +/- (P0+P_s),
    +/- (P0-P_s), +/- (P0+Q_-r), +/- (P0-Q_-r),
    +/- (P0+Q_-s), +/- (P0-Q_-s)``.

    Because ``E`` already lies on the curve, equal abscissas are sufficient:
    the two points differ by at most sign, and hence the sixth section is
    dependent.  The removed factors ``c*(c+1)*t*(t-1)*(t+1)`` are precisely
    the degenerate support/angle boundary.
    """

    c, t = map(Q, (c, t))
    return (
        c * c * t * t + c * c * t + c * t * t + c + 2 * t - 2,
        -c * c * t + c * c + c * t * t + c - 2 * t * t - 2 * t,
        c * t * t + 2 * c * t - c + t * t + t,
        c * t * t + 2 * c * t - c + t - 1,
        c * t * t - c * t + t * t - 2 * t - 1,
        -c * t - c + t * t - 2 * t - 1,
        -2 * c * c * t - 2 * c * c + c * t * t + c + t * t - t,
        2 * c * c * t * t - 2 * c * c * t - c * t * t - c - t - 1,
    )


def sixth_height_shell_current(
    c: int | Fraction, t: int | Fraction
) -> Fraction:
    """Product of the eight irreducible first-shell height-loss factors."""

    result = Q(1)
    for form in sixth_height_shell_forms(c, t):
        result *= form
    return result


def exposed_base_height_forms(
    c: int | Fraction, t: int | Fraction
) -> tuple[Fraction, Fraction]:
    """The two base-five loss sheets exposed by the ``1/5`` and ``1/7`` traps.

    The first is the chord relation
    ``P0+Q_-r=P_r+P_s+Q_-s``.  The second is ``P0+P_r=P_s``.
    They are kept separate from :func:`sixth_height_shell_forms` because the
    original rank-five lattice has already collapsed before the extra section
    is considered.
    """

    c, t = map(Q, (c, t))
    return (
        2 * c * t * t - c * t - c - 3 * t - 1,
        c * t * t + c * t + t - 1,
    )


def base_height_shell_forms(
    c: int | Fraction, t: int | Fraction
) -> tuple[Fraction, ...]:
    """Complete adjacent shell for relations inside the rank-five basis.

    These are the four nonboundary divisors obtained by comparing
    ``P0 +/- P_r`` with ``P_s`` and ``P0 +/- Q_-r`` with ``Q_-s``;
    exchanging ``r`` and ``s`` repeats the same factors.  They supplement the
    six lower/upper collinearity forms in :func:`cross_relation_forms`.
    """

    c, t = map(Q, (c, t))
    return (
        c * t * t + c * t + t - 1,
        c * t - c + t * t + t,
        c * t * t - c * t - t - 1,
        -c * t - c + t * t - t,
    )


def base_second_height_forms(
    c: int | Fraction, t: int | Fraction
) -> tuple[Fraction, Fraction]:
    """First two coefficient-two base-height walls and their triangle mate.

    The first contains ``P0+P_s=-2*Q_-r`` at ``(c,t)=(-9,1/2)``.
    The second is its Pythagorean transform ``t -> (1-t)/(1+t)`` and catches
    the relabeled ``t=1/3`` presentation of the same arithmetic curve.
    """

    c, t = map(Q, (c, t))
    return (
        c * c * t**3
        - 2 * c * c * t * t
        + c * c * t
        + 2 * c * t**3
        - 5 * c * t * t
        + 2 * c * t
        + c
        + t**3
        - 3 * t * t
        - 3 * t
        + 1,
        c * c * t**3
        - c * c * t * t
        + 2 * c * t**3
        - 3 * c * t * t
        - c * t
        - 3 * t * t
        + 1,
    )


def extra_section_r_plus_2s_carrier(
    c: int | Fraction,
    t: int | Fraction,
) -> Fraction:
    """Square carrier for the other support-edge difference ``x=r+2*s``.

    Unlike the ``r-s`` carrier, this current is linear in ``c``:

    ``(1+t^2)^2 + 12*t*(1-t^2)*(1+2*c)``.
    """

    c, t = map(Q, (c, t))
    transport = t * (1 - t * t)
    return (1 + t * t) ** 2 + 12 * transport * (1 + 2 * c)


def extra_section_r_plus_2s_parameter(
    t: int | Fraction,
    root: int | Fraction,
) -> Fraction:
    """Return the shape ``c`` for a prescribed edge-current square root."""

    t, root = map(Q, (t, root))
    transport = t * (1 - t * t)
    if transport == 0:
        raise ValueError("degenerate angle parameter")
    return (root * root - (1 + t * t) ** 2 - 12 * transport) / (
        24 * transport
    )


def extra_section_r_plus_2s(
    c: int | Fraction,
    t: int | Fraction,
    root: int | Fraction,
    w: int | Fraction = 1,
) -> tuple[Fraction, Fraction]:
    """Return the rational edge-difference point at ``x=r+2*s``."""

    c, t, root, w = map(Q, (c, t, root, w))
    if root * root != extra_section_r_plus_2s_carrier(c, t):
        raise ValueError("root is not on the edge-difference current")
    transport = t * (1 - t * t)
    r = transport * c * (1 + c) * w * w
    x = (1 + 2 * c) * r
    y = transport * c * c * (1 + c) ** 2 * root * w**3 / 2
    fiber = surface_fiber(c, t, w)
    assert y * y == x**3 - fiber.m * x + fiber.center_height**2 / 4
    return x, y


def extra_section_2r_plus_s_carrier(
    c: int | Fraction,
    t: int | Fraction,
) -> Fraction:
    """Square carrier for the reciprocal edge difference ``x=2*r+s``.

    This is the ``r+2*s`` current after interchanging the two primitive
    support coordinates and clearing the harmless square denominator.
    """

    c, t = map(Q, (c, t))
    transport = t * (1 - t * t)
    return c * c * (1 + t * t) ** 2 + 12 * transport * c * (c + 2)


def extra_section_2r_plus_s(
    c: int | Fraction,
    t: int | Fraction,
    root: int | Fraction,
    w: int | Fraction = 1,
) -> tuple[Fraction, Fraction]:
    """Return the reciprocal edge-difference point at ``x=2*r+s``."""

    c, t, root, w = map(Q, (c, t, root, w))
    if root * root != extra_section_2r_plus_s_carrier(c, t):
        raise ValueError("root is not on the reciprocal edge current")
    transport = t * (1 - t * t)
    r = transport * c * (1 + c) * w * w
    x = (2 + c) * r
    y = transport * c * (1 + c) ** 2 * root * w**3 / 2
    fiber = surface_fiber(c, t, w)
    assert y * y == x**3 - fiber.m * x + fiber.center_height**2 / 4
    return x, y


def linear_abscissa_carrier(
    c: int | Fraction,
    t: int | Fraction,
    constant: int | Fraction,
    support_coefficient: int | Fraction,
) -> Fraction:
    """Universal square carrier for ``x=(constant+coefficient*c)*r``.

    This exposes the full Eisenstein/A2 section geometry without enumerating
    Weierstrass abscissas.  If the returned value is ``root^2``,
    :func:`linear_abscissa_section` supplies the corresponding rational point.
    """

    c, t, constant, support_coefficient = map(
        Q, (c, t, constant, support_coefficient)
    )
    transport = t * (1 - t * t)
    linear = constant + support_coefficient * c
    support_norm = 1 + c + c * c
    return c * (1 + c) * (
        c * (1 + c) * (1 + t * t) ** 2
        + 4 * transport * (linear**3 - support_norm * linear)
    )


def linear_abscissa_section(
    c: int | Fraction,
    t: int | Fraction,
    constant: int | Fraction,
    support_coefficient: int | Fraction,
    root: int | Fraction,
    w: int | Fraction = 1,
) -> tuple[Fraction, Fraction]:
    """Return a point from the universal linear-abscissa carrier."""

    c, t, constant, support_coefficient, root, w = map(
        Q, (c, t, constant, support_coefficient, root, w)
    )
    if root * root != linear_abscissa_carrier(
        c, t, constant, support_coefficient
    ):
        raise ValueError("root is not on the linear-abscissa current")
    transport = t * (1 - t * t)
    r = transport * c * (1 + c) * w * w
    x = (constant + support_coefficient * c) * r
    y = transport * c * (1 + c) * root * w**3 / 2
    fiber = surface_fiber(c, t, w)
    assert y * y == x**3 - fiber.m * x + fiber.center_height**2 / 4
    return x, y


def a2_weight_carriers(
    c: int | Fraction, t: int | Fraction
) -> tuple[Fraction, Fraction]:
    """Carriers of the two smallest non-root A2 weight abscissas."""

    return (
        linear_abscissa_carrier(c, t, Q(2, 3), Q(1, 3)),
        linear_abscissa_carrier(c, t, Q(1, 3), Q(2, 3)),
    )


def rank_six_extra_section_example(
) -> tuple[tuple[int, int, int], tuple[tuple[int, int], ...]]:
    """A theory-selected specialization with six mod-2 independent sections.

    It comes from ``t=1/3``, the midpoint conic slope
    ``-2*(1+t^2)``, hence ``c=149`` and ``z=-980/3``.  The returned short
    model has already been divided by the available 10-dilation.
    """

    model = (0, -7_939_432_816, 4_928_844_010_000)
    points = (
        (0, 2_220_100),
        (596, 444_020),
        (88_804, 444_020),
        (-596, 3_108_140),
        (-88_804, 3_108_140),
        (-88_208, -4_351_396),
    )
    a2, a4, a6 = model
    assert all(y * y == x**3 + a2 * x * x + a4 * x + a6 for x, y in points)
    return model, points


def essential_discriminant(
    c: int | Fraction, t: int | Fraction
) -> Fraction:
    """Return the discriminant factor that depends only on surface shape.

    The complete short-Weierstrass discriminant is

    ``T^4*c^6*(1+c)^6*w^12 * essential_discriminant(c,t)``,

    where ``T=t*(1-t^2)``.  The removed factors are the support dilation and
    the three degenerate boundary components.
    """

    c, t = map(Q, (c, t))
    transport = t * (1 - t * t)
    return (
        64 * transport**2 * (1 + c + c * c) ** 3
        - 27 * c * c * (1 + c) ** 2 * (1 + t * t) ** 4
    )


def short_discriminant(fiber: CollisionFiber) -> Fraction:
    """Discriminant of ``y^2=x^3-m*x+q^2/4``."""

    return 64 * fiber.m**3 - 27 * fiber.center_height**4


def cross_relation_forms(
    c: int | Fraction, t: int | Fraction
) -> tuple[Fraction, ...]:
    """The six nondegenerate cross-line relation divisors.

    Vanishing of any entry makes the center section, one lower section, and
    one upper section collinear.  It therefore introduces a third visible
    Mordell--Weil relation and can reduce the seven sections from rank five to
    rank four.
    """

    c, t = map(Q, (c, t))
    return (
        -2 * c + t - 1,
        2 * c + t + 1,
        c * t - c - 2,
        c * t + c + 2,
        c * t - c + t + 1,
        c * t + c + t - 1,
    )


def rank_safe_contour_t(
    c: int | Fraction, clearance: int | Fraction = Q(-7, 9)
) -> Fraction:
    """Keep exact clearance from the rank-four wall through the ``q=29`` trap.

    The relevant wall is ``(c+1)*t-c+1=0``.  The record has value ``-7/9``;
    holding that value fixed gives a rational one-dimensional contour.
    """

    c, clearance = map(Q, (c, clearance))
    if c == -1:
        raise ValueError("the safe contour has a pole at c=-1")
    return (c - 1 + clearance) / (c + 1)


def _factor_denominator(value: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    prime = 2
    while prime * prime <= value:
        while value % prime == 0:
            factors[prime] = factors.get(prime, 0) + 1
            value //= prime
        prime = 3 if prime == 2 else prime + 2
    if value > 1:
        factors[value] = factors.get(value, 0) + 1
    return factors


def short_integral_scale(fiber: CollisionFiber) -> int:
    """Smallest integer dilation making the short model and sections integral.

    This is a presentation scale, not a claim that the resulting generalized
    Weierstrass equation is locally minimal.
    """

    weighted_values = [(fiber.m, 4), (fiber.center_height**2 / 4, 6)]
    weighted_values.extend((x, 2) for x, _ in fiber.seven_points())
    weighted_values.extend((y, 3) for _, y in fiber.seven_points())
    requirements: dict[int, int] = {}
    for value, weight in weighted_values:
        for prime, exponent in _factor_denominator(value.denominator).items():
            needed = (exponent + weight - 1) // weight
            requirements[prime] = max(requirements.get(prime, 0), needed)
    scale = 1
    for prime, exponent in requirements.items():
        scale *= prime**exponent
    return scale


def _q6_add(left: Quadratic6, right: Quadratic6) -> Quadratic6:
    return left[0] + right[0], left[1] + right[1]


def _q6_multiply(left: Quadratic6, right: Quadratic6) -> Quadratic6:
    return (
        left[0] * right[0] + 6 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def quadratic6_norm(value: Quadratic6) -> Fraction:
    """Norm of ``value[0] + value[1]*sqrt(6)``."""

    return value[0] * value[0] - 6 * value[1] * value[1]


def _q6_inverse(value: Quadratic6) -> Quadratic6:
    norm = quadratic6_norm(value)
    if norm == 0:
        raise ZeroDivisionError("zero divisor in Q(sqrt(6))")
    return value[0] / norm, -value[1] / norm


def _q6_power(value: Quadratic6, exponent: int) -> Quadratic6:
    result: Quadratic6 = (Q(1), Q(0))
    for _ in range(exponent):
        result = _q6_multiply(result, value)
    return result


def nodal_center_unit() -> Quadratic6:
    """Torus coordinate of the positive-center section at every nodal fiber."""

    return Q(-5), Q(-2)


def nodal_height_coordinate(unit: Quadratic6) -> Quadratic6:
    """Return ``H(U)=U*(1+U)/(1-U)^3`` in ``Q(sqrt(6))``."""

    one: Quadratic6 = (Q(1), Q(0))
    numerator = _q6_multiply(unit, _q6_add(one, unit))
    denominator = _q6_power(_q6_add(one, (-unit[0], -unit[1])), 3)
    return _q6_multiply(numerator, _q6_inverse(denominator))


def homogeneous_discriminant(
    support_numerator: int,
    support_denominator: int,
    angle_numerator: int,
    angle_denominator: int,
) -> int:
    """Bi-homogeneous arithmetic carrier for the shape discriminant.

    ``c=C/B`` and ``t=P/Q``.  The result has bidegree ``(6,8)`` in
    ``(C,B)`` and ``(P,Q)``.  Unlike the real-valued shape factor, its prime
    divisors survive rational denominator clearing.
    """

    c, b, p, q = (
        int(support_numerator),
        int(support_denominator),
        int(angle_numerator),
        int(angle_denominator),
    )
    support_norm = b * b + b * c + c * c
    support_product = b * c * (b + c)
    return (
        64 * p * p * (q * q - p * p) ** 2 * q * q * support_norm**3
        - 27 * support_product**2 * (q * q + p * p) ** 4
    )


def positive_chamber_wall_offsets(
    support_numerator: int,
    support_denominator: int,
    angle_numerator: int,
    angle_denominator: int,
) -> tuple[int, ...]:
    """Homogeneous offsets from all signed cross walls meeting ``c>1,0<t<1``."""

    c, b, p, q = (
        int(support_numerator),
        int(support_denominator),
        int(angle_numerator),
        int(angle_denominator),
    )
    return (
        (2 * c + b) * p - b * q,
        (c + b) * p - b * q,
        c * p - b * q,
        (c + 2 * b) * p - c * q,
        (c + b) * p - c * q,
        (c + b) * p - (c - b) * q,
    )


def primitive_support_square_carrier(
    support_numerator: int,
    support_denominator: int,
    angle_numerator: int,
    angle_denominator: int,
) -> int:
    """Square class whose vanishing permits primitive support ``(B,C)``.

    A rational dilation can make ``r=B,s=C`` exactly iff this integer is a
    square (up to the harmless sign choices outside the positive chamber).
    """

    c, b, p, q = (
        int(support_numerator),
        int(support_denominator),
        int(angle_numerator),
        int(angle_denominator),
    )
    return b * q * p * (q * q - p * p) * c * (b + c)


def is_integer_square(value: int) -> bool:
    if value < 0:
        return False
    root = isqrt(value)
    return root * root == value


def unit_offset_control_rhs(b: int) -> int:
    """Right side of the primitive-support control curve at ``P/Q=1/6``.

    On the nearest-wall characteristic ``2*C-5*B=-1``, primitive support is
    equivalent to ``Y^2 = unit_offset_control_rhs(B)``.
    """

    b = int(b)
    return 210 * b * (5 * b - 1) * (7 * b - 1)
