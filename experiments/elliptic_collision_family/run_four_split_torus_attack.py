"""Four split horizontal cubics from Eisenstein-torus doubling.

For a norm-one zero-sum support triple write

    X^2 + 3*b^2 = 4,       K = b^3-b.

The split-cubic discriminant is

    S^2 = 4-27*K^2,        S = X*(1-3*b^2).

Doubling ``(X+b*sqrt(-3))/2`` on the norm-one torus replaces ``b`` by
``X*b``.  Its product current is exactly ``K2=S*K``.  The compatibility
curve for square heights above the four currents ``+/-K,+/-K2`` is

    W^2 = z*(z-1)*(z-S^2).

It has the universal point ``(z,W)=(4,18*K)``.  The induced scale

    lambda = 36/(16-S^2)

makes both ``1-(lambda*K)^2`` and ``1-(lambda*K2)^2`` squares.  The final
cross-pair gate also factors identically as a square.  Consequently every
nondegenerate rational torus parameter gives a curve with four complete
horizontal support triples, without a search over Weierstrass coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt


Q = Fraction

WEYL_MATRICES: tuple[tuple[int, int, int, int], ...] = (
    (-1, -1, 0, 1),
    (-1, -1, 1, 0),
    (-1, 0, 0, -1),
    (-1, 0, 1, 1),
    (0, -1, -1, 0),
    (0, -1, 1, 1),
    (0, 1, -1, -1),
    (0, 1, 1, 0),
    (1, 0, -1, -1),
    (1, 0, 0, 1),
    (1, 1, -1, 0),
    (1, 1, 0, -1),
)


def primitive_parameter(t: Fraction) -> tuple[int, int]:
    """Return the coprime homogeneous pair ``(p,q)`` for ``t=p/q``."""

    t = Q(t)
    return t.numerator, t.denominator


def homogeneous_support_data(t: Fraction) -> tuple[int, int, int]:
    """Return the integral support invariants ``(D,R,N)``.

    ``D=p^2+p*q+q^2`` is the Eisenstein norm, ``R=p*q*(p+q)``, and
    ``N/D^3`` is the product current ``K``.
    """

    p, q = primitive_parameter(t)
    d = p * p + p * q + q * q
    r = p * q * (p + q)
    n = -p * q * (p - q) * (p + q) * (p + 2 * q) * (2 * p + q)
    return d, r, n


def integral_short_model(t: Fraction) -> tuple[int, int, int]:
    """Return an integral short model in the four-split Q-isomorphism class.

    The model is

        y^2 = x^3 - 9*D^4*x + 9*D^6 + 81*N^2/4.

    ``N`` is always even for a primitive pair, so the final coefficient is
    integral.  The model need not be locally minimal when ``t`` is a deck
    presentation such as ``1/4``; PARI removes that ordinary dilation.
    """

    d, _r, n = homogeneous_support_data(t)
    if n % 2:
        raise ArithmeticError("the homogeneous product current must be even")
    return 0, -9 * d**4, 9 * d**6 + 81 * n * n // 4


def conductor_norm_currents(t: Fraction) -> tuple[int, int]:
    """Return the two coprime-away-from-6 discriminant currents.

    Put ``A=2*D^3`` and ``B=9*R^2``.  Then

        F1 = Norm((A-6*B) + 3*B*sqrt(3)),
        F2 = Norm(A + B*sqrt(3)).

    The short-model discriminant is exactly ``3^6*F1*F2``.  For primitive
    ``(p,q)``, ``gcd(D,R)=1``; consequently a prime above 3 cannot divide
    both coefficients, and every common prime of ``F1,F2`` lies over 6.
    """

    d, r, _n = homogeneous_support_data(t)
    if gcd(d, r) != 1:
        raise ArithmeticError("primitive Eisenstein norm and product are not coprime")
    a = 2 * d**3
    b = 9 * r * r
    f1 = (a - 6 * b) ** 2 - 3 * (3 * b) ** 2
    f2 = a * a - 3 * b * b
    _a2, a4, a6 = integral_short_model(t)
    discriminant = -16 * (4 * a4**3 + 27 * a6**2)
    if discriminant != 3**6 * f1 * f2:
        raise ArithmeticError("conductor norm factorization failed")
    common = gcd(abs(f1), abs(f2))
    while common % 2 == 0:
        common //= 2
    while common % 3 == 0:
        common //= 3
    if common != 1:
        raise ArithmeticError("the norm currents share an unexpected prime")
    return f1, f2


def conductor_quartic_cores(t: Fraction) -> tuple[int, int]:
    """Return ``(F1/4,F2/4)`` as the two equianharmonic quartics.

    With ``R=2*r`` and

        g=(p-q)*(p+2*q)*(2*p+q)/2,

    the Eisenstein discriminant identity is ``D^3=g^2+27*r^2``.  It reduces
    the conductor currents to

        F1/4 = g^4-162*g^2*r^2-2187*r^4,
        F2/4 = g^4+ 54*g^2*r^2- 243*r^4.

    Both binary quartics have invariant ``I=0``; they are the two
    equianharmonic presentations joined by the 3-neighbor below.
    """

    p, q = primitive_parameter(t)
    d, r_twice, _n = homogeneous_support_data(t)
    if r_twice % 2:
        raise ArithmeticError("p*q*(p+q) must be even")
    difference_product = (p - q) * (p + 2 * q) * (2 * p + q)
    if difference_product % 2:
        raise ArithmeticError("the support difference product must be even")
    r = r_twice // 2
    g = difference_product // 2
    if d**3 != g * g + 27 * r * r:
        raise ArithmeticError("Eisenstein cubic discriminant identity failed")
    core1 = g**4 - 162 * g * g * r * r - 2187 * r**4
    core2 = g**4 + 54 * g * g * r * r - 243 * r**4
    f1, f2 = conductor_norm_currents(t)
    if (f1, f2) != (4 * core1, 4 * core2):
        raise ArithmeticError("quartic conductor reduction failed")
    return core1, core2


def three_neighbor(t: Fraction) -> Fraction:
    """The determinant-three support neighbor ``(p-q,p+2*q)``."""

    p, q = primitive_parameter(t)
    return Q(p - q, p + 2 * q)


def verify_three_neighbor_identity(t: Fraction) -> bool:
    """Verify ``F2(Tt)=-3^5*F1(t)`` before primitive rescaling.

    ``Fraction`` automatically removes a possible common factor three from
    the transformed pair.  Because the current is homogeneous of degree 12,
    that reduction contributes ``3^12``.  The returned equality includes
    precisely this deck-minimalization factor.
    """

    p, q = primitive_parameter(t)
    transformed_p = p - q
    transformed_q = p + 2 * q
    common = gcd(abs(transformed_p), abs(transformed_q))
    if common not in (1, 3):
        raise ArithmeticError("unexpected determinant-three content")
    f1, _f2 = conductor_norm_currents(t)
    _neighbor_f1, neighbor_f2 = conductor_norm_currents(three_neighbor(t))
    return neighbor_f2 * common**12 == -(3**5) * f1


def three_neighbor_weyl_fixed_discriminants() -> tuple[int, ...]:
    """Discriminants of all projective equations ``T(v)=W(v)``.

    The determinant-three map is ``T=[[1,-1],[1,2]]``.  Equating it
    projectively with each signed Eisenstein Weyl matrix gives a homogeneous
    quadratic in ``p,q``.  Every discriminant is ``-12,-3`` or ``12``;
    hence no nonzero rational support parameter can identify the two
    conductor vertices and collapse their prime supports.
    """

    values = []
    # At q=1 the cross-product equation is A*t^2+B*t+C.  Compute its
    # coefficients by evaluation, avoiding a symbolic dependency.
    for a, b, c, d in WEYL_MATRICES:
        def equation(t: int) -> int:
            left_p, left_q = t - 1, t + 2
            right_p, right_q = a * t + b, c * t + d
            return left_p * right_q - left_q * right_p

        constant = equation(0)
        at_one = equation(1)
        at_minus_one = equation(-1)
        linear = (at_one - at_minus_one) // 2
        quadratic = at_one - linear - constant
        values.append(linear * linear - 4 * quadratic * constant)
    result = tuple(values)
    if set(result) != {-12, -3, 12}:
        raise ArithmeticError("unexpected three-neighbor fixed discriminant")
    return result


def norm_unit_step(a: int, b: int, direction: int = 1) -> tuple[int, int]:
    """Multiply ``a+b*sqrt(3)`` by ``(2+sqrt(3))^direction``."""

    if direction == 1:
        return 2 * a + 3 * b, a + 2 * b
    if direction == -1:
        return 2 * a - 3 * b, -a + 2 * b
    raise ValueError("direction must be +1 or -1")


def _integer_cube_root(value: int) -> int:
    if value < 0:
        return -_integer_cube_root(-value)
    low, high = 0, 1
    while high**3 <= value:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if middle**3 <= value:
            low = middle
        else:
            high = middle
    return low


def cube_square_shape(a: int, b: int) -> tuple[int, int] | None:
    """Recover ``(D,R)`` when ``a=2*D^3`` and ``b=9*R^2``."""

    if a <= 0 or a % 2 or b < 0 or b % 9:
        return None
    d = _integer_cube_root(a // 2)
    r = isqrt(b // 9)
    if 2 * d**3 != a or 9 * r * r != b:
        return None
    return d, r


def unit_shape_hits(t: Fraction, radius: int) -> tuple[tuple[int, int, int], ...]:
    """Exact low-height cube/square hits in the fixed-``F2`` unit orbit.

    This is a recurrence audit, not a walk through rational parameters.  It
    follows the two deterministic Pell directions from one norm class and
    returns only orbit indices whose coefficients regain the required
    elliptic support shape.
    """

    if radius < 0:
        raise ValueError("radius must be nonnegative")
    d, r, _n = homogeneous_support_data(t)
    seed = (2 * d**3, 9 * r * r)
    hits: dict[int, tuple[int, int, int]] = {}
    for direction in (-1, 1):
        a, b = seed
        for index in range(radius + 1):
            shape = cube_square_shape(a, b)
            if shape is not None:
                hits[direction * index] = (direction * index, *shape)
            a, b = norm_unit_step(a, b, direction)
    return tuple(hits[index] for index in sorted(hits))


def modular_shape_indices(
    seed: tuple[int, int], modulus: int
) -> tuple[int, tuple[int, ...]]:
    """Complete one modular unit period for ``a=cube,b=18*square``.

    This gives an all-height obstruction for a fixed norm orbit whenever the
    returned index tuple is empty.  It is finite because multiplication by
    ``2+sqrt(3)`` is invertible modulo every modulus.
    """

    if modulus < 2:
        raise ValueError("modulus must be at least two")
    cubes = {value**3 % modulus for value in range(modulus)}
    shaped_b = {18 * value * value % modulus for value in range(modulus)}
    a, b = seed[0] % modulus, seed[1] % modulus
    start = a, b
    allowed = []
    index = 0
    while True:
        if a in cubes and b in shaped_b:
            allowed.append(index)
        a, b = (2 * a + 3 * b) % modulus, (a + 2 * b) % modulus
        index += 1
        if (a, b) == start:
            return index, tuple(allowed)
        if index > modulus * modulus + 1:
            raise ArithmeticError("unit orbit failed to close modulo modulus")


def first_nonunit_norm_obstructions() -> tuple[
    tuple[int, tuple[int, int], int, int], ...
]:
    """Certify that the core norm classes ``-11`` and ``13`` have no shape.

    In ``Z[sqrt(3)]`` the two prime norm classes have reduced generators
    ``1+/-2*sqrt(3)`` and ``4+/-sqrt(3)``.  Their associates exhaust the
    solutions because ``Q(sqrt(3))`` has fundamental unit ``2+sqrt(3)``.
    A complete recurrence period modulo the displayed modulus contains no
    simultaneous cube/square coefficient, at any height.
    """

    certificates = (
        (-11, (1, 2), 9, 18),
        (-11, (1, -2), 8, 4),
        (13, (4, 1), 7, 8),
        (13, (4, -1), 7, 8),
    )
    for _norm, seed, modulus, period in certificates:
        actual_period, allowed = modular_shape_indices(seed, modulus)
        if actual_period != period or allowed:
            raise ArithmeticError("fixed-norm modular obstruction failed")
    return certificates


def square_current_jacobians() -> tuple[
    tuple[int, int, int, int, int], tuple[int, int, int, int, int]
]:
    """Return the rank-zero Jacobians of the two square-current quartics.

    PARI's exact certificate is in ``four_split_torus_pari.gp``.  The first
    quartic is birational to the displayed nonminimal model and then to
    ``y^2=x^3+1``; the second is birational to ``y^2=x^3-27``.
    """

    return (0, 54, 0, 972, 52488), (0, -162, 0, 8748, -1417176)


def square_current_torsion_support_polynomials() -> tuple[
    tuple[int, int, int, int], tuple[int, int, int, int]
]:
    """Return the two cyclic cubics left by the nonboundary F2 torsion.

    The only finite nonboundary rational points on

        z^2=g^4+54*g^2*r^2-243*r^4

    have ``g/r=+/-3``.  Pulling those ratios back through the split cubic
    support map gives the two monic cubics below.  Their only possible
    rational roots are ``+/-1``, and neither is a root, so the cubics are
    cyclic irreducible rather than three rational linear factors.
    """

    polynomials = ((1, 0, -3, -1), (1, 3, 0, -1))
    for coefficients in polynomials:
        for root in (-1, 1):
            value = sum(
                coefficient * root ** (3 - index)
                for index, coefficient in enumerate(coefficients)
            )
            if value == 0:
                raise ArithmeticError("square-current support cubic split")
    return polynomials


def cube_second_current_jacobian() -> tuple[int, int, int, int, int]:
    """Return the rank-zero Jacobian closing ``F2/4 = cube``.

    Let ``F2/4=C^3`` and ``X=D^2``.  Since ``gcd(D,r)=1`` and ``D`` is
    prime to six on the primitive deck, factoring

        X^3-C^3=972*r^4

    forces

        X-C=324*u^4,       X^2+X*C+C^2=3*v^4.

    The 3-adic allocation is unique: the second factor has valuation exactly
    one because ``X=C=1 (mod 3)``.  Eliminating ``C`` gives

        (X-162*u^4)^2 = v^4-8748*u^8.

    For ``u != 0`` this is the quartic ``w^2=x^4-8748``.  Its Jacobian is
    the model below, minimalized by PARI to ``y^2=x^3+27*x`` with rank zero
    and rational torsion ``Z/2``.  That torsion is the point at infinity of
    the quartic, so the cube current has no nonboundary rational class.
    """

    return 0, 0, 0, 34_992, 0


@dataclass(frozen=True)
class FourSplitFiber:
    parameter: Fraction
    rho: Fraction
    m: Fraction
    a6: Fraction
    currents: tuple[Fraction, Fraction, Fraction, Fraction]
    heights: tuple[Fraction, Fraction, Fraction, Fraction]
    triples: tuple[tuple[Fraction, Fraction, Fraction], ...]

    @property
    def short_model(self) -> tuple[Fraction, Fraction, Fraction]:
        return Q(0), -self.m, self.a6

    @property
    def points(self) -> tuple[tuple[Fraction, Fraction], ...]:
        values = []
        for triple, height in zip(self.triples, self.heights):
            values.extend((root, height) for root in triple)
        return tuple(values)

    def verify(self) -> bool:
        if len(set(self.currents)) != 4:
            return False
        for current, triple, height in zip(
            self.currents, self.triples, self.heights
        ):
            if sum(triple) != 0:
                return False
            if sum(root * root for root in triple) != 2 * self.m:
                return False
            if -triple[0] * triple[1] * (
                triple[0] + triple[1]
            ) != current:
                return False
            for root in triple:
                if height * height != root**3 - self.m * root + self.a6:
                    return False
        return True


def support_coordinates(t: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    """Return ``(a,b,X=2*a+b)`` on ``a^2+a*b+b^2=1``."""

    t = Q(t)
    denominator = t * t + t + 1
    a = (t - 1) * (t + 1) / denominator
    b = -t * (t + 2) / denominator
    return a, b, 2 * a + b


def current_and_split_root(t: Fraction) -> tuple[Fraction, Fraction]:
    _a, b, x = support_coordinates(t)
    current = b**3 - b
    split_root = x * (1 - 3 * b * b)
    if split_root * split_root != 4 - 27 * current * current:
        raise ArithmeticError("split-current identity failed")
    return current, split_root


def doubled_support(t: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    """Double the norm-one torus point and return its zero-sum triple."""

    _a, b, x = support_coordinates(t)
    doubled_b = x * b
    doubled_x = (x * x - 3 * b * b) / 2
    doubled_a = (doubled_x - doubled_b) / 2
    return doubled_a, doubled_b, -doubled_a - doubled_b


def cross_height_root(t: Fraction) -> Fraction:
    """Square root of ``(1+lambda*K2)/(1+lambda*K)``."""

    t = Q(t)
    numerator = (
        2 * t**6
        + 18 * t**5
        + 15 * t**4
        - 40 * t**3
        - 45 * t**2
        - 6 * t
        + 2
    )
    denominator = (
        2 * t**6
        - 3 * t**4
        + 14 * t**3
        + 27 * t**2
        + 12 * t
        + 2
    )
    if denominator == 0:
        raise ValueError("cross-height chart is singular")
    return numerator / denominator


def build_four_split_fiber(t: Fraction) -> FourSplitFiber:
    """Construct the exact four-horizontal-triple fiber at ``t``."""

    t = Q(t)
    a, b, _x = support_coordinates(t)
    k1, split_root = current_and_split_root(t)
    base = (a, b, -a - b)
    doubled = doubled_support(t)
    k2 = -doubled[0] * doubled[1] * (
        doubled[0] + doubled[1]
    )
    if k2 != split_root * k1:
        raise ArithmeticError("torus doubling did not multiply the current by S")
    if not k1 or not k2 or k1 == k2 or k1 == -k2:
        raise ValueError("degenerate four-current parameter")

    lam = Q(36, 1) / (16 - split_root * split_root)
    d1 = lam * k1
    d2 = lam * k2
    u1 = (8 + split_root * split_root) / (16 - split_root * split_root)
    u2 = (7 * split_root * split_root - 16) / (
        16 - split_root * split_root
    )
    if u1 * u1 + d1 * d1 != 1:
        raise ArithmeticError("first Legendre height gate failed")
    if u2 * u2 + d2 * d2 != 1:
        raise ArithmeticError("second Legendre height gate failed")

    cross = cross_height_root(t)
    if cross * cross != (1 + d2) / (1 + d1):
        raise ArithmeticError("cross-height current did not factor")

    # Multiplying rho by a rational square only dilates the same Q-isomorphism
    # class.  This representative makes the first positive-current height a
    # square without extracting any roots.
    rho = lam * (1 + d1)
    m = rho * rho
    a6 = rho**3 / lam
    h1_plus = lam * (1 + d1) ** 2
    h1_minus = lam * (1 + d1) * u1
    h2_plus = h1_plus * cross
    h2_minus = h2_plus * u2 / (1 + d2)

    scaled_base = tuple(rho * value for value in base)
    scaled_doubled = tuple(rho * value for value in doubled)
    currents = (
        rho**3 * k1,
        -(rho**3 * k1),
        rho**3 * k2,
        -(rho**3 * k2),
    )
    heights = (h1_plus, h1_minus, h2_plus, h2_minus)
    triples = (
        scaled_base,
        tuple(-value for value in scaled_base),
        scaled_doubled,
        tuple(-value for value in scaled_doubled),
    )
    fiber = FourSplitFiber(t, rho, m, a6, currents, heights, triples)
    if not fiber.verify():
        raise ArithmeticError("four-split fiber verification failed")
    return fiber


def gp_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def gp_model(fiber: FourSplitFiber) -> str:
    return "[0,0,0,%s,%s]" % (
        gp_fraction(-fiber.m),
        gp_fraction(fiber.a6),
    )


def main() -> None:
    for t in (Q(1, 2), Q(1, 3), Q(1, 4), Q(2, 3), Q(2)):
        try:
            fiber = build_four_split_fiber(t)
        except ValueError:
            continue
        print(
            "t=", t,
            "model=", gp_model(fiber),
            "currents=", fiber.currents,
            "heights=", fiber.heights,
        )


if __name__ == "__main__":
    main()
