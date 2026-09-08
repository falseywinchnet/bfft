"""Exact Mordell--Weil recurrence on the minimal wall squareclass ``D=-23``.

The conductor-factorized angle surface leaves the binary quartic

    F(p)=p^4-8*p^3+2*p^2+8*p+1.

The local sieve in :mod:`run_wall_squareclass_sieve` proves that ``-23`` is
the smallest nondegenerate squareclass for ``F(p)=D*y^2``.  Shift ``u=p-2``;
then

    y^2 = 1 + 48/23*u + 22/23*u^2 - 1/23*u^4.

Direct elimination gives the birational map

    X=(2*(y+1)+d*u)/u^2,
    Y=((X^2-4*a)*u-X*d)/2

to

    Y^2=X^3+22/23*X^2+4/23*X-280/12167.

This curve has exact rank one.  ``GENERATOR`` generates its free part and
the four rational two-torsion cosets are deck presentations of the same angle
fiber.  The functions below follow its actual multiples, rather than quartic
interpolation shells.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, isqrt, prod
from typing import Optional


Q = Fraction
Point = Optional[tuple[Fraction, Fraction]]

A2 = Q(22, 23)
A4 = Q(4, 23)
A6 = Q(-280, 12167)
QUARTIC_A = Q(-1, 23)
QUARTIC_D = Q(48, 23)
GENERATOR = (Q(70, 529), Q(1680, 12167))
TORSION = (
    None,
    (Q(-14, 23), Q(0)),
    (Q(2, 23), Q(0)),
    (Q(-10, 23), Q(0)),
)


def add_points(left: Point, right: Point) -> Point:
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2 and y1 == -y2:
        return None
    if left == right:
        slope = (3 * x1 * x1 + 2 * A2 * x1 + A4) / (2 * y1)
    else:
        slope = (y2 - y1) / (x2 - x1)
    x3 = slope * slope - A2 - x1 - x2
    return x3, slope * (x1 - x3) - y1


def multiply(index: int, point: Point = GENERATOR) -> Point:
    if index < 0:
        if point is None:
            return None
        return multiply(-index, (point[0], -point[1]))
    result: Point = None
    while index:
        if index & 1:
            result = add_points(result, point)
        point = add_points(point, point)
        index //= 2
    return result


def quartic_point(point: Point) -> tuple[Fraction, Fraction]:
    """Inverse birational map to ``(p,y)`` on ``F(p)=-23*y^2``."""

    if point is None:
        return Q(2), Q(1)
    x, y = point
    denominator = x * x - 4 * QUARTIC_A
    u = (2 * y + x * QUARTIC_D) / denominator
    ordinate = (x * u * u - QUARTIC_D * u) / 2 - 1
    p = u + 2
    assert ordinate * ordinate == -(
        p**4 - 8 * p**3 + 2 * p * p + 8 * p + 1
    ) / 23
    return p, ordinate


def angle_parameter(p: Fraction) -> Fraction:
    return (
        (p + 1)
        * (p * p - 4 * p + 1)
        / ((p - 1) * (p * p + 4 * p + 1))
    )


def adjacent_shape(t: Fraction, index: Fraction = Q(5)) -> Fraction:
    return (
        (index - 1)
        * (index + 2) ** 2
        * t
        * (1 - t * t)
        / (index * (index - 2) * (t * t + 2 * t - 1) ** 2)
    )


def split_essential_current(p: Fraction, c: Fraction) -> Fraction:
    """Linear factor times quadratic norm of the essential discriminant."""

    r = (p * p - 1) / (2 * p)
    s = (r * r + 1) / (4 * r)
    support = c * (c + 1)
    root = 3 * s * s - 1
    linear = support - root
    quadratic = (
        9 * support * support * s**4
        - 6 * support * support * s * s
        + support * support
        - 9 * support * s * s
        + 2 * support
        - 3 * s * s
        + 1
    ) / root**2
    return linear * quadratic


def _primes_through(limit: int) -> tuple[int, ...]:
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    for value in range(2, isqrt(limit) + 1):
        if sieve[value]:
            sieve[value * value : limit + 1 : value] = b"\x00" * (
                (limit - value * value) // value + 1
            )
    return tuple(value for value in range(5, limit + 1) if sieve[value])


def integer_nth_root(value: int, exponent: int) -> int:
    """Floor of the positive ``exponent``-th root, computed exactly."""

    if value < 0 or exponent < 2:
        raise ValueError("expected a nonnegative value and exponent >= 2")
    if value < 2:
        return value
    root = 1 << ((value.bit_length() + exponent - 1) // exponent)
    while True:
        update = (
            (exponent - 1) * root + value // root ** (exponent - 1)
        ) // exponent
        if update >= root:
            break
        root = update
    while (root + 1) ** exponent <= value:
        root += 1
    while root**exponent > value:
        root -= 1
    return root


def perfect_power_exponent(value: int) -> int:
    """A prime exponent witnessing a perfect power, or zero."""

    if value < 2:
        return 0
    exponents = (2, 3) + _primes_through(value.bit_length())
    for exponent in exponents:
        root = integer_nth_root(value, exponent)
        if root**exponent == value:
            return exponent
    return 0


def fermat_composite_witness(
    value: int, bases: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17)
) -> Optional[int]:
    """Return a base that proves ``value`` composite, if one is immediate."""

    for base in bases:
        if value > base and pow(base, value - 1, value) != 1:
            return base
    return None


def _valuation(value: Fraction, prime: int) -> int:
    result = 0
    numerator = value.numerator
    denominator = value.denominator
    while numerator % prime == 0:
        numerator //= prime
        result += 1
    while denominator % prime == 0:
        denominator //= prime
        result -= 1
    return result


def conductor_divisor(
    multiple: int, trial_limit: int = 2_000_000
) -> tuple[tuple[int, ...], int]:
    """Certified conductor primes visible below ``trial_limit``.

    A visible prime is retained only if the normalized generalized model is
    integral and minimal there and the full discriminant has positive
    valuation.  Their product is therefore an unconditional divisor of the
    exact conductor, even though the residual current is not factored.
    """

    p, _ = quartic_point(multiply(multiple))
    t = angle_parameter(p)
    c = adjacent_shape(t)
    current = abs(split_essential_current(p, c).numerator)
    transport = t * (1 - t * t)
    r = transport * c * (1 + c)
    m = r * r * (1 + c + c * c)
    center = transport * c * c * (1 + c) ** 2 * (1 + t * t)
    discriminant = 64 * m**3 - 27 * center**4
    certified = []
    for prime in _primes_through(trial_limit):
        if current % prime:
            continue
        while current % prime == 0:
            current //= prime
        m_order = _valuation(m, prime)
        center_order = _valuation(center, prime)
        if (
            min(m_order, center_order) >= 0
            and not (m_order >= 4 and center_order >= 3)
            and _valuation(discriminant, prime) > 0
        ):
            certified.append(prime)
    return tuple(certified), prod(certified)


def clean_multiplicative_residual(
    multiple: int, trial_limit: int = 20_000_000
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Return the unfactored multiplicative current after local stripping.

    The current is first intersected with the numerator of the full
    discriminant.  Every prime shared with either coefficient, any coefficient
    denominator, or the current denominator is then removed.  All remaining
    prime divisors are therefore multiplicative in a minimal local model.
    Finally, exact trial factors through ``trial_limit`` are removed and
    returned separately.
    """

    p, _ = quartic_point(multiply(multiple))
    t = angle_parameter(p)
    c = adjacent_shape(t)
    essential = split_essential_current(p, c)
    transport = t * (1 - t * t)
    r = transport * c * (1 + c)
    m = r * r * (1 + c + c * c)
    center = transport * c * c * (1 + c) ** 2 * (1 + t * t)
    discriminant = 64 * m**3 - 27 * center**4
    residual = gcd(abs(essential.numerator), abs(discriminant.numerator))
    coefficient_support = abs(
        m.numerator
        * m.denominator
        * center.numerator
        * center.denominator
        * essential.denominator
        * discriminant.denominator
    )
    while True:
        common = gcd(residual, coefficient_support)
        if common == 1:
            break
        residual //= common
    factors = []
    for prime in _primes_through(trial_limit):
        if residual % prime:
            continue
        exponent = 0
        while residual % prime == 0:
            residual //= prime
            exponent += 1
        factors.append((prime, exponent))
    assert gcd(residual, coefficient_support) == 1
    assert discriminant.numerator % residual == 0
    return residual, tuple(factors)


def fourth_multiple_conductor_floor(
    trial_limit: int = 20_000_000,
) -> tuple[int, int, int]:
    """Composite/non-power certificate for the fourth recurrence class.

    Once trial factors are removed, a composite residual that is not a
    perfect power has at least two distinct prime factors.  Every residual
    prime is multiplicative by :func:`clean_multiplicative_residual`, and both
    exceed ``trial_limit``.  The returned strict floor is consequently an
    unconditional lower bound for the conductor contribution.
    """

    residual, _ = clean_multiplicative_residual(4, trial_limit)
    witness = fermat_composite_witness(residual)
    exponent = perfect_power_exponent(residual)
    if witness is None or exponent:
        raise AssertionError("the fourth residual certificate changed")
    return witness, exponent, (trial_limit + 1) ** 2


def main() -> None:
    for multiple in range(1, 5):
        point = multiply(multiple)
        p, _ = quartic_point(point)
        t = angle_parameter(p)
        primes, divisor = conductor_divisor(multiple)
        print(
            "multiple=", multiple,
            "p=", p,
            "t=", t,
            "parameter_bits=", max(p.numerator.bit_length(), p.denominator.bit_length()),
            "conductor_primes=", primes,
            "conductor_divisor=", divisor,
        )


if __name__ == "__main__":
    main()
