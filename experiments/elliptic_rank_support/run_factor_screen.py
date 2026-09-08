"""Factor the finite nearest-fiber audit; intended for the M4 Mini."""

from __future__ import annotations

from math import prod
from fractions import Fraction

import sympy as sp

from elliptic_rank_support import (
    RECORD_CONDUCTOR,
    integral_record_points,
    invariants,
    add_points,
    reduction_dependency_masks,
    support_pencils,
)


PRIMES = tuple(sp.primerange(5, 1000))


def rational_square_root(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    numerator = int(sp.integer_nthroot(value.numerator, 2)[0])
    denominator = int(sp.integer_nthroot(value.denominator, 2)[0])
    if numerator * numerator != value.numerator or denominator * denominator != value.denominator:
        return None
    return Fraction(numerator, denominator)


def halve_point(point, model):
    """Return one rational half of ``point``, or ``None`` if none exists."""

    if point is None:
        return None
    xp, _ = point
    a2, a4, a6 = map(Fraction, model)
    variable = sp.symbols("r")
    r = variable
    f = r**3 + sp.Rational(a2.numerator, a2.denominator) * r**2
    f += sp.Rational(a4.numerator, a4.denominator) * r
    f += sp.Rational(a6.numerator, a6.denominator)
    tangent_numerator = 3 * r**2 + 2 * sp.Rational(a2.numerator, a2.denominator) * r
    tangent_numerator += sp.Rational(a4.numerator, a4.denominator)
    quartic = sp.Poly(
        sp.expand(
            tangent_numerator**2
            - 4 * f * (sp.Rational(xp.numerator, xp.denominator) + a2 + 2 * r)
        ),
        r,
        domain=sp.QQ,
    )
    for root in sp.roots(quartic):
        if not root.is_Rational:
            continue
        xr = Fraction(int(root.p), int(root.q))
        value = xr**3 + a2 * xr**2 + a4 * xr + a6
        yr = rational_square_root(value)
        if yr is None:
            continue
        for signed_y in (yr, -yr):
            candidate = (xr, signed_y)
            if add_points(candidate, candidate, model) == point:
                return candidate
    return None


def saturate_visible_subgroup(model, points):
    """Saturate at 2 until rank five is certified or a relation is exposed."""

    generators = [(Fraction(x), Fraction(y)) for x, y in points[:5]]
    halvings = []
    seen_states = set()
    for _ in range(12):
        state = tuple(generators)
        if state in seen_states:
            return "dependent", generators, halvings + [("cycle", None)]
        seen_states.add(state)
        integer_points = tuple((point[0], point[1]) for point in generators)
        dependencies = reduction_dependency_masks(model, integer_points, PRIMES)
        if not dependencies:
            return "rank5", generators, halvings
        progressed = False
        for mask in sorted(dependencies):
            total = None
            indices = []
            for index, generator in enumerate(generators):
                if mask & (1 << index):
                    total = add_points(total, generator, model)
                    indices.append(index)
            if total is None:
                return "dependent", generators, halvings + [(mask, None)]
            half = halve_point(total, model)
            if half is None:
                continue
            negative_half = (half[0], -half[1])
            if half in generators or negative_half in generators:
                return "dependent", generators, halvings + [(mask, half)]
            generators[indices[0]] = half
            halvings.append((mask, half))
            progressed = True
            break
        if not progressed:
            return "unresolved", generators, halvings
    return "unresolved", generators, halvings


def main() -> None:
    pencils = support_pencils(integral_record_points())
    candidates = []
    for index, pencil in enumerate(pencils):
        base_model = pencil.integral_model(2)
        base_points = pencil.transported_points(2)
        if reduction_dependency_masks(base_model, base_points, PRIMES):
            continue
        for u in (1, 3):
            model = pencil.integral_model(u)
            points = pencil.transported_points(u)
            if model is None or points is None:
                continue
            c4, _, discriminant = invariants(model)
            factors = sp.factorint(abs(discriminant))
            radical = prod(int(prime) for prime in factors)
            # At p>=5, p|c4 implies additive reduction and conductor exponent
            # two.  The wild exponents at 2 and 3 are deliberately left as a
            # lower bound until a candidate survives the rank certificate.
            tame_lower_bound = radical * prod(
                int(prime)
                for prime in factors
                if prime >= 5 and c4 % prime == 0
            )
            if tame_lower_bound >= RECORD_CONDUCTOR:
                continue
            status, saturated_generators, halvings = saturate_visible_subgroup(model, points)
            candidates.append(
                {
                    "index": index,
                    "u": u,
                    "model": model,
                    "points": points,
                    "discriminant": discriminant,
                    "factors": factors,
                    "tame_conductor_lower_bound": tame_lower_bound,
                    "status": status,
                    "halvings": halvings,
                    "saturated_generators": saturated_generators,
                }
            )
    for candidate in candidates:
        if candidate["status"] != "dependent":
            print(candidate)
    print(f"tame-sub-record candidates={len(candidates)}")
    for status in ("dependent", "unresolved", "rank5"):
        print(status, sum(candidate["status"] == status for candidate in candidates))
    print(
        "rank-five certificates=",
        sum(candidate["status"] == "rank5" for candidate in candidates),
    )


if __name__ == "__main__":
    main()
