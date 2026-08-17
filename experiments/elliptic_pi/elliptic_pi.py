"""Radical-free elliptic/isogeny iterations for pi.

The cubic and quartic Borwein maps are expressed as selection of the small
positive root of a fixed-degree polynomial.  The polynomials below are stable
expanded forms: unlike the direct eliminated equations, they do not subtract
two quantities close to one near the cusp.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import mpmath as mp


def _precision_stages(dps: int) -> list[int]:
    target = dps + 20
    stages = []
    precision = min(50, target)
    while precision < target:
        stages.append(precision)
        precision = min(target, 2 * precision)
    stages.append(target)
    return stages


def _newton_staged(initial, polynomial_and_derivative, dps: int) -> mp.mpf:
    """Fixed-degree root extraction in O(M(n)) via precision doubling."""
    x = None
    for stage_index, precision in enumerate(_precision_stages(dps)):
        mp.mp.dps = precision
        if x is None:
            x = +initial()
            iterations = 7
        else:
            x = +x
            iterations = 2
        for _ in range(iterations):
            value, derivative = polynomial_and_derivative(x)
            x -= value / derivative
    mp.mp.dps = dps
    return +x


def positive_root_y0(dps: int) -> mp.mpf:
    """Small positive root of y^2+2y-1=0, without sqrt."""
    return _newton_staged(
        lambda: mp.mpf(2) / 5,
        lambda y: (y * y + 2 * y - 1, 2 * y + 2),
        dps,
    )


def positive_root_s0(dps: int) -> mp.mpf:
    """Small positive root of 2s^2+2s-1=0, without sqrt."""
    return _newton_staged(
        lambda: mp.mpf(1) / 3,
        lambda s: (2 * s * s + 2 * s - 1, 4 * s + 2),
        dps,
    )


def descend_lambda(u: mp.mpf, dps: int) -> mp.mpf:
    """Small root of 4(u-2)^2 v-u^2(1+v)^2=0."""
    u = +u

    def equation(v):
        value = 4 * (u - 2) ** 2 * v - u * u * (1 + v) ** 2
        derivative = 4 * (u - 2) ** 2 - 2 * u * u * (1 + v)
        return value, derivative

    return _newton_staged(lambda: u * u / 16, equation, dps)


def lambda2_branch_pair(u: mp.mpf):
    """Return both radical sheets of the descending degree-2 lambda map.

    With s=sqrt(1-u) and r=(1-s)/(1+s), the contracting sheet is v=r^2.
    Changing the sign of s gives 1/v.  The associated Landen multipliers for
    F(u)=(2/pi)K(sqrt(u)) are 1+r and 1+1/r on their respective continued
    period branches.
    """
    u = +u
    complement = mp.sqrt(1 - u)
    ratio = (1 - complement) / (1 + complement)
    small = ratio * ratio
    large = 1 / small
    return small, large, 1 + ratio, 1 + 1 / ratio


def lambda2_branch_invariants(u: mp.mpf):
    """Root-free trace/norm data for the two degree-2 sheets."""
    u = +u
    branch_trace = 16 / (u * u) - 16 / u + 2
    multiplier_trace_and_norm = 4 / u
    return branch_trace, mp.mpf(1), multiplier_trace_and_norm


def descend_lambda_degree3(u: mp.mpf, dps: int) -> mp.mpf:
    """Return lambda(3*tau) from u=lambda(tau) on the small real branch.

    Eliminating a=(u*v)^(1/4) from
        u*v=a^4,  u+v=2a(2-3a+2a^2)
    gives this quartic modular polynomial.  At the cusp v~u^3/256.
    """
    u = +u
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    coefficient3 = -256 * u3 + 384 * u2 - 132 * u
    coefficient2 = 384 * u3 - 762 * u2 + 384 * u
    coefficient1 = -132 * u3 + 384 * u2 - 256 * u

    def equation(v):
        value = (
            ((v + coefficient3) * v + coefficient2) * v + coefficient1
        ) * v + u4
        derivative = (
            (4 * v + 3 * coefficient3) * v + 2 * coefficient2
        ) * v + coefficient1
        return value, derivative

    return _newton_staged(lambda: u3 / 256, equation, dps)


def descend_lambda_degree3_normalized_data(u: mp.mpf, dps: int):
    """Return (v,r) for the normalized degree-3 descent v=u^3*r/256.

    The polynomial for r tends to 1-r at the cusp, keeping the selected root
    O(1) even when v itself is exponentially small.
    """
    u = +u
    u2 = u * u
    u3 = u2 * u
    coefficient1 = -(33 * u2 - 96 * u + 64) / 64
    coefficient2 = 3 * u3 * (64 * u2 - 127 * u + 64) / 32768
    coefficient3 = -u3 * u3 * (64 * u2 - 96 * u + 33) / 4194304
    coefficient4 = u2**4 / 4294967296

    def equation(r):
        value = (
            ((coefficient4 * r + coefficient3) * r + coefficient2) * r
            + coefficient1
        ) * r + 1
        derivative = (
            (4 * coefficient4 * r + 3 * coefficient3) * r
            + 2 * coefficient2
        ) * r + coefficient1
        return value, derivative

    ratio = _newton_staged(lambda: mp.mpf(1), equation, dps)
    return u3 * ratio / 256, ratio


def descend_lambda_degree3_normalized(u: mp.mpf, dps: int) -> mp.mpf:
    """Degree-3 descent solved in the normalized coordinate v=u^3*r/256."""
    return descend_lambda_degree3_normalized_data(u, dps)[0]


def lambda3_period_data(u: mp.mpf, v: mp.mpf):
    """Return (v', v'', M, M', alpha_correction) for P_3(u,v)=0.

    M=(K(u)/K(v))^2 is recovered algebraically from the isogeny derivative;
    no elliptic integral is evaluated.
    """
    u2 = u * u
    u3 = u2 * u
    v2 = v * v
    v3 = v2 * v

    pu = 4 * (
        u3
        + v3 * (-192 * u2 + 192 * u - 33)
        + v2 * (288 * u2 - 381 * u + 96)
        + v * (-99 * u2 + 192 * u - 64)
    )
    pv = (
        -132 * u3
        + 384 * u2
        - 256 * u
        + 4 * v3
        - 4 * v2 * (192 * u3 - 288 * u2 + 99 * u)
        - 4 * v * (-192 * u3 + 381 * u2 - 192 * u)
    )
    puu = 12 * (
        u2
        + v3 * (64 - 128 * u)
        + v2 * (192 * u - 127)
        + v * (64 - 66 * u)
    )
    puv = (
        -396 * u2
        + 768 * u
        - 256
        - 4 * v2 * (576 * u2 - 576 * u + 99)
        - 4 * v * (-576 * u2 + 762 * u - 192)
    )
    pvv = (
        768 * u3
        - 1524 * u2
        + 768 * u
        + 12 * v2
        - 12 * v * (128 * u3 - 192 * u2 + 66 * u)
    )

    first = -pu / pv
    second = -(puu + 2 * puv * first + pvv * first * first) / pv
    degree = mp.mpf(3)
    multiplier = degree * v * (1 - v) / (u * (1 - u) * first)
    log_derivative = (
        first * (1 - 2 * v) / (v * (1 - v))
        - (1 - 2 * u) / (u * (1 - u))
        - second / first
    )
    multiplier_derivative = multiplier * log_derivative
    correction = (
        degree * v
        - multiplier * u
        + u * (1 - u) * multiplier_derivative
    )
    return first, second, multiplier, multiplier_derivative, correction


def pi_lambda3_direct(dps: int, iterations: int | None = None) -> mp.mpf:
    """Direct degree-3 Legendre-lambda/alpha iteration from the collision curve."""
    mp.mp.dps = dps + 30
    bits = math.ceil(dps * math.log2(10))
    count = iterations if iterations is not None else cubic_iterations(bits)
    u = mp.mpf(1) / 2
    alpha = mp.mpf(1) / 2  # alpha(1), from Legendre's relation at lambda=1/2.
    sqrt_r = mp.mpf(1)
    for _ in range(count):
        v = descend_lambda_degree3(u, dps + 30)
        _, _, multiplier, _, correction = lambda3_period_data(u, v)
        alpha = multiplier * alpha + sqrt_r * correction
        u = v
        sqrt_r *= 3
    result = 1 / alpha
    mp.mp.dps = dps
    return +result


def terminal_period_series(u: mp.mpf, dps: int):
    """Return F, G=F^2, and h=G'/G at a small Legendre lambda u."""
    with mp.workdps(dps + 20):
        tolerance = mp.power(10, -(dps + 10))
        term = mp.mpf(1)
        function = mp.mpf(1)
        derivative = mp.mpf(0)
        k = 0
        while True:
            k += 1
            term *= ((mp.mpf(k) - mp.mpf("0.5")) / k) ** 2 * u
            function += term
            derivative_term = k * term / u
            derivative += derivative_term
            if abs(derivative_term) < tolerance:
                break
        square = function * function
        logarithmic_derivative = 2 * derivative / function
        return +function, +square, +logarithmic_derivative


def pi_lambda3_period_transport(dps: int, iterations: int | None = None) -> mp.mpf:
    """Direct degree-3 pi iteration with terminal period reconstruction.

    This eliminates the forward alpha/z state.  G and its logarithmic
    derivative are transported backward from the deep cusp using only the
    modular polynomial and its first two implicit derivatives.
    """
    mp.mp.dps = dps + 30
    bits = math.ceil(dps * math.log2(10))
    count = iterations if iterations is not None else cubic_iterations(bits)
    u = mp.mpf(1) / 2
    r_product = mp.mpf(1)
    q_product = mp.mpf(1)
    additive = mp.mpf(0)

    for _ in range(count):
        v = descend_lambda_degree3_normalized(u, dps + 30)
        first, second, _, _, _ = lambda3_period_data(u, v)
        period_ratio = first * u * (1 - u) / (3 * v * (1 - v))
        log_ratio_derivative = (
            second / first
            + (1 - 2 * u) / (u * (1 - u))
            - first * (1 - 2 * v) / (v * (1 - v))
        )
        additive -= q_product * log_ratio_derivative
        q_product *= first
        r_product *= period_ratio
        u = v

    _, terminal_g, terminal_h = terminal_period_series(u, dps + 30)
    initial_g = terminal_g / r_product
    initial_h = q_product * terminal_h + additive
    result = 4 / (initial_g * initial_h)
    mp.mp.dps = dps
    return +result


def pi_lambda3_composite_jet(dps: int, iterations: int | None = None) -> mp.mpf:
    """Product-free period reconstruction from the two-jet of f_3 composed N times."""
    mp.mp.dps = dps + 30
    bits = math.ceil(dps * math.log2(10))
    count = iterations if iterations is not None else cubic_iterations(bits)
    u = mp.mpf(1) / 2
    first_composite = mp.mpf(1)
    second_composite = mp.mpf(0)
    degree_power = mp.mpf(1)

    for _ in range(count):
        v = descend_lambda_degree3_normalized(u, dps + 30)
        first, second, _, _, _ = lambda3_period_data(u, v)
        second_composite = (
            second * first_composite * first_composite
            + first * second_composite
        )
        first_composite *= first
        degree_power *= 3
        u = v

    _, terminal_g, terminal_h = terminal_period_series(u, dps + 30)
    endpoint_logistic_derivative = (1 - 2 * u) / (u * (1 - u))
    initial_h = (
        first_composite * (terminal_h + endpoint_logistic_derivative)
        - second_composite / first_composite
    )
    initial_g = (
        4 * terminal_g * degree_power * u * (1 - u) / first_composite
    )
    result = 4 / (initial_g * initial_h)
    mp.mp.dps = dps
    return +result


def pi_lambda3_terminal_nome(dps: int, iterations: int | None = None) -> mp.mpf:
    """Derivative- and period-free extraction through the terminal elliptic nome.

    Since lambda(q)=16q(1+O(q)), replacing q(u_N) by u_N/16 has an
    exponentially smaller error than the requested precision with the guarded
    iteration count used here.
    """
    mp.mp.dps = dps + 30
    if iterations is None:
        # pi*3^N/log(10) controls the decimal exponent of u_N.  The elementary
        # bounds pi>3 and log(10)<7/3 give log(10)/pi<7/9; use 4/5.
        count = max(1, math.ceil(math.log((dps + 30) * 0.8, 3)))
    else:
        count = iterations
    u = mp.mpf(1) / 2
    for _ in range(count):
        u = descend_lambda_degree3_normalized(u, dps + 30)
    result = -mp.log(u / 16) / (mp.mpf(3) ** count)
    mp.mp.dps = dps
    return +result


def pi_lambda3_incremental_nome(dps: int, iterations: int | None = None) -> mp.mpf:
    """Incremental logarithmic nome correction along normalized descents.

    If v=u^3*r/256, then L(v)=3L(u)-log(r) for L(u)=-log(u/16).
    Since q(lambda(i))=exp(-pi), taking the cusp limit gives

        pi = log(32) - sum_j log(r_j)/3^(j+1).

    This prototype intentionally evaluates every retained correction at the
    target precision; it measures whether the identity improves the practical
    computation graph before attempting a relaxed-precision implementation.
    """
    mp.mp.dps = dps + 30
    if iterations is None:
        count = max(1, math.ceil(math.log((dps + 30) * 0.8, 3)))
    else:
        count = iterations
    u = mp.mpf(1) / 2
    inverse_weight = mp.mpf(1) / 3
    result = 5 * mp.log(2)
    for _ in range(count):
        u, ratio = descend_lambda_degree3_normalized_data(u, dps + 30)
        result -= mp.log(ratio) * inverse_weight
        inverse_weight /= 3
    mp.mp.dps = dps
    return +result


def quartic_small_root(y: mp.mpf, dps: int) -> mp.mpf:
    """Radical-free quartic Landen step.

    Stable polynomial:
        y^4(1+x)^4 - 8x(1+x^2) = 0.
    This equals (1-x)^4-(1-y^4)(1+x)^4=0 after expansion.
    """
    y = +y
    y4 = y**4

    def equation(x):
        one_plus = 1 + x
        value = y4 * one_plus**4 - 8 * x * (1 + x * x)
        derivative = 4 * y4 * one_plus**3 - 8 * (1 + 3 * x * x)
        return value, derivative

    return _newton_staged(lambda: y4 / 8, equation, dps)


def cubic_small_root(s: mp.mpf, dps: int) -> mp.mpf:
    """Radical-free cubic Landen step.

    Stable polynomial:
        s^3(1+2x)^3 - 9x(1+x+x^2) = 0.
    """
    s = +s
    s3 = s**3

    def equation(x):
        one_plus_2x = 1 + 2 * x
        value = s3 * one_plus_2x**3 - 9 * x * (1 + x + x * x)
        derivative = 6 * s3 * one_plus_2x**2 - 9 * (1 + 2 * x + 3 * x * x)
        return value, derivative

    return _newton_staged(lambda: s3 / 9, equation, dps)


def quartic_iterations(bits: int) -> int:
    # The asymptotic error is about exp(-2*pi*4^n); retain one guard iterate.
    return max(1, math.ceil(math.log(max(bits, 2) / 8, 4)) + 1)


def cubic_iterations(bits: int) -> int:
    return max(1, math.ceil(math.log(max(bits, 2) / 5, 3)) + 1)


def pi_quartic_implicit(dps: int, iterations: int | None = None) -> mp.mpf:
    """Borwein quartic pi iteration with polynomial root selection only."""
    mp.mp.dps = dps + 20
    bits = math.ceil(dps * math.log2(10))
    count = iterations if iterations is not None else quartic_iterations(bits)
    y = positive_root_y0(dps + 20)
    z = 2 * y * y
    for n in range(count):
        y = quartic_small_root(y, dps + 20)
        z = z * (1 + y) ** 4 - mp.mpf(2) ** (2 * n + 3) * y * (1 + y + y * y)
    result = 1 / z
    mp.mp.dps = dps
    return +result


def pi_quartic_radical(dps: int, iterations: int | None = None) -> mp.mpf:
    """Classical explicit-radical quartic iteration, used as an oracle."""
    mp.mp.dps = dps + 20
    bits = math.ceil(dps * math.log2(10))
    count = iterations if iterations is not None else quartic_iterations(bits)
    y = mp.sqrt(2) - 1
    z = 6 - 4 * mp.sqrt(2)
    for n in range(count):
        root = mp.root(1 - y**4, 4)
        y = (1 - root) / (1 + root)
        z = z * (1 + y) ** 4 - mp.mpf(2) ** (2 * n + 3) * y * (1 + y + y * y)
    result = 1 / z
    mp.mp.dps = dps
    return +result


def pi_lemniscatic_series(dps: int, terms: int | None = None) -> mp.mpf:
    """Conductor-two Q(i) CM series at tau=2i, where j=66^3."""

    mp.mp.dps = dps + 20
    if terms is None:
        bits = math.ceil((dps + 20) * math.log2(10))
        bits_per_term = math.log2(mp.mpf(66) ** 3 / 1728)
        terms = math.ceil(bits / bits_per_term) + 2
    coefficient = mp.mpf(1)
    total = mp.mpf(5)
    denominator = mp.mpf(66) ** 3
    for n in range(1, terms):
        coefficient *= (
            24 * (6 * n - 5) * (2 * n - 1) * (6 * n - 1)
            / (mp.mpf(n) ** 3 * denominator)
        )
        total += coefficient * (63 * n + 5)
    result = 11 * mp.sqrt(33) / (4 * total)
    mp.mp.dps = dps
    return +result


def pi_cubic_implicit(dps: int, iterations: int | None = None) -> mp.mpf:
    """Borwein cubic pi iteration with polynomial root selection only."""
    mp.mp.dps = dps + 20
    bits = math.ceil(dps * math.log2(10))
    count = iterations if iterations is not None else cubic_iterations(bits)
    a = mp.mpf(1) / 3
    s = positive_root_s0(dps + 20)
    power = mp.mpf(1)
    for _ in range(count):
        s = cubic_small_root(s, dps + 20)
        r = 1 + 2 * s
        r2 = r * r
        a = r2 * a - power * (r2 - 1)
        power *= 3
    result = 1 / a
    mp.mp.dps = dps
    return +result


def correct_bits(value: mp.mpf) -> float:
    with mp.workdps(mp.mp.dps + 30):
        error = abs(value - mp.pi)
        return math.inf if not error else float(-mp.log(error, 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--digits", type=int, default=1000)
    args = parser.parse_args()
    results = {}
    for name, function in (
        ("quartic_implicit", pi_quartic_implicit),
        ("quartic_radical", pi_quartic_radical),
        ("cubic_implicit", pi_cubic_implicit),
        ("lambda3_direct", pi_lambda3_direct),
        ("lambda3_period", pi_lambda3_period_transport),
        ("lambda3_jet", pi_lambda3_composite_jet),
        ("lambda3_nome", pi_lambda3_terminal_nome),
    ):
        started = time.perf_counter()
        value = function(args.digits + 20)
        elapsed = time.perf_counter() - started
        results[name] = {"seconds": elapsed, "correct_bits": correct_bits(value)}
    print(json.dumps({"digits": args.digits, "methods": results}, indent=2))


if __name__ == "__main__":
    main()
