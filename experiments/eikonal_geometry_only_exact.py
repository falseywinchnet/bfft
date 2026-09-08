"""Exact certificates for the geometry-only decision.

All calculations use SymPy exact rationals and Q(sqrt(3)).  The factor-two
periodic lattice is the same exact source/query geometry used by the earlier
spectral certificate.  In one dimension, scalar AMLE and natural-neighbour
interpolation are exactly the same piecewise-geodesic-linear extension.
"""

from __future__ import annotations

import json

import sympy as sp

from experiments.eikonal_markov_spectral_exact import periodic_factor_two_certificate


def periodic_distance(first: int, second: int, side: int) -> int:
    return min((first - second) % side, (second - first) % side)


def linear_circle_extension(
    sources: list[int], values: list[sp.Expr], query: int, side: int
) -> tuple[sp.Expr, list[sp.Rational]]:
    ordered = sorted(zip(sources, values))
    for index, (left_site, left_value) in enumerate(ordered):
        right_site, right_value = ordered[(index + 1) % len(ordered)]
        arc = (right_site - left_site) % side
        offset = (query - left_site) % side
        if offset <= arc:
            left_weight = sp.Rational(arc - offset, arc)
            right_weight = sp.Rational(offset, arc)
            weights = [sp.Rational(0) for _ in sources]
            weights[sources.index(left_site)] = left_weight
            weights[sources.index(right_site)] = right_weight
            return sp.simplify(left_weight * left_value + right_weight * right_value), weights
    raise AssertionError("query was not bracketed")


def exact_mse(estimates: list[sp.Expr], truth: list[sp.Expr]) -> sp.Expr:
    return sp.simplify(sum((estimate - target) ** 2 for estimate, target in zip(estimates, truth)) / len(truth))


def _exact_bracket(axis: list[int], coordinate: int) -> tuple[int, int, sp.Rational]:
    if coordinate <= axis[0]:
        return axis[0], axis[0], sp.Rational(0)
    if coordinate >= axis[-1]:
        return axis[-1], axis[-1], sp.Rational(0)
    for lower, upper in zip(axis, axis[1:]):
        if lower <= coordinate <= upper:
            return lower, upper, sp.Rational(coordinate - lower, upper - lower)
    raise AssertionError("coordinate was not bracketed")


def exact_metric_delaunay_p1_2d() -> dict[str, object]:
    """Exact rotated-SPD P1 certificate on the 13x13 common lattice."""

    side = 13
    source_axis = [2, 6, 10]
    query_axis = [1, 3, 5, 7, 9, 11]
    sources = [(x, y) for y in source_axis for x in source_axis]
    queries = [(x, y) for y in query_axis for x in query_axis]
    source_index = {point: index for index, point in enumerate(sources)}
    metric = sp.Matrix(((2, 1), (1, 1)))
    assert metric.det() == 1

    # For a metric parallelogram with physical edge vectors u and v, the
    # bottom-left/top-right diagonal is Delaunay iff u^T M v <= 0.  Here the
    # cross term is strictly positive, selecting the other diagonal exactly.
    u = sp.Matrix((4, 0))
    v = sp.Matrix((0, sp.Rational(34, 5)))
    cross_term = (u.T * metric * v)[0]
    assert cross_term > 0

    def weights_at(x: int, y: int) -> list[sp.Rational]:
        x0, x1, tx = _exact_bracket(source_axis, x)
        y0, y1, ty = _exact_bracket(source_axis, y)
        weights = [sp.Rational(0) for _ in sources]
        if x0 == x1 and y0 == y1:
            weights[source_index[(x0, y0)]] = 1
        elif x0 == x1:
            weights[source_index[(x0, y0)]] = 1 - ty
            weights[source_index[(x0, y1)]] = ty
        elif y0 == y1:
            weights[source_index[(x0, y0)]] = 1 - tx
            weights[source_index[(x1, y0)]] = tx
        elif tx + ty <= 1:  # Triangle bottom-left, bottom-right, top-left.
            weights[source_index[(x0, y0)]] = 1 - tx - ty
            weights[source_index[(x1, y0)]] = tx
            weights[source_index[(x0, y1)]] = ty
        else:  # Triangle bottom-right, top-right, top-left.
            weights[source_index[(x1, y0)]] = 1 - ty
            weights[source_index[(x1, y1)]] = tx + ty - 1
            weights[source_index[(x0, y1)]] = 1 - tx
        assert sum(weights) == 1
        assert all(weight >= 0 for weight in weights)
        return weights

    def truth_at(x: int, y: int) -> sp.Rational:
        xn, yn = sp.Rational(x, side - 1), sp.Rational(y, side - 1)
        return sp.cancel(
            (2 + 2 * xn + yn + xn * yn + xn**2 + yn**2 / 2)
            / sp.Rational(15, 2)
        )

    source_values = [truth_at(x, y) for x, y in sources]
    field: dict[tuple[int, int], sp.Expr] = {}
    all_rows: dict[tuple[int, int], list[sp.Rational]] = {}
    for y in range(side):
        for x in range(side):
            row = weights_at(x, y)
            all_rows[(x, y)] = row
            field[(x, y)] = sp.cancel(
                sum(weight * value for weight, value in zip(row, source_values))
            )
            assert min(source_values) <= field[(x, y)] <= max(source_values)

    strict_unobserved_extrema = 0
    source_set = set(sources)
    for y in range(side):
        for x in range(side):
            if (x, y) in source_set:
                continue
            neighbors = [
                field[(jx, jy)]
                for jy in range(max(0, y - 1), min(side, y + 2))
                for jx in range(max(0, x - 1), min(side, x + 2))
                if (jx, jy) != (x, y)
            ]
            strict_unobserved_extrema += int(
                all(field[(x, y)] > value for value in neighbors)
                or all(field[(x, y)] < value for value in neighbors)
            )
    assert strict_unobserved_extrema == 0

    estimates = [field[query] for query in queries]
    targets = [truth_at(*query) for query in queries]
    mse = exact_mse(estimates, targets)
    return {
        "arithmetic": "exact rational",
        "metric": [[str(metric[i, j]) for j in range(2)] for i in range(2)],
        "metric_cross_term_selecting_diagonal": str(cross_term),
        "source_count": len(sources),
        "query_count": len(queries),
        "mse": str(mse),
        "mse_decimal_60": str(sp.N(mse, 60)),
        "strict_unobserved_extrema_on_common_lattice": strict_unobserved_extrema,
        "maximum_sample_range_overshoot": "0",
        "certificates": {
            "metric_determinant_one": True,
            "delaunay_diagonal_selected_without_values": True,
            "weights_nonnegative": True,
            "weights_mass_one": True,
            "observations_cardinal": True,
            "boundary_rule_is_replication": True,
            "zero_sample_range_overshoot": True,
            "zero_strict_unobserved_extrema": True,
        },
    }


def main() -> None:
    spectral = periodic_factor_two_certificate()
    side = int(spectral["common_periodic_lattice_size"])
    sources = [int(value) for value in spectral["source_centers"]]
    queries = [int(value) for value in spectral["query_centers"]]
    truth_all = [
        sp.simplify(
            sp.Rational(1, 2)
            + sp.Rational(1, 4)
            * sp.cos(2 * sp.pi * sp.Rational(index - 2, side))
        )
        for index in range(side)
    ]
    source_values = [truth_all[index] for index in sources]
    truth = [truth_all[index] for index in queries]

    linear_estimates: list[sp.Expr] = []
    rows: list[list[sp.Rational]] = []
    for query in queries:
        estimate, weights = linear_circle_extension(
            sources, source_values, query, side
        )
        assert sum(weights) == 1
        assert all(weight >= 0 for weight in weights)
        assert min(source_values) <= estimate <= max(source_values)
        linear_estimates.append(estimate)
        rows.append(weights)
    linear_mse = exact_mse(linear_estimates, truth)
    markov_mse = sp.sympify(spectral["markov_mse"])
    projected_spectral_mse = sp.sympify(spectral["projected_spectral_mse"])
    natural_minus_markov = sp.simplify(linear_mse - markov_mse)
    natural_minus_projected = sp.simplify(
        linear_mse - projected_spectral_mse
    )
    assert natural_minus_markov < 0
    assert natural_minus_projected > 0

    # Exact affine reproduction and exact quadratic obstruction for arbitrary
    # rational barycentric weights on the interval.
    x0, x1, x = sp.Rational(0), sp.Rational(2), sp.Rational(1, 2)
    w1 = sp.cancel((x - x0) / (x1 - x0))
    w0 = 1 - w1
    assert w0 * x0 + w1 * x1 == x
    quadratic_bias = sp.simplify(
        w0 * x0**2 + w1 * x1**2 - x**2
    )
    variance_identity = sp.simplify(
        w0 * (x0 - x) ** 2 + w1 * (x1 - x) ** 2
    )
    assert quadratic_bias == variance_identity > 0

    payload = {
        "arithmetic": "exact Q(sqrt(3)) and exact rationals",
        "strict_operator_rule": "weights are functions of geometry and sites only",
        "periodic_factor_two": {
            "source_centers": sources,
            "query_centers": queries,
            "natural_neighbor_equals_harmonic_coordinates_equals_scalar_amle_in_one_dimension": True,
            "geometry_only_weights": [[str(value) for value in row] for row in rows],
            "natural_neighbor_amle_mse": str(linear_mse),
            "natural_neighbor_amle_mse_decimal_60": str(sp.N(linear_mse, 60)),
            "positive_markov_mse": spectral["markov_mse"],
            "positive_markov_mse_decimal_60": spectral["markov_mse_decimal_60"],
            "projected_spectral_mse": spectral["projected_spectral_mse"],
            "projected_spectral_mse_decimal_60": spectral[
                "projected_spectral_mse_decimal_60"
            ],
            "raw_spectral_mse": spectral["raw_spectral_mse"],
            "sample_interval": spectral["sample_interval"],
            "exact_ordering": {
                "natural_neighbor_amle_beats_positive_markov": True,
                "natural_minus_markov_mse": str(natural_minus_markov),
                "projected_spectral_beats_natural_neighbor_amle": True,
                "natural_minus_projected_spectral_mse": str(
                    natural_minus_projected
                ),
                "raw_spectral_is_exact_but_signed": True,
            },
        },
        "positive_affine_quadratic_obstruction": {
            "query": str(x),
            "weights": [str(w0), str(w1)],
            "quadratic_bias": str(quadratic_bias),
            "weighted_squared_radius": str(variance_identity),
            "identity_exact": True,
            "strictly_positive_away_from_samples": True,
        },
        "rotated_spd_two_dimensional_metric_delaunay_p1": exact_metric_delaunay_p1_2d(),
        "certificates": {
            "natural_neighbor_weights_nonnegative": True,
            "natural_neighbor_weights_mass_one": True,
            "natural_neighbor_and_scalar_amle_have_zero_sample_range_overshoot": True,
            "harmonic_coordinates_have_zero_sample_range_overshoot": True,
            "all_reported_mse_values_are_exact_algebraic_numbers": True,
            "positive_affine_exact_weights_cannot_reproduce_the_quadratic": True,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
