"""Exact certificates for ownerwise Riemannian harmonic cell interpolation.

The cell basis is the exact solution of a rational, monotone directional
finite-difference Dirichlet problem.  It is a discrete realization of

    -div(A grad(lambda_i)) = 0,   A = g^{-1},

with piecewise-linear vertex-hat boundary data.  All reported values are exact
SymPy rationals or exact expressions in Q(sqrt(3)); floating point is never
used to establish a claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import sympy as sp

from experiments.eikonal_geometry_only_exact import (
    exact_metric_delaunay_p1_2d,
    exact_mse,
)
from experiments.eikonal_markov_spectral_exact import periodic_factor_two_certificate


Point = tuple[int, int]


@dataclass(frozen=True)
class DirectionalStencil:
    horizontal: sp.Rational
    vertical: sp.Rational
    diagonal: sp.Rational
    diagonal_direction: Point


def monotone_directional_stencil(
    diffusion: sp.Matrix,
) -> DirectionalStencil:
    """Decompose a diagonally dominant 2-D SPD diffusion tensor exactly."""

    if diffusion != diffusion.T or diffusion.det() <= 0:
        raise ValueError("diffusion must be SPD")
    cross = sp.Rational(diffusion[0, 1])
    diagonal = abs(cross)
    horizontal = sp.Rational(diffusion[0, 0]) - diagonal
    vertical = sp.Rational(diffusion[1, 1]) - diagonal
    if horizontal < 0 or vertical < 0:
        raise ValueError("this exact stencil requires diagonal dominance")
    direction = (1, 1) if cross >= 0 else (1, -1)
    return DirectionalStencil(horizontal, vertical, diagonal, direction)


def boundary_hat_weights(point: Point, side: int) -> tuple[sp.Rational, ...]:
    """Exact BL, BR, TR, TL vertex hats on a square boundary."""

    x, y = point
    if not (x in (0, side) or y in (0, side)):
        raise ValueError("point is not on the cell boundary")
    tx, ty = sp.Rational(x, side), sp.Rational(y, side)
    if y == 0:
        return 1 - tx, tx, sp.Rational(0), sp.Rational(0)
    if x == side:
        return sp.Rational(0), 1 - ty, ty, sp.Rational(0)
    if y == side:
        return sp.Rational(0), sp.Rational(0), tx, 1 - tx
    return 1 - ty, sp.Rational(0), sp.Rational(0), ty


def exact_harmonic_cell_weights(
    diffusion: sp.Matrix, side: int = 4
) -> dict[Point, tuple[sp.Expr, ...]]:
    """Solve the monotone cell Dirichlet systems in exact arithmetic."""

    stencil = monotone_directional_stencil(diffusion)
    interior = [
        (x, y)
        for y in range(1, side)
        for x in range(1, side)
    ]
    row_of = {point: row for row, point in enumerate(interior)}
    matrix = sp.zeros(len(interior), len(interior))
    rhs = [sp.zeros(len(interior), 1) for _ in range(4)]
    directions = (
        ((1, 0), stencil.horizontal),
        ((0, 1), stencil.vertical),
        (stencil.diagonal_direction, stencil.diagonal),
    )
    center = 2 * sum(weight for _, weight in directions)
    if center <= 0:
        raise ValueError("stencil has no positive direction")

    for point, row in row_of.items():
        matrix[row, row] = center
        for (dx, dy), coefficient in directions:
            if coefficient == 0:
                continue
            for sign in (-1, 1):
                neighbor = (point[0] + sign * dx, point[1] + sign * dy)
                if neighbor in row_of:
                    matrix[row, row_of[neighbor]] -= coefficient
                else:
                    boundary = boundary_hat_weights(neighbor, side)
                    for basis in range(4):
                        rhs[basis][row] += coefficient * boundary[basis]

    assert matrix == matrix.T
    assert all(matrix[i, i] > 0 for i in range(matrix.rows))
    solutions = [matrix.inv() * vector for vector in rhs]
    weights: dict[Point, tuple[sp.Expr, ...]] = {}
    for y in range(side + 1):
        for x in range(side + 1):
            point = (x, y)
            if point in row_of:
                values = tuple(
                    sp.factor(solution[row_of[point]]) for solution in solutions
                )
            else:
                values = boundary_hat_weights(point, side)
            assert sp.simplify(sum(values)) == 1
            assert all(value >= 0 for value in values)
            weights[point] = values
    return weights


def exact_directional_energy(
    field: dict[Point, sp.Expr], stencil: DirectionalStencil, side: int
) -> sp.Expr:
    """Exact graph Dirichlet energy, counting every undirected edge once."""

    directions = (
        ((1, 0), stencil.horizontal),
        ((0, 1), stencil.vertical),
        (stencil.diagonal_direction, stencil.diagonal),
    )
    energy = sp.Rational(0)
    for y in range(side + 1):
        for x in range(side + 1):
            for (dx, dy), coefficient in directions:
                neighbor = (x + dx, y + dy)
                if coefficient and 0 <= neighbor[0] <= side and 0 <= neighbor[1] <= side:
                    energy += coefficient * (field[(x, y)] - field[neighbor]) ** 2
    return sp.factor(energy)


def exact_cell_energy_certificate(
    diffusion: sp.Matrix, side: int = 4
) -> dict[str, object]:
    """Compare harmonic, Delaunay-P1, and Q1 energy for one vertex hat."""

    stencil = monotone_directional_stencil(diffusion)
    weights = exact_harmonic_cell_weights(diffusion, side)
    harmonic = {point: row[2] for point, row in weights.items()}
    p1 = {
        (x, y): (
            sp.Rational(0)
            if x + y <= side
            else sp.Rational(x + y - side, side)
        )
        for y in range(side + 1)
        for x in range(side + 1)
    }
    q1 = {
        (x, y): sp.Rational(x * y, side**2)
        for y in range(side + 1)
        for x in range(side + 1)
    }
    harmonic_energy = exact_directional_energy(harmonic, stencil, side)
    p1_energy = exact_directional_energy(p1, stencil, side)
    q1_energy = exact_directional_energy(q1, stencil, side)
    assert harmonic_energy < p1_energy
    assert harmonic_energy < q1_energy
    return {
        "vertex_data": ["0", "0", "1", "0"],
        "harmonic_energy": str(harmonic_energy),
        "metric_delaunay_p1_energy": str(p1_energy),
        "q1_energy": str(q1_energy),
        "harmonic_strictly_minimal_against_both": True,
        "formal_reason": "unique minimizer of the strictly convex Dirichlet energy with fixed boundary hats",
    }


def _bracket(axis: list[int], coordinate: int) -> tuple[int, int, int]:
    clamped = min(max(coordinate, axis[0]), axis[-1])
    if clamped == axis[-1]:
        return axis[-2], axis[-1], clamped
    for lower, upper in zip(axis, axis[1:]):
        if lower <= clamped <= upper:
            return lower, upper, clamped
    raise AssertionError("coordinate was not bracketed")


def exact_two_dimensional_certificate() -> dict[str, object]:
    """Certify the harmonic-cell law on the existing rotated-SPD problem."""

    side = 13
    source_axis = [2, 6, 10]
    query_axis = [1, 3, 5, 7, 9, 11]
    sources = [(x, y) for y in source_axis for x in source_axis]
    source_index = {point: index for index, point in enumerate(sources)}
    queries = [(x, y) for y in query_axis for x in query_axis]

    metric = sp.Matrix(((2, 1), (1, 1)))
    spacing = sp.diag(1, sp.Rational(17, 10))
    # A lattice-coordinate scalar multiple of G^{-1}; the discarded positive
    # determinant factor does not change the homogeneous Dirichlet equation.
    diffusion = sp.simplify(spacing.inv() * metric.inv() * spacing.inv().T)
    stencil = monotone_directional_stencil(diffusion)
    cell_weights = exact_harmonic_cell_weights(diffusion, side=4)

    def truth_at(x: int, y: int) -> sp.Rational:
        xn, yn = sp.Rational(x, side - 1), sp.Rational(y, side - 1)
        return sp.cancel(
            (2 + 2 * xn + yn + xn * yn + xn**2 + yn**2 / 2)
            / sp.Rational(15, 2)
        )

    source_values = [truth_at(x, y) for x, y in sources]

    def global_row(x: int, y: int) -> list[sp.Expr]:
        x0, x1, xc = _bracket(source_axis, x)
        y0, y1, yc = _bracket(source_axis, y)
        local = cell_weights[(xc - x0, yc - y0)]
        corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        row = [sp.Rational(0) for _ in sources]
        for coefficient, corner in zip(local, corners):
            row[source_index[corner]] += coefficient
        assert sp.simplify(sum(row)) == 1
        assert all(value >= 0 for value in row)
        return row

    field: dict[Point, sp.Expr] = {}
    for y in range(side):
        for x in range(side):
            row = global_row(x, y)
            field[(x, y)] = sp.factor(
                sum(weight * value for weight, value in zip(row, source_values))
            )
            assert min(source_values) <= field[(x, y)] <= max(source_values)
            if (x, y) in source_index:
                assert field[(x, y)] == source_values[source_index[(x, y)]]

    strict_unobserved_extrema = 0
    for y in range(side):
        for x in range(side):
            if (x, y) in source_index:
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
    truth = [truth_at(*query) for query in queries]
    mse = exact_mse(estimates, truth)
    p1_mse = sp.Rational(exact_metric_delaunay_p1_2d()["mse"])
    return {
        "arithmetic": "exact rational",
        "metric": [[str(metric[i, j]) for j in range(2)] for i in range(2)],
        "spacing": [["1", "0"], ["0", "17/10"]],
        "lattice_diffusion": [
            [str(diffusion[i, j]) for j in range(2)] for i in range(2)
        ],
        "monotone_stencil": {
            "horizontal": str(stencil.horizontal),
            "vertical": str(stencil.vertical),
            "diagonal": str(stencil.diagonal),
            "diagonal_direction": list(stencil.diagonal_direction),
        },
        "cell_energy": exact_cell_energy_certificate(diffusion, side=4),
        "source_count": len(sources),
        "query_count": len(queries),
        "mse": str(mse),
        "p1_mse": str(p1_mse),
        "harmonic_minus_p1_mse": str(sp.factor(mse - p1_mse)),
        "harmonic_beats_p1": bool(mse < p1_mse),
        "strict_unobserved_extrema_on_common_lattice": strict_unobserved_extrema,
        "maximum_sample_range_overshoot": "0",
        "certificates": {
            "weights_nonnegative": True,
            "weights_mass_one": True,
            "observations_cardinal": True,
            "boundary_rule_is_replication": True,
            "zero_sample_range_overshoot": True,
            "zero_strict_unobserved_extrema": True,
            "discrete_operator_is_symmetric_m_matrix": True,
        },
    }


def exact_fidelity_sweep() -> dict[str, object]:
    """Compare harmonic cells and metric-Delaunay P1 on exact truth families."""

    side = 13
    source_axis = [2, 6, 10]
    query_axis = [1, 3, 5, 7, 9, 11]
    sources = [(x, y) for y in source_axis for x in source_axis]
    queries = [(x, y) for y in query_axis for x in query_axis]
    source_index = {point: index for index, point in enumerate(sources)}
    metric = sp.Matrix(((2, 1), (1, 1)))
    spacing = sp.diag(1, sp.Rational(17, 10))
    diffusion = sp.simplify(spacing.inv() * metric.inv() * spacing.inv().T)
    harmonic_local = exact_harmonic_cell_weights(diffusion, side=4)

    def rows_at(x: int, y: int) -> tuple[list[sp.Expr], list[sp.Expr]]:
        x0, x1, xc = _bracket(source_axis, x)
        y0, y1, yc = _bracket(source_axis, y)
        corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        harmonic = [sp.Rational(0) for _ in sources]
        for coefficient, corner in zip(
            harmonic_local[(xc - x0, yc - y0)], corners
        ):
            harmonic[source_index[corner]] += coefficient

        tx = sp.Rational(xc - x0, x1 - x0)
        ty = sp.Rational(yc - y0, y1 - y0)
        if tx + ty <= 1:
            p1_local = (1 - tx - ty, tx, sp.Rational(0), ty)
        else:
            p1_local = (sp.Rational(0), 1 - ty, tx + ty - 1, 1 - tx)
        p1 = [sp.Rational(0) for _ in sources]
        for coefficient, corner in zip(p1_local, corners):
            p1[source_index[corner]] += coefficient
        assert sum(harmonic) == sum(p1) == 1
        assert all(value >= 0 for value in harmonic + p1)
        return harmonic, p1

    rows = [rows_at(*query) for query in queries]

    def coordinates(x: int, y: int) -> tuple[sp.Rational, sp.Rational]:
        return sp.Rational(x, side - 1), sp.Rational(y, side - 1)

    truth_functions = {
        "x_squared": lambda x, y: coordinates(x, y)[0] ** 2,
        "y_squared": lambda x, y: coordinates(x, y)[1] ** 2,
        "xy": lambda x, y: coordinates(x, y)[0] * coordinates(x, y)[1],
        "metric_harmonic_quadratic": lambda x, y: (
            coordinates(x, y)[0] * coordinates(x, y)[1]
            + sp.Rational(10, 17) * coordinates(x, y)[0] ** 2
        ),
        "convex_quadratic": lambda x, y: (
            coordinates(x, y)[0] ** 2 + coordinates(x, y)[1] ** 2
        ),
        "ramp_kink_x": lambda x, y: abs(coordinates(x, y)[0] - sp.Rational(1, 2)),
        "oblique_step": lambda x, y: sp.Rational(int(2 * x + y >= 18)),
        "periodic_product": lambda x, y: sp.simplify(
            sp.Rational(1, 2)
            + sp.Rational(1, 8)
            * sp.cos(2 * sp.pi * sp.Rational(x - 2, side - 1))
            * sp.cos(2 * sp.pi * sp.Rational(y - 2, side - 1))
        ),
    }
    results: dict[str, object] = {}
    harmonic_wins = 0
    p1_wins = 0
    ties = 0
    for name, truth_fn in truth_functions.items():
        source_values = [sp.simplify(truth_fn(*point)) for point in sources]
        targets = [sp.simplify(truth_fn(*point)) for point in queries]
        harmonic_estimates = [
            sp.simplify(sum(weight * value for weight, value in zip(row[0], source_values)))
            for row in rows
        ]
        p1_estimates = [
            sp.simplify(sum(weight * value for weight, value in zip(row[1], source_values)))
            for row in rows
        ]
        harmonic_mse = exact_mse(harmonic_estimates, targets)
        p1_mse = exact_mse(p1_estimates, targets)
        difference = sp.simplify(harmonic_mse - p1_mse)
        if difference < 0:
            winner = "harmonic"
            harmonic_wins += 1
        elif difference > 0:
            winner = "p1"
            p1_wins += 1
        else:
            winner = "tie"
            ties += 1
        results[name] = {
            "harmonic_mse": str(harmonic_mse),
            "p1_mse": str(p1_mse),
            "harmonic_minus_p1": str(difference),
            "winner": winner,
        }
    return {
        "arithmetic": "exact rationals and Q(sqrt(3))",
        "truth_is_analytic_or_exactly_declared": True,
        "cases": results,
        "summary": {
            "harmonic_wins": harmonic_wins,
            "p1_wins": p1_wins,
            "ties": ties,
        },
    }


def exact_round_trip_and_stability_certificate() -> dict[str, object]:
    """Exact shift, nested-grid, positive-optimality, and Lanczos facts."""

    alpha, omega = sp.symbols("alpha omega", real=True)
    p1_round_trip = sp.simplify(
        ((1 - alpha) + alpha * sp.exp(sp.I * omega))
        * ((1 - alpha) + alpha * sp.exp(-sp.I * omega))
    )
    expected = 1 - 4 * alpha * (1 - alpha) * sp.sin(omega / 2) ** 2
    assert sp.trigsimp(sp.expand_complex(p1_round_trip) - expected) == 0

    lanczos2 = tuple(
        sp.Rational(value, 16) for value in (-1, 9, 9, -1)
    )
    lanczos3 = tuple(
        sp.Rational(value, 368) for value in (9, -50, 225, 225, -50, 9)
    )
    assert sum(lanczos2) == 1
    assert sum(lanczos3) == 1
    assert sum(abs(value) for value in lanczos2) == sp.Rational(5, 4)
    assert sum(abs(value) for value in lanczos3) == sp.Rational(71, 46)

    # Exact minimal variance for positive affine-exact weights at alpha=1/2:
    # a symmetric distribution with any mass outside {0,1} has a larger
    # second moment.  The general theorem follows from the integer inequality
    # (k-floor(alpha))(k-ceil(alpha)) >= 0.
    half_variance = sp.Rational(1, 4)
    return {
        "arithmetic": "exact rational and symbolic trigonometric identities",
        "nested_grid_round_trip": {
            "identity": True,
            "reason": "coarse vertices are target vertices and the return operator is cardinal",
        },
        "rank_obstruction": {
            "perfect_arbitrary_downsample_round_trip_possible": False,
            "reason": "rank(W_BA W_AB) <= |B| < |A|",
        },
        "uniform_shift_p1": {
            "forward_symbol": "(1-alpha)+alpha*exp(i*omega)",
            "round_trip_symbol": "1-4*alpha*(1-alpha)*sin(omega/2)^2",
            "round_trip_interval_for_0_le_alpha_le_1": "[0,1]",
            "half_shift_variance": str(half_variance),
            "minimum_low_frequency_loss_among_positive_affine_exact_integer_stencils": True,
        },
        "induced_l_infinity_norms": {
            "positive_harmonic_or_p1": "1",
            "normalized_half_shift_lanczos2": "5/4",
            "normalized_half_shift_lanczos3": "71/46",
        },
        "normalized_half_shift_weights": {
            "lanczos2": [str(value) for value in lanczos2],
            "lanczos3": [str(value) for value in lanczos3],
        },
        "two_dimensional_quarter_shift": exact_two_dimensional_round_trip(),
        "one_dimensional_step_edge": exact_lanczos_step_certificate(),
    }


def exact_lanczos_step_certificate() -> dict[str, object]:
    """Exact half-shift plateau errors for normalized Lanczos-2 and -3."""

    lanczos2_errors = (sp.Rational(-1, 16), sp.Rational(1, 16))
    lanczos3_errors = (sp.Rational(-41, 368), sp.Rational(41, 368))
    p1_errors = (sp.Rational(0), sp.Rational(0))
    lanczos2_mse = sum(value**2 for value in lanczos2_errors) / 2
    lanczos3_mse = sum(value**2 for value in lanczos3_errors) / 2
    p1_mse = sum(value**2 for value in p1_errors) / 2
    assert lanczos2_mse == sp.Rational(1, 256)
    assert lanczos3_mse == sp.Rational(1681, 135424)
    assert p1_mse == 0
    return {
        "truth": "Heaviside(x-1/2), sampled at integers; queries x=-1/2 and x=3/2",
        "queries_are_strictly_inside_constant_plateaus": True,
        "harmonic_equals_p1_mse": str(p1_mse),
        "normalized_lanczos2_mse": str(lanczos2_mse),
        "normalized_lanczos3_mse": str(lanczos3_mse),
        "harmonic_equals_p1_maximum_overshoot": "0",
        "normalized_lanczos2_maximum_overshoot": "1/16",
        "normalized_lanczos3_maximum_overshoot": "41/368",
    }


def exact_two_dimensional_round_trip() -> dict[str, object]:
    """Exact Fourier gains for a quarter-cell shift in the rotated metric."""

    metric = sp.Matrix(((2, 1), (1, 1)))
    spacing = sp.diag(1, sp.Rational(17, 10))
    diffusion = sp.simplify(spacing.inv() * metric.inv() * spacing.inv().T)
    harmonic = exact_harmonic_cell_weights(diffusion, side=4)[(1, 1)]
    p1 = (
        sp.Rational(1, 2),
        sp.Rational(1, 4),
        sp.Rational(0),
        sp.Rational(1, 4),
    )

    def gain(weights: tuple[sp.Expr, ...], zx: sp.Expr, zy: sp.Expr) -> sp.Expr:
        symbol = sp.expand(
            weights[0]
            + weights[1] * zx
            + weights[2] * zx * zy
            + weights[3] * zy
        )
        return sp.simplify(sp.expand_complex(symbol * sp.conjugate(symbol)))

    modes = {
        "x_pi_over_2": (sp.I, sp.Rational(1)),
        "y_pi_over_2": (sp.Rational(1), sp.I),
        "same_diagonal_pi_over_2": (sp.I, sp.I),
        "opposite_diagonal_pi_over_2": (sp.I, -sp.I),
        "x_pi": (-sp.Rational(1), sp.Rational(1)),
        "y_pi": (sp.Rational(1), -sp.Rational(1)),
        "both_pi": (-sp.Rational(1), -sp.Rational(1)),
    }
    results: dict[str, object] = {}
    harmonic_higher = 0
    p1_higher = 0
    ties = 0
    for name, (zx, zy) in modes.items():
        harmonic_gain = gain(harmonic, zx, zy)
        p1_gain = gain(p1, zx, zy)
        assert 0 <= harmonic_gain <= 1
        assert 0 <= p1_gain <= 1
        difference = sp.factor(harmonic_gain - p1_gain)
        if difference > 0:
            winner = "harmonic"
            harmonic_higher += 1
        elif difference < 0:
            winner = "p1"
            p1_higher += 1
        else:
            winner = "tie"
            ties += 1
        results[name] = {
            "harmonic_round_trip_gain": str(harmonic_gain),
            "p1_round_trip_gain": str(p1_gain),
            "harmonic_minus_p1": str(difference),
            "higher_retention": winner,
        }
    return {
        "arithmetic": "exact rational and Gaussian rational",
        "shift": ["1/4", "1/4"],
        "harmonic_forward_weights": [str(value) for value in harmonic],
        "p1_forward_weights": [str(value) for value in p1],
        "both_are_probability_rows": True,
        "return_symbol_is_complex_conjugate_by_central_symmetry": True,
        "round_trip_gain_is_squared_symbol_magnitude": True,
        "modes": results,
        "summary": {
            "harmonic_higher_retention": harmonic_higher,
            "p1_higher_retention": p1_higher,
            "ties": ties,
        },
    }


def exact_metric_flip_certificate() -> dict[str, object]:
    """Expose the P1 flip and the harmonic operator's continuous limit."""

    # Unit-square data: only the top-right value is one.  At q=(1/4,1/4),
    # the two metric-Delaunay triangulations and the tie Q1 rule disagree.
    p1_first_diagonal = sp.Rational(1, 4)
    p1_second_diagonal = sp.Rational(0)
    q1_tie = sp.Rational(1, 16)
    assert len({p1_first_diagonal, p1_second_diagonal, q1_tie}) == 3

    # For M(eps)=[[1,eps],[eps,1]], a scalar multiple of inverse metric is
    # A(eps)=[[1,-eps],[-eps,1]].  The monotone discrete operator has axis
    # weights 1-|eps| and a diagonal weight |eps|.  Although its diagonal
    # direction changes, that term vanishes at eps=0, so the matrix and its
    # inverse are continuous there.  Exact rational evaluations illustrate
    # the two sides without serving as the proof.
    values: dict[str, str] = {}
    for epsilon in (sp.Rational(-1, 10), sp.Rational(0), sp.Rational(1, 10)):
        diffusion = sp.Matrix(((1, -epsilon), (-epsilon, 1)))
        weights = exact_harmonic_cell_weights(diffusion, side=4)
        values[str(epsilon)] = str(weights[(1, 1)][2])

    return {
        "arithmetic": "exact rational",
        "delaunay_p1_at_query_one_quarter": {
            "first_diagonal_limit": str(p1_first_diagonal),
            "second_diagonal_limit": str(p1_second_diagonal),
            "q1_tie_value": str(q1_tie),
            "continuous_in_metric": False,
        },
        "harmonic_cell_top_right_coordinate_at_query_one_quarter": values,
        "harmonic_metric_continuity": {
            "continuous_at_zero": True,
            "proof": "L(eps) is continuous, uniformly SPD locally, and inversion is continuous on nonsingular matrices",
        },
    }


def main() -> None:
    periodic = periodic_factor_two_certificate()
    payload = {
        "method": "ownerwise Riemannian harmonic cell coordinates",
        "operator_rule": "metric, owners, sites, and query only; values never select weights",
        "one_dimensional_exact_mse": None,
        "two_dimensional": exact_two_dimensional_certificate(),
        "exact_fidelity_sweep": exact_fidelity_sweep(),
        "round_trip_and_stability": exact_round_trip_and_stability_certificate(),
        "metric_flip": exact_metric_flip_certificate(),
    }
    # Correct the 1-D method field: harmonic coordinates equal the exact P1
    # certificate, not the Markov comparator.
    spectral = periodic_factor_two_certificate()
    side = int(spectral["common_periodic_lattice_size"])
    truth_all = [
        sp.simplify(
            sp.Rational(1, 2)
            + sp.Rational(1, 4)
            * sp.cos(2 * sp.pi * sp.Rational(index - 2, side))
        )
        for index in range(side)
    ]
    sources = [int(value) for value in spectral["source_centers"]]
    queries = [int(value) for value in spectral["query_centers"]]
    source_values = [truth_all[index] for index in sources]
    estimates = []
    for query in queries:
        # Sources are four lattice units apart on the periodic circle.
        candidates = sorted(sources)
        for left_index, left in enumerate(candidates):
            right = candidates[(left_index + 1) % len(candidates)]
            arc = (right - left) % side
            offset = (query - left) % side
            if offset <= arc:
                estimates.append(
                    sp.Rational(arc - offset, arc)
                    * source_values[sources.index(left)]
                    + sp.Rational(offset, arc)
                    * source_values[sources.index(right)]
                )
                break
    payload["one_dimensional_exact_mse"] = str(
        exact_mse(estimates, [truth_all[index] for index in queries])
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
