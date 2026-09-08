"""Exact rational certificate for the sampling-trace Markov polynomial.

Run with a Python environment containing SymPy.  The graph is a 4x4 torus
whose Dirichlet stencil comes from the inverse of the single dimensionless SPD form

    (H n)^T M (H n),

with anisotropic Cartesian spacing H and a rotated determinant-one metric M.
The exact identity ``(H.T*M*H)^-1 = sum gamma_e e*e.T`` has nonnegative
lattice coefficients, so this is a Riemannian diffusion discretization rather
than reciprocal-distance graph weighting.
Every reported algebraic assertion and squared error is checked over exact
SymPy rationals; this is not a floating-point experiment.
"""

from __future__ import annotations

import json

import sympy as sp


def node(ix: int, iy: int, side: int) -> int:
    return (iy % side) * side + (ix % side)


def build_exact_laplacian(side: int) -> tuple[sp.Matrix, dict[tuple[int, int], sp.Rational]]:
    metric = sp.Matrix([[2, 1], [1, 1]])
    spacing = sp.diag(1, 2)
    assert metric.det() == 1
    pulled_back_metric = spacing.T * metric * spacing
    diffusion = pulled_back_metric.inv()
    directions = ((1, 0), (1, -1))
    weights: dict[tuple[int, int], sp.Rational] = {
        (1, 0): sp.Rational(1, 2),
        (1, -1): sp.Rational(1, 2),
    }
    reconstructed = sp.zeros(2)
    for direction, weight in weights.items():
        vector = sp.Matrix(direction)
        reconstructed += weight * vector * vector.T
    assert reconstructed == diffusion

    count = side * side
    adjacency = sp.zeros(count)
    for iy in range(side):
        for ix in range(side):
            i = node(ix, iy, side)
            for dx, dy in directions:
                j = node(ix + dx, iy + dy, side)
                weight = weights[(dx, dy)]
                adjacency[i, j] += weight
                adjacency[j, i] += weight
    degree = sp.diag(*[sum(adjacency[i, j] for j in range(count)) for i in range(count)])
    return degree - adjacency, weights


def sampling_trace_kernel(
    laplacian: sp.Matrix, sample_count: int
) -> tuple[sp.Matrix, sp.Matrix, int, sp.Rational, sp.Rational]:
    eigenvalues = laplacian.eigenvals()
    lambda_max = max(eigenvalues)
    count = laplacian.rows
    step = sp.eye(count) - laplacian / lambda_max

    assert step == step.T
    assert step * sp.ones(count, 1) == sp.ones(count, 1)
    assert all(entry >= 0 for entry in step)
    assert all(value >= 0 for value in step.eigenvals())

    target = sp.Integer(sample_count)
    previous = sp.eye(count)
    previous_trace = sp.trace(previous)
    for degree in range(1, count + 1):
        current = previous * step
        current_trace = sp.trace(current)
        if current_trace <= target <= previous_trace:
            alpha = sp.cancel((previous_trace - target) / (previous_trace - current_trace))
            kernel = sp.simplify((1 - alpha) * previous + alpha * current)
            return kernel, step, degree, alpha, lambda_max
        previous = current
        previous_trace = current_trace
    raise AssertionError("trace target was not bracketed")


def rational_truth(side: int) -> list[sp.Rational]:
    truth: list[sp.Rational] = []
    for iy in range(side):
        for ix in range(side):
            x = sp.Rational(ix, side - 1)
            y = sp.Rational(iy, side - 1)
            truth.append(sp.cancel((2 + 2 * x + y + x * y + x**2 + y**2 / 2) / sp.Rational(15, 2)))
    return truth


def interpolate_exact(
    kernel: sp.Matrix,
    samples: list[int],
    truth: list[sp.Rational],
) -> tuple[list[sp.Rational], sp.Rational, sp.Rational]:
    sample_values = [truth[index] for index in samples]
    estimates: list[sp.Rational] = []
    minimum_denominator: sp.Rational | None = None
    maximum_row_error = sp.Rational(0)
    for row in range(kernel.rows):
        denominator = sum(kernel[row, source] for source in samples)
        assert denominator > 0
        minimum_denominator = denominator if minimum_denominator is None else min(minimum_denominator, denominator)
        normalized = [sp.cancel(kernel[row, source] / denominator) for source in samples]
        assert all(weight >= 0 for weight in normalized)
        maximum_row_error = max(maximum_row_error, abs(sum(normalized) - 1))
        estimate = sp.cancel(sum(weight * value for weight, value in zip(normalized, sample_values)))
        estimates.append(sample_values[samples.index(row)] if row in samples else estimate)
    return estimates, sp.Rational(minimum_denominator), maximum_row_error


def exact_mse(
    estimates: list[sp.Rational], truth: list[sp.Rational], samples: list[int]
) -> sp.Rational:
    unsampled = [index for index in range(len(truth)) if index not in samples]
    return sp.cancel(sum((estimates[index] - truth[index]) ** 2 for index in unsampled) / len(unsampled))


def periodic_factor_two_certificate() -> dict[str, object]:
    """Exact 1-D factor-two resampling and convex-projection tradeoff.

    Three coarse pixel centers and six fine pixel centers are disjoint on the
    exact common periodic lattice Z/12Z.  The rank-three projector contains the
    constant and first sine/cosine pair, exactly the complete spectrum admitted
    by three observations.
    """

    side = 12
    sources = [2, 6, 10]
    queries = [1, 3, 5, 7, 9, 11]
    assert not set(sources).intersection(queries)

    projector = sp.zeros(side)
    for row in range(side):
        for column in range(side):
            angle = 2 * sp.pi * sp.Rational(row - column, side)
            projector[row, column] = sp.simplify((1 + 2 * sp.cos(angle)) / side)
    assert projector == projector.T
    assert all(sp.simplify(entry) == 0 for entry in projector * projector - projector)
    assert sp.trace(projector) == len(sources)
    assert projector * sp.ones(side, 1) == sp.ones(side, 1)

    truth = [
        sp.simplify(sp.Rational(1, 2) + sp.Rational(1, 4) * sp.cos(2 * sp.pi * sp.Rational(index - 2, side)))
        for index in range(side)
    ]
    source_values = [truth[index] for index in sources]
    raw: list[sp.Expr] = []
    weights: list[list[sp.Expr]] = []
    for row in range(side):
        denominator = sp.simplify(sum(projector[row, source] for source in sources))
        assert denominator == sp.Rational(1, 4)
        normalized = [sp.simplify(projector[row, source] / denominator) for source in sources]
        assert sp.simplify(sum(normalized)) == 1
        weights.append(normalized)
        raw.append(sp.simplify(sum(weight * value for weight, value in zip(normalized, source_values))))
    assert all(sp.simplify(raw[index] - truth[index]) == 0 for index in range(side))

    lower, upper = min(source_values), max(source_values)
    projected = [sp.Max(lower, sp.Min(upper, value)) for value in raw]
    projected = [sp.simplify(value.doit()) for value in projected]
    assert all(lower <= projected[index] <= upper for index in queries)
    raw_mse = sp.simplify(sum((raw[index] - truth[index]) ** 2 for index in queries) / len(queries))
    projected_mse = sp.simplify(
        sum((projected[index] - truth[index]) ** 2 for index in queries) / len(queries)
    )
    assert raw_mse == 0
    assert projected_mse == (2 - sp.sqrt(3)) / 96

    laplacian = sp.zeros(side)
    for index in range(side):
        laplacian[index, index] = 2
        laplacian[index, (index - 1) % side] = -1
        laplacian[index, (index + 1) % side] = -1
    markov_kernel, _, markov_degree, markov_alpha, markov_lambda_max = sampling_trace_kernel(
        laplacian, len(sources)
    )
    markov_estimates, _, _ = interpolate_exact(markov_kernel, sources, truth)
    markov_mse = sp.simplify(
        sum((markov_estimates[index] - truth[index]) ** 2 for index in queries) / len(queries)
    )

    nearest = []
    for query in queries:
        source = min(
            sources,
            key=lambda site: (min((query - site) % side, (site - query) % side), site),
        )
        nearest.append(truth[source])
    nearest_mse = sp.simplify(
        sum((estimate - truth[index]) ** 2 for estimate, index in zip(nearest, queries)) / len(queries)
    )

    return {
        "arithmetic": "exact Q(sqrt(3))",
        "common_periodic_lattice_size": side,
        "source_centers": sources,
        "query_centers": queries,
        "admitted_complete_mode_count": int(sp.trace(projector)),
        "minimum_raw_spectral_weight": str(min(weight for row in weights for weight in row)),
        "sample_interval": [str(lower), str(upper)],
        "truth_query_interval": [str(min(truth[index] for index in queries)), str(max(truth[index] for index in queries))],
        "raw_spectral_mse": str(raw_mse),
        "projected_spectral_mse": str(projected_mse),
        "projected_spectral_mse_decimal_60": str(sp.N(projected_mse, 60)),
        "projected_activation_count": sum(raw[index] != projected[index] for index in queries),
        "markov_lambda_max": str(markov_lambda_max),
        "markov_degree": markov_degree,
        "markov_alpha": str(markov_alpha),
        "markov_mse": str(markov_mse),
        "markov_mse_decimal_60": str(sp.N(markov_mse, 60)),
        "nearest_mse": str(nearest_mse),
        "nearest_mse_decimal_60": str(sp.N(nearest_mse, 60)),
        "certificates": {
            "source_and_query_lattices_disjoint": True,
            "projector_is_exact_rank_three_orthogonal_projector": True,
            "raw_spectral_reconstruction_exact_on_admitted_truth": True,
            "raw_spectral_weights_are_signed": True,
            "interval_projection_has_zero_sample_interval_overshoot": True,
            "interval_projection_strictly_increases_mse_for_unresolved_extremum": True,
        },
    }


def main() -> None:
    side = 4
    laplacian, weights = build_exact_laplacian(side)
    samples = [node(ix, iy, side) for iy in (0, 2) for ix in (0, 2)]
    kernel, step, degree, alpha, lambda_max = sampling_trace_kernel(laplacian, len(samples))

    count = side * side
    ones = sp.ones(count, 1)
    assert kernel == kernel.T
    assert kernel * ones == ones
    assert sp.trace(kernel) == len(samples)
    assert all(entry >= 0 for entry in kernel)
    assert all(value >= 0 for value in kernel.eigenvals())
    assert 0 <= alpha <= 1

    truth = rational_truth(side)
    sample_values = [truth[index] for index in samples]
    estimates, minimum_denominator, maximum_row_error = interpolate_exact(kernel, samples, truth)

    lower, upper = min(sample_values), max(sample_values)
    assert all(lower <= estimate <= upper for estimate in estimates)
    mse = exact_mse(estimates, truth, samples)

    coverage_kernel = sp.eye(count)
    coverage_degree = 0
    while min(sum(coverage_kernel[row, source] for source in samples) for row in range(count)) == 0:
        coverage_kernel *= step
        coverage_degree += 1
    coverage_estimates, _, _ = interpolate_exact(coverage_kernel, samples, truth)
    coverage_mse = exact_mse(coverage_estimates, truth, samples)

    strongest_estimates = truth.copy()
    for row in range(count):
        if row not in samples:
            strongest = max(samples, key=lambda source: (kernel[row, source], -source))
            strongest_estimates[row] = truth[strongest]
    strongest_mse = exact_mse(strongest_estimates, truth, samples)

    payload = {
        "arithmetic": "exact rational",
        "side": side,
        "node_count": count,
        "samples": samples,
        "metric": [["2", "1"], ["1", "1"]],
        "spacing": [["1", "0"], ["0", "2"]],
        "direction_weights": {str(key): str(value) for key, value in weights.items()},
        "lambda_max": str(lambda_max),
        "polynomial_degree": degree,
        "convex_power_coefficient": str(alpha),
        "trace": str(sp.trace(kernel)),
        "minimum_kernel_entry": str(min(kernel)),
        "maximum_kernel_entry": str(max(kernel)),
        "minimum_sample_denominator": str(minimum_denominator),
        "maximum_normalized_row_sum_error": str(maximum_row_error),
        "sample_range": [str(lower), str(upper)],
        "mse_fraction": str(mse),
        "mse_decimal_60": str(sp.N(mse, 60)),
        "exact_comparisons": {
            "coverage_polynomial_degree": coverage_degree,
            "coverage_polynomial_mse_fraction": str(coverage_mse),
            "coverage_polynomial_mse_decimal_60": str(sp.N(coverage_mse, 60)),
            "strongest_single_sample_mse_fraction": str(strongest_mse),
            "strongest_single_sample_mse_decimal_60": str(sp.N(strongest_mse, 60)),
            "trace_over_coverage_mse_ratio": str(sp.cancel(mse / coverage_mse)),
            "trace_over_strongest_mse_ratio": str(sp.cancel(mse / strongest_mse)),
        },
        "certificates": {
            "metric_determinant_one": True,
            "kernel_symmetric": True,
            "kernel_entrywise_nonnegative": True,
            "kernel_mass_one": True,
            "kernel_positive_semidefinite": True,
            "effective_rank_equals_sample_count": True,
            "normalized_weights_convex": True,
            "zero_sample_range_overshoot": True,
        },
        "periodic_factor_two_resampling": periodic_factor_two_certificate(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
