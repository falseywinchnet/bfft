import numpy as np
from scipy.sparse.linalg import splu

from bfft.vision import (assemble_normal, compact_support_operators,
                         coownership_graph, render_partition)


def test_fused_partition_matches_explicit_design():
    rng = np.random.default_rng(20260726)
    pixels, cells, width = 79, 9, 4
    owner = rng.integers(0, cells, pixels, dtype=np.int32)
    runner = (owner + rng.integers(1, cells, pixels, dtype=np.int32)) % cells
    valid = rng.random(pixels) > 0.15
    runner = np.where(valid, runner, owner).astype(np.int32)
    w1 = rng.uniform(0.55, 1.0, pixels)
    w1[~valid] = 1.0
    w2 = 1.0 - w1
    first = rng.normal(size=(pixels, width))
    second = rng.normal(size=(pixels, width))
    target = rng.normal(size=(pixels, 3))
    regularization = np.tile([1e-5, 2e-3, 2e-3, 2e-3], cells)

    graph = coownership_graph(owner, runner, valid, cells, width)
    gram, rhs, _ = assemble_normal(
        owner, runner, valid, w1, w2, first, second, target, graph,
        regularization)

    design = np.zeros((pixels, cells * width))
    for pixel in range(pixels):
        i = owner[pixel]
        j = runner[pixel]
        design[pixel, width * i:width * (i + 1)] += (
            w1[pixel] * first[pixel])
        if valid[pixel]:
            design[pixel, width * j:width * (j + 1)] += (
                w2[pixel] * second[pixel])
    reference = design.T @ design + np.diag(regularization)
    np.testing.assert_allclose(gram.toarray(), reference, atol=2e-14, rtol=0)
    np.testing.assert_allclose(rhs, design.T @ target, atol=2e-14, rtol=0)

    lu = splu(gram, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
              options={"SymmetricMode": True})
    coefficients = lu.solve(rhs).reshape(cells, width, 3)
    field, _, _ = render_partition(
        coefficients, owner, runner, valid, w1, w2, first, second)
    np.testing.assert_allclose(
        field, design @ coefficients.reshape(cells * width, 3),
        atol=2e-14, rtol=0)

def test_compact_support_operator_matches_dense_design():
    rng = np.random.default_rng(20260727)
    pixels, cells, samples = 37, 11, 149
    rows = rng.integers(0, pixels, samples, dtype=np.int32)
    sites = rng.integers(0, cells, samples, dtype=np.int32)
    weight = rng.uniform(0.0, 1.0, samples)
    basis_x = rng.normal(size=samples)
    basis_y = rng.normal(size=samples)
    operators = compact_support_operators(
        rows, sites, weight, basis_x, basis_y, pixels, cells)
    assert operators is not None
    forward, transpose, normal = operators

    design = np.zeros((pixels, 3 * cells))
    for sample in range(samples):
        site = sites[sample]
        design[rows[sample], 3 * site:3 * site + 3] += (
            weight[sample]
            * np.array([1.0, basis_x[sample], basis_y[sample]]))
    coefficient = rng.normal(size=3 * cells)
    pixel = rng.normal(size=pixels)
    np.testing.assert_allclose(
        forward(coefficient), design @ coefficient, atol=2e-14, rtol=0)
    np.testing.assert_allclose(
        transpose(pixel), design.T @ pixel, atol=2e-14, rtol=0)
    np.testing.assert_allclose(
        normal(coefficient),
        design.T @ design @ coefficient,
        atol=3e-14,
        rtol=0,
    )
