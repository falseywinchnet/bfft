import numpy as np
from scipy.sparse.linalg import splu

from bfft.vision import (assemble_normal, block_dct8_native,
                         compact_support_operators, coownership_graph,
                         inverse_block_dct8_native, nearest_code_native,
                         render_partition, weighted_kmeanspp_native)


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


def test_native_nearest_code_matches_scipy_vq():
    from scipy.cluster.vq import vq

    rng = np.random.default_rng(20260820)
    for dimensions in (3, 4):
        observations = rng.uniform(0, 255, (4099, dimensions)).astype(np.float32)
        codes = rng.uniform(0, 255, (193, dimensions)).astype(np.float32)
        native = nearest_code_native(observations, codes, threads=3)
        assert native is not None
        reference = vq(observations, codes, check_finite=False)[0]
        np.testing.assert_array_equal(native, reference)


def test_native_weighted_kmeanspp_preserves_numpy_rng_construction():
    source_rng = np.random.default_rng(20260822)
    observations = source_rng.uniform(0, 255, (4096, 3)).astype(np.float32)
    weights = source_rng.uniform(0.5, 4.0, len(observations))
    code_count = 61
    reference_rng = np.random.default_rng(508030340)
    native_rng = np.random.default_rng(508030340)
    reference = np.empty((code_count, 3), dtype=np.float32)
    first = reference_rng.choice(len(observations), p=weights / weights.sum())
    reference[0] = observations[first]
    closest = np.sum((observations - reference[0]) ** 2, axis=1)
    for index in range(1, code_count):
        probability = weights * closest
        choice = reference_rng.choice(
            len(observations), p=probability / probability.sum()
        )
        reference[index] = observations[choice]
        closest = np.minimum(
            closest,
            np.sum((observations - reference[index]) ** 2, axis=1),
        )
    native = weighted_kmeanspp_native(
        observations, weights, code_count, native_rng.random(code_count)
    )
    assert native is not None
    np.testing.assert_array_equal(native, reference)


def test_native_block_dct_matches_numpy_basis_and_round_trip():
    rng = np.random.default_rng(20260821)
    plane = rng.uniform(0, 255, (67, 91))
    coordinate = np.arange(8, dtype=np.float64)
    matrix = np.cos(
        (2.0 * coordinate[None, :] + 1.0) * coordinate[:, None] * np.pi / 16.0
    )
    matrix[0] *= 1.0 / np.sqrt(2.0)
    matrix *= 0.5
    native = block_dct8_native(plane, matrix, threads=3)
    assert native is not None
    padded = np.pad(plane, ((0, (-len(plane)) % 8), (0, (-plane.shape[1]) % 8)), mode="edge")
    blocks = padded.reshape(9, 8, 12, 8).transpose(0, 2, 1, 3)
    reference = np.einsum("ui,abij,vj->abuv", matrix, blocks - 128.0, matrix)
    np.testing.assert_allclose(native, reference, atol=2e-13, rtol=0)
    recovered = inverse_block_dct8_native(native, plane.shape, matrix, threads=3)
    np.testing.assert_allclose(recovered, plane, atol=4e-13, rtol=0)
