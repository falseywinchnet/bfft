import numpy as np

from port_needed.reverse_residual_flow import _principal_directions


def test_half_angle_direction_is_the_major_tensor_eigenvector():
    rng = np.random.default_rng(17)
    factor = rng.normal(size=(257, 2, 2))
    tensor = np.einsum("nij,nkj->nik", factor, factor)
    cxx = tensor[:, 0, 0]
    cxy = tensor[:, 0, 1]
    cyy = tensor[:, 1, 1]

    major, minor, direction = _principal_directions(cxx, cxy, cyy)

    applied = np.column_stack((
        cxx * direction[:, 0] + cxy * direction[:, 1],
        cxy * direction[:, 0] + cyy * direction[:, 1],
    ))
    np.testing.assert_allclose(
        applied, major[:, None] * direction, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        np.sum(direction * direction, axis=1), 1.0, atol=1e-14)
    assert np.all(major >= minor)


def test_isotropic_tensor_has_deterministic_axis():
    major, minor, direction = _principal_directions(
        np.array([3.0]), np.array([0.0]), np.array([3.0]))
    np.testing.assert_array_equal(direction, [[1.0, 0.0]])
    np.testing.assert_array_equal(major, [3.0])
    np.testing.assert_array_equal(minor, [3.0])
