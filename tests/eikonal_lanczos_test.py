import numpy as np

from port_needed.eikonal_lanczos import eikonal_lanczos_resize


def test_eikonal_lanczos_reproduces_constant_at_arbitrary_scale():
    image = np.full((19, 27, 3), (0.17, 0.43, 0.81), dtype=np.float64)
    labels = np.zeros((19, 27), dtype=np.int32)
    zero = np.zeros(labels.shape, dtype=np.float64)

    output = eikonal_lanczos_resize(
        image, (31, 44), labels, (zero, zero, zero))

    np.testing.assert_allclose(
        output,
        np.broadcast_to((0.17, 0.43, 0.81), output.shape),
        atol=1e-12,
    )


def test_eikonal_lanczos_never_mixes_structural_owners():
    image = np.zeros((20, 28, 3), dtype=np.float64)
    image[:, 14:] = 1.0
    labels = np.zeros((20, 28), dtype=np.int32)
    labels[:, 14:] = 1
    xx = np.ones(labels.shape, dtype=np.float64)
    zero = np.zeros(labels.shape, dtype=np.float64)

    output = eikonal_lanczos_resize(
        image, (37, 53), labels, (xx, zero, zero))

    assert np.all((output == 0.0) | (output == 1.0))
