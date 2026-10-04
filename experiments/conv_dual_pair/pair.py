"""Research-only matched dyadic cell-average pair with CONV moment admission.

Seven parent cells per synthesis phase; restriction averages two children.
The existing lineage scan/admission is retained. This is not a replacement
for point-valued CONV basin integration or its boundary convention.
"""
from dataclasses import dataclass
import math

import numpy as np

from experiments.conv_conservative_multiresolution import admit_moment_1d


MAXFLAT = np.array([201., -44., 5.]) / 1024.
BAND_OPTIMAL = np.array([
    (4096 - 483 * math.pi) / (2688 * math.pi),
    (-16384 + 3465 * math.pi) / (13440 * math.pi),
    (4096 - 945 * math.pi) / (13440 * math.pi),
])
RAW_CONV = np.array([1579 / 9216, -301 / 11520, 91 / 46080])
BASIN = np.array([11., -82., 709., 1604., 709., -82., 11.]) / 2880.
LINEAR_DUAL = np.array([
    -2981., -22222., -173472., 573838., 1456138.,
    573838., -173472., -22222., -2981.,
]) / 1103232.


def boundary_weights(offsets: tuple[int, ...]) -> np.ndarray:
    """Fixed rational closure compiled from the degree-six moment equations."""
    left = np.array([
        [-595,1384,-1619,1384,-761,240,-33],
        [-33,-364,691,-464,229,-68,9],
        [9,-96,-175,376,-149,40,-5],
    ], dtype=float) / 1024
    location=-offsets[0]
    if location < 3:
        return left[location]
    return -left[6-location,::-1]


def proposal(coarse, mode="maxflat"):
    z = np.asarray(coarse, dtype=np.float64)
    if z.ndim < 1 or len(z) < 7:
        raise ValueError("At least seven parent cells are required")
    coefficients = {"maxflat": MAXFLAT, "band": BAND_OPTIMAL, "raw-conv": RAW_CONV}[mode]
    out = np.zeros_like(z)
    for k, coefficient in enumerate(coefficients, 1):
        out[3:-3] += coefficient * (z[3+k:len(z)-3+k] - z[3-k:len(z)-3-k])
    # Shared polynomial closure for comparison; spectral optimality is an
    # infinite/interior-lattice result, not a boundary optimization claim.
    for i in (0, 1, 2, len(z)-3, len(z)-2, len(z)-1):
        start = min(max(i-3, 0), len(z)-7)
        weights = boundary_weights(tuple(range(start-i, start-i+7)))
        out[i] = np.tensordot(weights, z[start:start+7], axes=(0, 0))
    return out


def moment(coarse, mode="maxflat"):
    z = np.asarray(coarse, dtype=np.float64)
    return admit_moment_1d(z, proposal(z, mode))


def restrict(fine):
    x = np.asarray(fine, dtype=np.float64)
    if len(x) % 2:
        raise ValueError("Restriction needs complete child pairs")
    return (x[::2] + x[1::2]) * .5


def grow(coarse, mode="maxflat", detail=None):
    z = np.asarray(coarse, dtype=np.float64)
    mu = moment(z, mode)
    if detail is not None:
        mu = mu + np.asarray(detail, dtype=np.float64)
    out = np.empty((2*len(z),) + z.shape[1:])
    out[::2], out[1::2] = z-mu, z+mu
    return out


@dataclass(frozen=True)
class State:
    coarse: np.ndarray
    detail: np.ndarray
    mode: str


def analyze(fine, mode="maxflat"):
    x = np.asarray(fine, dtype=np.float64)
    z = restrict(x)
    return State(z, (x[1::2]-x[::2])*.5-moment(z, mode), mode)


def synthesize(state):
    return grow(state.coarse, state.mode, state.detail)


def apply_axis(operation, values, axis, **kwargs):
    return np.moveaxis(operation(np.moveaxis(values, axis, 0), **kwargs), 0, axis)


def grow2d(coarse, mode="maxflat"):
    return apply_axis(grow, apply_axis(grow, coarse, 1, mode=mode), 0, mode=mode)


def restrict2d(fine):
    # Reverse the order of the two synthesis factors.
    return apply_axis(restrict, apply_axis(restrict, fine, 0), 1)
