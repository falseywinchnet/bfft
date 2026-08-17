"""High-precision physical chart for the canonical Bidwell construction.

This is an initializer/control, not an imported proof method.  Constants are
transcribed from David Ellsworth's analytic SVG reconstruction:
https://kingbird.myphotos.cc/packing/square-17.svg
"""

from __future__ import annotations

import math

import numpy as np


REFERENCE_SIDE = 4.67553009360455095163411127048315
ANGLE_A_DEGREES = 39.8049589797677950558662536965423
ANGLE_B_DEGREES = 36.6237863834465893201859433547298
X1 = 0.640176192429266087584168543270317
V0 = 0.056838696870287317626412465614310
U1 = 0.276426766820379851031162763090775
U2 = 0.404478693276882795154318979238136
X3 = 1.84732482651028509718122846547049
Y4 = 3.11346013251256404552110105703834
R5 = 0.505927428006637436155510942971393


def _rotate(point: np.ndarray, theta: float) -> np.ndarray:
    cosine, sine = math.cos(theta), math.sin(theta)
    return np.asarray(
        [cosine * point[0] - sine * point[1],
         sine * point[0] + cosine * point[1]],
        dtype=np.float64,
    )


def reference_chart() -> np.ndarray:
    """Return centers and square phases for all seventeen unit squares."""

    side = REFERENCE_SIDE
    poses: list[tuple[float, float, float]] = []
    left = ((0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (0.5, side - 0.5))
    poses.extend((x, y, 0.0) for x, y in left)
    poses.extend((side - x, y, 0.0) for x, y in left)

    angle_a = math.radians(ANGLE_A_DEGREES)
    origins = (
        (0.0, 0.0),
        (1.0, V0),
        (U1, -1.0),
        (U1 + 1.0, V0 - 1.0),
        (U2, -2.0),
        (U2 + 1.0, V0 - 2.0),
    )
    for origin in origins:
        center = np.asarray([X1, 2.0]) + _rotate(
            np.asarray(origin) + 0.5, angle_a
        )
        poses.append((float(center[0]), float(center[1]), angle_a))

    poses.append((X3 + 0.5, side - 0.5, 0.0))
    angle_b = -math.radians(ANGLE_B_DEGREES)
    rotated_center = (
        np.asarray([X3 + 1.0, side - 1.0])
        + _rotate(np.asarray([0.5, 0.5 - R5]), angle_b)
    )
    poses.append((float(rotated_center[0]), float(rotated_center[1]), angle_b))
    poses.append((side - 0.5, Y4 - 0.5, 0.0))
    result = np.asarray(poses, dtype=np.float64)
    if result.shape != (17, 3):
        raise AssertionError(f"reference chart decoded to {result.shape}")
    return result
