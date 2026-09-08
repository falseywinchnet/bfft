"""Analytic outer certificates; assumptions and proofs in GUARANTEES.md.

Pure Python. Independent of filter particles and empirical covariance.
Floating point evaluation is not outward-rounded interval arithmetic.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Ball:
    center: tuple
    radius: float

    def contains(self, point, tolerance=0.0):
        return math.dist(self.center, point) <= self.radius + tolerance

    def radius_about(self, prediction):
        return self.radius + math.dist(self.center, prediction)


def pair_bound(t_a, y_a, error_a, t_b, y_b, error_b, target, acceleration_bound):
    """Bound x(target) given Euclidean observation errors and ||x''|| <= A.

    A is a transport regularity bound, not a fitted acceleration state.
    """
    values = (t_a, error_a, t_b, error_b, target, acceleration_bound)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Certificate inputs must be finite')
    if t_b <= t_a or target < t_b:
        raise ValueError('Require t_a < t_b <= target')
    if min(error_a, error_b, acceleration_bound) < 0:
        raise ValueError('Error and regularity bounds must be nonnegative')
    if not y_a or len(y_a) != len(y_b):
        raise ValueError('Positions must have the same nonzero dimension')
    if not all(math.isfinite(v) for v in (*y_a, *y_b)):
        raise ValueError('Positions must be finite')
    gap, horizon = t_b - t_a, target - t_b
    ratio = horizon / gap
    center = tuple(b + ratio * (b - a) for a, b in zip(y_a, y_b))
    radius = ((1 + ratio) * error_b + ratio * error_a
              + acceleration_bound * horizon * (gap + horizon) / 2)
    return Ball(center, radius)


def certificates(times, positions, errors, target, acceleration_bound):
    """All pair balls: their intersection contains every admissible position.

    Retain all balls to keep the intersection monotone as observations arrive
    at a fixed target. An empty intersection signals inconsistent assumptions.
    """
    if not (len(times) == len(positions) == len(errors)) or len(times) < 2:
        raise ValueError('At least two matched observations are required')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Observation times must strictly increase')
    return tuple(pair_bound(times[i], positions[i], errors[i],
                            times[j], positions[j], errors[j], target,
                            acceleration_bound)
                 for j in range(1, len(times)) for i in range(j))


def intrinsic_acceleration_bound(max_speed, max_speed_change, max_turn_rate):
    """x'=s*u, ||u||=1, |s'|<=a, ||u'||<=omega -> ||x''||<=A."""
    values = (max_speed, max_speed_change, max_turn_rate)
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Intrinsic bounds must be finite and nonnegative')
    return math.hypot(max_speed_change, max_speed * max_turn_rate)
