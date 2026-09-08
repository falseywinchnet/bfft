"""Structural checks for the Mellin-current coefficient engine."""

from mellin_current import EllipticCurve, dirichlet_coefficients, trace_at_prime
from analyze_prime_phase import phase_chamber
from run_probe import RECORD


CONTROL = EllipticCurve(37, 0, 0, 1, -1, 0)


def test_small_traces_and_hecke_recurrence() -> None:
    assert trace_at_prime(CONTROL, 2) == -2
    assert trace_at_prime(CONTROL, 3) == -3
    assert trace_at_prime(CONTROL, 5) == -2
    coefficients = dirichlet_coefficients(CONTROL, 25)
    assert coefficients[1] == 1
    assert coefficients[4] == coefficients[2] ** 2 - 2
    assert coefficients[6] == coefficients[2] * coefficients[3]
    assert coefficients[25] == coefficients[5] ** 2 - 5


def test_record_negative_prime_chamber() -> None:
    report = phase_chamber(RECORD, 127)
    assert report["negative_prefix_last_prime"] == 107
    assert report["first_positive_prime"] == 109


if __name__ == "__main__":
    test_small_traces_and_hecke_recurrence()
    test_record_negative_prime_chamber()
    print("elliptic Mellin-current tests passed")
