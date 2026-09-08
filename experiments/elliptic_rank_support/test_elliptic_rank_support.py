from fractions import Fraction

from elliptic_rank_support import (
    SupportPencil,
    integral_record_points,
    invariants,
    reduction_dependency_masks,
    support_pencils,
    verify_fiber,
)


def test_record_points_and_known_pencil() -> None:
    points = integral_record_points()
    assert len(points) == 23
    pencil = SupportPencil(
        (Fraction(53), Fraction(8), Fraction(-9, 2), Fraction(-1, 2)),
        ((-10, 23), (-8, -43), (-6, -49), (-1, 41), (3, 23), (4, -19)),
    )
    assert pencil.integral_model(1) == (-27, -268, 1696)
    assert invariants(pencil.integral_model(1))[2] == 6_497_949_952
    assert verify_fiber(pencil, 1)
    # This attractive low-discriminant fiber is correctly rejected: its first
    # five visible sections retain a mod-2 dependency.
    assert reduction_dependency_masks(
        pencil.integral_model(1), pencil.transported_points(1)
    )


def test_divisor_decomposition_contains_known_pencil() -> None:
    pencils = support_pencils(integral_record_points())
    cubics = {pencil.cubic for pencil in pencils}
    assert (Fraction(53), Fraction(8), Fraction(-9, 2), Fraction(-1, 2)) in cubics
    assert len(pencils) == 196


if __name__ == "__main__":
    test_record_points_and_known_pencil()
    test_divisor_decomposition_contains_known_pencil()
    print("elliptic_rank_support tests passed")
