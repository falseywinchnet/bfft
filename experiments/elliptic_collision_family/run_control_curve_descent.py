"""Finite local two-descent for the primitive-support control curve.

This does not enumerate values of B.  It classifies the square classes of the
three pairwise-coprime factors and removes locally impossible 2-Selmer cases.
"""

from __future__ import annotations

from itertools import product


SQUAREFREE_PRIMES = (2, 3, 5, 7)
LOCAL_MODULI = (8, 3, 5, 7)


def parity_compatible_classes() -> tuple[tuple[int, int, int], ...]:
    classes = []
    for assignment in product(range(3), repeat=len(SQUAREFREE_PRIMES)):
        parts = [1, 1, 1]
        for prime, slot in zip(SQUAREFREE_PRIMES, assignment):
            parts[slot] *= prime
        # B is odd on the unit-offset characteristic.
        if parts[0] % 2:
            classes.append(tuple(parts))
    return tuple(classes)


def locally_soluble(square_class: tuple[int, int, int], modulus: int) -> bool:
    """Test the two Pell equations in one square class modulo ``modulus``."""

    d_b, d_a, d_d = square_class
    squares = {value * value % modulus for value in range(modulus)}
    first_values = {(2 * d_a * square) % modulus for square in squares}
    second_values = {(2 * d_d * square) % modulus for square in squares}
    return any(
        (5 * d_b * u * u - 1) % modulus in first_values
        and (7 * d_b * u * u - 1) % modulus in second_values
        for u in range(modulus)
    )


def descent_report() -> tuple[tuple[int, int, tuple[tuple[int, int, int], ...]], ...]:
    survivors = parity_compatible_classes()
    report = []
    for modulus in LOCAL_MODULI:
        before = len(survivors)
        survivors = tuple(
            square_class
            for square_class in survivors
            if locally_soluble(square_class, modulus)
        )
        report.append((modulus, before, survivors))
    return tuple(report)


def main() -> None:
    print("parity-compatible classes:", len(parity_compatible_classes()))
    for modulus, before, survivors in descent_report():
        print(f"mod {modulus}: {before} -> {len(survivors)}")
        if len(survivors) <= 5:
            print("  ", survivors)


if __name__ == "__main__":
    main()

