"""Conductor-first audit on the split discriminant surface ``m=3*d^2``.

For these norms,

    4*m^3-27*a6^2 = 27*(2*d^3-a6)*(2*d^3+a6),

so a large model discriminant can still have a small conductor radical.  The
values ``13,19,37,41`` are not scanned parameters: they are precisely the
prime directions exposed by the non-Weyl ``m=481`` supports and its exact
rank-two Prym.  Difference-of-squares height compatibility gives a finite
complete curve list for each fixed norm.
"""

from __future__ import annotations

try:
    from .run_three_split_exact_audit import (
        ThreeSplitCurve,
        classify_three_split_curves,
        pari_audit,
    )
except ImportError:  # Direct execution from this directory.
    from run_three_split_exact_audit import (
        ThreeSplitCurve,
        classify_three_split_curves,
        pari_audit,
    )


STRUCTURAL_DIRECTIONS = (13, 19, 37, 41)


def split_discriminant_candidates() -> tuple[ThreeSplitCurve, ...]:
    maximum_norm = 3 * max(STRUCTURAL_DIRECTIONS) ** 2
    curves = classify_three_split_curves(
        maximum_norm=maximum_norm,
        discriminant_ceiling=1 << 4_096,
    )
    selected_norms = {3 * direction * direction for direction in STRUCTURAL_DIRECTIONS}
    return tuple(curve for curve in curves if curve.m in selected_norms)


def main() -> None:
    candidates = split_discriminant_candidates()
    for audit in pari_audit(candidates):
        curve = audit.curve
        print(
            f"m={curve.m} a6={curve.a6} currents={curve.currents} "
            f"conductor={audit.conductor} root={audit.root_number} "
            f"rank=[{audit.rank_lower},{audit.rank_upper}]"
        )


if __name__ == "__main__":
    main()
