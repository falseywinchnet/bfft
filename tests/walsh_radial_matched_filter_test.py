"""Algebraic checks for the radial matched-filter no-go exponent."""

from __future__ import annotations

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_radial_matched_filter import (
    asymptotic_radial_exponent,
    gaussian_importance_exponent,
    optimal_gaussian_target_width,
)


def test_paper_target_is_the_radial_matched_optimum() -> None:
    R = 0.400613
    r = optimal_gaussian_target_width(R)
    exponent = gaussian_importance_exponent(r, R)
    assert abs(r - 0.2222355) < 1e-6
    assert abs(exponent - 0.6038657533754043) < 1e-12
    assert math.isclose(
        asymptotic_radial_exponent(R),
        exponent,
        abs_tol=1e-14,
    )


def test_half_coset_source_has_radial_floor_above_half() -> None:
    # g(R)=1/2 occurs at R=2*t0 for the paper's elementary smoothing bound.
    R = 2.0 * 0.23147
    exponent = asymptotic_radial_exponent(R)
    assert exponent > 0.67
    assert exponent < 0.671


if __name__ == "__main__":
    test_paper_target_is_the_radial_matched_optimum()
    test_half_coset_source_has_radial_floor_above_half()
    print("walsh radial matched-filter tests passed")
