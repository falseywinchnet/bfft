"""Exact rational certificates for the CONV Eikonal-fusion analysis."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path


def _ratio(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def certificate() -> dict[str, object]:
    # Three Euclidean sources at 0, 1, and 2, queried at x=1/2.
    distances = (Fraction(1, 2), Fraction(1, 2), Fraction(3, 2))
    labels = (Fraction(0), Fraction(0), Fraction(1))
    kernels = tuple(1 / (distance * distance) for distance in distances)
    denominator = sum(kernels)
    shepard = sum(label * kernel for label, kernel in zip(labels, kernels)) / denominator
    local_linear = Fraction(0)
    assert shepard == Fraction(1, 19)
    assert local_linear == 0

    # At r=1, equal labels preserve U,V while arrival direction changes.
    radius = Fraction(1)
    label = Fraction(1, 3)
    u_value = 2 / (radius * radius)
    v_value = label * u_value
    opposite_gradient_norm_sq = Fraction(0)
    orthogonal_gradient_norm_sq = Fraction(8) / radius**6
    assert u_value == 2 and v_value == Fraction(2, 3)
    assert opposite_gradient_norm_sq != orthogonal_gradient_norm_sq

    # Exact simplification of the nodal order coordinate.
    lambda_1 = Fraction(7, 3)
    lambda_2 = Fraction(2, 5)
    normal_y_sq = Fraction(4, 9)
    trace = lambda_1 + lambda_2
    coherence = (lambda_1 - lambda_2) / trace
    spectral_eta = (1 - coherence) / 2 + coherence * normal_y_sq
    gyy_over_trace = (
        lambda_1 * normal_y_sq + lambda_2 * (1 - normal_y_sq)
    ) / trace
    assert spectral_eta == gyy_over_trace

    return {
        "all_source_does_not_collapse_to_local": {
            "distances": [_ratio(value) for value in distances],
            "labels": [_ratio(value) for value in labels],
            "inverse_square_kernels": [_ratio(value) for value in kernels],
            "formal_beta": _ratio(shepard),
            "cell_local_beta": _ratio(local_linear),
        },
        "no_autonomous_scalar_eikonal_closure": {
            "U": _ratio(u_value),
            "V": _ratio(v_value),
            "opposite_arrivals_gradient_norm_squared": _ratio(
                opposite_gradient_norm_sq
            ),
            "orthogonal_arrivals_gradient_norm_squared": _ratio(
                orthogonal_gradient_norm_sq
            ),
        },
        "nodal_coordinate_identity": {
            "spectral_form": _ratio(spectral_eta),
            "Gyy_over_trace": _ratio(gyy_over_trace),
        },
    }


def main() -> None:
    result = certificate()
    target = Path("output/support_geometry/eikonal_kernel_fusion_certificate.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
