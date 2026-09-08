"""Maximum spatial error in the Eikonal covector basis.

For the paired residual r = estimate-reference, form the spatial
product-moment tensor of its differential.  Its Rayleigh maximum is the
largest mean-square residual increment over every orientation, including all
orientations not aligned with the Cartesian sampling axes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
for directory in (ROOT, DEMO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backend import conv_basin_average, conv_resize, lanczos3_resize  # noqa: E402
from experiments.conv_fermi_owner_demo import (  # noqa: E402
    build_owner_atlas,
    synthesize_owner_charts,
)
from experiments.eikonal_current_pushforward_diagnostic import (  # noqa: E402
    analytic_feature,
    eikonal_fiber_resize,
)
from experiments.self_geometric_harmonic_interpolation import (  # noqa: E402
    _local_fourth_order_2jet,
)


Array = np.ndarray


@dataclass(frozen=True)
class EikonalPearsonMaximum:
    xx_product_moment: float
    xy_product_moment: float
    yy_product_moment: float
    xy_pearson: float
    maximum_mean_square_error: float
    maximum_rms_error: float
    principal_angle_degrees: float
    orthogonal_mean_square_error: float
    isotropy_ratio: float
    residual_mean: float
    residual_rms: float
    mean_gradient_x: float
    mean_gradient_y: float
    total_maximum_mean_square_error: float
    total_maximum_rms_error: float


def _gradient(values: Array) -> tuple[Array, Array]:
    field = np.asarray(values, dtype=np.float64)
    scalar = field.ndim == 2
    if scalar:
        field = field[..., None]
    if field.ndim != 3 or min(field.shape[:2]) < 5:
        raise ValueError("metric requires an HxW or HxWxC raster at least 5x5")
    _, gx, _ = _local_fourth_order_2jet(np.swapaxes(field, 0, 1))
    gx = np.swapaxes(gx, 0, 1)
    _, gy, _ = _local_fourth_order_2jet(field)
    return gx, gy


def eikonal_pearson_maximum(
    estimate: Array,
    reference: Array,
    *,
    crop: int = 2,
) -> EikonalPearsonMaximum:
    """Return the exact worst directional product moment of paired error.

    The Pearson tensor is the centered covariance of ``dr``.  Its largest
    eigenvalue is the maximum directional variance.  The mean-gradient outer
    product is retained separately and added back for the reported total
    maximum mean-square directional error.
    """

    trial = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    if trial.shape != truth.shape:
        raise ValueError("estimate and reference must have identical shapes")
    residual = trial - truth
    gx, gy = _gradient(residual)
    interior = np.s_[crop:-crop, crop:-crop] if crop else np.s_[:, :]
    gx = gx[interior]
    gy = gy[interior]
    mean_x = float(np.mean(gx))
    mean_y = float(np.mean(gy))
    centered_x = gx - mean_x
    centered_y = gy - mean_y
    xx = float(np.mean(centered_x * centered_x))
    xy = float(np.mean(centered_x * centered_y))
    yy = float(np.mean(centered_y * centered_y))
    tensor = np.array(((xx, xy), (xy, yy)), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    minimum = max(float(eigenvalues[0]), 0.0)
    maximum = max(float(eigenvalues[1]), 0.0)
    direction = eigenvectors[:, 1]
    angle = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    angle %= 180.0
    denominator = math.sqrt(xx * yy)
    correlation = xy / denominator if denominator > 0.0 else 0.0
    ratio = minimum / maximum if maximum > 0.0 else 1.0
    total_tensor = tensor + np.array(
        ((mean_x * mean_x, mean_x * mean_y),
         (mean_x * mean_y, mean_y * mean_y)), dtype=np.float64
    )
    total_maximum = max(float(np.linalg.eigvalsh(total_tensor)[1]), 0.0)
    return EikonalPearsonMaximum(
        xx_product_moment=xx,
        xy_product_moment=xy,
        yy_product_moment=yy,
        xy_pearson=float(np.clip(correlation, -1.0, 1.0)),
        maximum_mean_square_error=maximum,
        maximum_rms_error=math.sqrt(maximum),
        principal_angle_degrees=angle,
        orthogonal_mean_square_error=minimum,
        isotropy_ratio=ratio,
        residual_mean=float(np.mean(residual)),
        residual_rms=float(np.sqrt(np.mean(residual * residual))),
        mean_gradient_x=mean_x,
        mean_gradient_y=mean_y,
        total_maximum_mean_square_error=total_maximum,
        total_maximum_rms_error=math.sqrt(total_maximum),
    )


def run_audit(side: int = 129, factor: int = 4) -> dict[str, object]:
    angle = 37.5
    radians = math.radians(angle)
    normal = np.array((-math.sin(radians), math.cos(radians)))
    result: dict[str, object] = {}
    for family, width in (("ridge", 1.05), ("edge", 0.8)):
        truth = analytic_feature(side, angle, 0.35, family, width).astype(np.float32)
        coarse_side = (side - 1) // factor + 1
        coarse = conv_basin_average(truth, (coarse_side, coarse_side))
        candidates = {
            "CONV_basin_cycle": conv_resize(coarse, truth.shape),
            "straight_owner_chart_cycle": synthesize_owner_charts(
                coarse, truth.shape, build_owner_atlas(truth)
            ),
            "Lanczos3_cycle": lanczos3_resize(coarse, truth.shape),
            "exact_eikonal_fiber_cycle": eikonal_fiber_resize(
                coarse, truth.shape, normal
            ),
        }
        result[family] = {
            name: asdict(eikonal_pearson_maximum(value, truth))
            for name, value in candidates.items()
        }
    return {
        "definition": (
            "largest eigenvalue of the centered Pearson product-moment tensor "
            "of the residual differential"
        ),
        "angle_degrees": angle,
        "side": side,
        "factor": factor,
        "families": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=129)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "output/support_geometry/eikonal_pearson_maximum/audit.json",
    )
    args = parser.parse_args()
    result = run_audit(args.side, args.factor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
