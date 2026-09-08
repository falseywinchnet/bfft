"""Meyer splitting in the exact detail coordinate of a CONV scale transport."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from skimage import data


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conv_distilled_core import distilled_conv_resize  # noqa: E402
from experiments.meyer_conv_transport_split import ranged, signed  # noqa: E402
from experiments.meyer_eikonal_natural_probe import (  # noqa: E402
    correlation,
    gradient_magnitude,
    normalized,
)
from experiments.meyer_eikonal_transport_split import (  # noqa: E402
    metric_meyer,
    relative_mse,
    scenes,
)


OUT = ROOT / "experiments" / "out" / "meyer_conv_scale_state_split"


def conv_scale_state(
    image: np.ndarray,
    factor: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return transported coarse state and its exact full-grid detail."""

    source = np.asarray(image, dtype=np.float64)
    if source.ndim != 2 or min(source.shape) < 5 * int(factor):
        raise ValueError("CONV scale state needs a sufficiently large 2-D source")
    coarse_shape = tuple(max(5, int(np.ceil(side / factor))) for side in source.shape)
    coarse = np.asarray(distilled_conv_resize(source, coarse_shape), dtype=np.float64)
    transported = np.asarray(
        distilled_conv_resize(coarse, source.shape), dtype=np.float64)
    detail = source - transported
    return transported, detail, {
        "factor": int(factor),
        "coarse_height": int(coarse_shape[0]),
        "coarse_width": int(coarse_shape[1]),
        "detail_relative_l2": float(
            np.linalg.norm(detail) / max(np.linalg.norm(source), 1e-30)),
        "detail_rms": float(np.sqrt(np.mean(detail * detail))),
        "coordinate_recomposition_linf": float(
            np.max(np.abs(source - transported - detail))),
    }


def scale_state_meyer(
    image: np.ndarray,
    factor: int = 2,
    *,
    outer_iterations: int = 96,
    inner_iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply Meyer only to the CONV detail coordinate, then restore scale state."""

    source = np.asarray(image, dtype=np.float64)
    transported, detail, scale_diagnostic = conv_scale_state(source, factor)
    identity = (np.ones(source.shape), np.zeros(source.shape), np.ones(source.shape))
    detail_cartoon, texture, meyer_diagnostic = metric_meyer(
        detail,
        identity,
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
    )
    cartoon = transported + detail_cartoon
    return cartoon, texture, {
        **scale_diagnostic,
        **{f"meyer_{key}": value for key, value in meyer_diagnostic.items()},
        "recomposition_linf": float(np.max(np.abs(source - cartoon - texture))),
    }


def _render(rows: list[tuple], titles: tuple[str, ...], path: Path) -> None:
    cell, label_width, header = 128, 135, 30
    canvas = Image.new(
        "RGB", (label_width + (len(titles) - 1) * cell,
                header + len(rows) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(titles[1:]):
        draw.text((label_width + column * cell + 3, 8), title, fill="black")
    for row_index, row in enumerate(rows):
        name, *planes = row
        y0 = header + row_index * cell
        draw.text((4, y0 + 8), name, fill="black")
        texture_indices = set(range(2, len(planes), 2))
        scale = max(
            [float(np.percentile(np.abs(planes[index]), 99.5))
             for index in texture_indices] + [1.0])
        for column, plane in enumerate(planes):
            panel = signed(plane, scale) if column in texture_indices else ranged(plane)
            canvas.paste(
                Image.fromarray(np.uint8(panel * 255)).convert("RGB"),
                (label_width + column * cell, y0),
            )
    canvas.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, dict[str, float]]] = {
        "synthetic": {}, "natural": {}}
    synthetic_rows = []
    for name, (source, truth_u, truth_v) in scenes().items():
        identity = (np.ones(source.shape), np.zeros(source.shape), np.ones(source.shape))
        base_u, base_v, _ = metric_meyer(source, identity)
        two_u, two_v, two_diag = scale_state_meyer(source, 2)
        four_u, four_v, four_diag = scale_state_meyer(source, 4)
        records["synthetic"][name] = {
            "ordinary_cartoon_relative_mse": relative_mse(base_u, truth_u),
            "ordinary_texture_relative_mse": relative_mse(base_v, truth_v),
            "factor2_cartoon_relative_mse": relative_mse(two_u, truth_u),
            "factor2_texture_relative_mse": relative_mse(two_v, truth_v),
            "factor4_cartoon_relative_mse": relative_mse(four_u, truth_u),
            "factor4_texture_relative_mse": relative_mse(four_v, truth_v),
            **{f"factor2_{key}": value for key, value in two_diag.items()},
            **{f"factor4_{key}": value for key, value in four_diag.items()},
        }
        synthetic_rows.append((
            name, source, truth_u, truth_v, base_u, base_v,
            two_u, two_v, four_u, four_v,
        ))

    natural_sources = {
        "camera": normalized(data.camera()),
        "coins": normalized(data.coins()),
        "page": normalized(data.page()),
        "moon": normalized(data.moon()),
    }
    natural_rows = []
    for name, source in natural_sources.items():
        identity = (np.ones(source.shape), np.zeros(source.shape), np.ones(source.shape))
        base_u, base_v, _ = metric_meyer(source, identity)
        two_u, two_v, two_diag = scale_state_meyer(source, 2)
        four_u, four_v, four_diag = scale_state_meyer(source, 4)
        edge = gradient_magnitude(source)
        records["natural"][name] = {
            "ordinary_texture_rms": float(np.sqrt(np.mean(base_v * base_v))),
            "ordinary_edge_correlation": correlation(np.abs(base_v), edge),
            "factor2_texture_rms": float(np.sqrt(np.mean(two_v * two_v))),
            "factor2_edge_correlation": correlation(np.abs(two_v), edge),
            "factor4_texture_rms": float(np.sqrt(np.mean(four_v * four_v))),
            "factor4_edge_correlation": correlation(np.abs(four_v), edge),
            **{f"factor2_{key}": value for key, value in two_diag.items()},
            **{f"factor4_{key}": value for key, value in four_diag.items()},
        }
        natural_rows.append((
            name, source, base_u, base_v, two_u, two_v, four_u, four_v,
        ))

    (OUT / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    _render(
        synthetic_rows,
        ("label", "source", "truth U", "truth V", "ordinary U", "ordinary V",
         "CONV-2 U", "CONV-2 V", "CONV-4 U", "CONV-4 V"),
        OUT / "synthetic.png",
    )
    _render(
        natural_rows,
        ("label", "source", "ordinary U", "ordinary V", "CONV-2 U",
         "CONV-2 V", "CONV-4 U", "CONV-4 V"),
        OUT / "natural.png",
    )


if __name__ == "__main__":
    main()
