"""Use CONV target-excluded value transport as a lift before Meyer splitting.

The CONV charts define a pointwise interval of values supported without using
the target sample.  The source is projected into that interval, leaving the
smallest transport defect.  Meyer acts only on that defect; supported
structure is restored to the cartoon component after the split.
"""

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

from denoiser.dcnt import transport_uncertainty  # noqa: E402
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


OUT = ROOT / "experiments" / "out" / "meyer_conv_lifted_split"


def conv_transport_lift(
    image: np.ndarray,
    *,
    support: str = "hull",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return the CONV-supported value and its exact source defect.

    ``hull`` is the smallest interval containing all three target-excluded
    chart values. ``centre`` retains only their median and is included as a
    matched ablation of the interval geometry.
    """

    source = np.asarray(image, dtype=np.float64)
    law = transport_uncertainty(source)
    if support == "hull":
        lower = law.centre - law.convex_radius
        upper = law.centre + law.convex_radius
        transported = np.clip(source, lower, upper)
    elif support == "centre":
        transported = law.centre.copy()
    else:
        raise ValueError("support must be 'hull' or 'centre'")
    defect = source - transported
    return transported, defect, {
        "transported_fraction": float(
            np.linalg.norm(transported) / max(np.linalg.norm(source), 1e-30)),
        "defect_relative_l2": float(
            np.linalg.norm(defect) / max(np.linalg.norm(source), 1e-30)),
        "defect_rms": float(np.sqrt(np.mean(defect * defect))),
        "exact_linf": float(np.max(np.abs(source - transported - defect))),
    }


def lifted_meyer(
    image: np.ndarray,
    *,
    support: str = "hull",
    outer_iterations: int = 96,
    inner_iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Split only the component not admitted by target-excluded CONV transport."""

    source = np.asarray(image, dtype=np.float64)
    transported, defect, lift_diagnostic = conv_transport_lift(
        source, support=support)
    identity = (np.ones(source.shape), np.zeros(source.shape), np.ones(source.shape))
    defect_cartoon, texture, meyer_diagnostic = metric_meyer(
        defect,
        identity,
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
    )
    cartoon = transported + defect_cartoon
    return cartoon, texture, {
        **lift_diagnostic,
        **{f"meyer_{key}": value for key, value in meyer_diagnostic.items()},
        "recomposition_linf": float(np.max(np.abs(source - cartoon - texture))),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic_records: dict[str, dict[str, float]] = {}
    synthetic_rows = []
    for name, (source, truth_u, truth_v) in scenes().items():
        identity = (np.ones(source.shape), np.zeros(source.shape), np.ones(source.shape))
        base_u, base_v, _ = metric_meyer(source, identity)
        hull_u, hull_v, hull_diagnostic = lifted_meyer(source, support="hull")
        centre_u, centre_v, centre_diagnostic = lifted_meyer(source, support="centre")
        synthetic_records[name] = {
            "ordinary_cartoon_relative_mse": relative_mse(base_u, truth_u),
            "ordinary_texture_relative_mse": relative_mse(base_v, truth_v),
            "hull_cartoon_relative_mse": relative_mse(hull_u, truth_u),
            "hull_texture_relative_mse": relative_mse(hull_v, truth_v),
            "centre_cartoon_relative_mse": relative_mse(centre_u, truth_u),
            "centre_texture_relative_mse": relative_mse(centre_v, truth_v),
            **{f"hull_{key}": value for key, value in hull_diagnostic.items()},
            **{f"centre_{key}": value for key, value in centre_diagnostic.items()},
        }
        synthetic_rows.append((
            name, source, truth_u, truth_v, base_u, base_v,
            hull_u, hull_v, centre_u, centre_v,
        ))

    natural_sources = {
        "camera": normalized(data.camera()),
        "coins": normalized(data.coins()),
        "page": normalized(data.page()),
        "moon": normalized(data.moon()),
    }
    natural_records: dict[str, dict[str, float]] = {}
    natural_rows = []
    for name, source in natural_sources.items():
        identity = (np.ones(source.shape), np.zeros(source.shape), np.ones(source.shape))
        base_u, base_v, _ = metric_meyer(source, identity)
        hull_u, hull_v, hull_diagnostic = lifted_meyer(source, support="hull")
        centre_u, centre_v, centre_diagnostic = lifted_meyer(source, support="centre")
        edge = gradient_magnitude(source)
        natural_records[name] = {
            "ordinary_texture_rms": float(np.sqrt(np.mean(base_v * base_v))),
            "ordinary_texture_edge_correlation": correlation(np.abs(base_v), edge),
            "hull_texture_rms": float(np.sqrt(np.mean(hull_v * hull_v))),
            "hull_texture_edge_correlation": correlation(np.abs(hull_v), edge),
            "centre_texture_rms": float(np.sqrt(np.mean(centre_v * centre_v))),
            "centre_texture_edge_correlation": correlation(np.abs(centre_v), edge),
            **{f"hull_{key}": value for key, value in hull_diagnostic.items()},
            **{f"centre_{key}": value for key, value in centre_diagnostic.items()},
        }
        natural_rows.append((
            name, source, base_u, base_v, hull_u, hull_v, centre_u, centre_v,
        ))

    (OUT / "metrics.json").write_text(json.dumps({
        "synthetic": synthetic_records,
        "natural": natural_records,
    }, indent=2) + "\n")

    def render(rows: list[tuple], titles: tuple[str, ...], path: Path) -> None:
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

    render(
        synthetic_rows,
        ("label", "source", "truth U", "truth V", "ordinary U", "ordinary V",
         "hull-lift U", "hull-lift V", "centre-lift U", "centre-lift V"),
        OUT / "synthetic.png",
    )
    render(
        natural_rows,
        ("label", "source", "ordinary U", "ordinary V", "hull-lift U",
         "hull-lift V", "centre-lift U", "centre-lift V"),
        OUT / "natural.png",
    )


if __name__ == "__main__":
    main()
