"""Render a reproducible Cameraman gate for the terminal component posterior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .causal_information_lineage_2d import causal_information_lineage_law_2d
from .fmmt_certified import denoise_fmmt
from .probe_complete_moment_lineage_2d import displacement
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt
from .zero_residual_component_2d import zero_residual_component_readouts


def _save_image(path: Path, value: np.ndarray) -> None:
    pixels = np.uint8(np.round(np.clip(value, 0.0, 1.0) * 255.0))
    Image.fromarray(pixels).save(path)


def _render_comparison(
    output: Path,
    estimates: dict[str, np.ndarray],
    measured: dict[str, dict[str, float]],
) -> None:
    size = next(iter(estimates.values())).shape[0]
    scale = max(1, 384 // size)
    panel_size = size * scale
    title_height = 58
    canvas = Image.new(
        "RGB", (panel_size * len(estimates), panel_size + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for index, (name, value) in enumerate(estimates.items()):
        pixels = np.uint8(np.round(np.clip(value, 0.0, 1.0) * 255.0))
        panel = Image.fromarray(pixels).resize(
            (panel_size, panel_size), Image.Resampling.NEAREST).convert("RGB")
        left = index * panel_size
        canvas.paste(panel, (left, title_height))
        title = name if name == "truth" else (
            f"{name}\ntruth MSE {measured[name]['mse']:.5f} | "
            f"obs RMS {measured[name]['observation_displacement_rms']:.4f}")
        bounds = draw.multiline_textbbox((0, 0), title, font=font, align="center")
        width = bounds[2] - bounds[0]
        draw.multiline_text(
            (left + (panel_size - width) / 2.0, 8),
            title,
            fill="black",
            font=font,
            align="center",
        )
    canvas.save(output / "terminal_component_comparison.png")


def render_existing(output: Path) -> dict:
    """Finish a copied raw-image gate without rerunning transport."""
    files = {
        "truth": "truth.png",
        "observation": "observation.png",
        "FMMT": "fmmt.png",
        "HJ": "hj.png",
        "terminal component": "terminal_component.png",
    }
    estimates = {
        name: np.asarray(Image.open(output / filename).convert("L"), dtype=float)
        / 255.0
        for name, filename in files.items()
    }
    truth = estimates["truth"]
    observation = estimates["observation"]
    measured = {
        name: {
            **metrics(value, truth),
            **displacement(value, observation),
        }
        for name, value in estimates.items()
    }
    _render_comparison(output, estimates, measured)
    result = {
        "purpose": "render copied raw terminal-component visual gate",
        "source": "cameraman",
        "condition": "mixed replacement + uniform 0.25",
        "size": int(truth.shape[0]),
        "metrics": measured,
        "metric_precision": "8-bit copied images",
        "theory_status": "visual theorem probe; not promoted",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(size: int, output: Path) -> dict:
    truth = sources(size)["cameraman"]
    observation = corrupt(
        truth,
        "mixed replacement + uniform",
        amount=0.10,
        density=0.25,
        seed=271828,
    )
    law, law_diagnostic = causal_information_lineage_law_2d(
        observation,
        angular_count=4,
        quantile_count=16,
        population_phase=0.0,
        complete_residual_moment=True,
    )
    component, component_diagnostic = zero_residual_component_readouts(
        observation,
        law,
        nonzero_probability_mode="complete",
    )
    estimates = {
        "truth": truth,
        "observation": observation,
        "FMMT": denoise_fmmt(observation)[0],
        "HJ": law["causal_hj_simplex_collision_barycenter"],
        "terminal component": component["terminal_component_barycenter"],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in estimates.items():
        _save_image(output / f"{name.replace(' ', '_').lower()}.png", value)

    measured = {
        name: {
            **metrics(value, truth),
            **displacement(value, observation),
        }
        for name, value in estimates.items()
    }
    _render_comparison(output, estimates, measured)

    result = {
        "purpose": (
            "visual and observation-displacement gate for a complete-moment "
            "terminal zero/nonzero residual mixture"),
        "source": "cameraman",
        "condition": "mixed replacement + uniform 0.25",
        "size": size,
        "metrics": measured,
        "component_diagnostic": component_diagnostic,
        "transport_diagnostic": {
            "continuous_root_count": law_diagnostic["continuous_root_count"],
            "mean_branch_collision_population": law_diagnostic[
                "mean_branch_collision_population"],
            "mean_hj_simplex_collision_order": law_diagnostic[
                "mean_hj_simplex_collision_order"],
        },
        "theory_status": "visual theorem probe; not promoted",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-existing", action="store_true")
    args = parser.parse_args()
    result = render_existing(args.out) if args.render_existing else run(
        args.size, args.out)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
