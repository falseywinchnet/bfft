"""Expose the strongest analytic cases in which owner frames beat direct frames."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_eikonal_frame_bank import (  # noqa: E402
    _cases,
    _evaluate_methods,
    _grid,
)


SELECTED = (
    ("interface_45", (25, 27), (97, 105)),
    ("filament_45", (17, 19), (65, 73)),
    ("circle_r0.62", (17, 19), (65, 73)),
    ("radial_chirp_p0.00", (17, 19), (65, 73)),
)


def run(output: Path) -> None:
    functions = {name: function for _, name, function in _cases()}
    rows = []
    for name, coarse_shape, fine_shape in SELECTED:
        coarse_x, coarse_y = _grid(coarse_shape)
        fine_x, fine_y = _grid(fine_shape)
        function = functions[name]
        coarse = np.clip(function(coarse_x, coarse_y), 0.0, 1.0).astype(np.float32)
        truth = np.clip(function(fine_x, fine_y), 0.0, 1.0).astype(np.float64)
        results = _evaluate_methods(coarse, fine_shape)
        owner = np.asarray(results["owner_frames"], dtype=np.float64)
        direct = np.asarray(results["direct_frames_inverse_2"], dtype=np.float64)
        sinc = np.asarray(results["lanczos3_sinc"], dtype=np.float64)
        owner_error = owner - truth
        direct_error = direct - truth
        sinc_error = sinc - truth
        rows.append((
            name, truth, owner, direct, sinc, owner_error, direct_error,
            sinc_error,
            np.abs(direct_error) - np.abs(owner_error),
        ))

    figure, axes = plt.subplots(
        len(rows), 8, figsize=(19.2, 9.8), constrained_layout=True
    )
    headers = (
        "analytic truth", "owner frames", "direct action frames", "Lanczos-3",
        "owner residual", "direct residual", "Lanczos residual",
        "|direct error| - |owner error|",
    )
    for row_index, row in enumerate(rows):
        (
            name, truth, owner, direct, sinc, owner_error, direct_error,
            sinc_error, advantage,
        ) = row
        mse_owner = float(np.mean(owner_error * owner_error))
        mse_direct = float(np.mean(direct_error * direct_error))
        mse_sinc = float(np.mean(sinc_error * sinc_error))
        residual_limit = max(
            float(np.max(np.abs(owner_error))),
            float(np.max(np.abs(direct_error))),
            float(np.max(np.abs(sinc_error))),
        )
        advantage_limit = float(np.max(np.abs(advantage)))
        images = (
            truth, owner, direct, sinc, owner_error, direct_error, sinc_error,
            advantage,
        )
        for column, image in enumerate(images):
            axis = axes[row_index, column]
            if column < 4:
                axis.imshow(
                    image, cmap="gray", vmin=0.0, vmax=1.0,
                    interpolation="nearest",
                )
            elif column < 7:
                axis.imshow(
                    image, cmap="coolwarm",
                    vmin=-residual_limit, vmax=residual_limit,
                    interpolation="nearest",
                )
            else:
                axis.imshow(
                    image, cmap="coolwarm",
                    vmin=-advantage_limit, vmax=advantage_limit,
                    interpolation="nearest",
                )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(headers[column], fontsize=9)
        axes[row_index, 0].set_ylabel(
            f"{name}\nowner {mse_owner:.3e}\ndirect {mse_direct:.3e}\n"
            f"sinc {mse_sinc:.3e}\ndirect/owner {mse_direct/mse_owner:.3f}",
            fontsize=8,
        )
    figure.suptitle(
        "Strongest owner-frame wins; residual columns share one scale per row\n"
        "Final column: red means owner frames are locally closer to analytic truth",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    run(ROOT / "output/support_geometry/eikonal_frame_bank/owner_win_diagnostics.png")
