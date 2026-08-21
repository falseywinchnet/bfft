"""Probe posterior behavior outside additive, independent noise geometry.

The corruptions here are diagnostic instruments, not estimator branches.  They
ask whether one fixed transport law remains credible under photon counting and
under signal-dependent, row-correlated nuisance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage

from .backward_moment_smoother_2d import denoise_backward_moment_smoother_2d
from .canonical_variance_transport_2d import (
    denoise_canonical_variance_transport_2d,
)
from .continual_fabada_eikonal_2d import (
    denoise_anisotropic_moment_residual_posterior_2d,
    denoise_bounded_complete_moment_posterior_2d,
    denoise_complete_moment_residual_posterior_2d,
    denoise_continual_residual_posterior_2d,
)
from .fmmt_certified import denoise_fmmt
from .reflection_consistent_posterior_2d import (
    denoise_reflection_consistent_posterior_2d,
)
from .run_2d_denoiser_battery import metrics, sources


def poisson_observation(
    truth: np.ndarray,
    exposure: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Photon-counting observation in normalized intensity coordinates."""
    return np.clip(rng.poisson(np.maximum(truth, 0.0) * exposure) / exposure,
                   0.0, 1.0)


def row_correlated_signal_dependent_observation(
    truth: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Heteroscedastic nuisance whose innovations share horizontal ancestry."""
    innovations = rng.normal(size=truth.shape)
    correlated = ndimage.gaussian_filter1d(
        innovations, sigma=2.0, axis=1, mode="reflect")
    correlated /= max(float(np.std(correlated)), np.finfo(float).tiny)
    amplitude = scale * np.sqrt(np.maximum(truth, 1.0 / 255.0))
    return np.clip(truth + amplitude * correlated, 0.0, 1.0)


def run(size: int, seeds: int) -> dict[str, object]:
    selected = {
        name: image for name, image in sources(size).items()
        if name in {"cameraman", "geometric interfaces", "woven chirps"}
    }
    methods: dict[str, Callable[[np.ndarray], tuple[np.ndarray, dict]]] = {
        "observation": lambda value: (value.copy(), {}),
        "screened transport": denoise_fmmt,
        "transported residual posterior": denoise_continual_residual_posterior_2d,
        "complete-moment metric posterior": (
            denoise_complete_moment_residual_posterior_2d),
        "anisotropic-moment metric posterior": (
            denoise_anisotropic_moment_residual_posterior_2d),
        "backward complete-moment smoother": denoise_backward_moment_smoother_2d,
        "bounded complete-moment posterior": (
            denoise_bounded_complete_moment_posterior_2d),
        "Fisher-coordinate residual posterior": lambda value: (
            denoise_canonical_variance_transport_2d(value, "fisher")),
        "canonical-coordinate residual posterior": lambda value: (
            denoise_canonical_variance_transport_2d(value, "canonical")),
        "reflection-consistent posterior": denoise_reflection_consistent_posterior_2d,
    }
    rows: list[dict[str, object]] = []
    reflection_diagnostics: list[dict[str, object]] = []
    cases = (
        ("Poisson exposure 8", "poisson", 8.0),
        ("Poisson exposure 32", "poisson", 32.0),
        ("row-correlated signal-dependent 0.08", "row", 0.08),
        ("row-correlated signal-dependent 0.15", "row", 0.15),
    )
    for source, truth in selected.items():
        for condition, kind, level in cases:
            for seed in range(seeds):
                rng = np.random.default_rng(260821 + seed)
                if kind == "poisson":
                    observed = poisson_observation(truth, level, rng)
                else:
                    observed = row_correlated_signal_dependent_observation(
                        truth, level, rng)
                for method, estimator in methods.items():
                    estimate, diagnostic = estimator(observed)
                    rows.append({
                        "source": source,
                        "condition": condition,
                        "seed": seed,
                        "method": method,
                        **metrics(estimate, truth),
                    })
                    if method == "reflection-consistent posterior":
                        reflection_diagnostics.append({
                            "source": source,
                            "condition": condition,
                            "seed": seed,
                            "mean_authority": diagnostic[
                                "mean_reflection_authority"],
                            "mean_disagreement_squared": diagnostic[
                                "mean_reflection_disagreement_squared"],
                        })

    metric_names = ("mse", "ssim", "variance_ratio", "central_range_ratio",
                    "edge_retention", "mean_bias")

    def summarize(selected_rows: list[dict[str, object]]) -> dict[str, float]:
        return {
            name: float(np.mean([float(row[name]) for row in selected_rows]))
            for name in metric_names
        }

    summary = {
        method: summarize([row for row in rows if row["method"] == method])
        for method in methods
    }
    by_condition = {
        condition: {
            method: summarize([
                row for row in rows
                if row["condition"] == condition and row["method"] == method
            ])
            for method in methods
        }
        for condition, _kind, _level in cases
    }
    return {
        "status": "diagnostic only; no named-noise estimator branch",
        "size": size,
        "seeds": seeds,
        "corruption_role": (
            "known generators used only to falsify one fixed estimator law"),
        "summary": summary,
        "by_condition": by_condition,
        "reflection_diagnostics": reflection_diagnostics,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.size, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "summary": report["summary"],
        "by_condition": report["by_condition"],
    }, indent=2))


if __name__ == "__main__":
    main()
