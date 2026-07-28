#!/usr/bin/env python3
"""Continual sparse registration with an explicit entropy budget.

Temperature is only an indirect description of epistemic support.  This
experiment instead projects every batch posterior onto

    exp(mean_frame H[p(shift | photons)])) = K,

where K is the requested effective number of live translations.  The
projection stays on the exact Poisson-likelihood exponential family and is
found by a scalar bisection for inverse temperature.

The adaptive operator splits only the photons in the current batch.  Each
half proposes a posterior and is scored by how well it predicts the other
half.  A decayed evidence accumulator supplies friction, after which the
selected support budget is applied to the unsplit batch.  It is one-pass,
uses no clean image, and never revisits an earlier frame.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp

from finite_d_poisson_flow import (
    continual_likelihood_flow,
    poisson_thin,
)
from sparse_ray_transport import (
    RayBench,
    backproject,
    calibrated_radiance,
    capture,
    continual_ray_transport,
    marginal_poisson_score,
    metrics,
    oracle_registered_mean,
    shift_mask,
    source_image,
)


@dataclass(frozen=True)
class EntropyBench:
    budgets: tuple[float, ...] = (
        4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0)
    evidence_decays: tuple[float, ...] = (0.0, 0.5, 0.9, 0.98, 1.0)
    split_probability: float = 0.5
    split_seed_offset: int = 311
    bisection_steps: int = 36
    max_budget_steps_per_batch: int = 1
    robustness_seeds: int = 5


def poisson_shift_scores(
    counts: np.ndarray,
    belief: np.ndarray,
    config: RayBench,
    admitted: np.ndarray,
) -> np.ndarray:
    """Exact shift-dependent Poisson log likelihood for every frame."""
    rate = np.maximum(
        config.photons_at_white * belief + config.background_photons,
        1e-7,
    )
    score = np.fft.ifft2(
        np.fft.fft2(
            np.asarray(counts, dtype=np.float64), axes=(-2, -1)
        )
        * np.conj(np.fft.fft2(np.log(rate))),
        axes=(-2, -1),
    ).real
    return score[:, admitted]


def normalized_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(np.clip(shifted, -60.0, 0.0))
    probability /= np.maximum(
        np.sum(probability, axis=1, keepdims=True), 1e-30)
    return probability


def posterior_statistics(
    probability: np.ndarray,
) -> dict[str, float]:
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, 1e-30)), axis=1)
    return {
        "mean_entropy_nats": float(np.mean(entropy)),
        "effective_shifts": float(np.exp(np.mean(entropy))),
        "mean_frame_effective_shifts": float(np.mean(np.exp(entropy))),
        "mean_peak_probability": float(np.mean(
            np.max(probability, axis=1))),
    }


def entropy_project_scores(
    scores: np.ndarray,
    target_effective_shifts: float,
    *,
    bisection_steps: int = 36,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project likelihood scores onto a geometric-mean support budget.

    The result is ``softmax(beta * scores)``.  Because entropy decreases
    monotonically with nonnegative inverse temperature ``beta``, a scalar
    bisection gives the unique exponential-family member at the requested
    batch entropy.  Exact score ties impose the only possible lower bound.
    """
    score = np.asarray(scores, dtype=np.float64)
    if score.ndim != 2 or score.shape[1] == 0:
        raise ValueError("scores must have shape (frames, admitted shifts)")
    candidate_count = score.shape[1]
    target = float(np.clip(
        target_effective_shifts, 1.0, float(candidate_count)))
    target_entropy = float(np.log(target))
    centered = score - np.max(score, axis=1, keepdims=True)

    if target >= candidate_count * (1.0 - 1e-12):
        probability = np.full_like(centered, 1.0 / candidate_count)
        info = posterior_statistics(probability)
        return probability, {
            **info,
            "target_effective_shifts": target,
            "inverse_temperature": 0.0,
            "temperature": float("inf"),
            "entropy_residual_nats": (
                info["mean_entropy_nats"] - target_entropy),
        }

    def entropy_at(inverse_temperature: float) -> float:
        probability = normalized_probabilities(
            inverse_temperature * centered)
        return posterior_statistics(probability)["mean_entropy_nats"]

    low = 0.0
    high = 1.0
    while entropy_at(high) > target_entropy and high < 1e8:
        high *= 2.0
    for _ in range(bisection_steps):
        middle = 0.5 * (low + high)
        if entropy_at(middle) > target_entropy:
            low = middle
        else:
            high = middle
    inverse_temperature = 0.5 * (low + high)
    probability = normalized_probabilities(
        inverse_temperature * centered)
    info = posterior_statistics(probability)
    return probability, {
        **info,
        "target_effective_shifts": target,
        "inverse_temperature": inverse_temperature,
        "temperature": 1.0 / max(inverse_temperature, 1e-30),
        "entropy_residual_nats": (
            info["mean_entropy_nats"] - target_entropy),
    }


def embed_posterior(
    probability: np.ndarray,
    admitted: np.ndarray,
) -> np.ndarray:
    posterior = np.zeros(
        (len(probability),) + admitted.shape, dtype=np.float64)
    posterior[:, admitted] = probability
    return posterior


def initial_belief(
    counts: np.ndarray,
    config: RayBench,
) -> np.ndarray:
    initial_count = min(config.batch, len(counts))
    return gaussian_filter(
        np.maximum(
            np.mean(
                calibrated_radiance(
                    counts[:initial_count],
                    config.photons_at_white,
                    config.background_photons,
                ),
                axis=0,
            ),
            1e-7,
        ),
        1.0,
    )


def update_belief(
    belief: np.ndarray,
    counts: np.ndarray,
    posterior: np.ndarray,
    support: int,
    config: RayBench,
) -> np.ndarray:
    estimate = backproject(counts, posterior, config)
    old_support = min(support, config.support_cap)
    return np.maximum(
        (
            old_support * belief
            + len(counts) * estimate
        )
        / max(old_support + len(counts), 1),
        1e-7,
    )


def continual_entropy_budget_transport(
    counts: np.ndarray,
    config: RayBench,
    *,
    target_effective_shifts: float,
    bisection_steps: int = 36,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """One-pass transport with entropy, rather than temperature, held fixed."""
    admitted = shift_mask(config.grid, config.shift_radius)
    belief = initial_belief(counts, config)
    support = min(config.batch, len(counts))
    trace = []
    for start in range(support, len(counts), config.batch):
        batch = counts[start:start + config.batch]
        scores = poisson_shift_scores(batch, belief, config, admitted)
        probability, info = entropy_project_scores(
            scores,
            target_effective_shifts,
            bisection_steps=bisection_steps,
        )
        posterior = embed_posterior(probability, admitted)
        belief = update_belief(
            belief, batch, posterior, support, config)
        support += len(batch)
        record = {"frames": support, **info}
        if reference is not None:
            record.update(metrics(belief, reference))
        trace.append(record)
    return belief, {
        "mode": "continual_entropy_budget_transport",
        "passes_over_frames": 1,
        "target_effective_shifts": target_effective_shifts,
        "trace": trace,
    }


def cross_predictive_evidence(
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    budgets: tuple[float, ...],
    *,
    pixels: int,
    bisection_steps: int,
) -> np.ndarray:
    """Bidirectional evidence: each thinned stream predicts its complement."""
    evidence = []
    for budget in budgets:
        first, _ = entropy_project_scores(
            first_scores, budget, bisection_steps=bisection_steps)
        second, _ = entropy_project_scores(
            second_scores, budget, bisection_steps=bisection_steps)
        first_to_second = np.mean(logsumexp(
            np.log(np.maximum(first, 1e-30)) + second_scores,
            axis=1,
        ))
        second_to_first = np.mean(logsumexp(
            np.log(np.maximum(second, 1e-30)) + first_scores,
            axis=1,
        ))
        evidence.append(
            0.5 * (first_to_second + second_to_first) / max(pixels, 1))
    return np.asarray(evidence, dtype=np.float64)


class StreamingEntropyTransport:
    """Persistent batch-ingest form of the cumulative entropy operator."""

    def __init__(
        self,
        config: RayBench,
        entropy: EntropyBench,
        *,
        evidence_decay: float = 1.0,
    ) -> None:
        self.config = config
        self.entropy = entropy
        self.evidence_decay = float(evidence_decay)
        self.admitted = shift_mask(
            config.grid, config.shift_radius)
        first_scale = entropy.split_probability
        second_scale = 1.0 - first_scale
        self.first_config = replace(
            config,
            photons_at_white=config.photons_at_white * first_scale,
            background_photons=config.background_photons * first_scale,
        )
        self.second_config = replace(
            config,
            photons_at_white=config.photons_at_white * second_scale,
            background_photons=config.background_photons * second_scale,
        )
        self.reset()

    def reset(self) -> None:
        """Discard the gauge after a cut, reconfiguration, or hard move."""
        self.belief: np.ndarray | None = None
        self.support = 0
        self.evidence_state = np.zeros(
            len(self.entropy.budgets), dtype=np.float64)
        self.selected_index = len(self.entropy.budgets) - 1
        self.support_changes = 0
        self.trace: list[dict] = []
        self.rng = np.random.default_rng(
            self.config.seed + self.entropy.split_seed_offset)

    @property
    def selected_budget(self) -> float:
        return float(self.entropy.budgets[self.selected_index])

    def push(
        self,
        batch: np.ndarray,
        *,
        discontinuity: bool = False,
        reference: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Consume one batch once and return the current radiance belief."""
        counts = np.asarray(batch)
        expected = (
            len(counts), self.config.grid, self.config.grid)
        if not len(counts) or counts.ndim != 3 or counts.shape != expected:
            raise ValueError(
                "batch must have shape (frames, grid, grid)")
        if discontinuity:
            self.reset()
        if self.belief is None:
            self.belief = initial_belief(counts, self.config)
            self.support = len(counts)
            record = {
                "frames": self.support,
                "initialized": True,
                "selected_budget": self.selected_budget,
            }
            self.trace.append(record)
            return self.belief, record

        first = self.rng.binomial(
            counts,
            self.entropy.split_probability,
        ).astype(counts.dtype)
        second = counts - first
        first_scores = poisson_shift_scores(
            first, self.belief, self.first_config, self.admitted)
        second_scores = poisson_shift_scores(
            second, self.belief, self.second_config, self.admitted)
        batch_evidence = cross_predictive_evidence(
            first_scores,
            second_scores,
            self.entropy.budgets,
            pixels=self.config.grid * self.config.grid,
            bisection_steps=self.entropy.bisection_steps,
        )
        relative = batch_evidence - np.max(batch_evidence)
        self.evidence_state = (
            self.evidence_decay * self.evidence_state + relative)
        evidence_best = int(np.argmax(self.evidence_state))
        steps = max(
            int(self.entropy.max_budget_steps_per_batch), 1)
        previous_index = self.selected_index
        self.selected_index += int(np.clip(
            evidence_best - self.selected_index, -steps, steps))
        if self.selected_index != previous_index:
            self.support_changes += 1

        full_scores = poisson_shift_scores(
            counts, self.belief, self.config, self.admitted)
        probability, info = entropy_project_scores(
            full_scores,
            self.selected_budget,
            bisection_steps=self.entropy.bisection_steps,
        )
        posterior = embed_posterior(probability, self.admitted)
        self.belief = update_belief(
            self.belief,
            counts,
            posterior,
            self.support,
            self.config,
        )
        self.support += len(counts)
        sorted_evidence = np.sort(self.evidence_state)
        margin = (
            sorted_evidence[-1] - sorted_evidence[-2]
            if len(sorted_evidence) > 1 else 0.0
        )
        record = {
            "frames": self.support,
            "initialized": False,
            "selected_budget": self.selected_budget,
            "evidence_best_budget": float(
                self.entropy.budgets[evidence_best]),
            "batch_best_budget": float(self.entropy.budgets[
                int(np.argmax(batch_evidence))]),
            "evidence_margin_per_pixel": float(margin),
            "evidence_state": self.evidence_state.tolist(),
            "batch_cross_evidence": batch_evidence.tolist(),
            **info,
        }
        if reference is not None:
            record.update(metrics(self.belief, reference))
        self.trace.append(record)
        return self.belief, record

    def diagnostics(self) -> dict:
        selected = np.asarray([
            item["selected_budget"] for item in self.trace
            if not item["initialized"]
        ], dtype=np.float64)
        return {
            "mode": "streaming_adaptive_entropy_transport",
            "passes_over_frames": 1,
            "evidence_decay": self.evidence_decay,
            "support_changes": self.support_changes,
            "median_selected_budget": (
                float(np.median(selected))
                if len(selected) else float("nan")),
            "final_selected_budget": self.selected_budget,
            "trace": [
                item for item in self.trace
                if not item["initialized"]
            ],
        }


def continual_adaptive_entropy_transport(
    counts: np.ndarray,
    config: RayBench,
    entropy: EntropyBench,
    *,
    evidence_decay: float,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Compatibility wrapper over the persistent streaming operator."""
    stream = StreamingEntropyTransport(
        config, entropy, evidence_decay=evidence_decay)
    for start in range(0, len(counts), config.batch):
        stream.push(
            counts[start:start + config.batch],
            reference=reference,
        )
    if stream.belief is None:
        raise ValueError("counts must contain at least one frame")
    return stream.belief, stream.diagnostics()


def evaluate(
    estimate: np.ndarray,
    truth: np.ndarray,
    witness: np.ndarray,
    config: RayBench,
) -> dict[str, float]:
    return {
        "witness_log_evidence_per_pixel": marginal_poisson_score(
            witness, estimate, config),
        **metrics(estimate, truth),
    }


def robustness_audit(
    truth: np.ndarray,
    config: RayBench,
    entropy: EntropyBench,
    first_temperature_two: np.ndarray,
    first_cumulative: np.ndarray,
) -> dict:
    """Truth-only repeatability audit; it never feeds the operator."""
    records = []
    for offset in range(max(int(entropy.robustness_seeds), 1)):
        seed = config.seed + offset
        seeded_config = replace(config, seed=seed)
        if offset == 0:
            temperature_two = first_temperature_two
            cumulative = first_cumulative
        else:
            counts, _ = capture(
                truth,
                seeded_config.frames,
                seeded_config.photons_at_white,
                seeded_config.background_photons,
                seeded_config.shift_radius,
                seeded_config.seed,
            )
            temperature_two, _ = continual_likelihood_flow(
                counts,
                seeded_config,
                dimension=float("inf"),
                temperature=2.0,
            )
            cumulative, _ = continual_adaptive_entropy_transport(
                counts,
                seeded_config,
                entropy,
                evidence_decay=1.0,
            )
        baseline_metrics = metrics(temperature_two, truth)
        cumulative_metrics = metrics(cumulative, truth)
        records.append({
            "seed": seed,
            "temperature_two": baseline_metrics,
            "cumulative_entropy": cumulative_metrics,
            "psnr_delta_db": (
                cumulative_metrics["psnr_db"]
                - baseline_metrics["psnr_db"]
            ),
            "ssim_delta": (
                cumulative_metrics["ssim"]
                - baseline_metrics["ssim"]
            ),
        })
    psnr_delta = np.asarray(
        [item["psnr_delta_db"] for item in records])
    ssim_delta = np.asarray(
        [item["ssim_delta"] for item in records])
    return {
        "purpose": (
            "truth-only repeatability audit; no result feeds selection"
        ),
        "records": records,
        "mean_psnr_delta_db": float(np.mean(psnr_delta)),
        "mean_ssim_delta": float(np.mean(ssim_delta)),
        "psnr_wins": int(np.count_nonzero(psnr_delta > 0.0)),
        "ssim_wins": int(np.count_nonzero(ssim_delta > 0.0)),
        "trials": len(records),
    }


def run(
    config: RayBench,
    entropy: EntropyBench,
) -> tuple[dict, dict[str, np.ndarray]]:
    truth = source_image(config.grid)
    counts, shifts = capture(
        truth,
        config.frames,
        config.photons_at_white,
        config.background_photons,
        config.shift_radius,
        config.seed,
    )
    validation, _ = capture(
        truth,
        config.validation_frames,
        config.photons_at_white,
        config.background_photons,
        config.shift_radius,
        config.seed + 1,
    )
    standard, standard_info = continual_ray_transport(counts, config)
    temperature_two, temperature_two_info = continual_likelihood_flow(
        counts, config, dimension=float("inf"), temperature=2.0)

    fixed_records = []
    fixed_images = {}
    fixed_infos = {}
    for budget in entropy.budgets:
        estimate, info = continual_entropy_budget_transport(
            counts,
            config,
            target_effective_shifts=budget,
            bisection_steps=entropy.bisection_steps,
        )
        fixed_images[budget] = estimate
        fixed_infos[budget] = info
        fixed_records.append({
            "target_effective_shifts": budget,
            **evaluate(estimate, truth, validation, config),
            "mean_temperature": float(np.mean([
                item["temperature"] for item in info["trace"]
            ])),
            "final_temperature": info["trace"][-1]["temperature"],
            "final_effective_shifts": (
                info["trace"][-1]["effective_shifts"]),
        })
    selected_fixed = max(
        fixed_records,
        key=lambda item: item["witness_log_evidence_per_pixel"],
    )
    selected_fixed_image = fixed_images[
        selected_fixed["target_effective_shifts"]]

    adaptive_records = []
    adaptive_images = {}
    adaptive_infos = {}
    for decay in entropy.evidence_decays:
        estimate, info = continual_adaptive_entropy_transport(
            counts,
            config,
            entropy,
            evidence_decay=decay,
        )
        adaptive_images[decay] = estimate
        adaptive_infos[decay] = info
        adaptive_records.append({
            "evidence_decay": decay,
            **evaluate(estimate, truth, validation, config),
            "support_changes": info["support_changes"],
            "median_selected_budget": info["median_selected_budget"],
            "final_selected_budget": info["final_selected_budget"],
        })
    evidence_selected_adaptive = max(
        adaptive_records,
        key=lambda item: item["witness_log_evidence_per_pixel"],
    )
    cumulative_record = next(
        item for item in adaptive_records
        if item["evidence_decay"] == 1.0
    )
    cumulative_image = adaptive_images[1.0]
    cumulative_info = adaptive_infos[1.0]

    unregistered = np.mean(calibrated_radiance(
        counts,
        config.photons_at_white,
        config.background_photons,
    ), axis=0)
    oracle = oracle_registered_mean(counts, shifts, config)
    images = {
        "truth": truth,
        "one_frame": calibrated_radiance(
            counts[0],
            config.photons_at_white,
            config.background_photons,
        ),
        "unregistered": unregistered,
        "oracle": oracle,
        "standard": standard,
        "temperature_two": temperature_two,
        "fixed": selected_fixed_image,
        "adaptive": cumulative_image,
    }
    robustness = robustness_audit(
        truth,
        config,
        entropy,
        temperature_two,
        cumulative_image,
    )
    result = {
        "config": asdict(config),
        "entropy": asdict(entropy),
        "baselines": {
            "unregistered_mean": metrics(unregistered, truth),
            "oracle_registered_mean": metrics(oracle, truth),
            "standard_temperature_four": {
                **evaluate(standard, truth, validation, config),
                "final_effective_shifts": (
                    standard_info["trace"][-1]["effective_shifts"]),
            },
            "temperature_two": {
                **evaluate(
                    temperature_two, truth, validation, config),
                "final_effective_shifts": (
                    temperature_two_info["trace"][-1][
                        "effective_shifts"]),
            },
        },
        "fixed_entropy_budget": {
            "selection_rule": (
                "maximum marginal Poisson evidence on a separate capture"),
            "selected": selected_fixed,
            "sweep": fixed_records,
        },
        "adaptive_entropy_budget": {
            "operator_rule": (
                "each batch uses bidirectional complementary-photon "
                "evidence; cumulative friction retains all past evidence"
            ),
            "primary_cumulative": cumulative_record,
            "primary_trace": cumulative_info["trace"],
            "evidence_selected_decay": {
                **evidence_selected_adaptive,
                "selection_rule": (
                    "maximum marginal Poisson evidence on a separate capture"
                ),
            },
            "sweep": adaptive_records,
        },
        "robustness_audit": robustness,
    }
    return result, images


def render(
    result: dict,
    images: dict[str, np.ndarray],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(18, 7.8))
    panels = (
        ("truth", "Oracle", None),
        ("one_frame", "One sparse photon frame", None),
        ("unregistered", "Unregistered mean", "unregistered_mean"),
        ("oracle", "Oracle-registered mean", "oracle_registered_mean"),
        (
            "standard",
            "Original temperature 4",
            "standard_temperature_four",
        ),
        ("temperature_two", "Fixed temperature 2", "temperature_two"),
        ("fixed", "Heldout-selected entropy budget", "fixed"),
        ("adaptive", "Cumulative-evidence entropy flow", "adaptive"),
    )
    for axis, (key, title, metric_key) in zip(
        axes.ravel()[:8], panels
    ):
        axis.imshow(
            np.clip(images[key], 0.0, 1.0),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        if metric_key == "fixed":
            value = result["fixed_entropy_budget"]["selected"]
        elif metric_key == "adaptive":
            value = result["adaptive_entropy_budget"][
                "primary_cumulative"]
        elif metric_key:
            value = result["baselines"][metric_key]
        else:
            value = None
        if value is not None:
            title += (
                f"\n{value['psnr_db']:.2f} dB, "
                f"SSIM {value['ssim']:.3f}"
            )
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    fixed = result["fixed_entropy_budget"]["sweep"]
    budgets = np.asarray([
        item["target_effective_shifts"] for item in fixed])
    evidence = np.asarray([
        item["witness_log_evidence_per_pixel"] for item in fixed])
    psnr = np.asarray([item["psnr_db"] for item in fixed])
    evidence_axis = axes.ravel()[8]
    evidence_axis.plot(
        budgets, evidence - np.max(evidence), "o-", label="heldout evidence")
    evidence_axis.set_xscale("log", base=2)
    evidence_axis.set_xticks(budgets, [f"{value:g}" for value in budgets])
    evidence_axis.set_xlabel("effective-shift budget K")
    evidence_axis.set_ylabel("relative log evidence / pixel")
    audit_axis = evidence_axis.twinx()
    audit_axis.plot(
        budgets, psnr, "s--", color="tab:orange", label="oracle PSNR audit")
    audit_axis.set_ylabel("PSNR (dB), audit only")
    evidence_axis.set_title("Fixed support sweep")
    lines = evidence_axis.lines + audit_axis.lines
    evidence_axis.legend(
        lines, [line.get_label() for line in lines], fontsize=8)

    trace = result["adaptive_entropy_budget"]["primary_trace"]
    frames = [item["frames"] for item in trace]
    selected = [item["selected_budget"] for item in trace]
    achieved = [item["effective_shifts"] for item in trace]
    trace_axis = axes.ravel()[9]
    trace_axis.step(
        frames, selected, where="post", label="selected K")
    trace_axis.plot(
        frames, achieved, ".", alpha=0.65, label="achieved support")
    trace_axis.set_yscale("log", base=2)
    trace_axis.set_yticks(
        budgets, [f"{value:g}" for value in budgets])
    trace_axis.set_xlabel("frames consumed")
    trace_axis.set_ylabel("effective shifts")
    trace_axis.set_title(
        "Cumulative photon evidence\n"
        "(no forgetting, no oracle)"
    )
    trace_axis.legend(fontsize=8)

    config = result["config"]
    figure.suptitle(
        "Entropy-budgeted sparse registration: "
        f"{config['grid']}×{config['grid']}, "
        f"{config['frames']} frames"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--photons", type=float, default=0.08)
    parser.add_argument("--robustness-seeds", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "high_vision/out/entropy_budget_transport.png"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(
            "high_vision/out/entropy_budget_transport.json"),
    )
    args = parser.parse_args()
    config = RayBench(
        grid=args.grid,
        frames=args.frames,
        photons_at_white=args.photons,
    )
    result, images = run(
        config,
        EntropyBench(robustness_seeds=args.robustness_seeds),
    )
    render(result, images, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "baselines": result["baselines"],
        "selected_fixed": result["fixed_entropy_budget"]["selected"],
        "cumulative_adaptive": result[
            "adaptive_entropy_budget"]["primary_cumulative"],
        "evidence_selected_decay": result[
            "adaptive_entropy_budget"]["evidence_selected_decay"],
        "robustness_audit": {
            key: value
            for key, value in result["robustness_audit"].items()
            if key != "records"
        },
    }, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
