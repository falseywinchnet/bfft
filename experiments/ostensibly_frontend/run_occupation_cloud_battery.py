#!/usr/bin/env python3
"""Audit whether occupation-cloud registration discriminates phone identity."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.stats import spearmanr

from .crazy_climber_geometry import (
    CrazyClimberConfig,
    HessianRidgeConfig,
    crazy_climber_occupation,
    multiscale_hessian_ridge_surface,
)
from .depthmap_geometry import DepthmapGeometryConfig, multiscale_morphological_residual
from .fourier_unrolled_densified import (
    DEFAULT_APERTURES,
    fourier_unrolled_densified,
    reassigned_texture_baseline,
)
from .occupation_point_cloud import (
    CloudFitConfig,
    PointCloudConfig,
    cloud_distance,
    fit_phone_cloud,
    marginal_copula_cloud,
    occupation_to_point_cloud,
)
from .run_arctic_calibration import (
    DEFAULT_UTTERANCES,
    HOP_LENGTH,
    TARGET_SAMPLE_RATE,
    _timed_phones,
)
from .run_whole_patch_inventory import LABELS, OFFSETS
from .synthetic_vowels import apply_ssb_probe_channel
from .uncertainty_fusion import _as_float_audio


@dataclass(frozen=True)
class ReferenceWindow:
    label: str
    speaker: str
    utterance: str
    seconds0: float
    seconds1: float
    raw_label: str

    @property
    def duration(self) -> float:
        return self.seconds1 - self.seconds0

    @property
    def key(self) -> tuple[str, str, str, float, float]:
        return (
            self.label,
            self.speaker,
            self.utterance,
            self.seconds0,
            self.seconds1,
        )


def select_duration_matched_windows(
    windows: list[ReferenceWindow],
    target_duration: float,
    witnesses_per_label: int,
    forced: tuple[str, str, float, float] | None = None,
) -> list[ReferenceWindow]:
    """Select witnesses without looking at their geometric match distance."""

    if target_duration <= 0.0 or witnesses_per_label < 1:
        raise ValueError("invalid duration-matched selection")
    selected: list[ReferenceWindow] = []
    for label in sorted({item.label for item in windows}):
        candidates = sorted(
            (item for item in windows if item.label == label),
            key=lambda item: (
                abs(item.duration - target_duration),
                item.utterance,
                item.seconds0,
            ),
        )
        selected.extend(candidates[:witnesses_per_label])
    if forced is not None:
        forced_label, forced_utterance, forced_start, forced_stop = forced
        match = next(
            (
                item
                for item in windows
                if item.label == forced_label
                and item.utterance == forced_utterance
                and np.isclose(item.seconds0, forced_start)
                and np.isclose(item.seconds1, forced_stop)
            ),
            None,
        )
        if match is None:
            raise ValueError("forced positive-control window was not found")
        if match.key not in {item.key for item in selected}:
            selected.append(match)
    return sorted(selected, key=lambda item: item.key)


def _trace_to_cloud(
    trace: np.ndarray,
    morphology_config: DepthmapGeometryConfig,
    ridge_config: HessianRidgeConfig,
    climber_config: CrazyClimberConfig,
    cloud_config: PointCloudConfig,
) -> np.ndarray:
    residual, _ = multiscale_morphological_residual(trace, morphology_config)
    ridge = multiscale_hessian_ridge_surface(residual, ridge_config)
    occupation = crazy_climber_occupation(ridge.saliency, climber_config)
    return occupation_to_point_cloud(occupation.weighted, cloud_config).points


def _load_channel_audio(
    root: Path,
    speaker: str,
    utterance: str,
    utterance_index: int,
    noise_db: float,
) -> np.ndarray:
    wav_path = root / speaker / "wav" / f"{utterance}.wav"
    sample_rate, samples = wavfile.read(wav_path)
    samples = _as_float_audio(samples)
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        samples = resample_poly(samples, TARGET_SAMPLE_RATE, int(sample_rate))
    samples = apply_ssb_probe_channel(
        samples,
        sample_rate=TARGET_SAMPLE_RATE,
        noise_db=noise_db,
        seed=10_000 * (1 + utterance_index) + sum(map(ord, speaker)),
    )
    return samples


def _load_channel_baseline(
    root: Path,
    speaker: str,
    utterance: str,
    utterance_index: int,
    noise_db: float,
) -> np.ndarray:
    samples = _load_channel_audio(
        root,
        speaker,
        utterance,
        utterance_index,
        noise_db,
    )
    densified = fourier_unrolled_densified(
        samples,
        apertures=DEFAULT_APERTURES,
        target_n_fft=2048,
        hop_length=HOP_LENGTH,
        target_crop_rows=256,
        offsets=OFFSETS,
    )
    return reassigned_texture_baseline(
        densified.fused, densified.support_union
    ).merged


def _window_trace(
    baseline: np.ndarray,
    seconds0: float,
    seconds1: float,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers = HOP_LENGTH * np.arange(baseline.shape[1]) / TARGET_SAMPLE_RATE
    indices = np.flatnonzero((centers >= seconds0) & (centers < seconds1))
    if not indices.size:
        midpoint = 0.5 * (seconds0 + seconds1)
        indices = np.asarray((int(np.argmin(np.abs(centers - midpoint))),))
    return np.asarray(baseline[:rows, indices], dtype=np.float64), indices


def _bound_hits(row: dict, config: CloudFitConfig) -> list[str]:
    values_and_bounds = (
        ("row_scale", row["row_scale"], config.row_scale_bounds),
        ("frame_scale", row["frame_scale"], config.frame_scale_bounds),
        ("row_shift", row["row_shift"], config.row_shift_bounds),
        ("frame_shift", row["frame_shift"], config.frame_shift_bounds),
        ("pitch_drift", row["pitch_drift"], config.pitch_drift_bounds),
    )
    hits = []
    for name, value, bounds in values_and_bounds:
        tolerance = 0.01 * (bounds[1] - bounds[0])
        if value - bounds[0] <= tolerance or bounds[1] - value <= tolerance:
            hits.append(name)
    return hits


def _label_ranking(rows: list[dict], distance_key: str) -> list[dict]:
    labels = sorted({str(row["label"]) for row in rows})
    ranking = []
    for label in labels:
        witnesses = [row for row in rows if row["label"] == label]
        best = min(witnesses, key=lambda row: (row[distance_key], row["key"]))
        ranking.append(
            {
                "phone": label,
                "distance": best[distance_key],
                "witness_key": best["key"],
                "bound_hits": best["bound_hits"],
            }
        )
    return sorted(ranking, key=lambda row: (row["distance"], row["phone"]))


def _cloud_statistics(points: np.ndarray) -> dict[str, float]:
    return {
        "row_standard_deviation": float(np.std(points[:, 0])),
        "frame_standard_deviation": float(np.std(points[:, 1])),
        "height_mean": float(np.mean(points[:, 2])),
        "height_standard_deviation": float(np.std(points[:, 2])),
    }


def _spearman(rows: list[dict], field: str) -> float:
    value = spearmanr(
        [row[field] for row in rows],
        [row["postfit_distance"] for row in rows],
    ).statistic
    return float(value) if np.isfinite(value) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("pair", type=Path)
    parser.add_argument("--speaker", default="bdl")
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--query-arctic-speaker")
    parser.add_argument("--query-arctic-utterance", default="arctic_a0019")
    parser.add_argument("--query-phone", default="R")
    parser.add_argument(
        "--query-phone-ordinal",
        type=int,
        default=2,
        help="one-based occurrence of --query-phone in the query utterance",
    )
    parser.add_argument("--witnesses-per-label", type=int, default=2)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--points", type=int, default=32768)
    parser.add_argument("--fit-points", type=int, default=1024)
    parser.add_argument("--fit-iterations", type=int, default=24)
    parser.add_argument("--fit-population", type=int, default=6)
    parser.add_argument("--height-metric-scale", type=float, default=2.0)
    parser.add_argument(
        "--distance-mode",
        choices=("chamfer", "sliced_wasserstein"),
        default="chamfer",
    )
    parser.add_argument("--marginal-copula", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(args.utterances.split(","))
    if args.witnesses_per_label < 1:
        raise ValueError("witnesses per label must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "occupation_cloud_battery_checkpoint.json"
    checkpoint_rows = []
    if checkpoint_path.exists():
        checkpoint_rows = json.loads(checkpoint_path.read_text())["rows"]
    completed = {row["key"]: row for row in checkpoint_rows}

    with np.load(args.pair) as saved:
        radio_target_trace = np.asarray(
            saved["target_trace"][: args.rows], dtype=np.float64
        )
        saved_positive_trace = np.asarray(
            saved["source_trace"][: args.rows], dtype=np.float64
        )
    query_metadata = {
        "source": "daveandsimon",
        "hypothesized_phone": "R",
        "seconds0": 0.07466666666666667,
        "seconds1": 0.224,
    }
    if args.query_arctic_speaker is None:
        target_trace = radio_target_trace
    else:
        if args.query_phone_ordinal < 1:
            raise ValueError("query phone ordinal must be at least one")
        query_lab = (
            args.arctic_root
            / args.query_arctic_speaker
            / "lab"
            / f"{args.query_arctic_utterance}.lab"
        )
        query_windows = [
            item
            for item in _timed_phones(query_lab)
            if item[0] == args.query_phone
        ]
        if args.query_phone_ordinal > len(query_windows):
            raise ValueError("query phone ordinal exceeds available occurrences")
        query_window = query_windows[args.query_phone_ordinal - 1]
        query_utterance_index = utterances.index(args.query_arctic_utterance)
        query_baseline = _load_channel_baseline(
            args.arctic_root,
            args.query_arctic_speaker,
            args.query_arctic_utterance,
            query_utterance_index,
            args.noise_db,
        )
        target_trace, query_indices = _window_trace(
            query_baseline,
            query_window[1],
            query_window[2],
            args.rows,
        )
        query_metadata = {
            "source": "cmu_arctic",
            "speaker": args.query_arctic_speaker,
            "utterance": args.query_arctic_utterance,
            "hypothesized_phone": args.query_phone,
            "phone_ordinal": args.query_phone_ordinal,
            "seconds0": query_window[1],
            "seconds1": query_window[2],
            "frame0": int(query_indices[0]),
            "frame1_exclusive": int(query_indices[-1] + 1),
        }
    target_duration = target_trace.shape[1] * HOP_LENGTH / TARGET_SAMPLE_RATE

    available: list[ReferenceWindow] = []
    for utterance in utterances:
        lab_path = (
            args.arctic_root / args.speaker / "lab" / f"{utterance}.lab"
        )
        for label, seconds0, seconds1, raw_label in _timed_phones(lab_path):
            if label in LABELS:
                available.append(
                    ReferenceWindow(
                        label,
                        args.speaker,
                        utterance,
                        seconds0,
                        seconds1,
                        raw_label,
                    )
                )
    forced = ("R", "arctic_a0019", 1.83, 1.95)
    selected = select_duration_matched_windows(
        available,
        target_duration,
        args.witnesses_per_label,
        forced=forced,
    )

    morphology_config = DepthmapGeometryConfig()
    ridge_config = HessianRidgeConfig()
    climber_config = CrazyClimberConfig()
    cloud_config = PointCloudConfig(point_count=args.points)
    fit_config = CloudFitConfig(
        fit_point_count=args.fit_points,
        row_metric_scale=1.0 if args.marginal_copula else 0.25,
        height_metric_scale=args.height_metric_scale,
        distance_mode=args.distance_mode,
        global_iterations=args.fit_iterations,
        population_size=args.fit_population,
    )
    target_cloud = _trace_to_cloud(
        target_trace,
        morphology_config,
        ridge_config,
        climber_config,
        cloud_config,
    )
    if args.marginal_copula:
        target_cloud = marginal_copula_cloud(target_cloud)
    target_statistics = _cloud_statistics(target_cloud)

    baselines: dict[str, np.ndarray] = {}
    rows = []
    exact_positive_reproduced = False
    for index, window in enumerate(selected):
        key = (
            f"{window.speaker}:{window.utterance}:{window.label}:"
            f"{window.seconds0:.5f}-{window.seconds1:.5f}"
        )
        if key in completed:
            rows.append(completed[key])
            exact_positive_reproduced |= bool(
                completed[key].get("is_exact_positive", False)
            )
            print(
                f"resume {index + 1}/{len(selected)} {window.label} "
                f"{completed[key]['postfit_distance']:.5f}",
                flush=True,
            )
            continue
        if window.utterance not in baselines:
            utterance_index = utterances.index(window.utterance)
            print(f"building baseline {window.utterance}", flush=True)
            baselines[window.utterance] = _load_channel_baseline(
                args.arctic_root,
                args.speaker,
                window.utterance,
                utterance_index,
                args.noise_db,
            )
        reference_trace, frame_indices = _window_trace(
            baselines[window.utterance],
            window.seconds0,
            window.seconds1,
            args.rows,
        )
        is_exact_positive = bool(
            window.label == forced[0]
            and window.utterance == forced[1]
            and np.isclose(window.seconds0, forced[2])
            and np.isclose(window.seconds1, forced[3])
        )
        if is_exact_positive:
            exact_positive_reproduced = bool(
                np.array_equal(reference_trace, saved_positive_trace)
            )
            if not exact_positive_reproduced:
                raise RuntimeError("exact positive-control trace did not reproduce")
        reference_cloud = _trace_to_cloud(
            reference_trace,
            morphology_config,
            ridge_config,
            climber_config,
            cloud_config,
        )
        if args.marginal_copula:
            reference_cloud = marginal_copula_cloud(reference_cloud)
        reference_statistics = _cloud_statistics(reference_cloud)
        if args.marginal_copula:
            fit_distance = cloud_distance(target_cloud, reference_cloud, fit_config)
            fit_values = {
                "prefit_distance": fit_distance,
                "postfit_distance": fit_distance,
                "row_scale": 1.0,
                "frame_scale": 1.0,
                "row_shift": 0.0,
                "frame_shift": 0.0,
                "pitch_drift": 0.0,
                "optimizer_success": True,
            }
        else:
            fit = fit_phone_cloud(target_cloud, reference_cloud, fit_config)
            fit_values = {
                "prefit_distance": fit.prefit_distance,
                "postfit_distance": fit.postfit_distance,
                "row_scale": fit.row_scale,
                "frame_scale": fit.frame_scale,
                "row_shift": fit.row_shift,
                "frame_shift": fit.frame_shift,
                "pitch_drift": fit.pitch_drift,
                "optimizer_success": fit.optimizer_success,
            }
        row = {
            "key": key,
            "label": window.label,
            "speaker": window.speaker,
            "utterance": window.utterance,
            "seconds0": window.seconds0,
            "seconds1": window.seconds1,
            "duration_seconds": window.duration,
            "frame0": int(frame_indices[0]),
            "frame1_exclusive": int(frame_indices[-1] + 1),
            "frame_count": int(frame_indices.size),
            "is_exact_positive": is_exact_positive,
            **fit_values,
            "relative_improvement": (
                1.0
                - fit_values["postfit_distance"]
                / max(fit_values["prefit_distance"], 1e-30)
            ),
            **reference_statistics,
        }
        row["bound_hits"] = _bound_hits(row, fit_config)
        rows.append(row)
        checkpoint_path.write_text(
            json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"fit {index + 1}/{len(selected)} {window.label} "
            f"{fit_values['postfit_distance']:.5f}",
            flush=True,
        )

    prefit_ranking = _label_ranking(rows, "prefit_distance")
    postfit_ranking = _label_ranking(rows, "postfit_distance")
    r_rank = next(
        index + 1
        for index, item in enumerate(postfit_ranking)
        if item["phone"] == "R"
    )
    best_r = next(item for item in postfit_ranking if item["phone"] == "R")
    best_non_r = next(item for item in postfit_ranking if item["phone"] != "R")
    exact_positive = next(row for row in rows if row["is_exact_positive"])
    instance_ranking = sorted(
        rows, key=lambda row: (row["postfit_distance"], row["key"])
    )
    exact_positive_rank = next(
        index + 1
        for index, row in enumerate(instance_ranking)
        if row["is_exact_positive"]
    )
    result = {
        "method": "duration_matched_identical_occupation_cloud_phone_screen",
        "query": {
            **query_metadata,
            "trace_shape": list(target_trace.shape),
            "duration_seconds": target_duration,
            "cloud_statistics": target_statistics,
        },
        "reference_selection": {
            "speaker": args.speaker,
            "utterances": list(utterances),
            "witnesses_per_label": args.witnesses_per_label,
            "selection_uses_geometry": False,
            "selected_instances": len(selected),
            "selected_labels": len({item.label for item in selected}),
            "exact_positive_reproduced": exact_positive_reproduced,
        },
        "pipeline": {
            "morphology": asdict(morphology_config),
            "ridge": asdict(ridge_config),
            "climbers": asdict(climber_config),
            "point_cloud": asdict(cloud_config),
            "fit": asdict(fit_config),
            "marginal_copula": args.marginal_copula,
        },
        "summary": {
            "r_label_rank": r_rank,
            "label_count": len(postfit_ranking),
            "best_r_distance": best_r["distance"],
            "best_non_r_phone": best_non_r["phone"],
            "best_non_r_distance": best_non_r["distance"],
            "r_margin_vs_best_non_r": best_non_r["distance"] - best_r["distance"],
            "exact_positive_instance_rank": exact_positive_rank,
            "instance_count": len(rows),
            "exact_positive_prefit_distance": exact_positive["prefit_distance"],
            "exact_positive_postfit_distance": exact_positive["postfit_distance"],
            "instances_with_bound_hits": sum(bool(row["bound_hits"]) for row in rows),
            "spearman_distance_vs_reference_row_spread": _spearman(
                rows, "row_standard_deviation"
            ),
            "spearman_distance_vs_reference_frame_spread": _spearman(
                rows, "frame_standard_deviation"
            ),
            "spearman_distance_vs_reference_height_mean": _spearman(
                rows, "height_mean"
            ),
        },
        "prefit_label_ranking": prefit_ranking,
        "postfit_label_ranking": postfit_ranking,
        "instance_ranking": instance_ranking,
    }
    output = args.out / "occupation_cloud_battery.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
