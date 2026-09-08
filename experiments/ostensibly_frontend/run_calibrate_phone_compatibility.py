#!/usr/bin/env python3
"""Cross-fit empirical phone compatibility over frozen geometry scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .calibrated_geometry_fusion import (
    calibration_objective,
    fuse_rank_channels,
    rank_summary,
)
from .conditional_ridge_atlas import load_conditional_ridge_atlas
from .empirical_phone_compatibility import (
    EmpiricalPhoneCompatibility,
    POLICIES,
)
from .physical_interval_geometry import load_physical_interval_atlas
from .run_calibrate_geometry_fusion import (
    DEFAULT_UTTERANCES,
    _cloud_path,
    _payload,
    _windows,
)


def _configuration_key(prior_strength: float, policy: str) -> str:
    return f"{policy}:{prior_strength:g}"


def _target_ranks(
    compatibility: np.ndarray,
    targets: np.ndarray,
    labels: tuple[str, ...],
) -> np.ndarray:
    label_array = np.asarray(labels)
    target_indices = np.asarray(
        [int(np.flatnonzero(label_array == target)[0]) for target in targets],
        dtype=np.int64,
    )
    target_values = compatibility[np.arange(compatibility.shape[0]), target_indices]
    greater = np.sum(compatibility > target_values[:, None], axis=1)
    # Match the runtime's deterministic alphabetical phone tie break.
    earlier = np.arange(len(labels))[None, :] < target_indices[:, None]
    tied_before = np.sum((compatibility == target_values[:, None]) & earlier, axis=1)
    return (1 + greater + tied_before).astype(np.int64)


def _evaluate(
    scores: np.ndarray,
    targets: np.ndarray,
    labels: tuple[str, ...],
    train: np.ndarray,
    test: np.ndarray,
    prior_strength: float,
    policy: str,
) -> np.ndarray:
    calibrator = EmpiricalPhoneCompatibility(
        labels=labels,
        calibration_scores=scores[train],
        calibration_targets=targets[train],
        prior_strength=prior_strength,
        policy=policy,
    )
    compatibility = calibrator.score_matrix(scores[test])
    return _target_ranks(compatibility, targets[test], labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("bdl_topology_atlas", type=Path)
    parser.add_argument("slt_topology_atlas", type=Path)
    parser.add_argument("bdl_physical_atlas", type=Path)
    parser.add_argument("slt_physical_atlas", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--prior-strengths", default="0,2,4,8,16,32")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    prior_strengths = tuple(
        float(value) for value in args.prior_strengths.split(",") if value
    )
    policies = tuple(value for value in args.policies.split(",") if value)
    if (
        len(utterances) < 3
        or not prior_strengths
        or any(not np.isfinite(value) or value < 0.0 for value in prior_strengths)
        or not policies
        or any(value not in POLICIES for value in policies)
    ):
        raise ValueError("phone compatibility calibration configuration is invalid")
    configurations = tuple(
        (strength, policy) for policy in policies for strength in prior_strengths
    )
    payload = _payload()
    topology = {
        "bdl": load_conditional_ridge_atlas(args.bdl_topology_atlas),
        "slt": load_conditional_ridge_atlas(args.slt_topology_atlas),
    }
    physical = {
        "bdl": load_physical_interval_atlas(args.bdl_physical_atlas),
        "slt": load_physical_interval_atlas(args.slt_physical_atlas),
    }
    labels = tuple(sorted(set(topology["bdl"].labels.astype(str))))
    if (
        set(topology["slt"].labels.astype(str)) != set(labels)
        or set(physical["bdl"].labels.astype(str)) != set(labels)
        or set(physical["slt"].labels.astype(str)) != set(labels)
    ):
        raise ValueError("phone compatibility atlases have different labels")

    score_rows = []
    target_rows = []
    utterance_rows = []
    witness_rows = []
    direction_rows = []
    for reference_speaker, query_speaker in (("bdl", "slt"), ("slt", "bdl")):
        windows = _windows(args.arctic_root, query_speaker, utterances)
        for completed, (utterance, ordinal, window) in enumerate(windows, start=1):
            cloud = np.asarray(
                np.load(_cloud_path(args.raw_cache, window, payload)),
                dtype=np.float64,
            )
            topology_ranking = topology[reference_speaker].rank(cloud)
            mass_ranking = physical[reference_speaker].rank_with_weights(
                cloud,
                interval_weight=0.0,
                trajectory_weight=0.0,
                mass_weight=1.0,
            )
            fused = fuse_rank_channels(
                topology_ranking,
                mass_ranking,
                physical_weight=1.0,
                policy="geometric",
            )
            by_phone = {str(row["phone"]): float(row["score"]) for row in fused}
            score_rows.append([by_phone[label] for label in labels])
            target_rows.append(window.label)
            utterance_rows.append(utterance)
            witness_rows.append(f"{query_speaker}:{utterance}:{ordinal}")
            direction_rows.append(f"{reference_speaker}_to_{query_speaker}")
            if completed % 64 == 0 or completed == len(windows):
                print(
                    json.dumps(
                        {
                            "direction": direction_rows[-1],
                            "completed": completed,
                            "total": len(windows),
                        }
                    ),
                    flush=True,
                )
    scores = np.asarray(score_rows, dtype=np.float64)
    targets = np.asarray(target_rows)
    utterance_values = np.asarray(utterance_rows)
    all_indices = np.arange(scores.shape[0])

    crossfit_ranks = {
        _configuration_key(strength, policy): []
        for strength, policy in configurations
    }
    for held_out in utterances:
        train = all_indices[utterance_values != held_out]
        test = all_indices[utterance_values == held_out]
        for strength, policy in configurations:
            key = _configuration_key(strength, policy)
            crossfit_ranks[key].extend(
                _evaluate(scores, targets, labels, train, test, strength, policy)
            )
    summaries = {
        key: rank_summary(ranks) for key, ranks in crossfit_ranks.items()
    }
    selected_strength, selected_policy = min(
        configurations,
        key=lambda item: (
            calibration_objective(
                summaries[_configuration_key(item[0], item[1])]
            ),
            item[0],
            item[1],
        ),
    )

    nested_ranks = []
    nested_folds = []
    for held_out in utterances:
        outer_train = all_indices[utterance_values != held_out]
        outer_test = all_indices[utterance_values == held_out]
        inner_utterances = tuple(value for value in utterances if value != held_out)
        inner_summaries = {}
        for strength, policy in configurations:
            ranks = []
            for inner_held_out in inner_utterances:
                inner_train = outer_train[
                    utterance_values[outer_train] != inner_held_out
                ]
                inner_test = outer_train[
                    utterance_values[outer_train] == inner_held_out
                ]
                ranks.extend(
                    _evaluate(
                        scores,
                        targets,
                        labels,
                        inner_train,
                        inner_test,
                        strength,
                        policy,
                    )
                )
            inner_summaries[_configuration_key(strength, policy)] = rank_summary(
                ranks
            )
        fold_strength, fold_policy = min(
            configurations,
            key=lambda item: (
                calibration_objective(
                    inner_summaries[_configuration_key(item[0], item[1])]
                ),
                item[0],
                item[1],
            ),
        )
        ranks = _evaluate(
            scores,
            targets,
            labels,
            outer_train,
            outer_test,
            fold_strength,
            fold_policy,
        )
        nested_ranks.extend(ranks)
        nested_folds.append(
            {
                "utterance": held_out,
                "selected_prior_strength": fold_strength,
                "selected_policy": fold_policy,
                "validation": rank_summary(ranks),
            }
        )

    calibrator = EmpiricalPhoneCompatibility(
        labels=labels,
        calibration_scores=scores,
        calibration_targets=targets,
        prior_strength=selected_strength,
        policy=selected_policy,
        provenance={
            "geometry": "equal-weight geometric topology-by-mass rank percentile",
            "selection": "leave-one-utterance-out cross-speaker",
            "utterances": list(utterances),
        },
    )
    atlas_path = args.out.with_suffix(".npz")
    calibrator.save(atlas_path)
    result = {
        "method": "cross_fitted_finite_event_phone_compatibility",
        "selection_status": "calibration uses no radio queries",
        "geometry": "equal-weight geometric topology-by-mass rank percentile",
        "utterances": list(utterances),
        "labels": list(labels),
        "query_count": int(scores.shape[0]),
        "configurations": [
            {"prior_strength": strength, "policy": policy}
            for strength, policy in configurations
        ],
        "crossfit_summaries": summaries,
        "selected_prior_strength": selected_strength,
        "selected_policy": selected_policy,
        "selected_crossfit": summaries[
            _configuration_key(selected_strength, selected_policy)
        ],
        "nested_leave_one_utterance_out": {
            "summary": rank_summary(nested_ranks),
            "folds": nested_folds,
        },
        "calibrator": str(atlas_path),
        "witnesses": witness_rows,
        "targets": target_rows,
        "directions": direction_rows,
        "elapsed_seconds": perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "calibrator": str(atlas_path),
                "selected_prior_strength": selected_strength,
                "selected_policy": selected_policy,
                "selected_crossfit": result["selected_crossfit"],
                "nested": result["nested_leave_one_utterance_out"]["summary"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
