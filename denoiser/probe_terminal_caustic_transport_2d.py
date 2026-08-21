"""Gate the continuous scalar caustic on the full 2-D corruption catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fmmt_certified import denoise_fmmt
from .probe_population_phase_integral_2d import CONDITIONS
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt
from .terminal_caustic_transport_2d import (
    phase_integrated_terminal_caustic_readout_2d,
)


def _baseline_rows(path: Path | None, phase_count: int) -> dict[tuple[str, str], dict]:
    if path is None:
        return {}
    record = json.loads(path.read_text())
    result = {}
    for row in record["rows"]:
        matched = [
            item for item in row["counts"]
            if int(item["phase_count"]) == phase_count
        ]
        if len(matched) != 1:
            raise ValueError("baseline record lacks the requested phase count")
        result[(row["source"], row["condition"])] = matched[0]
    return result


def run(
    size: int,
    selected_sources: tuple[str, ...],
    phase_count: int,
    angular_count: int,
    quantile_count: int,
    baseline_record: Path | None,
) -> dict:
    catalogue = sources(size)
    baselines = _baseline_rows(baseline_record, phase_count)
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        for label, kind, amount, density in CONDITIONS:
            observation = (
                truth
                if kind is None
                else corrupt(
                    truth,
                    kind,
                    amount=amount,
                    density=density,
                    seed=11107,
                )
            )
            caustic, diagnostic = (
                phase_integrated_terminal_caustic_readout_2d(
                    observation,
                    angular_count=angular_count,
                    quantile_count=quantile_count,
                    phase_count=phase_count,
                )
            )
            fmmt = denoise_fmmt(observation)[0]
            row = {
                "source": source,
                "condition": label,
                "observation": metrics(observation, truth),
                "terminal_scalar_caustic": metrics(caustic, truth),
                "integrated_fmmt": metrics(fmmt, truth),
                "diagnostic": diagnostic,
            }
            baseline = baselines.get((source, label))
            if baseline is not None:
                row["hj_simplex_branch_barycenter"] = baseline[
                    "causal_phase_average_hj_simplex_collision_barycenter"]
            rows.append(row)

    methods = ["observation", "terminal_scalar_caustic", "integrated_fmmt"]
    matched_baselines = bool(rows) and all(
        "hj_simplex_branch_barycenter" in row for row in rows)
    if matched_baselines:
        methods.append("hj_simplex_branch_barycenter")

    def summarize(selected: list[dict]) -> dict:
        return {
            method: {
                key: float(np.mean([row[method][key] for row in selected]))
                for key in selected[0][method]
            }
            for method in methods
        }

    result = {
        "purpose": (
            "test whether the scalar pushforward Jacobian supplies the "
            "continuous terminal interface concentration missing from the "
            "causal-simplex HJ branch barycenter"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "conditions": [row[0] for row in CONDITIONS],
        "phase_count": int(phase_count),
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "baseline_record": str(baseline_record) if baseline_record else None,
        "summary": summarize(rows),
        "by_source": {
            source: summarize([row for row in rows if row["source"] == source])
            for source in selected_sources
        },
        "by_condition": {
            condition: summarize([
                row for row in rows if row["condition"] == condition
            ])
            for condition, *_ in CONDITIONS
        },
        "rows": rows,
    }
    if matched_baselines:
        result["case_wins_against_branch_barycenter"] = {
            metric: int(sum(
                row["terminal_scalar_caustic"][metric]
                < row["hj_simplex_branch_barycenter"][metric]
                for row in rows
            ))
            for metric in ("mse",)
        }
        result["case_wins_against_branch_barycenter"].update({
            metric: int(sum(
                row["terminal_scalar_caustic"][metric]
                > row["hj_simplex_branch_barycenter"][metric]
                for row in rows
            ))
            for metric in ("ssim", "edge_retention")
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument(
        "--sources",
        default=(
            "cameraman,tapered hair,geometric interfaces,woven chirps,"
            "line drawing,multiscale blobs"
        ),
    )
    parser.add_argument("--phase-count", type=int, default=4)
    parser.add_argument("--angular-count", type=int, default=4)
    parser.add_argument("--quantile-count", type=int, default=16)
    parser.add_argument("--baseline-record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
        args.phase_count,
        args.angular_count,
        args.quantile_count,
        args.baseline_record,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "case_wins_against_branch_barycenter": result.get(
            "case_wins_against_branch_barycenter"),
    }, indent=2))


if __name__ == "__main__":
    main()
