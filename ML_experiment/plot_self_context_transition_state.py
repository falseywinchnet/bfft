#!/usr/bin/env python3
"""Crossover-aligned optimizer-state atlas for Fusion versus Muon."""
from __future__ import annotations

import html
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAIRED = ROOT / "results_self_context_muon_transition_state.json"
FUSION = ROOT / "results_fusion_request_spectrum_transition.json"
SHADOW = ROOT / "results_fusion_shadow_muon_transition.json"
OUTPUT = ROOT / "self_context_transition_state.svg"

OFFSETS = list(range(-150, 26, 5))
MUON = "muon"
FUSION_NAME = "matrix_transport_longitudinal_fusion"


def run_map(payload, optimizer=None):
    return {
        (run["scenario"], run["seed"]): run
        for run in payload["runs"]
        if optimizer is None or run["optimizer"] == optimizer
    }


def history_map(run):
    return {row["step"]: row for row in run["history"]}


def path(points):
    return " ".join(
        ("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
        for index, (x, y) in enumerate(points)
    )


def polygon(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def x_map(offset, left, width):
    return left + (offset - OFFSETS[0]) / (OFFSETS[-1] - OFFSETS[0]) * width


def y_map(value, top, height, low=0.0, high=1.0):
    value = max(low, min(high, value))
    return top + (high - value) / (high - low) * height


def aggregate(rows_by_key, crosses, metric, transform=lambda value: value):
    result = {}
    for offset in OFFSETS:
        values = []
        for key, run in rows_by_key.items():
            row = history_map(run).get(crosses[key] + offset)
            if row is not None and row.get(metric) is not None:
                values.append(transform(row[metric]))
        if values:
            result[offset] = (
                min(values), statistics.median(values), max(values)
            )
    return result


def paired_ratio(muon_runs, fusion_runs, crosses):
    result = {}
    for offset in OFFSETS:
        values = []
        for key in crosses:
            muon = history_map(muon_runs[key]).get(crosses[key] + offset)
            fusion = history_map(fusion_runs[key]).get(crosses[key] + offset)
            if muon is not None and fusion is not None:
                ratio = muon["relative_mse"] / fusion["relative_mse"]
                values.append(math.log10(max(ratio, 1e-12)))
        if values:
            result[offset] = (
                min(values), statistics.median(values), max(values)
            )
    return result


def add_series(out, aggregate_values, left, top, width, height, color,
               low=0.0, high=1.0, dash=""):
    offsets = sorted(aggregate_values)
    median = [
        (x_map(offset, left, width),
         y_map(aggregate_values[offset][1], top, height, low, high))
        for offset in offsets
    ]
    lower = [
        (x_map(offset, left, width),
         y_map(aggregate_values[offset][0], top, height, low, high))
        for offset in offsets
    ]
    upper = [
        (x_map(offset, left, width),
         y_map(aggregate_values[offset][2], top, height, low, high))
        for offset in reversed(offsets)
    ]
    out.append(
        f'<polygon points="{polygon(lower + upper)}" fill="{color}" opacity=".10"/>'
    )
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(
        f'<path d="{path(median)}" fill="none" stroke="{color}" '
        f'stroke-width="3"{dash_attr}/>'
    )


def add_panel(out, left, top, width, height, title, subtitle,
              low=0.0, high=1.0, ticks=(0, .25, .5, .75, 1)):
    out.append(f'<text x="{left}" y="{top-31}" class="panel">{html.escape(title)}</text>')
    out.append(f'<text x="{left}" y="{top-12}" class="sub">{html.escape(subtitle)}</text>')
    out.append(
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
        'fill="#fff" stroke="#d5d9de"/>'
    )
    for tick in ticks:
        y = y_map(tick, top, height, low, high)
        out.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left+width}" y2="{y:.2f}" '
            'stroke="#eceef0"/>'
        )
        label = f"{10 ** tick:.1f}×" if low < 0 else f"{tick:.2g}"
        out.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" class="tick">{label}</text>'
        )
    for offset in (-150, -100, -50, 0, 25):
        x = x_map(offset, left, width)
        stroke = "#9da4ae" if offset == 0 else "#f0f1f3"
        dash = ' stroke-dasharray="5 4"' if offset == 0 else ""
        out.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+height}" '
            f'stroke="{stroke}"{dash}/>'
        )
        out.append(
            f'<text x="{x:.2f}" y="{top+height+18}" text-anchor="middle" class="tick">{offset:+d}</text>'
        )
    out.append(
        f'<text x="{left+width/2}" y="{top+height+39}" text-anchor="middle" class="sub">updates relative to first Muon/Fusion crossover</text>'
    )


def legend(out, x, y, entries):
    for label, color, dash in entries:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(
            f'<line x1="{x}" y1="{y}" x2="{x+28}" y2="{y}" stroke="{color}" '
            f'stroke-width="3"{dash_attr}/>'
        )
        out.append(f'<text x="{x+36}" y="{y+4}" class="legend">{html.escape(label)}</text>')
        x += 205


def main():
    paired = json.loads(PAIRED.read_text())
    fusion_payload = json.loads(FUSION.read_text())
    shadow_payload = json.loads(SHADOW.read_text())

    muon_runs = run_map(paired, MUON)
    paired_fusion_runs = run_map(paired, FUSION_NAME)
    fusion_runs = run_map(fusion_payload, FUSION_NAME)
    shadow_runs = run_map(shadow_payload, FUSION_NAME)
    crosses = {}
    for key in muon_runs:
        muon_history = history_map(muon_runs[key])
        fusion_history = history_map(paired_fusion_runs[key])
        crosses[key] = next(
            step for step in sorted(set(muon_history) & set(fusion_history))
            if muon_history[step]["relative_mse"]
            < fusion_history[step]["relative_mse"]
        )

    width, height = 1500, 990
    panel_w, panel_h = 610, 270
    panels = ((95, 190), (795, 190), (95, 590), (795, 590))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f7f3"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#20252b}'
        '.title{font-size:27px;font-weight:750}.deck{font-size:14px;fill:#4f5863}'
        '.panel{font-size:17px;font-weight:700}.sub{font-size:12px;fill:#606975}'
        '.tick{font-size:11px;fill:#68727d}.legend{font-size:12px;font-weight:650}'
        '.callout{font-size:13px;font-weight:650;fill:#27303a}</style>',
        '<text x="750" y="42" text-anchor="middle" class="title">The Fusion → polar crossover is created by the optimizer trajectory</text>',
        '<text x="750" y="69" text-anchor="middle" class="deck">Six paired self-context runs · fixed population and moving minibatches · crossover-aligned optimizer state</text>',
        '<rect x="160" y="96" width="1180" height="52" rx="8" fill="#e9eef3"/>',
        '<text x="750" y="118" text-anchor="middle" class="callout">Muon develops a narrow, gradient-aligned residual before it wins. A Muon shadow carried along Fusion does not.</text>',
        '<text x="750" y="138" text-anchor="middle" class="sub">The apparent trigger is therefore a signature of Muon’s own dynamics, not a state-independent phase detector available to Fusion.</text>',
    ]

    # Panel 1: loss crossover.
    left, top = panels[0]
    add_panel(out, left, top, panel_w, panel_h,
              "Outcome transition", "log₁₀(Muon relative MSE / Fusion relative MSE)",
              low=-.4, high=1.2, ticks=(-.3, 0, .3, .6, .9, 1.2))
    ratio = paired_ratio(muon_runs, paired_fusion_runs, crosses)
    add_series(out, ratio, left, top, panel_w, panel_h, "#7b3fb4", low=-.4, high=1.2)
    zero_y = y_map(0, top, panel_h, -.4, 1.2)
    out.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left+panel_w}" y2="{zero_y:.2f}" stroke="#20252b" stroke-width="1.5"/>')
    legend(out, left+215, top+22, [("Muon / Fusion", "#7b3fb4", "")])

    # Panel 2: state which exists on Muon's own trajectory.
    left, top = panels[1]
    add_panel(out, left, top, panel_w, panel_h,
              "Muon’s own pre-polar request", "The successful retrospective signature")
    muon_rank = aggregate(muon_runs, crosses, "state_request_stable_rank_fraction")
    muon_align = aggregate(muon_runs, crosses, "state_request_gradient_cosine")
    add_series(out, muon_rank, left, top, panel_w, panel_h, "#d05d38")
    add_series(out, muon_align, left, top, panel_w, panel_h, "#176a87")
    legend(out, left+105, top+22,
           [("participation-rank fraction", "#d05d38", ""),
            ("request/gradient cosine", "#176a87", "")])

    # Panel 3: same observer carried on Fusion's trajectory.
    left, top = panels[2]
    add_panel(out, left, top, panel_w, panel_h,
              "Shadow Muon state while Fusion acts", "Same EMA/polar observer; different parameter trajectory")
    shadow_rank = aggregate(shadow_runs, crosses, "state_shadow_muon_stable_rank_fraction")
    shadow_align = aggregate(shadow_runs, crosses, "state_shadow_muon_request_gradient_cosine")
    add_series(out, shadow_rank, left, top, panel_w, panel_h, "#d05d38")
    add_series(out, shadow_align, left, top, panel_w, panel_h, "#176a87")
    legend(out, left+105, top+22,
           [("participation-rank fraction", "#d05d38", ""),
            ("request/gradient cosine", "#176a87", "")])

    # Panel 4: Fusion's native state contains no sharp corresponding event.
    left, top = panels[3]
    add_panel(out, left, top, panel_w, panel_h,
              "Fusion’s native request", "No spectral or polar event at the loss crossover")
    fusion_rank = aggregate(fusion_runs, crosses, "state_fusion_request_stable_rank_fraction")
    fusion_polar = aggregate(fusion_runs, crosses, "state_fusion_request_polar_cosine")
    add_series(out, fusion_rank, left, top, panel_w, panel_h, "#b02668")
    add_series(out, fusion_polar, left, top, panel_w, panel_h, "#2f7d53")
    legend(out, left+105, top+22,
           [("participation-rank fraction", "#b02668", ""),
            ("request/polar cosine", "#2f7d53", "")])

    pop = sorted(cross for (scenario, _), cross in crosses.items() if scenario == "population")
    mini = sorted(cross for (scenario, _), cross in crosses.items() if scenario == "minibatch")
    out.append(
        f'<text x="750" y="952" text-anchor="middle" class="sub">Crossover updates: population {pop}; minibatch {mini}. Lines are medians across all six runs; translucent bands are full ranges.</text>'
    )
    out.append('</svg>')
    OUTPUT.write_text("\n".join(out))
    print(OUTPUT)


if __name__ == "__main__":
    main()
