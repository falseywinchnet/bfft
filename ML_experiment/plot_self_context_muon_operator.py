#!/usr/bin/env python3
"""Dependency-free SVG for the self-context Muon operator embedding."""
from __future__ import annotations

import html
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_self_context_muon_operator.json"
OUTPUT = ROOT / "self_context_muon_operator.svg"
ORDER = (
    "adamw", "muon", "matrix_transport_nesterov",
    "matrix_transport_longitudinal_fusion",
)
LABELS = {
    "adamw": "AdamW", "muon": "Muon",
    "matrix_transport_nesterov": "Nesterov Matrix Transport",
    "matrix_transport_longitudinal_fusion": "Longitudinal fusion",
}
COLORS = {
    "adamw": "#6f63d9", "muon": "#d26b38",
    "matrix_transport_nesterov": "#1261a0",
    "matrix_transport_longitudinal_fusion": "#b4216b",
}


def points(steps, values, left, top, width, height):
    answer = []
    for step, value in zip(steps, values):
        x = left + (step - 1) / 499 * width
        log_value = min(.2, max(-4.5, math.log10(max(value, 1e-12))))
        y = top + (.2 - log_value) / 4.7 * height
        answer.append((x, y))
    return answer


def path(values):
    return " ".join(("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
                    for index, (x, y) in enumerate(values))


def polygon(values):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in values)


def main():
    payload = json.loads(RESULTS.read_text())
    runs = payload["runs"]
    width, height = 1400, 680
    panel_w, panel_h = 600, 360
    panels = ((80, 160, "population", "Fixed population through self-context"),
              (750, 160, "minibatch", "Moving minibatches through self-context"))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#20242a}'
        '.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#59606b}'
        '.panel{font-size:17px;font-weight:700}.tick{font-size:11px;fill:#68707b}'
        '.legend{font-size:12px;font-weight:600}.result{font-size:12px}</style>',
        '<text x="700" y="38" text-anchor="middle" class="title">Muon’s operator learned through the full self-context network</text>',
        '<text x="700" y="62" text-anchor="middle" class="sub">16-dimensional orthogonal target · covariance condition 10⁴ · LR .003 · three paired seeds</text>',
    ]
    legend_x = 170
    for optimizer in ORDER:
        out.append(f'<line x1="{legend_x}" y1="105" x2="{legend_x+30}" y2="105" stroke="{COLORS[optimizer]}" stroke-width="4"/>')
        out.append(f'<text x="{legend_x+38}" y="109" class="legend">{html.escape(LABELS[optimizer])}</text>')
        legend_x += 220 if optimizer in {"adamw", "muon"} else 300

    for left, top, scenario, title in panels:
        out.append(f'<text x="{left}" y="{top-20}" class="panel">{title}</text>')
        out.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#fff" stroke="#d7d8dc"/>')
        for value, label in ((1, "10⁰"), (.1, "10⁻¹"), (.01, "10⁻²"),
                             (.003, "3×10⁻³"), (.001, "10⁻³"), (.0001, "10⁻⁴")):
            y = points([1], [value], left, top, panel_w, panel_h)[0][1]
            dash = ' stroke-dasharray="5 4"' if value in {.01, .003} else ''
            out.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+panel_w}" y2="{y:.2f}" stroke="#e1e2e5"{dash}/>')
            out.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" class="tick">{label}</text>')
        for step in (1, 100, 200, 300, 400, 500):
            x = points([step], [1], left, top, panel_w, panel_h)[0][0]
            out.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+panel_h}" stroke="#eeeff1"/>')
            out.append(f'<text x="{x:.2f}" y="{top+panel_h+18}" text-anchor="middle" class="tick">{step}</text>')
        for optimizer in ORDER:
            selected = [r for r in runs if r["scenario"] == scenario
                        and r["optimizer"] == optimizer]
            histories = [r["history"] for r in selected]
            steps = [row["step"] for row in histories[0]]
            columns = [[history[index]["relative_mse"] for history in histories]
                       for index in range(len(steps))]
            median = [statistics.median(column) for column in columns]
            lower = [min(column) for column in columns]
            upper = [max(column) for column in columns]
            med_points = points(steps, median, left, top, panel_w, panel_h)
            band = points(steps, lower, left, top, panel_w, panel_h)
            band += list(reversed(points(steps, upper, left, top, panel_w, panel_h)))
            color = COLORS[optimizer]
            out.append(f'<polygon points="{polygon(band)}" fill="{color}" opacity=".09"/>')
            out.append(f'<path d="{path(med_points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        out.append(f'<text x="{left+panel_w/2}" y="{top+panel_h+39}" text-anchor="middle" class="sub">optimizer updates</text>')

        summary_y = top + panel_h + 72
        for index, optimizer in enumerate(ORDER):
            selected = [r for r in runs if r["scenario"] == scenario
                        and r["optimizer"] == optimizer]
            step = statistics.mean(r["steps_to_1e-2"] for r in selected)
            final = statistics.mean(r["final_relative_mse"] for r in selected)
            x = left + (index % 2) * 300
            y = summary_y + (index // 2) * 23
            out.append(f'<text x="{x}" y="{y}" class="result" fill="{COLORS[optimizer]}">{html.escape(LABELS[optimizer])}: 10⁻² @ {step:.0f} · final {final:.2e}</text>')

    out.append('<text x="700" y="664" text-anchor="middle" class="sub">Lines are seed medians; translucent bands are full seed ranges. The target operator is identical; only gradient exposure changes.</text>')
    out.append('</svg>')
    OUTPUT.write_text("\n".join(out))
    print(OUTPUT)


if __name__ == "__main__":
    main()
