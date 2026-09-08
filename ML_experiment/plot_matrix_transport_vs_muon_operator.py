#!/usr/bin/env python3
"""Render the practical Muon/operator comparison as dependency-free SVG."""
from __future__ import annotations

import html
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "results_matrix_transport_vs_muon_operator.json"
FUSION = ROOT / "results_longitudinal_fusion_muon_operator.json"
OUTPUT = ROOT / "matrix_transport_vs_muon_operator.svg"

ORDER = (
    "adamw", "muon", "matrix_transport", "matrix_transport_nesterov",
    "matrix_transport_longitudinal_fusion",
)
LABELS = {
    "adamw": "AdamW", "muon": "Muon (5-step NS)",
    "matrix_transport": "Matrix Transport",
    "matrix_transport_nesterov": "Nesterov Matrix Transport",
    "matrix_transport_longitudinal_fusion": "Longitudinal fusion",
}
COLORS = {
    "adamw": "#6f63d9", "muon": "#d26b38",
    "matrix_transport": "#078d7c",
    "matrix_transport_nesterov": "#1261a0",
    "matrix_transport_longitudinal_fusion": "#b4216b",
}


def _curves(runs, protocol, scenario, optimizer):
    selected = [r for r in runs if r["protocol"] == protocol
                and r["scenario"] == scenario and r["optimizer"] == optimizer]
    histories = [r["history"] for r in selected]
    steps = [point["step"] for point in histories[0]]
    columns = [[history[i]["relative_loss"] for history in histories]
               for i in range(len(steps))]
    return (steps, [statistics.median(v) for v in columns],
            [min(v) for v in columns], [max(v) for v in columns])


def _points(steps, values, left, top, width, height):
    result = []
    for step, value in zip(steps, values):
        x = left + (step - 1) / 499 * width
        log_value = min(math.log10(8), max(-8, math.log10(max(value, 1e-30))))
        y = top + (math.log10(8) - log_value) / (math.log10(8) + 8) * height
        result.append((x, y))
    return result


def _path(points):
    return " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}"
                    for i, (x, y) in enumerate(points))


def _polygon(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def main():
    primary = json.loads(PRIMARY.read_text())
    fusion = json.loads(FUSION.read_text())
    runs = primary["runs"] + fusion["runs"]
    exact = statistics.median(
        row["exact_muon"]["relative_loss_after_one_step"]
        for row in primary["exact_one_step_certificates"]
    )
    width, height = 1420, 920
    panel_w, panel_h = 620, 315
    positions = ((80, 170), (750, 170), (80, 535), (750, 535))
    panels = (
        ("standing", "population", "Population · common LR 0.003"),
        ("standing", "block_sweep", "Rank-8 block stream · common LR 0.003"),
        ("operator_native", "population", "Population · native operator scale"),
        ("operator_native", "block_sweep", "Rank-8 block stream · native operator scale"),
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#20242a}'
        '.title{font-size:24px;font-weight:700}.subtitle{font-size:13px;fill:#5e6470}'
        '.panel-title{font-size:17px;font-weight:700}.tick{font-size:11px;fill:#666d78}'
        '.axis{font-size:12px;fill:#4f5661}.legend{font-size:12px;font-weight:600}</style>',
        '<text x="710" y="38" text-anchor="middle" class="title">Muon’s witness: exact polar algebra versus practical optimizers</text>',
        '<text x="710" y="62" text-anchor="middle" class="subtitle">Three seeds · condition 10⁴ · 32×32 operator · every run continues for 500 updates</text>',
    ]
    legend_x = 80
    for optimizer in ORDER:
        out.append(f'<line x1="{legend_x}" y1="105" x2="{legend_x+28}" y2="105" stroke="{COLORS[optimizer]}" stroke-width="4"/>')
        out.append(f'<text x="{legend_x+36}" y="109" class="legend">{html.escape(LABELS[optimizer])}</text>')
        legend_x += 150 if optimizer == "adamw" else 250

    y_ticks = ((1, "10⁰"), (1e-2, "10⁻²"), (1e-4, "10⁻⁴"),
               (1e-6, "10⁻⁶"), (1e-8, "10⁻⁸"))
    for (left, top), (protocol, scenario, title) in zip(positions, panels):
        out.append(f'<text x="{left}" y="{top-22}" class="panel-title">{html.escape(title)}</text>')
        out.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#fff" stroke="#d7d8dc"/>')
        for value, label in y_ticks:
            y = _points([1], [value], left, top, panel_w, panel_h)[0][1]
            dash = ' stroke-dasharray="5 4"' if value == 1e-2 else (' stroke-dasharray="2 4"' if value == 1e-4 else '')
            out.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+panel_w}" y2="{y:.2f}" stroke="#dfe1e5"{dash}/>')
            out.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" class="tick">{label}</text>')
        for step in (1, 100, 200, 300, 400, 500):
            x = _points([step], [1], left, top, panel_w, panel_h)[0][0]
            out.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+panel_h}" stroke="#ececef"/>')
            out.append(f'<text x="{x:.2f}" y="{top+panel_h+18}" text-anchor="middle" class="tick">{step}</text>')
        for optimizer in ORDER:
            steps, median, low, high = _curves(runs, protocol, scenario, optimizer)
            med = _points(steps, median, left, top, panel_w, panel_h)
            lower = _points(steps, low, left, top, panel_w, panel_h)
            upper = _points(steps, high, left, top, panel_w, panel_h)
            color = COLORS[optimizer]
            out.append(f'<polygon points="{_polygon(lower + list(reversed(upper)))}" fill="{color}" opacity="0.08"/>')
            out.append(f'<path d="{_path(med)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        out.append(f'<text x="{left+panel_w/2}" y="{top+panel_h+38}" text-anchor="middle" class="axis">optimizer updates</text>')
        out.append(f'<text transform="translate({left-55},{top+panel_h/2}) rotate(-90)" text-anchor="middle" class="axis">relative population loss</text>')
        if protocol == "operator_native" and scenario == "population":
            out.append(f'<circle cx="{left}" cy="{top+panel_h}" r="5" fill="#171a1f"/>')
            out.append(f'<text x="{left+12}" y="{top+panel_h-10}" class="tick">exact polar oracle: {exact:.1e} at step 1</text>')
            out.append(f'<text x="{left+panel_w-8}" y="{top+18}" text-anchor="end" class="tick">AdamW .01 · Muon 1 · Matrix 1/√32</text>')

    out.append('<text x="710" y="905" text-anchor="middle" class="subtitle">Lines: seed medians · bands: seed ranges · dashed threshold: 10⁻² · dotted threshold: 10⁻⁴</text>')
    out.append('</svg>')
    OUTPUT.write_text("\n".join(out))
    print(OUTPUT)


if __name__ == "__main__":
    main()
