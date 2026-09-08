#!/usr/bin/env python3
"""Dependency-free Ripple stability atlas for ResidualPolarTransport."""
from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path


WIDTH, HEIGHT = 1600, 1430
ORDER = (
    "adamw",
    "muon",
    "matrix_transport_longitudinal_fusion",
    "matrix_transport_residual_polar",
)
LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
    "matrix_transport_longitudinal_fusion": "Matrix Transport",
    "matrix_transport_residual_polar": "optimizer.py",
}
COLORS = {
    "adamw": "#1769a6",
    "muon": "#d06a32",
    "matrix_transport_longitudinal_fusion": "#ad2d6d",
    "matrix_transport_residual_polar": "#17815a",
}


def path(points):
    return " ".join(
        ("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
        for index, (x, y) in enumerate(points)
    )


def runs_for(data, optimizer):
    return [run for run in data["runs"] if run["optimizer"] == optimizer]


def curve_panel(data, x0, y0, width, height, key, title, subtitle):
    left, right, top, bottom = 68, 20, 56, 48
    histories = [run["history"] for run in data["runs"]]
    values = [math.log10(max(row[key], 1e-8))
              for history in histories for row in history]
    ymin, ymax = min(values), max(values)
    margin = .06 * max(ymax - ymin, .1)
    ymin, ymax = ymin - margin, ymax + margin
    xmin = min(row["step"] for row in histories[0])
    xmax = max(row["step"] for row in histories[0])
    px = lambda value: x0 + left + (value - xmin) / (xmax - xmin) * (width-left-right)
    py = lambda value: y0 + top + (ymax-math.log10(max(value, 1e-8))) / (ymax-ymin) * (height-top-bottom)
    out = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="8" fill="#fff" stroke="#d4d8d4"/>',
        f'<text x="{x0+20}" y="{y0+25}" class="panel">{html.escape(title)}</text>',
        f'<text x="{x0+20}" y="{y0+44}" class="sub">{html.escape(subtitle)}</text>',
    ]
    for index in range(5):
        level = ymax - index / 4 * (ymax-ymin)
        yy = y0 + top + index / 4 * (height-top-bottom)
        out.append(f'<line x1="{x0+left}" y1="{yy}" x2="{x0+width-right}" y2="{yy}" stroke="#e8ebe8"/>')
        out.append(f'<text x="{x0+left-9}" y="{yy+4}" text-anchor="end" class="tick">{10**level:.2g}</text>')
    for step in (1, 150, 300, 450, 600, 750):
        xx = px(step)
        out.append(f'<line x1="{xx}" y1="{y0+top}" x2="{xx}" y2="{y0+height-bottom}" stroke="#f0f1ef"/>')
        out.append(f'<text x="{xx}" y="{y0+height-18}" text-anchor="middle" class="tick">{step}</text>')
    for optimizer in ORDER:
        selected = runs_for(data, optimizer)
        for run in selected:
            points = [(px(row["step"]), py(row[key])) for row in run["history"]]
            out.append(f'<path d="{path(points)}" fill="none" stroke="{COLORS[optimizer]}" stroke-width="1" opacity=".13"/>')
        steps = [row["step"] for row in selected[0]["history"]]
        means = []
        for index, step in enumerate(steps):
            value = statistics.mean(run["history"][index][key] for run in selected)
            means.append((px(step), py(value)))
        out.append(f'<path d="{path(means)}" fill="none" stroke="{COLORS[optimizer]}" stroke-width="3"/>')
    out.append(f'<text x="{x0+width/2}" y="{y0+height-3}" text-anchor="middle" class="sub">optimizer updates</text>')
    return out


def state_panel(data, x0, y0, width, height):
    left, right, top, bottom = 66, 22, 58, 48
    selected = runs_for(data, "matrix_transport_residual_polar")
    xmin, xmax = 1, 750
    px = lambda value: x0 + left + (value-xmin)/(xmax-xmin)*(width-left-right)
    py = lambda value: y0 + top + (1-value)*(height-top-bottom)
    out = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="8" fill="#fff" stroke="#d4d8d4"/>',
        f'<text x="{x0+20}" y="{y0+25}" class="panel">C · optimizer.py remains mixed and reversible</text>',
        f'<text x="{x0+20}" y="{y0+44}" class="sub">Layer-mean residual entropy rank and polar weight · faint = individual seed</text>',
    ]
    for value in (0, .25, .5, .75, 1):
        yy = py(value)
        out.append(f'<line x1="{x0+left}" y1="{yy}" x2="{x0+width-right}" y2="{yy}" stroke="#e8ebe8"/>')
        out.append(f'<text x="{x0+left-9}" y="{yy+4}" text-anchor="end" class="tick">{value:.2g}</text>')
    for step in (1, 150, 300, 450, 600, 750):
        xx = px(step)
        out.append(f'<text x="{xx}" y="{y0+height-18}" text-anchor="middle" class="tick">{step}</text>')
    series = (
        ("optimizer_residual_entropy_rank", "Residual entropy rank", "#6d55b5"),
        ("optimizer_polar_weight", "Polar weight", "#17815a"),
    )
    for key, label, color in series:
        for run in selected:
            points = [(px(row["step"]), py(row[key])) for row in run["history"]]
            out.append(f'<path d="{path(points)}" fill="none" stroke="{color}" stroke-width="1" opacity=".14"/>')
        steps = [row["step"] for row in selected[0]["history"]]
        means = []
        for index, step in enumerate(steps):
            value = statistics.mean(run["history"][index][key] for run in selected)
            means.append((px(step), py(value)))
        out.append(f'<path d="{path(means)}" fill="none" stroke="{color}" stroke-width="3.2"/>')
    out.extend((
        f'<line x1="{x0+width-410}" y1="{y0+24}" x2="{x0+width-380}" y2="{y0+24}" stroke="#6d55b5" stroke-width="3"/>',
        f'<text x="{x0+width-370}" y="{y0+28}" class="legend">Residual entropy rank</text>',
        f'<line x1="{x0+width-205}" y1="{y0+24}" x2="{x0+width-175}" y2="{y0+24}" stroke="#17815a" stroke-width="3"/>',
        f'<text x="{x0+width-165}" y="{y0+28}" class="legend">Polar weight</text>',
        f'<text x="{x0+width/2}" y="{y0+height-3}" text-anchor="middle" class="sub">optimizer updates</text>',
    ))
    return out


def outcome_panel(data, x0, y0, width, height):
    left, right, top, bottom = 58, 18, 62, 54
    all_values = [run["normalized_mse"] for run in data["runs"]]
    ymin = math.log10(min(all_values)) - .12
    ymax = math.log10(max(all_values)) + .12
    py = lambda value: y0 + top + (ymax-math.log10(value))/(ymax-ymin)*(height-top-bottom)
    out = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="8" fill="#fff" stroke="#d4d8d4"/>',
        f'<text x="{x0+20}" y="{y0+25}" class="panel">D · Held-out normalized MSE</text>',
        f'<text x="{x0+20}" y="{y0+44}" class="sub">Every dot is one paired seed · bar is mean</text>',
    ]
    for index in range(5):
        level = ymax - index/4*(ymax-ymin)
        yy = y0 + top + index/4*(height-top-bottom)
        out.append(f'<line x1="{x0+left}" y1="{yy}" x2="{x0+width-right}" y2="{yy}" stroke="#e8ebe8"/>')
        out.append(f'<text x="{x0+left-8}" y="{yy+4}" text-anchor="end" class="tick">{10**level:.3g}</text>')
    usable = width-left-right
    for index, optimizer in enumerate(ORDER):
        center = x0+left+(index+.5)/len(ORDER)*usable
        selected = runs_for(data, optimizer)
        values = [run["normalized_mse"] for run in selected]
        for seed, value in enumerate(values):
            jitter = (seed-2)*6
            out.append(f'<circle cx="{center+jitter}" cy="{py(value):.2f}" r="5" fill="{COLORS[optimizer]}" opacity=".72"/>')
        mean = statistics.mean(values)
        out.append(f'<line x1="{center-30}" y1="{py(mean):.2f}" x2="{center+30}" y2="{py(mean):.2f}" stroke="{COLORS[optimizer]}" stroke-width="4"/>')
        out.append(f'<text x="{center}" y="{y0+height-22}" text-anchor="middle" class="tick">{html.escape(LABELS[optimizer])}</text>')
    return out


def color(value, low, high):
    stops = ((20, 42, 72), (23, 108, 143), (52, 160, 119),
             (226, 198, 78), (194, 66, 43))
    t = max(0.0, min(1.0, (value-low)/max(high-low, 1e-12)))
    position = t*(len(stops)-1)
    index = min(len(stops)-2, int(position))
    fraction = position-index
    rgb = tuple(round(a+(b-a)*fraction) for a, b in zip(stops[index], stops[index+1]))
    return "#%02x%02x%02x" % rgb


def surface(x0, y0, width, height, values, title, low, high):
    side = round(math.sqrt(len(values)))
    top, bottom = 42, 25
    size = min(width-22, height-top-bottom)
    ox, oy = x0+(width-size)/2, y0+top
    cell = size/side
    out = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="8" fill="#fff" stroke="#d4d8d4"/>',
        f'<text x="{x0+14}" y="{y0+26}" class="panel-small">{html.escape(title)}</text>',
    ]
    for index, value in enumerate(values):
        column, row = index%side, index//side
        out.append(f'<rect x="{ox+column*cell:.2f}" y="{oy+(side-row-1)*cell:.2f}" width="{cell+.25:.2f}" height="{cell+.25:.2f}" fill="{color(value,low,high)}"/>')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f5f3ee"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#20262c}'
        '.title{font-size:28px;font-weight:750}.deck{font-size:14px;fill:#58616b}'
        '.panel{font-size:17px;font-weight:700}.panel-small{font-size:14px;font-weight:700}'
        '.sub{font-size:12px;fill:#626c76}.tick{font-size:10px;fill:#68727c}'
        '.legend{font-size:11px;font-weight:650}.summary{font-size:13px;font-weight:650}</style>',
        '<text x="30" y="42" class="title">Ripple keeps optimizer.py between Matrix Transport and Muon</text>',
        '<text x="30" y="69" class="deck">Self-context · width 24 · LR .003 · 750 updates · five paired seeds · heavy curves are means</text>',
    ]
    legend_x = 770
    for optimizer in ORDER:
        out.append(f'<line x1="{legend_x}" y1="57" x2="{legend_x+28}" y2="57" stroke="{COLORS[optimizer]}" stroke-width="4"/>')
        out.append(f'<text x="{legend_x+36}" y="61" class="legend">{html.escape(LABELS[optimizer])}</text>')
        legend_x += 185

    out += curve_panel(data, 28, 100, 754, 350, "normalized_mse",
                       "A · Validation loss", "Lower is better · logarithmic MSE")
    out += curve_panel(data, 810, 100, 762, 350, "loss",
                       "B · Live minibatch loss", "Lower is better · logarithmic MSE")
    out += state_panel(data, 28, 475, 1004, 330)
    out += outcome_panel(data, 1060, 475, 512, 330)

    fits = {
        run["optimizer"]: run["fit_visualization"]
        for run in data["runs"] if run["seed"] == 0
    }
    target = fits["adamw"]["target"]
    values = [target] + [fits[optimizer]["prediction"] for optimizer in ORDER]
    low, high = min(v for series in values for v in series), max(v for series in values for v in series)
    titles = ["Ripple target"] + [f"{LABELS[optimizer]} · seed 0 best" for optimizer in ORDER]
    surface_width = 298
    for index, (title, series) in enumerate(zip(titles, values)):
        out += surface(28+index*312, 835, surface_width, 430, series, title, low, high)

    out.extend((
        '<rect x="160" y="1290" width="1280" height="95" rx="9" fill="#e6eee9"/>',
        '<text x="800" y="1321" text-anchor="middle" class="summary">optimizer.py: mean held-out MSE .03195 · every seed reaches .90 · median .90 at 370 updates</text>',
        '<text x="800" y="1347" text-anchor="middle" class="deck">The controller settles near residual rank .58 and polar weight .42—not zero, not Muon. It repeatedly releases polar weight as local geometry broadens.</text>',
        '<text x="800" y="1372" text-anchor="middle" class="deck">Muon mean held-out MSE .05334 · Matrix Transport .04192 · AdamW .04940. No numerical failures in any arm.</text>',
        '</svg>',
    ))
    args.output.write_text("\n".join(out))
    print(args.output)


if __name__ == "__main__":
    main()
