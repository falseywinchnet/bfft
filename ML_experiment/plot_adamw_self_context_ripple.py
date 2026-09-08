#!/usr/bin/env python3
"""Render a dependency-free SVG for the AdamW Ripple comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COLORS = {"adamw": "#1667a8", "adamw_source_aware": "#c45620"}
WIDTH, HEIGHT = 1500, 1500


def histories(data, arm):
    return [row["history"] for row in data["runs"] if row["arm"] == arm]


def polyline(points, color, opacity, width):
    value = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (f'<polyline points="{value}" fill="none" stroke="{color}" '
            f'stroke-opacity="{opacity}" stroke-width="{width}"/>')


def plot_panel(data, x0, y0, width, height, key, title, ylabel, log=False):
    left, right, top, bottom = 58, 18, 42, 50
    runs = [run for arm in COLORS for run in histories(data, arm)]
    values = [point[key] for run in runs for point in run]
    transformed = [math.log10(max(value, 1e-10)) for value in values] if log else values
    ymin, ymax = min(transformed), max(transformed)
    ymin -= .04 * (ymax - ymin)
    ymax += .04 * (ymax - ymin)
    steps = [point["step"] for point in runs[0]]
    xmin, xmax = min(steps), max(steps)
    px = lambda value: x0 + left + (value - xmin) / (xmax - xmin) * (width - left - right)
    py = lambda value: y0 + top + (ymax - (math.log10(max(value, 1e-10)) if log else value)) / (ymax - ymin) * (height - top - bottom)
    parts = [f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="6" fill="#fffdf8" stroke="#d5cfc3"/>',
             f'<text x="{x0+18}" y="{y0+26}" class="panel-title">{title}</text>']
    for index in range(5):
        yy = y0 + top + index / 4 * (height - top - bottom)
        value = ymax - index / 4 * (ymax - ymin)
        shown = 10 ** value if log else value
        parts.extend((
            f'<line x1="{x0+left}" y1="{yy}" x2="{x0+width-right}" y2="{yy}" stroke="#ddd7cc"/>',
            f'<text x="{x0+left-8}" y="{yy+4}" class="tick" text-anchor="end">{shown:.2g}</text>',
        ))
    for index in range(6):
        step = xmin + index / 5 * (xmax - xmin)
        xx = px(step)
        parts.append(f'<text x="{xx}" y="{y0+height-17}" class="tick" text-anchor="middle">{step:.0f}</text>')
    parts.append(f'<text x="{x0+width/2}" y="{y0+height-3}" class="axis" text-anchor="middle">optimizer steps</text>')
    parts.append(f'<text x="{x0+13}" y="{y0+height/2}" class="axis" text-anchor="middle" transform="rotate(-90 {x0+13} {y0+height/2})">{ylabel}</text>')
    for arm in COLORS:
        arm_runs = histories(data, arm)
        for run in arm_runs:
            parts.append(polyline([(px(row["step"]), py(row[key])) for row in run], COLORS[arm], .15, .9))
        means = []
        for index, step in enumerate(steps):
            means.append((px(step), py(sum(run[index][key] for run in arm_runs) / len(arm_runs))))
        parts.append(polyline(means, COLORS[arm], 1, 3.1))
    return "".join(parts)


def interpolate_color(value, low, high):
    stops = ((18, 38, 68), (24, 107, 143), (49, 161, 123),
             (224, 198, 82), (197, 70, 42))
    t = max(0.0, min(1.0, (value - low) / max(high - low, 1e-12)))
    position = t * (len(stops) - 1)
    index = min(len(stops) - 2, int(position))
    fraction = position - index
    rgb = tuple(round(a + (b - a) * fraction)
                for a, b in zip(stops[index], stops[index + 1]))
    return "#%02x%02x%02x" % rgb


def surface_panel(x0, y0, width, height, values, title, low, high):
    side = round(math.sqrt(len(values)))
    top, bottom = 42, 34
    size = min(width - 35, height - top - bottom)
    ox, oy = x0 + (width - size) / 2, y0 + top
    cell = size / side
    parts = [f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="6" fill="#fffdf8" stroke="#d5cfc3"/>',
             f'<text x="{x0+18}" y="{y0+26}" class="panel-title">{title}</text>']
    for index, value in enumerate(values):
        column, row = index % side, index // side
        parts.append(f'<rect x="{ox+column*cell:.2f}" y="{oy+(side-row-1)*cell:.2f}" width="{cell+.2:.2f}" height="{cell+.2:.2f}" fill="{interpolate_color(value,low,high)}"/>')
    parts.append(f'<text x="{x0+width/2}" y="{y0+height-10}" class="axis" text-anchor="middle">x₁  ·  −π to π</text>')
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    fits = {row["arm"]: row["fit_visualization"]
            for row in data["runs"] if row["seed"] == 0}
    target = fits["adamw"]["target"]
    ordinary = fits["adamw"]["prediction"]
    source = fits["adamw_source_aware"]["prediction"]
    low, high = min(target + ordinary + source), max(target + ordinary + source)

    panels = [
        plot_panel(data, 28, 120, 708, 410, "loss", "A · observed minibatch loss", "MSE · log", True),
        plot_panel(data, 764, 120, 708, 410, "normalized_mse", "B · validation loss", "normalized MSE · log", True),
        plot_panel(data, 28, 550, 708, 410, "score", "C · validation score", "1 / (1 + MSE)", False),
        surface_panel(764, 550, 708, 410, target, "D · Ripple target", low, high),
        surface_panel(28, 980, 708, 470, ordinary, "E · AdamW · seed 0 best", low, high),
        surface_panel(764, 980, 708, 470, source, "F · source-aware · seed 0 best", low, high),
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<style>.title{{font:700 28px system-ui;fill:#17202a}}.subtitle{{font:14px system-ui;fill:#5e6b76}}.panel-title{{font:700 15px system-ui;fill:#17202a}}.tick{{font:10px system-ui;fill:#69747c}}.axis{{font:11px system-ui;fill:#69747c}}</style>
<rect width="100%" height="100%" fill="#f4f0e8"/>
<text x="28" y="46" class="title">Ripple: preserving self-context source identity makes AdamW slower</text>
<text x="28" y="76" class="subtitle">Five paired seeds · width 24 · 750 steps · batch 256 · lr 0.003 · faint = individual seed · heavy = mean</text>
<line x1="1040" y1="70" x2="1070" y2="70" stroke="#1667a8" stroke-width="4"/><text x="1080" y="75" class="subtitle">AdamW</text>
<line x1="1200" y1="70" x2="1230" y2="70" stroke="#c45620" stroke-width="4"/><text x="1240" y="75" class="subtitle">source-aware AdamW</text>
{''.join(panels)}</svg>'''
    args.output.write_text(svg)
    print(args.output)


if __name__ == "__main__":
    main()
