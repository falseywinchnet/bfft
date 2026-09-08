#!/usr/bin/env python3
"""Dependency-free static figure for Matrix Transport on Ripple."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import ML_experiment.plot_adamw_self_context_ripple as plot


ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "results_matrix_transport_ripple_v4" / "results.json"
CONTROL = ROOT / "results_adamw_ripple_equal_ceiling" / "results.json"
OUTPUT = ROOT / "matrix_transport_ripple.svg"


def main():
    primary = json.loads(PRIMARY.read_text())
    control = json.loads(CONTROL.read_text())
    runs = []
    for row in primary["runs"]:
        runs.append({**row, "arm": row["optimizer"]})
    for row in control["runs"]:
        runs.append({**row, "arm": "adamw_equal_ceiling"})
    data = {"runs": runs}
    plot.COLORS = {
        "adamw": "#1667a8",
        "matrix_transport": "#c45620",
        "adamw_equal_ceiling": "#7b7f83",
    }

    fits = {
        row["arm"]: row["fit_visualization"]
        for row in runs if row["seed"] == 0
    }
    target = fits["adamw"]["target"]
    ordinary = fits["adamw"]["prediction"]
    transported = fits["matrix_transport"]["prediction"]
    low = min(target + ordinary + transported)
    high = max(target + ordinary + transported)
    panels = [
        plot.plot_panel(data, 28, 120, 708, 410, "loss",
                        "A · observed minibatch loss", "MSE · log", True),
        plot.plot_panel(data, 764, 120, 708, 410, "normalized_mse",
                        "B · validation loss", "normalized MSE · log", True),
        plot.plot_panel(data, 28, 550, 708, 410, "score",
                        "C · validation score", "1 / (1 + MSE)", False),
        plot.surface_panel(764, 550, 708, 410, target,
                           "D · Ripple target", low, high),
        plot.surface_panel(28, 980, 708, 470, ordinary,
                           "E · AdamW · seed 0 best", low, high),
        plot.surface_panel(764, 980, 708, 470, transported,
                           "F · Matrix Transport · seed 0 best", low, high),
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1500" viewBox="0 0 1500 1500">
<style>.title{{font:700 28px system-ui;fill:#17202a}}.subtitle{{font:14px system-ui;fill:#5e6b76}}.panel-title{{font:700 15px system-ui;fill:#17202a}}.tick{{font:10px system-ui;fill:#69747c}}.axis{{font:11px system-ui;fill:#69747c}}</style>
<rect width="100%" height="100%" fill="#f4f0e8"/>
<text x="28" y="46" class="title">Ripple: Euclidean Matrix Transport beats AdamW</text>
<text x="28" y="76" class="subtitle">Five paired seeds · width 24 · 750 steps · batch 256 · declared lr 0.003 · faint = seed · heavy = mean</text>
<line x1="820" y1="70" x2="850" y2="70" stroke="#1667a8" stroke-width="4"/><text x="860" y="75" class="subtitle">AdamW .003</text>
<line x1="1000" y1="70" x2="1030" y2="70" stroke="#c45620" stroke-width="4"/><text x="1040" y="75" class="subtitle">Matrix Transport</text>
<line x1="1230" y1="70" x2="1260" y2="70" stroke="#7b7f83" stroke-width="4"/><text x="1270" y="75" class="subtitle">AdamW .012</text>
{''.join(panels)}</svg>'''
    OUTPUT.write_text(svg)
    print(OUTPUT)


if __name__ == "__main__":
    main()
