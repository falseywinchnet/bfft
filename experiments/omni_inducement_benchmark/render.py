from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import torch

from metrics import predict


COLORS = {
    "linear": "#71717a", "dense_lelu": "#111827", "static_fourier_circle": "#a16207",
    "living_fourier_circle": "#d97706", "living_metric_graph": "#7c3aed",
    "learned_subspace_gram": "#2563eb", "hypersphere_atlas": "#0891b2",
    "soft_eikonal_pool": "#0d9488", "jet_transport": "#16a34a",
    "matrix_exponential": "#dc2626", "associative_shells": "#db2777",
    "banach_eikonal_sieve": "#9333ea", "projective_roles": "#4f46e5",
}


def _svg_begin(width, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#faf9f6"/>',
            '<style>text{font-family:ui-sans-serif,system-ui;fill:#18181b}.small{font-size:10px}.title{font-size:15px;font-weight:650}</style>']


def _escape(value): return html.escape(str(value))


@torch.no_grad()
def decision_atlas(path: Path, task, models: dict[str, torch.nn.Module], resolution: int = 100):
    xmin, xmax, ymin, ymax = task.visual_limits; columns = 4; panel = 260
    rows = math.ceil(len(models) / columns); width, height = columns * panel, rows * panel
    svg = _svg_begin(width, height)
    xs = torch.linspace(xmin, xmax, resolution); ys = torch.linspace(ymin, ymax, resolution)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij"); grid = torch.stack((xx.flatten(), yy.flatten()), 1)
    for index, (name, model) in enumerate(models.items()):
        ox, oy = (index % columns) * panel, (index // columns) * panel
        probability = torch.softmax(predict(model, grid, task.x_train, task.y_train), 1)[:, 1].reshape(resolution, resolution).numpy()
        cell = (panel - 32) / resolution
        for row in range(resolution):
            for column in range(resolution):
                p = probability[row, column]; red = int(42 + 198 * p); blue = int(220 - 178 * p)
                # row zero is Cartesian ymin and therefore belongs at the panel bottom.
                ypixel = oy + 18 + (resolution - 1 - row) * cell
                svg.append(f'<rect x="{ox + 16 + column * cell:.2f}" y="{ypixel:.2f}" width="{cell + .2:.2f}" height="{cell + .2:.2f}" fill="rgb({red},92,{blue})"/>')
        for point, label in zip(task.x_train[::max(1, len(task.x_train)//450)], task.y_train[::max(1, len(task.y_train)//450)]):
            px = ox + 16 + float((point[0] - xmin) / (xmax - xmin)) * (panel - 32)
            py = oy + 18 + (1 - float((point[1] - ymin) / (ymax - ymin))) * (panel - 32)
            color = "#082f49" if int(label) == 0 else "#7f1d1d"
            svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.25" fill="{color}"/>')
        svg.append(f'<rect x="{ox+16}" y="{oy+18}" width="{panel-32}" height="{panel-32}" fill="none" stroke="#27272a"/>')
        svg.append(f'<text class="small" x="{ox+18}" y="{oy+14}">{_escape(name)}</text>')
        svg.append(f'<text class="small" x="{ox+18}" y="{oy+panel-3}">x→ ; y↑ (not mirrored)</text>')
    svg.append('</svg>'); path.write_text("\n".join(svg))


def _project(x, y, z, ox, oy, scale=74, zscale=48):
    return ox + (x - y) * scale, oy + (x + y) * 25 - z * zscale


@torch.no_grad()
def surface_atlas(path: Path, task, models: dict[str, torch.nn.Module], resolution: int = 31):
    names = ["true_solution"] + list(models); columns = 3; panel_w, panel_h = 340, 245
    rows = math.ceil(len(names) / columns); svg = _svg_begin(columns * panel_w, rows * panel_h)
    xmin, xmax, ymin, ymax = task.visual_limits
    xs = torch.linspace(xmin, xmax, resolution); ys = torch.linspace(ymin, ymax, resolution)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij"); grid = torch.stack((xx.flatten(), yy.flatten()), 1)
    truth = 2 * task.truth(grid).float().reshape(resolution, resolution) - 1
    surfaces = {"true_solution": truth.numpy()}
    for name, model in models.items():
        probability = torch.softmax(predict(model, grid, task.x_train, task.y_train), 1)[:, 1]
        surfaces[name] = (2 * probability - 1).reshape(resolution, resolution).numpy()
    for index, name in enumerate(names):
        cx, top = (index % columns) * panel_w, (index // columns) * panel_h
        origin_x, origin_y = cx + panel_w / 2, top + 133
        surface = surfaces[name]
        for row in range(0, resolution, 2):
            points = []
            for column in range(resolution):
                xn = -1 + 2 * column / (resolution - 1); yn = -1 + 2 * row / (resolution - 1)
                points.append(_project(xn, yn, surface[row, column], origin_x, origin_y))
            error = float(np.mean(np.abs(surface[row] - truth.numpy()[row]))) if name != "true_solution" else 0
            color = f'rgb({int(35+190*min(1,error))},{int(125-75*min(1,error))},{int(190-120*min(1,error))})'
            svg.append('<polyline points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in points) + f'" fill="none" stroke="{color}" stroke-width="1.2"/>')
        for column in range(0, resolution, 3):
            points = []
            for row in range(resolution):
                xn = -1 + 2 * column / (resolution - 1); yn = -1 + 2 * row / (resolution - 1)
                points.append(_project(xn, yn, surface[row, column], origin_x, origin_y))
            svg.append('<polyline points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in points) + '" fill="none" stroke="#52525b" stroke-opacity=".45" stroke-width=".65"/>')
        svg.append(f'<text class="title" x="{cx+18}" y="{top+22}">{_escape(name)}</text>')
        if name != "true_solution":
            mse = float(np.mean(((surface + 1) / 2 - (truth.numpy() + 1) / 2) ** 2))
            svg.append(f'<text class="small" x="{cx+18}" y="{top+40}">dense-grid MSE {mse:.4f}; red = shape error</text>')
        svg.append(f'<text class="small" x="{cx+18}" y="{top+panel_h-8}">front: y min · right: x max · vertical: class score</text>')
    svg.append('</svg>'); path.write_text("\n".join(svg))


def learning_curves(path: Path, histories: dict[str, list[dict]], metric: str = "balanced_accuracy"):
    width, height = 1100, 650; left, top, right, bottom = 78, 55, 250, 75
    svg = _svg_begin(width, height); x0, x1 = left, width - right; y0, y1 = top, height - bottom
    svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#fff" stroke="#a1a1aa"/>')
    max_step = max(row["step"] for history in histories.values() for row in history)
    for tick in range(6):
        value = tick / 5; y = y1 - value * (y1 - y0)
        svg.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#e4e4e7"/><text class="small" x="{x0-34}" y="{y+4}">{value:.1f}</text>')
    for index, (name, history) in enumerate(histories.items()):
        points = []
        for row in history:
            x = x0 + row["step"] / max_step * (x1 - x0); y = y1 - row[metric] * (y1 - y0); points.append((x, y))
        color = COLORS.get(name, "#111827")
        svg.append('<polyline points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in points) + f'" fill="none" stroke="{color}" stroke-width="2.3"/>')
        ly = top + 18 + index * 21
        svg.append(f'<line x1="{x1+18}" y1="{ly-4}" x2="{x1+42}" y2="{ly-4}" stroke="{color}" stroke-width="3"/><text class="small" x="{x1+49}" y="{ly}">{_escape(name)}</text>')
    svg.append(f'<text class="title" x="{left}" y="{top-22}">Validation learning speed</text><text class="small" x="{(x0+x1)/2}" y="{height-25}">optimizer steps</text>')
    svg.append('</svg>'); path.write_text("\n".join(svg))
