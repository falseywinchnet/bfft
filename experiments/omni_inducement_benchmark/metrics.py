from __future__ import annotations

from collections import deque
import math

import numpy as np
import torch


@torch.no_grad()
def predict(model, x, context_x, context_y, chunk: int = 512):
    model.eval(); output = []
    for part in x.split(chunk):
        output.append(model(part, context_x, context_y))
    return torch.cat(output)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor):
    probability = torch.softmax(logits, 1)[:, 1]
    prediction = logits.argmax(1)
    recalls = []
    for label in (0, 1):
        selected = labels == label
        recalls.append(float((prediction[selected] == label).float().mean()) if selected.any() else math.nan)
    return {
        "accuracy": float((prediction == labels).float().mean()),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "min_class_recall": float(np.nanmin(recalls)),
        "class0_recall": recalls[0],
        "class1_recall": recalls[1],
        "probability_mse": float((probability - labels.float()).square().mean()),
    }


def tail_metrics(model, task, threshold: float = .8):
    bins = []
    for x, y in zip(task.tail_x, task.tail_y):
        bins.append(classification_metrics(predict(model, x, task.x_train, task.y_train), y))
    survival = 0
    for row in bins:
        if row["min_class_recall"] < threshold: break
        survival += 1
    return {
        "tail_accuracy": float(np.mean([row["accuracy"] for row in bins])),
        "tail_min_class_recall": float(np.mean([row["min_class_recall"] for row in bins])),
        "frontier_min_class_recall": bins[0]["min_class_recall"],
        "survival_bins_at_80pct": survival,
        "retention_auc": float(np.mean([row["min_class_recall"] for row in bins])),
        "tail_bins": bins,
    }


def _boundary(mask: np.ndarray):
    result = np.zeros_like(mask, dtype=bool)
    result[:-1] |= mask[:-1] != mask[1:]
    result[1:] |= mask[:-1] != mask[1:]
    result[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    result[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return result


def _dilate(mask: np.ndarray):
    result = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            result |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return result


def _components(mask: np.ndarray):
    seen = np.zeros_like(mask, dtype=bool); total = 0; height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if seen[y, x] or not mask[y, x]: continue
            total += 1; seen[y, x] = True; queue = deque([(y, x)])
            while queue:
                yy, xx = queue.popleft()
                for ny, nx in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; queue.append((ny, nx))
    return total


@torch.no_grad()
def shape_metrics(model, task, resolution: int = 120):
    if task.visual_limits is None: return {}
    xmin, xmax, ymin, ymax = task.visual_limits
    xs = torch.linspace(xmin, xmax, resolution); ys = torch.linspace(ymin, ymax, resolution)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij"); grid = torch.stack((xx.flatten(), yy.flatten()), 1)
    truth = task.truth(grid).reshape(resolution, resolution).numpy().astype(bool)
    probability = torch.softmax(predict(model, grid, task.x_train, task.y_train), 1)[:, 1]
    predicted = (probability.reshape(resolution, resolution).numpy() >= .5)
    true_boundary, pred_boundary = _boundary(truth), _boundary(predicted)
    precision = (pred_boundary & _dilate(true_boundary)).sum() / max(1, pred_boundary.sum())
    recall = (true_boundary & _dilate(pred_boundary)).sum() / max(1, true_boundary.sum())
    boundary_f1 = 2 * precision * recall / max(1e-9, precision + recall)
    true_components = _components(truth) + _components(~truth)
    pred_components = _components(predicted) + _components(~predicted)
    return {
        "grid_accuracy": float((predicted == truth).mean()),
        "grid_probability_mse": float(np.mean((probability.numpy() - truth.reshape(-1)) ** 2)),
        "boundary_f1": float(boundary_f1),
        "true_components": int(true_components),
        "predicted_components": int(pred_components),
        "component_count_error": int(abs(pred_components - true_components)),
    }
