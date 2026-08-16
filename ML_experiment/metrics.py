from __future__ import annotations

import math
import numpy as np
import torch


@torch.no_grad()
def predict(model, x, chunk=1024):
    model.eval(); return torch.cat([model(part) for part in x.split(chunk)])


def class_metrics(logits, labels):
    prediction = logits.argmax(1); probability = torch.softmax(logits, 1)
    recalls = []
    for label in range(logits.shape[1]):
        selected = labels == label
        recalls.append(float((prediction[selected] == label).float().mean()) if selected.any() else math.nan)
    one_hot = torch.nn.functional.one_hot(labels, logits.shape[1]).float()
    return {"score": float(np.nanmean(recalls)), "accuracy": float((prediction == labels).float().mean()),
            "min_class_recall": float(np.nanmin(recalls)), "probability_mse": float((probability-one_hot).square().mean())}


def regression_metrics(prediction, target):
    mse = float((prediction - target).square().mean()); variance = float(target.square().mean())
    return {"score": 1 / (1 + mse), "normalized_mse": mse,
            "r2": 1 - mse / max(variance, 1e-9), "rmse": math.sqrt(mse)}


def evaluate(model, task, x=None, y=None):
    x = task.x_test if x is None else x; y = task.y_test if y is None else y
    output = predict(model, x)
    return class_metrics(output, y) if task.kind == "classification" else regression_metrics(output, y)


def tail_metrics(model, task):
    if not task.tail_x: return {}
    rows = [evaluate(model, task, x, y) for x, y in zip(task.tail_x, task.tail_y)]
    if task.kind == "classification":
        survival = 0
        for row in rows:
            if row["min_class_recall"] < .8: break
            survival += 1
        return {"tail_score": float(np.mean([row["min_class_recall"] for row in rows])),
                "tail_survival": survival, "tail_bins": rows}
    return {"tail_score": float(np.mean([row["score"] for row in rows])), "tail_survival": None, "tail_bins": rows}


def jacobian_variability(model, x, maximum=24):
    model.eval(); selected = x[:maximum].detach(); jacobians = []
    for point in selected:
        point = point.clone().requires_grad_(True); output = model(point[None])[0]; rows = []
        for index in range(len(output)):
            rows.append(torch.autograd.grad(output[index], point, retain_graph=index + 1 < len(output))[0])
        jacobians.append(torch.stack(rows).detach())
    stack = torch.stack(jacobians); mean = stack.mean(0)
    variability = (stack - mean).square().sum((1, 2)).sqrt().mean() / mean.square().sum().sqrt().clamp_min(1e-8)
    centered = stack.flatten(1) - stack.flatten(1).mean(0)
    singular = torch.linalg.svdvals(centered)
    rank = 0 if float(variability) < 1e-6 else int((singular > singular.max().clamp_min(1e-12) * 1e-5).sum())
    return float(variability), rank


@torch.no_grad()
def soft_diagnostics(model, task):
    sample = task.x_val[:512]; model.set_diagnostic_mode("matched"); _ = model(sample); diagnostics = model.diagnostics()
    values = {}
    for layer, state in diagnostics.items():
        weight = state["weight"]
        values[f"{layer}_allocation_entropy"] = float(state["entropy"].mean())
        values[f"{layer}_allocation_variation"] = float(weight.std(0).mean())
        values[f"{layer}_metric_condition_median"] = float(state["condition"].median())
        values[f"{layer}_correction_ratio"] = float((state["correction_norm"]/(state["base_norm"]+1e-8)).mean())
    full = evaluate(model, task, task.x_val, task.y_val)["score"]
    values["matched_score"] = full
    for mode in ("base_only", "uniform", "mismatched"):
        model.set_diagnostic_mode(mode); values[f"{mode}_score"] = evaluate(model, task, task.x_val, task.y_val)["score"]
        values[f"{mode}_drop"] = full - values[f"{mode}_score"]
    model.set_diagnostic_mode("matched")
    return values
