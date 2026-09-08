"""Measure the backward channel introduced by self-context normalization.

The self-context forward function is held fixed.  At selected checkpoints we
differentiate the same scalar loss three ways:

``exact``
    Ordinary autograd through the normalized context proposal.
``detached``
    The same forward value with the context-mediated cotangent removed.
``nonexpansive``
    The same forward value with only the normalization Jacobian certified to
    have operator norm at most one.

Thus ``exact - detached`` is the parameter-gradient contribution caused by
the moving self-context chart.  No labels, validation values, or task identity
enter the decomposition rule.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ML_experiment.optimizers import make_optimizer
from ML_experiment.tasks import TASK_BUILDERS
from ML_experiment.variants import make_variant


def _loss(model, x, y, kind):
    output = model(x)
    if kind == "classification":
        return F.cross_entropy(output, y)
    return F.mse_loss(output, y)


def _gradient_pass(model, x, y, kind, mode):
    model.set_context_backward_mode(mode)
    model.zero_grad(set_to_none=True)
    loss = _loss(model, x, y, kind)
    loss.backward()
    gradients = {
        name: (
            torch.zeros_like(parameter)
            if parameter.grad is None else parameter.grad.detach().clone()
        )
        for name, parameter in model.named_parameters()
    }
    scales = {
        layer_name: {
            "mean": float(layer.last_context_backward_scale.mean()),
            "minimum": float(layer.last_context_backward_scale.min()),
            "maximum": float(layer.last_context_backward_scale.max()),
        }
        for layer_name, layer in (("up", model.up), ("down", model.down))
        if layer.last_context_backward_scale is not None
    }
    return float(loss.detach()), gradients, scales


def _inner(left, right, names):
    return sum(float(torch.sum(left[name] * right[name])) for name in names)


def _summarize(exact, detached, restrained, names):
    feedback = {name: exact[name] - detached[name] for name in names}
    exact_norm = math.sqrt(max(0.0, _inner(exact, exact, names)))
    direct_norm = math.sqrt(max(0.0, _inner(detached, detached, names)))
    feedback_norm = math.sqrt(max(0.0, _inner(feedback, feedback, names)))
    restrained_norm = math.sqrt(max(0.0, _inner(restrained, restrained, names)))
    denominator = max(direct_norm * feedback_norm, 1e-30)
    return {
        "exact_norm": exact_norm,
        "direct_norm": direct_norm,
        "context_feedback_norm": feedback_norm,
        "nonexpansive_norm": restrained_norm,
        "feedback_to_direct": feedback_norm / max(direct_norm, 1e-30),
        "feedback_to_exact": feedback_norm / max(exact_norm, 1e-30),
        "direct_feedback_cosine": _inner(detached, feedback, names) / denominator,
        "nonexpansive_to_exact": restrained_norm / max(exact_norm, 1e-30),
    }


def decompose(model, x, y, kind):
    exact_loss, exact, _ = _gradient_pass(model, x, y, kind, "exact")
    detached_loss, detached, _ = _gradient_pass(model, x, y, kind, "detached")
    restrained_loss, restrained, scales = _gradient_pass(
        model, x, y, kind, "nonexpansive"
    )
    model.set_context_backward_mode("exact")
    model.zero_grad(set_to_none=True)
    if not (
        exact_loss == detached_loss
        and exact_loss == restrained_loss
    ):
        raise RuntimeError("backward modes changed the forward loss")

    all_names = list(exact)
    groups = {
        "all": all_names,
        "embed": [name for name in all_names if name.startswith("embed.")],
        "up": [name for name in all_names if name.startswith("up.")],
        "down": [name for name in all_names if name.startswith("down.")],
        "output": [name for name in all_names if name.startswith("output.")],
    }
    return {
        "loss": exact_loss,
        "context_backward_scale": scales,
        "gradient": {
            group: _summarize(exact, detached, restrained, names)
            for group, names in groups.items() if names
        },
    }


@torch.no_grad()
def probe(model, x, y, kind):
    model.set_context_backward_mode("exact")
    output = model(x)
    value = (
        F.cross_entropy(output, y)
        if kind == "classification" else F.mse_loss(output, y)
    )
    result = {"loss": float(value)}
    if kind == "classification":
        result["accuracy"] = float((output.argmax(1) == y).float().mean())
    return result


def run(
    task_name, optimizer_name, seed, width, steps, batch, lr, checkpoints,
    training_backward_mode="exact",
):
    torch.manual_seed(10000 + seed)
    task = TASK_BUILDERS[task_name](seed)
    model = make_variant("self_context", task.input_dim, task.output_dim, width)
    model.set_context_backward_mode(training_backward_mode)
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(190000 + seed)
    probe_count = min(512, len(task.x_train))
    probe_x, probe_y = task.x_train[:probe_count], task.y_train[:probe_count]
    history, decomposition = [], []

    if 0 in checkpoints:
        decomposition.append({
            "step": 0,
            "probe": probe(model, probe_x, probe_y, task.kind),
            **decompose(model, probe_x, probe_y, task.kind),
        })
    for step in range(1, steps + 1):
        index = torch.randint(
            len(task.x_train), (batch,), generator=generator
        )
        x, y = task.x_train[index], task.y_train[index]
        model.set_context_backward_mode(training_backward_mode)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, x, y, task.kind)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        if step == 1 or step % 5 == 0 or step == steps:
            history.append({
                "step": step,
                "batch_loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **probe(model, probe_x, probe_y, task.kind),
                **({"update_gradient_ratio": float(optimizer.last_update_gradient_ratio)}
                   if hasattr(optimizer, "last_update_gradient_ratio") else {}),
                **({"certificate_scale": float(optimizer.last_certificate_scale)}
                   if hasattr(optimizer, "last_certificate_scale") else {}),
            })
        if step in checkpoints:
            decomposition.append({
                "step": step,
                "probe": probe(model, probe_x, probe_y, task.kind),
                **decompose(model, probe_x, probe_y, task.kind),
            })

    return {
        "task": task_name,
        "optimizer": optimizer_name,
        "optimizer_lr": lr,
        "training_backward_mode": training_backward_mode,
        "seed": seed,
        "width": width,
        "steps": steps,
        "history": history,
        "decomposition": decomposition,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/context_gradient_decomposition.json"))
    parser.add_argument("--tasks", default="nd_spiral_high_rank,ripple")
    parser.add_argument("--optimizers", default="adamw,lepton_transport,lepton_transport_guarded")
    parser.add_argument("--optimizer-lrs", default="adamw=.003,lepton_transport=.003,lepton_transport_guarded=.003")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--checkpoints", default="0,25,50,100,200,300,400,500")
    parser.add_argument("--training-backward-modes", default="exact")
    args = parser.parse_args()
    rates = dict(
        (name, float(value))
        for name, value in (
            assignment.split("=", 1)
            for assignment in args.optimizer_lrs.split(",")
        )
    )
    checkpoints = {int(value) for value in args.checkpoints.split(",")}
    runs = []
    for task_name in args.tasks.split(","):
        for optimizer_name in args.optimizers.split(","):
            for training_backward_mode in args.training_backward_modes.split(","):
                for seed in range(args.seeds):
                    row = run(
                        task_name, optimizer_name, seed, args.width, args.steps,
                        args.batch, rates[optimizer_name], checkpoints,
                        training_backward_mode,
                    )
                    runs.append(row)
                    final = row["history"][-1]
                    print(json.dumps({
                        "task": task_name,
                        "optimizer": optimizer_name,
                        "training_backward_mode": training_backward_mode,
                        "seed": seed,
                        "final": final,
                    }), flush=True)
    payload = {
        "configuration": {**vars(args), "out": str(args.out)},
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"complete": True, "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
