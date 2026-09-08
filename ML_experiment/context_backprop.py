"""Source-aware backward rules for self-context models.

The self-context forward value is identical in every backward mode. This lets
us split the parameter cotangent into a frozen-chart contribution and the
additional feedback caused by moving the chart itself. The split is
source-aware but task-blind: it never clips coordinates or inspects the loss
history or validation data.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch


@torch.no_grad()
def clip_context_gradient_channels_(parameters, max_norm, norm_type=2.0):
    """Clip ``.grad`` and its saved source channels by the same coefficient.

    ``torch.nn.utils.clip_grad_norm_`` only scales ``parameter.grad``. A
    source-aware optimizer reads the two saved cotangents instead, so clipping
    only their sum would silently bypass the declared global gradient bound.
    This helper preserves the exact identity

        grad == fixed_chart_grad + context_chart_grad

    after global clipping.
    """
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=max_norm, norm_type=norm_type
    )
    if not parameters or not torch.isfinite(total_norm):
        return total_norm
    coefficient = min(1.0, float(max_norm) / (float(total_norm) + 1e-6))
    if coefficient < 1.0:
        for parameter in parameters:
            fixed = getattr(parameter, "_fixed_chart_grad", None)
            chart = getattr(parameter, "_context_chart_grad", None)
            if fixed is not None:
                fixed.mul_(coefficient)
            if chart is not None:
                chart.mul_(coefficient)
    return total_norm


def _gradient_snapshot(model):
    return {
        parameter: (
            torch.zeros_like(parameter)
            if parameter.grad is None else parameter.grad.detach().clone()
        )
        for parameter in model.parameters()
        if parameter.requires_grad
    }


def _global_inner(left, right):
    return sum(
        torch.sum(left[parameter].double() * right[parameter].double())
        for parameter in left
    )


def backward_with_context_split(
    model,
    loss_closure: Callable[[], torch.Tensor],
    *,
    context_gain: float = 1.0,
    eps: float = 1e-30,
):
    """Backpropagate a globally certified self-context gradient.

    ``loss_closure`` must be deterministic across its two calls. The first
    pass holds every normalized self-context proposal fixed during backward;
    the second uses the locally nonexpansive normalization Jacobian. Their
    difference is the parameter-space chart-motion feedback.

    The feedback is uniformly scaled within each parameter tensor so that

        ||g_chart_applied[p]|| <= context_gain * ||g_fixed[p]||.

    At the default unit gain this implies
    ``<g_fixed + g_chart_applied, g_fixed> >= 0``. The chart can bend or
    cancel the frozen-chart decision but cannot reverse it to first order.
    This preserves the full non-elementwise shape inside every optimizer cell
    while preventing unrelated output-head gradients from concealing unstable
    chart feedback in a self-context tensor. A genuinely chart-only parameter
    (zero frozen-chart gradient) remains free to learn.
    """
    if context_gain < 0:
        raise ValueError("context_gain must be nonnegative")
    if not hasattr(model, "set_context_backward_mode"):
        raise TypeError("model does not expose self-context backward modes")

    model.set_context_backward_mode("detached")
    model.zero_grad(set_to_none=True)
    fixed_loss = loss_closure()
    fixed_loss.backward()
    fixed = _gradient_snapshot(model)

    model.set_context_backward_mode("nonexpansive")
    model.zero_grad(set_to_none=True)
    live_loss = loss_closure()
    live_loss.backward()
    live = _gradient_snapshot(model)

    if not torch.equal(fixed_loss.detach(), live_loss.detach()):
        raise RuntimeError("context backward modes changed the forward loss")

    chart = {
        parameter: live[parameter] - fixed[parameter]
        for parameter in fixed
    }
    fixed_norm = math.sqrt(max(0.0, float(_global_inner(fixed, fixed))))
    chart_norm = math.sqrt(max(0.0, float(_global_inner(chart, chart))))
    scales = {}

    for parameter in fixed:
        fixed_cell_norm = float(torch.linalg.vector_norm(fixed[parameter].double()))
        chart_cell_norm = float(torch.linalg.vector_norm(chart[parameter].double()))
        scale = (
            1.0
            if fixed_cell_norm <= eps
            else min(
                1.0,
                context_gain * fixed_cell_norm / max(chart_cell_norm, eps),
            )
        )
        scales[parameter] = scale
        applied_chart = chart[parameter].mul(scale)
        parameter.grad = fixed[parameter] + applied_chart
        # Experimental optimizer interface. These are the two source-aware
        # channels before ordinary optimizer state destroys their identity.
        parameter._fixed_chart_grad = fixed[parameter]
        parameter._context_chart_grad = applied_chart

    applied_chart_norm = math.sqrt(max(0.0, float(sum(
        torch.sum((scales[parameter] * chart[parameter].double()).square())
        for parameter in fixed
    ))))
    combined = {
        parameter: fixed[parameter] + scales[parameter] * chart[parameter]
        for parameter in fixed
    }
    combined_alignment = float(_global_inner(combined, fixed))
    denominator = max(fixed_norm * chart_norm, eps)
    return live_loss, {
        "fixed_chart_gradient_norm": fixed_norm,
        "context_chart_gradient_norm": chart_norm,
        "context_chart_applied_norm": applied_chart_norm,
        "context_chart_scale": min(scales.values(), default=1.0),
        "context_chart_mean_scale": (
            sum(scales.values()) / len(scales) if scales else 1.0
        ),
        "context_to_fixed_ratio": chart_norm / max(fixed_norm, eps),
        "context_fixed_cosine": float(_global_inner(chart, fixed)) / denominator,
        "fixed_chart_alignment": combined_alignment,
    }
