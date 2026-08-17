from __future__ import annotations

import math

import torch
import torch.nn as nn

from experiments.soft_eikonal_matched.models import (
    BudgetMatchedMLP,
    SoftEikonalNet,
    parameter_count,
)


class BudgetMatchedPairEikonal(nn.Module):
    """Two-sample Eikonal model with exactly the single-sample parameter budget."""

    def __init__(self, input_dim: int, output_dim: int, width: int, budget: int):
        super().__init__()
        pair_width = 1
        while parameter_count(SoftEikonalNet(2 * input_dim, 2 * output_dim, pair_width + 1)) <= budget:
            pair_width += 1
        self.input_dim, self.output_dim = input_dim, output_dim
        self.pair_width = pair_width
        self.core = SoftEikonalNet(2 * input_dim, 2 * output_dim, pair_width)
        remainder = budget - parameter_count(self.core)
        self.extra = nn.Parameter(torch.zeros(remainder))
        generator = torch.Generator().manual_seed(27191 + input_dim + output_dim + budget)
        if remainder:
            weight = torch.randn(remainder, 2 * output_dim, 2 * input_dim, generator=generator)
            bias = torch.randn(remainder, 2 * output_dim, generator=generator)
            self.register_buffer("basis_weight", weight / math.sqrt(max(1, 4 * input_dim * output_dim)))
            self.register_buffer("basis_bias", bias / math.sqrt(max(1, 2 * output_dim)))
        else:
            self.register_buffer("basis_weight", torch.empty(0, 2 * output_dim, 2 * input_dim))
            self.register_buffer("basis_bias", torch.empty(0, 2 * output_dim))
        assert parameter_count(self) == budget

    def forward(self, pair):
        result = self.core(pair)
        if self.extra.numel():
            weight = torch.einsum("r,roi->oi", self.extra, self.basis_weight)
            bias = torch.einsum("r,ro->o", self.extra, self.basis_bias)
            result = result + pair @ weight.T + bias
        return result


class PairZeroEvaluation(nn.Module):
    """Evaluate the first member of a pair while the second input is zero."""

    def __init__(self, pair_model: BudgetMatchedPairEikonal):
        super().__init__(); self.pair_model = pair_model

    def forward(self, x):
        pair = torch.cat((x, torch.zeros_like(x)), 1)
        output = self.pair_model(pair).view(len(x), 2, self.pair_model.output_dim)
        return output[:, 0]


def make_variant(name: str, input_dim: int, output_dim: int, width: int):
    reference = SoftEikonalNet(input_dim, output_dim, width)
    budget = parameter_count(reference)
    if name == "ordinary_mlp":
        model = BudgetMatchedMLP(input_dim, output_dim, width, budget)
    elif name == "paired_zero":
        model = BudgetMatchedPairEikonal(input_dim, output_dim, width, budget)
    elif name == "self_context":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25)
    elif name == "temperature_hard":
        model = SoftEikonalNet(input_dim, output_dim, width, temperature=.55)
    elif name == "temperature_soft":
        model = SoftEikonalNet(input_dim, output_dim, width, temperature=1.8)
    elif name in {"soft_eikonal", "garnish_instructive", "secant_relational", "allocation_secant"}:
        model = SoftEikonalNet(input_dim, output_dim, width)
    else:
        raise KeyError(name)
    assert parameter_count(model) == budget
    return model


VARIANTS = (
    "ordinary_mlp",
    "soft_eikonal",
    "self_context",
    "garnish_instructive",
    "paired_zero",
    "secant_relational",
    "allocation_secant",
    "temperature_hard",
    "temperature_soft",
)
