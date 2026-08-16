from __future__ import annotations

from ML_experiment.models import BudgetMatchedMLP, SoftEikonalNet, parameter_count


VARIANTS = (
    "ordinary_mlp",
    "self_context",
    "self_context_hard",
    "self_context_iterated",
    "self_context_uncertainty",
    "self_context_secant",
    "self_context_chart",
)

JET_VARIANTS = (
    "ordinary_mlp",
    "self_context",
    "self_context_jet_laplacian",
    "self_context_jet_factor",
    "self_context_jet_richardson",
    "self_context_jet_curvature_context",
    "self_context_nested",
)


def make_variant(name: str, input_dim: int, output_dim: int, width: int):
    reference = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25)
    budget = parameter_count(reference)
    if name == "ordinary_mlp":
        model = BudgetMatchedMLP(input_dim, output_dim, width, budget)
    elif name == "self_context_hard":
        model = SoftEikonalNet(input_dim, output_dim, width, temperature=.55,
                               self_context_strength=.25)
    elif name == "self_context_iterated":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               context_steps=2)
    elif name == "self_context_uncertainty":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               uncertainty_context=True)
    elif name in {"self_context", "self_context_secant", "self_context_chart"}:
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25)
    elif name.startswith("self_context_jet_"):
        mode = name.removeprefix("self_context_jet_")
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               jet_mode=mode)
    elif name == "self_context_nested":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               nested_self_context=True)
    else:
        raise KeyError(name)
    assert parameter_count(model) == budget
    return model
