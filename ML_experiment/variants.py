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
    "self_context_nested_chart",
)

TRANSPORT_VARIANTS = (
    "self_context",
    "self_context_iterated",
    "self_context_transport_heun",
    "self_context_transport_turn",
    "self_context_transport_self_ray_odd",
    "self_context_transport_self_ray_even",
    "self_context_transport_basis_ray_odd",
    "self_context_jet_curvature_context",
    "self_context_jet_curvature_bounded",
    "self_context_jet_curvature_geometric",
    "self_context_jet_curvature_detached",
)

RADIAL_LAB_VARIANTS = (
    "self_context",
    "self_context_source",
    "self_context_transport_self_ray_odd",
    "self_context_transport_self_ray_odd_source",
    "self_context_transport_self_ray_odd_midpoint",
)

RADIAL_SHELL_VARIANTS = (
    "self_context_jet_allocation_shell_mixture",
    "self_context_jet_allocation_shell_geodesic",
)

RADIAL_PARALLEL_VARIANTS = (
    "self_context_jet_curvature_context_parallel",
    "self_context_jet_curvature_chart_parallel",
)

RADIAL_FRAME_VARIANTS = (
    "self_context_jet_curvature_context_orthogonal",
    "self_context_tight_frame",
    "self_context_tight_frame_curvature",
    "self_context_tight_frame_24",
    "self_context_tight_frame_24_curvature",
    "self_context_stiefel_cycle",
    "self_context_stiefel_cycle_curvature",
    "self_context_stiefel_flow",
    "self_context_stiefel_flow_curvature",
    "self_context_stiefel_flow_smooth25_curvature",
    "self_context_stiefel_flow_smooth50_curvature",
    "self_context_stiefel_flow_24_curvature",
)

RADIAL_ENERGY_VARIANTS = (
    "self_context_ray_energy",
    "self_context_tight_frame_ray_energy",
    "self_context_tight_frame_curvature_ray_energy",
)

RADIAL_INTEGRAL_VARIANTS = (
    "self_context_jet_shell_mean",
    "self_context_jet_shell_midpoint",
    "self_context_jet_shell_mean_orthogonal",
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
    elif name == "self_context_source":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               value_mode="authentic")
    elif name == "self_context_ray_energy":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               value_mode="ray_energy")
    elif name == "self_context_tight_frame_ray_energy":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="tight", value_mode="ray_energy")
    elif name == "self_context_tight_frame_curvature_ray_energy":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="tight", value_mode="ray_energy",
                               jet_mode="curvature_context_orthogonal")
    elif name == "self_context_tight_frame":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="tight")
    elif name == "self_context_tight_frame_curvature":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="tight", jet_mode="curvature_context_orthogonal")
    elif name == "self_context_tight_frame_24":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="tight", directions=24)
    elif name == "self_context_tight_frame_24_curvature":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="tight", directions=24,
                               jet_mode="curvature_context_orthogonal")
    elif name == "self_context_stiefel_cycle":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="stiefel_cycle")
    elif name == "self_context_stiefel_cycle_curvature":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="stiefel_cycle",
                               jet_mode="curvature_context_orthogonal")
    elif name == "self_context_stiefel_flow":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="stiefel_flow")
    elif name == "self_context_stiefel_flow_curvature":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="stiefel_flow",
                               jet_mode="curvature_context_orthogonal")
    elif name == "self_context_stiefel_flow_smooth25_curvature":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="stiefel_flow", allocation_smoothing=.25,
                               jet_mode="curvature_context_orthogonal")
    elif name == "self_context_stiefel_flow_smooth50_curvature":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="stiefel_flow", allocation_smoothing=.5,
                               jet_mode="curvature_context_orthogonal")
    elif name == "self_context_stiefel_flow_24_curvature":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               primitive_mode="stiefel_flow", directions=24,
                               jet_mode="curvature_context_orthogonal")
    elif name.startswith("self_context_jet_"):
        mode = name.removeprefix("self_context_jet_")
        mode = {
            "curvature_bounded": "curvature_context_bounded",
            "curvature_geometric": "curvature_context_geometric",
            "curvature_detached": "curvature_context_detached",
            "curvature_context_parallel": "curvature_context_parallel",
            "curvature_chart_parallel": "curvature_chart_parallel",
            "curvature_context_orthogonal": "curvature_context_orthogonal",
        }.get(mode, mode)
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               jet_mode=mode)
    elif name == "self_context_nested":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               nested_self_context=True)
    elif name == "self_context_nested_chart":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               jet_mode="nested_chart")
    elif name == "self_context_transport_self_ray_odd_source":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               transport_mode="self_ray_odd", value_mode="authentic")
    elif name == "self_context_transport_self_ray_odd_midpoint":
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               transport_mode="self_ray_odd", value_mode="midpoint")
    elif name.startswith("self_context_transport_"):
        mode = name.removeprefix("self_context_transport_")
        model = SoftEikonalNet(input_dim, output_dim, width, self_context_strength=.25,
                               transport_mode=mode)
    else:
        raise KeyError(name)
    assert parameter_count(model) == budget
    return model
