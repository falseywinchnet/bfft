from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = math.pi / math.sqrt(3.0)

    def forward(self, x):
        return x * torch.sigmoid(self.scale * x)


class SoftEikonalLinear(nn.Module):
    """Dense affine map plus continuous metric allocation over fixed directions."""

    def __init__(self, n_in: int, n_out: int, directions: int = 12, rank: int = 4,
                 temperature: float = 1.0, self_context_strength: float = 0.0,
                 context_steps: int = 1, uncertainty_context: bool = False,
                 jet_mode: str = "none", nested_self_context: bool = False):
        super().__init__()
        self.directions, self.rank = directions, rank
        self.temperature = float(temperature)
        self.self_context_strength = float(self_context_strength)
        self.context_steps = int(context_steps)
        self.uncertainty_context = bool(uncertainty_context)
        self.nested_self_context = bool(nested_self_context)
        if jet_mode not in {"none", "laplacian", "factor", "richardson",
                            "curvature_context", "nested_chart"}:
            raise ValueError(jet_mode)
        self.jet_mode = jet_mode
        self.base = nn.Linear(n_in, n_out)
        self.metric = nn.Linear(n_in, rank * rank)
        generator = torch.Generator().manual_seed(9157 + n_in + n_out)
        primitive = torch.randn(directions, rank, n_in, generator=generator)
        self.register_buffer("primitive", F.normalize(primitive, dim=-1))
        self.shared = nn.Parameter(torch.randn(rank, n_out) / math.sqrt(rank))
        self.response = nn.Sequential(nn.Linear(4, 12), LELU(), nn.Linear(12, 1))
        self.scale = nn.Parameter(torch.tensor(-1.5))
        self.diagnostic_mode = "matched"
        self.last_diagnostics: dict[str, torch.Tensor] = {}
        self.last_weight: torch.Tensor | None = None

    def set_diagnostic_mode(self, mode: str):
        if mode not in {"matched", "mismatched", "uniform", "base_only"}:
            raise ValueError(mode)
        self.diagnostic_mode = mode

    def _allocate(self, x):
        batch = len(x)
        factor = self.metric(x).view(batch, self.rank, self.rank)
        metric = factor @ factor.transpose(1, 2) / self.rank
        projected = torch.einsum("dri,bi->bdr", self.primitive, x)
        cost = torch.einsum("bdr,brs,bds->bd", projected, metric, projected)
        norm = projected.square().mean(-1)
        stats = torch.stack((torch.log1p(cost), torch.log1p(norm), projected.mean(-1),
                             torch.log1p(projected.abs().mean(-1))), -1)
        response = self.response(stats).squeeze(-1)
        logits = response - cost / (cost.mean(1, keepdim=True) + 1e-5)
        weight = torch.softmax(logits / self.temperature, 1)
        return metric, projected, weight

    def _shell_curvature(self, x, pooled, weight, radius_scale=1.0):
        """Return directional even shell differences in the learned chart.

        The external model still receives one activation. These symmetric
        probes query the layer's own allocation field along the low-rank frame
        already selected by that activation. No labels or neighboring samples
        enter the construction and no full Jacobian is materialized.
        """
        batch = len(x)
        frame = torch.einsum("bd,dri->bri", weight, self.primitive)
        frame = F.normalize(frame, dim=-1)
        # The shell is one self-context step from the authentic chart point.
        # Euclidean norm makes the radius invariant to hidden width.
        radius = (self.self_context_strength * radius_scale
                  * x.norm(dim=1, keepdim=True).detach().clamp_min(1e-3))
        displacement = radius[:, None, :] * frame
        probes = torch.cat((x[:, None, :] + displacement,
                            x[:, None, :] - displacement), dim=1)
        _, probe_projected, probe_weight = self._allocate(probes.flatten(0, 1))
        probe_pooled = torch.einsum("bd,bdr->br", probe_weight, probe_projected)
        probe_pooled = probe_pooled.view(batch, 2 * self.rank, self.rank)
        plus, minus = probe_pooled[:, :self.rank], probe_pooled[:, self.rank:]
        return plus + minus - 2 * pooled[:, None, :]

    def _shell_context_curvature(self, x, projected, weight):
        """Lift the even shell response back into activation coordinates."""
        batch = len(x)
        center = torch.einsum("bd,bdr,dri->bi", weight, projected, self.primitive) / self.rank
        frame = torch.einsum("bd,dri->bri", weight, self.primitive)
        frame = F.normalize(frame, dim=-1)
        radius = (self.self_context_strength
                  * x.norm(dim=1, keepdim=True).detach().clamp_min(1e-3))
        displacement = radius[:, None, :] * frame
        probes = torch.cat((x[:, None, :] + displacement,
                            x[:, None, :] - displacement), dim=1)
        _, probe_projected, probe_weight = self._allocate(probes.flatten(0, 1))
        probe_context = torch.einsum(
            "bd,bdr,dri->bi", probe_weight, probe_projected, self.primitive
        ) / self.rank
        probe_context = probe_context.view(batch, 2 * self.rank, x.shape[1])
        plus, minus = probe_context[:, :self.rank], probe_context[:, self.rank:]
        return (plus + minus - 2 * center[:, None, :]).mean(dim=1)

    def _jet_state(self, x, pooled, weight):
        curvature = self._shell_curvature(x, pooled, weight)
        laplacian = curvature.mean(dim=1)
        if self.jet_mode == "laplacian":
            return laplacian
        if self.jet_mode == "richardson":
            outer = self._shell_curvature(x, pooled, weight, radius_scale=2.0).mean(dim=1)
            # Cancel the leading fourth-order shell error while preserving a
            # curvature state with the same units as the pooled response.
            return (4 * laplacian - .25 * outer) / 3
        if self.jet_mode == "factor":
            # K^T K is invariant to reordering/rotation of the sampled tangent
            # factors. Acting on the current response retains focal curvature
            # energy instead of flattening it.
            gram = curvature.transpose(1, 2) @ curvature / self.rank
            factored = torch.einsum("brs,bs->br", gram, pooled)
            target_rms = curvature.square().mean((1, 2), keepdim=False).sqrt().unsqueeze(1)
            factor_rms = factored.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            return factored * (target_rms / factor_rms)
        return torch.zeros_like(pooled)

    def forward(self, x):
        batch = len(x)
        chart_input = x
        normalized_context = None
        metric, projected, weight = self._allocate(x)
        initial_weight = weight
        for _ in range(self.context_steps if self.self_context_strength else 0):
            context = torch.einsum("bd,bdr,dri->bi", weight, projected, self.primitive) / self.rank
            context_rms = context.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            input_rms = x.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
            gain = self.self_context_strength
            if self.uncertainty_context:
                entropy = -(weight * torch.log(weight + 1e-9)).sum(1, keepdim=True) / math.log(self.directions)
                gain = gain * entropy
            normalized_context = context * (input_rms / context_rms)
            # Anchor every iteration to the authentic activation. Repeated
            # context steps refine the chart rather than accumulating drift.
            augmented = x + gain * normalized_context
            chart_input = augmented
            metric, projected, weight = self._allocate(augmented)
        if self.nested_self_context and normalized_context is not None:
            _, outer_projected, outer_weight = self._allocate(normalized_context)
            outer_context = torch.einsum(
                "bd,bdr,dri->bi", outer_weight, outer_projected, self.primitive
            ) / self.rank
            outer_rms = outer_context.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            input_rms = x.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
            outer_context = outer_context * (input_rms / outer_rms)
            nested_proposal = normalized_context + self.self_context_strength * outer_context
            chart_input = x + self.self_context_strength * nested_proposal
            metric, projected, weight = self._allocate(chart_input)
        if self.jet_mode == "curvature_context":
            curvature_context = self._shell_context_curvature(chart_input, projected, weight)
            curvature_rms = curvature_context.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            chart_rms = chart_input.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
            chart_input = chart_input + self.self_context_strength * curvature_context * (chart_rms / curvature_rms)
            metric, projected, weight = self._allocate(chart_input)
        if self.jet_mode == "nested_chart":
            # The first chart transition is a tangent displacement on the
            # allocation simplex. Lift that transition into activation space,
            # let the same continuous atlas interpret it, then allow the outer
            # chart to modify selection rather than position.
            transition_log = torch.log(weight + 1e-9) - torch.log(initial_weight + 1e-9)
            transition_log = transition_log - transition_log.mean(1, keepdim=True)
            view_context = torch.einsum("bdr,dri->bdi", projected, self.primitive) / self.rank
            transition = torch.einsum("bd,bdi->bi", transition_log, view_context)
            transition_rms = transition.square().mean(1, keepdim=True).sqrt()
            input_rms = x.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
            # Preserve infinitesimal transitions and bound only large ones.
            bounded_transition = transition * (input_rms / (input_rms + transition_rms + 1e-6))
            _, _, outer_weight = self._allocate(bounded_transition)
            outer_log = torch.log(outer_weight + 1e-9)
            outer_log = outer_log - outer_log.mean(1, keepdim=True)
            transition_strength = transition_rms / (input_rms + transition_rms + 1e-6)
            weight = torch.softmax(
                torch.log(weight + 1e-9)
                + self.self_context_strength * transition_strength * outer_log,
                dim=1,
            )
        matched_weight = weight
        self.last_weight = matched_weight
        if self.diagnostic_mode == "mismatched" and batch > 1:
            weight = torch.roll(weight, 1, 0)
        elif self.diagnostic_mode == "uniform":
            weight = torch.full_like(weight, 1 / self.directions)
        pooled = torch.einsum("bd,bdr->br", weight, projected)
        if self.jet_mode not in {"none", "curvature_context", "nested_chart"}:
            pooled = pooled + self._jet_state(chart_input, pooled, weight)
        correction = F.softplus(self.scale) * (pooled @ self.shared)
        if self.diagnostic_mode == "base_only":
            correction = torch.zeros_like(correction)
        eigenvalues = torch.linalg.eigvalsh(metric.detach()).clamp_min(1e-8)
        self.last_diagnostics = {
            "weight": matched_weight.detach(),
            "entropy": (-(matched_weight * torch.log(matched_weight + 1e-9)).sum(1) / math.log(self.directions)).detach(),
            "condition": (eigenvalues[:, -1] / eigenvalues[:, 0]).detach(),
            "base_norm": self.base(x).detach().norm(dim=1),
            "correction_norm": correction.detach().norm(dim=1),
        }
        return self.base(x) + correction


class SoftEikonalNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int,
                 temperature: float = 1.0, self_context_strength: float = 0.0,
                 context_steps: int = 1, uncertainty_context: bool = False,
                 jet_mode: str = "none", nested_self_context: bool = False):
        super().__init__()
        self.embed = nn.Linear(input_dim, width)
        self.up = SoftEikonalLinear(width, 2 * width, temperature=temperature,
                                    self_context_strength=self_context_strength,
                                    context_steps=context_steps, uncertainty_context=uncertainty_context,
                                    jet_mode=jet_mode, nested_self_context=nested_self_context)
        self.down = SoftEikonalLinear(2 * width, width, temperature=temperature,
                                      self_context_strength=self_context_strength,
                                      context_steps=context_steps, uncertainty_context=uncertainty_context,
                                      jet_mode=jet_mode, nested_self_context=nested_self_context)
        self.activation = LELU()
        self.output = nn.Linear(width, output_dim)

    def set_diagnostic_mode(self, mode: str):
        self.up.set_diagnostic_mode(mode); self.down.set_diagnostic_mode(mode)

    def forward(self, x):
        return self.output(self.down(self.activation(self.up(self.embed(x)))))

    def diagnostics(self):
        return {"up": self.up.last_diagnostics, "down": self.down.last_diagnostics}

    def allocation_weights(self):
        return self.up.last_weight, self.down.last_weight


class BudgetMatchedAffine(nn.Module):
    """An overparameterized network whose end-to-end function is exactly affine.

    Every trainable parameter is active. A three-affine-layer path consumes the
    bulk of the budget. Any exact-count remainder weights fixed affine basis
    maps, so matching does not rely on dead padding parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, parameter_budget: int):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim

        def deep_count(hidden: int):
            return ((input_dim + 1) * hidden + (hidden + 1) * hidden
                    + (hidden + 1) * output_dim)

        hidden = 1
        while deep_count(hidden + 1) <= parameter_budget:
            hidden += 1
        self.hidden = hidden
        self.first = nn.Linear(input_dim, hidden)
        self.middle = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, output_dim)
        remainder = parameter_budget - deep_count(hidden)
        self.extra = nn.Parameter(torch.zeros(remainder))
        generator = torch.Generator().manual_seed(12289 + input_dim + output_dim + parameter_budget)
        if remainder:
            basis_weight = torch.randn(remainder, output_dim, input_dim, generator=generator)
            basis_bias = torch.randn(remainder, output_dim, generator=generator)
            scale = math.sqrt(max(1, input_dim * output_dim))
            self.register_buffer("basis_weight", basis_weight / scale)
            self.register_buffer("basis_bias", basis_bias / math.sqrt(max(1, output_dim)))
        else:
            self.register_buffer("basis_weight", torch.empty(0, output_dim, input_dim))
            self.register_buffer("basis_bias", torch.empty(0, output_dim))

    def forward(self, x):
        result = self.output(self.middle(self.first(x)))
        if self.extra.numel():
            maps = torch.einsum("r,roi->oi", self.extra, self.basis_weight)
            bias = torch.einsum("r,ro->o", self.extra, self.basis_bias)
            result = result + x @ maps.T + bias
        return result

    @torch.no_grad()
    def collapsed(self):
        weight = self.output.weight @ self.middle.weight @ self.first.weight
        bias = self.output.weight @ (self.middle.weight @ self.first.bias + self.middle.bias) + self.output.bias
        if self.extra.numel():
            weight = weight + torch.einsum("r,roi->oi", self.extra, self.basis_weight)
            bias = bias + torch.einsum("r,ro->o", self.extra, self.basis_bias)
        return weight, bias


class BudgetMatchedMLP(nn.Module):
    """Ordinary encode-expand-LELU-contract-decode MLP at an exact budget.

    The latent width matches the soft model. The dense expansion is made as
    wide as the budget permits. At most ``2 * width`` remaining scalars weight
    fixed affine residual maps, so every counted parameter is active without
    changing the ordinary MLP's nonlinear feature class.
    """

    def __init__(self, input_dim: int, output_dim: int, width: int, parameter_budget: int):
        super().__init__()
        self.input_dim, self.output_dim, self.width = input_dim, output_dim, width

        fixed = (input_dim + 1) * width + width + (width + 1) * output_dim
        per_hidden = 2 * width + 1
        self.expansion = (parameter_budget - fixed) // per_hidden
        if self.expansion < 1:
            raise ValueError("parameter budget is too small for the requested MLP")
        self.encode = nn.Linear(input_dim, width)
        self.up = nn.Linear(width, self.expansion)
        self.activation = LELU()
        self.down = nn.Linear(self.expansion, width)
        self.decode = nn.Linear(width, output_dim)

        dense_count = fixed + per_hidden * self.expansion
        remainder = parameter_budget - dense_count
        self.extra = nn.Parameter(torch.zeros(remainder))
        generator = torch.Generator().manual_seed(17159 + input_dim + output_dim + parameter_budget)
        if remainder:
            basis_weight = torch.randn(remainder, output_dim, input_dim, generator=generator)
            basis_bias = torch.randn(remainder, output_dim, generator=generator)
            self.register_buffer("basis_weight", basis_weight / math.sqrt(max(1, input_dim * output_dim)))
            self.register_buffer("basis_bias", basis_bias / math.sqrt(max(1, output_dim)))
        else:
            self.register_buffer("basis_weight", torch.empty(0, output_dim, input_dim))
            self.register_buffer("basis_bias", torch.empty(0, output_dim))

    def forward(self, x):
        result = self.decode(self.down(self.activation(self.up(self.encode(x)))))
        if self.extra.numel():
            weight = torch.einsum("r,roi->oi", self.extra, self.basis_weight)
            bias = torch.einsum("r,ro->o", self.extra, self.basis_bias)
            result = result + x @ weight.T + bias
        return result


def parameter_count(model: nn.Module):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_matched_pair(input_dim: int, output_dim: int, width: int):
    soft = SoftEikonalNet(input_dim, output_dim, width)
    budget = parameter_count(soft)
    linear = BudgetMatchedAffine(input_dim, output_dim, budget)
    assert parameter_count(linear) == budget
    return linear, soft


def make_mlp_pair(input_dim: int, output_dim: int, width: int):
    soft = SoftEikonalNet(input_dim, output_dim, width)
    budget = parameter_count(soft)
    mlp = BudgetMatchedMLP(input_dim, output_dim, width, budget)
    assert parameter_count(mlp) == budget
    return mlp, soft
