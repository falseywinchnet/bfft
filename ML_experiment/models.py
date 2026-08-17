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
                 jet_mode: str = "none", nested_self_context: bool = False,
                 transport_mode: str = "none", value_mode: str = "transported",
                 primitive_mode: str = "random", allocation_smoothing: float = 0.0,
                 shell_metric_mode: str = "dynamic", shell_samples: int | None = None):
        super().__init__()
        self.directions, self.rank = directions, rank
        self.temperature = float(temperature)
        self.self_context_strength = float(self_context_strength)
        self.context_steps = int(context_steps)
        self.uncertainty_context = bool(uncertainty_context)
        self.nested_self_context = bool(nested_self_context)
        if jet_mode not in {"none", "laplacian", "factor", "richardson",
                            "shell_mean", "shell_midpoint", "shell_mean_orthogonal",
                            "curvature_context", "curvature_context_bounded",
                            "curvature_context_geometric", "curvature_context_detached",
                            "curvature_context_parallel", "curvature_chart_parallel",
                            "curvature_context_orthogonal",
                            "allocation_shell_mixture", "allocation_shell_geodesic",
                            "nested_chart"}:
            raise ValueError(jet_mode)
        self.jet_mode = jet_mode
        if transport_mode not in {"none", "heun", "turn", "self_ray_odd", "self_ray_even",
                                  "basis_ray_odd"}:
            raise ValueError(transport_mode)
        self.transport_mode = transport_mode
        if value_mode not in {"transported", "authentic", "midpoint",
                              "ray_energy", "transported_plus_energy"}:
            raise ValueError(value_mode)
        self.value_mode = value_mode
        if shell_metric_mode not in {"dynamic", "frozen"}:
            raise ValueError(shell_metric_mode)
        self.shell_metric_mode = shell_metric_mode
        self.shell_samples = rank if shell_samples is None else int(shell_samples)
        if not 1 <= self.shell_samples <= rank:
            raise ValueError(shell_samples)
        self.allocation_smoothing = float(allocation_smoothing)
        if not 0 <= self.allocation_smoothing <= 1:
            raise ValueError(allocation_smoothing)
        self.base = nn.Linear(n_in, n_out)
        self.metric = nn.Linear(n_in, rank * rank)
        generator = torch.Generator().manual_seed(9157 + n_in + n_out)
        if primitive_mode == "random":
            primitive = F.normalize(
                torch.randn(directions, rank, n_in, generator=generator), dim=-1
            )
        elif primitive_mode == "tight":
            # A union of orthonormal bases has a scalar frame operator: the
            # atlas begins without privileged aggregate directions while its
            # ray count and all trainable degrees of freedom remain unchanged.
            ray_count = directions * rank
            bases = []
            while sum(len(basis) for basis in bases) < ray_count:
                matrix = torch.randn(n_in, n_in, generator=generator)
                basis, _ = torch.linalg.qr(matrix)
                bases.append(basis.transpose(0, 1))
            primitive = torch.cat(bases, dim=0)[:ray_count].reshape(directions, rank, n_in)
        elif primitive_mode == "stiefel_cycle":
            if n_in < 2 * rank:
                raise ValueError("stiefel_cycle requires n_in >= 2 * rank")
            seed_matrix = torch.randn(n_in, 2 * rank, generator=generator)
            bundle, _ = torch.linalg.qr(seed_matrix)
            origin = bundle[:, :rank].transpose(0, 1)
            tangent = bundle[:, rank:2 * rank].transpose(0, 1)
            angle = torch.arange(directions) * (2 * math.pi / directions)
            primitive = (torch.cos(angle)[:, None, None] * origin[None]
                         + torch.sin(angle)[:, None, None] * tangent[None])
        elif primitive_mode == "stiefel_flow":
            if n_in % 2:
                raise ValueError("stiefel_flow requires an even input width")
            base_matrix = torch.randn(n_in, rank, generator=generator)
            base, _ = torch.linalg.qr(base_matrix)
            mixing_matrix = torch.randn(n_in, n_in, generator=generator)
            mixing, _ = torch.linalg.qr(mixing_matrix)
            canonical = torch.zeros(n_in, n_in)
            # Keep the geometric flow fixed when its discrete sampling is
            # refined; extra views resolve the connection rather than adding
            # higher-frequency rotations.
            maximum_frequency = max(1, min(5, (directions - 1) // 2))
            for plane in range(n_in // 2):
                frequency = 1 + plane % maximum_frequency
                canonical[2 * plane, 2 * plane + 1] = -frequency
                canonical[2 * plane + 1, 2 * plane] = frequency
            generator_matrix = mixing @ canonical @ mixing.transpose(0, 1)
            frames = []
            for direction in range(directions):
                angle = 2 * math.pi * direction / directions
                rotation = torch.matrix_exp(angle * generator_matrix)
                frames.append((rotation @ base).transpose(0, 1))
            primitive = torch.stack(frames)
        else:
            raise ValueError(primitive_mode)
        self.register_buffer("primitive", primitive)
        if self.shell_samples == rank:
            shell_mixer = torch.eye(rank)
        else:
            # Fixed dense directions form a deterministic Hutchinson trace
            # estimate over the complete learned tangent subspace.  Reducing
            # probes does not discard named rays or introduce task axes.
            mixer_source = torch.randn(rank, rank, generator=generator)
            shell_mixer, _ = torch.linalg.qr(mixer_source)
            shell_mixer = shell_mixer[:, :self.shell_samples].transpose(0, 1)
        self.register_buffer("shell_mixer", shell_mixer)
        self.shared = nn.Parameter(torch.randn(rank, n_out) / math.sqrt(rank))
        self.response = nn.Sequential(nn.Linear(4, 12), LELU(), nn.Linear(12, 1))
        self.scale = nn.Parameter(torch.tensor(-1.5))
        self.diagnostic_mode = "matched"
        # Training does not consume the spectral/JS summaries below.  Keep
        # them opt-in so measurement cannot tax the mechanism being measured.
        self.capture_diagnostics = False
        self.last_diagnostics: dict[str, torch.Tensor] = {}
        self.last_weight: torch.Tensor | None = None

    def set_diagnostic_mode(self, mode: str):
        if mode not in {"matched", "mismatched", "uniform", "base_only"}:
            raise ValueError(mode)
        self.diagnostic_mode = mode

    def set_diagnostics_enabled(self, enabled: bool):
        self.capture_diagnostics = enabled

    def _allocation_weights(self, metric, projected):
        cost = torch.einsum("bdr,brs,bds->bd", projected, metric, projected)
        norm = projected.square().mean(-1)
        stats = torch.stack((torch.log1p(cost), torch.log1p(norm), projected.mean(-1),
                             torch.log1p(projected.abs().mean(-1))), -1)
        response = self.response(stats).squeeze(-1)
        logits = response - cost / (cost.mean(1, keepdim=True) + 1e-5)
        weight = torch.softmax(logits / self.temperature, 1)
        if self.allocation_smoothing:
            neighbor = .5 * (torch.roll(weight, 1, 1) + torch.roll(weight, -1, 1))
            weight = ((1 - self.allocation_smoothing) * weight
                      + self.allocation_smoothing * neighbor)
        return weight

    def _allocate(self, x):
        batch = len(x)
        factor = self.metric(x).view(batch, self.rank, self.rank)
        metric = factor @ factor.transpose(1, 2) / self.rank
        projected = torch.einsum("dri,bi->bdr", self.primitive, x)
        weight = self._allocation_weights(metric, projected)
        return metric, projected, weight

    def _lift_context(self, projected, weight):
        return torch.einsum("bd,bdr,dri->bi", weight, projected, self.primitive) / self.rank

    @staticmethod
    def _normalize_like(state, reference):
        state_rms = state.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        reference_rms = reference.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
        return state * (reference_rms / state_rms), state_rms, reference_rms

    @staticmethod
    def _bound_like(state, reference_rms):
        state_rms = state.square().mean(1, keepdim=True).sqrt()
        bounded = state * (reference_rms / (reference_rms + state_rms + 1e-6))
        return bounded, state_rms

    def _shell_curvature(self, x, pooled, weight, radius_scale=1.0, orthogonal=False):
        """Return directional even shell differences in the learned chart.

        The external model still receives one activation. These symmetric
        probes query the layer's own allocation field along the low-rank frame
        already selected by that activation. No labels or neighboring samples
        enter the construction and no full Jacobian is materialized.
        """
        batch = len(x)
        frame = torch.einsum("bd,dri->bri", weight, self.primitive)
        frame = F.normalize(frame, dim=-1)
        if orthogonal:
            gram = frame @ frame.transpose(1, 2)
            identity = torch.eye(self.rank, device=x.device, dtype=x.dtype)[None]
            cholesky = torch.linalg.cholesky(gram + 1e-3 * identity)
            frame = torch.linalg.solve_triangular(cholesky, frame, upper=False)
        if self.shell_samples != self.rank:
            frame = torch.einsum("sr,bri->bsi", self.shell_mixer, frame)
        # The shell is one self-context step from the authentic chart point.
        # Euclidean norm makes the radius invariant to hidden width.
        radius = (self.self_context_strength * radius_scale
                  * x.norm(dim=1, keepdim=True).detach().clamp_min(1e-3))
        displacement = radius[:, None, :] * frame
        probes = torch.cat((x[:, None, :] + displacement,
                            x[:, None, :] - displacement), dim=1)
        _, probe_projected, probe_weight = self._allocate(probes.flatten(0, 1))
        probe_pooled = torch.einsum("bd,bdr->br", probe_weight, probe_projected)
        probe_pooled = probe_pooled.view(batch, 2 * self.shell_samples, self.rank)
        plus, minus = (probe_pooled[:, :self.shell_samples],
                       probe_pooled[:, self.shell_samples:])
        return plus + minus - 2 * pooled[:, None, :]

    def _shell_context_curvature(self, x, metric, projected, weight, orthogonal=False):
        """Lift the even shell response back into activation coordinates."""
        batch = len(x)
        center = torch.einsum("bd,bdr,dri->bi", weight, projected, self.primitive) / self.rank
        frame = torch.einsum("bd,dri->bri", weight, self.primitive)
        frame = F.normalize(frame, dim=-1)
        if orthogonal:
            # An Eikonal shell represents the learned tangent subspace, not
            # the arbitrary skew among the rays used to span it.  Whitening
            # makes the summed even difference invariant to that skew.
            gram = frame @ frame.transpose(1, 2)
            identity = torch.eye(self.rank, device=x.device, dtype=x.dtype)[None]
            cholesky = torch.linalg.cholesky(gram + 1e-3 * identity)
            frame = torch.linalg.solve_triangular(cholesky, frame, upper=False)
        if self.shell_samples != self.rank:
            frame = torch.einsum("sr,bri->bsi", self.shell_mixer, frame)
        radius = (self.self_context_strength
                  * x.norm(dim=1, keepdim=True).detach().clamp_min(1e-3))
        displacement = radius[:, None, :] * frame
        if self.shell_metric_mode == "frozen":
            # The center measures the local metric once.  Because every view
            # projection is linear, transport shell projections exactly rather
            # than recomputing either coordinates or geometry at each probe.
            delta_projected = torch.einsum(
                "dki,bsi->bsdk", self.primitive, displacement
            )
            center_projected = projected[:, None]
            probe_projected = torch.cat(
                (center_projected + delta_projected,
                 center_projected - delta_projected), dim=1
            ).flatten(0, 1)
            probe_metric = metric[:, None].expand(
                -1, 2 * self.shell_samples, -1, -1
            ).flatten(0, 1)
            probe_weight = self._allocation_weights(probe_metric, probe_projected)
        else:
            probes = torch.cat((x[:, None, :] + displacement,
                                x[:, None, :] - displacement), dim=1)
            _, probe_projected, probe_weight = self._allocate(probes.flatten(0, 1))
        probe_context = torch.einsum(
            "bd,bdr,dri->bi", probe_weight, probe_projected, self.primitive
        ) / self.rank
        probe_context = probe_context.view(batch, 2 * self.shell_samples, x.shape[1])
        plus, minus = (probe_context[:, :self.shell_samples],
                       probe_context[:, self.shell_samples:])
        return (plus + minus - 2 * center[:, None, :]).mean(dim=1)

    def _shell_allocation(self, x, weight):
        """Symmetric mean allocation on the learned rank-r Eikonal shell."""
        batch = len(x)
        frame = torch.einsum("bd,dri->bri", weight, self.primitive)
        frame = F.normalize(frame, dim=-1)
        radius = (self.self_context_strength
                  * x.norm(dim=1, keepdim=True).detach().clamp_min(1e-3))
        displacement = radius[:, None, :] * frame
        shell = torch.cat((x[:, None, :] + displacement,
                           x[:, None, :] - displacement), dim=1)
        _, _, shell_weight = self._allocate(shell.flatten(0, 1))
        return shell_weight.view(batch, 2 * self.rank, self.directions).mean(1)

    def _jet_state(self, x, pooled, weight):
        curvature = self._shell_curvature(
            x, pooled, weight, orthogonal=self.jet_mode == "shell_mean_orthogonal"
        )
        laplacian = curvature.mean(dim=1)
        if self.jet_mode in {"shell_mean", "shell_mean_orthogonal"}:
            # Each curvature factor is p+ + p- - 2p.  Half its mean moves p
            # exactly to the mean response over the symmetric shell.
            return .5 * laplacian
        if self.jet_mode == "shell_midpoint":
            return .25 * laplacian
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
        extra_diagnostics = {}
        chart_points = 1
        metric, projected, weight = self._allocate(x)
        authentic_projected = projected
        initial_weight = weight
        if self.transport_mode != "none" and self.self_context_strength:
            # Treat self-context as a vector field and retain how the field
            # changes along its own proposed flow.  These are parameter-free
            # predictor/corrector constructions, not an imposed shell atlas.
            context0 = self._lift_context(projected, weight)
            v0, context0_rms, input_rms = self._normalize_like(context0, x)
            gain = self.self_context_strength
            z_plus = x + gain * v0
            _, projected_plus, weight_plus = self._allocate(z_plus)
            chart_points += 1
            context_plus = self._lift_context(projected_plus, weight_plus)
            v_plus, _, _ = self._normalize_like(context_plus, x)
            innovation = v_plus - v0

            if self.transport_mode == "heun":
                flow = v0 + .5 * innovation
            elif self.transport_mode == "turn":
                coefficient = ((innovation * v0).sum(1, keepdim=True)
                               / v0.square().sum(1, keepdim=True).clamp_min(1e-6))
                turn = innovation - coefficient * v0
                bounded_turn, turn_rms = self._bound_like(turn, input_rms)
                flow = v0 + .5 * bounded_turn
                innovation_rms = innovation.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
                extra_diagnostics["turn_fraction"] = turn_rms / innovation_rms
            elif self.transport_mode == "basis_ray_odd":
                # Resolve the local differential along the allocator's own
                # rank-r eikonal basis rays.  Odd probe responses are
                # contracted covariantly back along the selected transport
                # direction; unlike the even shell, orientation is retained.
                frame = torch.einsum("bd,dri->bri", weight_plus, self.primitive)
                frame = F.normalize(frame, dim=-1)
                radius = (gain * z_plus.norm(dim=1, keepdim=True).detach().clamp_min(1e-3))
                displacement = radius[:, None, :] * frame
                probes = torch.cat((z_plus[:, None, :] + displacement,
                                    z_plus[:, None, :] - displacement), dim=1)
                _, probe_projected, probe_weight = self._allocate(probes.flatten(0, 1))
                chart_points += 2 * self.rank
                probe_context = self._lift_context(probe_projected, probe_weight)
                reference = z_plus[:, None, :].expand(-1, 2 * self.rank, -1).flatten(0, 1)
                probe_context, _, _ = self._normalize_like(probe_context, reference)
                probe_context = probe_context.view(batch, 2 * self.rank, x.shape[1])
                plus, minus = probe_context[:, :self.rank], probe_context[:, self.rank:]
                odd_factors = .5 * (plus - minus)

                gram = frame @ frame.transpose(1, 2)
                identity = torch.eye(self.rank, device=x.device, dtype=x.dtype)[None]
                direction = F.normalize(v_plus, dim=1)
                coordinates = torch.linalg.solve(
                    gram + 1e-3 * identity,
                    torch.einsum("bri,bi->br", frame, direction).unsqueeze(-1),
                ).squeeze(-1)
                ray_state = torch.einsum("br,bri->bi", coordinates, odd_factors)
                bounded_ray, _ = self._bound_like(ray_state, input_rms)
                flow = v0 + .5 * bounded_ray
                extra_diagnostics["basis_coordinate_rms"] = coordinates.square().mean(1, keepdim=True).sqrt()
            else:
                z_minus = x - gain * v0
                _, projected_minus, weight_minus = self._allocate(z_minus)
                chart_points += 1
                context_minus = self._lift_context(projected_minus, weight_minus)
                v_minus, _, _ = self._normalize_like(context_minus, x)
                if self.transport_mode == "self_ray_odd":
                    ray_state = .5 * (v_plus - v_minus)
                else:
                    ray_state = .5 * (v_plus + v_minus) - v0
                bounded_ray, _ = self._bound_like(ray_state, input_rms)
                flow = v0 + .5 * bounded_ray

            chart_input = x + gain * flow
            metric, projected, weight = self._allocate(chart_input)
            chart_points += 1
            normalized_context = v0
            innovation_rms = innovation.square().mean(1, keepdim=True).sqrt()
            extra_diagnostics.update({
                "context_raw_ratio": context0_rms / input_rms,
                "innovation_rms_ratio": innovation_rms / input_rms,
                "context_cosine": F.cosine_similarity(v0, v_plus, dim=1).unsqueeze(1),
            })
        else:
            for _ in range(self.context_steps if self.self_context_strength else 0):
                context = self._lift_context(projected, weight)
                normalized_context, context_rms, input_rms = self._normalize_like(context, x)
                gain = self.self_context_strength
                if self.uncertainty_context:
                    entropy = -(weight * torch.log(weight + 1e-9)).sum(1, keepdim=True) / math.log(self.directions)
                    gain = gain * entropy
                # Anchor every iteration to the authentic activation. Repeated
                # context steps refine the chart rather than accumulating drift.
                augmented = x + gain * normalized_context
                chart_input = augmented
                metric, projected, weight = self._allocate(augmented)
                chart_points += 1
            if normalized_context is not None:
                extra_diagnostics["context_raw_ratio"] = context_rms / input_rms
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
            chart_points += 2
        if self.jet_mode in {"curvature_context", "curvature_context_bounded",
                             "curvature_context_geometric", "curvature_context_detached",
                             "curvature_context_parallel", "curvature_chart_parallel",
                             "curvature_context_orthogonal"}:
            curvature_context = self._shell_context_curvature(
                chart_input, metric, projected, weight,
                orthogonal=self.jet_mode == "curvature_context_orthogonal",
            )
            curvature_rms = curvature_context.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            chart_rms = chart_input.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
            chart_points += 2 * self.shell_samples
            if self.jet_mode == "curvature_context_detached":
                curvature_context = curvature_context.detach()
                injected = curvature_context * (chart_rms / curvature_rms.detach())
                authority = torch.ones_like(curvature_rms)
            elif self.jet_mode == "curvature_context_bounded":
                injected, _ = self._bound_like(curvature_context, chart_rms)
                authority = curvature_rms / (chart_rms + curvature_rms)
            elif self.jet_mode == "curvature_context_geometric":
                # The geometric mean between the raw shell magnitude and the
                # old full-strength normalization.  Weak evidence remains weak
                # without becoming numerically invisible; strong evidence is
                # still capped at the chart scale.
                authority = (curvature_rms / (chart_rms + curvature_rms)).sqrt()
                injected = curvature_context * (chart_rms / curvature_rms) * authority
            elif self.jet_mode in {"curvature_context_parallel", "curvature_chart_parallel"}:
                # Curvature found the missing radial transitions, but its
                # component transverse to the established chart flow tore
                # otherwise closed level sets.  Keep the full signed second
                # difference while allowing it to change only distance along
                # an already acquired direction.
                full_scale = curvature_context * (chart_rms / curvature_rms)
                reference = (normalized_context
                             if self.jet_mode == "curvature_context_parallel"
                             else chart_input)
                coefficient = ((full_scale * reference).sum(1, keepdim=True)
                               / reference.square().sum(1, keepdim=True).clamp_min(1e-6))
                injected = coefficient * reference
                authority = injected.square().mean(1, keepdim=True).sqrt() / chart_rms
                extra_diagnostics["curvature_parallel_cosine"] = F.cosine_similarity(
                    curvature_context, reference, dim=1
                ).abs().unsqueeze(1)
            else:
                injected = curvature_context * (chart_rms / curvature_rms)
                authority = torch.ones_like(curvature_rms)
            chart_input = chart_input + self.self_context_strength * injected
            metric, projected, weight = self._allocate(chart_input)
            chart_points += 1
            extra_diagnostics.update({
                "curvature_raw_ratio": curvature_rms / chart_rms,
                "curvature_authority": authority,
            })
        if self.jet_mode in {"allocation_shell_mixture", "allocation_shell_geodesic"}:
            shell_weight = self._shell_allocation(chart_input, weight)
            chart_points += 2 * self.rank
            shell_midpoint = .5 * (weight + shell_weight)
            shell_js = .5 * (
                (weight * (torch.log(weight + 1e-9) - torch.log(shell_midpoint + 1e-9))).sum(1)
                + (shell_weight * (torch.log(shell_weight + 1e-9) - torch.log(shell_midpoint + 1e-9))).sum(1)
            )
            if self.jet_mode == "allocation_shell_mixture":
                weight = (1 - self.self_context_strength) * weight + self.self_context_strength * shell_weight
            else:
                center_log = torch.log(weight + 1e-9)
                shell_log = torch.log(shell_weight + 1e-9)
                weight = torch.softmax(
                    center_log + self.self_context_strength * (shell_log - center_log), dim=1
                )
            extra_diagnostics["shell_allocation_js"] = shell_js.unsqueeze(1)
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
        if self.value_mode == "authentic":
            output_projected = authentic_projected
        elif self.value_mode == "midpoint":
            output_projected = .5 * (authentic_projected + projected)
        elif self.value_mode == "ray_energy":
            output_projected = projected * torch.tanh(projected)
        elif self.value_mode == "transported_plus_energy":
            output_projected = projected + projected * torch.tanh(projected)
        else:
            output_projected = projected
        pooled = torch.einsum("bd,bdr->br", weight, output_projected)
        if self.jet_mode not in {"none", "curvature_context", "curvature_context_bounded",
                                "curvature_context_geometric", "curvature_context_detached",
                                "curvature_context_parallel", "curvature_chart_parallel",
                                "curvature_context_orthogonal",
                                "allocation_shell_mixture", "allocation_shell_geodesic",
                                "nested_chart"}:
            pooled = pooled + self._jet_state(chart_input, pooled, weight)
        correction = F.softplus(self.scale) * (pooled @ self.shared)
        if self.diagnostic_mode == "base_only":
            correction = torch.zeros_like(correction)
        base_output = self.base(x)
        if self.capture_diagnostics:
            eigenvalues = torch.linalg.eigvalsh(metric.detach()).clamp_min(1e-8)
            midpoint = .5 * (initial_weight + matched_weight)
            allocation_js = .5 * (
                (initial_weight * (torch.log(initial_weight + 1e-9) - torch.log(midpoint + 1e-9))).sum(1)
                + (matched_weight * (torch.log(matched_weight + 1e-9) - torch.log(midpoint + 1e-9))).sum(1)
            )
            self.last_diagnostics = {
                "weight": matched_weight.detach(),
                "entropy": (-(matched_weight * torch.log(matched_weight + 1e-9)).sum(1) / math.log(self.directions)).detach(),
                "condition": (eigenvalues[:, -1] / eigenvalues[:, 0]).detach(),
                "base_norm": base_output.detach().norm(dim=1),
                "correction_norm": correction.detach().norm(dim=1),
                "allocation_js": allocation_js.detach(),
                "chart_points": torch.full((batch,), chart_points, device=x.device),
                **{key: value.detach().squeeze(-1) for key, value in extra_diagnostics.items()},
            }
        return base_output + correction


class SoftEikonalNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int,
                 temperature: float = 1.0, self_context_strength: float = 0.0,
                 context_steps: int = 1, uncertainty_context: bool = False,
                 jet_mode: str = "none", nested_self_context: bool = False,
                 transport_mode: str = "none", value_mode: str = "transported",
                 primitive_mode: str = "random", directions: int = 12,
                 rank: int = 4, allocation_smoothing: float = 0.0,
                 shell_metric_mode: str = "dynamic", curvature_layers: str = "both",
                 shell_samples: int | None = None):
        super().__init__()
        if curvature_layers not in {"both", "up", "down"}:
            raise ValueError(curvature_layers)
        up_jet = jet_mode if curvature_layers in {"both", "up"} else "none"
        down_jet = jet_mode if curvature_layers in {"both", "down"} else "none"
        self.embed = nn.Linear(input_dim, width)
        self.up = SoftEikonalLinear(width, 2 * width, directions=directions, rank=rank,
                                    temperature=temperature,
                                    self_context_strength=self_context_strength,
                                    context_steps=context_steps, uncertainty_context=uncertainty_context,
                                    jet_mode=up_jet, nested_self_context=nested_self_context,
                                    transport_mode=transport_mode, value_mode=value_mode,
                                    primitive_mode=primitive_mode,
                                    allocation_smoothing=allocation_smoothing,
                                    shell_metric_mode=shell_metric_mode,
                                    shell_samples=shell_samples)
        self.down = SoftEikonalLinear(2 * width, width, directions=directions, rank=rank,
                                      temperature=temperature,
                                      self_context_strength=self_context_strength,
                                      context_steps=context_steps, uncertainty_context=uncertainty_context,
                                      jet_mode=down_jet, nested_self_context=nested_self_context,
                                      transport_mode=transport_mode, value_mode=value_mode,
                                      primitive_mode=primitive_mode,
                                      allocation_smoothing=allocation_smoothing,
                                      shell_metric_mode=shell_metric_mode,
                                      shell_samples=shell_samples)
        self.activation = LELU()
        self.output = nn.Linear(width, output_dim)

    def set_diagnostic_mode(self, mode: str):
        self.up.set_diagnostic_mode(mode); self.down.set_diagnostic_mode(mode)

    def set_diagnostics_enabled(self, enabled: bool):
        self.up.set_diagnostics_enabled(enabled); self.down.set_diagnostics_enabled(enabled)

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
