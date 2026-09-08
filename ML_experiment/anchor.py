"""Anchor: transported lead--lag momentum with live directional restraint.

Anchor carries a slow request in the moving frame defined by consecutive
normalized gradients.  The slow request is parallel-transported to the new
spherical midpoint, mixed with the current gradient to form a faster readout,
and only the readout's component along the live departure axis is restrained.

The optimizer uses no loss lookahead, per-example gradients, clipping, or
adaptive coordinate scaling.  Gradient clipping, if desired, belongs to the
training loop and is therefore shared with comparator optimizers.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch


def _blocks(value: torch.Tensor, cell: str) -> torch.Tensor:
    if cell == "tensor":
        return value.reshape(1, -1)
    if value.ndim >= 2:
        return value.reshape(value.shape[0], -1)
    return value.reshape(1, -1)


def _project_unit_ball(value: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(value, dim=1)
    scale = torch.maximum(norms, torch.ones_like(norms))
    return value / torch.clamp(scale[:, None], min=1e-300)


def _parallel_transport(
    value: torch.Tensor,
    frame_from: torch.Tensor,
    frame_to: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply the minimum rotation between two unit spherical frames."""
    axial = torch.sum(value * frame_from, dim=1)
    tangent = value - axial[:, None] * frame_from
    denominator = 1.0 + torch.sum(frame_from * frame_to, dim=1)
    transported_tangent = tangent - (
        torch.sum(tangent * frame_to, dim=1)
        / torch.clamp(denominator, min=1e-300)
    )[:, None] * (frame_from + frame_to)
    transported = axial[:, None] * frame_to + transported_tangent
    return torch.where(
        (denominator > eps)[:, None], transported, torch.zeros_like(transported)
    )


class Anchor(torch.optim.Optimizer):
    """Transported two-timescale momentum restrained on the live turn axis.

    With ``c=exp(-1)`` and ``a=1-c``, the transported slow request ``s`` and
    applied fast proposal ``f`` obey

        f       = a * transport(s) + c * gradient
        s_next  = a * transport(s) + c * f

    The final request is ``f - alpha*tau*P_d(f)``, where ``d`` is the
    consecutive-gradient departure and ``tau`` is its half-angle energy.
    This projection cannot amplify ``f``.  The slow memory is not restrained.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        *,
        alpha: float = 1.0,
        weight_decay: float = 0.0,
        cell: str = "row",
        eps: float = 1e-24,
        trust_radius: float | None = 0.02,
        recovery: float = 0.05,
        isotropy_cap: float | None = None,
        restrain_momentum: bool = False,
        gradient_trust: float | None = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if cell not in {"row", "tensor"}:
            raise ValueError("cell must be 'row' or 'tensor'")
        if trust_radius is not None and trust_radius <= 0.0:
            raise ValueError("trust_radius must be positive or None")
        if not 0.0 < recovery <= 1.0:
            raise ValueError("recovery must lie in (0, 1]")
        if isotropy_cap is not None and isotropy_cap < 1.0:
            raise ValueError("isotropy_cap must be at least 1 or None")
        if gradient_trust is not None and gradient_trust <= 0.0:
            raise ValueError("gradient_trust must be positive or None")
        super().__init__(params, dict(
            lr=float(lr), alpha=float(alpha), weight_decay=float(weight_decay),
            cell=cell, eps=float(eps), trust_radius=trust_radius,
            recovery=float(recovery), isotropy_cap=isotropy_cap,
            restrain_momentum=bool(restrain_momentum),
            gradient_trust=(
                None if gradient_trust is None else float(gradient_trust)
            ),
        ))
        self.last_ratio = 1.0
        self.last_turn = 0.0
        self.last_alignment = 1.0
        self.last_relative_step = 0.0
        self.last_lr_scale = 1.0
        self.last_proposal_concentration = 1.0
        self.last_update_concentration = 1.0
        self.last_memory_ratio = 1.0
        self.last_certificate_scale = 1.0
        self.last_update_gradient_ratio = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        c = math.exp(-1.0)
        retain = 1.0 - c
        raw_norm2 = 0.0
        applied_norm2 = 0.0
        turn_sum = 0.0
        cell_count = 0
        alignment = 0.0
        gradient_norm2 = 0.0
        parameter_norm2 = 0.0
        step_norm2 = 0.0
        scale_weighted = 0.0
        scale_weight = 0.0
        proposal_concentration = 1.0
        update_concentration = 1.0
        raw_memory_norm2 = 0.0
        stored_memory_norm2 = 0.0
        certificate_weighted = 0.0
        certificate_weight = 0.0

        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Anchor requires dense gradients")

                state = self.state[parameter]
                gradient_blocks = _blocks(gradient.detach(), group["cell"]).double()
                gradient_norm = torch.linalg.vector_norm(gradient_blocks, dim=1)
                gradient_unit = gradient_blocks / torch.clamp(
                    gradient_norm[:, None], min=1e-300
                )

                signature_value = state.get("anchor_signature")
                if signature_value is None:
                    frame_from = gradient_unit
                else:
                    frame_from = _blocks(signature_value, group["cell"]).double()
                    frame_from = frame_from / torch.clamp(
                        torch.linalg.vector_norm(frame_from, dim=1)[:, None],
                        min=1e-300,
                    )

                cosine = torch.sum(
                    gradient_unit * frame_from, dim=1
                ).clamp(-1.0, 1.0)
                turn = 0.5 * (1.0 - cosine)
                departure = gradient_unit - frame_from
                next_signature = _project_unit_ball(gradient_unit + frame_from)
                next_norm = torch.linalg.vector_norm(next_signature, dim=1)
                frame_to = next_signature / torch.clamp(
                    next_norm[:, None], min=1e-300
                )
                state["anchor_signature"] = next_signature.to(
                    gradient.dtype
                ).reshape_as(gradient)

                slow_value = state.get("anchor_slow")
                slow = (
                    torch.zeros_like(gradient_blocks)
                    if slow_value is None
                    else _blocks(slow_value, group["cell"]).double()
                )
                if signature_value is not None:
                    slow = _parallel_transport(
                        slow, frame_from, frame_to, group["eps"]
                    )
                    # At antipodal signature collapse there is no unique
                    # shortest transport.  Reset instead of inventing a turn.
                    slow = torch.where(
                        (next_norm > group["eps"])[:, None],
                        slow,
                        torch.zeros_like(slow),
                    )

                fast = retain * slow + c * gradient_blocks
                slow_next = retain * slow + c * fast

                departure_norm = torch.linalg.vector_norm(departure, dim=1)
                axis = torch.where(
                    (departure_norm > group["eps"])[:, None],
                    departure / torch.clamp(
                        departure_norm[:, None], min=1e-300
                    ),
                    torch.zeros_like(departure),
                )
                projection = torch.sum(fast * axis, dim=1)[:, None] * axis
                applied = fast - (group["alpha"] * turn)[:, None] * projection
                raw_slow_next = slow_next
                if group["restrain_momentum"]:
                    slow_projection = (
                        torch.sum(slow_next * axis, dim=1)[:, None] * axis
                    )
                    slow_next = slow_next - (
                        group["alpha"] * turn
                    )[:, None] * slow_projection

                if group["trust_radius"] is None:
                    lr_scale = torch.ones_like(turn)
                else:
                    parameter_blocks = _blocks(
                        parameter.detach(), group["cell"]
                    ).double()
                    parameter_scale = torch.linalg.vector_norm(
                        parameter_blocks, dim=1
                    ).clamp_min(1.0)
                    proposed_step = group["lr"] * torch.linalg.vector_norm(
                        applied, dim=1
                    )
                    target_scale = torch.clamp(
                        group["trust_radius"] * parameter_scale
                        / torch.clamp(proposed_step, min=1e-300),
                        max=1.0,
                    )
                    previous_scale = state.get("anchor_lr_scale")
                    if previous_scale is None:
                        previous_scale = torch.ones_like(target_scale)
                    else:
                        previous_scale = previous_scale.double()
                    recovered = previous_scale + group["recovery"] * (
                        target_scale - previous_scale
                    )
                    lr_scale = torch.where(
                        target_scale < previous_scale,
                        target_scale,
                        recovered,
                    )
                    state["anchor_lr_scale"] = lr_scale.to(gradient.dtype)
                proposal_row_norm = torch.linalg.vector_norm(applied, dim=1)
                proposal_rms = torch.sqrt(torch.mean(proposal_row_norm.square()))
                if proposal_rms > group["eps"]:
                    proposal_concentration = max(
                        proposal_concentration,
                        float((proposal_row_norm.max() / proposal_rms).item()),
                    )
                tentative = applied * lr_scale[:, None]
                if group["isotropy_cap"] is not None and parameter.ndim >= 2:
                    tentative_row_norm = torch.linalg.vector_norm(
                        tentative, dim=1
                    )
                    tentative_rms = torch.sqrt(torch.mean(
                        tentative_row_norm.square()
                    ))
                    isotropy_scale = torch.clamp(
                        group["isotropy_cap"] * tentative_rms
                        / torch.clamp(tentative_row_norm, min=1e-300),
                        max=1.0,
                    )
                    lr_scale = lr_scale * isotropy_scale
                if group["gradient_trust"] is not None:
                    proposed_norm = (
                        group["lr"]
                        * torch.linalg.vector_norm(applied * lr_scale[:, None], dim=1)
                    )
                    live_limit = group["gradient_trust"] * gradient_norm
                    certificate_scale = torch.clamp(
                        live_limit / torch.clamp(proposed_norm, min=1e-300),
                        max=1.0,
                    )
                    lr_scale = lr_scale * certificate_scale
                    proposal_energy = torch.sum(applied.square(), dim=1)
                    certificate_weighted += float(torch.sum(
                        certificate_scale * proposal_energy
                    ).item())
                    certificate_weight += float(torch.sum(proposal_energy).item())
                actual = applied * lr_scale[:, None]
                actual_row_norm = torch.linalg.vector_norm(actual, dim=1)
                actual_rms = torch.sqrt(torch.mean(actual_row_norm.square()))
                if actual_rms > group["eps"]:
                    update_concentration = max(
                        update_concentration,
                        float((actual_row_norm.max() / actual_rms).item()),
                    )

                state["anchor_slow"] = slow_next.to(
                    gradient.dtype
                ).reshape_as(gradient)
                raw_memory_norm2 += float(torch.sum(raw_slow_next.square()).item())
                stored_memory_norm2 += float(torch.sum(slow_next.square()).item())
                parameter_norm2 += float(
                    torch.sum(parameter.detach().double().square()).item()
                )
                if group["weight_decay"]:
                    decay = 1.0 - (
                        group["lr"] * group["weight_decay"] * lr_scale
                    )
                    _blocks(parameter, group["cell"]).mul_(
                        decay.to(parameter.dtype)[:, None]
                    )
                parameter.add_(
                    actual.to(gradient.dtype).reshape_as(gradient),
                    alpha=-group["lr"],
                )

                raw_norm2 += float(torch.sum(fast.square()).item())
                applied_norm2 += float(torch.sum(actual.square()).item())
                gradient_norm2 += float(torch.sum(gradient_blocks.square()).item())
                alignment += float(torch.sum(actual * gradient_blocks).item())
                step_norm2 += float(
                    group["lr"] ** 2 * torch.sum(actual.square()).item()
                )
                proposal_energy = torch.sum(applied.square(), dim=1)
                scale_weighted += float(torch.sum(
                    lr_scale * proposal_energy
                ).item())
                scale_weight += float(torch.sum(proposal_energy).item())
                turn_sum += float(torch.sum(turn).item())
                cell_count += int(turn.numel())

        self.last_ratio = math.sqrt(applied_norm2 / max(raw_norm2, 1e-300))
        self.last_turn = turn_sum / max(cell_count, 1)
        self.last_alignment = alignment / math.sqrt(
            max(applied_norm2 * gradient_norm2, 1e-300)
        )
        self.last_relative_step = math.sqrt(
            step_norm2 / max(parameter_norm2, 1e-300)
        )
        self.last_lr_scale = scale_weighted / max(scale_weight, 1e-300)
        self.last_proposal_concentration = proposal_concentration
        self.last_update_concentration = update_concentration
        self.last_memory_ratio = math.sqrt(
            stored_memory_norm2 / max(raw_memory_norm2, 1e-300)
        )
        self.last_certificate_scale = (
            certificate_weighted / max(certificate_weight, 1e-300)
            if certificate_weight else 1.0
        )
        self.last_update_gradient_ratio = math.sqrt(
            step_norm2 / max(gradient_norm2, 1e-300)
        )
        if self.last_ratio > 1.0 + 2e-6:
            raise RuntimeError("Anchor amplified its lead request")
        return loss


class RestrainedMomentumAnchor(Anchor):
    """Anchor whose live departure geometry also shapes stored momentum.

    Standing Anchor contracts only the fast request that is applied to the
    parameters.  This variant applies the same rank-one contraction to the
    next slow state before storing it.  Rejected departure energy therefore
    cannot be silently accumulated and returned by momentum on a later step.
    The contraction strength remains the live half-angle turn energy, so a
    nearly unchanged frame leaves the momentum gambit almost untouched.
    """

    def __init__(self, params, *args, **kwargs):
        kwargs["restrain_momentum"] = True
        super().__init__(params, *args, **kwargs)
