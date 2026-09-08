"""Faithful Wolf request generation with projected-signature wrappers."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from experiments.sgd_transport_restraint.optimizer import (
    _apply_live_axis_restraint,
    _local_blocks,
    _parallel_transport_blocks,
    _project_unit_ball,
)


class WolfTransportV2(torch.optim.Optimizer):
    """Wolf's lead-lag momentum core with explicit transport constraints."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        *,
        alpha: float = 1.0,
        cell: str = "row",
        eps: float = 1e-24,
        restrain_output: bool = False,
        restrain_state: bool = False,
        transport_state: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if restrain_state and not restrain_output:
            raise ValueError("restrain_state requires restrain_output")
        defaults = dict(
            lr=lr,
            alpha=alpha,
            cell=cell,
            eps=eps,
            gate="soft",
            fusion=1,
            restrain_output=restrain_output,
            restrain_state=restrain_state,
            transport_state=transport_state,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                self.state[parameter]["wolf_slow"] = torch.zeros_like(parameter)
        self.last_ratio = 1.0
        self.last_turn = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        c = 0.367879441
        retain = 1.0 - c
        raw_norm2 = 0.0
        actual_norm2 = 0.0
        turn_sum = 0.0
        cell_count = 0

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                state = self.state[parameter]
                grad_blocks = _local_blocks(grad, group["cell"]).double()
                grad_norm = torch.linalg.vector_norm(grad_blocks, dim=1)
                grad_unit = grad_blocks / torch.clamp(
                    grad_norm[:, None], min=1e-300
                )
                signature = state.get("wolf2_signature")
                if signature is None:
                    frame_from = grad_unit
                else:
                    frame_from = _local_blocks(
                        signature, group["cell"]
                    ).double()
                    frame_from = frame_from / torch.clamp(
                        torch.linalg.vector_norm(frame_from, dim=1)[:, None],
                        min=1e-300,
                    )
                cosine = torch.sum(
                    grad_unit * frame_from, dim=1
                ).clamp(-1.0, 1.0)
                turns = 0.5 * (1.0 - cosine)
                # The live BFFT transport axis is the departure between the
                # consecutive observed requests.  It is tangent at their
                # spherical midpoint, so it remains the correct axis after
                # transporting Wolf's carried state into the new frame.
                departure = grad_unit - frame_from
                next_signature = _project_unit_ball(grad_unit + frame_from)
                frame_to = next_signature / torch.clamp(
                    torch.linalg.vector_norm(next_signature, dim=1)[:, None],
                    min=1e-300,
                )
                state["wolf2_signature"] = next_signature.to(
                    grad.dtype
                ).reshape_as(grad)

                slow = _local_blocks(
                    state["wolf_slow"], group["cell"]
                ).double()
                if group["transport_state"] and signature is not None:
                    slow = _parallel_transport_blocks(
                        slow, frame_from, frame_to, group["eps"]
                    )
                fast = retain * slow + c * grad_blocks
                slow_next = retain * slow + c * fast

                if group["restrain_output"]:
                    applied = _apply_live_axis_restraint(
                        fast, departure, turns, group
                    )
                    if group["restrain_state"]:
                        slow_next = _apply_live_axis_restraint(
                            slow_next, departure, turns, group
                        )
                else:
                    applied = fast

                state["wolf_slow"] = slow_next.to(
                    grad.dtype
                ).reshape_as(grad)
                parameter.add_(
                    applied.to(grad.dtype).reshape_as(grad), alpha=-group["lr"]
                )
                raw_norm2 += float(torch.sum(fast.square()).item())
                actual_norm2 += float(torch.sum(applied.square()).item())
                turn_sum += float(torch.sum(turns).item())
                cell_count += int(turns.numel())

        self.last_ratio = math.sqrt(actual_norm2 / max(raw_norm2, 1e-300))
        self.last_turn = turn_sum / max(cell_count, 1)
        if self.last_ratio > 1.0 + 2e-6:
            raise RuntimeError("Wolf V2 transport amplified the request")
        return loss


class TransportWolf(torch.optim.Optimizer):
    """Wolf from PyITD, optionally restraining its final requested motion.

    ``mode='none'`` is the source Wolf algorithm. Shape/scalar/rotate modes
    leave Wolf's double leaky state, sign gate, fallback, and random request
    untouched, then transform only the request applied to the parameters.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-3,
        *,
        alpha: float = 1.0,
        cell: str = "row",
        eps: float = 1e-24,
        mode: str = "none",
        observer_source: str = "request",
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if cell not in {"row", "tensor"}:
            raise ValueError("cell must be 'row' or 'tensor'")
        if mode not in {"none", "scalar", "shape", "rotate"}:
            raise ValueError("invalid Wolf transport mode")
        if observer_source not in {"request", "gradient"}:
            raise ValueError("observer_source must be 'request' or 'gradient'")
        super().__init__(
            params,
            dict(
                lr=lr, alpha=alpha, cell=cell, eps=eps, mode=mode,
                observer_source=observer_source,
            ),
        )
        for group in self.param_groups:
            for parameter in group["params"]:
                self.state[parameter]["wolf_integrator"] = torch.zeros_like(
                    parameter
                )
        self.last_ratio = 1.0
        self.last_turn = 0.0
        self.last_fallback_fraction = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        c = 0.367879441
        retain = 1.0 - c
        entries = []
        raw_norm2 = 0.0
        shape_norm2 = 0.0
        turn_sum = 0.0
        cell_count = 0
        fallback_count = 0
        value_count = 0

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                state = self.state[parameter]
                old_integrator = state["wolf_integrator"]
                leaky_request = retain * old_integrator + c * grad
                state["wolf_integrator"] = (
                    retain * old_integrator + c * leaky_request
                )
                sign_agreement = torch.sign(leaky_request) * torch.sign(grad)
                noisy_request = leaky_request * (
                    1.0 + c * (2.0 * torch.rand_like(leaky_request) - 1.0)
                )
                mask = sign_agreement > 0.0
                request = torch.where(mask, noisy_request, parameter.detach())

                blocks = _local_blocks(request, group["cell"]).double()
                norms = torch.linalg.vector_norm(blocks, dim=1)
                valid = norms > group["eps"]
                request_unit = blocks / torch.clamp(
                    norms[:, None], min=1e-300
                )
                if group["observer_source"] == "gradient":
                    observer_blocks = _local_blocks(
                        grad, group["cell"]
                    ).double()
                    observer_norms = torch.linalg.vector_norm(
                        observer_blocks, dim=1
                    )
                    observer_unit = observer_blocks / torch.clamp(
                        observer_norms[:, None], min=1e-300
                    )
                else:
                    observer_unit = request_unit
                signature = state.get("wolf_signature")
                if signature is None:
                    reference = observer_unit
                else:
                    reference = _local_blocks(
                        signature, group["cell"]
                    ).double()
                    reference = reference / torch.clamp(
                        torch.linalg.vector_norm(reference, dim=1)[:, None],
                        min=1e-300,
                    )
                cosine = torch.sum(
                    request_unit * reference, dim=1
                ).clamp(-1.0, 1.0)
                turns = 0.5 * (1.0 - cosine)
                departure = request_unit - reference
                departure_norm2 = torch.sum(departure.square(), dim=1)
                projection_scale = torch.sum(
                    blocks * departure, dim=1
                ) / torch.clamp(departure_norm2, min=1e-300)
                candidate_blocks = blocks - (
                    group["alpha"] * turns * projection_scale
                )[:, None] * departure
                candidate_blocks = torch.where(
                    valid[:, None], candidate_blocks, blocks
                )

                next_signature = _project_unit_ball(observer_unit + reference)
                state["wolf_signature"] = next_signature.to(
                    request.dtype
                ).reshape_as(request)
                candidate = candidate_blocks.to(request.dtype).reshape_as(request)
                candidate_norms = torch.linalg.vector_norm(candidate_blocks, dim=1)
                rotated_blocks = candidate_blocks * (
                    norms / torch.clamp(candidate_norms, min=1e-300)
                )[:, None]
                rotated_blocks = torch.where(
                    valid[:, None], rotated_blocks, blocks
                )
                rotated = rotated_blocks.to(request.dtype).reshape_as(request)
                entries.append((group, parameter, request, candidate, rotated))
                raw_norm2 += float(torch.sum(request.double().square()).item())
                shape_norm2 += float(torch.sum(candidate.double().square()).item())
                turn_sum += float(torch.sum(turns).item())
                cell_count += int(turns.numel())
                fallback_count += int(torch.count_nonzero(~mask).item())
                value_count += parameter.numel()

        shape_ratio = math.sqrt(shape_norm2 / max(raw_norm2, 1e-300))
        actual_norm2 = 0.0
        for group, parameter, request, candidate, rotated in entries:
            if group["mode"] == "shape":
                applied = candidate
            elif group["mode"] == "scalar":
                applied = request * shape_ratio
            elif group["mode"] == "rotate":
                applied = rotated
            else:
                applied = request
            actual_norm2 += float(torch.sum(applied.double().square()).item())
            parameter.add_(applied, alpha=-group["lr"])

        self.last_ratio = math.sqrt(actual_norm2 / max(raw_norm2, 1e-300))
        self.last_turn = turn_sum / max(cell_count, 1)
        self.last_fallback_fraction = fallback_count / max(value_count, 1)
        if self.last_ratio > 1.0 + 2e-6:
            raise RuntimeError("transport-amplified Wolf request")
        return loss
