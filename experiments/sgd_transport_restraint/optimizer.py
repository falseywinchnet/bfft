"""Restraint-only SGD experiments derived from BFFT transport logic.

``SignatureTransportSGD`` is the live arm: it carries the bounded projected
state from the reduced Split-Bregman recursion and uses that state only to
shape the current gradient. ``ResponseTransportSGD`` is the retained negative
control based on a same-batch finite response.

Neither optimizer uses velocity, extrapolation, gradient amplification, a
Hessian inverse, or learning-rate adaptation.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def _local_blocks(value: torch.Tensor, cell: str = "row") -> torch.Tensor:
    """Use one transport cell per output unit, as in a pointwise flux field."""
    if cell == "tensor":
        return value.reshape(1, -1)
    if value.ndim >= 2:
        return value.reshape(value.shape[0], -1)
    return value.reshape(1, -1)


def _project_unit_ball(value: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(value, dim=1)
    return value / torch.clamp(
        torch.maximum(norms, torch.ones_like(norms))[:, None], min=1e-300
    )


def _parallel_transport_blocks(
    value: torch.Tensor,
    frame_from: torch.Tensor,
    frame_to: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Minimum spherical-frame rotation, with reset at antipodal collapse."""
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


def _restrain_blocks(blocks, departure, turns, valid, state, group):
    departure_norm2 = torch.sum(departure.square(), dim=1)
    if group["gate"] == "hard":
        restraint = group["alpha"] * (
            departure_norm2 > group["eps"]
        ).to(turns.dtype)
    elif group["gate"] == "amplitude":
        restraint = group["alpha"] * torch.sqrt(turns)
    elif group["gate"] == "quartic":
        restraint = group["alpha"] * turns.square()
    elif group["gate"] == "reversal":
        previous_departure = state.get("previous_departure")
        if previous_departure is None:
            restraint = torch.zeros_like(turns)
        else:
            previous_blocks = _local_blocks(
                previous_departure, group["cell"]
            ).double()
            previous_norm2 = torch.sum(previous_blocks.square(), dim=1)
            reversal_cosine = torch.sum(
                departure * previous_blocks, dim=1
            ) / torch.sqrt(torch.clamp(
                departure_norm2 * previous_norm2, min=1e-300
            ))
            restraint = group["alpha"] * torch.clamp(
                -reversal_cosine, min=0.0, max=1.0
            )
    else:
        restraint = group["alpha"] * turns
    if group["fusion"] > 1:
        restraint = 1.0 - torch.pow(1.0 - restraint, group["fusion"])
    prior_normals = state.get("normal_history", [])
    orthonormal: list[torch.Tensor] = []
    projection = torch.zeros_like(blocks.double())
    for vector in [departure, *prior_normals[: group["normal_rank"] - 1]]:
        axis = vector.double()
        for previous_axis in orthonormal:
            axis = axis - torch.sum(
                axis * previous_axis, dim=1
            )[:, None] * previous_axis
        axis_norm = torch.linalg.vector_norm(axis, dim=1)
        axis = torch.where(
            (axis_norm > group["eps"])[:, None],
            axis / torch.clamp(axis_norm[:, None], min=1e-300),
            torch.zeros_like(axis),
        )
        orthonormal.append(axis)
        projection = projection + torch.sum(
            blocks.double() * axis, dim=1
        )[:, None] * axis
    state["normal_history"] = [
        departure.detach().clone(),
        *[value.detach().clone() for value in prior_normals],
    ][: group["normal_rank"]]
    candidate = blocks.double() - restraint[:, None] * projection
    return torch.where(valid[:, None], candidate, blocks.double())


def _apply_live_axis_restraint(
    value: torch.Tensor,
    departure: torch.Tensor,
    turns: torch.Tensor,
    group: dict,
) -> torch.Tensor:
    """Apply the current rank-one signature projector to an arbitrary state."""
    departure_norm = torch.linalg.vector_norm(departure, dim=1)
    axis = torch.where(
        (departure_norm > group["eps"])[:, None],
        departure / torch.clamp(departure_norm[:, None], min=1e-300),
        torch.zeros_like(departure),
    )
    if group["gate"] == "hard":
        restraint = group["alpha"] * (
            departure_norm > group["eps"]
        ).to(turns.dtype)
    elif group["gate"] == "amplitude":
        restraint = group["alpha"] * torch.sqrt(turns)
    elif group["gate"] == "quartic":
        restraint = group["alpha"] * turns.square()
    else:
        restraint = group["alpha"] * turns
    if group["fusion"] > 1:
        restraint = 1.0 - torch.pow(1.0 - restraint, group["fusion"])
    projection = torch.sum(value * axis, dim=1)[:, None] * axis
    return value - restraint[:, None] * projection


class SignatureTransportSGD(torch.optim.Optimizer):
    """Restraint-only SGD with a carried projected Bregman signature.

    For each local parameter cell, let ``u`` be the normalized current
    gradient request.  The observer is the reduced Split-Bregman recursion

        q_k     = project(t_k, unit_ball)
        t_{k+1} = u_{k+1} + q_k

    from ``cartoon_state_algebra.py``.  The departure ``u - q/|q|`` is the
    locally incompatible direction.  Shape mode restrains only the current
    gradient's component on that direction.  Scalar mode applies exactly the
    same total norm reduction uniformly and is therefore the damping control.

    The signature observes history but is never added to the update.  Hence
    the optimizer cannot manufacture a force or amplify the SGD gradient.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        *,
        alpha: float = 1.0,
        eps: float = 1e-24,
        fusion: int = 1,
        gate: str = "soft",
        mode: str = "shape",
        observer: str = "single",
        cell: str = "row",
        normal_rank: int = 1,
        momentum: float = 0.0,
        transport_momentum: bool = False,
        normalized_momentum: bool = False,
        project_momentum: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if not isinstance(fusion, int) or fusion < 1:
            raise ValueError("fusion must be a positive integer")
        if mode not in {"shape", "rotate", "scalar", "none"}:
            raise ValueError(
                "mode must be 'shape', 'rotate', 'scalar', or 'none'"
            )
        if gate not in {
            "soft", "quartic", "amplitude", "reversal", "hard"
        }:
            raise ValueError(
                "gate must be 'soft', 'quartic', 'amplitude', "
                "'reversal', or 'hard'"
            )
        if observer not in {"single", "capacity", "cycle2"}:
            raise ValueError(
                "observer must be 'single', 'capacity', or 'cycle2'"
            )
        if cell not in {"row", "tensor"}:
            raise ValueError("cell must be 'row' or 'tensor'")
        if not isinstance(normal_rank, int) or normal_rank < 1:
            raise ValueError("normal_rank must be a positive integer")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if transport_momentum and observer != "single":
            raise ValueError(
                "transported momentum currently requires observer='single'"
            )
        if project_momentum and momentum <= 0.0:
            raise ValueError("project_momentum requires positive momentum")
        super().__init__(
            params,
            dict(
                lr=lr, alpha=alpha, eps=eps, fusion=fusion,
                gate=gate, mode=mode, observer=observer, cell=cell,
                normal_rank=normal_rank, momentum=momentum,
                transport_momentum=transport_momentum,
                normalized_momentum=normalized_momentum,
                project_momentum=project_momentum,
            ),
        )
        self.last_turn = 0.0
        self.last_ratio = 1.0
        self.last_active_fraction = 0.0
        self._step_index = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        entries: list[
            tuple[
                dict, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, bool, torch.Tensor, torch.Tensor,
            ]
        ] = []
        raw_norm2 = 0.0
        shape_norm2 = 0.0
        turn_sum = 0.0
        cell_count = 0
        active_count = 0

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("SignatureTransportSGD requires dense gradients")
                grad = parameter.grad.detach()
                blocks = _local_blocks(grad, group["cell"])
                norms = torch.linalg.vector_norm(blocks.double(), dim=1)
                valid = norms > group["eps"]
                unit = blocks.double() / torch.clamp(norms[:, None], min=1e-300)
                state = self.state[parameter]
                if group["observer"] in {"single", "capacity"}:
                    signature = state.get("signature")
                    if signature is None:
                        reference = (
                            unit if group["observer"] == "single"
                            else torch.zeros_like(unit)
                        )
                        turns = torch.zeros_like(norms)
                        departure = torch.zeros_like(unit)
                    else:
                        reference = _local_blocks(
                            signature, group["cell"]
                        ).double()
                        if group["observer"] == "single":
                            reference = reference / torch.clamp(
                                torch.linalg.vector_norm(
                                    reference, dim=1
                                )[:, None], min=1e-300
                            )
                            cosine = torch.sum(
                                unit * reference, dim=1
                            ).clamp(-1.0, 1.0)
                            turns = 0.5 * (1.0 - cosine)
                        else:
                            load = torch.clamp(
                                torch.linalg.vector_norm(reference, dim=1),
                                min=0.0, max=1.0,
                            )
                            turns = load * torch.clamp(
                                torch.sum((unit - reference).square(), dim=1)
                                / 4.0,
                                min=0.0, max=1.0,
                            )
                        departure = unit - reference
                    next_signature = _project_unit_ball(unit + reference)
                    if (
                        group["observer"] == "single"
                        and group["normal_rank"] > 1
                        and state.get("normal_history")
                    ):
                        next_base = next_signature / torch.clamp(
                            torch.linalg.vector_norm(
                                next_signature, dim=1
                            )[:, None], min=1e-300
                        )
                        transport_denominator = 1.0 + torch.sum(
                            reference * next_base, dim=1
                        )
                        transported_history = []
                        for old_axis in state["normal_history"]:
                            coefficient = torch.sum(
                                old_axis.double() * next_base, dim=1
                            ) / torch.clamp(
                                transport_denominator, min=1e-12
                            )
                            transported = old_axis.double() - (
                                coefficient[:, None]
                                * (reference + next_base)
                            )
                            transported = torch.where(
                                (transport_denominator > 1e-12)[:, None],
                                transported,
                                torch.zeros_like(transported),
                            )
                            transported_history.append(transported)
                        state["normal_history"] = transported_history
                    state["signature"] = next_signature.to(
                        grad.dtype
                    ).reshape_as(grad)
                else:
                    phase = self._step_index & 1
                    key = f"cycle_signature_{phase}"
                    other_key = f"cycle_signature_{1 - phase}"
                    current = state.get(key)
                    if current is None:
                        current_reference = unit
                    else:
                        current_reference = _local_blocks(
                            current, group["cell"]
                        ).double()
                        current_reference = current_reference / torch.clamp(
                            torch.linalg.vector_norm(
                                current_reference, dim=1
                            )[:, None], min=1e-300
                        )
                    next_signature = _project_unit_ball(
                        unit + current_reference
                    )
                    next_reference = next_signature / torch.clamp(
                        torch.linalg.vector_norm(next_signature, dim=1)[:, None],
                        min=1e-300,
                    )
                    state[key] = next_signature.to(grad.dtype).reshape_as(grad)
                    other = state.get(other_key)
                    if other is None:
                        turns = torch.zeros_like(norms)
                        departure = torch.zeros_like(unit)
                    else:
                        other_reference = _local_blocks(
                            other, group["cell"]
                        ).double()
                        other_reference = other_reference / torch.clamp(
                            torch.linalg.vector_norm(
                                other_reference, dim=1
                            )[:, None], min=1e-300
                        )
                        cosine = torch.sum(
                            next_reference * other_reference, dim=1
                        ).clamp(-1.0, 1.0)
                        turns = 0.5 * (1.0 - cosine)
                        departure = next_reference - other_reference

                candidate_blocks = _restrain_blocks(
                    blocks, departure, turns, valid, state, group
                )
                state["previous_departure"] = departure.to(
                    grad.dtype
                ).reshape_as(grad)

                candidate = candidate_blocks.to(grad.dtype).reshape_as(grad)
                candidate_norms = torch.linalg.vector_norm(
                    candidate_blocks, dim=1
                )
                rotated_blocks = candidate_blocks * (
                    norms / torch.clamp(candidate_norms, min=1e-300)
                )[:, None]
                rotated_blocks = torch.where(
                    valid[:, None], rotated_blocks, blocks.double()
                )
                rotated = rotated_blocks.to(grad.dtype).reshape_as(grad)
                old_momentum = state.get("momentum_buffer")
                if old_momentum is None:
                    momentum_base = torch.zeros_like(grad)
                elif group["transport_momentum"]:
                    old_blocks = _local_blocks(
                        old_momentum, group["cell"]
                    ).double()
                    frame_from = reference
                    frame_to = next_signature / torch.clamp(
                        torch.linalg.vector_norm(
                            next_signature, dim=1
                        )[:, None], min=1e-300
                    )
                    momentum_base = _parallel_transport_blocks(
                        old_blocks, frame_from, frame_to, group["eps"]
                    ).to(grad.dtype).reshape_as(grad)
                else:
                    momentum_base = old_momentum
                entries.append(
                    (
                        group, parameter, grad, candidate, rotated,
                        momentum_base, old_momentum is not None,
                        departure, turns,
                    )
                )
                raw_norm2 += float(torch.sum(grad.double().square()).item())
                shape_norm2 += float(torch.sum(candidate.double().square()).item())
                turn_sum += float(torch.sum(turns).item())
                cell_count += int(turns.numel())
                active_count += int(torch.count_nonzero(turns > 1e-12).item())

        shape_ratio = math.sqrt(shape_norm2 / max(raw_norm2, 1e-300))
        actual_norm2 = 0.0
        has_momentum = False
        for (
            group, parameter, grad, candidate, rotated,
            momentum_base, had_momentum, departure, turns,
        ) in entries:
            if group["mode"] == "shape":
                effective = candidate
            elif group["mode"] == "rotate":
                effective = rotated
            elif group["mode"] == "scalar":
                effective = grad * shape_ratio
            else:
                effective = grad
            if group["momentum"] > 0.0:
                momentum_input = grad if group["project_momentum"] else effective
                if group["normalized_momentum"] and had_momentum:
                    accumulated = (
                        (1.0 - group["momentum"]) * momentum_input
                        + group["momentum"] * momentum_base
                    )
                else:
                    accumulated = (
                        momentum_input + group["momentum"] * momentum_base
                    )
                if group["project_momentum"]:
                    accumulated_blocks = _local_blocks(
                        accumulated, group["cell"]
                    ).double()
                    update = _apply_live_axis_restraint(
                        accumulated_blocks, departure, turns, group
                    ).to(grad.dtype).reshape_as(grad)
                else:
                    update = accumulated
                self.state[parameter]["momentum_buffer"] = update.clone()
                has_momentum = True
            else:
                update = effective
            actual_norm2 += float(torch.sum(update.double().square()).item())
            parameter.add_(update, alpha=-group["lr"])

        ratio = math.sqrt(actual_norm2 / max(raw_norm2, 1e-300)) if entries else 1.0
        if not has_momentum and ratio > 1.0 + 2e-6:
            raise RuntimeError(f"signature restraint amplified the gradient: {ratio}")
        self.last_turn = turn_sum / max(cell_count, 1)
        self.last_ratio = ratio
        self.last_active_fraction = active_count / max(cell_count, 1)
        self._step_index += 1
        return loss


class ResponseTransportSGD(torch.optim.Optimizer):
    """SGD with an optional same-batch consecutive-state response restraint.

    ``mode="shape"`` removes a fraction of the projection of the current
    gradient onto the observed response ``y``. The gate is the local
    normalized request motion ``||y|| / ||g_current||``, matching the target-
    motion signature in the Meyer cycle. ``mode="scalar"`` is norm-matched.

    Call ``step(observe=False)`` for the first step on a minibatch and
    ``step(observe=True)`` for the second step on that same minibatch.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        *,
        alpha: float = 1.0,
        mode: str = "shape",
        eps: float = 1e-24,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if mode not in {"shape", "scalar", "none"}:
            raise ValueError("mode must be 'shape', 'scalar', or 'none'")
        super().__init__(params, dict(lr=lr, alpha=alpha, mode=mode, eps=eps))
        self.last_coherence = 0.0
        self.last_ratio = 1.0
        self.last_response_norm = 0.0
        self.last_active_fraction = 0.0

    @torch.no_grad()
    def step(self, closure=None, *, observe: bool = False):
        loss = closure() if closure is not None else None

        entries: list[
            tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("ResponseTransportSGD requires dense gradients")

                grad = parameter.grad.detach()
                state = self.state[parameter]
                previous_parameter = state.get("previous_parameter")
                previous_grad = state.get("previous_grad")

                if observe and previous_parameter is not None and previous_grad is not None:
                    displacement = parameter.detach() - previous_parameter
                    response = grad - previous_grad
                else:
                    displacement = torch.zeros_like(parameter)
                    response = torch.zeros_like(grad)
                entries.append((group, parameter, grad, displacement, response))

        shape_candidates: list[torch.Tensor] = []
        effective_norm2 = 0.0
        raw_norm2 = 0.0
        response_norm2 = 0.0
        coherence_sum = 0.0
        block_count = 0
        active_count = 0
        for group, parameter, grad, displacement, response in entries:
            alpha = group["alpha"]
            if grad.ndim >= 2:
                grad_blocks = grad.reshape(grad.shape[0], -1)
                step_blocks = displacement.reshape(displacement.shape[0], -1)
                response_blocks = response.reshape(response.shape[0], -1)
            else:
                grad_blocks = grad.reshape(1, -1)
                step_blocks = displacement.reshape(1, -1)
                response_blocks = response.reshape(1, -1)

            response_block_norm2 = torch.sum(response_blocks.double().square(), dim=1)
            grad_block_norm2 = torch.sum(grad_blocks.double().square(), dim=1)
            coherence = torch.clamp(
                torch.sqrt(response_block_norm2)
                / torch.sqrt(torch.clamp(grad_block_norm2, min=1e-300)),
                min=0.0,
                max=1.0,
            )
            grad_response = torch.sum(
                grad_blocks.double() * response_blocks.double(), dim=1
            )
            projection_scale = grad_response / torch.clamp(response_block_norm2, min=1e-300)
            candidate_blocks = grad_blocks.double() - (
                alpha * coherence * projection_scale
            )[:, None] * response_blocks.double()
            candidate = candidate_blocks.to(grad.dtype).reshape_as(grad)
            shape_candidates.append(candidate)

            raw_norm2 += float(torch.sum(grad.double().square()).item())
            effective_norm2 += float(torch.sum(candidate.double().square()).item())
            response_norm2 += float(torch.sum(response.double().square()).item())
            coherence_sum += float(torch.sum(coherence).item())
            block_count += int(coherence.numel())
            active_count += int(torch.count_nonzero(coherence > 0.0).item())

        shape_ratio = (
            math.sqrt(effective_norm2 / max(raw_norm2, 1e-300)) if entries else 1.0
        )

        actual_norm2 = 0.0
        for (group, parameter, grad, _, _), candidate in zip(entries, shape_candidates):
            mode = group["mode"]
            if mode == "shape":
                effective = candidate
            elif mode == "scalar":
                effective = grad * shape_ratio
            else:
                effective = grad
            actual_norm2 += float(torch.sum(effective.double().square()).item())
            # Store the pre-update state. On the next same-batch step this
            # makes displacement exactly theta_1 - theta_0.
            state = self.state[parameter]
            state["previous_parameter"] = parameter.detach().clone()
            state["previous_grad"] = grad.clone()
            parameter.add_(effective, alpha=-group["lr"])

        ratio = math.sqrt(actual_norm2 / max(raw_norm2, 1e-300)) if entries else 1.0
        if ratio > 1.0 + 2e-6:
            raise RuntimeError(f"response restraint amplified the gradient: {ratio}")

        self.last_coherence = coherence_sum / max(block_count, 1)
        self.last_ratio = ratio
        self.last_response_norm = math.sqrt(max(response_norm2, 0.0))
        self.last_active_fraction = active_count / max(block_count, 1)
        return loss
