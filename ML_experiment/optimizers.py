"""Small CPU-compatible Muon + AdamW optimizer used by the experiments.

The implementation follows ``torch.optim.Muon``.  Muon is applied only to
hidden two-dimensional weights; vectors, biases, input embeddings, and output
heads remain on AdamW.
"""
from __future__ import annotations

import math

import torch

from ML_experiment.anchor import Anchor, RestrainedMomentumAnchor, _parallel_transport
from ML_experiment.optimizer import ResidualPolarTransport


def zeropower_newton_schulz5(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Approximately replace ``matrix`` by its semi-orthogonal polar factor."""
    if matrix.ndim != 2:
        raise ValueError("Muon supports only 2-D gradients")
    a, b, c = 3.4445, -4.7750, 2.0315
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.float().T if transposed else matrix.float()
    x = x / x.norm().clamp_min(eps)
    for _ in range(steps):
        gram = x @ x.T
        update = b * gram + c * (gram @ gram)
        x = a * x + update @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


def bregman_soft_polar(matrix: torch.Tensor, epsilon: float = 0.05):
    """Gradient of a smoothed nuclear norm, the dual map used by Lepton.

    If ``matrix = U diag(s) V^T``, the returned singular values are
    ``s / sqrt(s^2 + epsilon^2)``.  Unlike Muon's hard polar factor, weak
    singular modes remain weak.  The map is ``grad h*`` for
    ``h*(Y) = sum_i sqrt(sigma_i(Y)^2 + epsilon^2)``.
    """
    if matrix.ndim != 2:
        raise ValueError("Lepton supports only 2-D gradients")
    if epsilon <= 0:
        raise ValueError("Lepton epsilon must be positive")
    source_dtype = matrix.dtype
    left, singular, right = torch.linalg.svd(matrix.float(), full_matrices=False)
    mapped = singular / torch.sqrt(singular.square() + float(epsilon) ** 2)
    return ((left * mapped.unsqueeze(0)) @ right).to(source_dtype)


def _transport_matrix_request(value, frame_from, frame_to, eps=1e-12):
    """Apply Anchor's minimum spherical rotation to one complete matrix."""
    shape = value.shape
    transported = _parallel_transport(
        value.reshape(1, -1),
        frame_from.reshape(1, -1),
        frame_to.reshape(1, -1),
        eps,
    )
    return transported.reshape(shape)


def _split_hidden_matrix_parameters(model):
    matrix_parameters, auxiliary_parameters = [], []
    for name, parameter in model.named_parameters():
        is_edge = name.startswith(("embed.", "encode.", "output.", "decode."))
        if parameter.ndim == 2 and not is_edge:
            matrix_parameters.append(parameter)
        else:
            auxiliary_parameters.append(parameter)
    if not matrix_parameters or not auxiliary_parameters:
        raise ValueError(
            "hybrid matrix optimizer requires hidden matrices and auxiliary parameters"
        )
    return matrix_parameters, auxiliary_parameters


class RidgeAdamW(torch.optim.Optimizer):
    """AdamW with an isotropic ridge under each tensor's diagonal metric.

    Ordinary Adam divides every coordinate by its own RMS.  On the first
    noiseless step this turns a dense matrix gradient into its elementwise
    sign, which is not equivariant to a rotation of the matrix coordinates.
    We instead use

        sqrt(v_hat + ridge * mean(v_hat))

    as the denominator.  The added scalar is rotation invariant and prevents
    quiet coordinates from receiving an arbitrarily large relative gain.  A
    zero ridge is exactly AdamW; this is deliberately not a matrix inverse,
    polar map, SVD, or Shampoo factorization.
    """

    def __init__(self, params, lr=3e-3, betas=(.9, .999), eps=1e-8,
                 weight_decay=1e-4, ridge=1.0):
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam betas must lie in [0, 1)")
        if ridge < 0:
            raise ValueError("ridge must be nonnegative")
        defaults = dict(
            lr=float(lr), betas=(float(beta1), float(beta2)), eps=float(eps),
            weight_decay=float(weight_decay), ridge=float(ridge),
        )
        super().__init__(params, defaults)
        self.last_denominator_cv = 0.0
        self.last_ridge_fraction = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        cvs, fractions = [], []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("RidgeAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                mean = state["exp_avg"]
                square = state["exp_avg_sq"]
                mean.lerp_(gradient, 1 - beta1)
                square.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
                corrected_square = square / bias2
                isotropic = corrected_square.mean()
                ridge_term = group["ridge"] * isotropic
                denominator = (corrected_square + ridge_term).sqrt().add_(group["eps"])
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.addcdiv_(
                    mean, denominator, value=-group["lr"] / bias1
                )

                rms = corrected_square.sqrt()
                rms_mean = rms.mean().clamp_min(group["eps"])
                cvs.append(float(rms.std(unbiased=False) / rms_mean))
                fractions.append(float(
                    ridge_term / (corrected_square.mean() + ridge_term + group["eps"])
                ))
        self.last_denominator_cv = sum(cvs) / len(cvs) if cvs else 0.0
        self.last_ridge_fraction = sum(fractions) / len(fractions) if fractions else 0.0
        return loss


class TurnAdamW(torch.optim.Optimizer):
    """AdamW with a tensorwise reversal brake on the applied step size.

    Adam's moment equations are unchanged.  Each tensor additionally stores
    the previous normalized raw gradient and one scalar ``lr_scale``.  A
    negative consecutive-gradient cosine is evidence that the last adaptive
    request crossed a local valley, so the scale contracts exponentially.
    Coherent gradients recover the declared ceiling slowly.  The rule uses no
    loss value, line search, matrix factorization, or task-specific geometry.
    """

    def __init__(self, params, lr=0.1, betas=(.9, .999), eps=1e-8,
                 weight_decay=1e-4, turn_brake=1.5, recovery=.02,
                 min_scale=.01, initial_scale=1.0,
                 coherence_threshold=None, coherent_growth=1.03,
                 incoherent_decay=1.0, target_from_coherence=False,
                 coherence_ema_beta=.9, binary_coherence_target=False,
                 hold_steps=0, gradient_trust=None,
                 gradient_trust_max=None, gradient_trust_threshold=.3,
                 global_gradient_trust=False):
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam betas must lie in [0, 1)")
        if (turn_brake <= 0 or not 0 < recovery <= 1
                or not 0 < min_scale <= initial_scale <= 1
                or coherent_growth <= 1 or not 0 < incoherent_decay <= 1
                or not 0 <= coherence_ema_beta < 1 or hold_steps < 0):
            raise ValueError("invalid turn-brake configuration")
        if coherence_threshold is not None and not -1 <= coherence_threshold <= 1:
            raise ValueError("coherence_threshold must lie in [-1, 1]")
        if gradient_trust is not None and gradient_trust <= 0:
            raise ValueError("gradient_trust must be positive")
        if gradient_trust_max is not None and (
            gradient_trust is None or gradient_trust_max < gradient_trust
        ):
            raise ValueError("gradient_trust_max requires a smaller base trust")
        defaults = dict(
            lr=float(lr), betas=(float(beta1), float(beta2)), eps=float(eps),
            weight_decay=float(weight_decay), turn_brake=float(turn_brake),
            recovery=float(recovery), min_scale=float(min_scale),
            initial_scale=float(initial_scale),
            coherence_threshold=(
                None if coherence_threshold is None else float(coherence_threshold)
            ),
            coherent_growth=float(coherent_growth),
            incoherent_decay=float(incoherent_decay),
            target_from_coherence=bool(target_from_coherence),
            coherence_ema_beta=float(coherence_ema_beta),
            binary_coherence_target=bool(binary_coherence_target),
            hold_steps=int(hold_steps),
            gradient_trust=(
                None if gradient_trust is None else float(gradient_trust)
            ),
            gradient_trust_max=(
                None if gradient_trust_max is None else float(gradient_trust_max)
            ),
            gradient_trust_threshold=float(gradient_trust_threshold),
            global_gradient_trust=bool(global_gradient_trust),
        )
        super().__init__(params, defaults)
        self.last_turn = 0.0
        self.last_lr_scale = 1.0
        self.last_update_gradient_ratio = 0.0
        self.last_certificate_scale = 1.0
        self.global_coherence_ema = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        turns, scales, ratios, certificates = [], [], [], []
        observed_cosines = []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("TurnAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["lr_scale"] = group["initial_scale"]
                    state["coherence_ema"] = 0.0
                norm = torch.linalg.vector_norm(gradient)
                if float(norm) > group["eps"]:
                    signature = gradient / norm
                    if "gradient_signature" in state:
                        cosine = float(torch.sum(
                            state["gradient_signature"] * signature
                        ).clamp(-1.0, 1.0))
                        turns.append((1.0 - cosine) * .5)
                        observed_cosines.append(cosine)
                        if (group["target_from_coherence"]
                                or group["gradient_trust_max"] is not None):
                            ema_beta = group["coherence_ema_beta"]
                            state["coherence_ema"] = (
                                ema_beta * state["coherence_ema"]
                                + (1 - ema_beta) * cosine
                            )
                        if state["step"] < group["hold_steps"]:
                            state["lr_scale"] = group["initial_scale"]
                        elif group["target_from_coherence"]:
                            threshold = group["coherence_threshold"]
                            if group["binary_coherence_target"]:
                                evidence = float(
                                    state["coherence_ema"] >= threshold
                                )
                            else:
                                evidence = max(
                                    0.0,
                                    min(
                                        1.0,
                                        (state["coherence_ema"] - threshold)
                                        / max(1.0 - threshold, group["eps"]),
                                    ),
                                )
                            target = (
                                group["initial_scale"]
                                + (1.0 - group["initial_scale"]) * evidence
                            )
                            if cosine < 0:
                                target = min(
                                    target,
                                    state["lr_scale"] * math.exp(
                                        group["turn_brake"] * cosine
                                    ),
                                )
                            state["lr_scale"] += group["recovery"] * (
                                target - state["lr_scale"]
                            )
                            state["lr_scale"] = max(
                                group["min_scale"], state["lr_scale"]
                            )
                        elif cosine < 0:
                            state["lr_scale"] = max(
                                group["min_scale"],
                                state["lr_scale"] * math.exp(
                                    group["turn_brake"] * cosine
                                ),
                            )
                        elif group["coherence_threshold"] is None:
                            state["lr_scale"] += group["recovery"] * (
                                1.0 - state["lr_scale"]
                            )
                        elif cosine >= group["coherence_threshold"]:
                            state["lr_scale"] = min(
                                1.0,
                                state["lr_scale"] * group["coherent_growth"],
                            )
                        else:
                            state["lr_scale"] = max(
                                group["min_scale"],
                                state["lr_scale"] * group["incoherent_decay"],
                            )
                        state["gradient_signature"].copy_(signature)
                    else:
                        state["gradient_signature"] = signature.clone()

                state["step"] += 1
                mean = state["exp_avg"]
                square = state["exp_avg_sq"]
                mean.lerp_(gradient, 1 - beta1)
                square.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
                denominator = (square / bias2).sqrt().add_(group["eps"])
                effective_lr = group["lr"] * state["lr_scale"]
                request = mean / denominator
                raw_update_norm = (
                    effective_lr * torch.linalg.vector_norm(request) / bias1
                )
                certificate = 1.0
                if group["gradient_trust"] is not None and float(raw_update_norm) > group["eps"]:
                    trust = group["gradient_trust"]
                    if group["gradient_trust_max"] is not None:
                        threshold = group["gradient_trust_threshold"]
                        coherence = (
                            self.global_coherence_ema
                            if group["global_gradient_trust"]
                            else state["coherence_ema"]
                        )
                        evidence = max(
                            0.0,
                            min(
                                1.0,
                                (coherence - threshold)
                                / max(1.0 - threshold, group["eps"]),
                            ),
                        )
                        trust += (
                            group["gradient_trust_max"] - trust
                        ) * evidence
                    certificate = min(
                        1.0,
                        float(
                            trust * norm
                            / raw_update_norm.clamp_min(group["eps"])
                        ),
                    )
                parameter.mul_(1 - effective_lr * group["weight_decay"])
                parameter.add_(request, alpha=-effective_lr * certificate / bias1)
                scales.append(state["lr_scale"])
                applied_norm = raw_update_norm * certificate
                ratios.append(float(applied_norm / norm.clamp_min(group["eps"])))
                certificates.append(certificate)
        self.last_turn = sum(turns) / len(turns) if turns else 0.0
        self.last_lr_scale = sum(scales) / len(scales) if scales else 1.0
        self.last_update_gradient_ratio = (
            sum(ratios) / len(ratios) if ratios else 0.0
        )
        self.last_certificate_scale = (
            sum(certificates) / len(certificates) if certificates else 1.0
        )
        if observed_cosines:
            observed = sum(observed_cosines) / len(observed_cosines)
            self.global_coherence_ema = (
                .9 * self.global_coherence_ema + .1 * observed
            )
        return loss


class EuclideanTransport(torch.optim.Optimizer):
    """Rotation-equivariant adaptive descent with transported momentum.

    Euclidean tangent spaces share the identity connection. The changing
    object here is instead each block's normalized gradient frame. Previous
    momentum is moved by the unique minimum rotation taking the old frame to
    the live frame before the momentum recurrence. Scale adaptation is one
    scalar RMS per block, never one denominator per coordinate::

        q_t = g_t / sqrt(E[mean(g_t^2)])
        m_t = beta1 * T(m_{t-1}) + (1-beta1) * q_t
        theta_t = (1-lr*wd) theta_{t-1} - lr * mhat_t

    Matrix rows are blocks by default; vectors and scalars are one block. A
    component of transported memory transverse to the live gradient is not
    identified by the two observed frames, so it is retained only in
    proportion to positive consecutive-frame coherence. The axial component
    is transported exactly. The resulting request is finally projected onto
    the current Euclidean descent half-space.
    """

    def __init__(self, params, lr=3e-3, betas=(.9, .999), eps=1e-8,
                 weight_decay=1e-4, cell="row", transport_eps=1e-7,
                 restrain_transverse=True):
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("moment betas must lie in [0, 1)")
        if cell not in {"row", "tensor"}:
            raise ValueError("cell must be 'row' or 'tensor'")
        if lr <= 0 or eps <= 0 or transport_eps <= 0 or weight_decay < 0:
            raise ValueError("invalid EuclideanTransport configuration")
        defaults = dict(
            lr=float(lr), betas=(float(beta1), float(beta2)), eps=float(eps),
            weight_decay=float(weight_decay), cell=cell,
            transport_eps=float(transport_eps),
            restrain_transverse=bool(restrain_transverse),
        )
        super().__init__(params, defaults)
        self.last_turn = 0.0
        self.last_transport_ratio = 1.0
        self.last_history_alignment = 1.0
        self.last_transverse_retention = 1.0
        self.last_update_gradient_ratio = 0.0

    @staticmethod
    def _blocks(value, cell):
        if cell == "row" and value.ndim >= 2:
            return value.reshape(value.shape[0], -1)
        return value.reshape(1, -1)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        turns, transport_ratios, alignments = [], [], []
        transverse_retentions, update_ratios = [], []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("EuclideanTransport requires dense gradients")
                blocks = self._blocks(gradient.detach(), group["cell"])
                cells = blocks.shape[0]
                norms = torch.linalg.vector_norm(blocks, dim=1)
                live = norms > group["transport_eps"]
                signature = blocks / norms.clamp_min(group["transport_eps"])[:, None]

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(blocks)
                    state["mean_square"] = torch.zeros(
                        cells, device=blocks.device, dtype=blocks.dtype
                    )
                momentum = state["momentum"]
                previous = state.get("signature")
                if previous is not None:
                    previous_live = torch.linalg.vector_norm(previous, dim=1) > group["transport_eps"]
                    comparable = live & previous_live
                    cosine = torch.sum(previous * signature, dim=1).clamp(-1.0, 1.0)
                    movable = comparable & ((1.0 + cosine) > group["transport_eps"])
                    if bool(torch.any(movable)):
                        before = torch.linalg.vector_norm(momentum[movable], dim=1)
                        moved = _parallel_transport(
                            momentum[movable], previous[movable],
                            signature[movable], group["transport_eps"],
                        )
                        if group["restrain_transverse"]:
                            axial = torch.sum(
                                moved * signature[movable], dim=1
                            )[:, None] * signature[movable]
                            transverse = moved - axial
                            retention = cosine[movable].clamp(0.0, 1.0)
                            moved = axial + retention[:, None] * transverse
                            transverse_retentions.extend(retention.tolist())
                        momentum[movable] = moved
                        after = torch.linalg.vector_norm(moved, dim=1)
                        valid = before > group["transport_eps"]
                        if bool(torch.any(valid)):
                            transport_ratios.extend((after[valid] / before[valid]).tolist())
                    reset = comparable & ~((1.0 + cosine) > group["transport_eps"])
                    momentum[reset] = 0
                    turns.extend((.5 * (1.0 - cosine[comparable])).tolist())
                    previous[live] = signature[live]
                else:
                    state["signature"] = signature.clone()

                state["step"] += 1
                mean_square = state["mean_square"]
                instantaneous_square = blocks.square().mean(dim=1)
                mean_square.lerp_(instantaneous_square, 1.0 - beta2)
                bias2 = 1.0 - beta2 ** state["step"]
                denominator = (mean_square / bias2).sqrt().add_(group["eps"])
                normalized = blocks / denominator[:, None]
                momentum.lerp_(normalized, 1.0 - beta1)
                bias1 = 1.0 - beta1 ** state["step"]
                request = momentum / bias1

                inner = torch.sum(request * blocks, dim=1)
                reversed_request = inner < 0
                if bool(torch.any(reversed_request)):
                    request[reversed_request] -= (
                        inner[reversed_request]
                        / norms[reversed_request].square().clamp_min(group["eps"])
                    )[:, None] * blocks[reversed_request]
                request_norm = torch.linalg.vector_norm(request, dim=1)
                valid = live & (request_norm > group["transport_eps"])
                if bool(torch.any(valid)):
                    alignments.extend((
                        torch.sum(request[valid] * signature[valid], dim=1)
                        / request_norm[valid]
                    ).tolist())
                    update_ratios.extend((
                        group["lr"] * request_norm[valid]
                        / norms[valid].clamp_min(group["transport_eps"])
                    ).tolist())

                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(request.reshape_as(parameter), alpha=-group["lr"])

        self.last_turn = sum(turns) / len(turns) if turns else 0.0
        self.last_transport_ratio = (
            sum(transport_ratios) / len(transport_ratios)
            if transport_ratios else 1.0
        )
        self.last_history_alignment = (
            sum(alignments) / len(alignments) if alignments else 1.0
        )
        self.last_transverse_retention = (
            sum(transverse_retentions) / len(transverse_retentions)
            if transverse_retentions else 1.0
        )
        self.last_update_gradient_ratio = (
            sum(update_ratios) / len(update_ratios) if update_ratios else 0.0
        )
        return loss


class MatrixTransport(torch.optim.Optimizer):
    """Transported momentum under a non-elementwise Euclidean matrix metric.

    For a matrix gradient ``G`` the second moment is represented by the two
    coordinate-free covariance factors

        L = E[G G^T / columns],   R = E[G^T G / rows].

    The live request ``L^-1/4 G R^-1/4`` is the inverse square-root action of
    their Kronecker product. It is equivariant to independent orthogonal
    changes of row and column coordinates. Momentum is accumulated only after
    this whitening, and is minimum-rotation transported between consecutive
    whitened gradient frames before accumulation. The Euclidean identity is
    the fallback connection: an observed frame earns only its positive cosine
    fraction of the full rotation. The signed coherence also integrates one
    bounded scalar learning-rate gain, growing under repeated agreement and
    shrinking under contradiction. Vectors use the unique isotropic scalar
    second moment. No coordinate owns an elementwise RMS. Matrix roots are
    cached until the bias-corrected covariance has moved by five percent, with
    a hard ten-step staleness bound; this changes evaluation frequency, not
    the metric being estimated. ``longitudinal_fusion`` is an experimental
    Meyer/Hodge translation: it leaves Nesterov's transverse momentum intact
    while moving only its axial component toward the live metric request,
    admitted by positive consecutive-frame coherence. It is retained as a
    measured negative arm, not enabled by the primary optimizer factory.
    """

    def __init__(self, params, lr=3e-3, betas=(.9, .999), eps=1e-8,
                 matrix_ridge=1e-4, weight_decay=1e-4,
                 transport_eps=1e-7, restrain_transverse=True,
                 gain_rate=.02, min_gain=.25, max_gain=4.0,
                 target_coherence=0.0, root_drift_threshold=.05,
                 max_root_staleness=10, gauge_transport=False,
                 nesterov=False, adaptive_nesterov=False,
                 longitudinal_fusion=False, closure_certificate=True,
                 residual_polar=False, residual_polar_beta=.9):
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("moment betas must lie in [0, 1)")
        if (lr <= 0 or eps <= 0 or matrix_ridge <= 0
                or transport_eps <= 0 or weight_decay < 0 or gain_rate < 0
                or not 0 < min_gain <= 1 <= max_gain
                or not -1 <= target_coherence <= 1
                or root_drift_threshold < 0 or max_root_staleness < 1
                or not 0 <= residual_polar_beta < 1):
            raise ValueError("invalid MatrixTransport configuration")
        gauge_mode = (
            "covector" if gauge_transport is True
            else "none" if gauge_transport is False
            else str(gauge_transport)
        )
        if gauge_mode not in {"none", "covector", "tangent"}:
            raise ValueError("gauge_transport must be none, covector, or tangent")
        super().__init__(params, dict(
            lr=float(lr), betas=(float(beta1), float(beta2)), eps=float(eps),
            matrix_ridge=float(matrix_ridge), weight_decay=float(weight_decay),
            transport_eps=float(transport_eps),
            restrain_transverse=bool(restrain_transverse),
            gain_rate=float(gain_rate), min_gain=float(min_gain),
            max_gain=float(max_gain),
            target_coherence=float(target_coherence),
            root_drift_threshold=float(root_drift_threshold),
            max_root_staleness=int(max_root_staleness),
            gauge_transport=gauge_mode,
            nesterov=bool(nesterov),
            adaptive_nesterov=bool(adaptive_nesterov),
            longitudinal_fusion=bool(longitudinal_fusion),
            closure_certificate=bool(closure_certificate),
            residual_polar=bool(residual_polar),
            residual_polar_beta=float(residual_polar_beta),
        ))
        self.last_turn = 0.0
        self.last_transport_ratio = 1.0
        self.last_history_alignment = 1.0
        self.last_transverse_retention = 1.0
        self.last_update_gradient_ratio = 0.0
        self.last_metric_condition = 1.0
        self.last_lr_scale = 1.0
        self.last_root_refresh_fraction = 1.0
        self.last_root_staleness = 0.0
        self.last_gauge_transport_ratio = 1.0
        self.last_gauge_transport_deformation = 0.0
        self.last_gauge_turn_removed = 0.0
        self.last_gauge_event_count = 0.0
        self.last_longitudinal_closure = 0.0
        self.last_longitudinal_certificate = 0.0
        self.last_transverse_momentum_ratio = 0.0
        self.last_residual_entropy_rank = 1.0
        self.last_polar_weight = 0.0
        self._gauge_norm_events = 0
        self._gauge_ratio_sum = 0.0
        self._gauge_deformation_sum = 0.0
        self._gauge_turn_events = 0
        self._gauge_turn_removed_sum = 0.0
        self._lr_gain = 1.0
        self._polar_weight = 0.0

    def set_residual_entropy_rank(self, normalized_rank):
        """Supply the normalized entropy rank of the unresolved operator.

        Rank one is isotropic across the available operator dimension; low
        rank is anisotropic.  The polar share is the complementary anisotropy,
        low-pass filtered to prevent minibatch spectra from flipping geometry.
        The observation is set by the training closure and never reads a
        holdout loss or a hand-selected iteration number.
        """
        rank = max(0.0, min(1.0, float(normalized_rank)))
        beta = self.param_groups[0]["residual_polar_beta"]
        target = 1.0 - rank
        self._polar_weight = (
            beta * self._polar_weight + (1.0 - beta) * target
        )
        self.last_residual_entropy_rank = rank
        self.last_polar_weight = self._polar_weight

    @staticmethod
    def _inverse_fourth_root(matrix, ridge_fraction, eps):
        eigenvalues, eigenvectors = torch.linalg.eigh(
            .5 * (matrix + matrix.transpose(0, 1))
        )
        mean = eigenvalues.clamp_min(0).mean()
        floor = ridge_fraction * mean + eps * eps
        regularized = eigenvalues.clamp_min(floor)
        inverse = regularized.pow(-.25)
        root = (eigenvectors * inverse.unsqueeze(0)) @ eigenvectors.transpose(0, 1)
        condition = float(regularized.max() / regularized.min())
        return root, condition

    @staticmethod
    def _gauge_transport(value, old_left, old_right, new_left, new_right):
        """Re-express a whitened covector under new left/right roots.

        If ``value = old_left @ raw @ old_right``, this returns
        ``new_left @ raw @ new_right`` without storing the raw covector.
        """
        raw_left = torch.linalg.solve(old_left, value)
        right_change = torch.linalg.solve(old_right, new_right)
        return new_left @ raw_left @ right_change

    @staticmethod
    def _tangent_gauge_transport(
        value, old_left, old_right, new_left, new_right
    ):
        """Re-express one physical tangent under new whitening roots.

        With ``physical = left @ coordinates @ right``, this is the unique
        coordinate change satisfying
        ``new_left @ moved @ new_right == old_left @ value @ old_right``.
        """
        left_changed = torch.linalg.solve(new_left, old_left @ value)
        right_change = torch.linalg.solve(new_right, old_right).transpose(0, 1)
        return left_changed @ right_change

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        turns, ratios, alignments, retentions, coherences = [], [], [], [], []
        closures, closure_certificates, transverse_momentum_ratios = [], [], []
        update_ratios, conditions, root_refreshes, root_ages = [], [], [], []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("MatrixTransport requires dense gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(parameter)
                    if parameter.ndim >= 2:
                        rows = parameter.shape[0]
                        columns = parameter.numel() // rows
                        state["left"] = torch.zeros(
                            rows, rows, device=parameter.device, dtype=parameter.dtype
                        )
                        state["right"] = torch.zeros(
                            columns, columns, device=parameter.device,
                            dtype=parameter.dtype,
                        )
                    else:
                        state["mean_square"] = torch.zeros(
                            (), device=parameter.device, dtype=parameter.dtype
                        )
                state["step"] += 1
                bias2 = 1.0 - beta2 ** state["step"]

                if parameter.ndim >= 2:
                    rows = parameter.shape[0]
                    matrix = gradient.reshape(rows, -1)
                    columns = matrix.shape[1]
                    left = state["left"]
                    right = state["right"]
                    left.mul_(beta2).add_(
                        matrix @ matrix.transpose(0, 1),
                        alpha=(1.0 - beta2) / columns,
                    )
                    right.mul_(beta2).add_(
                        matrix.transpose(0, 1) @ matrix,
                        alpha=(1.0 - beta2) / rows,
                    )
                    corrected_left = left / bias2
                    corrected_right = right / bias2
                    last_root_step = state.get("last_root_step", 0)
                    age = state["step"] - last_root_step
                    reference_left = state.get("root_reference_left")
                    if reference_left is None:
                        drift = math.inf
                    else:
                        left_drift = torch.linalg.vector_norm(
                            corrected_left - reference_left
                        ) / torch.linalg.vector_norm(reference_left).clamp_min(
                            group["eps"]
                        )
                        right_reference = state["root_reference_right"]
                        right_drift = torch.linalg.vector_norm(
                            corrected_right - right_reference
                        ) / torch.linalg.vector_norm(right_reference).clamp_min(
                            group["eps"]
                        )
                        drift = max(float(left_drift), float(right_drift))
                    refresh = (
                        reference_left is None
                        or age >= group["max_root_staleness"]
                        or drift >= group["root_drift_threshold"]
                    )
                    if refresh:
                        left_root, left_condition = self._inverse_fourth_root(
                            corrected_left, group["matrix_ridge"], group["eps"]
                        )
                        right_root, right_condition = self._inverse_fourth_root(
                            corrected_right, group["matrix_ridge"], group["eps"]
                        )
                        old_left_root = state.get("left_root")
                        old_right_root = state.get("right_root")
                        if (group["gauge_transport"] != "none"
                                and old_left_root is not None
                                and old_right_root is not None):
                            old_momentum = state["momentum"].reshape(
                                rows, columns
                            ).clone()
                            before = torch.linalg.vector_norm(old_momentum)
                            gauge_map = (
                                self._gauge_transport
                                if group["gauge_transport"] == "covector"
                                else self._tangent_gauge_transport
                            )
                            moved_momentum = gauge_map(
                                old_momentum, old_left_root, old_right_root,
                                left_root, right_root,
                            )
                            state["momentum"].copy_(
                                moved_momentum.reshape_as(parameter)
                            )
                            after = torch.linalg.vector_norm(moved_momentum)
                            if float(before) > group["transport_eps"]:
                                ratio = float(after / before)
                                deformation = float(
                                    torch.linalg.vector_norm(
                                        moved_momentum - old_momentum
                                    ) / before
                                )
                                self._gauge_norm_events += 1
                                self._gauge_ratio_sum += ratio
                                self._gauge_deformation_sum += deformation
                            if "signature" in state:
                                old_signature = state["signature"].reshape(
                                    rows, columns
                                ).clone()
                                moved_signature = gauge_map(
                                    old_signature, old_left_root, old_right_root,
                                    left_root, right_root,
                                ).reshape(1, -1)
                                moved_norm = torch.linalg.vector_norm(
                                    moved_signature, dim=1
                                )
                                state["signature"].copy_(
                                    moved_signature / moved_norm.clamp_min(
                                        group["transport_eps"]
                                    )[:, None]
                                )
                                state["gauge_apparent_signature"] = (
                                    old_signature.reshape(1, -1).clone()
                                )
                        state["left_root"] = left_root
                        state["right_root"] = right_root
                        state["root_reference_left"] = corrected_left.clone()
                        state["root_reference_right"] = corrected_right.clone()
                        state["root_condition"] = math.sqrt(
                            left_condition * right_condition
                        )
                        state["last_root_step"] = state["step"]
                        age = 0
                    else:
                        left_root = state["left_root"]
                        right_root = state["right_root"]
                    live_request = (
                        left_root @ matrix @ right_root
                    ).reshape_as(parameter)
                    conditions.append(state["root_condition"])
                    root_refreshes.append(float(refresh))
                    root_ages.append(float(age))
                else:
                    square = state["mean_square"]
                    square.mul_(beta2).add_(
                        gradient.square().mean(), alpha=1.0 - beta2
                    )
                    live_request = gradient / (square / bias2).sqrt().add(group["eps"])

                flat_live = live_request.reshape(1, -1)
                live_norm = torch.linalg.vector_norm(flat_live, dim=1)
                signature = flat_live / live_norm.clamp_min(group["transport_eps"])[:, None]
                momentum = state["momentum"].reshape(1, -1)
                previous = state.get("signature")
                apparent_previous = state.pop("gauge_apparent_signature", None)
                if apparent_previous is not None and previous is not None:
                    apparent_cosine = float(torch.sum(
                        apparent_previous * signature
                    ).clamp(-1.0, 1.0))
                    corrected_cosine = float(torch.sum(
                        previous * signature
                    ).clamp(-1.0, 1.0))
                    self._gauge_turn_events += 1
                    self._gauge_turn_removed_sum += .5 * (
                        corrected_cosine - apparent_cosine
                    )
                if previous is not None and float(live_norm) > group["transport_eps"]:
                    cosine = torch.sum(previous * signature, dim=1).clamp(-1.0, 1.0)
                    state["frame_confidence"] = float(cosine.clamp(0.0, 1.0))
                    turns.append(float(.5 * (1.0 - cosine)))
                    coherences.append(float(cosine))
                    if float(1.0 + cosine) > group["transport_eps"]:
                        before = torch.linalg.vector_norm(momentum)
                        fully_moved = _parallel_transport(
                            momentum, previous, signature, group["transport_eps"]
                        )
                        if group["restrain_transverse"]:
                            axial = torch.sum(
                                fully_moved * signature, dim=1
                            )[:, None] * signature
                            transverse = fully_moved - axial
                            retention = cosine.clamp(0.0, 1.0)
                            fully_moved = axial + retention[:, None] * transverse
                            retentions.append(float(retention))
                        # Euclidean identity transport is the prior. The frame
                        # observation earns only its positive cosine fraction
                        # of the full minimum rotation; contradictory frames
                        # do not get to rewrite momentum.
                        confidence = cosine.clamp(0.0, 1.0)
                        moved = (
                            (1.0 - confidence)[:, None] * momentum
                            + confidence[:, None] * fully_moved
                        )
                        momentum.copy_(moved)
                        after = torch.linalg.vector_norm(momentum)
                        if float(before) > group["transport_eps"]:
                            ratios.append(float(after / before))
                    previous.copy_(signature)
                else:
                    state["signature"] = signature.clone()
                    state["frame_confidence"] = 1.0

                momentum.lerp_(flat_live, 1.0 - beta1)
                averaged_request = momentum.reshape_as(parameter) / (
                    1.0 - beta1 ** state["step"]
                )
                if group["longitudinal_fusion"]:
                    # Rank-one Euclidean analogue of the Meyer/Hodge drop.
                    # The live metric request proposes closure of the
                    # identifiable longitudinal inconsistency. Consecutive
                    # frame coherence is the one-sided admission certificate:
                    # a stable frame earns the full closure, while a moving or
                    # contradictory frame falls back to Nesterov. Transverse
                    # (route-predicting) momentum is unchanged by the drop.
                    averaged_flat = averaged_request.reshape(1, -1)
                    axial_history = torch.sum(
                        averaged_flat * signature, dim=1
                    )[:, None] * signature
                    transverse_history = averaged_flat - axial_history
                    nesterov_request = (
                        beta1 * averaged_flat + (1.0 - beta1) * flat_live
                    )
                    certificate = (
                        state["frame_confidence"]
                        if group["closure_certificate"] else 1.0
                    )
                    axial_history_value = torch.sum(
                        averaged_flat * signature, dim=1
                    )
                    correction = (
                        certificate * beta1
                        * (live_norm - axial_history_value)[:, None]
                        * signature
                    )
                    request = (nesterov_request + correction).reshape_as(parameter)
                    closures.append(float(
                        torch.linalg.vector_norm(correction)
                        / live_norm.clamp_min(group["transport_eps"])
                    ))
                    closure_certificates.append(float(certificate))
                    transverse_momentum_ratios.append(float(
                        beta1 * torch.linalg.vector_norm(transverse_history)
                        / live_norm.clamp_min(group["transport_eps"])
                    ))
                elif group["adaptive_nesterov"]:
                    memory_weight = beta1 * state["frame_confidence"]
                    request = (
                        memory_weight * averaged_request
                        + (1.0 - memory_weight) * live_request
                    )
                elif group["nesterov"]:
                    request = (
                        beta1 * averaged_request + (1.0 - beta1) * live_request
                    )
                else:
                    request = averaged_request
                if (group["residual_polar"] and parameter.ndim >= 2
                        and self._polar_weight > 0.0):
                    # Interpolate geometry without changing the request's
                    # Frobenius norm.  Fusion supplies the magnitude and
                    # transported direction; the polar factor progressively
                    # equalizes its supported singular modes as the empirical
                    # residual operator becomes anisotropic.
                    request_matrix = request.reshape(rows, columns)
                    request_norm = torch.linalg.vector_norm(request_matrix)
                    if float(request_norm) > group["transport_eps"]:
                        polar = zeropower_newton_schulz5(request_matrix, 5)
                        polar.mul_(
                            request_norm
                            / torch.linalg.vector_norm(polar).clamp_min(
                                group["transport_eps"]
                            )
                        )
                        blended = torch.lerp(
                            request_matrix, polar, self._polar_weight
                        )
                        blended.mul_(
                            request_norm
                            / torch.linalg.vector_norm(blended).clamp_min(
                                group["transport_eps"]
                            )
                        )
                        request = blended.reshape_as(parameter)
                inner = torch.sum(request.double() * gradient.double())
                gradient_norm2 = gradient.double().square().sum()
                if float(inner) < 0.0 and float(gradient_norm2) > group["eps"]:
                    request.add_(gradient, alpha=-float(inner / gradient_norm2))
                request_norm = torch.linalg.vector_norm(request)
                gradient_norm = torch.linalg.vector_norm(gradient)
                if float(request_norm) > group["transport_eps"] and float(gradient_norm) > group["transport_eps"]:
                    alignments.append(float(
                        torch.sum(request.double() * gradient.double())
                        / (request_norm.double() * gradient_norm.double())
                    ))
                    update_ratios.append(float(
                        group["lr"] * self._lr_gain
                        * request_norm / gradient_norm
                    ))
                effective_lr = group["lr"] * self._lr_gain
                parameter.mul_(1.0 - effective_lr * group["weight_decay"])
                parameter.add_(request, alpha=-effective_lr)

        self.last_turn = sum(turns) / len(turns) if turns else 0.0
        self.last_transport_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        self.last_history_alignment = (
            sum(alignments) / len(alignments) if alignments else 1.0
        )
        self.last_transverse_retention = (
            sum(retentions) / len(retentions) if retentions else 1.0
        )
        self.last_update_gradient_ratio = (
            sum(update_ratios) / len(update_ratios) if update_ratios else 0.0
        )
        self.last_metric_condition = (
            sum(conditions) / len(conditions) if conditions else 1.0
        )
        self.last_root_refresh_fraction = (
            sum(root_refreshes) / len(root_refreshes)
            if root_refreshes else 0.0
        )
        self.last_root_staleness = (
            sum(root_ages) / len(root_ages) if root_ages else 0.0
        )
        self.last_gauge_transport_ratio = (
            self._gauge_ratio_sum / self._gauge_norm_events
            if self._gauge_norm_events else 1.0
        )
        self.last_gauge_transport_deformation = (
            self._gauge_deformation_sum / self._gauge_norm_events
            if self._gauge_norm_events else 0.0
        )
        self.last_gauge_turn_removed = (
            self._gauge_turn_removed_sum / self._gauge_turn_events
            if self._gauge_turn_events else 0.0
        )
        self.last_gauge_event_count = float(self._gauge_turn_events)
        self.last_longitudinal_closure = (
            sum(closures) / len(closures) if closures else 0.0
        )
        self.last_longitudinal_certificate = (
            sum(closure_certificates) / len(closure_certificates)
            if closure_certificates else 0.0
        )
        self.last_transverse_momentum_ratio = (
            sum(transverse_momentum_ratios) / len(transverse_momentum_ratios)
            if transverse_momentum_ratios else 0.0
        )
        if coherences:
            group = self.param_groups[0]
            mean_coherence = sum(coherences) / len(coherences)
            log_gain = math.log(self._lr_gain) + group["gain_rate"] * (
                mean_coherence - group["target_coherence"]
            )
            self._lr_gain = min(
                group["max_gain"],
                max(group["min_gain"], math.exp(log_gain)),
            )
        self.last_lr_scale = self._lr_gain
        return loss


class SourceAwareAdamW(torch.optim.Optimizer):
    """AdamW whose memory belongs to the frozen self-context chart.

    A self-context backward pass supplies two cotangents on every parameter:

    ``g_fixed``
        The ordinary learning decision with the context proposal held fixed.
    ``g_chart``
        The additional feedback caused by moving that proposal's chart.

    Ordinary AdamW adds them before either moment recurrence, allowing moving
    coordinates to write persistent first- and second-moment history. Here
    only ``g_fixed`` enters Adam's state. The instantaneous chart request is
    read through the same diagonal metric, tensorwise capped by the fixed
    request, and then projected against reversing the current fixed-chart
    decision. Decoupled weight decay remains exactly AdamW's.

    Parameters that have never had a nonzero frozen-chart contribution are
    treated as genuinely chart-only parameters and receive ordinary AdamW.
    If a fixed-chart signal later appears, their old unanchored state is reset
    before switching permanently to the source-aware rule.
    """

    def __init__(self, params, lr=3e-3, betas=(.9, .999), eps=1e-8,
                 weight_decay=1e-4, context_gain=1.0, source_eps=1e-12):
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam betas must lie in [0, 1)")
        if (lr <= 0 or eps <= 0 or weight_decay < 0
                or context_gain < 0 or source_eps <= 0):
            raise ValueError("invalid SourceAwareAdamW configuration")
        defaults = dict(
            lr=float(lr), betas=(float(beta1), float(beta2)), eps=float(eps),
            weight_decay=float(weight_decay), context_gain=float(context_gain),
            source_eps=float(source_eps),
        )
        super().__init__(params, defaults)
        self.last_context_request_scale = 1.0
        self.last_context_request_ratio = 0.0
        self.last_fixed_alignment = 1.0
        self.last_reversal_fraction = 0.0
        self.last_chart_only_fraction = 0.0

    @staticmethod
    def _initialize_state(state, parameter):
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(parameter)
        state["exp_avg_sq"] = torch.zeros_like(parameter)
        state["anchored"] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        scales, ratios, alignments = [], [], []
        reversed_cells = 0
        chart_only_cells = 0
        total_cells = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("SourceAwareAdamW requires dense gradients")
                fixed = getattr(parameter, "_fixed_chart_grad", None)
                chart = getattr(parameter, "_context_chart_grad", None)
                if fixed is None or chart is None:
                    raise RuntimeError(
                        "SourceAwareAdamW requires backward_with_context_split"
                    )

                total_cells += 1
                state = self.state[parameter]
                if not state:
                    self._initialize_state(state, parameter)

                fixed_norm = torch.linalg.vector_norm(fixed)
                fixed_is_live = float(fixed_norm) > group["source_eps"]
                if fixed_is_live and not state["anchored"]:
                    # Discard any chart-only history before this parameter
                    # acquires a frozen-chart decision.
                    state["step"] = 0
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()
                    state["anchored"] = True

                if state["anchored"]:
                    persistent_gradient = fixed
                    chart_gradient = chart if fixed_is_live else torch.zeros_like(chart)
                else:
                    # With no fixed chart there is no source conflict and no
                    # direction against which a chart request can be certified.
                    persistent_gradient = chart
                    chart_gradient = torch.zeros_like(chart)
                    chart_only_cells += 1

                state["step"] += 1
                mean = state["exp_avg"]
                square = state["exp_avg_sq"]
                mean.lerp_(persistent_gradient, 1.0 - beta1)
                square.mul_(beta2).addcmul_(
                    persistent_gradient, persistent_gradient, value=1.0 - beta2
                )
                bias1 = 1.0 - beta1 ** state["step"]
                bias2 = 1.0 - beta2 ** state["step"]
                denominator = (square / bias2).sqrt().add_(group["eps"])
                fixed_request = (mean / bias1) / denominator
                chart_request = chart_gradient / denominator

                fixed_request_norm = torch.linalg.vector_norm(fixed_request)
                chart_request_norm = torch.linalg.vector_norm(chart_request)
                ratio = float(
                    chart_request_norm
                    / fixed_request_norm.clamp_min(group["source_eps"])
                )
                scale = min(
                    1.0,
                    group["context_gain"]
                    * float(fixed_request_norm)
                    / max(float(chart_request_norm), group["source_eps"]),
                )
                request = fixed_request + scale * chart_request

                if state["anchored"] and fixed_is_live:
                    inner = torch.sum(request.double() * fixed.double())
                    if float(inner) < 0.0:
                        request.add_(
                            fixed,
                            alpha=-float(inner / fixed.double().square().sum()),
                        )
                        reversed_cells += 1
                    request_norm = torch.linalg.vector_norm(request)
                    alignment = float(
                        torch.sum(request.double() * fixed.double())
                        / (
                            request_norm.double()
                            * fixed_norm.double()
                        ).clamp_min(group["source_eps"])
                    )
                    alignments.append(alignment)

                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(request, alpha=-group["lr"])
                scales.append(scale)
                ratios.append(ratio)

        self.last_context_request_scale = (
            sum(scales) / len(scales) if scales else 1.0
        )
        self.last_context_request_ratio = (
            sum(ratios) / len(ratios) if ratios else 0.0
        )
        self.last_fixed_alignment = (
            sum(alignments) / len(alignments) if alignments else 1.0
        )
        self.last_reversal_fraction = (
            reversed_cells / total_cells if total_cells else 0.0
        )
        self.last_chart_only_fraction = (
            chart_only_cells / total_cells if total_cells else 0.0
        )
        return loss


class TransportedAdamW(torch.optim.Optimizer):
    """AdamW with transported row moments and non-elementwise covariance.

    Every matrix row is treated as one Euclidean cell. If consecutive raw
    row-gradient signatures define the minimum rotation ``R``, the previous
    first and second moments are moved into the current frame before the Adam
    recurrence::

        m <- R m
        V <- R V R^T
        m <- beta1 m + (1-beta1) g
        V <- beta2 V + (1-beta2) g g^T
        update <- (sqrt(V) + eps I)^-1 m

    Vectors form one cell. This is full-matrix Adam inside each row, not a
    matrix polar map: rows never interact, and the first deterministic step is
    row-RMS normalization rather than Muon's semi-orthogonal factor. Decoupled
    weight decay is unchanged.
    """

    def __init__(self, params, lr=3e-3, betas=(.9, .999), eps=1e-8,
                 weight_decay=1e-4, transport_eps=1e-7,
                 match_adam_rms=True):
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam betas must lie in [0, 1)")
        if lr <= 0 or eps <= 0 or transport_eps <= 0 or weight_decay < 0:
            raise ValueError("invalid TransportedAdamW configuration")
        defaults = dict(
            lr=float(lr), betas=(float(beta1), float(beta2)), eps=float(eps),
            weight_decay=float(weight_decay), transport_eps=float(transport_eps),
            match_adam_rms=bool(match_adam_rms),
        )
        super().__init__(params, defaults)
        self.last_turn = 0.0
        self.last_transport_ratio = 1.0
        self.last_update_gradient_ratio = 0.0
        self.last_covariance_rank = 0.0

    @staticmethod
    def _blocks(value):
        if value.ndim >= 2:
            return value.reshape(value.shape[0], -1)
        return value.reshape(1, -1)

    @staticmethod
    def _transport_matrices(frame_from, frame_to, eps):
        cells, dimension = frame_from.shape
        basis = torch.eye(
            dimension, device=frame_from.device, dtype=frame_from.dtype
        ).expand(cells, -1, -1)
        transported = _parallel_transport(
            basis.reshape(cells * dimension, dimension),
            frame_from[:, None, :].expand(-1, dimension, -1).reshape(
                cells * dimension, dimension
            ),
            frame_to[:, None, :].expand(-1, dimension, -1).reshape(
                cells * dimension, dimension
            ),
            eps,
        )
        # Rows are transported basis vectors, so row vectors act on the right.
        return transported.reshape(cells, dimension, dimension)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        turns, transport_ratios, update_ratios, ranks = [], [], [], []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("TransportedAdamW requires dense gradients")
                blocks = self._blocks(gradient)
                cells, dimension = blocks.shape
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["first"] = torch.zeros_like(blocks)
                    state["second"] = torch.zeros(
                        cells, dimension, dimension,
                        device=blocks.device, dtype=blocks.dtype,
                    )
                first = state["first"]
                second = state["second"]
                norms = torch.linalg.vector_norm(blocks, dim=1)
                live = norms > group["transport_eps"]
                current = blocks / norms.clamp_min(group["transport_eps"])[:, None]

                previous = state.get("signature")
                if previous is not None and bool(torch.any(live)):
                    previous_live = (
                        torch.linalg.vector_norm(previous, dim=1)
                        > group["transport_eps"]
                    )
                    cosine = torch.sum(previous * current, dim=1).clamp(-1.0, 1.0)
                    non_antipodal = (1.0 + cosine) > group["transport_eps"]
                    # A row whose earlier gradients were zero has no prior
                    # frame. Initialize it in the live frame instead of
                    # treating the zero signature as a unit direction.
                    movable = live & previous_live & non_antipodal
                    if bool(torch.any(movable)):
                        transform = self._transport_matrices(
                            previous[movable], current[movable],
                            group["transport_eps"],
                        )
                        before = torch.linalg.vector_norm(first[movable], dim=1)
                        moved_first = torch.einsum(
                            "bi,bij->bj", first[movable], transform
                        )
                        moved_second = (
                            transform.transpose(1, 2)
                            @ second[movable]
                            @ transform
                        )
                        first[movable] = moved_first
                        second[movable] = .5 * (
                            moved_second + moved_second.transpose(1, 2)
                        )
                        after = torch.linalg.vector_norm(moved_first, dim=1)
                        valid_before = before > group["transport_eps"]
                        if bool(torch.any(valid_before)):
                            transport_ratios.extend(
                                (after[valid_before] / before[valid_before]).tolist()
                            )
                    reset = live & previous_live & ~non_antipodal
                    if bool(torch.any(reset)):
                        first[reset] = 0
                        second[reset] = 0
                    comparable = live & previous_live
                    turns.extend((.5 * (1.0 - cosine[comparable])).tolist())
                    previous[live] = current[live]
                elif previous is None:
                    state["signature"] = current.clone()

                state["step"] += 1
                first.lerp_(blocks, 1.0 - beta1)
                second.mul_(beta2).add_(
                    torch.einsum("bi,bj->bij", blocks, blocks),
                    alpha=1.0 - beta2,
                )
                bias1 = 1.0 - beta1 ** state["step"]
                bias2 = 1.0 - beta2 ** state["step"]
                corrected_first = first / bias1
                corrected_second = second / bias2
                eigenvalues, eigenvectors = torch.linalg.eigh(corrected_second)
                eigenvalues = eigenvalues.clamp_min(0.0)
                coordinates = torch.einsum(
                    "bij,bj->bi", eigenvectors.transpose(1, 2), corrected_first
                )
                largest = eigenvalues[:, -1:].clamp_min(group["eps"] ** 2)
                supported = eigenvalues > torch.maximum(
                    torch.finfo(eigenvalues.dtype).eps * dimension * largest,
                    eigenvalues.new_tensor(group["eps"] ** 2),
                )
                inverse = torch.where(
                    supported,
                    1.0 / (eigenvalues.sqrt() + group["eps"]),
                    torch.zeros_like(eigenvalues),
                )
                coordinates = coordinates * inverse
                update = torch.einsum("bij,bj->bi", eigenvectors, coordinates)
                if group["match_adam_rms"]:
                    # A rank-one full-covariance request has row norm one,
                    # whereas Adam's first sign request has per-coordinate RMS
                    # one and therefore row norm sqrt(d). Match that declared
                    # learning-rate convention without inspecting gradients,
                    # loss, or task identity.
                    update.mul_(math.sqrt(dimension))

                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update.reshape_as(parameter), alpha=-group["lr"])
                update_norm = group["lr"] * torch.linalg.vector_norm(update, dim=1)
                update_ratios.extend((
                    update_norm / norms.clamp_min(group["transport_eps"])
                ).tolist())
                ranks.extend((
                    supported
                ).sum(1).float().tolist())

        self.last_turn = sum(turns) / len(turns) if turns else 0.0
        self.last_transport_ratio = (
            sum(transport_ratios) / len(transport_ratios)
            if transport_ratios else 1.0
        )
        self.last_update_gradient_ratio = (
            sum(update_ratios) / len(update_ratios) if update_ratios else 0.0
        )
        self.last_covariance_rank = sum(ranks) / len(ranks) if ranks else 0.0
        return loss


class Muon(torch.optim.Optimizer):
    """Single-process Muon for hidden matrix parameters."""

    def __init__(self, params, lr=3e-3, weight_decay=1e-4, momentum=.95,
                 nesterov=True, ns_steps=5, adjust_lr="match_rms_adamw"):
        defaults = dict(lr=float(lr), weight_decay=float(weight_decay),
                        momentum=float(momentum), nesterov=bool(nesterov),
                        ns_steps=int(ns_steps), adjust_lr=adjust_lr)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError("Muon parameter groups must contain only matrices")

    @staticmethod
    def _adjusted_lr(lr, shape, mode):
        rows, columns = shape
        if mode == "match_rms_adamw":
            return lr * .2 * math.sqrt(max(rows, columns))
        if mode == "original":
            return lr * math.sqrt(max(1.0, rows / columns))
        if mode == "spectral_unclamped":
            return lr * math.sqrt(rows / columns)
        raise ValueError(mode)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1 - group["momentum"])
                update = (gradient.lerp(buffer, group["momentum"])
                          if group["nesterov"] else buffer)
                update = zeropower_newton_schulz5(update, group["ns_steps"])
                parameter.mul_(1 - lr * group["weight_decay"])
                adjusted = self._adjusted_lr(lr, parameter.shape, group["adjust_lr"])
                parameter.add_(update, alpha=-adjusted)
        return loss


class TransportMuon(Muon):
    """Muon whose matrix momentum is transported through gradient-frame turns.

    Each parameter matrix is one geometric cell.  Before the momentum buffer
    consumes a new gradient, Anchor's minimum spherical rotation carries the
    old buffer from the previous full-matrix gradient signature to their
    spherical midpoint.  Muon's Newton--Schulz spectral decision is unchanged.
    """

    def __init__(self, params, lr=3e-3, weight_decay=1e-4, momentum=.95,
                 nesterov=True, ns_steps=5, adjust_lr="match_rms_adamw",
                 transport_eps=1e-12):
        super().__init__(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum,
            nesterov=nesterov, ns_steps=ns_steps, adjust_lr=adjust_lr,
        )
        for group in self.param_groups:
            group["transport_eps"] = float(transport_eps)
        self.last_turn = 0.0
        self.last_transport_ratio = 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        turns, ratios = [], []
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["transport_eps"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("TransportMuon does not support sparse gradients")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]

                norm = torch.linalg.vector_norm(gradient)
                if float(norm) > eps:
                    current = gradient / norm
                    if "gradient_signature" not in state:
                        state["gradient_signature"] = current.clone()
                    else:
                        previous = state["gradient_signature"]
                        cosine = torch.sum(previous * current).clamp(-1.0, 1.0)
                        midpoint = previous + current
                        midpoint_norm = torch.linalg.vector_norm(midpoint)
                        target = (midpoint / midpoint_norm
                                  if float(midpoint_norm) > eps else current)
                        before = torch.linalg.vector_norm(buffer)
                        transported = _transport_matrix_request(
                            buffer, previous, target, eps
                        )
                        after = torch.linalg.vector_norm(transported)
                        buffer.copy_(transported)
                        state["gradient_signature"].copy_(target)
                        turns.append(float((1.0 - cosine) * 0.5))
                        if float(before) > eps:
                            ratios.append(float(after / before))

                buffer.lerp_(gradient, 1 - group["momentum"])
                request = (gradient.lerp(buffer, group["momentum"])
                           if group["nesterov"] else buffer)
                update = zeropower_newton_schulz5(request, group["ns_steps"])
                parameter.mul_(1 - lr * group["weight_decay"])
                adjusted = self._adjusted_lr(
                    lr, parameter.shape, group["adjust_lr"]
                )
                parameter.add_(update, alpha=-adjusted)
        self.last_turn = sum(turns) / len(turns) if turns else 0.0
        self.last_transport_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        return loss


class Lepton(torch.optim.Optimizer):
    """Smooth Bregman continuation of Muon's hard spectral polar map.

    The momentum buffer is a discounted dual coordinate built from normalized
    matrix gradients.  ``bregman_soft_polar`` maps it to the primal request.
    As epsilon approaches zero, the map approaches exact Muon on supported
    singular modes; finite epsilon restrains weak, uncertain modes.
    """

    def __init__(self, params, lr=3e-3, weight_decay=1e-4, momentum=.95,
                 nesterov=True, bregman_epsilon=.05,
                 adjust_lr="match_rms_adamw", norm_eps=1e-7):
        defaults = dict(
            lr=float(lr), weight_decay=float(weight_decay),
            momentum=float(momentum), nesterov=bool(nesterov),
            bregman_epsilon=float(bregman_epsilon), adjust_lr=adjust_lr,
            norm_eps=float(norm_eps),
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError("Lepton parameter groups must contain only matrices")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Lepton does not support sparse gradients")
                normalized = gradient / gradient.norm().clamp_min(group["norm_eps"])
                state = self.state[parameter]
                if "dual_buffer" not in state:
                    state["dual_buffer"] = torch.zeros_like(gradient)
                buffer = state["dual_buffer"]
                buffer.lerp_(normalized, 1 - group["momentum"])
                request = (normalized.lerp(buffer, group["momentum"])
                           if group["nesterov"] else buffer)
                update = bregman_soft_polar(
                    request, group["bregman_epsilon"]
                )
                parameter.mul_(1 - lr * group["weight_decay"])
                adjusted = Muon._adjusted_lr(
                    lr, parameter.shape, group["adjust_lr"]
                )
                parameter.add_(update, alpha=-adjusted)
        return loss


class TransportLepton(Lepton):
    """Lepton with Anchor transport applied to its Bregman dual state."""

    def __init__(self, params, lr=3e-3, weight_decay=1e-4, momentum=.95,
                 nesterov=True, bregman_epsilon=.05,
                 adjust_lr="match_rms_adamw", norm_eps=1e-7,
                 transport_eps=1e-12, frame_target="midpoint",
                 gradient_trust=None):
        super().__init__(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum,
            nesterov=nesterov, bregman_epsilon=bregman_epsilon,
            adjust_lr=adjust_lr, norm_eps=norm_eps,
        )
        for group in self.param_groups:
            group["transport_eps"] = float(transport_eps)
            if frame_target not in {"midpoint", "current"}:
                raise ValueError(frame_target)
            group["frame_target"] = frame_target
            group["gradient_trust"] = (
                None if gradient_trust is None else float(gradient_trust)
            )
            if group["gradient_trust"] is not None and group["gradient_trust"] <= 0:
                raise ValueError("gradient_trust must be positive")
        self.last_turn = 0.0
        self.last_transport_ratio = 1.0
        self.last_frame_residual = 0.0
        self.last_history_alignment = 1.0
        self.last_gradient_norm = 0.0
        self.last_update_norm = 0.0
        self.last_update_gradient_ratio = 0.0
        self.last_certificate_scale = 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        turns, ratios, residuals = [], [], []
        alignments, gradient_norms, update_norms, update_gradient_ratios = [], [], [], []
        certificate_scales = []
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["transport_eps"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("TransportLepton does not support sparse gradients")
                gradient_norm = gradient.norm()
                normalized = gradient / gradient_norm.clamp_min(group["norm_eps"])
                state = self.state[parameter]
                if "dual_buffer" not in state:
                    state["dual_buffer"] = torch.zeros_like(gradient)
                buffer = state["dual_buffer"]

                if float(gradient_norm) > group["norm_eps"]:
                    current = normalized
                    if "dual_signature" not in state:
                        state["dual_signature"] = current.clone()
                    else:
                        previous = state["dual_signature"]
                        cosine = torch.sum(previous * current).clamp(-1.0, 1.0)
                        if group["frame_target"] == "current":
                            target = current
                        else:
                            midpoint = previous + current
                            midpoint_norm = torch.linalg.vector_norm(midpoint)
                            target = (midpoint / midpoint_norm
                                      if float(midpoint_norm) > eps else current)
                        before = torch.linalg.vector_norm(buffer)
                        transported = _transport_matrix_request(
                            buffer, previous, target, eps
                        )
                        after = torch.linalg.vector_norm(transported)
                        buffer.copy_(transported)
                        state["dual_signature"].copy_(target)
                        turns.append(float((1.0 - cosine) * 0.5))
                        residuals.append(float(
                            (1.0 - torch.sum(target * current).clamp(-1.0, 1.0))
                            * 0.5
                        ))
                        if float(before) > eps:
                            ratios.append(float(after / before))

                buffer_norm = torch.linalg.vector_norm(buffer)
                if float(buffer_norm) > eps and float(gradient_norm) > group["norm_eps"]:
                    alignments.append(float(
                        torch.sum((buffer / buffer_norm) * normalized).clamp(-1.0, 1.0)
                    ))

                buffer.lerp_(normalized, 1 - group["momentum"])
                request = (normalized.lerp(buffer, group["momentum"])
                           if group["nesterov"] else buffer)
                update = bregman_soft_polar(
                    request, group["bregman_epsilon"]
                )
                parameter.mul_(1 - lr * group["weight_decay"])
                adjusted = Muon._adjusted_lr(
                    lr, parameter.shape, group["adjust_lr"]
                )
                raw_update_norm = adjusted * torch.linalg.vector_norm(update)
                certificate_scale = 1.0
                if group["gradient_trust"] is not None and float(raw_update_norm) > eps:
                    certificate_scale = min(
                        1.0,
                        float(
                            group["gradient_trust"] * gradient_norm
                            / raw_update_norm.clamp_min(eps)
                        ),
                    )
                parameter.add_(update, alpha=-adjusted * certificate_scale)
                update_norm = raw_update_norm * certificate_scale
                gradient_norms.append(float(gradient_norm))
                update_norms.append(float(update_norm))
                update_gradient_ratios.append(float(
                    update_norm / gradient_norm.clamp_min(group["norm_eps"])
                ))
                certificate_scales.append(certificate_scale)
        self.last_turn = sum(turns) / len(turns) if turns else 0.0
        self.last_transport_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        self.last_frame_residual = (
            sum(residuals) / len(residuals) if residuals else 0.0
        )
        self.last_history_alignment = (
            sum(alignments) / len(alignments) if alignments else 1.0
        )
        self.last_gradient_norm = (
            sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
        )
        self.last_update_norm = (
            sum(update_norms) / len(update_norms) if update_norms else 0.0
        )
        self.last_update_gradient_ratio = (
            sum(update_gradient_ratios) / len(update_gradient_ratios)
            if update_gradient_ratios else 0.0
        )
        self.last_certificate_scale = (
            sum(certificate_scales) / len(certificate_scales)
            if certificate_scales else 1.0
        )
        return loss


class MuonWithAuxAdamW:
    """Optimizer facade combining Muon hidden matrices with AdamW auxiliaries."""

    def __init__(self, model, lr=3e-3, weight_decay=1e-4, muon_lr=None,
                 momentum=.95, adjust_lr="match_rms_adamw"):
        muon_parameters, auxiliary_parameters = _split_hidden_matrix_parameters(model)
        self.muon = Muon(muon_parameters, lr=muon_lr or lr,
                         weight_decay=weight_decay, momentum=momentum,
                         adjust_lr=adjust_lr)
        self.adamw = torch.optim.AdamW(auxiliary_parameters, lr=lr,
                                       weight_decay=weight_decay,
                                       betas=(.9, .95))

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()


class TransportMuonWithAuxAdamW:
    """Hybrid facade for transported hidden matrices and AdamW auxiliaries."""

    def __init__(self, model, lr=3e-3, weight_decay=1e-4, muon_lr=None,
                 momentum=.95, adjust_lr="match_rms_adamw"):
        matrix_parameters, auxiliary_parameters = _split_hidden_matrix_parameters(model)
        self.muon = TransportMuon(
            matrix_parameters, lr=muon_lr or lr, weight_decay=weight_decay,
            momentum=momentum, adjust_lr=adjust_lr,
        )
        self.adamw = torch.optim.AdamW(
            auxiliary_parameters, lr=lr, weight_decay=weight_decay,
            betas=(.9, .95),
        )

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()


class LeptonWithAuxAdamW:
    """Hybrid facade for Bregman hidden matrices and AdamW auxiliaries."""

    def __init__(self, model, lr=3e-3, weight_decay=1e-4, lepton_lr=None,
                 momentum=.95, bregman_epsilon=.05,
                 adjust_lr="match_rms_adamw"):
        matrix_parameters, auxiliary_parameters = _split_hidden_matrix_parameters(model)
        self.lepton = Lepton(
            matrix_parameters, lr=lepton_lr or lr, weight_decay=weight_decay,
            momentum=momentum, bregman_epsilon=bregman_epsilon,
            adjust_lr=adjust_lr,
        )
        self.adamw = torch.optim.AdamW(
            auxiliary_parameters, lr=lr, weight_decay=weight_decay,
            betas=(.9, .95),
        )

    def zero_grad(self, set_to_none=True):
        self.lepton.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.lepton.step()
        self.adamw.step()


class TransportLeptonWithAuxAdamW(LeptonWithAuxAdamW):
    """Hybrid facade for transported Bregman matrices and AdamW auxiliaries."""

    def __init__(self, model, lr=3e-3, weight_decay=1e-4, lepton_lr=None,
                 momentum=.95, bregman_epsilon=.05,
                 adjust_lr="match_rms_adamw", frame_target="midpoint",
                 gradient_trust=None):
        matrix_parameters, auxiliary_parameters = _split_hidden_matrix_parameters(model)
        self.lepton = TransportLepton(
            matrix_parameters, lr=lepton_lr or lr, weight_decay=weight_decay,
            momentum=momentum, bregman_epsilon=bregman_epsilon,
            adjust_lr=adjust_lr, frame_target=frame_target,
            gradient_trust=gradient_trust,
        )
        self.adamw = torch.optim.AdamW(
            auxiliary_parameters, lr=lr, weight_decay=weight_decay,
            betas=(.9, .95),
        )

    @property
    def last_turn(self):
        return self.lepton.last_turn

    @property
    def last_transport_ratio(self):
        return self.lepton.last_transport_ratio

    @property
    def last_frame_residual(self):
        return self.lepton.last_frame_residual

    @property
    def last_history_alignment(self):
        return self.lepton.last_history_alignment

    @property
    def last_gradient_norm(self):
        return self.lepton.last_gradient_norm

    @property
    def last_update_norm(self):
        return self.lepton.last_update_norm

    @property
    def last_update_gradient_ratio(self):
        return self.lepton.last_update_gradient_ratio

    @property
    def last_certificate_scale(self):
        return self.lepton.last_certificate_scale


class RidgeAdamWithAuxAdamW:
    """Use ridged Adam only on hidden matrices and ordinary AdamW elsewhere."""

    def __init__(self, model, lr=3e-3, weight_decay=1e-4, ridge=1.0):
        matrix_parameters, auxiliary_parameters = _split_hidden_matrix_parameters(model)
        self.matrix = RidgeAdamW(
            matrix_parameters, lr=lr, weight_decay=weight_decay,
            betas=(.9, .999), ridge=ridge,
        )
        self.adamw = torch.optim.AdamW(
            auxiliary_parameters, lr=lr, weight_decay=weight_decay,
        )

    def zero_grad(self, set_to_none=True):
        self.matrix.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.matrix.step()
        self.adamw.step()

    @property
    def last_denominator_cv(self):
        return self.matrix.last_denominator_cv

    @property
    def last_ridge_fraction(self):
        return self.matrix.last_ridge_fraction


def make_optimizer(model, name: str, lr: float, weight_decay: float = 1e-4):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "euclidean_transport":
        return EuclideanTransport(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if name == "matrix_transport":
        return MatrixTransport(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if name == "matrix_transport_gauge":
        return MatrixTransport(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            gauge_transport=True,
        )
    if name == "matrix_transport_nesterov":
        return MatrixTransport(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            nesterov=True,
        )
    if name == "matrix_transport_longitudinal_fusion":
        return MatrixTransport(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            longitudinal_fusion=True,
        )
    if name == "matrix_transport_residual_polar":
        return ResidualPolarTransport(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
    if name == "matrix_transport_longitudinal_fusion_full":
        return MatrixTransport(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            longitudinal_fusion=True, closure_certificate=False,
        )
    if name == "matrix_transport_gauge_nesterov":
        return MatrixTransport(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            gauge_transport=True, nesterov=True,
        )
    if name == "matrix_transport_tangent_nesterov":
        return MatrixTransport(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            gauge_transport="tangent", nesterov=True,
        )
    if name == "matrix_transport_adaptive_nesterov":
        return MatrixTransport(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            adaptive_nesterov=True,
        )
    if name == "matrix_transport_tangent_adaptive_nesterov":
        return MatrixTransport(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            gauge_transport="tangent", adaptive_nesterov=True,
        )
    if name == "adamw_source_aware":
        return SourceAwareAdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if name == "adamw_transport":
        return TransportedAdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if name == "adamw_ridge":
        return RidgeAdamWithAuxAdamW(
            model, lr=lr, weight_decay=weight_decay, ridge=1.0
        )
    if name == "adamw_ridge_025":
        return RidgeAdamWithAuxAdamW(
            model, lr=lr, weight_decay=weight_decay, ridge=.25
        )
    if name == "adamw_turn":
        return TurnAdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if name == "adamw_coherence":
        return TurnAdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            initial_scale=.03, coherence_threshold=.3, recovery=.2,
            target_from_coherence=True, coherence_ema_beta=.9,
        )
    if name == "adamw_coherence_latch":
        return TurnAdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            initial_scale=.03, coherence_threshold=.5, recovery=.5,
            target_from_coherence=True, coherence_ema_beta=.9,
            binary_coherence_target=True,
        )
    if name == "adamw_warm_turn":
        return TurnAdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            initial_scale=.03, hold_steps=100, recovery=.05,
        )
    if name == "adamw_turn_certified":
        return TurnAdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            gradient_trust=1.0,
        )
    if name == "adamw_turn_guarded":
        return TurnAdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            gradient_trust=10.0,
        )
    if name == "adamw_turn_adaptive_guard":
        return TurnAdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            gradient_trust=1.0, gradient_trust_max=10.0,
            gradient_trust_threshold=.3,
        )
    if name == "adamw_turn_global_guard":
        return TurnAdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            gradient_trust=1.0, gradient_trust_max=10.0,
            gradient_trust_threshold=.3, global_gradient_trust=True,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if name == "anchor":
        return Anchor(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            trust_radius=0.02,
        )
    if name == "anchor_cured":
        return Anchor(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            trust_radius=0.02, gradient_trust=100.0,
        )
    if name == "anchor_restrained_momentum":
        return RestrainedMomentumAnchor(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            trust_radius=0.02,
        )
    if name == "anchor_isotropic":
        return Anchor(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            trust_radius=0.02, isotropy_cap=2.0,
        )
    if name == "anchor_fixed":
        return Anchor(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            trust_radius=None,
        )
    if name == "muon":
        return MuonWithAuxAdamW(model, lr=lr, weight_decay=weight_decay)
    if name == "muon_original":
        return MuonWithAuxAdamW(model, lr=lr, weight_decay=weight_decay,
                                adjust_lr="original")
    if name == "muon_transport":
        return TransportMuonWithAuxAdamW(
            model, lr=lr, weight_decay=weight_decay
        )
    if name == "lepton":
        return LeptonWithAuxAdamW(
            model, lr=lr, weight_decay=weight_decay
        )
    if name == "lepton_transport":
        return TransportLeptonWithAuxAdamW(
            model, lr=lr, weight_decay=weight_decay
        )
    if name == "lepton_transport_current":
        return TransportLeptonWithAuxAdamW(
            model, lr=lr, weight_decay=weight_decay, frame_target="current"
        )
    if name == "lepton_transport_certified":
        return TransportLeptonWithAuxAdamW(
            model, lr=lr, weight_decay=weight_decay, gradient_trust=1.0
        )
    if name == "lepton_transport_current_certified":
        return TransportLeptonWithAuxAdamW(
            model, lr=lr, weight_decay=weight_decay, frame_target="current",
            gradient_trust=1.0,
        )
    if name == "lepton_transport_guarded":
        return TransportLeptonWithAuxAdamW(
            model, lr=lr, weight_decay=weight_decay, gradient_trust=100.0
        )
    if name == "lepton_transport_cured":
        return TransportLeptonWithAuxAdamW(
            model, lr=lr, weight_decay=weight_decay, gradient_trust=100.0
        )
    if name == "lepton_transport_current_guarded":
        return TransportLeptonWithAuxAdamW(
            model, lr=lr, weight_decay=weight_decay, frame_target="current",
            gradient_trust=100.0,
        )
    raise KeyError(name)
