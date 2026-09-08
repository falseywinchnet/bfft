"""Residual-anisotropy-governed matrix transport optimizer.

``ResidualPolarTransport`` combines four operations:

1. a non-elementwise Kronecker-factored matrix metric;
2. minimum-rotation transport of momentum between live gradient frames;
3. longitudinal fusion, which closes only the live axial inconsistency;
4. a reversible interpolation toward polar geometry when the unresolved
   operator becomes spectrally anisotropic.

The polar controller never reads a validation loss or an iteration schedule.

QUICK START: AUTOMATIC LINEAR-MODULE OBSERVATION
------------------------------------------------

Attach the optimizer *before the first forward pass*. ``attach_model`` installs
one forward hook and one full-backward hook on each optimized
``torch.nn.Linear``. The hooks observe the relation between that layer's input
activation and its output cotangent before the batch is contracted into one
weight gradient::

    model = MyModel()
    optimizer = ResidualPolarTransport(model.parameters(), lr=3e-3)
    optimizer.attach_model(model)

    for x, target in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

    optimizer.remove_observers()

The observed entropy ranks are training-backprop state. No label is inspected
by the optimizer, and no validation or held-out value is used. Biases and other
vectors retain transported Fusion behavior but cannot receive a polar matrix
shape.

Observation need not cover every linear module. In a network where a named
matrix is the first contraction after a nonlinearity, it can be used as the
only local observer::

    observed = {model.down.base, model.down.metric}
    optimizer.attach_model(
        model,
        module_filter=lambda module: module in observed,
    )

Only observed weights receive locally governed polar shape. Unobserved matrix
weights remain on Matrix Transport unless an explicit global rank is supplied.
This is often the correct reduction for consecutive affine maps: without an
intervening nonlinearity their residual operators are coordinate-transformed
views of the same relation, whereas a nonlinearity creates a genuinely new
local tangent geometry worth observing.

QUICK START: EXPLICIT RESIDUAL OBSERVATION
------------------------------------------

If hooks are undesirable, supply a source activation and its training residual
explicitly before ``step``. All leading dimensions are flattened into samples;
the final dimensions are source and residual coordinates::

    optimizer = ResidualPolarTransport(model.parameters(), lr=3e-3)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    optimizer.observe_residual(x, prediction - target)
    loss = criterion(prediction, target)
    loss.backward()
    optimizer.step()

By default this queues one global rank for all matrix parameters. Pass
``parameters=module.weight`` (or an iterable of parameters) to govern only a
specified matrix. A parameter-specific observation overrides the global value
for that parameter on that update. For a custom observer, a precomputed rank
can be queued with ``set_residual_entropy_rank``.

USAGE CONTRACT AND LIMITATIONS
------------------------------

* Calling neither observer is valid. Polar weight remains zero and the class
  behaves as longitudinally fused Matrix Transport.
* Attach once, before the first observed forward. Calling ``attach_model`` a
  second time removes the old hooks before installing new ones.
* Shared/reused linear modules are supported with a LIFO activation stack.
  Multiple backward passes before one ``step`` are treated as gradient
  accumulation; their rank observations are averaged on that step.
* Forwards inside ``torch.no_grad`` are ignored, so ordinary evaluation does
  not retain activations. A grad-enabled forward that is intentionally not
  followed by backward should be followed by ``clear_observations``.
* If an accumulated or partially completed update is abandoned without
  calling ``step``, call ``clear_observations`` so its geometry is not consumed
  by the next update.
* Hooks and random-sketch caches are runtime objects and are not serialized in
  ``state_dict``. After restoring a checkpoint into a new optimizer, call
  ``attach_model`` again. Optimizer moments, matrix roots, rank state, polar
  weights, and LR gains are serialized normally.
* ``remove_observers`` must be called before permanently discarding or replacing
  an attached optimizer. Context-manager cleanup is intentionally not assumed.
* The observer uses float32 and fixed deterministic two-sided sketches when a
  feature space exceeds ``observer_dim``. Increase that dimension for a more
  faithful spectrum at additional cost; decrease it for cheaper observation.
* Automatic hooks are intended for eager, single-process PyTorch. Some
  ``torch.compile`` graphs treat full-backward hooks as graph breaks. Under DDP
  the locally observed ranks are not automatically all-reduced, so use a
  synchronized explicit rank if identical optimizer state across workers is
  required.
* This reference keeps full row and column covariance factors for every matrix:
  state is ``O(rows² + columns²)`` and a root refresh performs dense symmetric
  eigendecompositions. Cached roots make the measured small/medium-matrix use
  practical; very large transformer matrices require a future blocked or
  low-rank root backend rather than pretending this file is memory-free.
* AMP loss scaling does not change entropy rank because the statistic is scale
  invariant. Normal ``GradScaler.step(optimizer)`` still needs gradients to be
  unscaled in the standard way before the optimizer update.
* ``attach_model`` recognizes ``nn.Linear`` by default. A ``module_filter`` may
  admit a custom linear-like module only if it has a two-dimensional ``weight``,
  tensor input in ``inputs[0]``, and tensor cotangent in ``grad_output[0]``.
* Dense real-valued gradients are required. Sparse and complex parameters are
  rejected.

The most useful runtime diagnostics are ``last_residual_entropy_rank``,
``last_polar_weight``, ``last_lr_scale``, ``last_turn``,
``last_longitudinal_closure``, and ``last_metric_condition``.

The explicit observer reproduces the evaluated operator experiment; the hook
path is its task-agnostic, layer-local generalization.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Callable

import torch


__all__ = ["ResidualPolarTransport", "Optimizer"]


def _minimum_rotation(
    value: torch.Tensor,
    frame_from: torch.Tensor,
    frame_to: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Transport vectors by the minimum rotation between unit frames."""
    axial = torch.sum(value * frame_from, dim=1)
    tangent = value - axial[:, None] * frame_from
    denominator = 1.0 + torch.sum(frame_from * frame_to, dim=1)
    transported_tangent = tangent - (
        torch.sum(tangent * frame_to, dim=1)
        / denominator.clamp_min(1e-30)
    )[:, None] * (frame_from + frame_to)
    transported = axial[:, None] * frame_to + transported_tangent
    return torch.where(
        (denominator > eps)[:, None], transported, torch.zeros_like(transported)
    )


def _zeropower_newton_schulz5(
    matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate the semi-orthogonal polar factor without an SVD."""
    if matrix.ndim != 2:
        raise ValueError("polar shaping requires a matrix")
    a, b, c = 3.4445, -4.7750, 2.0315
    transposed = matrix.shape[0] > matrix.shape[1]
    value = matrix.T if transposed else matrix
    value = value / torch.linalg.vector_norm(value).clamp_min(eps)
    for _ in range(steps):
        gram = value @ value.T
        value = a * value + (b * gram + c * (gram @ gram)) @ value
    return value.T if transposed else value


def _entropy_rank_fraction(matrix: torch.Tensor, eps: float = 1e-20) -> float:
    """Spectral entropy rank divided by the smaller matrix dimension."""
    singular = torch.linalg.svdvals(matrix.float())
    energy = singular.square()
    total = energy.sum()
    if not torch.isfinite(total) or float(total) <= eps:
        return 1.0
    probability = energy / total
    entropy_rank = torch.exp(-torch.sum(
        probability * torch.log(probability.clamp_min(eps))
    ))
    return float((entropy_rank / min(matrix.shape)).clamp(0.0, 1.0))


class ResidualPolarTransport(torch.optim.Optimizer):
    """Transported matrix optimizer with reversible residual-polar geometry.

    Matrix gradients are read through two covariance factors

    ``L = EMA(G G^T / columns)`` and ``R = EMA(G^T G / rows)``.

    Their inverse fourth roots produce the live request
    ``L^-1/4 G R^-1/4``. Momentum is transported into each new live frame,
    then longitudinal fusion closes the currently identifiable axial error
    without deleting transverse route information.

    A normalized residual-operator entropy rank ``r`` controls polar weight.
    The target is ``1-r``: isotropic residuals return to Fusion, while a
    persistent narrow tail earns singular-value equalization. Polar and Fusion
    requests are norm matched before interpolation, so this controller changes
    geometry rather than silently changing the learning rate.

    Args:
        params: Iterable of parameters or optimizer parameter groups.
        lr: Declared learning rate.
        betas: Momentum and covariance decay factors.
        matrix_ridge: Relative eigenvalue floor for matrix roots.
        weight_decay: Decoupled weight decay.
        polar_beta: Low-pass coefficient when polar weight increases.
        polar_release_beta: Coefficient when polar weight decreases. Defaults
            to ``polar_beta``; use a smaller value for faster retreat.
        polar_steps: Newton--Schulz iterations for the polar factor.
        observer_dim: Maximum activation and cotangent sketch dimension.
        observer_ridge: Relative ridge in the residual-operator regression.
        observer_center: Remove batch means before residual regression.
        observer_seed: Deterministic seed for fixed random sketches.
        root_drift_threshold: Relative covariance movement that refreshes roots.
        max_root_staleness: Hard upper bound between root refreshes.
        gain_rate: Rate of the bounded coherence-controlled scalar LR gain.
        min_gain: Minimum scalar LR gain.
        max_gain: Maximum scalar LR gain.
        target_coherence: Neutral frame coherence for scalar LR adaptation.
        transport_eps: Numerical floor for frame transport.
        eps: General numerical floor.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3e-3,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        matrix_ridge: float = 1e-4,
        weight_decay: float = 1e-4,
        polar_beta: float = 0.9,
        polar_release_beta: float | None = None,
        polar_steps: int = 5,
        observer_dim: int = 32,
        observer_ridge: float = 1e-7,
        observer_center: bool = True,
        observer_seed: int = 1729,
        root_drift_threshold: float = 0.05,
        max_root_staleness: int = 10,
        gain_rate: float = 0.02,
        min_gain: float = 0.25,
        max_gain: float = 4.0,
        target_coherence: float = 0.0,
        transport_eps: float = 1e-7,
        eps: float = 1e-8,
    ) -> None:
        beta1, beta2 = betas
        release_beta = polar_beta if polar_release_beta is None else polar_release_beta
        if lr <= 0 or weight_decay < 0 or matrix_ridge <= 0:
            raise ValueError("lr and matrix_ridge must be positive; weight_decay nonnegative")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must lie in [0, 1)")
        if not 0 <= polar_beta < 1 or not 0 <= release_beta < 1:
            raise ValueError("polar smoothing coefficients must lie in [0, 1)")
        if polar_steps < 1 or observer_dim < 1 or observer_ridge <= 0:
            raise ValueError("invalid polar or observer configuration")
        if root_drift_threshold < 0 or max_root_staleness < 1:
            raise ValueError("invalid root cache configuration")
        if gain_rate < 0 or not 0 < min_gain <= 1 <= max_gain:
            raise ValueError("invalid learning-rate gain bounds")
        if not -1 <= target_coherence <= 1 or eps <= 0 or transport_eps <= 0:
            raise ValueError("invalid coherence or numerical floor")

        defaults = dict(
            lr=float(lr),
            betas=(float(beta1), float(beta2)),
            matrix_ridge=float(matrix_ridge),
            weight_decay=float(weight_decay),
            polar_beta=float(polar_beta),
            polar_release_beta=float(release_beta),
            polar_steps=int(polar_steps),
            root_drift_threshold=float(root_drift_threshold),
            max_root_staleness=int(max_root_staleness),
            gain_rate=float(gain_rate),
            min_gain=float(min_gain),
            max_gain=float(max_gain),
            target_coherence=float(target_coherence),
            transport_eps=float(transport_eps),
            eps=float(eps),
            lr_gain=1.0,
        )
        super().__init__(params, defaults)

        self.observer_dim = int(observer_dim)
        self.observer_ridge = float(observer_ridge)
        self.observer_center = bool(observer_center)
        self.observer_seed = int(observer_seed)
        self._pending_global_ranks: list[float] = []
        self._pending_parameter_ranks: dict[int, list[float]] = {}
        self._observer_projections: dict[tuple, torch.Tensor] = {}
        self._activation_stacks: dict[int, list[torch.Tensor]] = {}
        self._observer_handles: list[torch.utils.hooks.RemovableHandle] = []

        self.last_turn = 0.0
        self.last_lr_scale = 1.0
        self.last_history_alignment = 1.0
        self.last_metric_condition = 1.0
        self.last_root_refresh_fraction = 1.0
        self.last_root_staleness = 0.0
        self.last_longitudinal_closure = 0.0
        self.last_longitudinal_certificate = 0.0
        self.last_residual_entropy_rank = 1.0
        self.last_polar_weight = 0.0

    @staticmethod
    def _working_dtype(parameter: torch.Tensor) -> torch.dtype:
        return torch.float64 if parameter.dtype == torch.float64 else torch.float32

    @staticmethod
    def _inverse_fourth_root(
        matrix: torch.Tensor, ridge_fraction: float, eps: float
    ) -> tuple[torch.Tensor, float]:
        eigenvalues, eigenvectors = torch.linalg.eigh(
            0.5 * (matrix + matrix.T)
        )
        mean = eigenvalues.clamp_min(0).mean()
        floor = ridge_fraction * mean + eps * eps
        regularized = eigenvalues.clamp_min(floor)
        inverse = regularized.pow(-0.25)
        root = (eigenvectors * inverse.unsqueeze(0)) @ eigenvectors.T
        condition = float(regularized.max() / regularized.min())
        return root, condition

    def _initialize_state(
        self, parameter: torch.nn.Parameter, state: dict
    ) -> None:
        dtype = self._working_dtype(parameter)
        state["step"] = 0
        state["momentum"] = torch.zeros_like(parameter, dtype=dtype)
        state["polar_weight"] = 0.0
        state["residual_entropy_rank"] = 1.0
        if parameter.ndim >= 2:
            rows = parameter.shape[0]
            columns = parameter.numel() // rows
            state["left"] = torch.zeros(
                rows, rows, device=parameter.device, dtype=dtype
            )
            state["right"] = torch.zeros(
                columns, columns, device=parameter.device, dtype=dtype
            )
        else:
            state["mean_square"] = torch.zeros(
                (), device=parameter.device, dtype=dtype
            )

    def _projection(
        self,
        size: int,
        reduced: int,
        device: torch.device,
        dtype: torch.dtype,
        tag: int,
    ) -> torch.Tensor:
        key = (size, reduced, str(device), dtype, tag)
        cached = self._observer_projections.get(key)
        if cached is not None:
            return cached
        generator = torch.Generator(device="cpu")
        seed = self.observer_seed + 104729 * tag + 1009 * size + 9176 * reduced
        generator.manual_seed(seed % (2**63 - 1))
        raw = torch.randn(size, reduced, generator=generator, dtype=torch.float32)
        projection = torch.linalg.qr(raw, mode="reduced").Q.to(
            device=device, dtype=dtype
        )
        self._observer_projections[key] = projection
        return projection

    @torch.no_grad()
    def _estimate_residual_rank(
        self,
        activations: torch.Tensor,
        residuals: torch.Tensor,
        cache_tag: int = 0,
    ) -> float:
        if activations.ndim < 2 or residuals.ndim < 2:
            raise ValueError("activations and residuals need a feature dimension")
        x = activations.detach().reshape(-1, activations.shape[-1]).float()
        delta = residuals.detach().reshape(-1, residuals.shape[-1]).float()
        count = min(x.shape[0], delta.shape[0])
        if count < 2:
            return 1.0
        x, delta = x[:count], delta[:count]
        finite = torch.isfinite(x).all(dim=1) & torch.isfinite(delta).all(dim=1)
        x, delta = x[finite], delta[finite]
        if x.shape[0] < 2:
            return 1.0
        if self.observer_center:
            x = x - x.mean(dim=0, keepdim=True)
            delta = delta - delta.mean(dim=0, keepdim=True)

        if x.shape[1] > self.observer_dim:
            projection = self._projection(
                x.shape[1], self.observer_dim, x.device, x.dtype,
                2 * cache_tag + 1,
            )
            x = x @ projection
        if delta.shape[1] > self.observer_dim:
            projection = self._projection(
                delta.shape[1], self.observer_dim, delta.device, delta.dtype,
                2 * cache_tag + 2,
            )
            delta = delta @ projection

        gram = x.T @ x / x.shape[0]
        cross = x.T @ delta / x.shape[0]
        scale = torch.trace(gram).clamp_min(1e-20) / gram.shape[0]
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        operator = torch.linalg.solve(
            gram + self.observer_ridge * scale * identity,
            cross,
        )
        return _entropy_rank_fraction(operator)

    def set_residual_entropy_rank(
        self,
        normalized_rank: float,
        parameters: Iterable[torch.nn.Parameter] | torch.nn.Parameter | None = None,
    ) -> None:
        """Queue an externally measured residual entropy rank for the next step."""
        rank = max(0.0, min(1.0, float(normalized_rank)))
        if parameters is None:
            self._pending_global_ranks.append(rank)
            return
        if isinstance(parameters, torch.nn.Parameter):
            parameters = (parameters,)
        for parameter in parameters:
            self._pending_parameter_ranks.setdefault(id(parameter), []).append(rank)

    @torch.no_grad()
    def observe_residual(
        self,
        activations: torch.Tensor,
        residuals: torch.Tensor,
        parameters: Iterable[torch.nn.Parameter] | torch.nn.Parameter | None = None,
    ) -> float:
        """Estimate and queue covariance-corrected residual anisotropy.

        ``activations`` and ``residuals`` share all leading sample dimensions.
        Their last dimensions are treated as source and target coordinates.
        The returned number is normalized entropy rank, not anisotropy; polar
        target weight is its complement.
        """
        rank = self._estimate_residual_rank(activations, residuals)
        self.set_residual_entropy_rank(rank, parameters)
        return rank

    def attach_model(
        self,
        model: torch.nn.Module,
        module_filter: Callable[[torch.nn.Module], bool] | None = None,
    ) -> "ResidualPolarTransport":
        """Observe local residual operators on the model's linear modules.

        The forward hook retains each module's input activation until its
        backward hook receives the output cotangent. Fixed random projections
        bound the observer at ``observer_dim`` without changing the optimizer
        update or using labels, task identity, or validation data.
        """
        self.remove_observers()
        optimized = {id(parameter) for group in self.param_groups
                     for parameter in group["params"]}

        def eligible(module: torch.nn.Module) -> bool:
            if module_filter is not None:
                return bool(module_filter(module))
            return isinstance(module, torch.nn.Linear)

        for module_index, module in enumerate(model.modules()):
            weight = getattr(module, "weight", None)
            if (not eligible(module) or not isinstance(weight, torch.nn.Parameter)
                    or id(weight) not in optimized or weight.ndim != 2):
                continue
            module_key = id(module)
            self._activation_stacks[module_key] = []

            def forward_hook(
                hooked_module, inputs, output, *, key=module_key
            ):
                del hooked_module
                if (torch.is_grad_enabled() and inputs
                        and isinstance(inputs[0], torch.Tensor)
                        and isinstance(output, torch.Tensor)
                        and output.requires_grad):
                    self._activation_stacks[key].append(inputs[0].detach())

            def backward_hook(
                hooked_module, grad_input, grad_output, *, key=module_key,
                parameter=weight, tag=module_index + 1,
            ):
                del hooked_module, grad_input
                stack = self._activation_stacks.get(key)
                if not stack or not grad_output or not isinstance(grad_output[0], torch.Tensor):
                    return
                activations = stack.pop()
                with torch.no_grad():
                    rank = self._estimate_residual_rank(
                        activations, grad_output[0], cache_tag=tag
                    )
                self.set_residual_entropy_rank(rank, parameter)

            self._observer_handles.append(
                module.register_forward_hook(forward_hook)
            )
            self._observer_handles.append(
                module.register_full_backward_hook(backward_hook)
            )
        return self

    def remove_observers(self) -> None:
        """Remove hooks installed by :meth:`attach_model`."""
        for handle in self._observer_handles:
            handle.remove()
        self._observer_handles.clear()
        self.clear_observations()

    def clear_observations(self) -> None:
        """Discard queued ranks and retained activations from an abandoned step."""
        self._pending_global_ranks.clear()
        self._pending_parameter_ranks.clear()
        for stack in self._activation_stacks.values():
            stack.clear()

    def _consume_rank(self, parameter: torch.nn.Parameter, state: dict, group: dict) -> None:
        local = self._pending_parameter_ranks.get(id(parameter))
        observations = local if local else self._pending_global_ranks
        if not observations or parameter.ndim < 2:
            return
        rank = sum(observations) / len(observations)
        target = 1.0 - rank
        current = float(state["polar_weight"])
        beta = (
            group["polar_beta"] if target >= current
            else group["polar_release_beta"]
        )
        state["polar_weight"] = beta * current + (1.0 - beta) * target
        state["residual_entropy_rank"] = rank

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        turns, alignments, conditions = [], [], []
        root_refreshes, root_ages, closures, certificates = [], [], [], []
        observed_ranks, polar_weights, gains = [], [], []

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            group_coherences = []
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("ResidualPolarTransport requires dense gradients")
                if parameter.is_complex():
                    raise RuntimeError("ResidualPolarTransport does not support complex parameters")
                state = self.state[parameter]
                if not state:
                    self._initialize_state(parameter, state)
                self._consume_rank(parameter, state, group)

                work_gradient = gradient.detach().to(state["momentum"].dtype)
                state["step"] += 1
                bias2 = 1.0 - beta2 ** state["step"]

                if parameter.ndim >= 2:
                    rows = parameter.shape[0]
                    matrix = work_gradient.reshape(rows, -1)
                    columns = matrix.shape[1]
                    left, right = state["left"], state["right"]
                    left.mul_(beta2).add_(
                        matrix @ matrix.T, alpha=(1.0 - beta2) / columns
                    )
                    right.mul_(beta2).add_(
                        matrix.T @ matrix, alpha=(1.0 - beta2) / rows
                    )
                    corrected_left = left / bias2
                    corrected_right = right / bias2
                    age = state["step"] - state.get("last_root_step", 0)
                    reference_left = state.get("root_reference_left")
                    if reference_left is None:
                        drift = math.inf
                    else:
                        left_drift = torch.linalg.vector_norm(
                            corrected_left - reference_left
                        ) / torch.linalg.vector_norm(reference_left).clamp_min(group["eps"])
                        reference_right = state["root_reference_right"]
                        right_drift = torch.linalg.vector_norm(
                            corrected_right - reference_right
                        ) / torch.linalg.vector_norm(reference_right).clamp_min(group["eps"])
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
                    live_request = (left_root @ matrix @ right_root).reshape_as(
                        work_gradient
                    )
                    conditions.append(float(state["root_condition"]))
                    root_refreshes.append(float(refresh))
                    root_ages.append(float(age))
                else:
                    square = state["mean_square"]
                    square.mul_(beta2).add_(
                        work_gradient.square().mean(), alpha=1.0 - beta2
                    )
                    live_request = work_gradient / (
                        (square / bias2).sqrt() + group["eps"]
                    )

                flat_live = live_request.reshape(1, -1)
                live_norm = torch.linalg.vector_norm(flat_live, dim=1)
                signature = flat_live / live_norm.clamp_min(group["transport_eps"])[:, None]
                momentum = state["momentum"].reshape(1, -1)
                previous = state.get("signature")
                if previous is not None and float(live_norm) > group["transport_eps"]:
                    cosine = torch.sum(previous * signature, dim=1).clamp(-1.0, 1.0)
                    confidence = cosine.clamp(0.0, 1.0)
                    state["frame_confidence"] = float(confidence)
                    turns.append(float(0.5 * (1.0 - cosine)))
                    group_coherences.append(float(cosine))
                    if float(1.0 + cosine) > group["transport_eps"]:
                        fully_moved = _minimum_rotation(
                            momentum, previous, signature, group["transport_eps"]
                        )
                        axial = torch.sum(fully_moved * signature, dim=1)[:, None] * signature
                        transverse = fully_moved - axial
                        fully_moved = axial + confidence[:, None] * transverse
                        momentum.copy_(
                            (1.0 - confidence)[:, None] * momentum
                            + confidence[:, None] * fully_moved
                        )
                    previous.copy_(signature)
                else:
                    state["signature"] = signature.clone()
                    state["frame_confidence"] = 1.0

                momentum.lerp_(flat_live, 1.0 - beta1)
                averaged = momentum / (1.0 - beta1 ** state["step"])

                # Longitudinal fusion: Nesterov transverse history is kept,
                # while positive frame confidence admits closure on live axis.
                nesterov = beta1 * averaged + (1.0 - beta1) * flat_live
                axial_history_value = torch.sum(averaged * signature, dim=1)
                certificate = float(state["frame_confidence"])
                correction = (
                    certificate * beta1
                    * (live_norm - axial_history_value)[:, None]
                    * signature
                )
                request = (nesterov + correction).reshape_as(work_gradient)
                closures.append(float(
                    torch.linalg.vector_norm(correction)
                    / live_norm.clamp_min(group["transport_eps"])
                ))
                certificates.append(certificate)

                if parameter.ndim >= 2 and state["polar_weight"] > 0.0:
                    request_matrix = request.reshape(rows, columns)
                    request_norm = torch.linalg.vector_norm(request_matrix)
                    if float(request_norm) > group["transport_eps"]:
                        polar = _zeropower_newton_schulz5(
                            request_matrix, group["polar_steps"]
                        )
                        polar.mul_(
                            request_norm
                            / torch.linalg.vector_norm(polar).clamp_min(
                                group["transport_eps"]
                            )
                        )
                        blended = torch.lerp(
                            request_matrix, polar, float(state["polar_weight"])
                        )
                        blended.mul_(
                            request_norm
                            / torch.linalg.vector_norm(blended).clamp_min(
                                group["transport_eps"]
                            )
                        )
                        request = blended.reshape_as(work_gradient)

                # Local Euclidean descent certificate.
                inner = torch.sum(request.double() * work_gradient.double())
                gradient_norm2 = work_gradient.double().square().sum()
                if float(inner) < 0.0 and float(gradient_norm2) > group["eps"]:
                    request.add_(
                        work_gradient, alpha=-float(inner / gradient_norm2)
                    )

                request_norm = torch.linalg.vector_norm(request)
                gradient_norm = torch.linalg.vector_norm(work_gradient)
                if (float(request_norm) > group["transport_eps"]
                        and float(gradient_norm) > group["transport_eps"]):
                    alignments.append(float(
                        torch.sum(request.double() * work_gradient.double())
                        / (request_norm.double() * gradient_norm.double())
                    ))

                effective_lr = group["lr"] * group["lr_gain"]
                parameter.mul_(1.0 - effective_lr * group["weight_decay"])
                parameter.add_(request.to(parameter.dtype), alpha=-effective_lr)

                if parameter.ndim >= 2:
                    observed_ranks.append(float(state["residual_entropy_rank"]))
                    polar_weights.append(float(state["polar_weight"]))

            if group_coherences:
                mean_coherence = sum(group_coherences) / len(group_coherences)
                log_gain = math.log(group["lr_gain"]) + group["gain_rate"] * (
                    mean_coherence - group["target_coherence"]
                )
                group["lr_gain"] = min(
                    group["max_gain"],
                    max(group["min_gain"], math.exp(log_gain)),
                )
            gains.append(float(group["lr_gain"]))

        self.last_turn = sum(turns) / len(turns) if turns else 0.0
        self.last_history_alignment = (
            sum(alignments) / len(alignments) if alignments else 1.0
        )
        self.last_metric_condition = (
            sum(conditions) / len(conditions) if conditions else 1.0
        )
        self.last_root_refresh_fraction = (
            sum(root_refreshes) / len(root_refreshes) if root_refreshes else 0.0
        )
        self.last_root_staleness = (
            sum(root_ages) / len(root_ages) if root_ages else 0.0
        )
        self.last_longitudinal_closure = (
            sum(closures) / len(closures) if closures else 0.0
        )
        self.last_longitudinal_certificate = (
            sum(certificates) / len(certificates) if certificates else 0.0
        )
        self.last_residual_entropy_rank = (
            sum(observed_ranks) / len(observed_ranks) if observed_ranks else 1.0
        )
        self.last_polar_weight = (
            sum(polar_weights) / len(polar_weights) if polar_weights else 0.0
        )
        self.last_lr_scale = sum(gains) / len(gains) if gains else 1.0

        self._pending_global_ranks.clear()
        self._pending_parameter_ranks.clear()
        return loss


# A convenient import for a downloaded single-file optimizer:
# ``from optimizer import Optimizer``.
Optimizer = ResidualPolarTransport
