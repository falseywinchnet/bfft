"""Small CPU-compatible Muon + AdamW optimizer used by the experiments.

The implementation follows ``torch.optim.Muon``.  Muon is applied only to
hidden two-dimensional weights; vectors, biases, input embeddings, and output
heads remain on AdamW.
"""
from __future__ import annotations

import math

import torch


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


class MuonWithAuxAdamW:
    """Optimizer facade combining Muon hidden matrices with AdamW auxiliaries."""

    def __init__(self, model, lr=3e-3, weight_decay=1e-4, muon_lr=None,
                 momentum=.95, adjust_lr="match_rms_adamw"):
        muon_parameters, auxiliary_parameters = [], []
        for name, parameter in model.named_parameters():
            is_edge = name.startswith(("embed.", "encode.", "output.", "decode."))
            if parameter.ndim == 2 and not is_edge:
                muon_parameters.append(parameter)
            else:
                auxiliary_parameters.append(parameter)
        if not muon_parameters or not auxiliary_parameters:
            raise ValueError("hybrid Muon requires both hidden matrices and auxiliary parameters")
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


def make_optimizer(model, name: str, lr: float, weight_decay: float = 1e-4):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "muon":
        return MuonWithAuxAdamW(model, lr=lr, weight_decay=weight_decay)
    if name == "muon_original":
        return MuonWithAuxAdamW(model, lr=lr, weight_decay=weight_decay,
                                adjust_lr="original")
    raise KeyError(name)
