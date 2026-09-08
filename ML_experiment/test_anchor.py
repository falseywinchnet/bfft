from __future__ import annotations

import math

import torch

from ML_experiment.anchor import Anchor, RestrainedMomentumAnchor, _parallel_transport


def _set_gradient(parameter, values):
    parameter.grad = torch.tensor(values, dtype=parameter.dtype)


def test_anchor_constant_request_has_exact_lead_lag_recurrence():
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = Anchor(
        [parameter], lr=1.0, weight_decay=0.0, trust_radius=None
    )
    c = math.exp(-1.0)
    retain = 1.0 - c
    gradient = torch.tensor([2.0, -3.0], dtype=torch.float64)

    _set_gradient(parameter, gradient.tolist())
    before = parameter.detach().clone()
    optimizer.step()
    first_fast = c * gradient
    first_slow = c * first_fast
    assert torch.allclose(before - parameter.detach(), first_fast)
    assert torch.allclose(optimizer.state[parameter]["anchor_slow"], first_slow)

    _set_gradient(parameter, gradient.tolist())
    before = parameter.detach().clone()
    optimizer.step()
    second_fast = retain * first_slow + c * gradient
    second_slow = retain * first_slow + c * second_fast
    assert torch.allclose(before - parameter.detach(), second_fast)
    assert torch.allclose(optimizer.state[parameter]["anchor_slow"], second_slow)


def test_anchor_never_amplifies_its_fast_proposal():
    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = Anchor([parameter], lr=1.0)
    for values in (
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [-1.0, 0.5, 3.0],
        [0.25, -2.0, 1.0],
    ):
        _set_gradient(parameter, values)
        optimizer.step()
        assert math.isfinite(optimizer.last_ratio)
        assert optimizer.last_ratio <= 1.0 + 2e-6


def test_parallel_transport_preserves_norm_and_resets_at_antipode():
    value = torch.tensor([[2.0, 3.0, -4.0]], dtype=torch.float64)
    frame_from = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    frame_to = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    transported = _parallel_transport(value, frame_from, frame_to, 1e-24)
    assert torch.allclose(
        torch.linalg.vector_norm(transported, dim=1),
        torch.linalg.vector_norm(value, dim=1),
    )
    assert torch.equal(
        _parallel_transport(value, frame_from, -frame_from, 1e-24),
        torch.zeros_like(value),
    )


def test_dynamic_anchor_caps_local_relative_displacement():
    parameter = torch.nn.Parameter(torch.tensor([10.0, 0.0], dtype=torch.float64))
    optimizer = Anchor([parameter], lr=1.0, trust_radius=0.02)
    _set_gradient(parameter, [100.0, 0.0])
    before = parameter.detach().clone()
    optimizer.step()
    relative = torch.linalg.vector_norm(before - parameter.detach()) / 10.0
    assert relative <= 0.02 * (1.0 + 1e-12)
    assert optimizer.last_lr_scale < 1.0


def test_restrained_momentum_contracts_the_stored_departure_component():
    standard_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    restrained_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    standard = Anchor(
        [standard_parameter], lr=1.0, weight_decay=0.0, trust_radius=None
    )
    restrained = RestrainedMomentumAnchor(
        [restrained_parameter], lr=1.0, weight_decay=0.0, trust_radius=None
    )

    for values in ([1.0, 0.0], [0.0, 1.0]):
        _set_gradient(standard_parameter, values)
        _set_gradient(restrained_parameter, values)
        standard.step()
        restrained.step()

    standard_slow = standard.state[standard_parameter]["anchor_slow"]
    restrained_slow = restrained.state[restrained_parameter]["anchor_slow"]
    departure = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    departure /= torch.linalg.vector_norm(departure)
    standard_component = torch.dot(standard_slow, departure).abs()
    restrained_component = torch.dot(restrained_slow, departure).abs()
    assert restrained_component < standard_component
    assert restrained.last_memory_ratio < 1.0


def test_restrained_momentum_matches_anchor_in_a_constant_frame():
    standard_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    restrained_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    standard = Anchor(
        [standard_parameter], lr=1.0, weight_decay=0.0, trust_radius=None
    )
    restrained = RestrainedMomentumAnchor(
        [restrained_parameter], lr=1.0, weight_decay=0.0, trust_radius=None
    )
    for _ in range(4):
        _set_gradient(standard_parameter, [2.0, -1.0])
        _set_gradient(restrained_parameter, [2.0, -1.0])
        standard.step()
        restrained.step()
    assert torch.allclose(standard_parameter, restrained_parameter)
    assert torch.allclose(
        standard.state[standard_parameter]["anchor_slow"],
        restrained.state[restrained_parameter]["anchor_slow"],
    )
