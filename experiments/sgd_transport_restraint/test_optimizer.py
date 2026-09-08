from __future__ import annotations

import math

import torch

from experiments.sgd_transport_restraint.optimizer import (
    ResponseTransportSGD,
    SignatureTransportSGD,
    _parallel_transport_blocks,
)
from experiments.sgd_transport_restraint.wolf_optimizer import WolfTransportV2


def _set_grad(parameter: torch.nn.Parameter, values: list[float]) -> None:
    parameter.grad = torch.tensor(values, dtype=parameter.dtype)


def test_first_step_is_exact_sgd_and_records_preupdate_state():
    p = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    opt = ResponseTransportSGD([p], lr=0.25, mode="shape")
    _set_grad(p, [4.0, 2.0])
    opt.step(observe=False)
    assert torch.equal(p, torch.tensor([1.0, -1.5]))
    assert torch.equal(opt.state[p]["previous_parameter"], torch.tensor([2.0, -1.0]))
    assert torch.equal(opt.state[p]["previous_grad"], torch.tensor([4.0, 2.0]))


def test_shape_restraint_removes_only_observed_response_component():
    p = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    opt = ResponseTransportSGD([p], lr=1.0, alpha=1.0, mode="shape")

    _set_grad(p, [-1.0, 0.0])
    opt.step(observe=False)  # p = [1, 0], s = [1, 0]
    _set_grad(p, [1.0, 3.0])  # y = [2, 3], s dot y > 0
    before = p.detach().clone()
    raw = p.grad.detach().clone()
    response = raw - torch.tensor([-1.0, 0.0])
    opt.step(observe=True)
    effective = before - p.detach()

    # Removed part is parallel to y; orthogonal component is untouched.
    removed = raw - effective
    cross = removed[0] * response[1] - removed[1] * response[0]
    assert abs(float(cross)) < 1e-6
    assert opt.last_coherence > 0.0
    assert opt.last_ratio <= 1.0 + 1e-6


def test_zero_request_motion_does_not_restrain():
    p = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    opt = ResponseTransportSGD([p], lr=1.0, alpha=1.0, mode="shape")
    _set_grad(p, [-1.0, 0.0])
    opt.step(observe=False)  # s = [1, 0]
    _set_grad(p, [-1.0, 0.0])  # y = 0
    before = p.detach().clone()
    raw = p.grad.detach().clone()
    opt.step(observe=True)
    assert torch.allclose(before - p.detach(), raw)
    assert opt.last_coherence == 0.0
    assert opt.last_ratio == 1.0


def test_none_mode_matches_sgd_for_a_two_step_pair():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    q = torch.nn.Parameter(p.detach().clone())
    ours = ResponseTransportSGD([p], lr=0.1, mode="none")
    sgd = torch.optim.SGD([q], lr=0.1)
    for observe, grad in [(False, [2.0, -1.0]), (True, [-3.0, 4.0])]:
        _set_grad(p, grad)
        _set_grad(q, grad)
        ours.step(observe=observe)
        sgd.step()
    assert torch.equal(p, q)


def test_scalar_control_matches_shape_gradient_norm():
    p = torch.nn.Parameter(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
    q = torch.nn.Parameter(p.detach().clone())
    shape = ResponseTransportSGD([p], lr=1.0, alpha=0.8, mode="shape")
    scalar = ResponseTransportSGD([q], lr=1.0, alpha=0.8, mode="scalar")
    first = torch.tensor([[-1.0, 0.0], [0.0, -2.0]])
    second = torch.tensor([[2.0, 1.0], [3.0, 1.0]])
    measured = []
    for opt, parameter in ((shape, p), (scalar, q)):
        parameter.grad = first.clone()
        opt.step(observe=False)
        before = parameter.detach().clone()
        parameter.grad = second.clone()
        opt.step(observe=True)
        measured.append(torch.linalg.vector_norm(before - parameter.detach()))
    assert torch.allclose(measured[0], measured[1])
    assert not torch.allclose(p, q)


def test_signature_constant_request_is_exact_sgd():
    p = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    opt = SignatureTransportSGD([p], lr=0.25, mode="shape")
    for _ in range(4):
        _set_grad(p, [4.0, 2.0])
        before = p.detach().clone()
        opt.step()
        assert torch.allclose(before - p.detach(), torch.tensor([1.0, 0.5]))
        assert opt.last_ratio <= 1.0 + 1e-6


def test_signature_scalar_is_norm_matched_but_not_shape_matched():
    p = torch.nn.Parameter(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
    q = torch.nn.Parameter(p.detach().clone())
    shape = SignatureTransportSGD([p], lr=1.0, mode="shape")
    scalar = SignatureTransportSGD([q], lr=1.0, mode="scalar")
    first = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    second = torch.tensor([[-1.0, 1.0], [2.0, -1.0]])
    updates = []
    for opt, parameter in ((shape, p), (scalar, q)):
        parameter.grad = first.clone()
        opt.step()
        before = parameter.detach().clone()
        parameter.grad = second.clone()
        opt.step()
        updates.append(before - parameter.detach())
        assert opt.last_ratio <= 1.0 + 1e-6
    assert torch.allclose(
        torch.linalg.vector_norm(updates[0]),
        torch.linalg.vector_norm(updates[1]),
    )
    assert not torch.allclose(updates[0], updates[1])


def test_hard_signature_is_exact_bisector_projection():
    p = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
    opt = SignatureTransportSGD(
        [p], lr=1.0, alpha=1.0, gate="hard", mode="shape"
    )
    _set_grad(p, [1.0, 0.0])
    opt.step()
    before = p.detach().clone()
    _set_grad(p, [0.0, 2.0])
    opt.step()
    update = before - p.detach()
    # ||g|| (u + r) / 2 for orthogonal unit requests u=[0,1], r=[1,0].
    assert torch.allclose(update, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert math.isclose(opt.last_ratio, 2.0**-0.5, rel_tol=1e-12)


def test_rotate_preserves_each_local_cell_norm():
    p = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float64))
    opt = SignatureTransportSGD([p], lr=1.0, mode="rotate")
    p.grad = torch.tensor([[1.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    opt.step()
    before = p.detach().clone()
    raw = torch.tensor([[0.0, 2.0], [4.0, 0.0]], dtype=torch.float64)
    p.grad = raw.clone()
    opt.step()
    update = before - p.detach()
    assert torch.allclose(
        torch.linalg.vector_norm(update, dim=1),
        torch.linalg.vector_norm(raw, dim=1),
    )
    assert math.isclose(opt.last_ratio, 1.0, rel_tol=1e-12)
    assert not torch.allclose(update, raw)


def test_two_fused_soft_cells_have_closed_form_gate():
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = SignatureTransportSGD(
        [p], lr=1.0, gate="soft", fusion=2, mode="shape"
    )
    _set_grad(p, [1.0, 0.0])
    opt.step()
    before = p.detach().clone()
    _set_grad(p, [0.0, 2.0])
    opt.step()
    # Orthogonal requests have tau=1/2, hence k=1-(1-tau)^2=3/4.
    assert torch.allclose(
        before - p.detach(),
        torch.tensor([0.75, 1.25], dtype=torch.float64),
    )


def test_capacity_signature_loses_authority_when_it_collapses():
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = SignatureTransportSGD(
        [p], lr=1.0, observer="capacity", mode="shape"
    )
    _set_grad(p, [1.0, 0.0])
    opt.step()  # seats q=[1,0]
    before = p.detach().clone()
    _set_grad(p, [-1.0, 0.0])
    opt.step()  # full reversal is restrained and q collapses to zero
    assert torch.equal(before, p.detach())
    before = p.detach().clone()
    _set_grad(p, [-1.0, 0.0])
    opt.step()  # collapsed q has no authority; request passes and reseats q
    assert torch.equal(before - p.detach(), torch.tensor([-1.0, 0.0]))


def test_rank_two_normal_bundle_never_amplifies():
    p = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    opt = SignatureTransportSGD(
        [p], lr=0.1, normal_rank=2, mode="shape"
    )
    for values in (
        [1.0, 2.0, 0.0, -1.0],
        [-2.0, 1.0, 3.0, 0.0],
        [0.5, -3.0, 1.0, 2.0],
    ):
        before = p.detach().clone()
        _set_grad(p, values)
        raw_norm = torch.linalg.vector_norm(p.grad)
        opt.step()
        update_norm = torch.linalg.vector_norm(before - p.detach()) / 0.1
        assert update_norm <= raw_norm * (1.0 + 1e-12)


def test_ordinary_momentum_mode_matches_torch_sgd():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    ours = SignatureTransportSGD(
        [p], lr=0.1, momentum=0.9, mode="none"
    )
    reference = torch.optim.SGD([q], lr=0.1, momentum=0.9)
    for values in ([2.0, -1.0], [-3.0, 4.0], [0.5, 2.0]):
        _set_grad(p, values)
        _set_grad(q, values)
        ours.step()
        reference.step()
    assert torch.allclose(p, q, rtol=1e-14, atol=1e-14)


def test_parallel_transport_preserves_norm_and_resets_at_antipode():
    value = torch.tensor([[2.0, 3.0, -4.0]], dtype=torch.float64)
    frame_from = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    frame_to = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    transported = _parallel_transport_blocks(
        value, frame_from, frame_to, 1e-24
    )
    assert torch.allclose(
        torch.linalg.vector_norm(transported, dim=1),
        torch.linalg.vector_norm(value, dim=1),
    )
    assert torch.allclose(
        _parallel_transport_blocks(
            value, frame_from, -frame_from, 1e-24
        ),
        torch.zeros_like(value),
    )


def test_normalized_momentum_preserves_a_constant_request():
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = SignatureTransportSGD(
        [p], lr=0.1, momentum=0.9,
        normalized_momentum=True, mode="none",
    )
    expected_update = torch.tensor([0.2, -0.3], dtype=torch.float64)
    for _ in range(5):
        before = p.detach().clone()
        _set_grad(p, [2.0, -3.0])
        opt.step()
        assert torch.allclose(before - p.detach(), expected_update)


def test_wolf_v2_is_the_exact_lead_lag_core():
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = WolfTransportV2([p], lr=1.0)
    c = 0.367879441
    retain = 1.0 - c
    gradient = torch.tensor([2.0, -3.0], dtype=torch.float64)

    _set_grad(p, gradient.tolist())
    before = p.detach().clone()
    opt.step()
    first_fast = c * gradient
    first_slow = c * first_fast
    assert torch.allclose(before - p.detach(), first_fast)
    assert torch.allclose(opt.state[p]["wolf_slow"], first_slow)

    _set_grad(p, gradient.tolist())
    before = p.detach().clone()
    opt.step()
    second_fast = retain * first_slow + c * gradient
    second_slow = retain * first_slow + c * second_fast
    assert torch.allclose(before - p.detach(), second_fast)
    assert torch.allclose(opt.state[p]["wolf_slow"], second_slow)


def test_wolf_live_axis_restraint_never_amplifies_the_proposal():
    p = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    opt = WolfTransportV2(
        [p], lr=1.0, transport_state=True, restrain_output=True
    )
    for values in (
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [-1.0, 0.5, 3.0],
        [0.25, -2.0, 1.0],
    ):
        _set_grad(p, values)
        opt.step()
        assert math.isfinite(opt.last_ratio)
        assert opt.last_ratio <= 1.0 + 2e-6
