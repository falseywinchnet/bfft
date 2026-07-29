import numpy as np

from experiments import cartoon_fourier_transport as fourier
from experiments import cartoon_hodge_motion_cycle as cycle


def test_none_schedule_is_independent_of_schedule_parameters():
    image = fourier.tree_experiment._load_image("synthetic", 32)
    a, events_a = cycle.run_cycle(
        image, max_cost=16, placement="none", start=1, gap=1
    )
    b, events_b = cycle.run_cycle(
        image, max_cost=16, placement="none", start=7, gap=9
    )
    assert not events_a and not events_b
    assert a[-1].cost == b[-1].cost == 16
    assert np.array_equal(a[-1].u, b[-1].u)
    assert np.array_equal(a[-1].v, b[-1].v)


def test_hodge_reseat_preserves_static_continuation_target():
    image = fourier.tree_experiment._load_image("synthetic", 32)
    state = None
    u = image.copy()
    for _ in range(4):
        u, state = cycle.meyer.rof_sb(
            image, 0.05, eta=0.10, state=state, sweeps=1
        )
    accelerated, alpha, _ = cycle.hodge_state_drop(
        image, 0.05, 0.10, u, state
    )
    assert alpha > 0.0
    continued, state = cycle.meyer.rof_sb(
        image, 0.05, eta=0.10, state=state, sweeps=1
    )
    assert np.isfinite(continued).all()
    assert state.u is continued
    assert not np.array_equal(accelerated, continued)


def test_motion_can_reenable_a_replacement_hodge_shot():
    image = fourier.tree_experiment._load_image("cameraman", 32)
    _, events = cycle.run_cycle(
        image,
        max_cost=48,
        placement="replace",
        side="u",
        start=4,
        gap=1,
    )
    accepted_after_motion = [
        event for event in events
        if event.accepted and event.target_motion > 0.0
    ]
    assert accepted_after_motion


def test_additive_cycle_cost_is_charged_explicitly():
    image = fourier.tree_experiment._load_image("synthetic", 32)
    points, events = cycle.run_cycle(
        image,
        max_cost=20,
        placement="add",
        side="u",
        start=2,
        gap=4,
        hodge_cost=2.0,
    )
    assert events
    assert points[-1].cost <= 20
    assert points[-1].cost == 2 * points[-1].outer + 2 * len(events)


def test_finite_motion_burst_beats_equal_cost_baseline():
    image = fourier.tree_experiment._load_image("synthetic", 32)
    reference, _ = cycle.run_cycle(
        image, max_cost=2048, placement="none"
    )
    baseline, _ = cycle.run_cycle(
        image, max_cost=64, placement="none"
    )
    burst, events = cycle.run_cycle(
        image,
        max_cost=64,
        placement="add",
        side="u",
        start=2,
        gap=1,
        shot_limit=6,
        hodge_cost=2.0,
    )
    baseline_error = cycle.combined_error(
        baseline[-1], reference[-1].u, reference[-1].v, image
    )
    burst_error = cycle.combined_error(
        burst[-1], reference[-1].u, reference[-1].v, image
    )
    assert len(events) == 6
    assert sum(event.accepted for event in events) == 6
    assert burst_error < 0.80 * baseline_error
