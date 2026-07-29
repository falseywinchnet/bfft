#!/usr/bin/env python3
"""Sparse Hodge -> moving target -> Hodge cycles for the Meyer split.

The static one-shot loses leverage after its initial drop because subsequent
Split-Bregman sweeps keep solving the same ROF target.  Meyer's outer
alternation is different:

    u <- ROF(f-v, lambda)
    w <- ROF(f-u, 1/mu)
    v <- (f-u)-w.

Updating either side moves the other side's target.  This experiment asks
whether that motion restores enough Hodge leverage to justify another shot.

Two placements are kept distinct:

``add``
    Run the ordinary ROF sweep, then add a Hodge shot.

``replace``
    On scheduled cycles, use Hodge against the moved target instead of the
    ordinary sweep.  The opposite ROF side still performs the motion.

Schedules may shoot the cartoon side, texture-survivor side, both, or
alternate sides.  Results are compared at equal transform-equivalent cost.
One ordinary Split-Bregman sweep costs one unit; the measured native Hodge
shot defaults to two.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cartoon_fourier_transport as fourier  # noqa: E402
import meyer_bregman as meyer  # noqa: E402


@dataclass
class HodgeEvent:
    outer: int
    side: str
    alpha: float
    objective_gain: float
    target_motion: float

    @property
    def accepted(self) -> bool:
        return self.alpha > 0.0


@dataclass
class CyclePoint:
    outer: int
    cost: float
    u: np.ndarray
    v: np.ndarray
    events: int
    accepted: int


def hodge_state_drop(
    target: np.ndarray,
    c: float,
    eta: float,
    current: np.ndarray,
    state: meyer.RofState,
) -> tuple[np.ndarray, float, float]:
    """Apply one Hodge drop and re-seat a live Split-Bregman state."""
    pre_x, pre_y = fourier.routed_preflux(
        current,
        (eta * state.bx, eta * state.by),
        target,
        c,
    )
    scale = np.maximum(1.0, np.hypot(pre_x, pre_y))
    flux_x = pre_x / scale
    flux_y = pre_y / scale
    raw = target + meyer.div(flux_x, flux_y) / c
    accepted, alpha = fourier._segment_taylor_drop(
        current, raw, target, c
    )
    old_objective = fourier.objective(current, target, c)
    new_objective = fourier.objective(accepted, target, c)
    gain = max(0.0, old_objective - new_objective)
    if alpha > 0.0:
        state.u = accepted.copy()
        state.bx = flux_x / eta
        state.by = flux_y / eta
        state.dx, state.dy = meyer.grad(accepted)
    return accepted, alpha, gain


def _scheduled_side(
    outer: int,
    *,
    start: int,
    gap: int,
    side: str,
    shot_limit: int | None,
) -> str | None:
    if outer < start or (outer - start) % gap:
        return None
    shot_index = (outer - start) // gap
    if shot_limit is not None and shot_index >= shot_limit:
        return None
    if side != "alternate":
        return side
    return "u" if shot_index % 2 == 0 else "w"


def run_cycle(
    image: np.ndarray,
    *,
    max_cost: float,
    lam: float = 0.05,
    mu: float = 40.0,
    eta_u: float | None = None,
    eta_w: float | None = None,
    placement: str = "add",
    side: str = "u",
    start: int = 8,
    gap: int = 16,
    shot_limit: int | None = None,
    hodge_cost: float = 2.0,
) -> tuple[list[CyclePoint], list[HodgeEvent]]:
    """Run a cost-bounded moving-target schedule."""
    if placement not in ("none", "add", "replace"):
        raise ValueError("placement must be none, add, or replace")
    if side not in ("u", "w", "both", "alternate"):
        raise ValueError("side must be u, w, both, or alternate")
    if start < 1 or gap < 1 or hodge_cost <= 0.0 or (
        shot_limit is not None and shot_limit < 1
    ):
        raise ValueError(
            "start, gap, hodge_cost, and any shot_limit must be positive"
        )
    eta_u = 2.0 * lam if eta_u is None else eta_u
    eta_w = 10.0 / mu if eta_w is None else eta_w
    c_w = 1.0 / mu

    state_u = None
    state_w = None
    u = np.zeros_like(image)
    v = np.zeros_like(image)
    cost = 0.0
    outer = 0
    points: list[CyclePoint] = []
    events: list[HodgeEvent] = []
    last_targets: dict[str, np.ndarray] = {}

    while True:
        next_outer = outer + 1
        scheduled = (
            None
            if placement == "none"
            else _scheduled_side(
                next_outer,
                start=start,
                gap=gap,
                side=side,
                shot_limit=shot_limit,
            )
        )
        shoot_u = scheduled in ("u", "both")
        shoot_w = scheduled in ("w", "both")
        replace_u = placement == "replace" and shoot_u and \
            state_u is not None
        replace_w = placement == "replace" and shoot_w and \
            state_w is not None
        outer_cost = (
            (hodge_cost if replace_u else 1.0)
            + (hodge_cost if replace_w else 1.0)
            + (hodge_cost if placement == "add" and shoot_u else 0.0)
            + (hodge_cost if placement == "add" and shoot_w else 0.0)
        )
        if cost + outer_cost > max_cost + 1e-12:
            break

        outer = next_outer
        target_u = image - v
        if replace_u:
            u, alpha, gain = hodge_state_drop(
                target_u, lam, eta_u, u, state_u
            )
            cost += hodge_cost
        else:
            u, state_u = meyer.rof_sb(
                target_u,
                lam,
                eta=eta_u,
                state=state_u,
                sweeps=1,
            )
            cost += 1.0
            if placement == "add" and shoot_u:
                u, alpha, gain = hodge_state_drop(
                    target_u, lam, eta_u, u, state_u
                )
                cost += hodge_cost
            else:
                alpha = gain = 0.0
        if shoot_u:
            previous = last_targets.get("u")
            motion = (
                0.0
                if previous is None
                else float(
                    np.linalg.norm(target_u - previous)
                    / max(np.linalg.norm(target_u), 1e-30)
                )
            )
            events.append(HodgeEvent(outer, "u", alpha, gain, motion))
            last_targets["u"] = target_u.copy()

        target_w = image - u
        if replace_w:
            w, alpha, gain = hodge_state_drop(
                target_w, c_w, eta_w, state_w.u, state_w
            )
            cost += hodge_cost
        else:
            w, state_w = meyer.rof_sb(
                target_w,
                c_w,
                eta=eta_w,
                state=state_w,
                sweeps=1,
            )
            cost += 1.0
            if placement == "add" and shoot_w:
                w, alpha, gain = hodge_state_drop(
                    target_w, c_w, eta_w, w, state_w
                )
                cost += hodge_cost
            else:
                alpha = gain = 0.0
        if shoot_w:
            previous = last_targets.get("w")
            motion = (
                0.0
                if previous is None
                else float(
                    np.linalg.norm(target_w - previous)
                    / max(np.linalg.norm(target_w), 1e-30)
                )
            )
            events.append(HodgeEvent(outer, "w", alpha, gain, motion))
            last_targets["w"] = target_w.copy()

        v = target_w - w
        points.append(
            CyclePoint(
                outer=outer,
                cost=cost,
                u=u.copy(),
                v=v.copy(),
                events=len(events),
                accepted=sum(event.accepted for event in events),
            )
        )
    return points, events


def combined_error(
    point: CyclePoint,
    reference_u: np.ndarray,
    reference_v: np.ndarray,
    image: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.linalg.norm(point.u - reference_u) ** 2
            + np.linalg.norm(point.v - reference_v) ** 2
        )
        / max(np.linalg.norm(image), 1e-30)
    )


def point_at_cost(points: list[CyclePoint], budget: float) -> CyclePoint:
    eligible = [point for point in points if point.cost <= budget + 1e-12]
    if not eligible:
        raise ValueError("trajectory has no point within the requested budget")
    return eligible[-1]


def run_case(
    name: str,
    size: int,
    budgets: list[int],
    reference_cost: int,
    hodge_cost: float,
) -> None:
    image = fourier.tree_experiment._load_image(name, size)
    reference, _ = run_cycle(
        image, max_cost=reference_cost, placement="none"
    )
    reference_u = reference[-1].u
    reference_v = reference[-1].v
    maximum = max(budgets)
    baseline, _ = run_cycle(image, max_cost=maximum, placement="none")

    candidates = []
    # The broad pilot included add/replace on each side independently.
    # Texture-only schedules were dominated throughout; retain the four
    # meaningful families for the costed grid.
    for placement, side in (
        ("add", "u"),
        ("replace", "u"),
        ("replace", "alternate"),
        ("add", "both"),
    ):
        for start in (2, 4, 8):
            for gap in (1, 2, 4, 8, 16, 32):
                for shot_limit in (1, 2, 3, 4, 6, None):
                    points, events = run_cycle(
                        image,
                        max_cost=maximum,
                        placement=placement,
                        side=side,
                        start=start,
                        gap=gap,
                        shot_limit=shot_limit,
                        hodge_cost=hodge_cost,
                    )
                    candidates.append(
                        (
                            placement,
                            side,
                            start,
                            gap,
                            shot_limit,
                            points,
                            events,
                        )
                    )

    print(f"\n{name} {size}x{size}, Hodge price {hodge_cost:g} sweeps")
    for budget in budgets:
        base_point = point_at_cost(baseline, budget)
        base_error = combined_error(
            base_point, reference_u, reference_v, image
        )
        ranked = []
        for (
            placement,
            side,
            start,
            gap,
            shot_limit,
            points,
            events,
        ) in candidates:
            try:
                point = point_at_cost(points, budget)
            except ValueError:
                continue
            error = combined_error(point, reference_u, reference_v, image)
            used_events = [event for event in events
                           if event.outer <= point.outer]
            ranked.append(
                (
                    error / base_error,
                    placement,
                    side,
                    start,
                    gap,
                    shot_limit,
                    point,
                    used_events,
                    error,
                )
            )
        best = min(ranked, key=lambda item: item[0])
        (
            ratio,
            placement,
            side,
            start,
            gap,
            shot_limit,
            point,
            events,
            error,
        ) = best
        accepted = sum(event.accepted for event in events)
        reacquired = sum(
            event.accepted and event.target_motion > 0.0 for event in events
        )
        print(
            f"budget {budget:4d}: baseline outer {base_point.outer:3d} "
            f"error {base_error:.4e}; best {placement}/{side} "
            f"start {start} gap {gap} limit {shot_limit}, "
            f"outer {point.outer:3d}, "
            f"shots {len(events)}/{accepted} accepted "
            f"({reacquired} after motion), error {error:.4e}, "
            f"ratio {ratio:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[32, 64, 128, 256]
    )
    parser.add_argument("--reference-cost", type=int, default=2048)
    parser.add_argument("--hodge-cost", type=float, default=2.0)
    parser.add_argument(
        "--images", nargs="+", default=["cameraman", "synthetic"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.images:
        run_case(
            name,
            args.size,
            args.budgets,
            args.reference_cost,
            args.hodge_cost,
        )


if __name__ == "__main__":
    main()
