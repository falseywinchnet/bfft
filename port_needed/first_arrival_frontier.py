"""Local reductions on an irreversible first-arrival partition.

No runner-up field is constructed.  A boundary edge already contains the
needed stopping information: if pixel p was accepted by one source and q by
another, then

    T[p] + cost(p -> q) - T[q]

is the excess action with which p's front failed to purchase q.  Reducing
this directed collision slack by accepted label measures terminal pressure
without soft ownership or a second-best source.
"""

from __future__ import annotations

import numpy as np


def _bincount(labels, values, cells):
    return np.bincount(
        labels, weights=values, minlength=cells).astype(np.float64)


def first_arrival_frontier(
    labels: np.ndarray,
    distance: np.ndarray,
    mxx: np.ndarray,
    mxy: np.ndarray,
    myy: np.ndarray,
    measure: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return accepted-mass and directed terminal-energy reductions."""
    labels = np.asarray(labels, dtype=np.int32)
    distance = np.asarray(distance, dtype=np.float64)
    mxx = np.asarray(mxx, dtype=np.float64)
    mxy = np.asarray(mxy, dtype=np.float64)
    myy = np.asarray(myy, dtype=np.float64)
    if not (
        labels.shape == distance.shape == mxx.shape == mxy.shape == myy.shape
    ):
        raise ValueError("all first-arrival fields must have identical shape")
    if np.any(labels < 0) or np.any(~np.isfinite(distance)):
        raise ValueError("first-arrival partition must be complete")
    density = (
        np.ones(labels.shape, dtype=np.float64)
        if measure is None
        else np.maximum(np.asarray(measure, dtype=np.float64), 0.0)
    )
    cells = int(np.max(labels)) + 1
    flat_label = labels.ravel()
    flat_density = density.ravel()
    flat_distance = distance.ravel()
    mass = _bincount(flat_label, flat_density, cells)
    accepted_action = _bincount(
        flat_label, flat_density * flat_distance, cells)
    maximum_action = np.zeros(cells, dtype=np.float64)
    np.maximum.at(maximum_action, flat_label, flat_distance)

    terminal_mask = np.zeros(labels.shape, dtype=bool)
    terminal_action_sum = np.zeros(cells, dtype=np.float64)
    terminal_slack_sum = np.zeros(cells, dtype=np.float64)
    terminal_slack_max = np.zeros(cells, dtype=np.float64)
    terminal_fraction_sum = np.zeros(cells, dtype=np.float64)
    terminal_fraction_min = np.ones(cells, dtype=np.float64)
    terminal_fraction_max = np.zeros(cells, dtype=np.float64)
    terminal_edges = np.zeros(cells, dtype=np.int64)
    slack_map = np.zeros(labels.shape, dtype=np.float64)
    fraction_sum_map = np.zeros(labels.shape, dtype=np.float64)
    fraction_count_map = np.zeros(labels.shape, dtype=np.int32)

    # Each undirected interface is visited once, then reduced in both
    # directions so every cell receives its own failed-purchase action.
    for dy, dx in ((0, 1), (1, 0)):
        ys = slice(0, labels.shape[0] - dy)
        xs = slice(0, labels.shape[1] - dx)
        yd = slice(dy, labels.shape[0])
        xd = slice(dx, labels.shape[1])
        source_label = labels[ys, xs]
        target_label = labels[yd, xd]
        crossing = source_label != target_label
        if not np.any(crossing):
            continue
        a = 0.5 * (mxx[ys, xs] + mxx[yd, xd])
        b = 0.5 * (mxy[ys, xs] + mxy[yd, xd])
        c = 0.5 * (myy[ys, xs] + myy[yd, xd])
        step = np.sqrt(np.maximum(
            dx * dx * a + 2.0 * dx * dy * b + dy * dy * c,
            1e-30,
        ))
        source_distance = distance[ys, xs]
        target_distance = distance[yd, xd]
        source_slack = np.maximum(
            source_distance + step - target_distance, 0.0)
        target_slack = np.maximum(
            target_distance + step - source_distance, 0.0)
        # Linear interpolation of both accepted fronts along the interface
        # edge. This is the subpixel point where their arrival actions meet.
        source_fraction = np.clip(
            (target_distance + step - source_distance)
            / np.maximum(2.0 * step, 1e-30),
            0.0,
            1.0,
        )
        target_fraction = 1.0 - source_fraction

        for cell_label, action, slack, fraction, mask_slice in (
            (
                source_label,
                source_distance,
                source_slack,
                source_fraction,
                (ys, xs),
            ),
            (
                target_label,
                target_distance,
                target_slack,
                target_fraction,
                (yd, xd),
            ),
        ):
            selected_label = cell_label[crossing]
            selected_action = action[crossing]
            selected_slack = slack[crossing]
            selected_fraction = fraction[crossing]
            terminal_action_sum += _bincount(
                selected_label, selected_action, cells)
            terminal_slack_sum += _bincount(
                selected_label, selected_slack, cells)
            terminal_fraction_sum += _bincount(
                selected_label, selected_fraction, cells)
            terminal_edges += np.bincount(
                selected_label, minlength=cells).astype(np.int64)
            np.maximum.at(
                terminal_slack_max, selected_label, selected_slack)
            np.minimum.at(
                terminal_fraction_min, selected_label, selected_fraction)
            np.maximum.at(
                terminal_fraction_max, selected_label, selected_fraction)
            view = terminal_mask[mask_slice]
            view[crossing] = True
            slack_view = slack_map[mask_slice]
            slack_view[crossing] = np.maximum(
                slack_view[crossing], selected_slack)
            fraction_sum_view = fraction_sum_map[mask_slice]
            fraction_sum_view[crossing] += selected_fraction
            fraction_count_view = fraction_count_map[mask_slice]
            fraction_count_view[crossing] += 1

    safe_mass = np.maximum(mass, 1e-30)
    safe_edges = np.maximum(terminal_edges, 1)
    terminal_fraction_min[terminal_edges == 0] = 0.5
    terminal_fraction_max[terminal_edges == 0] = 0.5
    fraction_map = np.full(labels.shape, 0.5, dtype=np.float64)
    boundary_pixel = fraction_count_map > 0
    fraction_map[boundary_pixel] = (
        fraction_sum_map[boundary_pixel]
        / fraction_count_map[boundary_pixel]
    )
    return {
        "mass": mass,
        "mean_accepted_action": accepted_action / safe_mass,
        "maximum_accepted_action": maximum_action,
        "terminal_edges": terminal_edges,
        "mean_terminal_action": terminal_action_sum / safe_edges,
        "mean_collision_slack": terminal_slack_sum / safe_edges,
        "maximum_collision_slack": terminal_slack_max,
        "mean_terminal_fraction": terminal_fraction_sum / safe_edges,
        "minimum_terminal_fraction": terminal_fraction_min,
        "maximum_terminal_fraction": terminal_fraction_max,
        "terminal_mask": terminal_mask,
        "collision_slack_map": slack_map,
        "terminal_fraction_map": fraction_map,
    }
