"""Hodge projection of local T-cap arrows onto global relative depth.

Each amodal port contributes two unit observations: cap region is in front of
each terminating-stem side.  Their least-squares projection onto a scalar
region potential is the exact gradient component of this directed evidence;
the remainder is cyclic/inconsistent depth evidence.  No region is selected
as foreground and the additive gauge is fixed independently per component.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def build_depth_hodge(
    ports: dict[str, np.ndarray],
    region_count: int,
    *,
    port_weight: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    count = len(ports["cap_region"])
    cap = np.repeat(np.asarray(ports["cap_region"], dtype=np.int32), 2)
    behind = np.column_stack((
        ports["left_region"], ports["right_region"])).ravel().astype(np.int32)
    row = np.repeat(np.arange(count, dtype=np.int32), 2)
    row = np.arange(2 * count, dtype=np.int32)
    incidence = sparse.csr_matrix((
        np.concatenate((np.ones(2 * count), -np.ones(2 * count))),
        (np.concatenate((row, row)), np.concatenate((cap, behind))),
    ), shape=(2 * count, region_count))
    if port_weight is None:
        edge_weight = np.ones(2 * count, dtype=np.float64)
    else:
        value = np.asarray(port_weight, dtype=np.float64)
        if value.shape != (count,):
            raise ValueError("port weight must have one value per T port")
        edge_weight = np.repeat(np.maximum(value, 0.0), 2)
    weighted = incidence.multiply(np.sqrt(edge_weight)[:, None])
    target = np.sqrt(edge_weight)
    laplacian = (weighted.T @ weighted).tocsr()
    rhs = np.asarray(weighted.T @ target).ravel()
    adjacency = sparse.csr_matrix((
        np.ones(4 * count),
        (
            np.concatenate((cap, behind)),
            np.concatenate((behind, cap)),
        ),
    ), shape=(region_count, region_count))
    _, component = connected_components(
        adjacency, directed=False, return_labels=True)
    potential = np.zeros(region_count, dtype=np.float64)
    for identifier in np.unique(component):
        member = np.flatnonzero(component == identifier)
        if len(member) <= 1:
            continue
        system = laplacian[member][:, member].toarray()
        # The rank-one mean gauge changes only the null direction.
        system += np.ones_like(system) / float(len(member))
        potential[member] = np.linalg.solve(system, rhs[member])
    predicted = np.asarray(incidence @ potential).ravel()
    residual = 1.0 - predicted
    port_first_drop = predicted[0::2]
    port_second_drop = predicted[1::2]
    port_agreement = 0.5 * (port_first_drop + port_second_drop)
    denominator = max(float(np.sum(edge_weight)), 1e-30)
    explained = 1.0 - float(np.sum(edge_weight * residual * residual)) / denominator
    return {
        "region_potential": potential,
        "edge_cap": cap,
        "edge_behind": behind,
        "edge_weight": edge_weight,
        "edge_predicted_drop": predicted,
        "edge_cycle_residual": residual,
        "port_first_drop": port_first_drop,
        "port_second_drop": port_second_drop,
        "port_agreement": port_agreement,
        "explained_fraction": explained,
    }


def summarize_depth_hodge(hodge: dict) -> dict:
    agreement = np.asarray(hodge["port_agreement"])
    residual = np.asarray(hodge["edge_cycle_residual"])
    return {
        "explained_fraction": float(hodge["explained_fraction"]),
        "positive_port_agreement": int(np.count_nonzero(agreement > 0.0)),
        "negative_port_agreement": int(np.count_nonzero(agreement < 0.0)),
        "port_agreement_quantiles": [
            float(value) for value in np.quantile(
                agreement, (0.0, 0.25, 0.5, 0.75, 1.0))
        ] if len(agreement) else [0.0] * 5,
        "cycle_residual_quantiles": [
            float(value) for value in np.quantile(
                residual, (0.0, 0.25, 0.5, 0.75, 1.0))
        ] if len(residual) else [0.0] * 5,
    }
