"""Lift raw T-junction and focus evidence onto one-sided contour runs.

The returned arrays are a coordinate chart, not a foreground decision.  A
component keeps its exact owner region, cap-junction persistence, signed
content continuation, and contrast-normalized boundary-focus ownership as
separate observables.
"""

from __future__ import annotations

import numpy as np


def _incidence_lookup(bundle: dict) -> dict[tuple[int, int], int]:
    incidence = bundle["incidence"]
    return {
        (int(arc), int(region)): identifier
        for identifier, (arc, region) in enumerate(zip(
            incidence["arc"], incidence["region"]))
    }


def build_depth_contour_transport(
    complex_: dict,
    bundle: dict,
    contour: dict,
    junction_depth: dict,
    focus_interfaces: dict,
) -> dict[str, np.ndarray]:
    """Aggregate independent asymmetric observations without thresholding."""
    incidence = bundle["incidence"]
    component = np.asarray(
        contour["incidence_component"], dtype=np.int32)
    component_count = int(contour["component_count"])
    lookup = _incidence_lookup(bundle)

    # Orient the interface-focus margin toward each one-sided incidence.
    arc = np.asarray(incidence["arc"], dtype=np.int32)
    region = np.asarray(incidence["region"], dtype=np.int32)
    topology_arc = complex_["topology"]["arc"]
    first = np.asarray(topology_arc["cell_first"], dtype=np.int32)
    margin = np.asarray(
        focus_interfaces["first_match_margin"], dtype=np.float64)
    reliability = np.asarray(
        focus_interfaces["reliability"], dtype=np.float64)
    oriented_margin = np.where(region == first[arc], margin[arc], -margin[arc])
    incidence_length = np.asarray(incidence["length"], dtype=np.float64)
    focus_mass = incidence_length * reliability[arc]
    component_focus_mass = np.bincount(
        component, weights=focus_mass, minlength=component_count)
    component_focus_margin = np.bincount(
        component,
        weights=focus_mass * oriented_margin,
        minlength=component_count,
    ) / np.maximum(component_focus_mass, 1e-30)
    component_focus_reliability = np.bincount(
        component,
        weights=incidence_length * reliability[arc],
        minlength=component_count,
    ) / np.maximum(
        np.bincount(
            component, weights=incidence_length, minlength=component_count),
        1e-30,
    )

    cap_junction_count = np.zeros(component_count, dtype=np.int32)
    cap_record_count = np.zeros(component_count, dtype=np.int32)
    cap_tangent_sum = np.zeros(component_count, dtype=np.float64)
    transition_names = sorted(
        name for name in junction_depth
        if name.startswith("cap_") and name.endswith("_transition_cosine")
    )
    transition_sum = {
        name: np.zeros(component_count, dtype=np.float64)
        for name in transition_names
    }
    component_junctions: list[set[int]] = [
        set() for _ in range(component_count)
    ]
    record_component_offset = [0]
    record_component = []
    cap_arc_offset = junction_depth["cap_arc_offset"]
    for record, (junction, owner) in enumerate(zip(
        junction_depth["junction"], junction_depth["cap_region"]
    )):
        start = int(cap_arc_offset[record])
        stop = int(cap_arc_offset[record + 1])
        components = sorted({
            int(component[lookup[(int(arc_identifier), int(owner))]])
            for arc_identifier in junction_depth["cap_arc"][start:stop]
            if (int(arc_identifier), int(owner)) in lookup
        })
        record_component.extend(components)
        record_component_offset.append(len(record_component))
        for identifier in components:
            cap_record_count[identifier] += 1
            component_junctions[identifier].add(int(junction))
            cap_tangent_sum[identifier] += float(
                junction_depth["cap_tangent_continuation"][record])
            for name in transition_names:
                transition_sum[name][identifier] += float(
                    junction_depth[name][record])
    for identifier, values in enumerate(component_junctions):
        cap_junction_count[identifier] = len(values)

    denominator = np.maximum(cap_record_count, 1)
    result = {
        "record_component_offset": np.asarray(
            record_component_offset, dtype=np.int64),
        "record_component": np.asarray(record_component, dtype=np.int32),
        "component_cap_record_count": cap_record_count,
        "component_cap_junction_count": cap_junction_count,
        "component_cap_tangent_continuation": cap_tangent_sum / denominator,
        "component_focus_match_margin": component_focus_margin,
        "component_focus_reliability": component_focus_reliability,
        "component_focus_evidence_mass": component_focus_mass,
        "arc_focus_match_margin_first": margin,
        "arc_focus_reliability": reliability,
    }
    result.update({
        f"component_{name}": values / denominator
        for name, values in transition_sum.items()
    })
    return result


def summarize_depth_contour_transport(
    depth: dict,
    contour: dict,
) -> dict:
    cap_count = depth["component_cap_junction_count"]
    reliability = depth["component_focus_reliability"]
    both = (cap_count > 0) & (reliability > 0.0)
    persistent = cap_count > 1
    return {
        "contour_components": int(contour["component_count"]),
        "components_with_t_cap": int(np.count_nonzero(cap_count > 0)),
        "components_with_repeated_t_caps": int(np.count_nonzero(persistent)),
        "maximum_t_caps_on_component": int(np.max(cap_count, initial=0)),
        "components_with_focus_evidence": int(np.count_nonzero(
            reliability > 0.0)),
        "components_with_t_cap_and_focus": int(np.count_nonzero(both)),
        "focus_reliability_quantiles": (
            [float(value) for value in np.quantile(
                reliability[reliability > 0.0],
                (0.0, 0.25, 0.5, 0.75, 1.0))]
            if np.any(reliability > 0.0) else [0.0] * 5
        ),
    }
