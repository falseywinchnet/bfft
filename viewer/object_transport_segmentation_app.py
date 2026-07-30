#!/usr/bin/env python3
"""DearPyGui laboratory for emergent objects on transport-cell support."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "viewer", ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import dearpygui.dearpygui as dpg  # noqa: E402
import gallery  # noqa: E402
from experiments.transport_object_support import (  # noqa: E402
    ObjectSupportConfig,
    build_cell_interface_graph,
    infer_object_support,
)
from experiments.sparse_finsler_elastica import (  # noqa: E402
    SparseElasticaConfig,
    build_sparse_elastica_graph,
    elastica_common_surround_relations,
    finsler_saliency_closing,
    intrinsic_arc_speed,
    label_intrinsic_arc_parts,
    project_pixel_field_to_arcs,
    render_elastica_arcs,
)
from experiments.contour_object_hierarchy import (  # noqa: E402
    ContourHierarchyConfig,
    infer_contour_object_hierarchy,
)
from experiments.object_hierarchy_diagnostics import (  # noqa: E402
    analyze_object_hierarchy,
    format_selection_report,
    selection_report,
)
from experiments.embedded_interface_topology import (  # noqa: E402
    build_embedded_interface_topology,
    render_arc_ids,
)
from experiments.transport_object_hierarchy import (  # noqa: E402
    ParentHierarchyConfig,
    infer_parent_objects,
)
from experiments.transport_border_ownership import (  # noqa: E402
    infer_transport_border_ownership,
)
from experiments.transport_scene_assembly import (  # noqa: E402
    SceneAssemblyConfig,
    infer_scene_assemblies,
)
from experiments.transport_relation_forensics import (  # noqa: E402
    transport_anchor_cell_fields,
    transport_relation_forensics,
)
from experiments.transport_graph_scattering import (  # noqa: E402
    scattering_anchor_field,
    transport_graph_scattering,
)
from experiments.transport_edge_relation import (  # noqa: E402
    aggregate_signed_relations,
    signed_relation_field,
    transport_edge_relation,
)
from experiments.transport_focus_forensics import (  # noqa: E402
    focus_likeness,
    transport_focus_forensics,
    transport_focus_interfaces,
)
from port_needed import SegmentingConfig, build_segmenting_representation  # noqa: E402
from voronoi_itd import (  # noqa: E402
    VoronoiITDConfig,
    extract_intrinsic_voronoi_support,
)
from segmenting_veroni_app import (  # noqa: E402
    _fit_rgb,
    _rgb,
    colour_map,
    overlay_boundaries,
    overlay_motion,
    overlay_sites,
    site_ids,
)


PANEL = 650
SOURCE = "object_source_texture"
RESULT = "object_result_texture"
VIEWS = (
    "Anchor cell colour likeness",
    "Anchor cell transport-action likeness",
    "Anchor cell metric-tensor likeness",
    "Anchor cell action + metric likeness",
    "Anchor cell full-state likeness",
    "Anchor cell local graph-scattering likeness",
    "Anchor cell multiscale graph-scattering likeness",
    "Anchor cell centered edge relation",
    "Anchor cell focus likeness",
    "Relative defocus radius",
    "Focus evidence confidence",
    "Edge-only defocus radius",
    "Edge-only focus confidence",
    "Texture defocus radius",
    "Texture focus confidence",
    "Texture focus model fit",
    "Texture focus identifiability",
    "Chromatic focus sign",
    "Chromatic focus confidence",
    "Chromatic focus spread",
    "Autofocus cell selection score",
    "Autofocus-adjusted peak persistence",
    "BFFT cross-scale persistence",
    "BFFT fine-scale null activity",
    "BFFT persistence ratio",
    "Transport-cell defocus",
    "Transport-cell focus evidence coverage",
    "Focus boundary-to-side match",
    "Focus boundary ownership reliability",
    "Soft object IDs",
    "Soft object IDs + object boundaries",
    "Hard object IDs",
    "Hard object IDs + object boundaries",
    "Intrinsic Voronoi support IDs",
    "Intrinsic Finsler contour evidence",
    "Finsler two-sided completion",
    "Finsler completion lift",
    "Finsler lift on canonical interfaces",
    "Finsler common-surround collisions",
    "Intrinsic boundary alignment",
    "Closed-contour barrier",
    "Contour-ultrametric soft IDs",
    "Contour-ultrametric parent IDs",
    "Contour parents over reconstruction",
    "Contour leaf merge altitude",
    "Parent object IDs",
    "Parent IDs over reconstruction",
    "Compound assembly IDs",
    "Compound assembly over reconstruction",
    "Compound assembly proposals",
    "Exterior-reachable substrate",
    "Bounded support basins",
    "Frame substrate seeds",
    "Anchor colour similarity",
    "Anchor transport-state similarity",
    "Anchor transport-action similarity",
    "Anchor metric-tensor similarity",
    "Anchor boundary-role transport similarity",
    "Anchor full relational similarity",
    "Anchor part centered edge relation",
    "Parent junction proposals",
    "Enclosed seam attachment proposals",
    "Frame exposure / substrate evidence",
    "T-junction depth order",
    "T-implied contour sides",
    "Focus-implied contour sides",
    "Combined contour ownership sides",
    "T-implied frontness",
    "Transport-depth extrapolation",
    "Accepted parent relations",
    "Surround completion proposals",
    "Accepted surround completions",
    "Connected material atoms",
    "Connected support fragment IDs",
    "Intra-site topology cuts",
    "Unconstrained path disagreement",
    "Embedded interface arc IDs",
    "Embedded interface junctions",
    "Object IDs over reconstruction",
    "Object confidence",
    "Soft winner weight",
    "Distance to hard-object interface",
    "Object saddle ambiguity",
    "Object waterline",
    "Distance from winning core",
    "Boundary-distance core altitude",
    "Object connected-material quotient",
    "Object material-component count",
    "Material single-object quotient",
    "Material object-count",
    "Selected highpoints",
    "All provisional highpoints",
    "Seed strength",
    "Boundary enclosure",
    "Fused interface barrier",
    "Raw interface barrier",
    "Short-contact reliability",
    "Direct target jump",
    "Region colour similarity failure",
    "Cartoon interface jump",
    "Cartoon contribution to barrier",
    "Unresolved cartoon interfaces",
    "Object-cut cartoon interfaces",
    "Transport glass jump",
    "Transport-normal action",
    "Transport support-signature jump",
    "Focus discontinuity",
    "Unanchored focus-scale difference",
    "Focus interface reliability",
    "Latent support frontier",
    "Direct visual witness",
    "Additive barrier control",
    "Anchored barrier control",
    "Cross-scale null reliability",
    "Certified material joins",
    "Canonical boundary confidence",
    "Canonical Site IDs",
    "Canonical reconstruction",
    "Original",
)


class State:
    def __init__(self):
        self.image = None
        self.name = "(none)"
        self.rgb = None
        self.result = None
        self.objects = None
        self.graph = None
        self.intrinsic_support = None
        self.finsler_support = None
        self.busy = False
        self.status = "Choose an image, then build cells and object support."
        self.lock = threading.Lock()
        self.buffers: dict[str, np.ndarray] = {}
        self.texture_shapes: dict[str, tuple[int, int]] = {}
        self.texture_items: dict[str, str | int] = {}
        self.texture_generation: dict[str, int] = {}
        self.anchor_part = 0
        self.anchor_cell = 0
        self.anchor_cell_fields = None


S = State()


def alloc_texture(tag: str, height: int, width: int) -> None:
    old_item = S.texture_items.get(tag)
    if old_item is None:
        texture_item = tag
        generation = 0
    else:
        generation = S.texture_generation.get(tag, 0) + 1
        # DearPyGui can retain a deleted string alias until a later frame.
        # A generated integer item ID has no alias lifecycle and therefore
        # cannot collide during rapid gallery size changes.
        texture_item = dpg.generate_uuid()
    S.buffers[tag] = np.ones(height * width * 4, dtype=np.float32)
    S.texture_shapes[tag] = (height, width)
    S.texture_items[tag] = texture_item
    S.texture_generation[tag] = generation
    with dpg.texture_registry():
        dpg.add_raw_texture(
            width,
            height,
            S.buffers[tag],
            tag=texture_item,
            format=dpg.mvFormat_Float_rgba,
        )
    scale = PANEL / max(height, width)
    item = "object_source_image" if tag == SOURCE else "object_result_image"
    if dpg.does_item_exist(item):
        dpg.configure_item(
            item,
            texture_tag=texture_item,
            width=max(1, int(width * scale)),
            height=max(1, int(height * scale)),
        )
    # The old item is deleted only after the image points at the replacement.
    if (
        old_item is not None
        and old_item != texture_item
        and dpg.does_item_exist(old_item)
    ):
        dpg.delete_item(old_item)


def push_texture(tag: str, image: np.ndarray) -> None:
    display = _rgb(image).astype(np.float32)
    if max(display.shape[:2]) > PANEL:
        display = _fit_rgb(display, PANEL).astype(np.float32)
    height, width = display.shape[:2]
    required = height * width * 4
    if (
        S.texture_shapes.get(tag) != (height, width)
        or tag not in S.buffers
        or S.buffers[tag].size != required
    ):
        alloc_texture(tag, height, width)
    pixels = S.buffers[tag].reshape(height, width, 4)
    pixels[..., :3] = display
    pixels[..., 3] = 1.0
    dpg.set_value(S.texture_items[tag], S.buffers[tag])


def _blend_object_boundaries(
    image: np.ndarray,
    object_labels: np.ndarray,
) -> np.ndarray:
    return overlay_boundaries(image, object_labels)


_CURRENT = object()


def current_view(
    rgb: np.ndarray | object = _CURRENT,
    result: dict | None | object = _CURRENT,
    objects: dict | None | object = _CURRENT,
) -> np.ndarray:
    if rgb is _CURRENT or result is _CURRENT or objects is _CURRENT:
        with S.lock:
            rgb, result, objects = S.rgb, S.result, S.objects
    if rgb is None:
        return np.full((8, 8, 3), 0.08)
    view = dpg.get_value("object_view")
    if view == "Original" or result is None:
        return rgb
    reconstruction = result["record"]["rgb"]
    if view == "Canonical reconstruction" or objects is None:
        return reconstruction
    labels = objects["graph"]["labels"]
    source_labels = result["labels"]
    geometry = result["geometry"]
    if view == "BFFT cross-scale persistence":
        return colour_map(np.log1p(np.asarray(
            geometry["persistent_activity"], dtype=np.float64)))
    if view == "BFFT fine-scale null activity":
        return colour_map(np.log1p(np.asarray(
            geometry["local_null_activity"], dtype=np.float64)))
    if view == "BFFT persistence ratio":
        return colour_map(geometry["null_confidence"])
    forensics = objects.get("relation_forensics")
    cell_forensic_views = {
        "Anchor cell colour likeness": "colour",
        "Anchor cell transport-action likeness": "action",
        "Anchor cell metric-tensor likeness": "metric",
        "Anchor cell action + metric likeness": "action_metric",
        "Anchor cell full-state likeness": "full_state",
    }
    if S.anchor_cell_fields is not None and view in cell_forensic_views:
        field = S.anchor_cell_fields[cell_forensic_views[view]]
        return colour_map(field[labels])
    scattering = objects.get("graph_scattering")
    if scattering is not None:
        if view == "Anchor cell local graph-scattering likeness":
            return colour_map(scattering_anchor_field(
                scattering, S.anchor_cell, maximum_scale=1)[labels])
        if view == "Anchor cell multiscale graph-scattering likeness":
            return colour_map(scattering_anchor_field(
                scattering, S.anchor_cell)[labels])
    edge_relation = objects.get("edge_relation")
    if (
        edge_relation is not None
        and view == "Anchor cell centered edge relation"
    ):
        signed = signed_relation_field(edge_relation, S.anchor_cell)
        return colour_map((0.5 + 0.5 * signed)[labels])
    focus = objects.get("focus_forensics")
    if focus is not None:
        if view == "Anchor cell focus likeness":
            return colour_map(
                focus_likeness(focus, S.anchor_cell)[labels])
        if view == "Relative defocus radius":
            # Confidence is intentionally shown separately. Flat interiors
            # contain no focus observation and must not masquerade as sharp.
            return colour_map(focus["defocus_radius"])
        if view == "Focus evidence confidence":
            return colour_map(focus["confidence"])
        if view == "Edge-only defocus radius":
            return colour_map(focus["edge_defocus_radius"])
        if view == "Edge-only focus confidence":
            return colour_map(focus["edge_confidence"])
        if view == "Texture defocus radius":
            return colour_map(focus["texture_blur_sigma"])
        if view == "Texture focus confidence":
            return colour_map(focus["texture_focus_confidence"])
        if view == "Texture focus model fit":
            return colour_map(focus["texture_fit_r2"])
        if view == "Texture focus identifiability":
            return colour_map(focus["texture_curvature_gain"])
        if view == "Chromatic focus sign":
            signed = np.asarray(focus["chromatic_focus_sign"])
            nonzero = np.abs(signed[
                np.asarray(focus["chromatic_focus_confidence"]) > 0])
            scale = (
                float(np.percentile(nonzero, 90.0))
                if nonzero.size else 1.0
            )
            return colour_map(
                0.5 + 0.5 * np.tanh(signed / max(scale, 1e-12)))
        if view == "Chromatic focus confidence":
            return colour_map(focus["chromatic_focus_confidence"])
        if view == "Chromatic focus spread":
            return colour_map(focus["chromatic_focus_spread"])
        if view == "Autofocus cell selection score":
            return colour_map(objects["autofocus_cell_score"][labels])
        if view == "Autofocus-adjusted peak persistence":
            return colour_map(
                objects["autofocus_selection_prominence"][labels])
        if view == "Transport-cell defocus":
            return colour_map(
                focus["cell_effective_scale"][labels])
        if view == "Transport-cell focus evidence coverage":
            return colour_map(
                focus["cell_evidence_coverage"][labels])
        interface_focus = focus.get("interface")
        if interface_focus is not None and view in (
            "Focus boundary-to-side match",
            "Focus boundary ownership reliability",
        ):
            arc_labels = objects["embedded_arc_labels"]
            raster = np.zeros(labels.shape, dtype=np.float64)
            present = arc_labels > 0
            arc = arc_labels[present] - 1
            if view == "Focus boundary-to-side match":
                margin = interface_focus["first_match_margin"]
                nonzero = np.abs(margin[interface_focus["reliability"] > 0])
                scale = (
                    float(np.percentile(nonzero, 80.0))
                    if nonzero.size else 1.0
                )
                raster[present] = (
                    0.5
                    + 0.5
                    * np.tanh(margin[arc] / max(scale, 1e-12))
                    * interface_focus["reliability"][arc]
                )
            else:
                raster[present] = interface_focus["reliability"][arc]
            return colour_map(raster)
    forensic_views = {
        "Anchor colour similarity": "colour_similarity",
        "Anchor transport-state similarity": "state_similarity",
        "Anchor transport-action similarity": "action_similarity",
        "Anchor metric-tensor similarity": "metric_similarity",
        "Anchor boundary-role transport similarity":
            "boundary_transport_similarity",
        "Anchor full relational similarity": "boundary_full_similarity",
    }
    if forensics is not None and view in forensic_views:
        matrix = forensics[forensic_views[view]]
        anchor = int(np.clip(S.anchor_part, 0, len(matrix) - 1))
        return colour_map(matrix[anchor][objects["object_labels"]])
    if (
        edge_relation is not None
        and view == "Anchor part centered edge relation"
    ):
        matrix = objects["part_edge_relation"]
        anchor = int(np.clip(S.anchor_part, 0, len(matrix) - 1))
        return colour_map(
            (0.5 + 0.5 * matrix[anchor])[objects["object_labels"]])
    object_labels = objects["object_labels"]
    if view == "Soft object IDs":
        return objects["soft_ids"]
    if view == "Soft object IDs + object boundaries":
        return _blend_object_boundaries(
            objects["soft_ids"], object_labels)
    if view == "Hard object IDs":
        return objects["hard_ids"]
    if view == "Hard object IDs + object boundaries":
        return _blend_object_boundaries(
            objects["hard_ids"], object_labels)
    intrinsic = objects.get("intrinsic_support")
    if view == "Intrinsic Voronoi support IDs":
        return (
            np.zeros_like(reconstruction)
            if intrinsic is None
            else site_ids(intrinsic.owner)
        )
    finsler = objects.get("finsler_support")
    if view == "Intrinsic Finsler contour evidence":
        return (
            np.zeros_like(reconstruction)
            if finsler is None
            else colour_map(finsler["saliency_map"])
        )
    if view == "Finsler two-sided completion":
        return (
            np.zeros_like(reconstruction)
            if finsler is None
            else colour_map(finsler["completed_map"])
        )
    if view == "Finsler completion lift":
        return (
            np.zeros_like(reconstruction)
            if finsler is None
            else colour_map(finsler["lift_map"])
        )
    if view == "Finsler common-surround collisions":
        if finsler is None or "relations" not in finsler:
            return np.zeros_like(reconstruction)
        relation = finsler["relations"]
        tau = max(float(finsler["closing"]["tau"]), 1e-12)
        score = (
            np.asarray(relation["strength"], dtype=np.float64)
            * np.exp(
                -np.asarray(relation["action"], dtype=np.float64) / tau)
        )
        height, width = finsler["topology"]["shape"]
        stride = width + 1
        vertex = np.asarray(relation["vertex"], dtype=np.int64)
        x = vertex % stride
        y = vertex // stride
        collision = np.zeros((height, width), dtype=np.float64)
        for dy, dx in (
            (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)
        ):
            py, px = y + dy, x + dx
            valid = (
                (py >= 0) & (py < height)
                & (px >= 0) & (px < width)
            )
            np.maximum.at(
                collision,
                (py[valid], px[valid]),
                score[valid],
            )
        return colour_map(collision)
    if view == "Finsler lift on canonical interfaces":
        return colour_map(
            objects["interface_maps"]["finsler_contour_completion"])
    if view == "Intrinsic boundary alignment":
        return colour_map(
            objects["interface_maps"]["intrinsic_boundary_alignment"])
    if view == "Closed-contour barrier":
        return colour_map(objects["interface_maps"]["cycle_barrier"])
    contour = objects.get("contour_hierarchy")
    if contour is not None:
        if view == "Contour-ultrametric soft IDs":
            return overlay_boundaries(
                contour["soft_ids"],
                object_labels,
            )
        if view == "Contour-ultrametric parent IDs":
            return overlay_boundaries(
                contour["parent_ids"],
                contour["parent_labels"],
            )
        if view == "Contour parents over reconstruction":
            coloured = (
                0.58 * reconstruction + 0.42 * contour["parent_ids"])
            return overlay_boundaries(
                coloured, contour["parent_labels"])
        if view == "Contour leaf merge altitude":
            return colour_map(contour["leaf_merge_altitude"])
    parent = objects.get("parent_hierarchy")
    assembly = objects.get("scene_assembly")
    if assembly is not None:
        if view == "Compound assembly IDs":
            return overlay_boundaries(
                assembly["assembly_ids"],
                assembly["assembly_labels"],
            )
        if view == "Compound assembly over reconstruction":
            coloured = (
                0.58 * reconstruction + 0.42 * assembly["assembly_ids"])
            return overlay_boundaries(
                coloured, assembly["assembly_labels"])
        if view == "Compound assembly proposals":
            relation = assembly["relations"]
            accepted = assembly["accepted"]
            means = objects["diagnostics"]["objects"]
            height, width = labels.shape
            centers = np.column_stack((
                means["centroid_x"] / max(width, 1),
                means["centroid_y"] / max(height, 1),
            ))
            return overlay_motion(
                reconstruction,
                centers[relation["first"][accepted]],
                centers[relation["second"][accepted]],
            )
        exterior = assembly["exterior_reachability"]
        if view == "Exterior-reachable substrate":
            cell = exterior["cell_is_exterior"]
            return colour_map(
                cell[objects["graph"]["labels"]].astype(np.float64))
        if view == "Bounded support basins":
            return exterior["basin_ids"]
        if view == "Frame substrate seeds":
            return colour_map(
                exterior["frame_seed_map"].astype(np.float64))
    if parent is not None:
        if view == "Parent object IDs":
            return overlay_boundaries(
                parent["parent_ids"],
                parent["parent_labels"],
            )
        if view == "Parent IDs over reconstruction":
            coloured = (
                0.58 * reconstruction + 0.42 * parent["parent_ids"])
            return overlay_boundaries(
                coloured, parent["parent_labels"])
        if view == "Parent junction proposals":
            topology = parent["topology"]
            relation = parent["junction_relations"]
            junction_id = relation["junction"]
            height, width = labels.shape
            centers = np.column_stack((
                topology["junction"]["x"][junction_id] / max(width, 1),
                topology["junction"]["y"][junction_id] / max(height, 1),
            ))
            return overlay_sites(reconstruction, centers)
        if view == "Enclosed seam attachment proposals":
            relation = parent["enclosed_seam_relations"]
            means = objects["diagnostics"]["objects"]
            height, width = labels.shape
            centers = np.column_stack((
                means["centroid_x"] / max(width, 1),
                means["centroid_y"] / max(height, 1),
            ))
            return overlay_motion(
                reconstruction,
                centers[relation["first"]],
                centers[relation["second"]],
            )
        if view == "Frame exposure / substrate evidence":
            exposure = parent["frame_geometry"]["frame_exposure"]
            return colour_map(exposure[parent["topology_part_labels"]])
        if view == "T-junction depth order":
            relation = parent["junction_relations"]
            means = objects["diagnostics"]["objects"]
            height, width = labels.shape
            centers = np.column_stack((
                means["centroid_x"] / max(width, 1),
                means["centroid_y"] / max(height, 1),
            ))
            front = np.concatenate((
                centers[relation["surround"]],
                centers[relation["surround"]],
            ), axis=0)
            behind = np.concatenate((
                centers[relation["first"]],
                centers[relation["second"]],
            ), axis=0)
            return overlay_motion(reconstruction, front, behind)
        ownership = objects.get("border_ownership")
        if ownership is not None and view in (
            "T-implied contour sides",
            "Focus-implied contour sides",
            "Combined contour ownership sides",
        ):
            maps = {
                "T-implied contour sides": ownership["junction_arc_maps"],
                "Focus-implied contour sides": ownership["focus_arc_maps"],
                "Combined contour ownership sides": ownership["arc_maps"],
            }[view]
            out = 0.22 * reconstruction
            front = maps["front_boundary"] > 0.0
            back = maps["back_boundary"] > 0.0
            out[front] = np.column_stack((
                np.zeros(np.count_nonzero(front)),
                0.35 + 0.65 * maps["front_boundary"][front],
                np.ones(np.count_nonzero(front)),
            ))
            out[back] = np.column_stack((
                np.ones(np.count_nonzero(back)),
                np.full(np.count_nonzero(back), 0.12),
                0.25 + 0.45 * maps["back_boundary"][back],
            ))
            return out
        if ownership is not None and view == "T-implied frontness":
            return colour_map(np.clip(
                0.5 + 0.5 * ownership["observed_frontness"],
                0.0,
                1.0,
            ))
        if ownership is not None and view == "Transport-depth extrapolation":
            value = np.asarray(
                ownership["support_frontness"], dtype=np.float64)
            nonzero = np.abs(value[np.abs(value) > 0.0])
            scale = (
                float(np.percentile(nonzero, 80.0))
                if nonzero.size else 1.0
            )
            return colour_map(
                0.5 + 0.5 * np.tanh(value / max(scale, 1e-12)))
        if view == "Accepted parent relations":
            accepted = parent["accepted_relations"]
            means = objects["diagnostics"]["objects"]
            height, width = labels.shape
            centers = np.column_stack((
                means["centroid_x"] / max(width, 1),
                means["centroid_y"] / max(height, 1),
            ))
            return overlay_motion(
                reconstruction,
                centers[accepted["first"]],
                centers[accepted["second"]],
            )
        if view in (
            "Surround completion proposals",
            "Accepted surround completions",
        ):
            completion = parent["completion_relations"]
            if view == "Accepted surround completions":
                index = parent["accepted_completion_indices"]
            else:
                index = np.arange(len(completion["score"]))
            means = objects["diagnostics"]["objects"]
            height, width = labels.shape
            centers = np.column_stack((
                means["centroid_x"] / max(width, 1),
                means["centroid_y"] / max(height, 1),
            ))
            return overlay_motion(
                reconstruction,
                centers[completion["part_first"][index]],
                centers[completion["part_second"][index]],
            )
    if view == "Connected material atoms":
        return overlay_boundaries(
            objects["material_atom_ids"],
            objects["material_atom_labels"],
        )
    if view == "Connected support fragment IDs":
        return site_ids(labels)
    if view == "Intra-site topology cuts":
        source = np.asarray(source_labels, dtype=np.int32)
        fragment = np.asarray(labels, dtype=np.int32)
        cut = np.zeros(fragment.shape, dtype=bool)
        horizontal = (
            (source[:, :-1] == source[:, 1:])
            & (fragment[:, :-1] != fragment[:, 1:])
        )
        vertical = (
            (source[:-1] == source[1:])
            & (fragment[:-1] != fragment[1:])
        )
        cut[:, :-1] |= horizontal
        cut[:, 1:] |= horizontal
        cut[:-1] |= vertical
        cut[1:] |= vertical
        out = 0.28 * reconstruction
        out[cut] = (0.0, 1.0, 0.92)
        return out
    if view == "Unconstrained path disagreement":
        unconstrained = objects["propagation"]["unconstrained"]
        disagreement = (
            unconstrained["best_seed"]
            != objects["propagation"]["best_seed"]
        )
        return colour_map(disagreement[labels].astype(np.float64))
    if view == "Embedded interface arc IDs":
        arc_labels = objects["embedded_arc_labels"]
        colour = site_ids(arc_labels)
        out = 0.12 * reconstruction
        mask = arc_labels > 0
        out[mask] = colour[mask]
        return out
    if view == "Embedded interface junctions":
        topology = objects["embedded_topology"]
        height, width = labels.shape
        centers = np.column_stack((
            topology["junction"]["x"] / max(width, 1),
            topology["junction"]["y"] / max(height, 1),
        ))
        return overlay_sites(reconstruction, centers)
    if view == "Object IDs over reconstruction":
        coloured = 0.58 * reconstruction + 0.42 * objects["soft_ids"]
        return _blend_object_boundaries(coloured, object_labels)
    if view == "Object confidence":
        return colour_map(objects["confidence"])
    if view == "Soft winner weight":
        return colour_map(objects["soft_winner_weight"])
    if view == "Distance to hard-object interface":
        return colour_map(objects["object_interface_distance"])
    if view == "Object saddle ambiguity":
        return colour_map(objects["saddle_margin"])
    if view == "Object waterline":
        return colour_map(objects["waterline"])
    if view == "Distance from winning core":
        return colour_map(objects["distance_altitude"])
    if view == "Boundary-distance core altitude":
        return colour_map(objects["core_altitude_map"])
    diagnostics = objects.get("diagnostics")
    if diagnostics is not None:
        quotient = diagnostics["material"]
        cell = labels
        object_id = objects["object_id_per_cell"]
        material_id = quotient["material_id_per_cell"]
        if view == "Object connected-material quotient":
            value = quotient["object_connected_material_quotient"][
                object_id
            ]
            return colour_map(value[cell])
        if view == "Object material-component count":
            value = quotient["object_material_component_count"][object_id]
            return colour_map(value[cell])
        if view == "Material single-object quotient":
            value = quotient["material_object_quotient"][material_id]
            return colour_map(value[cell])
        if view == "Material object-count":
            value = quotient["material_object_count"][material_id]
            return colour_map(value[cell])
    if view == "Selected highpoints":
        height, width = labels.shape
        centers = np.column_stack((
            objects["graph"]["node_x"] / max(width, 1),
            objects["graph"]["node_y"] / max(height, 1),
        ))
        return overlay_sites(
            reconstruction,
            centers[objects["selected_seeds"]],
        )
    if view == "All provisional highpoints":
        height, width = labels.shape
        centers = np.column_stack((
            objects["graph"]["node_x"] / max(width, 1),
            objects["graph"]["node_y"] / max(height, 1),
        ))
        return overlay_sites(
            reconstruction,
            centers[objects["highpoints"]],
        )
    if view == "Seed strength":
        return colour_map(objects["seed_score_map"])
    if view == "Boundary enclosure":
        return colour_map(objects["enclosure_map"])
    interface = objects["interface_maps"]
    interface_views = {
        "Fused interface barrier": "barrier",
        "Raw interface barrier": "raw_barrier",
        "Short-contact reliability": "contact_reliability",
        "Direct target jump": "target_jump",
        "Region colour similarity failure": "region_colour_jump",
        "Cartoon interface jump": "cartoon_jump",
        "Cartoon contribution to barrier": "cartoon_barrier_contribution",
        "Unresolved cartoon interfaces": "unresolved_cartoon_jump",
        "Object-cut cartoon interfaces": "resolved_cartoon_jump",
        "Transport glass jump": "glass_jump",
        "Transport-normal action": "transport_action",
        "Transport support-signature jump": "support_jump",
        "Focus discontinuity": "focus_jump",
        "Unanchored focus-scale difference": "focus_jump_raw",
        "Focus interface reliability": "focus_reliability",
        "Latent support frontier": "latent_support_frontier",
        "Direct visual witness": "visual_witness",
        "Additive barrier control": "additive_barrier",
        "Anchored barrier control": "anchored_barrier",
        "Cross-scale null reliability": "null_reliability",
        "Certified material joins": "material_join",
        "Finsler lift on canonical interfaces":
            "finsler_contour_completion",
    }
    if view in interface_views:
        return colour_map(interface[interface_views[view]])
    if view == "Canonical boundary confidence":
        return colour_map(result["geometry"]["boundary_confidence"])
    if view == "Canonical Site IDs":
        return site_ids(source_labels)
    return objects["soft_ids"]


def _object_config() -> ObjectSupportConfig:
    return ObjectSupportConfig(
        boundary_weight=float(dpg.get_value("object_boundary_weight")),
        target_jump_weight=float(dpg.get_value("object_target_weight")),
        region_colour_weight=float(dpg.get_value("object_region_weight")),
        cartoon_jump_weight=float(dpg.get_value("object_cartoon_weight")),
        glass_jump_weight=float(dpg.get_value("object_glass_weight")),
        transport_weight=float(dpg.get_value("object_transport_weight")),
        support_jump_weight=float(dpg.get_value("object_support_weight")),
        focus_jump_weight=float(dpg.get_value("object_focus_weight")),
        focus_seed_weight=float(
            dpg.get_value("object_focus_seed_weight")),
        null_suppression=float(dpg.get_value("object_null_suppression")),
        fragment_jump_threshold=float(
            dpg.get_value("object_fragment_jump")),
        anchored_barriers=bool(
            dpg.get_value("object_anchored_barriers")),
        material_tolerance=float(dpg.get_value("object_material_tolerance")),
        material_boundary_ceiling=float(
            dpg.get_value("object_material_boundary")),
        short_contact_scale=float(
            dpg.get_value("object_short_contact_scale")),
        short_contact_prior=float(
            dpg.get_value("object_short_contact_prior")),
        contour_cycle_weight=float(
            dpg.get_value("object_contour_cycle_weight")),
        intrinsic_contour_weight=float(
            dpg.get_value("object_intrinsic_contour_weight")),
        finsler_contour_weight=float(
            dpg.get_value("object_finsler_contour_weight")),
        barrier_scale=float(dpg.get_value("object_barrier_scale")),
        detail_weight=float(dpg.get_value("object_detail_weight")),
        enclosure_weight=float(dpg.get_value("object_enclosure_weight")),
        core_weight=float(dpg.get_value("object_core_weight")),
        peak_prominence=float(dpg.get_value("object_peak_prominence")),
        confidence_temperature=float(
            dpg.get_value("object_confidence_temperature")),
    )


def _contour_hierarchy_config() -> ContourHierarchyConfig:
    return ContourHierarchyConfig(
        waterline=float(
            dpg.get_value("object_contour_parent_waterline")),
        soft_temperature=float(
            dpg.get_value("object_contour_soft_temperature")),
    )


def _cell_config() -> SegmentingConfig:
    return SegmentingConfig(
        allocation_method="causal_density",
        allocation_max_side=int(dpg.get_value("object_allocation_side")),
        tgfd_sweeps=int(dpg.get_value("object_tgfd_sweeps")),
        flow_sweeps=int(dpg.get_value("object_flow_sweeps")),
        metric_strength=float(dpg.get_value("object_metric_strength")),
        safety_cells=int(dpg.get_value("object_safety_cells")),
        curvature_limited_density=bool(
            dpg.get_value("object_curvature_density")),
        null_evidence_strength=float(dpg.get_value("object_cell_null")),
        boundary_jump_strength=float(dpg.get_value("object_cell_boundary")),
        interface_coverage_strength=float(
            dpg.get_value("object_interface_coverage")),
        characteristic_passes=int(
            dpg.get_value("object_characteristic_passes")),
        ridge_count=int(dpg.get_value("object_ridges")),
        soft_support_passes=int(dpg.get_value("object_soft_passes")),
        soft_support_coupling=float(
            dpg.get_value("object_soft_coupling")),
    )


def _finsler_config() -> SparseElasticaConfig:
    return SparseElasticaConfig(
        curvature_scale=float(
            dpg.get_value("object_finsler_curvature")),
        speed_floor=float(
            dpg.get_value("object_finsler_speed_floor")),
        boundary_weight=float(
            dpg.get_value("object_finsler_boundary")),
        colour_weight=float(
            dpg.get_value("object_finsler_colour")),
        support_weight=float(
            dpg.get_value("object_finsler_support")),
    )


def _attach_finsler_support(
    result: dict,
    rgb: np.ndarray,
    intrinsic,
    canonical_graph: dict,
) -> dict:
    """Measure and project one sparse lifted completion onto object edges."""

    started = time.perf_counter()
    topology = build_embedded_interface_topology(intrinsic.owner)
    config = _finsler_config()
    evidence = intrinsic_arc_speed(
        topology,
        intrinsic.owner,
        rgb,
        intrinsic.support_measure,
        result["geometry"]["boundary_confidence"],
        config,
    )
    lifted = build_sparse_elastica_graph(
        topology, evidence["speed"], config)
    closing = finsler_saliency_closing(
        lifted,
        evidence["saliency"],
        continuation_scale=float(
            dpg.get_value("object_finsler_continuation")),
    )
    saliency_map = render_elastica_arcs(
        topology, evidence["saliency"])
    completed_map = render_elastica_arcs(
        topology, closing["saliency"])
    lift_map = render_elastica_arcs(
        topology, closing["lift"])
    projected = project_pixel_field_to_arcs(
        canonical_graph["interface_topology"], lift_map)
    canonical_graph["finsler_contour_completion"] = projected
    record = {
        "config": config,
        "topology": topology,
        "evidence": evidence,
        "graph": lifted,
        "closing": closing,
        "saliency_map": saliency_map,
        "completed_map": completed_map,
        "lift_map": lift_map,
        "canonical_completion": projected,
        "milliseconds": 1000.0 * (
            time.perf_counter() - started),
    }
    canonical_graph["finsler_support"] = record
    return record


def _parent_config() -> ParentHierarchyConfig:
    return ParentHierarchyConfig(
        continuation_floor=float(
            dpg.get_value("object_parent_continuation")),
        polarity_floor=float(
            dpg.get_value("object_parent_polarity")),
        relation_floor=float(
            dpg.get_value("object_parent_relation")),
        minimum_junction_support=int(
            dpg.get_value("object_parent_junction_support")),
        tangent_span=int(
            dpg.get_value("object_parent_tangent_span")),
        junction_attraction=bool(
            dpg.get_value("object_parent_junction_attraction")),
        enclosed_seam_dominance=float(
            dpg.get_value("object_parent_enclosed_dominance")),
        containment=bool(
            dpg.get_value("object_parent_containment")),
        containment_dominance=float(
            dpg.get_value("object_parent_containment_dominance")),
        surround_completion=bool(
            dpg.get_value("object_parent_surround_completion")),
        completion_gap_scale=float(
            dpg.get_value("object_parent_completion_gap")),
        completion_polarity_floor=float(
            dpg.get_value("object_parent_completion_polarity")),
        completion_relation_floor=float(
            dpg.get_value("object_parent_completion_relation")),
        completion_collision_support=int(
            dpg.get_value("object_parent_completion_support")),
    )


def _assembly_config() -> SceneAssemblyConfig:
    return SceneAssemblyConfig(
        relation_floor=float(
            dpg.get_value("object_assembly_relation")),
        colour_scale=float(
            dpg.get_value("object_assembly_colour")),
        support_scale=float(
            dpg.get_value("object_assembly_support")),
        frame_penalty=float(
            dpg.get_value("object_assembly_frame")),
        exterior_barrier=float(
            dpg.get_value("object_assembly_exterior_barrier")),
        frame_seed_exposure=float(
            dpg.get_value("object_assembly_frame_seed")),
        include_enclosed_seams=bool(
            dpg.get_value("object_assembly_enclosed")),
        include_completions=bool(
            dpg.get_value("object_assembly_completions")),
    )


def _metrics_text(result: dict, objects: dict) -> str:
    graph = objects["graph"]
    edge = graph["edge"]
    cartoon_mass = (
        np.asarray(edge["length"], dtype=np.float64)
        * np.asarray(objects["evidence"]["cartoon_jump"], dtype=np.float64)
    )
    cartoon_total = float(np.sum(cartoon_mass))
    object_crossing = (
        objects["object_id_per_cell"][edge["first"]]
        != objects["object_id_per_cell"][edge["second"]]
    )
    cartoon_resolved = (
        float(np.sum(cartoon_mass[object_crossing]))
        / max(cartoon_total, 1e-12)
    )
    diagnostic = objects.get("diagnostics")
    quotient_text = ""
    if diagnostic is not None:
        quotient = diagnostic["material"]
        quotient_text = (
            f" | {quotient['material_count']} diagnostic basins"
        )
    contour = objects.get("contour_hierarchy")
    contour_text = ""
    if contour is not None:
        contour_text = (
            f" | contour ultrametric {contour['leaf_count']} leaves → "
            f"{contour['parent_count']} displayed parents"
        )
    parent = objects.get("parent_hierarchy")
    parent_text = ""
    if parent is not None:
        parent_text = (
            f" | {parent['part_count']} parts → "
            f"{parent['parent_count']} parent hypotheses "
            f"({len(parent['accepted_relations']['score'])} seams + "
            f"{len(parent['accepted_completion_indices'])} completions; "
            f"{len(parent['enclosed_seam_relations']['score'])} "
            "enclosed alternatives)"
        )
    assembly = objects.get("scene_assembly")
    assembly_text = ""
    if assembly is not None:
        assembly_text = (
            f" | {assembly['assembly_count']} compound assemblies "
            f"({int(np.count_nonzero(assembly['accepted']))} relations) "
            f"/ {assembly['exterior_reachability']['basin_count']} "
            "bounded support basins"
        )
    ownership = objects.get("border_ownership")
    ownership_text = ""
    if ownership is not None:
        relation = ownership["directed_relations"]
        agreement = ownership["relation_agreement_fraction"]
        held_out = ownership["leave_one_out_agreement_fraction"]
        ownership_text = (
            f" | {len(relation['front'])} directed depth relations"
            + (
                f", support agreement {100.0 * agreement:.0f}%"
                if np.isfinite(agreement)
                else ""
            )
            + (
                f" / held-out {100.0 * held_out:.0f}%"
                if np.isfinite(held_out)
                else ""
            )
        )
    finsler = objects.get("finsler_support")
    finsler_text = ""
    if finsler is not None:
        relations = finsler.get("relations")
        collision_count = (
            len(relations["action"]) if relations is not None else 0)
        finsler_text = (
            f" | sparse Finsler "
            f"{finsler['graph']['state_count']} states / "
            f"{finsler['milliseconds']:.0f} ms / "
            f"{collision_count} shared-surround collisions"
        )
    return (
        f"{len(result['centers'])} transport cells → "
        f"{graph['cells']} connected support fragments → "
        f"{int(objects['material_atom_per_cell'].max(initial=-1)) + 1} "
        f"material atoms → "
        f"{len(objects['highpoints'])} provisional highpoints → "
        f"{len(objects['selected_seeds'])} persistent objects | "
        f"{len(graph['edge']['first'])} literal interfaces | "
        f"cartoon cut capture {100.0 * cartoon_resolved:.0f}% | "
        f"PSNR {result['record']['psnr']:.2f} dB | "
        f"cell build {result['timing']['total_ms']:.0f} ms | "
        f"graph {objects['timing']['graph_ms']:.0f} ms | "
        f"object hierarchy {objects['timing']['analysis_ms']:.0f} ms"
        f"{finsler_text}"
        f"{quotient_text}"
        f"{contour_text}"
        f" | {objects['embedded_topology']['arc']['count']} arcs / "
        f"{objects['embedded_topology']['junction']['count']} junctions"
        f"{parent_text}"
        f"{assembly_text}"
        f"{ownership_text}"
    )


def _attach_diagnostics(
    result: dict,
    objects: dict,
    rgb: np.ndarray,
) -> None:
    objects["diagnostics"] = analyze_object_hierarchy(
        result,
        objects,
        barrier_threshold=float(
            dpg.get_value("object_diagnostic_waterline")),
    )
    topology = objects["graph"].get("interface_topology")
    if topology is None:
        topology = build_embedded_interface_topology(result["labels"])
    objects["embedded_topology"] = topology
    objects["embedded_arc_labels"] = render_arc_ids(topology)
    finsler = objects.get("finsler_support")
    if finsler is not None:
        arc_parts = label_intrinsic_arc_parts(
            finsler["topology"], objects["object_labels"])
        relations = elastica_common_surround_relations(
            finsler["graph"],
            arc_parts,
            finsler["evidence"]["saliency"],
        )
        finsler["arc_parts"] = arc_parts
        finsler["relations"] = relations
    objects["contour_hierarchy"] = infer_contour_object_hierarchy(
        objects, _contour_hierarchy_config())
    parent = infer_parent_objects(objects, _parent_config())
    objects["parent_hierarchy"] = parent
    objects["border_ownership"] = infer_transport_border_ownership(
        objects, parent)
    objects["scene_assembly"] = infer_scene_assemblies(
        objects, parent, _assembly_config())
    objects["relation_forensics"] = transport_relation_forensics(objects)
    objects["graph_scattering"] = transport_graph_scattering(
        objects, scales=5)
    objects["edge_relation"] = transport_edge_relation(
        objects, scales=3)
    objects["part_edge_relation"] = aggregate_signed_relations(
        objects["edge_relation"],
        objects["object_id_per_cell"],
        objects["graph"]["area"],
    )
    # Focus is now built before the support forest because its reliable
    # discontinuity participates in interface resistance. Keep this fallback
    # for records made by older callers.
    if "focus_forensics" not in objects:
        objects["focus_forensics"] = transport_focus_forensics(
            rgb,
            objects["graph"]["labels"],
        )
        objects["focus_forensics"]["interface"] = (
            transport_focus_interfaces(
                objects["focus_forensics"],
                objects["graph"]["labels"],
                topology,
            )
        )
    S.anchor_cell = int(np.clip(
        S.anchor_cell, 0, max(int(objects["graph"]["cells"]) - 1, 0)))
    S.anchor_cell_fields = transport_anchor_cell_fields(
        objects, S.anchor_cell)


def refresh() -> None:
    with S.lock:
        rgb, result, objects = S.rgb, S.result, S.objects
    if rgb is None:
        return
    push_texture(SOURCE, rgb)
    push_texture(RESULT, current_view(rgb, result, objects))
    dpg.set_value(
        "object_metrics",
        "" if result is None or objects is None
        else _metrics_text(result, objects),
    )


def _work_side() -> int:
    return (
        0
        if dpg.get_value("object_full")
        else int(dpg.get_value("object_work_side"))
    )


def build_worker() -> None:
    if S.busy or S.image is None:
        return
    S.busy = True
    try:
        rgb = _fit_rgb(S.image, _work_side())
        S.status = "Building the canonical one-decomposition transport cells..."
        result = build_segmenting_representation(rgb, _cell_config())
        S.status = "Measuring the intrinsic Voronoi support geometry..."
        luminance = rgb[..., :3] @ np.array(
            [0.2125, 0.7154, 0.0721])
        intrinsic = extract_intrinsic_voronoi_support(
            luminance,
            VoronoiITDConfig(),
            guidance=rgb,
        )
        object_config = _object_config()
        graph = build_cell_interface_graph(
            result,
            rgb,
            fragment_jump_threshold=(
                object_config.fragment_jump_threshold),
        )
        if object_config.finsler_contour_weight > 0.0:
            S.status = (
                "Marching the optional sparse Finsler-elastica support...")
            finsler = _attach_finsler_support(
                result, rgb, intrinsic, graph)
        else:
            finsler = None
        S.status = "Building the maximum-support object tree..."
        objects = infer_object_support(
            result,
            rgb,
            object_config,
            graph=graph,
            intrinsic_owner=intrinsic.owner,
        )
        objects["intrinsic_support"] = intrinsic
        objects["finsler_support"] = finsler
        _attach_diagnostics(result, objects, rgb)
        with S.lock:
            S.rgb = rgb
            S.result = result
            S.objects = objects
            S.graph = objects["graph"]
            S.intrinsic_support = intrinsic
            S.finsler_support = finsler
        S.status = (
            f"{S.name}: object-support hierarchy complete. "
            "Use the evidence views to find where it is blind."
        )
    except Exception as exc:
        S.status = f"Build failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def object_worker() -> None:
    if S.busy:
        return
    with S.lock:
        rgb, result, graph, intrinsic = (
            S.rgb, S.result, S.graph, S.intrinsic_support)
    if rgb is None or result is None:
        S.status = "Build the transport cells first."
        return
    S.busy = True
    try:
        S.status = "Recomputing only the sparse object-support hierarchy..."
        config = _object_config()
        rebuild_graph = (
            graph is None
            or abs(
                float(graph.get("fragment_jump_threshold", np.inf))
                - float(config.fragment_jump_threshold)
            ) > 1e-12
        )
        if rebuild_graph:
            graph = build_cell_interface_graph(
                result,
                rgb,
                fragment_jump_threshold=config.fragment_jump_threshold,
            )
        finsler = (
            _attach_finsler_support(result, rgb, intrinsic, graph)
            if (
                intrinsic is not None
                and config.finsler_contour_weight > 0.0
            )
            else None
        )
        objects = infer_object_support(
            result,
            rgb,
            config,
            graph=graph,
            intrinsic_owner=(
                None if intrinsic is None else intrinsic.owner),
        )
        objects["intrinsic_support"] = intrinsic
        objects["finsler_support"] = finsler
        _attach_diagnostics(result, objects, rgb)
        with S.lock:
            S.objects = objects
            S.graph = objects["graph"]
            S.finsler_support = finsler
        S.status = (
            f"{S.name}: object controls updated without rebuilding "
            "the reconstruction cells."
        )
    except Exception as exc:
        S.status = f"Object analysis failed: {type(exc).__name__}: {exc}"
    finally:
        S.busy = False


def adopt(image: np.ndarray, name: str) -> None:
    S.image = np.asarray(image, dtype=np.float64)
    S.name = name
    rgb = _fit_rgb(S.image, _work_side())
    with S.lock:
        S.rgb, S.result, S.objects, S.graph = rgb, None, None, None
        S.intrinsic_support = None
        S.finsler_support = None
        S.anchor_part = 0
        S.anchor_cell = 0
        S.anchor_cell_fields = None
    S.status = f"{name} loaded. Press Build cells + objects."
    push_texture(SOURCE, rgb)
    push_texture(RESULT, np.full_like(rgb, 0.08))


def cb_gallery(sender, label) -> None:
    try:
        key = gallery.key_for_label(label)
        adopt(gallery.load(key), gallery.describe(key)["label"])
    except Exception as exc:
        S.status = f"Gallery load failed: {type(exc).__name__}: {exc}"


def cb_file(sender, app_data) -> None:
    candidates = list((app_data.get("selections") or {}).values())
    candidates.append(app_data.get("file_path_name", ""))
    path = next((Path(p) for p in candidates if p and Path(p).is_file()), None)
    if path is None:
        S.status = "Could not resolve the selected image."
        return
    try:
        from skimage.io import imread
        adopt(imread(path), path.name)
    except Exception as exc:
        S.status = f"Image load failed: {type(exc).__name__}: {exc}"


def cb_inspect(sender=None, app_data=None) -> None:
    if not dpg.does_item_exist("object_result_image"):
        return
    if not dpg.is_item_hovered("object_result_image"):
        return
    with S.lock:
        result, objects = S.result, S.objects
    if result is None or objects is None:
        return
    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    left, top = dpg.get_item_rect_min("object_result_image")
    width, height = dpg.get_item_rect_size("object_result_image")
    if width <= 0 or height <= 0:
        return
    labels = np.asarray(result["labels"])
    x = int(np.clip(
        (mouse_x - left) * labels.shape[1] / width,
        0,
        labels.shape[1] - 1,
    ))
    y = int(np.clip(
        (mouse_y - top) * labels.shape[0] / height,
        0,
        labels.shape[0] - 1,
    ))
    diagnostic = objects.get("diagnostics")
    report = selection_report(
        result,
        objects,
        y,
        x,
        quotient=None if diagnostic is None else diagnostic["material"],
        adjacency=None if diagnostic is None else diagnostic["adjacency"],
    )
    parent = objects.get("parent_hierarchy")
    parent_line = ""
    if parent is not None:
        parent_id = int(parent["parent_id_per_part"][report["object_id"]])
        parent_line = (
            f"\npart {report['object_id']} → parent hypothesis {parent_id}"
        )
    ownership = objects.get("border_ownership")
    if ownership is not None:
        part = report["object_id"]
        parent_line += (
            f"\nobserved border ownership "
            f"{ownership['observed_frontness_per_part'][part]:+.3f}; "
            f"transport-depth extrapolation "
            f"{ownership['support_frontness_per_part'][part]:+.3f}"
        )
    forensics = objects.get("relation_forensics")
    if forensics is not None:
        S.anchor_cell = int(objects["graph"]["labels"][y, x])
        S.anchor_cell_fields = transport_anchor_cell_fields(
            objects, S.anchor_cell)
        S.anchor_part = report["object_id"]
        parent_line += (
            f"\nforensic anchors: support fragment {S.anchor_cell}; "
            f"object part {S.anchor_part}"
        )
        for label, key in (
            ("colour", "colour_similarity"),
            ("transport state", "state_similarity"),
            ("transport action", "action_similarity"),
            ("metric tensor", "metric_similarity"),
            ("boundary role", "boundary_transport_similarity"),
            ("full relation", "boundary_full_similarity"),
        ):
            values = forensics[key][S.anchor_part].copy()
            values[S.anchor_part] = -np.inf
            order = np.argsort(values)[::-1][:3]
            parent_line += (
                f"\n{label} nearest: "
                + ", ".join(
                    f"{int(index)} ({values[index]:.3f})"
                    for index in order
                )
            )
    dpg.set_value(
        "object_inspector",
        format_selection_report(report, neighbours=7) + parent_line,
    )
    refresh()


def slider(
    tag: str,
    label: str,
    default,
    low,
    high,
    *,
    floating: bool = False,
    width: int = 250,
) -> None:
    function = dpg.add_slider_float if floating else dpg.add_slider_int
    function(
        label=label,
        tag=tag,
        default_value=default,
        min_value=low,
        max_value=high,
        width=width,
    )


def build_ui(labels: list[str], default_label: str) -> None:
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=cb_file,
        tag="object_file_dialog",
        width=900,
        height=520,
    ):
        dpg.add_file_extension(
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"
            "{.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp}")
        dpg.add_file_extension(".*")
    with dpg.window(tag="object_root"):
        with dpg.group(horizontal=True):
            dpg.add_combo(
                labels,
                default_value=default_label,
                width=390,
                tag="object_gallery",
                callback=cb_gallery,
            )
            dpg.add_button(
                label="Load image...",
                callback=lambda: dpg.show_item("object_file_dialog"),
            )
            dpg.add_button(
                label="Build cells + objects",
                tag="object_build",
                callback=lambda: threading.Thread(
                    target=build_worker, daemon=True).start(),
            )
            dpg.add_button(
                label="Recompute objects only",
                tag="object_recompute",
                callback=lambda: threading.Thread(
                    target=object_worker, daemon=True).start(),
            )
        dpg.add_text("", tag="object_status", wrap=1500)
        dpg.add_text("", tag="object_metrics", wrap=1500)
        with dpg.collapsing_header(
            label="Canonical transport cells",
            default_open=False,
        ):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="process every source pixel",
                    tag="object_full",
                    default_value=False,
                )
                slider(
                    "object_work_side", "otherwise longest side",
                    768, 128, 3840, width=320)
                slider(
                    "object_allocation_side", "allocation grid side",
                    512, 128, 1536, width=300)
                slider(
                    "object_safety_cells", "site safety ceiling",
                    32768, 256, 65536, width=300)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="anchored witnesses only (research control)",
                    tag="object_anchored_barriers",
                    default_value=False,
                )
                slider(
                    "object_tgfd_sweeps", "single Meyer sweeps",
                    1, 1, 64)
                slider(
                    "object_flow_sweeps", "fixed glass sweeps",
                    24, 4, 64)
                slider(
                    "object_metric_strength", "metric strength",
                    1.5, 0.0, 8.0, floating=True)
                slider(
                    "object_characteristic_passes", "front relaxation",
                    0, 0, 4)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="curvature-limited population",
                    tag="object_curvature_density",
                    default_value=True,
                )
                slider(
                    "object_cell_null", "cell null evidence",
                    0.5, 0.0, 1.0, floating=True)
                slider(
                    "object_cell_boundary", "cell boundary action",
                    24.0, 0.0, 48.0, floating=True)
                slider(
                    "object_interface_coverage", "interface coverage",
                    0.4, 0.0, 1.0, floating=True)
            with dpg.group(horizontal=True):
                slider("object_ridges", "ridge finish", 1, 0, 2)
                slider(
                    "object_soft_passes", "soft cover passes",
                    16, 0, 64)
                slider(
                    "object_soft_coupling", "soft cover sharing",
                    0.8, 0.0, 2.0, floating=True)
        with dpg.collapsing_header(
            label="Object interface evidence",
            default_open=True,
        ):
            with dpg.group(horizontal=True):
                slider(
                    "object_boundary_weight", "decisive boundary",
                    1.5, 0.0, 4.0, floating=True)
                slider(
                    "object_target_weight", "direct target jump",
                    0.15, 0.0, 3.0, floating=True)
                slider(
                    "object_region_weight", "region colour difference",
                    0.05, 0.0, 3.0, floating=True)
            with dpg.group(horizontal=True):
                slider(
                    "object_cartoon_weight", "cartoon jump",
                    0.10, 0.0, 3.0, floating=True)
                slider(
                    "object_glass_weight", "glass jump",
                    0.25, 0.0, 3.0, floating=True)
                slider(
                    "object_transport_weight", "transport-normal action",
                    0.65, 0.0, 3.0, floating=True)
                slider(
                    "object_support_weight", "support-signature jump",
                    1.0, 0.0, 3.0, floating=True)
                slider(
                    "object_focus_weight", "focus discontinuity",
                    0.50, 0.0, 3.0, floating=True)
                slider(
                    "object_focus_seed_weight", "autofocus selection",
                    0.35, 0.0, 1.0, floating=True)
                slider(
                    "object_null_suppression", "null suppression",
                    0.90, 0.0, 1.0, floating=True)
                slider(
                    "object_fragment_jump",
                    "intra-site discontinuity cut",
                    0.12, 0.01, 0.40, floating=True)
            with dpg.group(horizontal=True):
                slider(
                    "object_material_tolerance",
                    "unchanged-material OKLab tolerance",
                    0.012, 0.0, 0.05, floating=True)
                slider(
                    "object_material_boundary",
                    "material boundary ceiling",
                    0.08, 0.0, 0.5, floating=True)
                slider(
                    "object_diagnostic_waterline",
                    "diagnostic material waterline",
                    0.35, 0.0, 1.0, floating=True)
            with dpg.group(horizontal=True):
                slider(
                    "object_short_contact_scale",
                    "short-contact shrinkage",
                    0.0, 0.0, 2.0, floating=True)
                slider(
                    "object_short_contact_prior",
                    "short-contact barrier prior",
                    0.5, 0.0, 1.0, floating=True)
            with dpg.group(horizontal=True):
                slider(
                    "object_contour_cycle_weight",
                    "legacy cycle projection",
                    0.0, 0.0, 1.0, floating=True)
                slider(
                    "object_intrinsic_contour_weight",
                    "legacy intrinsic alignment",
                    0.0, 0.0, 1.0, floating=True)
            dpg.add_text(
                "Only literal neighbours are compared. Similar face and "
                "bridge cells cannot merge without crossing the measured "
                "interfaces between them. The legacy cycle controls are "
                "disabled: cell-perimeter closedness is not object evidence.")
        with dpg.collapsing_header(
            label="Sparse Finsler-elastica completion",
            default_open=False,
        ):
            with dpg.group(horizontal=True):
                slider(
                    "object_finsler_contour_weight",
                    "completed-boundary contribution",
                    0.0, 0.0, 1.0, floating=True)
                slider(
                    "object_finsler_curvature",
                    "curvature radius",
                    10.0, 0.0, 32.0, floating=True)
                slider(
                    "object_finsler_continuation",
                    "intrinsic continuation span",
                    2.0, 0.25, 16.0, floating=True)
                slider(
                    "object_finsler_speed_floor",
                    "off-contour speed floor",
                    0.025, 0.005, 0.25, floating=True)
            with dpg.group(horizontal=True):
                slider(
                    "object_finsler_boundary",
                    "canonical border evidence",
                    2.5, 0.0, 4.0, floating=True)
                slider(
                    "object_finsler_colour",
                    "variable-distance colour evidence",
                    0.35, 0.0, 2.0, floating=True)
                slider(
                    "object_finsler_support",
                    "intrinsic support discontinuity",
                    0.15, 0.0, 2.0, floating=True)
            dpg.add_text(
                "Two directed states per intrinsic interface arc replace "
                "the dense x-y-angle grid. One min-plus march requires "
                "coherent support from both contour directions, then projects "
                "only the new completion lift onto canonical object edges. "
                "Recompute objects updates this pass without rebuilding cells.")
        with dpg.collapsing_header(
            label="Highpoints and waterline",
            default_open=True,
        ):
            with dpg.group(horizontal=True):
                slider(
                    "object_detail_weight", "high-detail germ strength",
                    0.35, 0.0, 1.5, floating=True)
                slider(
                    "object_enclosure_weight", "boundary enclosure strength",
                    0.15, 0.0, 1.5, floating=True)
                slider(
                    "object_core_weight", "boundary-distance core strength",
                    1.0, 0.0, 2.0, floating=True)
                slider(
                    "object_barrier_scale", "waterline barrier scale",
                    1.5, 0.25, 10.0, floating=True)
            with dpg.group(horizontal=True):
                slider(
                    "object_peak_prominence", "persistent object prominence",
                    0.30, 0.0, 0.7, floating=True)
                slider(
                    "object_confidence_temperature",
                    "soft membership temperature",
                    0.12, 0.01, 0.5, floating=True)
            dpg.add_text(
                "Local detail highpoints are born simultaneously. The "
                "maximum-support forest gathers highpoints connected through "
                "permeable cell interfaces; only topologically persistent "
                "highpoints retain distinct IDs.")
        with dpg.collapsing_header(
            label="Closed-contour ultrametric",
            default_open=True,
        ):
            with dpg.group(horizontal=True):
                slider(
                    "object_contour_parent_waterline",
                    "displayed parent waterline",
                    0.50, 0.0, 1.0, floating=True)
                slider(
                    "object_contour_soft_temperature",
                    "soft cophenetic temperature",
                    0.12, 0.01, 0.5, floating=True)
            dpg.add_text(
                "Part membership is the minimax merge altitude on the "
                "closed-contour quotient graph. One Kruskal pass returns the "
                "entire hierarchy; this waterline changes only its displayed "
                "cut, not the measured cells or contours.")
        with dpg.collapsing_header(
            label="Part → parent topology",
            default_open=True,
        ):
            with dpg.group(horizontal=True):
                slider(
                    "object_parent_continuation",
                    "outer-contour continuation",
                    0.60, 0.0, 1.0, floating=True)
                slider(
                    "object_parent_polarity",
                    "common-surround polarity",
                    0.10, 0.0, 1.0, floating=True)
                slider(
                    "object_parent_relation",
                    "parent relation waterline",
                    0.48, 0.0, 1.0, floating=True)
                slider(
                    "object_parent_tangent_span",
                    "arc tangent span",
                    8, 1, 32)
                slider(
                    "object_parent_junction_support",
                    "seam endpoint support",
                    2, 1, 4)
                dpg.add_checkbox(
                    label="experimental T-junction attraction",
                    tag="object_parent_junction_attraction",
                    default_value=False,
                )
                slider(
                    "object_parent_enclosed_dominance",
                    "enclosed seam exterior dominance",
                    0.72, 0.4, 1.0, floating=True)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="attach enclosed details to their container",
                    tag="object_parent_containment",
                    default_value=True,
                )
                slider(
                    "object_parent_containment_dominance",
                    "containment boundary dominance",
                    0.97, 0.5, 1.0, floating=True)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="common-surround first-arrival completion",
                    tag="object_parent_surround_completion",
                    default_value=True,
                )
                slider(
                    "object_parent_completion_gap",
                    "completion transport reach",
                    3.0, 0.25, 12.0, floating=True)
                slider(
                    "object_parent_completion_polarity",
                    "completion contrast polarity",
                    0.10, 0.0, 1.0, floating=True)
            with dpg.group(horizontal=True):
                slider(
                    "object_parent_completion_relation",
                    "completion relation waterline",
                    0.22, 0.0, 1.0, floating=True)
                slider(
                    "object_parent_completion_support",
                    "completion collision support",
                    2, 1, 8)
            dpg.add_text(
                "Parts are retained. T-junction continuation supplies a "
                "directed front/behind proposal; it does not imply that the "
                "two regions behind share an object. Experimental attraction "
                "is therefore off by default. Containment and common-surround "
                "completion remain independent parent mechanisms.")
        with dpg.collapsing_header(
            label="Compound foreground assembly",
            default_open=True,
        ):
            with dpg.group(horizontal=True):
                slider(
                    "object_assembly_relation",
                    "assembly relation waterline",
                    0.46, 0.0, 1.0, floating=True)
                slider(
                    "object_assembly_colour",
                    "lawful colour compatibility",
                    0.16, 0.02, 0.5, floating=True)
                slider(
                    "object_assembly_support",
                    "lawful support compatibility",
                    2.25, 0.25, 6.0, floating=True)
                slider(
                    "object_assembly_frame",
                    "frame-rooted substrate penalty",
                    2.0, 0.0, 8.0, floating=True)
                slider(
                    "object_assembly_exterior_barrier",
                    "exterior reachability waterline",
                    0.46, 0.0, 1.0, floating=True)
                slider(
                    "object_assembly_frame_seed",
                    "broad frame exposure seed",
                    0.12, 0.0, 0.8, floating=True)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="include enclosed material seams",
                    tag="object_assembly_enclosed",
                    default_value=True,
                )
                dpg.add_checkbox(
                    label="include accepted amodal completions",
                    tag="object_assembly_completions",
                    default_value=True,
                )
            dpg.add_text(
                "Geometry alone creates assembly candidates. Appearance is "
                "consulted only inside those lawful containment/contact "
                "relations; continuous frame exposure suppresses the basal "
                "substrate. Assembly IDs do not claim material identity.")
        with dpg.group(horizontal=True):
            dpg.add_text("Right panel")
            dpg.add_combo(
                VIEWS,
                default_value=VIEWS[0],
                tag="object_view",
                callback=lambda: refresh(),
                width=350,
            )
        dpg.add_text(
            "Click the right panel to anchor both the raw support-fragment "
            "likeness microscope and the later object-part comparisons.",
        )
        dpg.add_text("", tag="object_inspector", wrap=1500)
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Original")
                dpg.add_image(SOURCE, tag="object_source_image")
            with dpg.group():
                dpg.add_text("Emergent object support")
                dpg.add_image(RESULT, tag="object_result_image")
    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=cb_inspect)


def main() -> int:
    keys = gallery.available()
    labels = gallery.labels(keys)
    key = "pikachu" if "pikachu" in keys else keys[0]
    default_label = labels[keys.index(key)]
    dpg.create_context()
    alloc_texture(SOURCE, 8, 8)
    alloc_texture(RESULT, 8, 8)
    build_ui(labels, default_label)
    dpg.create_viewport(
        title="BFFT Vision — Emergent Object Support",
        width=1580,
        height=1020,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("object_root", True)
    cb_gallery(None, default_label)
    last = 0.0
    while dpg.is_dearpygui_running():
        dpg.set_value("object_status", S.status)
        dpg.configure_item("object_build", enabled=not S.busy)
        dpg.configure_item(
            "object_recompute",
            enabled=(not S.busy and S.result is not None),
        )
        now = time.perf_counter()
        if now - last > 0.15:
            refresh()
            last = now
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
