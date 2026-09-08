"""Compile and render a 3-D room through genuine retained transport blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .child_volume import (
    AxisAlignedVolume,
    BoundaryFiber,
    ChildVolumeResponseSystem,
    VolumeMedium,
    compile_child_volume_rank_blocks,
)
from .light_tensor import LightTensor6
from .render_standard_scene import build_standard_scene, render_scene
from .retained_diffuse import NativeDiffuseTransportPlan
from .transport import Patch, compile_hierarchical_field, solve_transport


def _feed_global_light_through_child(
    patches: tuple[Patch, ...],
    area_indices: np.ndarray,
) -> tuple[tuple[Patch, ...], dict[str, object]]:
    """Turn the ceiling fixture into the egress face of a private child slab."""
    fixture = tuple(patches[int(index)] for index in area_indices)
    ingress = tuple(
        BoundaryFiber(
            np.array((patch.center[0], 4.0, patch.center[2])),
            np.array((0.0, 1.0, 0.0)),
            np.array((0.0, -1.0, 0.0)),
            patch.area,
            2.0 * np.pi,
        )
        for patch in fixture
    )
    egress = tuple(
        BoundaryFiber(
            patch.center,
            np.array((0.0, -1.0, 0.0)),
            np.array((0.0, -1.0, 0.0)),
            patch.area,
            2.0 * np.pi,
        )
        for patch in fixture
    )
    medium = VolumeMedium(
        density=1.0,
        extinction=0.10,
        linear_diffusive_extinction=0.24,
        circular_diffusive_extinction=0.07,
    )
    compiled = compile_child_volume_rank_blocks(
        AxisAlignedVolume(
            np.array((-1.15, 3.92, -3.0)),
            np.array((1.15, 4.0, -1.3)),
        ),
        ingress,
        egress,
        (),
        medium=medium,
        admissibility=0.35,
    )
    # Three spectral representatives also carry distinct polarization-mode
    # distributions.  The child does not convert them into RGB; RGB is only
    # the terminal display basis used after the optical boundary response.
    modes = (
        LightTensor6(1.0, 610.0, 18.0**2, 0.0, 0.72, 0.0, 1),
        LightTensor6(1.0, 545.0, 15.0**2, 55.0, 0.24, 0.35, 2),
        LightTensor6(1.0, 460.0, 13.0**2, 110.0, 0.08, -0.82, 3),
    )
    ingress_powers = np.stack([patch.emission for patch in fixture]).T
    child = ChildVolumeResponseSystem(
        compiled.topology,
        absolute_cutoff=1.0e-14,
        cache_strategy="retained",
        execution_backend="native",
    )
    response = child.evaluate_packed(modes, ingress_powers)
    updated = list(patches)
    for local_index, raw_index in enumerate(area_indices):
        patch = patches[int(raw_index)]
        updated[int(raw_index)] = Patch(
            center=patch.center,
            normal=patch.normal,
            area=patch.area,
            radius=patch.radius,
            albedo=patch.albedo,
            emission=response.powers[:, local_index],
            object_id=patch.object_id,
        )
    diagnostics: dict[str, object] = {
        "ingress_fibers": len(ingress),
        "egress_fibers": len(egress),
        "retained_blocks": len(compiled.topology.blocks),
        "stored_coefficients": compiled.topology.retained_stored_coefficients,
        "represented_expanded_pairs": compiled.topology.retained_expanded_coefficients,
        "execution_backend": response.execution_backend,
        "block_applications": response.block_applications,
        "gathered_source_coefficients": response.gathered_source_coefficients,
        "scattered_receiver_coefficients": response.scattered_receiver_coefficients,
        "input_power": float(np.sum(ingress_powers)),
        "egress_power": float(np.sum(response.powers)),
        "mode_centre_nm": [mode.centre_nm for mode in modes],
        "mode_linear_peakedness": [mode.linear_peakedness for mode in modes],
        "mode_chirality": [mode.chirality for mode in modes],
        "geometry": dict(compiled.geometry_diagnostics),
    }
    return tuple(updated), diagnostics


def render_retained_room(
    output: Path,
    *,
    width: int,
    height: int,
    admissibility: float,
    transport_scale: float,
    verify_reference: bool,
) -> dict[str, object]:
    patches, occluders, groups, spheres = build_standard_scene()
    patches, child_diagnostics = _feed_global_light_through_child(
        patches, groups["area_light"]
    )
    emission = np.stack([patch.emission for patch in patches])
    albedo = np.stack([patch.albedo for patch in patches])

    started = time.perf_counter()
    field = compile_hierarchical_field(
        patches,
        occluders=occluders,
        admissibility=admissibility,
        transport_scale=transport_scale,
    )
    geometry_compile_seconds = time.perf_counter() - started

    source_width = np.array(
        [block.source.size for block in field.blocks], dtype=np.int64
    )
    receiver_width = np.array(
        [block.receiver.size for block in field.blocks], dtype=np.int64
    )
    expanded_pairs = int(np.sum(source_width * receiver_width))

    started = time.perf_counter()
    plan = NativeDiffuseTransportPlan(field)
    native_plan_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        solved = plan.solve(
            emission,
            albedo,
            relative_tolerance=2.0e-8,
            absolute_tolerance=1.0e-12,
        )
        native_solve_seconds = time.perf_counter() - started
    finally:
        plan.close()

    reference_seconds = None
    reference_maximum_absolute_error = None
    reference_l1_error = None
    if verify_reference:
        started = time.perf_counter()
        reference = solve_transport(
            field,
            emission,
            albedo,
            relative_tolerance=2.0e-8,
            absolute_tolerance=1.0e-12,
        )
        reference_seconds = time.perf_counter() - started
        difference = np.abs(reference.outgoing - solved.outgoing)
        reference_maximum_absolute_error = float(np.max(difference))
        reference_l1_error = float(np.sum(difference))

    started = time.perf_counter()
    image = render_scene(
        patches,
        groups,
        spheres,
        solved.outgoing,
        width=width,
        height=height,
    )
    camera_gather_seconds = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(output, optimize=True)

    depth_work = [
        {
            "depth": work.depth,
            "active_nodes": work.active_nodes,
            "block_applications": work.block_applications,
            "gathered_source_coefficients": work.gathered_source_coefficients,
            "scattered_receiver_coefficients": work.scattered_receiver_coefficients,
        }
        for work in solved.depth_work
    ]
    diagnostics: dict[str, object] = {
        "scene": "3-D Cornell-style room fed through a retained child optical volume",
        "transport_representation": "polarized global source -> child retained response -> dual-tree parent geometry -> native residual march",
        "native_backend": solved.backend,
        "width": width,
        "height": height,
        "surface_nodes": len(patches),
        "sphere_occluders": len(occluders),
        "admissibility": admissibility,
        "transport_scale": transport_scale,
        "geometry_compile_seconds": geometry_compile_seconds,
        "native_plan_seconds": native_plan_seconds,
        "native_solve_seconds": native_solve_seconds,
        "camera_gather_seconds": camera_gather_seconds,
        "transport_depths": solved.depth_count,
        "retained_blocks": len(field.blocks),
        "retained_stored_coefficients": field.stored_coefficients,
        "represented_expanded_pairs": expanded_pairs,
        "dense_matrix_coefficients_not_materialized": len(patches) ** 2,
        "blocks_with_multiple_sources": int(np.count_nonzero(source_width > 1)),
        "blocks_with_multiple_receivers": int(np.count_nonzero(receiver_width > 1)),
        "blocks_many_to_many": int(
            np.count_nonzero((source_width > 1) & (receiver_width > 1))
        ),
        "maximum_source_width": int(np.max(source_width, initial=0)),
        "maximum_receiver_width": int(np.max(receiver_width, initial=0)),
        "total_native_block_applications": solved.block_applications,
        "total_native_gathered_coefficients": solved.gathered_source_coefficients,
        "total_native_scattered_coefficients": solved.scattered_receiver_coefficients,
        "maximum_conservation_error": float(
            np.max(np.abs(solved.conservation_error), initial=0.0)
        ),
        "reference_seconds": reference_seconds,
        "reference_maximum_absolute_error": reference_maximum_absolute_error,
        "reference_l1_error": reference_l1_error,
        "child_volume": child_diagnostics,
        "geometry_diagnostics": dict(field.diagnostics),
        "depth_work": depth_work,
        "image": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/retained_field_room.png"),
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--admissibility", type=float, default=0.35)
    parser.add_argument("--transport-scale", type=float, default=0.80)
    parser.add_argument("--verify-reference", action="store_true")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        raise ValueError("render dimensions must be positive")
    diagnostics = render_retained_room(
        args.out,
        width=args.width,
        height=args.height,
        admissibility=args.admissibility,
        transport_scale=args.transport_scale,
        verify_reference=args.verify_reference,
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
