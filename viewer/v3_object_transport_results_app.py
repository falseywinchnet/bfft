#!/usr/bin/env python3
"""Dear PyGui microscope for saved V3 object-transport experiments.

This viewer is deliberately read-only: it never reruns V3 and never uses a
landmark to change an inferred matrix. A landmark or mouse click merely selects
one already-computed region column for inspection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    ROOT / "experiments" / "v3_object_transport" / "results"
    / "v3_object_transport_audit_256_bundle"
)
LANDMARKS = (
    ROOT / "experiments" / "v3_object_transport" / "assets"
    / "landmarks.json"
)
PANEL = 430
TEXTURES = ("source", "heat", "recomposition")


@dataclass(frozen=True)
class ViewSpec:
    path: str
    key: str


VIEWS = {
    "Parts · canonical complete": ViewSpec(
        "participation_algebra/{image}/participation_algebra.npz", "complete"),
    "Proposal · heat": ViewSpec(
        "proposal_topology_transport/{image}/full.npz", "heat_kernel"),
    "Proposal · transported parts": ViewSpec(
        "proposal_topology_transport/{image}/full.npz",
        "transported_base_kernel"),
    "Content · raw wavelet leader": ViewSpec(
        "wavelet_leader_transport/{image}/wavelet_leader_transport.npz",
        "leader"),
    "Content · raw payload via proposals": ViewSpec(
        "wavelet_leader_transport/{image}/wavelet_leader_transport.npz",
        "proposal_transported_leader"),
    "Parts + raw payload · via proposals": ViewSpec(
        "wavelet_leader_transport/{image}/wavelet_leader_transport.npz",
        "proposal_transported_base_leader"),
    "Content · leader scale law": ViewSpec(
        "wavelet_leader_scale_law/{image}/wavelet_leader_transport.npz",
        "leader"),
    "Content · scale-law payload via proposals": ViewSpec(
        "wavelet_leader_scale_law/{image}/wavelet_leader_transport.npz",
        "proposal_transported_leader"),
    "Incidence · transition role": ViewSpec(
        "wavelet_incidence_transport/{image}/transition_only.npz",
        "role_kernel"),
    "Incidence · transition complete": ViewSpec(
        "wavelet_incidence_transport/{image}/transition_only.npz",
        "complete_kernel"),
    "Incidence · transition + proposals": ViewSpec(
        "wavelet_incidence_transport/{image}/transition_only.npz",
        "proposal_transported_complete_kernel"),
    "Incidence · ordered-endpoint role": ViewSpec(
        "wavelet_incidence_transport/{image}/ordered_endpoints.npz",
        "role_kernel"),
    "Incidence · ordered endpoints + proposals": ViewSpec(
        "wavelet_incidence_transport/{image}/ordered_endpoints.npz",
        "proposal_transported_complete_kernel"),
    "NULL · shuffled ordered endpoints + proposals": ViewSpec(
        "wavelet_incidence_transport/{image}/shuffled_ordered_endpoints.npz",
        "proposal_transported_complete_kernel"),
    "Split diffusion · heat": ViewSpec(
        "wavelet_split_transport/{image}/full.npz", "split_heat_kernel"),
    "Split diffusion · transported parts": ViewSpec(
        "wavelet_split_transport/{image}/full.npz",
        "transported_base_kernel"),
    "NULL · shuffled split diffusion": ViewSpec(
        "wavelet_split_transport/{image}/shuffled_content_alignment.npz",
        "transported_base_kernel"),
    "Role-gated content · connection": ViewSpec(
        "wavelet_gated_transport/{image}/full.npz", "content_connection"),
    "Role-gated content · transported parts": ViewSpec(
        "wavelet_gated_transport/{image}/full.npz",
        "transported_base_kernel"),
    "NULL · shuffled role-gated content": ViewSpec(
        "wavelet_gated_transport/{image}/shuffled_content_alignment.npz",
        "transported_base_kernel"),
    "Packet coordinate · raw payload": ViewSpec(
        "object_packet_algebra/{image}/object_packet_algebra.npz",
        "raw_payload"),
    "Packet coordinate · scale-law payload": ViewSpec(
        "object_packet_algebra/{image}/object_packet_algebra.npz",
        "scale_law_payload"),
    "Packet coordinate · ordered role": ViewSpec(
        "object_packet_algebra/{image}/object_packet_algebra.npz",
        "ordered_endpoint_role"),
    "Packet scalar · complete (rejected)": ViewSpec(
        "object_packet_algebra/{image}/object_packet_algebra.npz", "complete"),
    "NULL · shuffled scalar packet": ViewSpec(
        "object_packet_algebra/{image}/object_packet_algebra.npz",
        "shuffled_complete"),
}


DEFAULT_ANCHOR = {
    "pikachu_hard": "body",
    "coffee": "cup_wall",
    "astronaut": "flag_blue",
    "checker": "black_a",
    "coins": "coin_00",
}


class State:
    def __init__(self, results: Path, landmarks: dict):
        self.results = results
        self.landmarks = landmarks
        self.image_name = ""
        self.source = np.zeros((8, 8, 3), dtype=np.float64)
        self.labels = np.zeros((8, 8), dtype=np.int32)
        self.matrix = np.eye(1)
        self.matrix_path = Path()
        self.matrix_key = ""
        self.anchor_region = 0
        self.anchor_pixel = (0, 0)
        self.anchor_label = "region 0"
        self.texture_ids: dict[str, int] = {}
        self.display_size = (PANEL, PANEL)


S: State


def _rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.shape[2] > 3:
        value = value[..., :3]
    if value.max(initial=0.0) > 1.5:
        value /= 255.0
    return np.clip(value, 0.0, 1.0)


def _magma(field: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(field, dtype=np.float64), 0.0, 1.0)
    stops = np.asarray([
        (0.00, 0, 0, 4),
        (0.20, 45, 17, 95),
        (0.40, 127, 30, 110),
        (0.60, 210, 62, 78),
        (0.80, 249, 142, 8),
        (1.00, 252, 253, 191),
    ], dtype=np.float64)
    position = value * (len(stops) - 1)
    lower = np.minimum(position.astype(np.int32), len(stops) - 2)
    fraction = position - lower
    return (
        stops[lower, 1:] + fraction[..., None]
        * (stops[lower + 1, 1:] - stops[lower, 1:])
    ) / 255.0


def _signed(field: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(field, dtype=np.float64), -1.0, 1.0)
    negative = np.clip(-value, 0.0, 1.0)
    positive = np.clip(value, 0.0, 1.0)
    neutral = 1.0 - np.maximum(negative, positive)
    blue = np.asarray((0.10, 0.34, 0.95))
    red = np.asarray((0.98, 0.29, 0.12))
    gray = np.asarray((0.08, 0.08, 0.10))
    return (
        negative[..., None] * blue + positive[..., None] * red
        + neutral[..., None] * gray)


def _region_boundaries(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return boundary


def _anchor_boundary(labels: np.ndarray, anchor: int) -> np.ndarray:
    member = labels == anchor
    interior = member.copy()
    interior[1:] &= member[:-1]
    interior[:-1] &= member[1:]
    interior[:, 1:] &= member[:, :-1]
    interior[:, :-1] &= member[:, 1:]
    return member & ~interior


def _rgba(image: np.ndarray) -> np.ndarray:
    rgb = _rgb(image).astype(np.float32)
    alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
    return np.concatenate((rgb, alpha), axis=2).ravel()


def _allocate_textures(height: int, width: int) -> None:
    scale = PANEL / max(height, width)
    display_width = max(1, int(round(width * scale)))
    display_height = max(1, int(round(height * scale)))
    S.display_size = (display_width, display_height)
    old = dict(S.texture_ids)
    S.texture_ids.clear()
    for name in TEXTURES:
        item = dpg.generate_uuid()
        dpg.add_dynamic_texture(
            width=width,
            height=height,
            default_value=np.ones(height * width * 4, dtype=np.float32),
            tag=item,
            parent="v3_texture_registry",
        )
        S.texture_ids[name] = item
        dpg.configure_item(
            f"v3_{name}_image", texture_tag=item,
            width=display_width, height=display_height)
    for item in old.values():
        if dpg.does_item_exist(item):
            dpg.delete_item(item)


def _push(name: str, image: np.ndarray) -> None:
    dpg.set_value(S.texture_ids[name], _rgba(image))


def _point_region(name: str) -> tuple[int, tuple[int, int]]:
    xy = S.landmarks[S.image_name][name]["xy"]
    x = int(round(float(xy[0]) * (S.labels.shape[1] - 1)))
    y = int(round(float(xy[1]) * (S.labels.shape[0] - 1)))
    return int(S.labels[y, x]), (x, y)


def _set_landmark(name: str) -> None:
    region, pixel = _point_region(name)
    S.anchor_region = region
    S.anchor_pixel = pixel
    S.anchor_label = name
    dpg.set_value("v3_anchor_combo", name)
    refresh()


def _load_matrix() -> None:
    view = dpg.get_value("v3_view_combo")
    spec = VIEWS[view]
    path = S.results / spec.path.format(image=S.image_name)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        if spec.key not in data.files:
            raise KeyError(f"{spec.key}; available: {', '.join(data.files)}")
        matrix = np.asarray(data[spec.key], dtype=np.float64)
    region_count = int(S.labels.max()) + 1
    if matrix.shape != (region_count, region_count):
        raise ValueError(
            f"matrix {matrix.shape} does not match {region_count} V3 regions")
    S.matrix = matrix
    S.matrix_path = path
    S.matrix_key = spec.key


def _load_image(name: str) -> None:
    image_dir = S.results / name
    S.source = _rgb(np.asarray(Image.open(image_dir / "source.png").convert(
        "RGB")))
    with np.load(image_dir / "v3_stages.npz") as data:
        S.labels = np.asarray(data["compound_labels"], dtype=np.int32)
    S.image_name = name
    _allocate_textures(*S.source.shape[:2])
    landmark_names = list(S.landmarks[name])
    dpg.configure_item("v3_anchor_combo", items=landmark_names)
    anchor = DEFAULT_ANCHOR.get(name, landmark_names[0])
    S.anchor_region, S.anchor_pixel = _point_region(anchor)
    S.anchor_label = anchor
    dpg.set_value("v3_anchor_combo", anchor)
    _load_matrix()
    refresh()


def _landmark_inspector(field: np.ndarray) -> str:
    rows = []
    for name, record in S.landmarks[S.image_name].items():
        region, _ = _point_region(name)
        rows.append((name, record["instance"], region, float(field[region])))
    width = max(len(row[0]) for row in rows)
    return "\n".join(
        f"{name:<{width}}  r{region:<5d}  {value:+.6f}  [{instance}]"
        for name, instance, region, value in rows)


def refresh() -> None:
    if not S.image_name:
        return
    anchor = int(np.clip(S.anchor_region, 0, len(S.matrix) - 1))
    field = np.asarray(S.matrix[:, anchor], dtype=np.float64)
    pixel_field = field[S.labels]
    signed = bool(dpg.get_value("v3_signed"))
    heat = _signed(pixel_field) if signed else _magma(pixel_field)
    boundaries = _region_boundaries(S.labels)
    heat[boundaries] *= 0.30
    anchor_edge = _anchor_boundary(S.labels, anchor)
    heat[anchor_edge] = (0.10, 1.00, 0.86)

    source = S.source.copy()
    source[anchor_edge] = (0.10, 1.00, 0.86)
    alpha = np.clip(pixel_field, 0.0, 1.0)
    recomposition = source * (0.10 + 0.90 * alpha[..., None])
    recomposition[anchor_edge] = (0.10, 1.00, 0.86)
    _push("source", source)
    _push("heat", heat)
    _push("recomposition", recomposition)

    positive = np.clip(field, 0.0, None)
    status = (
        f"{S.image_name} | {len(field):,} compound regions | "
        f"anchor {S.anchor_label} = r{anchor} at {S.anchor_pixel} | "
        f"range [{field.min():+.5f}, {field.max():+.5f}] | "
        f"positive mass {positive.sum():.3f}\n"
        f"{S.matrix_path.relative_to(S.results)} :: {S.matrix_key}\n\n"
        f"Landmark similarities to r{anchor}:\n{_landmark_inspector(field)}"
    )
    dpg.set_value("v3_inspector", status)


def cb_image(sender, value) -> None:
    try:
        dpg.set_value("v3_status", "Loading image and saved V3 atoms…")
        _load_image(value)
        dpg.set_value("v3_status", "Ready. Click any panel to change anchor.")
    except Exception as error:
        dpg.set_value("v3_status", f"Load failed: {type(error).__name__}: {error}")


def cb_view(sender=None, value=None) -> None:
    try:
        dpg.set_value("v3_status", "Loading saved operator matrix…")
        _load_matrix()
        refresh()
        dpg.set_value("v3_status", "Ready. No inference was rerun.")
    except Exception as error:
        dpg.set_value("v3_status", f"View unavailable: {type(error).__name__}: {error}")


def cb_anchor(sender, value) -> None:
    try:
        _set_landmark(value)
    except Exception as error:
        dpg.set_value("v3_status", f"Anchor failed: {type(error).__name__}: {error}")


def cb_click() -> None:
    hovered = next((
        f"v3_{name}_image" for name in TEXTURES
        if dpg.is_item_hovered(f"v3_{name}_image")
    ), None)
    if hovered is None:
        return
    left, top = dpg.get_item_rect_min(hovered)
    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    display_width, display_height = S.display_size
    x = int(np.clip(
        (mouse_x - left) / max(display_width, 1) * S.labels.shape[1],
        0, S.labels.shape[1] - 1))
    y = int(np.clip(
        (mouse_y - top) / max(display_height, 1) * S.labels.shape[0],
        0, S.labels.shape[0] - 1))
    S.anchor_region = int(S.labels[y, x])
    S.anchor_pixel = (x, y)
    S.anchor_label = f"clicked ({x}, {y})"
    dpg.set_value("v3_status", f"Selected compound region {S.anchor_region}.")
    refresh()


def _available_images(results: Path, landmarks: dict) -> list[str]:
    return [
        name for name in landmarks
        if (results / name / "source.png").exists()
        and (results / name / "v3_stages.npz").exists()
    ]


def build_ui(images: list[str], initial_image: str, initial_view: str) -> None:
    with dpg.texture_registry(tag="v3_texture_registry"):
        dpg.add_static_texture(
            width=1, height=1, default_value=(0.08, 0.08, 0.10, 1.0),
            tag="v3_placeholder_texture")
    with dpg.window(tag="v3_root", label="V3 Object Transport Results"):
        dpg.add_text("Saved V3 object-transport microscope", color=(110, 235, 210))
        dpg.add_text(str(S.results), color=(155, 155, 165))
        with dpg.group(horizontal=True):
            dpg.add_text("Image")
            dpg.add_combo(
                images, default_value=initial_image, tag="v3_image_combo",
                callback=cb_image, width=170)
            dpg.add_text("Anchor")
            dpg.add_combo(
                [], tag="v3_anchor_combo", callback=cb_anchor, width=190)
            dpg.add_checkbox(
                label="signed heat", tag="v3_signed", default_value=False,
                callback=lambda: refresh())
        with dpg.group(horizontal=True):
            dpg.add_text("Saved operator")
            dpg.add_combo(
                list(VIEWS), default_value=initial_view, tag="v3_view_combo",
                callback=cb_view, width=520)
        dpg.add_text(
            "Landmarks are evaluation-only. Click source, heat, or recomposition "
            "to inspect any compound region. Cyan outlines the anchor atom.")
        dpg.add_text("Starting…", tag="v3_status", color=(240, 190, 90))
        with dpg.group(horizontal=True):
            for name, title in (
                ("source", "Source + anchor V3 atom"),
                ("heat", "Saved participation field"),
                ("recomposition", "Continuous-alpha recomposition"),
            ):
                with dpg.group():
                    dpg.add_text(title)
                    dpg.add_image(
                        "v3_placeholder_texture", tag=f"v3_{name}_image")
        dpg.add_text("", tag="v3_inspector", wrap=1380)
    with dpg.handler_registry():
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                    callback=lambda: cb_click())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--image")
    parser.add_argument("--view", choices=tuple(VIEWS))
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    landmarks = json.loads(LANDMARKS.read_text())["images"]
    images = _available_images(results, landmarks)
    if not images:
        raise SystemExit(f"no V3 result images found under {results}")
    initial_image = args.image if args.image in images else images[0]
    if "pikachu_hard" in images and args.image is None:
        initial_image = "pikachu_hard"
    initial_view = args.view or "Incidence · transition + proposals"
    global S
    S = State(results, landmarks)
    dpg.create_context()
    build_ui(images, initial_image, initial_view)
    dpg.create_viewport(
        title="BFFT — V3 Object Transport Results",
        width=1420,
        height=900,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("v3_root", True)
    cb_image(None, initial_image)
    dpg.start_dearpygui()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
