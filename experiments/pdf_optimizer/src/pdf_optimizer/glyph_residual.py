"""Canonical glyph envelopes and foreground-owned MRC residuals.

This is an experimental representation model, not a production PDF pass.  It
tests whether scan-specific binary edge variation can be removed from the
JBIG2 mask and represented more cheaply by the continuous foreground plane.

The construction deliberately keeps three ideas separate:

* signed-distance fields lift bitmap components into an eikonal height field;
* Fourier-circle energy is an alignment/matching confidence, not an image
  prior;
* the Meyer cartoon determines reusable topology while the exact source mask
  is used to form a common superset envelope for every member of a cluster.

Because each envelope contains every aligned source occurrence, replacing a
source mask ``M`` by the canonical mask ``C`` never removes opacity.  The
decoded source composite can therefore be moved into the foreground wherever
``C`` is set, including paper-coloured pixels added by the envelope.  Before
foreground coding this is an exact change of representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import bz2
import math
import zlib
from typing import Callable, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


@dataclass(frozen=True)
class GlyphResidualConfig:
    """Controls for one page-level canonical-envelope experiment."""

    minimum_area: int = 10
    maximum_area: int = 2_000
    minimum_repetitions: int = 4
    component_padding: int = 3
    descriptor_size: int = 24
    radial_bins: int = 8
    dimension_tolerance: float = 0.18
    descriptor_distance: float = 0.30
    spatial_distance: float = 0.02
    maximum_shift: int = 2
    envelope_dilation: int = 0
    meyer_virtual_passes: int = 8

    def __post_init__(self) -> None:
        if self.minimum_area < 1 or self.maximum_area < self.minimum_area:
            raise ValueError("invalid component-area interval")
        if self.minimum_repetitions < 2:
            raise ValueError("minimum_repetitions must be at least two")
        if self.component_padding < 1:
            raise ValueError("component_padding must be positive")
        if self.descriptor_size < 8 or self.radial_bins < 2:
            raise ValueError("descriptor resolution is too small")
        if not 0 <= self.dimension_tolerance < 1:
            raise ValueError("dimension_tolerance must be in [0, 1)")
        if self.maximum_shift < 0:
            raise ValueError("maximum_shift must be non-negative")
        if self.envelope_dilation < 0:
            raise ValueError("envelope_dilation must be non-negative")


@dataclass(frozen=True)
class Component:
    label: int
    y0: int
    x0: int
    y1: int
    x1: int
    area: int
    centroid_y: float
    centroid_x: float
    radial: np.ndarray
    spatial: np.ndarray

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


@dataclass(frozen=True)
class CanonicalMaskResult:
    source_mask: np.ndarray
    canonical_mask: np.ndarray
    labels: np.ndarray
    components: tuple[Component, ...]
    clusters: tuple[tuple[int, ...], ...]
    clustered_components: int
    added_pixels: int
    removed_pixels: int
    meyer_used: bool

    def diagnostics(self) -> dict:
        values = {
            "components": len(self.components),
            "clusters": len(self.clusters),
            "cluster_sizes": [len(cluster) for cluster in self.clusters],
            "clustered_components": self.clustered_components,
            "source_pixels": int(np.count_nonzero(self.source_mask)),
            "canonical_pixels": int(np.count_nonzero(self.canonical_mask)),
            "added_pixels": self.added_pixels,
            "removed_pixels": self.removed_pixels,
            "source_fraction": float(np.mean(self.source_mask)),
            "canonical_fraction": float(np.mean(self.canonical_mask)),
            "meyer_used": self.meyer_used,
        }
        return values


@dataclass(frozen=True)
class GlobalCanonicalMaskResult:
    source_masks: tuple[np.ndarray, ...]
    canonical_masks: tuple[np.ndarray, ...]
    cluster_sizes: tuple[int, ...]
    component_counts: tuple[int, ...]
    clustered_components: int
    added_pixels: tuple[int, ...]
    removed_pixels: tuple[int, ...]
    meyer_used: bool

    def diagnostics(self) -> dict:
        return {
            "pages": len(self.source_masks),
            "components": list(self.component_counts),
            "clusters": len(self.cluster_sizes),
            "cluster_sizes": list(self.cluster_sizes),
            "clustered_components": self.clustered_components,
            "source_pixels": [int(np.count_nonzero(mask)) for mask in self.source_masks],
            "canonical_pixels": [int(np.count_nonzero(mask)) for mask in self.canonical_masks],
            "added_pixels": list(self.added_pixels),
            "removed_pixels": list(self.removed_pixels),
            "meyer_used": self.meyer_used,
        }


@dataclass(frozen=True)
class AtlasOccurrence:
    """One placement of a cropped shared glyph core."""

    page: int
    glyph: int
    y: int
    x: int


@dataclass(frozen=True)
class AtlasCorrection:
    """One cropped page-local ink exception to a shared glyph core."""

    occurrence: int
    page: int
    y: int
    x: int
    bitmap: np.ndarray


@dataclass(frozen=True)
class GlobalCoreAtlasResult:
    """Exact decomposition of source masks into atlas cores and residual ink."""

    source_masks: tuple[np.ndarray, ...]
    core_masks: tuple[np.ndarray, ...]
    residual_masks: tuple[np.ndarray, ...]
    remainder_masks: tuple[np.ndarray, ...]
    glyphs: tuple[np.ndarray, ...]
    occurrences: tuple[AtlasOccurrence, ...]
    corrections: tuple[AtlasCorrection, ...]
    cluster_sizes: tuple[int, ...]
    component_counts: tuple[int, ...]
    meyer_used: bool
    core_mode: str

    def diagnostics(self) -> dict:
        source_pixels = [int(np.count_nonzero(mask)) for mask in self.source_masks]
        core_pixels = [int(np.count_nonzero(mask)) for mask in self.core_masks]
        residual_pixels = [int(np.count_nonzero(mask)) for mask in self.residual_masks]
        return {
            "pages": len(self.source_masks),
            "components": list(self.component_counts),
            "glyphs": len(self.glyphs),
            "cluster_sizes": list(self.cluster_sizes),
            "occurrences": len(self.occurrences),
            "source_pixels": source_pixels,
            "core_pixels": core_pixels,
            "residual_pixels": residual_pixels,
            "remainder_pixels": [
                int(np.count_nonzero(mask)) for mask in self.remainder_masks
            ],
            "correction_pixels": sum(
                int(np.count_nonzero(item.bitmap)) for item in self.corrections
            ),
            "corrections": len(self.corrections),
            "core_fraction": [
                core / source if source else 0.0
                for core, source in zip(core_pixels, source_pixels)
            ],
            "meyer_used": self.meyer_used,
            "core_mode": self.core_mode,
        }


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Return a positive-inside Euclidean signed-distance height field."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("signed_distance expects a two-dimensional mask")
    if not binary.any():
        return -ndi.distance_transform_edt(~binary)
    if binary.all():
        return ndi.distance_transform_edt(binary)
    return ndi.distance_transform_edt(binary) - ndi.distance_transform_edt(~binary)


def _resample(field: np.ndarray, size: int, *, order: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(field, dtype=np.float32), mode="F")
    method = Image.Resampling.BILINEAR if order else Image.Resampling.NEAREST
    return np.asarray(image.resize((size, size), method), dtype=np.float64)


def fourier_circle_descriptor(mask: np.ndarray, *, size: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return radial Fourier energy and a coarse signed-distance shape.

    Circle pooling makes the first half insensitive to direction.  The second
    half intentionally retains a low-resolution spatial check so rotationally
    different glyphs with similar spectra are not fused.
    """

    height = signed_distance(mask)
    scale = max(float(np.max(np.abs(height))), 1.0)
    sampled = _resample(height / scale, size, order=1)
    sampled -= float(np.mean(sampled))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(sampled))) ** 2
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(yy - (size - 1) / 2.0, xx - (size - 1) / 2.0)
    edges = np.linspace(0.0, float(radius.max()) + 1e-9, bins + 1)
    radial = np.array(
        [float(np.mean(spectrum[(radius >= a) & (radius < b)])) for a, b in zip(edges[:-1], edges[1:])],
        dtype=np.float64,
    )
    radial = np.log1p(radial)
    radial /= max(float(np.linalg.norm(radial)), 1e-12)
    spatial_size = max(6, size // 3)
    spatial = _resample(height / scale, spatial_size, order=1)
    spatial /= max(float(np.linalg.norm(spatial)), 1e-12)
    return radial, spatial.ravel()


def extract_components(mask: np.ndarray, config: GlyphResidualConfig) -> tuple[np.ndarray, tuple[Component, ...]]:
    """Label eligible eight-connected components and measure descriptors."""

    source = np.asarray(mask, dtype=bool)
    labels, count = ndi.label(source, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndi.find_objects(labels, max_label=count)
    components: list[Component] = []
    for label, bounds in enumerate(objects, 1):
        if bounds is None:
            continue
        ys, xs = bounds
        local = labels[ys, xs] == label
        area = int(np.count_nonzero(local))
        if area < config.minimum_area or area > config.maximum_area:
            continue
        height, width = local.shape
        # Rule out long rules and page decorations before descriptor work.
        if max(height / max(width, 1), width / max(height, 1)) > 12.0:
            continue
        cy, cx = ndi.center_of_mass(local)
        radial, spatial = fourier_circle_descriptor(
            local, size=config.descriptor_size, bins=config.radial_bins
        )
        components.append(
            Component(
                label=label,
                y0=int(ys.start),
                x0=int(xs.start),
                y1=int(ys.stop),
                x1=int(xs.stop),
                area=area,
                centroid_y=float(cy),
                centroid_x=float(cx),
                radial=radial,
                spatial=spatial,
            )
        )
    return labels, tuple(components)


def _relative_difference(left: int, right: int) -> float:
    return abs(left - right) / max(left, right, 1)


def cluster_components(
    components: Iterable[Component], config: GlyphResidualConfig
) -> tuple[tuple[int, ...], ...]:
    """Greedily form conservative appearance clusters.

    The representative is the first component.  A cluster is retained only
    when it reaches the configured repetition count; rejected small groups do
    not change the source mask.
    """

    values = tuple(components)
    order = sorted(range(len(values)), key=lambda index: (-values[index].area, index))
    groups: list[list[int]] = []
    dimension_index: dict[tuple[int, int], list[int]] = {}
    for index in order:
        item = values[index]
        best: tuple[float, int] | None = None
        low_h = max(1, math.ceil(item.height * (1.0 - config.dimension_tolerance)))
        high_h = max(low_h, math.floor(item.height / (1.0 - config.dimension_tolerance)))
        low_w = max(1, math.ceil(item.width * (1.0 - config.dimension_tolerance)))
        high_w = max(low_w, math.floor(item.width / (1.0 - config.dimension_tolerance)))
        candidates = {
            group_index
            for height in range(low_h, high_h + 1)
            for width in range(low_w, high_w + 1)
            for group_index in dimension_index.get((height, width), ())
        }
        for group_index in candidates:
            group = groups[group_index]
            representative = values[group[0]]
            if _relative_difference(item.height, representative.height) > config.dimension_tolerance:
                continue
            if _relative_difference(item.width, representative.width) > config.dimension_tolerance:
                continue
            radial = float(np.linalg.norm(item.radial - representative.radial))
            spatial = float(np.sqrt(np.mean((item.spatial - representative.spatial) ** 2)))
            if radial > config.descriptor_distance or spatial > config.spatial_distance:
                continue
            score = radial + 2.0 * spatial
            if best is None or score < best[0]:
                best = (score, group_index)
        if best is None:
            groups.append([index])
            dimension_index.setdefault((item.height, item.width), []).append(
                len(groups) - 1
            )
        else:
            groups[best[1]].append(index)
    retained = [tuple(group) for group in groups if len(group) >= config.minimum_repetitions]
    return tuple(sorted(retained, key=lambda group: (-len(group), group[0])))


def _component_patch(labels: np.ndarray, component: Component) -> np.ndarray:
    return labels[component.y0 : component.y1, component.x0 : component.x1] == component.label


def _canvas_geometry(group: Iterable[Component], padding: int) -> tuple[int, int]:
    items = tuple(group)
    return (
        max(item.height for item in items) + 2 * padding,
        max(item.width for item in items) + 2 * padding,
    )


def _centered_patch(patch: np.ndarray, shape: tuple[int, int], shift: tuple[int, int] = (0, 0)) -> np.ndarray:
    target = np.zeros(shape, dtype=bool)
    y0 = (shape[0] - patch.shape[0]) // 2 + shift[0]
    x0 = (shape[1] - patch.shape[1]) // 2 + shift[1]
    y1 = y0 + patch.shape[0]
    x1 = x0 + patch.shape[1]
    if y0 < 0 or x0 < 0 or y1 > shape[0] or x1 > shape[1]:
        raise ValueError("component patch does not fit canonical canvas")
    target[y0:y1, x0:x1] = patch
    return target


def _best_shift(reference: np.ndarray, patch: np.ndarray, maximum: int) -> tuple[int, int]:
    best = (-math.inf, 0, 0)
    signed = signed_distance(patch)
    for dy in range(-maximum, maximum + 1):
        for dx in range(-maximum, maximum + 1):
            shifted = ndi.shift(signed, (dy, dx), order=0, mode="constant", cval=float(signed.min()))
            score = float(np.sum(reference * shifted))
            if score > best[0]:
                best = (score, dy, dx)
    return best[1], best[2]


def _default_meyer(field: np.ndarray, virtual_passes: int) -> tuple[np.ndarray, np.ndarray]:
    from bfft import meyer_split_jump_measure

    return meyer_split_jump_measure(field, virtual_passes=virtual_passes)


def canonicalize_mask(
    mask: np.ndarray,
    config: GlyphResidualConfig = GlyphResidualConfig(),
    *,
    meyer_splitter: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
    precomputed: CanonicalMaskResult | None = None,
    selected_cluster_indices: Iterable[int] | None = None,
) -> CanonicalMaskResult:
    """Replace recurring components by common Meyer-cartoon envelopes."""

    source = np.asarray(mask, dtype=bool)
    if precomputed is None:
        labels, components = extract_components(source, config)
        clusters = cluster_components(components, config)
    else:
        if not np.array_equal(source, precomputed.source_mask):
            raise ValueError("precomputed components belong to a different source mask")
        labels = precomputed.labels
        components = precomputed.components
        clusters = precomputed.clusters
    enabled = (
        None
        if selected_cluster_indices is None
        else frozenset(int(index) for index in selected_cluster_indices)
    )
    canonical = source.copy()
    splitter = meyer_splitter or _default_meyer
    meyer_used = True
    clustered = 0

    for cluster_index, cluster in enumerate(clusters):
        if enabled is not None and cluster_index not in enabled:
            continue
        members = [components[index] for index in cluster]
        shape = _canvas_geometry(members, config.component_padding + config.maximum_shift)
        initial = [
            _centered_patch(_component_patch(labels, member), shape)
            for member in members
        ]
        reference = signed_distance(initial[0])
        aligned: list[np.ndarray] = []
        shifts: list[tuple[int, int]] = []
        cartoons: list[np.ndarray] = []
        for member, patch in zip(members, initial):
            shift = _best_shift(reference, patch, config.maximum_shift)
            centered = _centered_patch(
                _component_patch(labels, member), shape, shift=shift
            )
            height = signed_distance(centered)
            height /= max(float(np.max(np.abs(height))), 1.0)
            try:
                cartoon, _texture = splitter(height, config.meyer_virtual_passes)
            except (ImportError, OSError, RuntimeError):
                # The exact experiment reports this fallback; it is retained
                # for synthetic tests and hosts without the native library.
                cartoon = ndi.gaussian_filter(height, 0.75, mode="nearest")
                meyer_used = False
            aligned.append(centered)
            shifts.append(shift)
            cartoons.append(np.asarray(cartoon, dtype=np.float64))

        # The Meyer median is the reusable topology estimate.  The opacity
        # support remains the exact aligned union: expanding the support from
        # a smoothed core was measured to double coverage and made the
        # foreground residual dominate all mask savings.
        _median_cartoon = np.median(np.stack(cartoons), axis=0)
        union = np.logical_or.reduce(aligned)
        if config.envelope_dilation:
            union = ndi.binary_dilation(union, iterations=config.envelope_dilation)
        envelope = union

        for member, shift in zip(members, shifts):
            canonical[labels == member.label] = False
            base_y = member.y0 - (shape[0] - member.height) // 2 - shift[0]
            base_x = member.x0 - (shape[1] - member.width) // 2 - shift[1]
            y0 = max(0, base_y)
            x0 = max(0, base_x)
            y1 = min(canonical.shape[0], base_y + shape[0])
            x1 = min(canonical.shape[1], base_x + shape[1])
            ey0 = y0 - base_y
            ex0 = x0 - base_x
            canonical[y0:y1, x0:x1] |= envelope[
                ey0 : ey0 + (y1 - y0), ex0 : ex0 + (x1 - x0)
            ]
            clustered += 1

    removed = int(np.count_nonzero(source & ~canonical))
    added = int(np.count_nonzero(canonical & ~source))
    if removed:
        raise RuntimeError("canonical envelope violated the source-superset invariant")
    return CanonicalMaskResult(
        source_mask=source,
        canonical_mask=canonical,
        labels=labels,
        components=components,
        clusters=clusters,
        clustered_components=clustered,
        added_pixels=added,
        removed_pixels=removed,
        meyer_used=meyer_used,
    )


def canonicalize_masks_global(
    masks: Iterable[np.ndarray],
    config: GlyphResidualConfig = GlyphResidualConfig(),
    *,
    meyer_splitter: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> GlobalCanonicalMaskResult:
    """Build one set of canonical component envelopes across several pages.

    This is the font-atlas form of :func:`canonicalize_mask`: clustering is
    global, so every occurrence in one cluster receives exactly the same
    aligned envelope even when it appears on a different page.
    """

    sources = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    if not sources:
        raise ValueError("at least one mask is required")
    labelled: list[np.ndarray] = []
    page_components: list[tuple[Component, ...]] = []
    occurrences: list[tuple[int, Component]] = []
    for page_index, source in enumerate(sources):
        labels, components = extract_components(source, config)
        labelled.append(labels)
        page_components.append(components)
        occurrences.extend((page_index, component) for component in components)
    components = tuple(component for _page, component in occurrences)
    clusters = cluster_components(components, config)
    canonical = [source.copy() for source in sources]
    splitter = meyer_splitter or _default_meyer
    meyer_used = True
    clustered = 0

    for cluster in clusters:
        members = [occurrences[index] for index in cluster]
        component_values = [component for _page, component in members]
        shape = _canvas_geometry(
            component_values, config.component_padding + config.maximum_shift
        )
        initial = [
            _centered_patch(
                _component_patch(labelled[page_index], component), shape
            )
            for page_index, component in members
        ]
        reference = signed_distance(initial[0])
        aligned: list[np.ndarray] = []
        shifts: list[tuple[int, int]] = []
        cartoons: list[np.ndarray] = []
        for (page_index, component), patch in zip(members, initial):
            shift = _best_shift(reference, patch, config.maximum_shift)
            centered = _centered_patch(
                _component_patch(labelled[page_index], component), shape, shift=shift
            )
            height = signed_distance(centered)
            height /= max(float(np.max(np.abs(height))), 1.0)
            try:
                cartoon, _texture = splitter(height, config.meyer_virtual_passes)
            except (ImportError, OSError, RuntimeError):
                cartoon = ndi.gaussian_filter(height, 0.75, mode="nearest")
                meyer_used = False
            aligned.append(centered)
            shifts.append(shift)
            cartoons.append(np.asarray(cartoon, dtype=np.float64))
        # The shared cartoon is the glyph identity.  Exact opacity remains the
        # aligned union so the source-superset invariant is maintained.
        _shared_cartoon = np.median(np.stack(cartoons), axis=0)
        envelope = np.logical_or.reduce(aligned)
        if config.envelope_dilation:
            envelope = ndi.binary_dilation(
                envelope, iterations=config.envelope_dilation
            )

        for (page_index, component), shift in zip(members, shifts):
            page_mask = canonical[page_index]
            page_mask[labelled[page_index] == component.label] = False
            base_y = component.y0 - (shape[0] - component.height) // 2 - shift[0]
            base_x = component.x0 - (shape[1] - component.width) // 2 - shift[1]
            y0 = max(0, base_y)
            x0 = max(0, base_x)
            y1 = min(page_mask.shape[0], base_y + shape[0])
            x1 = min(page_mask.shape[1], base_x + shape[1])
            ey0 = y0 - base_y
            ex0 = x0 - base_x
            page_mask[y0:y1, x0:x1] |= envelope[
                ey0 : ey0 + (y1 - y0), ex0 : ex0 + (x1 - x0)
            ]
            clustered += 1

    removed = tuple(
        int(np.count_nonzero(source & ~candidate))
        for source, candidate in zip(sources, canonical)
    )
    if any(removed):
        raise RuntimeError("global canonical envelope violated the source-superset invariant")
    added = tuple(
        int(np.count_nonzero(candidate & ~source))
        for source, candidate in zip(sources, canonical)
    )
    return GlobalCanonicalMaskResult(
        source_masks=sources,
        canonical_masks=tuple(canonical),
        cluster_sizes=tuple(len(cluster) for cluster in clusters),
        component_counts=tuple(len(values) for values in page_components),
        clustered_components=clustered,
        added_pixels=added,
        removed_pixels=removed,
        meyer_used=meyer_used,
    )


def canonicalize_masks_global_core(
    masks: Iterable[np.ndarray],
    config: GlyphResidualConfig = GlyphResidualConfig(),
    *,
    meyer_splitter: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
    cartoon_floor: float = 0.0,
    core_mode: str = "intersection",
) -> GlobalCoreAtlasResult:
    """Factor recurring ink into conservative shared cores and exact residuals.

    Unlike :func:`canonicalize_masks_global`, this construction takes the
    aligned *intersection*.  A shared core is therefore a subset of every
    source occurrence.  The source bitmap is recovered exactly as
    ``core | residual``; no paper-coloured foreground correction is needed.

    Meyer cartoon support is allowed to shrink the exact intersection but
    never enlarge it.  This deliberately gives texture and damaged edges to
    the page-local residual while reserving stable glyph topology for the
    atlas.
    """

    if core_mode not in {"intersection", "median"}:
        raise ValueError("core_mode must be 'intersection' or 'median'")
    sources = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    if not sources:
        raise ValueError("at least one mask is required")
    labelled: list[np.ndarray] = []
    page_components: list[tuple[Component, ...]] = []
    occurrence_values: list[tuple[int, Component]] = []
    for page_index, source in enumerate(sources):
        labels, components = extract_components(source, config)
        labelled.append(labels)
        page_components.append(components)
        occurrence_values.extend((page_index, component) for component in components)

    components = tuple(component for _page, component in occurrence_values)
    clusters = cluster_components(components, config)
    core_pages = [np.zeros_like(source, dtype=bool) for source in sources]
    remainder_pages = [source.copy() for source in sources]
    splitter = meyer_splitter or _default_meyer
    meyer_used = True
    glyphs: list[np.ndarray] = []
    placements: list[AtlasOccurrence] = []
    corrections: list[AtlasCorrection] = []
    retained_sizes: list[int] = []

    for cluster in clusters:
        members = [occurrence_values[index] for index in cluster]
        component_values = [component for _page, component in members]
        shape = _canvas_geometry(
            component_values, config.component_padding + config.maximum_shift
        )
        initial = [
            _centered_patch(
                _component_patch(labelled[page_index], component), shape
            )
            for page_index, component in members
        ]
        reference = signed_distance(initial[0])
        aligned: list[np.ndarray] = []
        shifts: list[tuple[int, int]] = []
        cartoons: list[np.ndarray] = []
        for (page_index, component), patch in zip(members, initial):
            shift = _best_shift(reference, patch, config.maximum_shift)
            centered = _centered_patch(
                _component_patch(labelled[page_index], component), shape, shift=shift
            )
            height = signed_distance(centered)
            height /= max(float(np.max(np.abs(height))), 1.0)
            try:
                cartoon, _texture = splitter(height, config.meyer_virtual_passes)
            except (ImportError, OSError, RuntimeError):
                cartoon = ndi.gaussian_filter(height, 0.75, mode="nearest")
                meyer_used = False
            aligned.append(centered)
            shifts.append(shift)
            cartoons.append(np.asarray(cartoon, dtype=np.float64))

        shared_cartoon = np.median(np.stack(cartoons), axis=0)
        stable_support = (
            np.logical_and.reduce(aligned)
            if core_mode == "intersection"
            else np.mean(np.stack(aligned), axis=0) >= 0.5
        )
        core = stable_support & (shared_cartoon >= float(cartoon_floor))
        bounds = ndi.find_objects(core.astype(np.uint8), max_label=1)
        if not bounds or bounds[0] is None:
            continue
        ys, xs = bounds[0]
        cropped = core[ys, xs].copy()
        glyph_index = len(glyphs)

        pending: list[tuple[int, Component, int, int, int, int, np.ndarray]] = []
        valid = True
        for (page_index, component), shift in zip(members, shifts):
            base_y = component.y0 - (shape[0] - component.height) // 2 - shift[0]
            base_x = component.x0 - (shape[1] - component.width) // 2 - shift[1]
            y = base_y + int(ys.start)
            x = base_x + int(xs.start)
            page = core_pages[page_index]
            if y < 0 or x < 0 or y + cropped.shape[0] > page.shape[0] or x + cropped.shape[1] > page.shape[1]:
                valid = False
                break
            if core_mode == "intersection" and np.any(
                cropped
                & ~sources[page_index][
                    y : y + cropped.shape[0], x : x + cropped.shape[1]
                ]
            ):
                valid = False
                break
            pending.append(
                (page_index, component, y, x, base_y, base_x, aligned[len(pending)])
            )
        if not valid:
            continue

        glyphs.append(cropped)
        retained_sizes.append(len(cluster))
        for page_index, component, y, x, base_y, base_x, aligned_member in pending:
            core_pages[page_index][y : y + cropped.shape[0], x : x + cropped.shape[1]] |= cropped
            occurrence_index = len(placements)
            placements.append(AtlasOccurrence(page_index, glyph_index, y, x))
            remainder_pages[page_index][labelled[page_index] == component.label] = False
            correction = aligned_member ^ core
            correction_bounds = ndi.find_objects(
                correction.astype(np.uint8), max_label=1
            )
            if correction_bounds and correction_bounds[0] is not None:
                cys, cxs = correction_bounds[0]
                corrections.append(
                    AtlasCorrection(
                        occurrence_index,
                        page_index,
                        base_y + int(cys.start),
                        base_x + int(cxs.start),
                        correction[cys, cxs].copy(),
                    )
                )

    residuals = tuple(source ^ core for source, core in zip(sources, core_pages))
    correction_pages = [np.zeros_like(source, dtype=bool) for source in sources]
    for item in corrections:
        correction_pages[item.page][
            item.y : item.y + item.bitmap.shape[0],
            item.x : item.x + item.bitmap.shape[1],
        ] |= item.bitmap
    for source, core, residual, remainder, correction in zip(
        sources, core_pages, residuals, remainder_pages, correction_pages
    ):
        if not np.array_equal(source, core ^ residual):
            raise RuntimeError("global glyph core XOR decomposition is not exact")
        if core_mode == "intersection" and not np.array_equal(
            source, (core ^ correction) | remainder
        ):
            raise RuntimeError("glyph-local correction decomposition is not exact")
    return GlobalCoreAtlasResult(
        source_masks=sources,
        core_masks=tuple(core_pages),
        residual_masks=residuals,
        remainder_masks=tuple(remainder_pages),
        glyphs=tuple(glyphs),
        occurrences=tuple(placements),
        corrections=tuple(corrections),
        cluster_sizes=tuple(retained_sizes),
        component_counts=tuple(len(values) for values in page_components),
        meyer_used=meyer_used,
        core_mode=core_mode,
    )


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varints must be non-negative")
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift > 63:
            raise ValueError("invalid atlas varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def serialize_core_atlas(result: GlobalCoreAtlasResult, *, level: int = 9) -> bytes:
    """Serialize a compact experimental atlas and occurrence program.

    The payload is intentionally simple and independently decodable: cropped
    row-packed glyph bitmaps followed by delta-coded placements.  Deflate is
    applied to the whole program so repeated dimensions and placement deltas
    share one entropy model.
    """

    payload = bytearray()
    payload += _encode_varint(len(result.source_masks))
    for mask in result.source_masks:
        payload += _encode_varint(mask.shape[0])
        payload += _encode_varint(mask.shape[1])
    payload += _encode_varint(len(result.glyphs))
    for glyph in result.glyphs:
        packed = np.packbits(glyph, axis=1, bitorder="big").tobytes()
        payload += _encode_varint(glyph.shape[0])
        payload += _encode_varint(glyph.shape[1])
        payload += _encode_varint(len(packed))
        payload += packed

    payload += _encode_varint(len(result.occurrences))
    for item in result.occurrences:
        payload += _encode_varint(item.page)
        payload += _encode_varint(item.y)
        payload += _encode_varint(item.x)
        payload += _encode_varint(item.glyph)
    return b"GCA1" + zlib.compress(bytes(payload), level)


def _decode_core_atlas_program(
    data: bytes,
) -> tuple[list[tuple[int, int]], list[np.ndarray], list[AtlasOccurrence]]:
    if not data.startswith(b"GCA1"):
        raise ValueError("not a glyph-core atlas")
    payload = zlib.decompress(data[4:])
    offset = 0
    page_count, offset = _decode_varint(payload, offset)
    shapes = []
    for _ in range(page_count):
        height, offset = _decode_varint(payload, offset)
        width, offset = _decode_varint(payload, offset)
        shapes.append((height, width))
    glyph_count, offset = _decode_varint(payload, offset)
    glyphs = []
    for _ in range(glyph_count):
        height, offset = _decode_varint(payload, offset)
        width, offset = _decode_varint(payload, offset)
        byte_count, offset = _decode_varint(payload, offset)
        packed = np.frombuffer(payload[offset : offset + byte_count], dtype=np.uint8)
        offset += byte_count
        unpacked = np.unpackbits(packed, bitorder="big").reshape(height, -1)[:, :width]
        glyphs.append(unpacked.astype(bool))
    occurrence_count, offset = _decode_varint(payload, offset)
    occurrences = []
    for _ in range(occurrence_count):
        page, offset = _decode_varint(payload, offset)
        y, offset = _decode_varint(payload, offset)
        x, offset = _decode_varint(payload, offset)
        glyph_index, offset = _decode_varint(payload, offset)
        occurrences.append(AtlasOccurrence(page, glyph_index, y, x))
    if offset != len(payload):
        raise ValueError("trailing bytes in glyph-core atlas")
    return shapes, glyphs, occurrences


def decode_core_atlas(data: bytes) -> tuple[np.ndarray, ...]:
    """Decode :func:`serialize_core_atlas` into page-sized core masks."""

    shapes, glyphs, occurrences = _decode_core_atlas_program(data)
    pages = [np.zeros(shape, dtype=bool) for shape in shapes]
    for item in occurrences:
        glyph = glyphs[item.glyph]
        pages[item.page][
            item.y : item.y + glyph.shape[0], item.x : item.x + glyph.shape[1]
        ] |= glyph
    return tuple(pages)


def serialize_core_corrections(
    result: GlobalCoreAtlasResult, *, level: int = 9
) -> bytes:
    """Serialize cropped per-occurrence missing-ink sprites."""

    payload = bytearray()
    ordered = sorted(result.corrections, key=lambda item: item.occurrence)
    payload += _encode_varint(len(ordered))
    previous_occurrence = 0
    for index, item in enumerate(ordered):
        payload += _encode_varint(
            item.occurrence - previous_occurrence if index else item.occurrence
        )
        payload += _encode_varint(item.page)
        payload += _encode_varint(item.y)
        payload += _encode_varint(item.x)
        payload += _encode_varint(item.bitmap.shape[0])
        payload += _encode_varint(item.bitmap.shape[1])
        packed = np.packbits(item.bitmap, axis=1, bitorder="big").tobytes()
        payload += _encode_varint(len(packed))
        payload += packed
        previous_occurrence = item.occurrence
    return b"GCC1" + zlib.compress(bytes(payload), level)


def decode_core_corrections(
    data: bytes, shapes: Iterable[tuple[int, int]]
) -> tuple[np.ndarray, ...]:
    """Decode :func:`serialize_core_corrections` into page-sized masks."""

    if not data.startswith(b"GCC1"):
        raise ValueError("not a glyph-core correction stream")
    payload = zlib.decompress(data[4:])
    offset = 0
    count, offset = _decode_varint(payload, offset)
    pages = [np.zeros(tuple(shape), dtype=bool) for shape in shapes]
    previous_occurrence = 0
    for index in range(count):
        occurrence_delta, offset = _decode_varint(payload, offset)
        _occurrence = previous_occurrence + occurrence_delta if index else occurrence_delta
        page, offset = _decode_varint(payload, offset)
        y, offset = _decode_varint(payload, offset)
        x, offset = _decode_varint(payload, offset)
        height, offset = _decode_varint(payload, offset)
        width, offset = _decode_varint(payload, offset)
        byte_count, offset = _decode_varint(payload, offset)
        packed = np.frombuffer(payload[offset : offset + byte_count], dtype=np.uint8)
        offset += byte_count
        bitmap = np.unpackbits(packed, bitorder="big").reshape(height, -1)[:, :width]
        pages[page][y : y + height, x : x + width] |= bitmap.astype(bool)
        previous_occurrence = _occurrence
    if offset != len(payload):
        raise ValueError("trailing bytes in glyph-core corrections")
    return tuple(pages)


def decode_refined_core_atlas(atlas_data: bytes, correction_data: bytes) -> tuple[np.ndarray, ...]:
    """Apply each XOR correction to its glyph before page compositing."""

    shapes, glyphs, occurrences = _decode_core_atlas_program(atlas_data)
    if not correction_data.startswith(b"GCC1"):
        raise ValueError("not a glyph-core correction stream")
    payload = zlib.decompress(correction_data[4:])
    offset = 0
    count, offset = _decode_varint(payload, offset)
    correction_records: dict[int, tuple[int, int, np.ndarray]] = {}
    previous_occurrence = 0
    for index in range(count):
        occurrence_delta, offset = _decode_varint(payload, offset)
        occurrence = previous_occurrence + occurrence_delta if index else occurrence_delta
        _page, offset = _decode_varint(payload, offset)
        y, offset = _decode_varint(payload, offset)
        x, offset = _decode_varint(payload, offset)
        height, offset = _decode_varint(payload, offset)
        width, offset = _decode_varint(payload, offset)
        byte_count, offset = _decode_varint(payload, offset)
        packed = np.frombuffer(payload[offset : offset + byte_count], dtype=np.uint8)
        offset += byte_count
        bitmap = np.unpackbits(packed, bitorder="big").reshape(height, -1)[:, :width]
        correction_records[occurrence] = (y, x, bitmap.astype(bool))
        previous_occurrence = occurrence
    if offset != len(payload):
        raise ValueError("trailing bytes in glyph-core corrections")

    pages = [np.zeros(shape, dtype=bool) for shape in shapes]
    for occurrence_index, item in enumerate(occurrences):
        glyph = glyphs[item.glyph]
        correction = correction_records.get(occurrence_index)
        if correction is None:
            pages[item.page][
                item.y : item.y + glyph.shape[0], item.x : item.x + glyph.shape[1]
            ] |= glyph
            continue
        correction_y, correction_x, bitmap = correction
        y0 = min(item.y, correction_y)
        x0 = min(item.x, correction_x)
        y1 = max(item.y + glyph.shape[0], correction_y + bitmap.shape[0])
        x1 = max(item.x + glyph.shape[1], correction_x + bitmap.shape[1])
        refined = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        refined[
            item.y - y0 : item.y - y0 + glyph.shape[0],
            item.x - x0 : item.x - x0 + glyph.shape[1],
        ] |= glyph
        refined[
            correction_y - y0 : correction_y - y0 + bitmap.shape[0],
            correction_x - x0 : correction_x - x0 + bitmap.shape[1],
        ] ^= bitmap
        pages[item.page][y0:y1, x0:x1] |= refined
    return tuple(pages)


def subset_core_atlas(
    result: GlobalCoreAtlasResult, glyph_indices: Iterable[int]
) -> GlobalCoreAtlasResult:
    """Return an exact atlas containing only selected glyph classes."""

    selected = tuple(sorted(set(int(index) for index in glyph_indices)))
    if any(index < 0 or index >= len(result.glyphs) for index in selected):
        raise IndexError("glyph index is out of range")
    glyph_map = {old: new for new, old in enumerate(selected)}
    occurrence_map: dict[int, int] = {}
    occurrences = []
    for old_index, item in enumerate(result.occurrences):
        if item.glyph not in glyph_map:
            continue
        occurrence_map[old_index] = len(occurrences)
        occurrences.append(
            AtlasOccurrence(item.page, glyph_map[item.glyph], item.y, item.x)
        )
    corrections = tuple(
        AtlasCorrection(
            occurrence_map[item.occurrence],
            item.page,
            item.y,
            item.x,
            item.bitmap,
        )
        for item in result.corrections
        if item.occurrence in occurrence_map
    )
    glyphs = tuple(result.glyphs[index] for index in selected)
    core_pages = [np.zeros_like(source, dtype=bool) for source in result.source_masks]
    for item in occurrences:
        glyph = glyphs[item.glyph]
        core_pages[item.page][
            item.y : item.y + glyph.shape[0], item.x : item.x + glyph.shape[1]
        ] |= glyph
    placeholder = GlobalCoreAtlasResult(
        source_masks=result.source_masks,
        core_masks=tuple(core_pages),
        residual_masks=tuple(
            source ^ core for source, core in zip(result.source_masks, core_pages)
        ),
        remainder_masks=tuple(np.zeros_like(source) for source in result.source_masks),
        glyphs=glyphs,
        occurrences=tuple(occurrences),
        corrections=corrections,
        cluster_sizes=tuple(result.cluster_sizes[index] for index in selected),
        component_counts=result.component_counts,
        meyer_used=result.meyer_used,
        core_mode=result.core_mode,
    )
    refined = decode_refined_core_atlas(
        serialize_core_atlas(placeholder), serialize_core_corrections(placeholder)
    )
    remainders = tuple(
        source & ~selected_ink
        for source, selected_ink in zip(result.source_masks, refined)
    )
    if any(
        not np.array_equal(source, selected_ink | remainder)
        for source, selected_ink, remainder in zip(
            result.source_masks, refined, remainders
        )
    ):
        raise RuntimeError("selected glyph atlas is not a source subset")
    return GlobalCoreAtlasResult(
        source_masks=placeholder.source_masks,
        core_masks=placeholder.core_masks,
        residual_masks=placeholder.residual_masks,
        remainder_masks=remainders,
        glyphs=placeholder.glyphs,
        occurrences=placeholder.occurrences,
        corrections=placeholder.corrections,
        cluster_sizes=placeholder.cluster_sizes,
        component_counts=placeholder.component_counts,
        meyer_used=placeholder.meyer_used,
        core_mode=placeholder.core_mode,
    )


def _fixed_refinement_geometry(
    result: GlobalCoreAtlasResult,
) -> tuple[list[tuple[int, int, int, int]], dict[int, AtlasCorrection]]:
    corrections = {item.occurrence: item for item in result.corrections}
    geometry = []
    for glyph_index, glyph in enumerate(result.glyphs):
        min_y = 0
        min_x = 0
        max_y = glyph.shape[0]
        max_x = glyph.shape[1]
        for occurrence_index, occurrence in enumerate(result.occurrences):
            if occurrence.glyph != glyph_index:
                continue
            correction = corrections.get(occurrence_index)
            if correction is None:
                continue
            rel_y = correction.y - occurrence.y
            rel_x = correction.x - occurrence.x
            min_y = min(min_y, rel_y)
            min_x = min(min_x, rel_x)
            max_y = max(max_y, rel_y + correction.bitmap.shape[0])
            max_x = max(max_x, rel_x + correction.bitmap.shape[1])
        geometry.append((min_y, min_x, max_y, max_x))
    return geometry, corrections


def serialize_fixed_refinement_atlas(
    result: GlobalCoreAtlasResult, *, level: int = 9
) -> bytes:
    """Serialize fixed-shape, class-conditioned XOR refinement bitmaps.

    Geometry is stored once per glyph class.  Every occurrence then carries a
    same-shape delta with no rectangle header, allowing Deflate to learn across
    aligned exception fields instead of separately compressed cropped sprites.
    """

    geometry, corrections = _fixed_refinement_geometry(result)
    payload = bytearray()
    payload += _encode_varint(len(result.source_masks))
    for mask in result.source_masks:
        payload += _encode_varint(mask.shape[0])
        payload += _encode_varint(mask.shape[1])
    payload += _encode_varint(len(result.glyphs))
    expanded_glyphs = []
    for glyph, (min_y, min_x, max_y, max_x) in zip(result.glyphs, geometry):
        height = max_y - min_y
        width = max_x - min_x
        expanded = np.zeros((height, width), dtype=bool)
        expanded[-min_y : -min_y + glyph.shape[0], -min_x : -min_x + glyph.shape[1]] = glyph
        expanded_glyphs.append(expanded)
        packed = np.packbits(expanded, axis=1, bitorder="big").tobytes()
        payload += _encode_varint(height)
        payload += _encode_varint(width)
        payload += _encode_varint(len(packed))
        payload += packed
    payload += _encode_varint(len(result.occurrences))
    for occurrence_index, item in enumerate(result.occurrences):
        min_y, min_x, _max_y, _max_x = geometry[item.glyph]
        expanded = expanded_glyphs[item.glyph]
        payload += _encode_varint(item.page)
        payload += _encode_varint(item.y + min_y)
        payload += _encode_varint(item.x + min_x)
        payload += _encode_varint(item.glyph)
        delta = np.zeros_like(expanded)
        correction = corrections.get(occurrence_index)
        if correction is not None:
            rel_y = correction.y - (item.y + min_y)
            rel_x = correction.x - (item.x + min_x)
            delta[
                rel_y : rel_y + correction.bitmap.shape[0],
                rel_x : rel_x + correction.bitmap.shape[1],
            ] = correction.bitmap
        payload += np.packbits(delta, axis=1, bitorder="big").tobytes()
    return b"GCF1" + zlib.compress(bytes(payload), level)


def decode_fixed_refinement_atlas(data: bytes) -> tuple[np.ndarray, ...]:
    """Decode :func:`serialize_fixed_refinement_atlas`."""

    if not data.startswith(b"GCF1"):
        raise ValueError("not a fixed-refinement glyph atlas")
    payload = zlib.decompress(data[4:])
    offset = 0
    page_count, offset = _decode_varint(payload, offset)
    shapes = []
    for _ in range(page_count):
        height, offset = _decode_varint(payload, offset)
        width, offset = _decode_varint(payload, offset)
        shapes.append((height, width))
    glyph_count, offset = _decode_varint(payload, offset)
    glyphs = []
    for _ in range(glyph_count):
        height, offset = _decode_varint(payload, offset)
        width, offset = _decode_varint(payload, offset)
        byte_count, offset = _decode_varint(payload, offset)
        packed = np.frombuffer(payload[offset : offset + byte_count], dtype=np.uint8)
        offset += byte_count
        glyphs.append(
            np.unpackbits(packed, bitorder="big").reshape(height, -1)[:, :width].astype(bool)
        )
    occurrence_count, offset = _decode_varint(payload, offset)
    pages = [np.zeros(shape, dtype=bool) for shape in shapes]
    for _ in range(occurrence_count):
        page, offset = _decode_varint(payload, offset)
        y, offset = _decode_varint(payload, offset)
        x, offset = _decode_varint(payload, offset)
        glyph_index, offset = _decode_varint(payload, offset)
        glyph = glyphs[glyph_index]
        byte_count = glyph.shape[0] * ((glyph.shape[1] + 7) // 8)
        packed = np.frombuffer(payload[offset : offset + byte_count], dtype=np.uint8)
        offset += byte_count
        delta = np.unpackbits(packed, bitorder="big").reshape(glyph.shape[0], -1)[:, : glyph.shape[1]].astype(bool)
        pages[page][y : y + glyph.shape[0], x : x + glyph.shape[1]] |= glyph ^ delta
    if offset != len(payload):
        raise ValueError("trailing bytes in fixed-refinement glyph atlas")
    return tuple(pages)


def serialize_split_refinement_atlas(
    result: GlobalCoreAtlasResult,
) -> bytes:
    """Serialize class geometry, positions, and XOR fields with separate models."""

    geometry, corrections = _fixed_refinement_geometry(result)
    header = bytearray()
    header += _encode_varint(len(result.source_masks))
    for mask in result.source_masks:
        header += _encode_varint(mask.shape[0])
        header += _encode_varint(mask.shape[1])
    header += _encode_varint(len(result.glyphs))
    expanded_glyphs = []
    for glyph, (min_y, min_x, max_y, max_x) in zip(result.glyphs, geometry):
        expanded = np.zeros((max_y - min_y, max_x - min_x), dtype=bool)
        expanded[-min_y : -min_y + glyph.shape[0], -min_x : -min_x + glyph.shape[1]] = glyph
        expanded_glyphs.append(expanded)
        packed = np.packbits(expanded, axis=1, bitorder="big").tobytes()
        header += _encode_varint(expanded.shape[0])
        header += _encode_varint(expanded.shape[1])
        header += _encode_varint(len(packed))
        header += packed

    positions = bytearray()
    deltas = bytearray()
    for glyph_index, expanded in enumerate(expanded_glyphs):
        members = sorted(
            (
                (occurrence_index, item)
                for occurrence_index, item in enumerate(result.occurrences)
                if item.glyph == glyph_index
            ),
            key=lambda pair: (pair[1].page, pair[1].y, pair[1].x),
        )
        page_groups: list[tuple[int, list[tuple[int, AtlasOccurrence]]]] = []
        for occurrence_index, item in members:
            if not page_groups or page_groups[-1][0] != item.page:
                page_groups.append((item.page, []))
            page_groups[-1][1].append((occurrence_index, item))
        positions += _encode_varint(len(page_groups))
        previous_page = 0
        min_y, min_x, _max_y, _max_x = geometry[glyph_index]
        for page_group_index, (page, values) in enumerate(page_groups):
            positions += _encode_varint(
                page - previous_page if page_group_index else page
            )
            positions += _encode_varint(len(values))
            previous_page = page
            previous_y = 0
            previous_x = 0
            for value_index, (occurrence_index, item) in enumerate(values):
                y = item.y + min_y
                x = item.x + min_x
                y_delta = y - previous_y if value_index else y
                positions += _encode_varint(y_delta)
                positions += _encode_varint(
                    x - previous_x if value_index and y_delta == 0 else x
                )
                previous_y = y
                previous_x = x
                delta = np.zeros_like(expanded)
                correction = corrections.get(occurrence_index)
                if correction is not None:
                    rel_y = correction.y - y
                    rel_x = correction.x - x
                    delta[
                        rel_y : rel_y + correction.bitmap.shape[0],
                        rel_x : rel_x + correction.bitmap.shape[1],
                    ] = correction.bitmap
                deltas += np.packbits(delta, axis=1, bitorder="big").tobytes()

    header_stream = zlib.compress(bytes(header), 9)
    position_stream = zlib.compress(bytes(positions), 9)
    delta_stream = bz2.compress(bytes(deltas), 9)
    return (
        b"GCS1"
        + len(header_stream).to_bytes(4, "big")
        + len(position_stream).to_bytes(4, "big")
        + len(delta_stream).to_bytes(4, "big")
        + header_stream
        + position_stream
        + delta_stream
    )


def decode_split_refinement_atlas(data: bytes) -> tuple[np.ndarray, ...]:
    """Decode :func:`serialize_split_refinement_atlas`."""

    if not data.startswith(b"GCS1") or len(data) < 16:
        raise ValueError("not a split-refinement glyph atlas")
    header_length = int.from_bytes(data[4:8], "big")
    position_length = int.from_bytes(data[8:12], "big")
    delta_length = int.from_bytes(data[12:16], "big")
    if 16 + header_length + position_length + delta_length != len(data):
        raise ValueError("invalid split-refinement stream lengths")
    header_start = 16
    position_start = header_start + header_length
    delta_start = position_start + position_length
    header = zlib.decompress(data[header_start:position_start])
    positions = zlib.decompress(data[position_start:delta_start])
    deltas = bz2.decompress(data[delta_start:])

    header_offset = 0
    page_count, header_offset = _decode_varint(header, header_offset)
    shapes = []
    for _ in range(page_count):
        height, header_offset = _decode_varint(header, header_offset)
        width, header_offset = _decode_varint(header, header_offset)
        shapes.append((height, width))
    glyph_count, header_offset = _decode_varint(header, header_offset)
    glyphs = []
    for _ in range(glyph_count):
        height, header_offset = _decode_varint(header, header_offset)
        width, header_offset = _decode_varint(header, header_offset)
        byte_count, header_offset = _decode_varint(header, header_offset)
        packed = np.frombuffer(
            header[header_offset : header_offset + byte_count], dtype=np.uint8
        )
        header_offset += byte_count
        glyphs.append(
            np.unpackbits(packed, bitorder="big").reshape(height, -1)[:, :width].astype(bool)
        )
    if header_offset != len(header):
        raise ValueError("trailing split-refinement header bytes")

    pages = [np.zeros(shape, dtype=bool) for shape in shapes]
    position_offset = 0
    delta_offset = 0
    for glyph in glyphs:
        page_group_count, position_offset = _decode_varint(positions, position_offset)
        previous_page = 0
        byte_count = glyph.shape[0] * ((glyph.shape[1] + 7) // 8)
        for page_group_index in range(page_group_count):
            page_delta, position_offset = _decode_varint(positions, position_offset)
            page = previous_page + page_delta if page_group_index else page_delta
            count, position_offset = _decode_varint(positions, position_offset)
            previous_page = page
            previous_y = 0
            previous_x = 0
            for value_index in range(count):
                y_delta, position_offset = _decode_varint(positions, position_offset)
                x_value, position_offset = _decode_varint(positions, position_offset)
                y = previous_y + y_delta if value_index else y_delta
                x = previous_x + x_value if value_index and y_delta == 0 else x_value
                packed = np.frombuffer(
                    deltas[delta_offset : delta_offset + byte_count], dtype=np.uint8
                )
                delta_offset += byte_count
                delta = np.unpackbits(packed, bitorder="big").reshape(glyph.shape[0], -1)[:, : glyph.shape[1]].astype(bool)
                pages[page][y : y + glyph.shape[0], x : x + glyph.shape[1]] |= glyph ^ delta
                previous_y = y
                previous_x = x
    if position_offset != len(positions) or delta_offset != len(deltas):
        raise ValueError("trailing split-refinement payload bytes")
    return tuple(pages)


def transcode_fixed_refinement_to_split(data: bytes) -> bytes:
    """Losslessly transcode a GCF1 occurrence stream into the split GCS1 form."""

    if not data.startswith(b"GCF1"):
        raise ValueError("not a fixed-refinement glyph atlas")
    raw = zlib.decompress(data[4:])
    offset = 0
    page_count, offset = _decode_varint(raw, offset)
    shapes = []
    for _ in range(page_count):
        height, offset = _decode_varint(raw, offset)
        width, offset = _decode_varint(raw, offset)
        shapes.append((height, width))
    glyph_count, offset = _decode_varint(raw, offset)
    glyphs = []
    for _ in range(glyph_count):
        height, offset = _decode_varint(raw, offset)
        width, offset = _decode_varint(raw, offset)
        byte_count, offset = _decode_varint(raw, offset)
        packed = raw[offset : offset + byte_count]
        offset += byte_count
        glyphs.append((height, width, packed))
    occurrence_count, offset = _decode_varint(raw, offset)
    records = []
    for _ in range(occurrence_count):
        page, offset = _decode_varint(raw, offset)
        y, offset = _decode_varint(raw, offset)
        x, offset = _decode_varint(raw, offset)
        glyph, offset = _decode_varint(raw, offset)
        height, width, _packed = glyphs[glyph]
        byte_count = height * ((width + 7) // 8)
        delta = raw[offset : offset + byte_count]
        offset += byte_count
        records.append((glyph, page, y, x, delta))
    if offset != len(raw):
        raise ValueError("trailing fixed-refinement bytes")

    header = bytearray()
    header += _encode_varint(len(shapes))
    for height, width in shapes:
        header += _encode_varint(height)
        header += _encode_varint(width)
    header += _encode_varint(len(glyphs))
    for height, width, packed in glyphs:
        header += _encode_varint(height)
        header += _encode_varint(width)
        header += _encode_varint(len(packed))
        header += packed
    positions = bytearray()
    deltas = bytearray()
    for glyph_index in range(len(glyphs)):
        members = sorted(
            (item for item in records if item[0] == glyph_index),
            key=lambda item: (item[1], item[2], item[3]),
        )
        page_groups: list[tuple[int, list[tuple[int, int, bytes]]]] = []
        for _glyph, page, y, x, delta in members:
            if not page_groups or page_groups[-1][0] != page:
                page_groups.append((page, []))
            page_groups[-1][1].append((y, x, delta))
        positions += _encode_varint(len(page_groups))
        previous_page = 0
        for page_group_index, (page, values) in enumerate(page_groups):
            positions += _encode_varint(
                page - previous_page if page_group_index else page
            )
            positions += _encode_varint(len(values))
            previous_page = page
            previous_y = 0
            previous_x = 0
            for value_index, (y, x, delta) in enumerate(values):
                y_delta = y - previous_y if value_index else y
                positions += _encode_varint(y_delta)
                positions += _encode_varint(
                    x - previous_x if value_index and y_delta == 0 else x
                )
                previous_y = y
                previous_x = x
                deltas += delta
    header_stream = zlib.compress(bytes(header), 9)
    position_stream = zlib.compress(bytes(positions), 9)
    delta_stream = bz2.compress(bytes(deltas), 9)
    return (
        b"GCS1"
        + len(header_stream).to_bytes(4, "big")
        + len(position_stream).to_bytes(4, "big")
        + len(delta_stream).to_bytes(4, "big")
        + header_stream
        + position_stream
        + delta_stream
    )


def compose_mrc(background: np.ndarray, foreground: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Composite RGB MRC planes at mask resolution."""

    foreground = np.asarray(foreground, dtype=np.uint8)
    binary = np.asarray(mask, dtype=bool)
    if foreground.shape[:2] != binary.shape or foreground.shape[2:] != (3,):
        raise ValueError("foreground and mask dimensions differ")
    if np.asarray(background).shape[:2] != binary.shape:
        background = np.asarray(
            Image.fromarray(np.asarray(background, dtype=np.uint8), "RGB").resize(
                (binary.shape[1], binary.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
    else:
        background = np.asarray(background, dtype=np.uint8)
    return np.where(binary[..., None], foreground, background)


def foreground_owned_composite(
    background: np.ndarray,
    foreground: np.ndarray,
    source_mask: np.ndarray,
    canonical_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move source composite pixels under a superset mask into foreground.

    Returns ``(source_composite, candidate_foreground, enlarged_background)``.
    The candidate recomposition is bit-exact before a foreground codec is
    applied.
    """

    source = np.asarray(source_mask, dtype=bool)
    canonical = np.asarray(canonical_mask, dtype=bool)
    if np.any(source & ~canonical):
        raise ValueError("canonical mask must contain the source mask")
    foreground = np.asarray(foreground, dtype=np.uint8)
    background_array = np.asarray(background, dtype=np.uint8)
    if background_array.shape[:2] != source.shape:
        enlarged = np.asarray(
            Image.fromarray(background_array, "RGB").resize(
                (source.shape[1], source.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
    else:
        enlarged = background_array
    composite = np.where(source[..., None], foreground, enlarged)
    # Preserve the source foreground in hidden pixels.  It is usually an
    # extremely low-entropy ink-colour field; replacing it with the paper
    # background would make the foreground codec redundantly store a full
    # scanned page.  Only newly exposed envelope pixels need paper-coloured
    # corrections.
    candidate = foreground.copy()
    candidate[canonical] = composite[canonical]
    verified = np.where(canonical[..., None], candidate, enlarged)
    if not np.array_equal(composite, verified):
        raise RuntimeError("foreground ownership transfer was not exact")
    return composite, candidate, enlarged


def denoise_foreground_texture_family(
    candidate: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    *,
    texture_gains: Iterable[float],
    meyer_splitter: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
    virtual_passes: int = 8,
) -> dict[float, np.ndarray]:
    """Shrink Meyer texture at several gains after one decomposition.

    ``texture_gain=1`` is an exact no-op.  Lower values are perceptual and are
    terminal-measured after coding.  The decomposition acts only on the mask's
    bounding rectangle, with a small guard to keep padding bounded.
    """

    gains = tuple(dict.fromkeys(float(value) for value in texture_gains))
    if not gains:
        raise ValueError("texture_gains must not be empty")
    if any(not 0.0 <= gain <= 1.0 for gain in gains):
        raise ValueError("texture gains must be in [0, 1]")
    result = {
        gain: np.asarray(candidate, dtype=np.uint8).copy()
        for gain in gains if gain == 1.0
    }
    active_gains = tuple(gain for gain in gains if gain != 1.0)
    if not active_gains:
        return result
    binary = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return {
            gain: np.asarray(candidate, dtype=np.uint8).copy()
            for gain in gains
        }
    guard = 8
    y0, y1 = max(0, int(ys.min()) - guard), min(binary.shape[0], int(ys.max()) + guard + 1)
    x0, x1 = max(0, int(xs.min()) - guard), min(binary.shape[1], int(xs.max()) + guard + 1)
    value = np.asarray(candidate, dtype=np.float64)
    base = np.asarray(background, dtype=np.float64)
    delta = value[y0:y1, x0:x1] - base[y0:y1, x0:x1]
    luma_weights = np.array([0.2126, 0.7152, 0.0722])
    luma = delta @ luma_weights
    local_mask = binary[y0:y1, x0:x1]
    luma[~local_mask] = 0.0
    scale = max(float(np.max(np.abs(luma))), 1.0)
    splitter = meyer_splitter or _default_meyer
    try:
        cartoon, texture = splitter(luma / scale, virtual_passes)
    except (ImportError, OSError, RuntimeError):
        cartoon = ndi.gaussian_filter(luma / scale, 0.8, mode="nearest")
        texture = luma / scale - cartoon
    active = np.abs(luma) > 1e-6
    for gain in active_gains:
        target_luma = scale * (cartoon + gain * texture)
        ratio = np.ones_like(luma)
        ratio[active] = target_luma[active] / luma[active]
        reconstructed = base[y0:y1, x0:x1] + delta * ratio[..., None]
        output = np.asarray(candidate, dtype=np.uint8).copy()
        output[y0:y1, x0:x1][local_mask] = np.clip(
            np.rint(reconstructed[local_mask]), 0, 255
        ).astype(np.uint8)
        result[gain] = output
    return {gain: result[gain] for gain in gains}


def denoise_foreground_texture(
    candidate: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    *,
    texture_gain: float,
    meyer_splitter: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
    virtual_passes: int = 8,
) -> np.ndarray:
    """Single-gain convenience wrapper around the shared Meyer split."""

    return denoise_foreground_texture_family(
        candidate,
        background,
        mask,
        texture_gains=(texture_gain,),
        meyer_splitter=meyer_splitter,
        virtual_passes=virtual_passes,
    )[float(texture_gain)]


def rmse(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None = None) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if mask is not None:
        difference = difference[np.asarray(mask, dtype=bool)]
    return float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0


def edge_rmse(left: np.ndarray, right: np.ndarray) -> float:
    weights = np.array([0.2126, 0.7152, 0.0722])
    a = np.asarray(left, dtype=np.float64) @ weights
    b = np.asarray(right, dtype=np.float64) @ weights
    edge_a = np.hypot(ndi.sobel(a, axis=0), ndi.sobel(a, axis=1))
    edge_b = np.hypot(ndi.sobel(b, axis=0), ndi.sobel(b, axis=1))
    return rmse(edge_a, edge_b)


def config_dict(config: GlyphResidualConfig) -> dict:
    return asdict(config)
