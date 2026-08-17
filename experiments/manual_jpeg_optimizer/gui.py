"""Dear PyGui front end for the five-stage JPEG laboratory."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json
import threading
import traceback

import numpy as np

from .core import (
    JPEGConfig,
    analyze_five_stages,
    decode,
    encode,
    image_metrics,
    infer_source_quality,
    load_rgb,
    optimize_jpeg,
    save_report,
)


def _rgba_texture(image: np.ndarray) -> tuple[int, int, list[float]]:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
    rgba = np.concatenate((rgb, alpha), axis=-1)
    return rgb.shape[1], rgb.shape[0], rgba.ravel().tolist()


class JPEGApp:
    def __init__(self, dpg):
        self.dpg = dpg
        self.source: Path | None = None
        self.rgb: np.ndarray | None = None
        self.processed_rgb: np.ndarray | None = None
        self.processed_label = "source preprocessing"
        self.processed_quality = 80
        self.processed_subsampling = 2
        self.texture_tags: list[str] = []
        self.optimize_cancel = threading.Event()

    def cancel_search(self, announce: bool = False):
        self.optimize_cancel.set()
        if announce:
            self.dpg.set_value("status", "Stopping the active rate–distortion trace…")

    def config(self) -> JPEGConfig:
        dpg = self.dpg
        return JPEGConfig(
            quality=dpg.get_value("quality"),
            subsampling=int(dpg.get_value("subsampling")),
            cartoon_sigma=dpg.get_value("cartoon_sigma"),
            chroma_projection=dpg.get_value("chroma_projection"),
            luma_texture_shrink=dpg.get_value("luma_texture_shrink"),
            phase_degrees=dpg.get_value("phase_degrees"),
            region_threshold=dpg.get_value("region_threshold"),
        )

    def choose_source(self, _sender=None, app_data=None):
        if app_data and app_data.get("file_path_name"):
            self.open(Path(app_data["file_path_name"]))

    def open(self, path: Path):
        try:
            self.cancel_search()
            self.source = path
            self.rgb = load_rgb(path)
            inferred_quality = infer_source_quality(path)
            self.dpg.set_value("quality", inferred_quality)
            self.processed_quality = inferred_quality
            self.dpg.set_value("source_path", str(path))
            self.dpg.set_value("status", f"Loaded {path.name}: {path.stat().st_size:,} bytes")
            self.refresh()
        except Exception:
            self.dpg.set_value("status", traceback.format_exc())

    def open_text_path(self):
        value = self.dpg.get_value("source_path").strip()
        if value:
            self.open(Path(value).expanduser())

    def refresh(self):
        if self.rgb is None:
            return
        try:
            stages = analyze_five_stages(self.rgb, self.config())
            from .core import preprocess
            self.processed_rgb = preprocess(self.rgb, self.config()).rgb
            self.processed_label = "manual controls"
            self.processed_quality = self.config().quality
            self.processed_subsampling = self.config().subsampling
            self.render_stages(stages)
            self.dpg.set_value("status", f"Rendered {len(stages)} views; controls are live after Update stages")
        except Exception:
            self.dpg.set_value("status", traceback.format_exc())

    def render_stages(self, stages):
        dpg = self.dpg
        for tag in self.texture_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self.texture_tags.clear()
        dpg.delete_item("stage_tabs", children_only=True)
        with dpg.texture_registry(show=False):
            for index, (name, image) in enumerate(stages.items()):
                width, height, data = _rgba_texture(image)
                tag = f"stage_texture_{index}"
                dpg.add_static_texture(width, height, data, tag=tag)
                self.texture_tags.append(tag)
        for index, (name, image) in enumerate(stages.items()):
            height, width = image.shape[:2]
            with dpg.tab(label=name.replace("_", " ").title(), parent="stage_tabs"):
                dpg.add_text(f"{width} x {height}")
                dpg.add_image(f"stage_texture_{index}", width=min(width, 900), height=min(height, 600))

    def globally_relax(self):
        if self.rgb is None:
            self.dpg.set_value("status", "Choose a source JPEG first")
            return
        self.cancel_search()
        try:
            from .certified_relaxation import (
                RelaxationConfig, _coefficients, coefficients_to_rgb,
                solve_coefficients,
            )
            from .core import _region_labels, rgb_to_ycc
            from .spectral_relaxation import frame_views, solve_spectral_relaxation

            self.dpg.set_value("status", "Solving global three-channel spectral relaxation…")
            ycc = rgb_to_ycc(self.rgb)
            labels, _ = _region_labels(ycc, 1.2, self.config().region_threshold)
            source = _coefficients(ycc)
            spectral = solve_spectral_relaxation(
                source,
                labels,
                cross_region_weight=self.dpg.get_value("global_cross_weight"),
            )
            inner_config = RelaxationConfig(
                rate_lambda=self.dpg.get_value("global_rate_lambda"),
                connection_lambda=self.dpg.get_value("global_connection_lambda"),
                cross_region_weight=self.dpg.get_value("global_cross_weight"),
                iterations=500,
                relative_gap_tolerance=1e-5,
            )
            coefficients, certificate = solve_coefficients(
                source, labels, inner_config, fixed_frames=spectral.frames
            )
            relaxed_rgb = coefficients_to_rgb(
                coefficients, labels.shape, self.rgb.shape[:2]
            )
            stages = analyze_five_stages(
                relaxed_rgb,
                replace(
                    self.config(), chroma_projection=0.0,
                    luma_texture_shrink=0.0, phase_degrees=0.0,
                ),
            )
            self.processed_rgb = relaxed_rgb
            self.processed_label = "spectral diagnostic"
            self.processed_quality = self.config().quality
            self.processed_subsampling = self.config().subsampling
            stages.update(frame_views(spectral.frames, labels.shape, self.rgb.shape[:2]))
            self.render_stages(stages)
            self.dpg.set_value(
                "status",
                f"Global optimum {spectral.relaxed_optimum:.12g}; eigen residual "
                f"{spectral.eigen_residual:.2e}; inner gap {certificate['relative_gap']:.2e}",
            )
        except Exception:
            self.dpg.set_value("status", traceback.format_exc())

    def ownership_relax(self):
        """Run the conserved-composition ownership/bifurcation experiment."""
        if self.rgb is None:
            self.dpg.set_value("status", "Choose a source JPEG first")
            return
        self.cancel_search()
        try:
            from PIL import Image
            from .certified_relaxation import _coefficients, coefficients_to_rgb
            from .core import _region_labels, infer_source_quality, rgb_to_ycc
            from .ownership_bifurcation import (
                BifurcationConfig, bifurcate_coefficients,
            )

            self.dpg.set_value("status", "Transporting owned constituents and solving the bifurcation tree…")
            ycc = rgb_to_ycc(self.rgb)
            labels, _ = _region_labels(ycc, 1.2, self.config().region_threshold)
            source = _coefficients(ycc)
            quality = (
                infer_source_quality(self.source)
                if self.source is not None else self.config().quality
            )
            result = bifurcate_coefficients(
                source,
                labels,
                quality,
                BifurcationConfig(
                    rate_lambda=self.dpg.get_value("ownership_rate_lambda"),
                    branch_penalty=self.dpg.get_value("ownership_branch_penalty"),
                    maximum_depth=self.dpg.get_value("ownership_maximum_depth"),
                    minimum_atoms=self.dpg.get_value("ownership_minimum_atoms"),
                    maximum_condition=self.dpg.get_value("ownership_maximum_condition"),
                    cross_region_weight=self.dpg.get_value("ownership_cross_weight"),
                ),
            )
            relaxed_rgb = coefficients_to_rgb(
                result.coefficients, labels.shape, self.rgb.shape[:2]
            )
            stages = analyze_five_stages(
                relaxed_rgb,
                replace(
                    self.config(), chroma_projection=0.0,
                    luma_texture_shrink=0.0, phase_degrees=0.0,
                ),
            )
            self.processed_rgb = relaxed_rgb
            self.processed_label = "ownership bifurcation"
            self.processed_quality = quality
            self.processed_subsampling = self.config().subsampling

            # A block owns 63 AC atoms.  Its modal leaf gives a compact view
            # of the causal partition; the change view shows where a routed
            # quantization path departed from the identity path.
            owners = result.leaf_of_atom.reshape(labels.size, 63)
            dominant = np.empty(labels.size, dtype=np.int32)
            for block, row in enumerate(owners):
                dominant[block] = np.bincount(row).argmax()
            colors = np.column_stack((
                (dominant * 73 + 31) % 256,
                (dominant * 151 + 67) % 256,
                (dominant * 199 + 101) % 256,
            )).astype(np.uint8).reshape(*labels.shape, 3)
            changed = np.any(np.abs(result.coefficients - source) > 1e-9, axis=(1, 2))
            change_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
            change_rgb.reshape(-1, 3)[changed] = (255, 180, 32)
            size = (self.rgb.shape[1], self.rgb.shape[0])
            stages["ownership_bifurcation"] = np.asarray(
                Image.fromarray(colors).resize(size, Image.Resampling.NEAREST)
            )
            stages["routed_departures"] = np.asarray(
                Image.fromarray(change_rgb).resize(size, Image.Resampling.NEAREST)
            )
            self.render_stages(stages)
            self.dpg.set_value(
                "status",
                f"Owned leaves {result.leaf_count}; changed coefficients "
                f"{result.changed_quantized_coefficients:,}; composition error "
                f"{result.prequantization_max_composition_error:.2e}; "
                f"Bellman value {result.objective:.6g}",
            )
        except Exception:
            self.dpg.set_value("status", traceback.format_exc())

    def spatial_dct_relax(self):
        """Transport signed ownership on the actual block/frequency graph."""
        if self.rgb is None:
            self.dpg.set_value("status", "Choose a source JPEG first")
            return
        self.cancel_search()
        try:
            from PIL import Image
            from .certified_relaxation import _coefficients, coefficients_to_rgb
            from .core import _region_labels, infer_source_quality, preprocess, rgb_to_ycc
            from .spatial_dct_transport import (
                SpatialDCTTransportConfig, transport_spatial_dct,
            )
            from .balanced_regions import (
                BalancedRegionConfig, balanced_bifurcation_regions,
            )

            self.dpg.set_value("status", "Solving simultaneous spatial/frequency DCT ownership flow…")
            quality = (
                infer_source_quality(self.source)
                if self.source is not None else self.config().quality
            )
            prepared = preprocess(self.rgb, self.config())
            ycc = rgb_to_ycc(prepared.rgb)
            if self.dpg.get_value("dct_region_mode") == "balanced":
                balanced = balanced_bifurcation_regions(
                    ycc,
                    quality,
                    BalancedRegionConfig(
                        target_regions=self.dpg.get_value("dct_region_count"),
                        minimum_blocks=self.dpg.get_value("dct_region_minimum"),
                    ),
                )
                labels = balanced.labels
            else:
                labels, _ = _region_labels(ycc, 1.2, self.config().region_threshold)
            source = _coefficients(ycc)
            result = transport_spatial_dct(
                source,
                labels,
                quality,
                SpatialDCTTransportConfig(
                    transport_lambda=self.dpg.get_value("dct_transport_lambda"),
                    frequency_weight=self.dpg.get_value("dct_frequency_weight"),
                    cross_region_weight=self.dpg.get_value("dct_cross_weight"),
                    luma_mobility=self.dpg.get_value("dct_y_mobility"),
                    cb_mobility=self.dpg.get_value("dct_cb_mobility"),
                    cr_mobility=self.dpg.get_value("dct_cr_mobility"),
                ),
            )
            transported_rgb = coefficients_to_rgb(
                result.coefficients, labels.shape, self.rgb.shape[:2]
            )
            stages = analyze_five_stages(
                transported_rgb,
                replace(
                    self.config(), chroma_projection=0.0,
                    luma_texture_shrink=0.0, phase_degrees=0.0,
                ),
            )
            self.processed_rgb = transported_rgb
            self.processed_label = "spatial/frequency DCT ownership"
            self.processed_quality = quality
            self.processed_subsampling = self.config().subsampling
            block_count = labels.size
            spatial = np.zeros(block_count, dtype=np.float64)
            frequency = np.zeros(block_count, dtype=np.float64)
            magnitude = np.sum(
                np.abs(result.positive_flow) + np.abs(result.negative_flow), axis=1
            )
            edge_block = result.edge_left // 63
            np.add.at(spatial, edge_block[result.edge_kind == 0], magnitude[result.edge_kind == 0])
            np.add.at(frequency, edge_block[result.edge_kind == 1], magnitude[result.edge_kind == 1])

            def flow_view(value, color):
                scale = np.quantile(value, 0.99) + 1e-15
                level = np.clip(value / scale, 0.0, 1.0).reshape(labels.shape)
                image = np.uint8(level[..., None] * np.asarray(color)[None, None, :])
                return np.asarray(Image.fromarray(image).resize(
                    (self.rgb.shape[1], self.rgb.shape[0]), Image.Resampling.NEAREST
                ))

            stages["spatial_ownership_flow"] = flow_view(spatial, (255, 170, 30))
            stages["frequency_escape_flow"] = flow_view(frequency, (70, 180, 255))
            self.render_stages(stages)
            self.dpg.set_value(
                "status",
                f"Global DCT transport {result.objective:.6g}; mass errors "
                f"+{result.positive_mass_error:.2e}/-{result.negative_mass_error:.2e}; "
                f"flow residual {result.flow_divergence_residual:.2e}",
            )
        except Exception:
            self.dpg.set_value("status", traceback.format_exc())

    def save_processed(self):
        if self.processed_rgb is None or self.source is None or self.rgb is None:
            self.dpg.set_value("status", "Choose and process a source JPEG first")
            return
        try:
            output_text = self.dpg.get_value("output_path").strip()
            output = (
                Path(output_text).expanduser()
                if output_text else self.source.with_name(self.source.stem + "_processed.jpg")
            )
            data = encode(
                self.processed_rgb,
                JPEGConfig(
                    quality=int(self.processed_quality),
                    subsampling=int(self.processed_subsampling),
                ),
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            ssim, psnr, edge_psnr = image_metrics(self.rgb, decode(data))
            report = {
                "source": str(self.source),
                "output": str(output),
                "pipeline": self.processed_label,
                "output_bytes": len(data),
                "quality": self.processed_quality,
                "subsampling": self.processed_subsampling,
                "ssim": ssim,
                "psnr_db": psnr,
                "edge_psnr_db": edge_psnr,
            }
            output.with_suffix(".json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            self.dpg.set_value(
                "status",
                f"Saved displayed {self.processed_label} to {output}: "
                f"{len(data):,} B, SSIM {ssim:.6f}",
            )
        except Exception:
            self.dpg.set_value("status", traceback.format_exc())

    def optimize(self):
        if self.source is None:
            self.dpg.set_value("status", "Choose a source JPEG first")
            return
        output_text = self.dpg.get_value("output_path").strip()
        output = Path(output_text).expanduser() if output_text else self.source.with_name(self.source.stem + "_optimized.jpg")
        target = int(self.dpg.get_value("target_bytes"))
        exhaustive = bool(self.dpg.get_value("exhaustive"))
        self.cancel_search()
        cancel = threading.Event()
        self.optimize_cancel = cancel
        self.dpg.set_value("status", "Tracing the measured rate–distortion frontier…")

        def work():
            try:
                def progress(index, total, candidate):
                    if index % max(1, total // 100) == 0 or index == total:
                        metric = (
                            "rate probe" if np.isnan(candidate.ssim)
                            else f"SSIM {candidate.ssim:.6f}"
                        )
                        self.dpg.set_value(
                            "status",
                            f"Frontier trace {index} (at most {total}): "
                            f"{candidate.size_bytes:,} B, {metric}",
                        )
                result = optimize_jpeg(
                    self.source, output, target_bytes=target,
                    exhaustive=exhaustive, progress=progress,
                    cancelled=cancel.is_set,
                )
                save_report(result, output.with_suffix(".json"))
                best = result.best
                self.dpg.set_value(
                    "status",
                    f"Saved {output}: {best.size_bytes:,} B, SSIM {best.ssim:.6f}, "
                    f"PSNR {best.psnr_db:.2f} dB",
                )
            except InterruptedError:
                pass
            except Exception:
                self.dpg.set_value("status", traceback.format_exc())

        threading.Thread(target=work, daemon=True).start()

    def jpegli_fuse(self):
        if self.source is None:
            self.dpg.set_value("status", "Choose a source JPEG first")
            return
        self.cancel_search()
        output_text = self.dpg.get_value("output_path").strip()
        output = (
            Path(output_text).expanduser()
            if output_text
            else self.source.with_name(self.source.stem + "_jpegli_fused.jpg")
        )
        self.dpg.set_value("status", "Jpegli ownership trellis running…")

        def work():
            try:
                from .jpegli_fusion import encode_jpegli_fusion

                report = encode_jpegli_fusion(
                    self.source,
                    output,
                    quality=int(self.dpg.get_value("jpegli_quality")),
                    target_bytes=(
                        int(self.dpg.get_value("target_bytes"))
                        if self.dpg.get_value("jpegli_use_target") else 0
                    ),
                    regions=int(self.dpg.get_value("dct_region_count")),
                    minimum_region_blocks=int(self.dpg.get_value("dct_region_minimum")),
                    field_strength=float(self.dpg.get_value("jpegli_field_strength")),
                    edge_protection=float(self.dpg.get_value("jpegli_edge_protection")),
                    transport_lambda=float(self.dpg.get_value("jpegli_transport_lambda")),
                    frequency_weight=float(self.dpg.get_value("dct_frequency_weight")),
                    cross_region_weight=float(self.dpg.get_value("dct_cross_weight")),
                    trellis_lambda=float(self.dpg.get_value("jpegli_trellis_lambda")),
                    ownership_weight=float(self.dpg.get_value("jpegli_ownership_weight")),
                    trellis_edge_weight=float(self.dpg.get_value("jpegli_trellis_edge_weight")),
                    trellis_luma_weight=float(self.dpg.get_value("jpegli_trellis_luma_weight")),
                    trellis_chroma_weight=float(self.dpg.get_value("jpegli_trellis_chroma_weight")),
                    quant_luma_tilt=float(self.dpg.get_value("jpegli_quant_luma_tilt")),
                    quant_chroma_tilt=float(self.dpg.get_value("jpegli_quant_chroma_tilt")),
                )
                self.dpg.set_value(
                    "status",
                    f"Saved fused Jpegli {output}: {report['output_bytes']:,} B, "
                    f"SSIM {report['ssim']:.6f}, edge {report['edge_psnr_db']:.2f} dB",
                )
            except Exception:
                self.dpg.set_value("status", traceback.format_exc())

        threading.Thread(target=work, daemon=True).start()


def run_gui() -> None:
    try:
        import dearpygui.dearpygui as dpg
    except ImportError as error:  # pragma: no cover - depends on desktop extras
        raise SystemExit(
            "Dear PyGui is not installed. Install the jpeg-lab extra: "
            "python -m pip install -e '.[jpeg-lab]'"
        ) from error

    dpg.create_context()
    app = JPEGApp(dpg)
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=app.choose_source,
        tag="source_dialog",
        width=720,
        height=480,
    ):
        dpg.add_file_extension(".jpg", color=(240, 190, 90, 255))
        dpg.add_file_extension(".jpeg", color=(240, 190, 90, 255))
        dpg.add_file_extension(".png", color=(110, 200, 255, 255))
        dpg.add_file_extension(".*")

    with dpg.window(tag="primary", label="Manual JPEG Optimizer"):
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="source_path", width=-180, hint="Source PNG or JPEG")
            dpg.add_button(label="Open path", callback=lambda: app.open_text_path())
            dpg.add_button(label="Browse…", callback=lambda: dpg.show_item("source_dialog"))
        with dpg.collapsing_header(label="Five-stage and alignment controls", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_slider_int(tag="quality", label="Quality", default_value=80, min_value=1, max_value=100, width=250)
                dpg.add_combo(("0", "1", "2"), tag="subsampling", label="Sampling (0=444, 1=422, 2=420)", default_value="2", width=100)
                dpg.add_slider_float(tag="cartoon_sigma", label="Cartoon sigma", default_value=1.2, min_value=0.1, max_value=4.0, width=220)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(tag="chroma_projection", label="Aligned chroma projection", default_value=0.0, min_value=0.0, max_value=1.0, width=250)
                dpg.add_slider_float(tag="luma_texture_shrink", label="Flat luma texture shrink", default_value=0.0, min_value=0.0, max_value=0.5, width=250)
                dpg.add_slider_float(tag="phase_degrees", label="Phase rotation", default_value=0.0, min_value=-45.0, max_value=45.0, width=220)
                dpg.add_slider_float(tag="region_threshold", label="Bloom threshold", default_value=0.58, min_value=0.05, max_value=0.95, width=220)
            dpg.add_button(label="Update stages", callback=lambda: app.refresh())
        with dpg.collapsing_header(label="Measured rate–distortion search", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="output_path", hint="Output JPEG (optional)", width=500)
                dpg.add_input_int(tag="target_bytes", label="Target bytes", default_value=29200, width=140)
                dpg.add_checkbox(tag="exhaustive", label="Exhaustive")
                dpg.add_button(label="Optimize", callback=lambda: app.optimize())
                dpg.add_button(label="Stop search", callback=lambda: app.cancel_search(True))
                dpg.add_button(label="Save displayed result", callback=lambda: app.save_processed())
        with dpg.collapsing_header(label="Ownership transport and optimal bifurcation", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_input_float(tag="ownership_rate_lambda", label="Transport pressure", default_value=0.0, min_value=0.0, width=160)
                dpg.add_input_float(tag="ownership_branch_penalty", label="Branch penalty", default_value=0.0, min_value=0.0, width=150)
                dpg.add_input_int(tag="ownership_maximum_depth", label="Maximum depth", default_value=8, min_value=0, max_value=16, width=130)
                dpg.add_input_int(tag="ownership_minimum_atoms", label="Minimum atoms", default_value=128, min_value=8, width=130)
            with dpg.group(horizontal=True):
                dpg.add_input_float(tag="ownership_maximum_condition", label="Bifurcation condition", default_value=12.0, min_value=1.0, width=180)
                dpg.add_input_float(tag="ownership_cross_weight", label="Cross-region transport", default_value=0.05, min_value=0.0, max_value=1.0, width=180)
                dpg.add_button(label="Solve ownership tree", callback=lambda: app.ownership_relax())
        with dpg.collapsing_header(label="Spatial/frequency DCT ownership", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_input_float(tag="dct_transport_lambda", label="Annealing pressure", default_value=0.009, min_value=0.0, width=160)
                dpg.add_input_float(tag="dct_frequency_weight", label="Frequency escape", default_value=0.1, min_value=0.0, width=160)
                dpg.add_input_float(tag="dct_cross_weight", label="Cross-region flow", default_value=0.05, min_value=0.0, max_value=1.0, width=160)
            with dpg.group(horizontal=True):
                dpg.add_input_float(tag="dct_y_mobility", label="Y mobility", default_value=0.15, min_value=0.0, width=130)
                dpg.add_input_float(tag="dct_cb_mobility", label="Cb mobility", default_value=1.75, min_value=0.0, width=130)
                dpg.add_input_float(tag="dct_cr_mobility", label="Cr mobility", default_value=1.75, min_value=0.0, width=130)
                dpg.add_button(label="Solve DCT ownership flow", callback=lambda: app.spatial_dct_relax())
            with dpg.group(horizontal=True):
                dpg.add_combo(("balanced", "signature"), tag="dct_region_mode", label="Region atlas", default_value="balanced", width=120)
                dpg.add_input_int(tag="dct_region_count", label="Regions", default_value=256, min_value=2, width=120)
                dpg.add_input_int(tag="dct_region_minimum", label="Minimum blocks", default_value=24, min_value=2, width=140)
        with dpg.collapsing_header(label="Jpegli ownership trellis", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_input_int(tag="jpegli_quality", label="Jpegli quality", default_value=72, min_value=1, max_value=100, width=130)
                dpg.add_checkbox(tag="jpegli_use_target", label="Use target bytes", default_value=False)
                dpg.add_input_float(tag="jpegli_trellis_lambda", label="Trellis rate", default_value=0.0695, min_value=0.0, width=130)
                dpg.add_input_float(tag="jpegli_ownership_weight", label="Ownership shadow", default_value=0.05, min_value=0.0, width=150)
                dpg.add_input_float(tag="jpegli_trellis_edge_weight", label="Edge frequency weight", default_value=1.0, min_value=0.0, width=160)
                dpg.add_input_float(tag="jpegli_trellis_luma_weight", label="Y allocation", default_value=1.0, min_value=0.0, width=120)
                dpg.add_input_float(tag="jpegli_trellis_chroma_weight", label="Cb/Cr allocation", default_value=1.0, min_value=0.0, width=130)
            with dpg.group(horizontal=True):
                dpg.add_input_float(tag="jpegli_field_strength", label="Dead-zone opportunity", default_value=0.25, min_value=0.0, width=170)
                dpg.add_input_float(tag="jpegli_edge_protection", label="Region edge protection", default_value=2.0, min_value=0.0, width=170)
                dpg.add_input_float(tag="jpegli_transport_lambda", label="Transport witness", default_value=0.002, min_value=0.0, width=150)
                dpg.add_input_float(tag="jpegli_quant_luma_tilt", label="Y quant tilt", default_value=-0.5, width=120)
                dpg.add_input_float(tag="jpegli_quant_chroma_tilt", label="Cb/Cr quant tilt", default_value=0.0, width=130)
                dpg.add_button(label="Encode + save fused JPEG", callback=lambda: app.jpegli_fuse())
        with dpg.collapsing_header(label="Spectral marginal diagnostic", default_open=False):
            with dpg.group(horizontal=True):
                dpg.add_input_float(tag="global_rate_lambda", label="Rate relaxation", default_value=0.5, min_value=0.0, width=160)
                dpg.add_input_float(tag="global_connection_lambda", label="Connection relaxation", default_value=0.5, min_value=0.0, width=180)
                dpg.add_input_float(tag="global_cross_weight", label="Cross-region connection", default_value=0.05, min_value=0.0, max_value=1.0, width=180)
                dpg.add_button(label="Run spectral diagnostic", callback=lambda: app.globally_relax())
        dpg.add_text("Choose a PNG or JPEG to begin", tag="status", wrap=1200)
        dpg.add_separator()
        with dpg.tab_bar(tag="stage_tabs"):
            pass

    dpg.create_viewport(title="Manual JPEG Optimizer", width=1280, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    dpg.start_dearpygui()
    dpg.destroy_context()
