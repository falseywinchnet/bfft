"""Dear PyGui front end for the ownership-aware PNG optimizer."""

from __future__ import annotations

from pathlib import Path
import threading
import traceback

import numpy as np
from PIL import Image

from .core import PNGConfig, PNGOptimizationResult, optimize_png


def _preview(image: np.ndarray, maximum: tuple[int, int] = (760, 500)) -> tuple[int, int, list[float]]:
    rgba = Image.fromarray(np.asarray(image, dtype=np.uint8), "RGBA")
    rgba.thumbnail(maximum, Image.Resampling.LANCZOS)
    values = np.asarray(rgba, dtype=np.float32) / 255.0
    return rgba.width, rgba.height, values.ravel().tolist()


class PNGApp:
    def __init__(self, dpg):
        self.dpg = dpg
        self.source: Path | None = None
        self.source_rgba: np.ndarray | None = None
        self.result: PNGOptimizationResult | None = None
        self.texture_tags: list[str] = []
        self.busy = False

    def choose_source(self, _sender=None, app_data=None):
        if app_data and app_data.get("file_path_name"):
            self.open(Path(app_data["file_path_name"]))

    def open_text_path(self):
        value = self.dpg.get_value("source_path").strip()
        if value:
            self.open(Path(value).expanduser())

    def open(self, path: Path):
        try:
            with Image.open(path) as image:
                self.source_rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                mode = image.mode
            self.source = path
            self.result = None
            self.dpg.set_value("source_path", str(path))
            self.dpg.set_value("output_path", str(path.with_name(path.stem + "_optimized.png")))
            self.dpg.set_value(
                "status", f"Loaded {path.name}: {path.stat().st_size:,} B, {mode}, "
                f"{self.source_rgba.shape[1]} × {self.source_rgba.shape[0]}"
            )
            self.dpg.set_value("metrics", "No processed result yet")
            self.render()
        except Exception:
            self.dpg.set_value("status", traceback.format_exc())

    def config(self, lossless: bool = False) -> PNGConfig:
        dpg = self.dpg
        strength = float(dpg.get_value("ownership_strength"))
        if dpg.get_value("automatic_ownership"):
            strength = -1.0
        return PNGConfig(
            target_bytes=max(0, int(dpg.get_value("target_bytes"))),
            minimum_ssim=max(0.0, min(1.0, float(dpg.get_value("minimum_ssim")))),
            colors=0 if dpg.get_value("automatic_colors") else int(dpg.get_value("colors")),
            minimum_colors=int(dpg.get_value("minimum_colors")),
            dither=str(dpg.get_value("dither")),
            quantizer=str(dpg.get_value("quantizer")),
            lloyd_iterations=int(dpg.get_value("lloyd_iterations")),
            palette_edge_weight=float(dpg.get_value("palette_edge_weight")),
            diffusion_strength=float(dpg.get_value("diffusion_strength")),
            diffusion_edge_barrier=float(dpg.get_value("diffusion_edge_barrier")),
            ownership_strength=strength,
            ownership_iterations=int(dpg.get_value("ownership_iterations")),
            edge_protection=float(dpg.get_value("edge_protection")),
            palette_transport=bool(dpg.get_value("palette_transport")),
            filter_search="thorough" if dpg.get_value("thorough") else "fast",
            lossless=lossless,
        )

    def render(self):
        dpg = self.dpg
        dpg.delete_item("preview_tabs", children_only=True)
        for tag in self.texture_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self.texture_tags.clear()
        images: list[tuple[str, np.ndarray]] = []
        if self.source_rgba is not None:
            images.append(("Original", self.source_rgba))
        if self.result is not None:
            images.append(("Optimized", self.result.winner.rgba))
        with dpg.texture_registry(show=False):
            for index, (_, image) in enumerate(images):
                width, height, values = _preview(image)
                tag = f"png_preview_texture_{index}"
                dpg.add_static_texture(width, height, values, tag=tag)
                self.texture_tags.append(tag)
        for index, (name, image) in enumerate(images):
            with dpg.tab(label=name, parent="preview_tabs"):
                dpg.add_text(f"{image.shape[1]} × {image.shape[0]}")
                dpg.add_image(self.texture_tags[index])

    def run(self, lossless: bool = False):
        if self.source is None:
            self.dpg.set_value("status", "Choose a source PNG first")
            return
        if self.busy:
            self.dpg.set_value("status", "An optimization is already running")
            return
        self.busy = True
        config = self.config(lossless=lossless)
        self.dpg.set_value("progress", 0.0)
        self.dpg.set_value(
            "status",
            "Optimizing lossless scanlines…" if lossless else "Tracing the PNG rate–distortion frontier…",
        )

        def work():
            try:
                def progress(current: int, total: int, message: str):
                    self.dpg.set_value("progress", min(1.0, current / max(total, 1)))
                    self.dpg.set_value("status", f"{current}/{total}: {message}")

                result = optimize_png(self.source, config=config, progress=progress)
                self.result = result
                winner = result.winner
                self.dpg.set_value(
                    "metrics",
                    f"{result.source_bytes:,} → {winner.size:,} B  "
                    f"({100.0 * winner.size / result.source_bytes:.1f}%)\n"
                    f"SSIM {winner.ssim:.6f}   PSNR {winner.psnr_db:.2f} dB   "
                    f"edge {winner.edge_psnr_db:.2f} dB\n"
                    f"{winner.colors or 'truecolor'} colors · {winner.palette_order} order · "
                    f"{winner.filter_policy} · ownership {winner.ownership_strength:g}\n"
                    f"{winner.dither} diffusion {winner.diffusion_strength:g} · "
                    f"smooth-transition coverage {winner.smooth_transition_coverage:.3f}\n"
                    f"{len(result.candidates)} measured candidates in {result.elapsed_seconds:.2f} s",
                )
                self.dpg.set_value(
                    "status", f"Ready to save: {winner.size:,} B, SSIM {winner.ssim:.6f}"
                )
                self.render()
            except Exception:
                self.dpg.set_value("status", traceback.format_exc())
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def save(self):
        if self.result is None:
            self.dpg.set_value("status", "Run an optimization before saving")
            return
        try:
            text = self.dpg.get_value("output_path").strip()
            if not text:
                assert self.source is not None
                output = self.source.with_name(self.source.stem + "_optimized.png")
            else:
                output = Path(text).expanduser()
            if output.suffix.lower() != ".png":
                output = output.with_suffix(".png")
            self.result.save(output, output.with_suffix(".json"))
            self.dpg.set_value(
                "status", f"Saved {output}: {self.result.winner.size:,} B and {output.with_suffix('.json').name}"
            )
        except Exception:
            self.dpg.set_value("status", traceback.format_exc())


def run_gui(initial_source: str | Path | None = None) -> None:
    try:
        import dearpygui.dearpygui as dpg
    except ImportError as error:  # pragma: no cover
        raise SystemExit(
            "Dear PyGui is not installed. Install the png-lab extra: "
            "python -m pip install -e '.[png-lab]'"
        ) from error

    dpg.create_context()
    app = PNGApp(dpg)
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=app.choose_source,
        tag="png_source_dialog",
        width=720,
        height=480,
    ):
        dpg.add_file_extension(".png", color=(110, 200, 255, 255))
        dpg.add_file_extension(".*")

    with dpg.window(tag="png_primary", label="Ownership PNG Optimizer"):
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="source_path", width=-180, hint="Source PNG")
            dpg.add_button(label="Open path", callback=lambda: app.open_text_path())
            dpg.add_button(label="Browse…", callback=lambda: dpg.show_item("png_source_dialog"))
        with dpg.collapsing_header(label="Measured objective", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_input_int(tag="target_bytes", label="Target bytes (0 = quality frontier)", default_value=0, min_value=0, width=180)
                dpg.add_input_float(tag="minimum_ssim", label="Minimum SSIM", default_value=0.90, min_value=0.0, max_value=1.0, width=150, format="%.4f")
                dpg.add_checkbox(tag="automatic_colors", label="Automatic colors", default_value=False)
                dpg.add_input_int(tag="colors", label="Colors", default_value=256, min_value=2, max_value=256, width=110)
                dpg.add_input_int(tag="minimum_colors", label="Minimum colors", default_value=8, min_value=2, max_value=256, width=130)
            with dpg.group(horizontal=True):
                dpg.add_combo(("auto", "edge-lloyd", "lloyd-rgb", "median-cut", "maximum-coverage", "fast-octree"), tag="quantizer", label="Palette allocator", default_value="auto", width=180)
                dpg.add_input_int(tag="lloyd_iterations", label="Lloyd passes", default_value=10, min_value=0, max_value=30, width=115)
                dpg.add_input_float(tag="palette_edge_weight", label="Palette edge weight", default_value=1.5, min_value=0.0, width=145)
            with dpg.group(horizontal=True):
                dpg.add_combo(("none", "selective", "floyd", "auto"), tag="dither", label="Dither", default_value="selective", width=110)
                dpg.add_checkbox(tag="palette_transport", label="Transport palette ownership", default_value=True)
                dpg.add_checkbox(tag="thorough", label="Thorough terminal filters", default_value=False)
        with dpg.collapsing_header(label="Spatial ownership flow", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(tag="automatic_ownership", label="Automatic annealing ladder", default_value=True)
                dpg.add_input_float(tag="ownership_strength", label="Boundary pressure", default_value=0.0015, min_value=0.0, width=150, format="%.6f")
                dpg.add_input_int(tag="ownership_iterations", label="Flow passes", default_value=2, min_value=0, max_value=8, width=120)
                dpg.add_input_float(tag="edge_protection", label="Edge protection", default_value=8.0, min_value=0.0, width=140)
            with dpg.group(horizontal=True):
                dpg.add_input_float(tag="diffusion_strength", label="Diffusion strength", default_value=0.9, min_value=0.0, max_value=2.0, width=145)
                dpg.add_input_float(tag="diffusion_edge_barrier", label="Diffusion edge barrier", default_value=3.0, min_value=0.0, width=165)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Optimize PNG", callback=lambda: app.run(False))
            dpg.add_button(label="Lossless optimize", callback=lambda: app.run(True))
            dpg.add_input_text(tag="output_path", width=-190, hint="Output PNG")
            dpg.add_button(label="Save processed result", callback=lambda: app.save())
        dpg.add_progress_bar(tag="progress", default_value=0.0, width=-1)
        dpg.add_text("Choose the source PNG to begin", tag="status", wrap=1200)
        dpg.add_text("No processed result yet", tag="metrics", wrap=1200)
        dpg.add_separator()
        with dpg.tab_bar(tag="preview_tabs"):
            pass

    dpg.create_viewport(title="Ownership PNG Optimizer", width=1280, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("png_primary", True)
    if initial_source is not None:
        app.open(Path(initial_source).expanduser())
    dpg.start_dearpygui()
    dpg.destroy_context()
