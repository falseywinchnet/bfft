"""Tk GUI for the transport-locked vectorizer."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import webbrowser

import numpy as np
from PIL import Image, ImageTk

from .core import Vectorization, VectorizerConfig, vectorize_array


def _tk():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as error:  # pragma: no cover - platform packaging
        raise RuntimeError(
            "The GUI needs Python's Tk support. The command-line converter is still available."
        ) from error
    return tk, ttk, filedialog, messagebox


class VectorizerApp:
    def __init__(self, root) -> None:
        tk, ttk, filedialog, messagebox = _tk()
        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox
        self.root = root
        self.root.title("Transport-Locked PNG → SVG")
        self.root.minsize(980, 680)
        self.source_path: Path | None = None
        self.source_rgba: np.ndarray | None = None
        self.result: Vectorization | None = None
        self.events: queue.Queue = queue.Queue()
        self.photos: dict[str, ImageTk.PhotoImage] = {}

        self.values = {
            "colors": tk.StringVar(value="12"),
            "detail_colors": tk.StringVar(value="6"),
            "coarse_side": tk.StringVar(value="160"),
            "minimum_region": tk.StringVar(value="10"),
            "simplify": tk.StringVar(value="0.85"),
            "curve_tolerance": tk.StringVar(value="0.65"),
            "subpixel_smoothing": tk.StringVar(value="4"),
            "seam_overlap": tk.StringVar(value="0.65"),
            "alpha_mode": tk.StringVar(value="auto"),
            "trim_transparent": tk.BooleanVar(value=True),
        }
        self.status = tk.StringVar(value="Open a PNG to begin.")
        self._build()
        self.root.after(80, self._poll)

    def _build(self) -> None:
        ttk = self.ttk
        shell = ttk.Frame(self.root, padding=14)
        shell.pack(fill="both", expand=True)
        toolbar = ttk.Frame(shell)
        toolbar.pack(fill="x", pady=(0, 10))
        ttk.Button(toolbar, text="Open PNG…", command=self.open_png).pack(side="left")
        self.convert_button = ttk.Button(toolbar, text="Convert", command=self.convert)
        self.convert_button.pack(side="left", padx=8)
        self.export_button = ttk.Button(toolbar, text="Export SVG…", command=self.export, state="disabled")
        self.export_button.pack(side="left")
        self.open_button = ttk.Button(toolbar, text="Open SVG", command=self.open_svg, state="disabled")
        self.open_button.pack(side="left", padx=8)

        body = ttk.Panedwindow(shell, orient="horizontal")
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, padding=(0, 0, 14, 0))
        previews = ttk.Frame(body)
        body.add(controls, weight=0)
        body.add(previews, weight=1)

        ttk.Label(controls, text="Vector basis", font=("TkDefaultFont", 14, "bold")).pack(anchor="w", pady=(0, 8))
        fields = [
            ("Structural colors", "colors"),
            ("Residual colors", "detail_colors"),
            ("Coarse side", "coarse_side"),
            ("Minimum island", "minimum_region"),
            ("Simplification", "simplify"),
            ("Curve tolerance", "curve_tolerance"),
            ("Subpixel passes", "subpixel_smoothing"),
            ("Seam overlap", "seam_overlap"),
        ]
        grid = ttk.Frame(controls)
        grid.pack(fill="x")
        for row, (label, key) in enumerate(fields):
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(grid, textvariable=self.values[key], width=10).grid(row=row, column=1, sticky="e", padx=(12, 0), pady=4)
        ttk.Checkbutton(
            controls, text="Trim transparent border",
            variable=self.values["trim_transparent"],
        ).pack(anchor="w", pady=(12, 4))
        ttk.Label(controls, text="Transparency").pack(anchor="w", pady=(8, 2))
        ttk.Combobox(
            controls, textvariable=self.values["alpha_mode"],
            values=("auto", "cutout", "preserve"), state="readonly", width=12,
        ).pack(anchor="w")
        ttk.Label(
            controls,
            text="Seam overlap closes antialiasing pinholes.\nSubpixel passes smooth raster staircases\nwhile persistent corners remain pinned.",
            foreground="#555555", justify="left",
        ).pack(anchor="w", pady=(10, 0))

        preview_grid = ttk.Frame(previews)
        preview_grid.pack(fill="both", expand=True)
        preview_grid.columnconfigure((0, 1), weight=1)
        preview_grid.rowconfigure(1, weight=1)
        ttk.Label(preview_grid, text="Source", anchor="center").grid(row=0, column=0, sticky="ew")
        ttk.Label(preview_grid, text="Region preview", anchor="center").grid(row=0, column=1, sticky="ew")
        self.source_label = ttk.Label(preview_grid, anchor="center", relief="solid")
        self.source_label.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self.result_label = ttk.Label(preview_grid, anchor="center", relief="solid")
        self.result_label.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=5)
        ttk.Label(
            preview_grid,
            text="The preview shows the region model; exported SVG additionally applies smooth curves and seam closure.",
            foreground="#555555", anchor="center",
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(shell, textvariable=self.status, anchor="w").pack(fill="x", pady=(10, 0))

    def _config(self) -> VectorizerConfig:
        return VectorizerConfig(
            colors=int(self.values["colors"].get()),
            detail_colors=int(self.values["detail_colors"].get()),
            coarse_side=int(self.values["coarse_side"].get()),
            minimum_region=int(self.values["minimum_region"].get()),
            simplify=float(self.values["simplify"].get()),
            curve_tolerance=float(self.values["curve_tolerance"].get()),
            subpixel_smoothing=int(self.values["subpixel_smoothing"].get()),
            seam_overlap=float(self.values["seam_overlap"].get()),
            alpha_mode=self.values["alpha_mode"].get(),
            trim_transparent=bool(self.values["trim_transparent"].get()),
        )

    def _show(self, array: np.ndarray, target, key: str) -> None:
        image = Image.fromarray(array, "RGBA")
        image.thumbnail((470, 520), Image.Resampling.LANCZOS)
        # Checkerboard makes transparency distinct from accidental white gaps.
        checker = Image.new("RGBA", image.size, (235, 235, 235, 255))
        pixels = np.asarray(checker).copy()
        yy, xx = np.indices((image.height, image.width))
        dark = ((xx // 12 + yy // 12) & 1).astype(bool)
        pixels[dark, :3] = 211
        checker = Image.fromarray(pixels, "RGBA")
        checker.alpha_composite(image)
        photo = ImageTk.PhotoImage(checker)
        self.photos[key] = photo
        target.configure(image=photo)

    def open_png(self) -> None:
        filename = self.filedialog.askopenfilename(
            title="Open PNG", filetypes=[("PNG images", "*.png"), ("All files", "*")]
        )
        if not filename:
            return
        try:
            with Image.open(filename) as image:
                rgba = np.asarray(image.convert("RGBA"))
        except Exception as error:
            self.messagebox.showerror("Could not open image", str(error))
            return
        self.source_path = Path(filename)
        self.source_rgba = rgba
        self.result = None
        self.export_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self._show(rgba, self.source_label, "source")
        self.result_label.configure(image="")
        self.status.set(f"Loaded {self.source_path.name} — {rgba.shape[1]}×{rgba.shape[0]}")
        self.convert()

    def convert(self) -> None:
        if self.source_rgba is None:
            self.open_png()
            return
        try:
            config = self._config()
        except ValueError as error:
            self.messagebox.showerror("Invalid setting", str(error))
            return
        source = self.source_rgba.copy()
        title = self.source_path.name if self.source_path else "vectorized image"
        self.convert_button.configure(state="disabled")
        self.status.set("Converting…")

        def work() -> None:
            try:
                self.events.put(("done", vectorize_array(source, config, title=title)))
            except Exception as error:  # pragma: no cover - UI boundary
                self.events.put(("error", error))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self) -> None:
        try:
            kind, payload = self.events.get_nowait()
        except queue.Empty:
            self.root.after(80, self._poll)
            return
        self.convert_button.configure(state="normal")
        if kind == "error":
            self.status.set("Conversion failed.")
            self.messagebox.showerror("Conversion failed", str(payload))
        else:
            self.result = payload
            approximation = self.result.palette_rgba[self.result.labels]
            self._show(approximation, self.result_label, "result")
            self.export_button.configure(state="normal")
            self.status.set(
                f"Ready — {self.result.diagnostics['paths']} paths, "
                f"{self.result.diagnostics['loops']} loops, "
                f"{self.result.diagnostics['svg_bytes'] / 1024:.1f} KiB, "
                f"{self.result.diagnostics['total_ms'] / 1000:.2f} s"
            )
        self.root.after(80, self._poll)

    def export(self) -> None:
        if self.result is None:
            return
        initial = (self.source_path.stem + ".svg") if self.source_path else "output.svg"
        filename = self.filedialog.asksaveasfilename(
            title="Export SVG", defaultextension=".svg", initialfile=initial,
            filetypes=[("SVG images", "*.svg")],
        )
        if not filename:
            return
        self.result.save(filename)
        self.exported_path = Path(filename)
        self.open_button.configure(state="normal")
        self.status.set(f"Saved {self.exported_path}")

    def open_svg(self) -> None:
        path = getattr(self, "exported_path", None)
        if path:
            webbrowser.open(path.resolve().as_uri())


def main() -> int:
    tk, _ttk, _filedialog, _messagebox = _tk()
    root = tk.Tk()
    VectorizerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
