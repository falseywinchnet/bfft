"""Render real 3-D room scenes through scalar and retained diffuse transport."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np
from PIL import Image, ImageDraw


SCENES = ("standard", "aperture-canyon", "mirror-relay", "occlusion-garden")


def _run(
    executable: Path,
    scene: str,
    backend: str,
    width: int,
    height: int,
    output: Path,
) -> dict[str, object]:
    completed = subprocess.run(
        (
            str(executable),
            "--width",
            str(width),
            "--height",
            str(height),
            "--terminal-error",
            "1",
            "--scene",
            scene,
            "--transport-backend",
            backend,
            "--out",
            str(output),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median(records: list[dict[str, object]], key: str) -> float:
    return statistics.median(float(record[key]) for record in records)


def run(
    executable: Path,
    width: int,
    height: int,
    repeats: int,
    output: Path,
    montage_path: Path,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    render_root = output.parent / "native_retained_room_frames"
    render_root.mkdir(parents=True, exist_ok=True)
    results = []
    panels = []
    for scene in SCENES:
        backend_records: dict[str, list[dict[str, object]]] = {}
        backend_frames: dict[str, Path] = {}
        for backend in ("scalar", "retained"):
            records = []
            for repeat in range(repeats):
                frame = render_root / f"{scene}_{backend}_{repeat}.ppm"
                records.append(
                    _run(executable, scene, backend, width, height, frame)
                )
                backend_frames[backend] = frame
            backend_records[backend] = records
        scalar_image = np.asarray(Image.open(backend_frames["scalar"]), dtype=np.int16)
        retained_image = np.asarray(Image.open(backend_frames["retained"]), dtype=np.int16)
        difference = np.abs(scalar_image - retained_image)
        scalar = backend_records["scalar"]
        retained = backend_records["retained"]
        results.append(
            {
                "scene": scene,
                "width": width,
                "height": height,
                "repeats": repeats,
                "byte_identical": bool(np.array_equal(scalar_image, retained_image)),
                "maximum_channel_difference": int(np.max(difference, initial=0)),
                "changed_channel_count": int(np.count_nonzero(difference)),
                "scalar_sha256": _digest(backend_frames["scalar"]),
                "retained_sha256": _digest(backend_frames["retained"]),
                "scalar_median_total_ms": _median(scalar, "total_ms"),
                "retained_median_total_ms": _median(retained, "total_ms"),
                "scalar_median_field_compile_ms": _median(
                    scalar, "field_compile_ms"
                ),
                "retained_median_field_compile_ms": _median(
                    retained, "field_compile_ms"
                ),
                "scalar_median_transport_solve_ms": _median(
                    scalar, "transport_solve_ms"
                ),
                "retained_median_transport_solve_ms": _median(
                    retained, "transport_solve_ms"
                ),
                "retained_plan_ms": float(retained[-1]["retained_plan_ms"]),
                "diffuse_nodes": int(retained[-1]["diffuse_nodes"]),
                "sparse_couplings": int(retained[-1]["transport_couplings"]),
                "retained_blocks": int(retained[-1]["retained_transport_blocks"]),
                "retained_coefficients": int(
                    retained[-1]["retained_transport_coefficients"]
                ),
                "retained_block_applications": int(
                    retained[-1]["retained_block_applications"]
                ),
                "retained_scattered_coefficients": int(
                    retained[-1]["retained_scattered_coefficients"]
                ),
                "transport_iterations": int(retained[-1]["transport_iterations"]),
            }
        )
        image = Image.open(backend_frames["retained"]).convert("RGB")
        panel = Image.new("RGB", (width, height + 30), (17, 18, 22))
        panel.paste(image, (0, 30))
        ImageDraw.Draw(panel).text((9, 8), f"{scene} - retained", fill=(240, 242, 247))
        panels.append(panel)
        if scene == "standard":
            image.save(output.parent / "native_retained_room_standard_m4.png", optimize=True)
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    montage = Image.new("RGB", (columns * width, rows * (height + 30)), (17, 18, 22))
    for index, panel in enumerate(panels):
        montage.paste(panel, ((index % columns) * width, (index // columns) * (height + 30)))
    montage.save(montage_path, optimize=True)
    result = {
        "description": "real 3-D room renders with identical geometry, camera, materials, and terminal evaluator",
        "scalar_oracle": "edge-wise CSR diffuse residual march",
        "candidate": "source-column retained blocks through compiled arm64 NEON mode plan",
        "scenes": results,
        "montage": str(montage_path),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("experiments/photonic_transport_field/native/regime_scene_native"),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/native_retained_room_m4.json")
    )
    parser.add_argument(
        "--montage",
        type=Path,
        default=Path("/tmp/native_retained_room_m4.png"),
    )
    args = parser.parse_args()
    if args.width < 16 or args.height < 16 or args.repeats < 1:
        raise ValueError("invalid room dogfood dimensions or repeat count")
    result = run(
        args.executable,
        args.width,
        args.height,
        args.repeats,
        args.out,
        args.montage,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
