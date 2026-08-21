"""Deterministic synthetic ground truth and deliberately inefficient captures."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass(frozen=True)
class Scene:
    name: str
    description: str
    tags: tuple[str, ...]
    render: Callable[[int, int], Image.Image]
    probes: tuple[dict[str, object], ...] = ()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _downsample(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _geometry(size: int, scale: int) -> Image.Image:
    n = size * scale
    image = Image.new("RGBA", (n, n), (241, 238, 228, 255))
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [(0, int(.74*n)), (int(.39*n), int(.08*n)), (int(.55*n), int(.78*n))],
        fill=(25, 45, 78, 255),
    )
    draw.ellipse(
        (int(.38*n), int(.18*n), int(.91*n), int(.71*n)),
        fill=(231, 91, 67, 255), outline=(72, 27, 30, 255), width=2*scale,
    )
    draw.rounded_rectangle(
        (int(.08*n), int(.61*n), int(.84*n), int(.91*n)),
        radius=int(.055*n), fill=(54, 171, 142, 255),
    )
    draw.line(
        [(int(.05*n), int(.16*n)), (int(.94*n), int(.88*n))],
        fill=(255, 210, 58, 255), width=scale,
    )
    return _downsample(image, size)


def _chroma_edges(size: int, scale: int) -> Image.Image:
    n = size * scale
    image = Image.new("RGBA", (n, n), (118, 118, 118, 255))
    draw = ImageDraw.Draw(image)
    # These pairs have deliberately similar Rec.709 luma but very different hue.
    colors = ((214, 72, 55, 255), (25, 132, 196, 255), (159, 101, 30, 255), (41, 125, 69, 255))
    width = n // len(colors)
    for index, color in enumerate(colors):
        x0 = index * width
        draw.rectangle((x0, 0, n if index == len(colors)-1 else x0+width, n), fill=color)
    draw.polygon(
        [(int(.08*n), int(.78*n)), (int(.47*n), int(.13*n)), (int(.91*n), int(.74*n))],
        fill=(187, 74, 169, 255),
    )
    for radius, color in ((.26, (51, 132, 103, 255)), (.17, (203, 74, 83, 255)), (.08, (36, 128, 177, 255))):
        r = int(radius*n)
        cx, cy = int(.51*n), int(.52*n)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=color)
    return _downsample(image, size)


def _gradients(size: int, scale: int) -> Image.Image:
    n = size * scale
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    xn, yn = x / max(n-1, 1), y / max(n-1, 1)
    radius = np.sqrt((xn-.72)**2 + (yn-.70)**2)
    fine = 2.2*np.sin(2*math.pi*(xn*3.1 + yn*.7))
    rgb = np.empty((n, n, 4), dtype=np.float64)
    rgb[..., 0] = 92 + 76*xn + 18*yn + fine
    rgb[..., 1] = 123 + 52*xn + 28*yn + fine
    rgb[..., 2] = 171 + 42*xn - 20*yn + 14*np.clip(.58-radius, 0, 1) + fine
    rgb[..., 3] = 255
    return _downsample(Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), "RGBA"), size)


def _thin_lines(size: int, scale: int) -> Image.Image:
    n = size * scale
    image = Image.new("RGBA", (n, n), (250, 247, 239, 255))
    draw = ImageDraw.Draw(image)
    for index, width in enumerate((1, 1, 2, 3, 5)):
        y = int((.10 + index*.075)*n)
        draw.line((int(.05*n), y, int(.95*n), y+int(.16*n)), fill=(18, 31, 45, 255), width=width*scale)
    for index, angle in enumerate((7, 19, 33, 47, 71)):
        cx = int((.13 + index*.18)*n)
        length = int(.32*n)
        dx = math.cos(math.radians(angle))*length/2
        dy = math.sin(math.radians(angle))*length/2
        draw.line((cx-dx, .72*n-dy, cx+dx, .72*n+dy), fill=(191, 47+index*18, 71, 255), width=scale)
    draw.text((int(.08*n), int(.43*n)), "Aa 8x  curve / edge", font=_font(31*scale), fill=(22, 56, 88, 255))
    draw.arc((int(.59*n), int(.50*n), int(.94*n), int(.88*n)), 195, 518, fill=(30, 143, 121, 255), width=2*scale)
    return _downsample(image, size)


def _texture_boundary(size: int, scale: int) -> Image.Image:
    n = size * scale
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    xn, yn = x/n, y/n
    rng = np.random.default_rng(508030340)
    grain = rng.normal(0, 1, (n, n))
    woven = 10*np.sin(2*math.pi*(xn*28 + yn*4)) + 7*np.sin(2*math.pi*(yn*39-xn*3)) + 3*grain
    boundary = .47 + .08*np.sin(2*math.pi*yn*1.7)
    textured = xn > boundary
    base = np.stack((154+20*yn, 132+28*yn, 103+30*yn), axis=-1)
    rgb = base + textured[..., None]*woven[..., None]*np.array((1.0, .82, .55))
    rgba = np.concatenate((np.clip(rgb, 0, 255), np.full((n, n, 1), 255.0)), axis=-1)
    return _downsample(Image.fromarray(np.rint(rgba).astype(np.uint8), "RGBA"), size)


def _phase_frequency(size: int, scale: int) -> Image.Image:
    n = size * scale
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    xn, yn = x/n, y/n
    bands = np.floor(xn*4).astype(int)
    frequencies = np.choose(bands, (5.0, 11.0, 23.0, 43.0))
    angles = np.choose(bands, (.0, .25, .55, .78))*math.pi
    phase = frequencies*(xn*np.cos(angles) + yn*np.sin(angles))*2*math.pi
    carrier = np.sin(phase)
    envelope = .45 + .55*np.sin(math.pi*yn)**2
    r = 126 + 54*carrier*envelope
    g = 132 + 45*np.sin(phase+2.1)*envelope
    b = 137 + 50*np.sin(phase-2.0)*envelope
    rgba = np.stack((r, g, b, np.full_like(r, 255)), axis=-1)
    return _downsample(Image.fromarray(np.clip(np.rint(rgba), 0, 255).astype(np.uint8), "RGBA"), size)


def _transparency(size: int, scale: int) -> Image.Image:
    n = size * scale
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    xn, yn = x/n, y/n
    radius = np.sqrt((xn-.5)**2+(yn-.5)**2)
    alpha = np.clip((.48-radius)*5.0, 0, 1)
    alpha *= .35 + .65*np.clip((xn-.05)/.9, 0, 1)
    rgb = np.stack((220-70*yn, 62+95*xn, 125+65*yn), axis=-1)
    rgba = np.concatenate((rgb, (255*alpha)[..., None]), axis=-1)
    image = Image.fromarray(np.clip(np.rint(rgba), 0, 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((int(.12*n), int(.16*n), int(.70*n), int(.42*n)), radius=int(.06*n), fill=(25, 116, 187, 112))
    draw.ellipse((int(.51*n), int(.46*n), int(.92*n), int(.87*n)), fill=(240, 148, 45, 183))
    return _downsample(image, size)


def _mixed(size: int, scale: int) -> Image.Image:
    n = size * scale
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    xn, yn = x/n, y/n
    rng = np.random.default_rng(612612)
    cloud = 13*np.sin(2*math.pi*(xn*.8+yn*.23)) + 5*np.sin(2*math.pi*(xn*4.2-yn*1.3))
    grain = rng.normal(0, 1.4, (n, n))
    rgb = np.stack((72+82*yn+cloud, 115+75*yn+cloud*.7, 157+55*yn+cloud*.35), axis=-1)
    rgb += grain[..., None]
    rgba = np.concatenate((np.clip(rgb, 0, 255), np.full((n, n, 1), 255)), axis=-1)
    image = Image.fromarray(np.rint(rgba).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(image)
    draw.polygon([(0, int(.77*n)), (int(.28*n), int(.46*n)), (int(.49*n), int(.75*n)), (int(.72*n), int(.37*n)), (n, int(.73*n)), (n, n), (0, n)], fill=(47, 76, 61, 255))
    draw.rectangle((int(.08*n), int(.68*n), int(.25*n), int(.91*n)), fill=(181, 117, 63, 255))
    draw.line((int(.02*n), int(.20*n), int(.96*n), int(.38*n)), fill=(247, 232, 173, 255), width=scale)
    draw.text((int(.55*n), int(.80*n)), "MIX", font=_font(34*scale), fill=(244, 236, 211, 255))
    return _downsample(image, size)


def _scenes() -> tuple[Scene, ...]:
    return (
        Scene("geometry", "Curves, oblique boundaries, corners, and one-pixel diagonals.", ("edges", "curves", "flat"), _geometry),
        Scene("chroma_edges", "Nearly equal-luma hue boundaries that expose chroma bleeding.", ("edges", "chroma"), _chroma_edges),
        Scene("gradients", "Low-slope multichannel and radial ramps that expose banding.", ("gradient", "banding"), _gradients, (
            {"kind": "gradient", "rect": [.08, .08, .92, .42], "axis": "x"},
            {"kind": "gradient", "rect": [.08, .48, .48, .92], "axis": "y"},
        )),
        Scene("thin_lines", "Text, arcs, and one-to-five-pixel lines at varied angles.", ("edges", "text", "thin-lines"), _thin_lines),
        Scene("texture_boundary", "Flat and stochastic/woven regions separated by a curved edge.", ("texture", "flat", "edges"), _texture_boundary, (
            {"kind": "flat", "rect": [.07, .12, .34, .88]},
            {"kind": "texture", "rect": [.68, .12, .94, .88]},
        )),
        Scene("phase_frequency", "Oriented RGB phase gratings across spatial frequencies.", ("texture", "frequency", "phase"), _phase_frequency, (
            {"kind": "texture", "rect": [.02, .18, .23, .82]},
            {"kind": "texture", "rect": [.77, .18, .98, .82]},
        )),
        Scene("transparency", "Soft and hard alpha boundaries for palette/transparency behavior.", ("alpha", "edges", "gradient"), _transparency),
        Scene("mixed", "A deterministic photo/graphic hybrid with sky, texture, text, and silhouettes.", ("mixed", "edges", "texture", "gradient"), _mixed),
    )


def _composite_rgb(image: Image.Image) -> Image.Image:
    background = Image.new("RGBA", image.size, (239, 241, 244, 255))
    return Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")


def _contact_sheet(images: list[tuple[str, Image.Image]], size: int) -> Image.Image:
    columns = 4
    label_height = 28
    thumb = size // 2
    rows = math.ceil(len(images)/columns)
    sheet = Image.new("RGB", (columns*thumb, rows*(thumb+label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(14)
    for index, (name, image) in enumerate(images):
        x = (index % columns)*thumb
        y = (index // columns)*(thumb+label_height)
        sheet.paste(_composite_rgb(image).resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y))
        draw.text((x+6, y+thumb+5), name, font=font, fill=(20, 20, 20))
    return sheet


def generate_suite(root: str | Path, *, size: int = 384, supersample: int = 4) -> dict[str, object]:
    """Generate references and same-pixel, deliberately inefficient upload inputs."""
    if size < 128:
        raise ValueError("size must be at least 128")
    if supersample < 1:
        raise ValueError("supersample must be positive")
    root = Path(root)
    references = root / "references"
    upload = root / "upload"
    controls = {
        "control-blur": root/"controls"/"blur",
        "control-banded": root/"controls"/"banded",
        "control-halo": root/"controls"/"halo",
    }
    for directory in (
        references, upload, root/"candidates"/"tinypng", root/"candidates"/"ours",
        *controls.values(),
    ):
        directory.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, object]] = []
    preview: list[tuple[str, Image.Image]] = []
    for scene in _scenes():
        image = scene.render(size, supersample).convert("RGBA")
        reference_path = references / f"{scene.name}.png"
        png_path = upload / f"png__{scene.name}.png"
        jpeg_path = upload / f"jpeg__{scene.name}.jpg"
        image.save(reference_path, format="PNG", optimize=True, compress_level=9)
        # PNG is pixel-identical but intentionally uses unfiltered stored DEFLATE.
        image.save(png_path, format="PNG", optimize=False, compress_level=0)
        # JPEG is a high-quality, 4:4:4, non-optimized first-generation capture.
        _composite_rgb(image).save(
            jpeg_path, format="JPEG", quality=95, subsampling=0,
            optimize=False, progressive=False,
        )
        image.filter(ImageFilter.GaussianBlur(radius=1.15)).save(
            controls["control-blur"]/f"png__{scene.name}.png", optimize=True,
        )
        banded = np.asarray(image, dtype=np.uint8).copy()
        banded[..., :3] = np.minimum(255, ((banded[..., :3].astype(np.uint16)+8)//16)*16).astype(np.uint8)
        Image.fromarray(banded, "RGBA").save(
            controls["control-banded"]/f"png__{scene.name}.png", optimize=True,
        )
        image.filter(ImageFilter.UnsharpMask(radius=2.2, percent=240, threshold=0)).save(
            controls["control-halo"]/f"png__{scene.name}.png", optimize=True,
        )
        cases.append({
            "name": scene.name,
            "description": scene.description,
            "tags": list(scene.tags),
            "reference": str(reference_path.relative_to(root)),
            "upload_png": str(png_path.relative_to(root)),
            "upload_jpeg": str(jpeg_path.relative_to(root)),
            "probes": list(scene.probes),
        })
        preview.append((scene.name, image))

    manifest: dict[str, object] = {
        "schema": 1,
        "size": [size, size],
        "supersample": supersample,
        "jpeg_composite": [239, 241, 244],
        "cases": cases,
        "protocol": {
            "upload": "Upload every file in upload/ to TinyPNG without renaming.",
            "returned": "Place returned files in candidates/tinypng/.",
            "comparison": "Run our codecs at each TinyPNG file's exact byte target, then evaluate against references/.",
            "controls": "Blur, coarse banding, and halo controls calibrate metric direction before optimizer comparison.",
        },
    }
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    _contact_sheet(preview, size).save(root/"contact_sheet.png", optimize=True)
    return manifest
