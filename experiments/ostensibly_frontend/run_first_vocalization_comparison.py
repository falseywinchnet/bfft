#!/usr/bin/env python3
"""Compare the first extracted vocalization with a literal phone synthesis."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile

from .fourier_unrolled_densified import (
    DEFAULT_APERTURES,
    fourier_unrolled_densified,
    reassigned_texture_baseline,
)
from .synthetic_consonants import CONSONANTS, synthesize_consonant
from .synthetic_vowels import (
    DIPHTHONGS,
    FORMANTS,
    VOICE_PROBES,
    apply_ssb_probe_channel,
    synthesize_english_vowel,
)


SAMPLE_RATE = 48_000
HOP_LENGTH = 512


def _phone(
    label: str,
    duration: float,
    voice_index: int,
    seed: int,
) -> np.ndarray:
    voice = VOICE_PROBES[voice_index]
    if label in FORMANTS or label in DIPHTHONGS:
        return synthesize_english_vowel(
            label,
            voice,
            sample_rate=SAMPLE_RATE,
            duration=duration,
            seed=seed,
        )
    if label in CONSONANTS:
        return synthesize_consonant(
            label,
            voice,
            sample_rate=SAMPLE_RATE,
            duration=duration,
            seed=seed,
        )
    raise KeyError(label)


def _crossfade(chunks: tuple[np.ndarray, ...], overlap: int) -> np.ndarray:
    output = np.asarray(chunks[0], dtype=np.float64).copy()
    for chunk in chunks[1:]:
        right = np.asarray(chunk, dtype=np.float64)
        count = min(overlap, output.size // 2, right.size // 2)
        if count:
            phase = np.linspace(0.0, np.pi / 2.0, count)
            mixed = output[-count:] * np.cos(phase) ** 2 + right[:count] * np.sin(
                phase
            ) ** 2
            output = np.concatenate((output[:-count], mixed, right[count:]))
        else:
            output = np.concatenate((output, right))
    return output


def synthesize_phone_string(
    phones: tuple[str, ...],
    duration_seconds: float,
    voice_index: int,
    seed: int,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Create one duration-matched literal phone string with 10 ms joins."""

    if not phones or duration_seconds <= 0.0:
        raise ValueError("phone synthesis needs content and positive duration")
    overlap_seconds = 0.010
    generated_total = duration_seconds + overlap_seconds * (len(phones) - 1)
    base_weights = {
        "AO": 1.30,
        "AY": 1.55,
        "L": 0.95,
        "R": 0.90,
        "T": 0.80,
    }
    weights = np.asarray(
        [base_weights.get(phone, 1.0) for phone in phones], dtype=np.float64
    )
    durations = generated_total * weights / np.sum(weights)
    chunks = tuple(
        _phone(phone, float(duration), voice_index, seed + 101 * index)
        for index, (phone, duration) in enumerate(zip(phones, durations, strict=True))
    )
    joined = _crossfade(chunks, int(round(overlap_seconds * SAMPLE_RATE)))
    target_count = int(round(duration_seconds * SAMPLE_RATE))
    if joined.size != target_count:
        source = np.linspace(0.0, 1.0, joined.size)
        target = np.linspace(0.0, 1.0, target_count)
        joined = np.interp(target, source, joined)
    joined /= max(float(np.max(np.abs(joined))), 1e-30)
    return joined, tuple(map(float, durations))


def duration_match(samples: np.ndarray, target_count: int) -> np.ndarray:
    """Deterministically stretch one generated waveform to the query duration."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("duration matching needs a nontrivial mono waveform")
    if target_count < 2:
        raise ValueError("target duration must contain at least two samples")
    if values.size == target_count:
        return values.copy()
    source = np.linspace(0.0, 1.0, values.size)
    target = np.linspace(0.0, 1.0, target_count)
    return np.interp(target, source, values)


def _trace(samples: np.ndarray) -> np.ndarray:
    densified = fourier_unrolled_densified(
        samples,
        apertures=DEFAULT_APERTURES,
        target_n_fft=2048,
        hop_length=HOP_LENGTH,
        target_crop_rows=256,
    )
    baseline = reassigned_texture_baseline(
        densified.fused, densified.support_union
    )
    return np.asarray(baseline.merged[:128], dtype=np.float64)


def _display(field: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(field, dtype=np.float64), 0.0)
    scale = max(float(np.percentile(values[values > 0.0], 99.7)), 1e-30)
    normalized = np.clip(values / scale, 0.0, 1.0)
    return np.log1p(12.0 * normalized) / np.log(13.0)


def _plot(
    real: np.ndarray,
    synthetic: np.ndarray,
    duration_seconds: float,
    phones: tuple[str, ...],
    synthetic_title: str | None = None,
) -> bytes:
    def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", size
            )
        except OSError:
            return ImageFont.load_default()

    def colorize(field: np.ndarray) -> Image.Image:
        values = np.flipud(_display(field))
        stops = np.asarray(
            [
                (0.00, 0, 0, 4),
                (0.20, 47, 15, 61),
                (0.40, 120, 28, 109),
                (0.60, 191, 54, 96),
                (0.80, 249, 142, 8),
                (1.00, 252, 253, 191),
            ],
            dtype=np.float64,
        )
        rgb = np.empty((*values.shape, 3), dtype=np.float64)
        for channel_index in range(3):
            rgb[..., channel_index] = np.interp(
                values, stops[:, 0], stops[:, channel_index + 1]
            )
        return Image.fromarray(np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8))

    width, height = 1500, 570
    canvas = Image.new("RGB", (width, height), (17, 19, 24))
    draw = ImageDraw.Draw(canvas)
    foreground = (235, 237, 241)
    muted = (168, 173, 184)
    frame = (88, 94, 108)
    draw.text(
        (width // 2, 18),
        "Same Fourier-unrolled → reassigned-support → merged-ridge print",
        font=font(21),
        fill=foreground,
        anchor="ma",
    )
    panels = (
        (real, "Extracted first vocalization — ‘all right’"),
        (
            synthetic,
            synthetic_title or f"Literal synthetic — {' '.join(phones)}",
        ),
    )
    panel_width, panel_height = 650, 400
    for panel_index, (field, title) in enumerate(panels):
        x0 = 72 + panel_index * 742
        y0 = 92
        heatmap = colorize(field).resize(
            (panel_width, panel_height), Image.Resampling.BILINEAR
        )
        canvas.paste(heatmap, (x0, y0))
        draw.rectangle(
            (x0, y0, x0 + panel_width, y0 + panel_height), outline=frame, width=1
        )
        draw.text(
            (x0 + panel_width // 2, 62),
            title,
            font=font(18),
            fill=foreground,
            anchor="ma",
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = x0 + int(round(fraction * panel_width))
            draw.line((x, y0 + panel_height, x, y0 + panel_height + 5), fill=frame)
            draw.text(
                (x, y0 + panel_height + 9),
                f"{fraction * duration_seconds:.2f}",
                font=font(13),
                fill=muted,
                anchor="ma",
            )
        draw.text(
            (x0 + panel_width // 2, y0 + panel_height + 36),
            "time (s)",
            font=font(14),
            fill=foreground,
            anchor="ma",
        )
        for value in (0, 32, 64, 96, 128):
            y = y0 + panel_height - int(round(value / 128.0 * panel_height))
            draw.line((x0 - 5, y, x0, y), fill=frame)
            draw.text(
                (x0 - 10, y),
                str(value),
                font=font(13),
                fill=muted,
                anchor="rm",
            )
    label_layer = Image.new("RGBA", (230, 24), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text(
        (115, 12),
        "unrolled row (low → high)",
        font=font(14),
        fill=foreground,
        anchor="mm",
    )
    rotated = label_layer.rotate(90, expand=True)
    canvas.paste(rotated, (6, 175), rotated)
    buffer = BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extracted_trace", type=Path)
    parser.add_argument("--phones", default="AO,L,R,AY,T")
    parser.add_argument("--voice-index", type=int, default=1)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--generated-npz",
        type=Path,
        help="Use an existing generated waveform instead of the literal phone string",
    )
    parser.add_argument(
        "--generated-key",
        default="synthetic_samples",
        help="Waveform key inside --generated-npz",
    )
    parser.add_argument(
        "--generated-title",
        default="New articulatory form — generated ‘all’",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--visualization", type=Path, required=True)
    args = parser.parse_args()
    phones = tuple(value for value in args.phones.split(",") if value)
    if not 0 <= args.voice_index < len(VOICE_PROBES):
        raise ValueError("voice index is outside the deterministic probe bank")
    with np.load(args.extracted_trace, allow_pickle=False) as document:
        real = np.asarray(document["query_trace"], dtype=np.float64)
    duration_seconds = real.shape[1] * HOP_LENGTH / SAMPLE_RATE
    target_count = int(round(duration_seconds * SAMPLE_RATE))
    if args.generated_npz is None:
        clean, durations = synthesize_phone_string(
            phones,
            duration_seconds,
            args.voice_index,
            args.seed,
        )
        channel = apply_ssb_probe_channel(
            clean,
            sample_rate=SAMPLE_RATE,
            noise_db=args.noise_db,
            seed=args.seed + 7001,
        )
        synthetic_title = None
        waveform_name = "synthetic-all-right.wav"
    else:
        with np.load(args.generated_npz, allow_pickle=False) as document:
            generated = np.asarray(document[args.generated_key], dtype=np.float64)
        channel = duration_match(generated, target_count)
        durations = ()
        synthetic_title = args.generated_title
        waveform_name = "generated-comparison-source.wav"
    synthetic = _trace(channel)
    png = _plot(
        real,
        synthetic,
        duration_seconds,
        phones,
        synthetic_title=synthetic_title,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "first-vocalization-comparison.png").write_bytes(png)
    wavfile.write(
        args.out / waveform_name,
        SAMPLE_RATE,
        np.asarray(np.clip(channel, -1.0, 1.0) * 32767.0, dtype=np.int16),
    )
    np.savez_compressed(
        args.out / "first-vocalization-comparison.npz",
        real_trace=real.astype(np.float32),
        synthetic_trace=synthetic.astype(np.float32),
        synthetic_samples=channel.astype(np.float32),
        phones=np.asarray(phones),
        phone_durations=np.asarray(durations, dtype=np.float64),
    )
    encoded = base64.b64encode(png).decode("ascii")
    fragment = f"""<div id="first-vocalization-comparison" style="color:var(--foreground);font:12px/1.4 ui-sans-serif,system-ui,sans-serif">
  <div style="font-size:14px;font-weight:600;margin:0 0 8px">First bounded vocalization versus literal phone synthesis</div>
  <img src="data:image/png;base64,{encoded}" alt="Side-by-side Fourier-unrolled ridge prints of the extracted first vocalization all right and the selected generated waveform" style="display:block;width:100%;height:auto" />
  <div style="margin-top:6px;color:var(--muted-foreground)">Frames 4–80 · {duration_seconds:.3f} s · both panels independently scaled at their 99.7th percentile · identical print pipeline · no learned model</div>
</div>
"""
    args.visualization.parent.mkdir(parents=True, exist_ok=True)
    args.visualization.write_text(fragment, encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "visualization": str(args.visualization),
        "phones": list(phones),
        "duration_seconds": duration_seconds,
        "voice": VOICE_PROBES[args.voice_index].name,
        "phone_durations": durations,
        "real_shape": list(real.shape),
        "synthetic_shape": list(synthetic.shape),
    }, indent=2))


if __name__ == "__main__":
    main()
