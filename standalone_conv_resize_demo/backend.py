"""Self-contained native resampling backend used by both demos."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
from typing import Literal

import numpy as np


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "native" / "conv_native.c"
BUILD = ROOT / ".native_build"
_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None


def _library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt":
        return ".dll"
    return ".so"


def build_native(*, force: bool = False) -> Path:
    """Compile the bundled C backend and return the resulting library."""

    identity = (
        SOURCE.read_bytes() + sys.platform.encode() + platform.machine().encode()
    )
    digest = hashlib.sha256(identity).hexdigest()[:12]
    BUILD.mkdir(exist_ok=True)
    target = BUILD / f"conv_native_{digest}{_library_suffix()}"
    if target.exists() and not force:
        return target
    compiler = os.environ.get("CC", "cc")
    if sys.platform == "darwin":
        command = [
            compiler, "-std=c11", "-O3", "-DNDEBUG", "-dynamiclib",
            "-fvisibility=hidden", "-pthread", str(SOURCE), "-o", str(target),
        ]
    elif os.name == "nt":
        raise RuntimeError(
            "Automatic native compilation currently expects Clang or GCC. "
            "Run from a MinGW shell or set CC to a compatible compiler."
        )
    else:
        command = [
            compiler, "-std=c11", "-O3", "-DNDEBUG", "-shared", "-fPIC",
            "-fvisibility=hidden", "-pthread", str(SOURCE), "-lm", "-o", str(target),
        ]
    if platform.machine().lower() in {"arm64", "aarch64"}:
        command.insert(3, "-mcpu=native")
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            "Native backend compilation failed:\n"
            + completed.stdout + completed.stderr
        )
    return target


def native_library() -> ctypes.CDLL:
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        library = ctypes.CDLL(str(build_native()))
        pointer = ctypes.POINTER(ctypes.c_float)
        for name, extra in (
            ("conv_resize_lines_f32", []),
            ("conv_basin_average_lines_f32", []),
            ("fir_resize_lines_f32", [ctypes.c_int]),
            ("linear_resize_lines_f32", []),
        ):
            function = getattr(library, name)
            function.argtypes = [
                pointer, ctypes.c_int, ctypes.c_int,
                pointer, ctypes.c_int, *extra,
            ]
            function.restype = ctypes.c_int
        double_pointer = ctypes.POINTER(ctypes.c_double)
        library.conv_basin_average_lines_f64.argtypes = [
            double_pointer, ctypes.c_int, ctypes.c_int,
            double_pointer, ctypes.c_int,
        ]
        library.conv_basin_average_lines_f64.restype = ctypes.c_int
        library.conv_moment_lines_f32.argtypes = [
            pointer,ctypes.c_int,ctypes.c_int,pointer,
        ]
        library.conv_moment_lines_f32.restype = ctypes.c_int
        library.conv_admit_moments_2d_f32.argtypes = [
            pointer, pointer, pointer, pointer,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            pointer, pointer, pointer,
        ]
        library.conv_admit_moments_2d_f32.restype = ctypes.c_int
        library.conv_four_child_moment_atlas_f32.argtypes = [
            pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, pointer,
        ]
        library.conv_four_child_moment_atlas_f32.restype = ctypes.c_int
        library.conv_evaluate_lines_f32.argtypes = [
            pointer, ctypes.c_int, ctypes.c_int, pointer, pointer,
        ]
        library.conv_evaluate_lines_f32.restype = ctypes.c_int
        library.conv_evaluate_profile_f32.argtypes = [
            pointer, ctypes.c_int, ctypes.c_int, pointer, pointer,
            ctypes.c_int,
        ]
        library.conv_evaluate_profile_f32.restype = ctypes.c_int
        library.conv_evaluate_profile_2d_f32.argtypes = [
            pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            pointer, pointer, pointer, ctypes.c_int,
        ]
        library.conv_evaluate_profile_2d_f32.restype = ctypes.c_int
        library.conv_backend_version.restype = ctypes.c_int
        library.conv_backend_has_neon.restype = ctypes.c_int
        library.conv_backend_threads.restype = ctypes.c_int
        library.easu_resize_f32.argtypes = [
            pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            pointer, ctypes.c_int, ctypes.c_int,
        ]
        library.easu_resize_f32.restype = ctypes.c_int
        library.conv_oriented_chord_blend_f32.argtypes = [
            pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            pointer, pointer, ctypes.c_int, ctypes.c_int,
        ]
        library.conv_oriented_chord_blend_f32.restype = ctypes.c_int
        library.conv_q1_order_blend_f32.argtypes = [
            pointer, ctypes.c_int, ctypes.c_int,
            pointer, pointer, ctypes.c_int,
            pointer, ctypes.c_int, ctypes.c_int,
        ]
        library.conv_q1_order_blend_f32.restype = ctypes.c_int
        library.conv_prepare_profile_f32.argtypes = [
            pointer, ctypes.c_int, ctypes.c_int, pointer,
        ]
        library.conv_prepare_profile_f32.restype = ctypes.c_int
        library.conv_fused_terminal.argtypes = [pointer]*5 + [ctypes.c_int]*5 + [pointer]
        library.conv_fused_terminal.restype = ctypes.c_int
        if library.conv_backend_version() != 1:
            raise RuntimeError("unsupported native backend ABI")
        _LIB = library
        return library


def backend_description() -> str:
    library = native_library()
    simd = "ARM NEON" if library.conv_backend_has_neon() else "scalar C"
    return (
        f"native C ABI 1, {simd}, float32, "
        f"{library.conv_backend_threads()} workers"
    )


def _resize_axis(
    values: Array,
    target: int,
    axis: int,
    method: Literal["conv", "fir", "linear"],
    *,
    radius: int = 3,
    workspace_megabytes: int = 96,
) -> Array:
    field = np.asarray(values, dtype=np.float32)
    target = int(target)
    axis = int(axis)
    if target < 1:
        raise ValueError("target extent must be positive")
    moved = np.moveaxis(field, axis, 0)
    n = moved.shape[0]
    if method == "conv" and n < 5:
        raise ValueError("CONV* requires at least five source anchors per axis")
    lines = np.ascontiguousarray(moved.reshape(n, -1))
    output = np.empty((target, lines.shape[1]), dtype=np.float32)
    if method == "conv":
        bytes_per_lane = max(1, (n - 1) * 5 * 4)
        chunk = max(4, workspace_megabytes * 1024 * 1024 // bytes_per_lane)
        chunk = max(4, chunk - chunk % 4)
        function = native_library().conv_resize_lines_f32
        arguments: tuple[object, ...] = ()
    elif method == "fir":
        chunk = max(4, 1 << 15)
        function = native_library().fir_resize_lines_f32
        arguments = (int(radius),)
    elif method == "linear":
        chunk = max(4, 1 << 16)
        function = native_library().linear_resize_lines_f32
        arguments = ()
    else:
        raise ValueError(f"unknown method {method!r}")
    pointer = ctypes.POINTER(ctypes.c_float)
    for start in range(0, lines.shape[1], chunk):
        stop = min(lines.shape[1], start + chunk)
        source_part = np.ascontiguousarray(lines[:, start:stop])
        output_part = np.empty((target, stop-start), dtype=np.float32)
        status = function(
            source_part.ctypes.data_as(pointer), n, stop-start,
            output_part.ctypes.data_as(pointer), target, *arguments,
        )
        if status:
            raise RuntimeError(f"native {method} resize failed with status {status}")
        output[:, start:stop] = output_part
    restored = output.reshape((target,) + moved.shape[1:])
    return np.moveaxis(restored, 0, axis)


def resize(
    values: Array,
    target_shape: tuple[int, ...] | int,
    method: Literal["conv", "fir", "linear"] = "conv",
    *,
    radius: int = 3,
) -> Array:
    """Endpoint-aligned 1-D or separable 2-D arbitrary resize.

    ``conv`` is exact CONV* analysis, ordered-current projection, and quintic
    synthesis in float32. ``fir`` is a scale-widened normalized Lanczos
    polyphase FIR; unlike point interpolation, it is antialiased on reduction.
    """

    field = np.asarray(values, dtype=np.float32)
    if isinstance(target_shape, int):
        target = (int(target_shape),)
    else:
        target = tuple(int(value) for value in target_shape)
    if field.ndim == 1:
        if len(target) != 1:
            raise ValueError("one-dimensional input needs one target extent")
        return _resize_axis(field, target[0], 0, method, radius=radius)
    if field.ndim not in (2, 3) or len(target) != 2:
        raise ValueError("input must be N, HxW, or HxWxC")
    height, width = target
    along_x = _resize_axis(field, width, 1, method, radius=radius)
    return _resize_axis(along_x, height, 0, method, radius=radius)


def conv_resize(values: Array, target_shape: tuple[int, ...] | int) -> Array:
    return resize(values, target_shape, "conv")


def conv_evaluate_lines(values: Array, positions: Array) -> Array:
    """Evaluate one arbitrary CONV site in each column of an N-by-L array."""

    lines = np.ascontiguousarray(values, dtype=np.float32)
    sites = np.ascontiguousarray(positions, dtype=np.float32)
    if lines.ndim != 2 or lines.shape[0] < 5:
        raise ValueError("CONV line evaluation requires an N-by-L array, N >= 5")
    if sites.shape != (lines.shape[1],):
        raise ValueError("positions must contain one site per line")
    output = np.empty(lines.shape[1], dtype=np.float32)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = native_library().conv_evaluate_lines_f32(
        lines.ctypes.data_as(pointer), lines.shape[0], lines.shape[1],
        sites.ctypes.data_as(pointer), output.ctypes.data_as(pointer),
    )
    if status:
        raise RuntimeError(f"native CONV line evaluation failed with status {status}")
    return output


def conv_evaluate_profile(values: Array, positions: Array) -> Array:
    """Evaluate one scalar or vector CONV profile at arbitrary sites."""

    field = np.asarray(values, dtype=np.float32)
    scalar = field.ndim == 1
    if scalar:
        field = field[:, None]
    if field.ndim != 2 or field.shape[0] < 5:
        raise ValueError("CONV profile evaluation requires N or N-by-C, N >= 5")
    field = np.ascontiguousarray(field)
    sites = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1)
    output = np.empty((sites.size, field.shape[1]), dtype=np.float32)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = native_library().conv_evaluate_profile_f32(
        field.ctypes.data_as(pointer), field.shape[0], field.shape[1],
        sites.ctypes.data_as(pointer), output.ctypes.data_as(pointer),
        sites.size,
    )
    if status:
        raise RuntimeError(f"native CONV profile evaluation failed with status {status}")
    return output[:, 0] if scalar else output


def conv_evaluate_profile_2d(
    values: Array, x_positions: Array, y_positions: Array
) -> Array:
    """Evaluate the tensor-covariant two-order CONV profile at arbitrary sites."""

    field = np.asarray(values, dtype=np.float32)
    scalar = field.ndim == 2
    if scalar:
        field = field[..., None]
    if field.ndim != 3 or min(field.shape[:2]) < 5:
        raise ValueError("2-D CONV profile evaluation requires HxW or HxWxC, H,W >= 5")
    x = np.ascontiguousarray(x_positions, dtype=np.float32).reshape(-1)
    y = np.ascontiguousarray(y_positions, dtype=np.float32).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("2-D CONV coordinates must have matching shapes")
    field = np.ascontiguousarray(field)
    output = np.empty((x.size, field.shape[2]), dtype=np.float32)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = native_library().conv_evaluate_profile_2d_f32(
        field.ctypes.data_as(pointer), field.shape[0], field.shape[1],
        field.shape[2], x.ctypes.data_as(pointer), y.ctypes.data_as(pointer),
        output.ctypes.data_as(pointer), x.size,
    )
    if status:
        raise RuntimeError(f"native 2-D CONV profile evaluation failed with status {status}")
    return output[:, 0] if scalar else output


def _basin_average_axis(values: Array, target: int, axis: int) -> Array:
    field = np.asarray(values, dtype=np.float32)
    moved = np.moveaxis(field, axis, 0)
    source_count = moved.shape[0]
    if source_count < 5 or target < 1 or target > source_count:
        raise ValueError("CONV basin analysis requires 1 <= target <= source")
    lines = np.ascontiguousarray(moved.reshape(source_count, -1))
    output = np.empty((target, lines.shape[1]), dtype=np.float32)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = native_library().conv_basin_average_lines_f32(
        lines.ctypes.data_as(pointer), source_count, lines.shape[1],
        output.ctypes.data_as(pointer), int(target),
    )
    if status:
        raise RuntimeError(f"native CONV basin analysis failed with status {status}")
    return np.moveaxis(output.reshape((target,) + moved.shape[1:]), 0, axis)


def conv_basin_average_lines_f64(values: Array, target: int) -> Array:
    """Double-precision ordered-CONV basin analysis for an N-by-lanes bank."""

    lines = np.ascontiguousarray(values, dtype=np.float64)
    if lines.ndim != 2 or lines.shape[0] < 5:
        raise ValueError("CONV float64 basin lines require N-by-lanes, N >= 5")
    if target < 1 or target > lines.shape[0]:
        raise ValueError("CONV basin analysis requires 1 <= target <= source")
    output = np.empty((int(target), lines.shape[1]), dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    status = native_library().conv_basin_average_lines_f64(
        lines.ctypes.data_as(pointer), lines.shape[0], lines.shape[1],
        output.ctypes.data_as(pointer), int(target),
    )
    if status:
        raise RuntimeError(
            f"native float64 CONV basin analysis failed with status {status}"
        )
    return output


def conv_passband_compensate_lines_f64(
    values: Array, *, compensation: float = 0.0
) -> Array:
    """Apply the moment-neutral fourth-order term to an existing line bank."""

    lines = np.ascontiguousarray(values, dtype=np.float64)
    if lines.ndim != 2 or lines.shape[0] < 1:
        raise ValueError("CONV passband compensation requires N-by-lanes")
    strength = float(compensation)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("passband compensation must be finite and in [0, 1]")
    output = lines.copy()
    if strength == 0.0 or output.shape[0] < 3:
        return output

    curvature = output[:-2] - 2.0 * output[1:-1] + output[2:]
    # The first and last canonical targets average half-width endpoint
    # basins.  Their centres therefore sit at one quarter of the ordinary
    # target spacing, not on the interior affine sequence.  Those two
    # boundary curvatures are geometric bookkeeping rather than recoverable
    # passband detail, so exclude them.  Every retained D2 row separately has
    # zero zeroth and first moment.
    curvature[0] = 0.0
    curvature[-1] = 0.0
    correction = np.zeros_like(output)
    correction[:-2] += curvature
    correction[1:-1] -= 2.0 * curvature
    correction[2:] += curvature
    output += (strength / 16.0) * correction
    return output


def conv_passband_basin_lines_f64(
    values: Array,
    target: int,
    *,
    compensation: float = 0.0,
) -> Array:
    """Float64 CONV basin analysis with moment-neutral passband recovery.

    The canonical ordered-CONV basin average remains the analysis operator.
    On its target lattice, this variant admits a controlled fraction of the
    fourth-order residual ``D2.T @ D2``.  Dividing by 16 normalizes the
    residual to unit gain at the one-dimensional Nyquist frequency, so a
    compensation of 0.10 means at most ten percent extra Nyquist response.

    Because the second-difference operator annihilates constants and affine
    ramps, and its transpose is orthogonal to both target-lattice zeroth and
    first moments, the correction does not alter DC, line centroid, or affine
    data.  Its low-frequency response begins at fourth order rather than the
    second order of an ordinary unsharp mask.  This makes it suitable for a
    small, explicitly certified reduction-kernel adjustment; it is not a
    general-purpose sharpening stage.
    """

    canonical = conv_basin_average_lines_f64(values, target)
    return conv_passband_compensate_lines_f64(
        canonical, compensation=compensation
    )


def conv_basin_average(
    values: Array, target_shape: tuple[int, ...] | int
) -> Array:
    """Average the ordered CONV profile over target-grid Voronoi basins."""

    field = np.asarray(values, dtype=np.float32)
    target = (int(target_shape),) if isinstance(target_shape, int) else tuple(
        map(int, target_shape)
    )
    if field.ndim == 1:
        if len(target) != 1:
            raise ValueError("one-dimensional input needs one target extent")
        return _basin_average_axis(field, target[0], 0)
    if field.ndim not in (2, 3) or len(target) != 2:
        raise ValueError("input must be N, HxW, or HxWxC")
    # Synthesis is horizontal then vertical.  Canonical basin analysis is the
    # reverse Cartesian composition: vertical then horizontal.
    along_y = _basin_average_axis(field, target[0], 0)
    return _basin_average_axis(along_y, target[1], 1)


def conv_transport_resize(
    values: Array, target_shape: tuple[int, ...] | int
) -> Array:
    """Scale-global CONV: basin integration on reduction, direct synthesis on enlargement."""

    field = np.asarray(values, dtype=np.float32)
    target = (int(target_shape),) if isinstance(target_shape, int) else tuple(
        map(int, target_shape)
    )
    if field.ndim == 1:
        if len(target) != 1:
            raise ValueError("one-dimensional input needs one target extent")
        return (
            _basin_average_axis(field, target[0], 0)
            if target[0] < field.shape[0]
            else _resize_axis(field, target[0], 0, "conv")
        )
    if field.ndim not in (2, 3) or len(target) != 2:
        raise ValueError("input must be N, HxW, or HxWxC")
    output = field
    # Complete reducing factors in reverse synthesis order.
    if target[0] < output.shape[0]:
        output = _basin_average_axis(output, target[0], 0)
    if target[1] < output.shape[1]:
        output = _basin_average_axis(output, target[1], 1)
    # Complete enlarging factors in the declared synthesis order.
    if target[1] > output.shape[1]:
        output = _resize_axis(output, target[1], 1, "conv")
    if target[0] > output.shape[0]:
        output = _resize_axis(output, target[0], 0, "conv")
    return output


def polyphase_fir_resize(
    values: Array, target_shape: tuple[int, ...] | int, *, radius: int = 8
) -> Array:
    return resize(values, target_shape, "fir", radius=radius)


def lanczos3_resize(values: Array, target_shape: tuple[int, ...] | int) -> Array:
    return resize(values, target_shape, "fir", radius=3)


def linear_resize(values: Array, target_shape: tuple[int, ...] | int) -> Array:
    return resize(values, target_shape, "linear")


def conv_oriented_chord_blend(
    values: Array, baseline: Array, target_shape: tuple[int, int]
) -> Array:
    """Native bounded tangent-chord blend over a supplied CONV baseline."""

    field = np.asarray(values, dtype=np.float32)
    scalar = field.ndim == 2
    if scalar:
        field = field[..., None]
    base = np.asarray(baseline, dtype=np.float32)
    if scalar and base.ndim == 2:
        base = base[..., None]
    target = tuple(map(int, target_shape))
    if field.ndim != 3 or base.shape != target + (field.shape[2],):
        raise ValueError("oriented chord inputs have inconsistent shapes")
    field = np.ascontiguousarray(field)
    base = np.ascontiguousarray(base)
    output = np.empty_like(base)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = native_library().conv_oriented_chord_blend_f32(
        field.ctypes.data_as(pointer), field.shape[0], field.shape[1],
        field.shape[2], base.ctypes.data_as(pointer),
        output.ctypes.data_as(pointer), target[0], target[1],
    )
    if status:
        raise RuntimeError(f"native oriented chord failed with status {status}")
    return output[..., 0] if scalar else output


def q1_order_blend(
    eta: Array,
    forward: Array,
    reverse: Array,
) -> Array:
    """Interpolate nodal order coordinates and blend both orders in one pass."""

    coordinate = np.ascontiguousarray(eta, dtype=np.float32)
    first = np.asarray(forward, dtype=np.float32)
    second = np.asarray(reverse, dtype=np.float32)
    scalar = first.ndim == 2
    if scalar:
        first = first[..., None]
        second = second[..., None]
    if (
        coordinate.ndim != 2
        or coordinate.shape[0] < 2
        or coordinate.shape[1] < 2
        or first.ndim != 3
        or second.shape != first.shape
    ):
        raise ValueError("Q1 blend inputs have inconsistent shapes")
    first = np.ascontiguousarray(first)
    second = np.ascontiguousarray(second)
    output = np.empty_like(first)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = native_library().conv_q1_order_blend_f32(
        coordinate.ctypes.data_as(pointer),
        coordinate.shape[0], coordinate.shape[1],
        first.ctypes.data_as(pointer), second.ctypes.data_as(pointer),
        first.shape[2], output.ctypes.data_as(pointer),
        first.shape[0], first.shape[1],
    )
    if status:
        raise RuntimeError(f"native Q1 order blend failed with status {status}")
    return output[..., 0] if scalar else output


def easu_resize(values: Array, target_shape: tuple[int, int]) -> Array:
    """Native FP32 AMD FSR 1.0 EASU reference geometry, without RCAS."""

    field = np.asarray(values, dtype=np.float32)
    scalar = field.ndim == 2
    if scalar:
        field = field[..., None]
    if field.ndim != 3:
        raise ValueError("EASU input must be HxW or HxWxC")
    field = np.ascontiguousarray(field)
    height, width, channels = field.shape
    out_height, out_width = map(int, target_shape)
    output = np.empty((out_height, out_width, channels), dtype=np.float32)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = native_library().easu_resize_f32(
        field.ctypes.data_as(pointer), height, width, channels,
        output.ctypes.data_as(pointer), out_height, out_width,
    )
    if status:
        raise RuntimeError(f"native EASU resize failed with status {status}")
    return output[..., 0] if scalar else output


def conv_moment_axis(values: Array, axis: int) -> Array:
    """Exact half-cell moment proposal of the admitted CONV profile."""

    field=np.asarray(values,dtype=np.float32)
    moved=np.moveaxis(field,int(axis),0)
    n=moved.shape[0]
    if n<5:
        raise ValueError("CONV moments require at least five cells per axis")
    lines=np.ascontiguousarray(moved.reshape(n,-1))
    output=np.empty_like(lines)
    pointer=ctypes.POINTER(ctypes.c_float)
    status=native_library().conv_moment_lines_f32(
        lines.ctypes.data_as(pointer),n,lines.shape[1],
        output.ctypes.data_as(pointer),
    )
    if status:
        raise RuntimeError(f"native CONV moment analysis failed with status {status}")
    restored=output.reshape(moved.shape)
    return np.moveaxis(restored,0,int(axis))


def admit_moments_2d(
    coarse: Array, horizontal: Array, vertical: Array, mixed: Array
) -> tuple[Array, Array, Array]:
    """Native complete face-current admission for a 2-D moment field."""

    value = np.asarray(coarse, dtype=np.float32)
    scalar = value.ndim == 2
    if scalar:
        value = value[..., None]
    if value.ndim != 3 or min(value.shape[:2]) < 2:
        raise ValueError("moment admission requires an HxW or HxWxC field")
    proposals = []
    for proposal in (horizontal, vertical, mixed):
        item = np.asarray(proposal, dtype=np.float32)
        if scalar and item.ndim == 2:
            item = item[..., None]
        if item.shape != value.shape:
            raise ValueError("every proposed moment must have the coarse shape")
        proposals.append(np.ascontiguousarray(item))
    value = np.ascontiguousarray(value)
    outputs = [np.empty_like(value) for _ in range(3)]
    pointer = ctypes.POINTER(ctypes.c_float)
    status = native_library().conv_admit_moments_2d_f32(
        value.ctypes.data_as(pointer),
        proposals[0].ctypes.data_as(pointer),
        proposals[1].ctypes.data_as(pointer),
        proposals[2].ctypes.data_as(pointer),
        value.shape[0], value.shape[1], value.shape[2],
        outputs[0].ctypes.data_as(pointer),
        outputs[1].ctypes.data_as(pointer),
        outputs[2].ctypes.data_as(pointer),
    )
    if status:
        raise RuntimeError(f"native moment admission failed with status {status}")
    if scalar:
        return tuple(item[..., 0] for item in outputs)
    return tuple(outputs)


def conv_four_child_moment_atlas(values: Array) -> Array:
    """Native admitted four-child atlas for one conservative dyadic level."""

    field = np.asarray(values, dtype=np.float32)
    scalar = field.ndim == 2
    if scalar:
        field = field[..., None]
    if field.ndim != 3 or min(field.shape[:2]) < 5:
        raise ValueError(
            "moment synthesis requires an HxW or HxWxC field of at least 5x5"
        )
    field = np.ascontiguousarray(field)
    height, width, channels = field.shape
    output = np.empty((2 * height, 2 * width, channels), dtype=np.float32)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = native_library().conv_four_child_moment_atlas_f32(
        field.ctypes.data_as(pointer), height, width, channels,
        output.ctypes.data_as(pointer),
    )
    if status:
        raise RuntimeError(f"native four-child moment atlas failed with status {status}")
    return output[..., 0] if scalar else output


def conv_two_order_synthesis(
    values: Array, target_shape: tuple[int, int], eta: Array,
    *, workspace_megabytes: int = 96,
) -> Array:
    """Original two admissions with fused terminal synthesis and Q1 blend.

    Retain the existing bounded-workspace route when both terminal profiles
    and the horizontal Q1 table would exceed its 96 MiB workspace budget.
    """
    field = np.asarray(values, dtype=np.float32)
    target = tuple(map(int, target_shape))
    coordinate = np.ascontiguousarray(eta, dtype=np.float32)
    if (field.ndim not in (2, 3) or len(target) != 2
            or min(field.shape[:2]) < 5 or min(target) < 1
            or coordinate.shape != field.shape[:2]
            or (field.ndim == 3 and field.shape[2] < 1)):
        raise ValueError("two-order synthesis requires HxW or HxWxC data, matching eta, and positive target extents")
    h, w = field.shape[:2]
    oh, ow = target
    channels = 1 if field.ndim == 2 else field.shape[2]
    workspace = 20*channels*((h-1)*ow+(w-1)*oh) + 8*h*ow
    if workspace > max(0, int(workspace_megabytes))*1024*1024:
        forward = conv_resize(field, target)
        reverse = np.swapaxes(conv_resize(np.swapaxes(field, 0, 1), (ow, oh)), 0, 1)
        return q1_order_blend(coordinate, forward, reverse)
    x = np.ascontiguousarray(_resize_axis(field, ow, 1, "conv").reshape(h, ow*channels))
    y = np.ascontiguousarray(np.moveaxis(_resize_axis(field, oh, 0, "conv"), 1, 0).reshape(w, oh*channels))
    library = native_library()
    pointer = ctypes.POINTER(ctypes.c_float)
    def prepare(lines):
        n, lanes = lines.shape
        current = np.empty((n-1, 5, lanes), dtype=np.float32)
        status = library.conv_prepare_profile_f32(
            lines.ctypes.data_as(pointer), n, lanes, current.ctypes.data_as(pointer))
        if status:
            raise RuntimeError(f"native CONV profile preparation failed with status {status}")
        return current
    cy, cx = prepare(x), prepare(y)
    output = np.empty(target if field.ndim == 2 else target+(channels,), dtype=np.float32)
    arguments = [array.ctypes.data_as(pointer) for array in (x, y, cy, cx, coordinate)]
    status = library.conv_fused_terminal(
        *arguments, h, w, oh, ow, channels, output.ctypes.data_as(pointer))
    if status:
        raise RuntimeError(f"native fused CONV synthesis failed with status {status}")
    return output
