"""ctypes bridge for the SIMD retained-rank child transport kernel."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


Array = np.ndarray
_U32_MAX = np.iinfo(np.uint32).max


class _Block(ctypes.Structure):
    _fields_ = [
        ("source_offset", ctypes.c_uint32),
        ("source_count", ctypes.c_uint32),
        ("receiver_offset", ctypes.c_uint32),
        ("receiver_count", ctypes.c_uint32),
        ("source_base", ctypes.c_uint32),
        ("receiver_base", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("optical_depth", ctypes.c_double),
        ("extinction", ctypes.c_double),
        ("linear_diffusive_extinction", ctypes.c_double),
        ("circular_diffusive_extinction", ctypes.c_double),
    ]


class _Stats(ctypes.Structure):
    _fields_ = [
        ("block_applications", ctypes.c_uint64),
        ("gathered_source_coefficients", ctypes.c_uint64),
        ("scattered_receiver_coefficients", ctypes.c_uint64),
        ("contiguous_source_blocks", ctypes.c_uint64),
        ("contiguous_receiver_blocks", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class NativeApplyStats:
    block_applications: int
    gathered_source_coefficients: int
    scattered_receiver_coefficients: int
    contiguous_source_blocks: int
    contiguous_receiver_blocks: int


def _library_names() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("libphotonic_retained.dylib",)
    if os.name == "nt":
        return ("photonic_retained.dll",)
    return ("libphotonic_retained.so",)


def candidate_library_paths() -> tuple[Path, ...]:
    explicit = os.environ.get("PFT_RETAINED_LIBRARY")
    paths = [Path(explicit)] if explicit else []
    native = Path(__file__).resolve().parent / "native"
    paths.extend(native / name for name in _library_names())
    return tuple(paths)


def _load_library() -> ctypes.CDLL:
    failures = []
    for path in candidate_library_paths():
        try:
            library = ctypes.CDLL(str(path))
        except OSError as error:
            failures.append(f"{path}: {error}")
            continue
        function = library.pft_retained_apply_f64
        function.restype = ctypes.c_int
        function.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(_Block),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(_Stats),
        ]
        library.pft_retained_backend.restype = ctypes.c_char_p
        library.pft_retained_backend.argtypes = []
        library.pft_retained_plan_create_f64.restype = ctypes.c_int
        library.pft_retained_plan_create_f64.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(_Block),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.pft_retained_plan_apply_f64.restype = ctypes.c_int
        library.pft_retained_plan_apply_f64.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(_Stats),
        ]
        library.pft_retained_plan_destroy.restype = None
        library.pft_retained_plan_destroy.argtypes = [ctypes.c_void_p]
        library.pft_retained_mode_plan_create_f64.restype = ctypes.c_int
        library.pft_retained_mode_plan_create_f64.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.pft_retained_mode_plan_apply_f64.restype = ctypes.c_int
        library.pft_retained_mode_plan_apply_f64.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(_Stats),
        ]
        library.pft_retained_mode_plan_destroy.restype = None
        library.pft_retained_mode_plan_destroy.argtypes = [ctypes.c_void_p]
        return library
    raise RuntimeError(
        "native retained-rank library is unavailable; build it with "
        "`make -C experiments/photonic_transport_field/native retained`\n"
        + "\n".join(failures)
    )


def native_available() -> bool:
    try:
        _load_library()
    except RuntimeError:
        return False
    return True


def _contiguous_base(indices: Array) -> int | None:
    if indices.size == 0:
        return None
    base = int(indices[0])
    expected = np.arange(base, base + indices.size, dtype=np.uint32)
    return base if np.array_equal(indices, expected) else None


class NativeRetainedPlan:
    """Immutable packed retained-rank execution plan.

    The plan stores each block as two factor vectors.  It does not contain or
    construct the Cartesian source-by-receiver edge expansion.
    """

    def __init__(
        self,
        node_count: int,
        blocks: Iterable[
            tuple[Array, Array, Array, Array, float, float, float, float]
        ],
    ) -> None:
        if node_count < 1 or node_count > _U32_MAX:
            raise ValueError("native node count must fit a nonzero uint32 range")
        self.node_count = int(node_count)
        descriptors = []
        source_nodes = []
        source_factors = []
        receiver_nodes = []
        receiver_factors = []
        for specification in blocks:
            (
                raw_sources,
                raw_source_factors,
                raw_receivers,
                raw_receiver_factors,
                optical_depth,
                extinction,
                linear_extinction,
                circular_extinction,
            ) = specification
            sources = np.asarray(raw_sources, dtype=np.uint32)
            receivers = np.asarray(raw_receivers, dtype=np.uint32)
            u = np.asarray(raw_source_factors, dtype=np.float64)
            v = np.asarray(raw_receiver_factors, dtype=np.float64)
            if sources.ndim != 1 or receivers.ndim != 1:
                raise ValueError("native block indices must be vectors")
            if u.shape != sources.shape or v.shape != receivers.shape:
                raise ValueError("native block factors do not align")
            if (
                sources.size == 0
                or receivers.size == 0
                or np.any(sources >= node_count)
                or np.any(receivers >= node_count)
                or not np.all(np.isfinite(u))
                or not np.all(np.isfinite(v))
            ):
                raise ValueError("invalid native retained block")
            source_base = _contiguous_base(sources)
            receiver_base = _contiguous_base(receivers)
            flags = (1 if source_base is not None else 0) | (
                2 if receiver_base is not None else 0
            )
            descriptors.append(
                _Block(
                    len(source_nodes),
                    sources.size,
                    len(receiver_nodes),
                    receivers.size,
                    source_base or 0,
                    receiver_base or 0,
                    flags,
                    0,
                    float(optical_depth),
                    float(extinction),
                    float(linear_extinction),
                    float(circular_extinction),
                )
            )
            source_nodes.extend(int(value) for value in sources)
            source_factors.extend(float(value) for value in u)
            receiver_nodes.extend(int(value) for value in receivers)
            receiver_factors.extend(float(value) for value in v)
        self._blocks = (_Block * len(descriptors))(*descriptors)
        self._source_nodes = np.ascontiguousarray(source_nodes, dtype=np.uint32)
        self._source_factors = np.ascontiguousarray(source_factors, dtype=np.float64)
        self._receiver_nodes = np.ascontiguousarray(receiver_nodes, dtype=np.uint32)
        self._receiver_factors = np.ascontiguousarray(receiver_factors, dtype=np.float64)
        self._library = _load_library()
        self._handle = ctypes.c_void_p()
        status = self._library.pft_retained_plan_create_f64(
            self.node_count,
            len(self._blocks),
            self._blocks,
            self._source_nodes.size,
            self._pointer(self._source_nodes, ctypes.c_uint32),
            self._pointer(self._source_factors, ctypes.c_double),
            self._receiver_nodes.size,
            self._pointer(self._receiver_nodes, ctypes.c_uint32),
            self._pointer(self._receiver_factors, ctypes.c_double),
            ctypes.byref(self._handle),
        )
        if status:
            raise RuntimeError(f"native retained plan creation failed with status {status}")

    @classmethod
    def from_transport_field(cls, field) -> "NativeRetainedPlan":
        """Pack a geometric hierarchical field without expanding its blocks."""
        return cls(
            int(field.node_count),
            (
                (
                    block.source,
                    block.source_factor,
                    block.receiver,
                    block.receiver_factor,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                for block in field.blocks
            ),
        )

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            self._library.pft_retained_plan_destroy(handle)
            handle.value = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def source_coefficient_count(self) -> int:
        return int(self._source_nodes.size)

    @property
    def receiver_coefficient_count(self) -> int:
        return int(self._receiver_nodes.size)

    @property
    def stored_coefficient_count(self) -> int:
        return self.source_coefficient_count + self.receiver_coefficient_count

    @property
    def backend(self) -> str:
        return self._library.pft_retained_backend().decode("ascii")

    @staticmethod
    def _pointer(array: Array, ctype: type[ctypes._SimpleCData]):
        return array.ctypes.data_as(ctypes.POINTER(ctype))

    def apply(
        self,
        powers: Array,
        active_nodes: Array,
        linear_peakedness: Array,
        chirality: Array,
    ) -> tuple[Array, NativeApplyStats]:
        power = np.ascontiguousarray(powers, dtype=np.float64)
        active = np.ascontiguousarray(active_nodes, dtype=np.uint8)
        linear = np.ascontiguousarray(linear_peakedness, dtype=np.float64)
        circular = np.ascontiguousarray(chirality, dtype=np.float64)
        if power.ndim != 2 or power.shape[1] != self.node_count:
            raise ValueError("native powers must have shape modes by nodes")
        if active.shape != (self.node_count,):
            raise ValueError("native active-node mask has the wrong shape")
        if linear.shape != (power.shape[0],) or circular.shape != linear.shape:
            raise ValueError("native mode factors do not align with powers")
        if not all(np.all(np.isfinite(value)) for value in (power, linear, circular)):
            raise ValueError("native retained inputs must be finite")
        output = np.empty_like(power)
        scratch = np.empty(power.shape[0], dtype=np.float64)
        stats = _Stats()
        status = self._library.pft_retained_plan_apply_f64(
            self._handle,
            power.shape[0],
            self._pointer(active, ctypes.c_uint8),
            self._pointer(linear, ctypes.c_double),
            self._pointer(circular, ctypes.c_double),
            self._pointer(power, ctypes.c_double),
            self._pointer(output, ctypes.c_double),
            self._pointer(scratch, ctypes.c_double),
            ctypes.byref(stats),
        )
        if status:
            raise RuntimeError(f"native retained-rank kernel failed with status {status}")
        return output, NativeApplyStats(
            int(stats.block_applications),
            int(stats.gathered_source_coefficients),
            int(stats.scattered_receiver_coefficients),
            int(stats.contiguous_source_blocks),
            int(stats.contiguous_receiver_blocks),
        )

    def bind_modes(
        self,
        linear_peakedness: Array,
        chirality: Array,
    ) -> "NativeRetainedModePlan":
        return NativeRetainedModePlan(self, linear_peakedness, chirality)


class NativeRetainedModePlan:
    """A retained topology with its invariant extinction table precomputed."""

    def __init__(
        self,
        plan: NativeRetainedPlan,
        linear_peakedness: Array,
        chirality: Array,
    ) -> None:
        self.plan = plan
        self._linear = np.ascontiguousarray(linear_peakedness, dtype=np.float64)
        self._circular = np.ascontiguousarray(chirality, dtype=np.float64)
        if (
            self._linear.ndim != 1
            or self._circular.shape != self._linear.shape
            or self._linear.size == 0
            or not np.all(np.isfinite(self._linear))
            or not np.all(np.isfinite(self._circular))
        ):
            raise ValueError("bound native mode factors must be finite vectors")
        self._handle = ctypes.c_void_p()
        status = plan._library.pft_retained_mode_plan_create_f64(
            plan._handle,
            self._linear.size,
            plan._pointer(self._linear, ctypes.c_double),
            plan._pointer(self._circular, ctypes.c_double),
            ctypes.byref(self._handle),
        )
        if status:
            raise RuntimeError(f"native retained mode binding failed with status {status}")

    @property
    def mode_count(self) -> int:
        return self._linear.size

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            self.plan._library.pft_retained_mode_plan_destroy(handle)
            handle.value = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def apply(
        self,
        powers: Array,
        active_nodes: Array,
    ) -> tuple[Array, NativeApplyStats]:
        power = np.ascontiguousarray(powers, dtype=np.float64)
        active = np.ascontiguousarray(active_nodes, dtype=np.uint8)
        if power.shape != (self.mode_count, self.plan.node_count):
            raise ValueError("bound native powers have the wrong shape")
        if active.shape != (self.plan.node_count,):
            raise ValueError("bound native active-node mask has the wrong shape")
        if not np.all(np.isfinite(power)):
            raise ValueError("bound native powers must be finite")
        output = np.empty_like(power)
        scratch = np.empty(self.mode_count, dtype=np.float64)
        stats = _Stats()
        status = self.plan._library.pft_retained_mode_plan_apply_f64(
            self._handle,
            self.plan._pointer(active, ctypes.c_uint8),
            self.plan._pointer(power, ctypes.c_double),
            self.plan._pointer(output, ctypes.c_double),
            self.plan._pointer(scratch, ctypes.c_double),
            ctypes.byref(stats),
        )
        if status:
            raise RuntimeError(
                f"bound native retained-rank kernel failed with status {status}"
            )
        return output, NativeApplyStats(
            int(stats.block_applications),
            int(stats.gathered_source_coefficients),
            int(stats.scattered_receiver_coefficients),
            int(stats.contiguous_source_blocks),
            int(stats.contiguous_receiver_blocks),
        )
