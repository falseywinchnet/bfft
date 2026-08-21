"""Optional ctypes binding for the exact reflected exposure operator."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)
_INT64_POINTER = ctypes.POINTER(ctypes.c_int64)
_INT_POINTER = ctypes.POINTER(ctypes.c_int)
_LIBRARY: ctypes.CDLL | None = None
_LOAD_ATTEMPTED = False


def _candidate_libraries() -> list[Path]:
    override = os.environ.get("PERSONAL_DEBLURRER_NATIVE")
    if override and override.lower() not in {"0", "false", "off", "no"}:
        return [Path(override).expanduser()]
    root = Path(__file__).resolve().parent / "native"
    return [
        root / "libpersonal_deblurrer.dylib",
        root / "libpersonal_deblurrer.so",
    ]


def _load_library() -> ctypes.CDLL | None:
    global _LIBRARY, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LIBRARY
    _LOAD_ATTEMPTED = True
    disabled = os.environ.get("PERSONAL_DEBLURRER_NATIVE", "").lower()
    if disabled in {"0", "false", "off", "no"}:
        return None
    for candidate in _candidate_libraries():
        if not candidate.is_file():
            continue
        try:
            library = ctypes.CDLL(str(candidate))
            library.pdeb_path_operator_abi_version.argtypes = []
            library.pdeb_path_operator_abi_version.restype = ctypes.c_int
            abi_version = library.pdeb_path_operator_abi_version()
            if abi_version not in (5, 6):
                continue
            library.pdeb_path_operator_backend.argtypes = []
            library.pdeb_path_operator_backend.restype = ctypes.c_char_p
            arguments = [
                _DOUBLE_POINTER,
                _DOUBLE_POINTER,
                ctypes.c_int64,
                ctypes.c_int,
                _INT64_POINTER,
                _DOUBLE_POINTER,
                ctypes.c_int,
            ]
            library.pdeb_path_forward.argtypes = arguments
            library.pdeb_path_forward.restype = ctypes.c_int
            library.pdeb_path_adjoint.argtypes = arguments
            library.pdeb_path_adjoint.restype = ctypes.c_int
            library.pdeb_spatial_forward.argtypes = arguments
            library.pdeb_spatial_forward.restype = ctypes.c_int
            library.pdeb_spatial_adjoint.argtypes = arguments
            library.pdeb_spatial_adjoint.restype = ctypes.c_int
            batch_arguments = [*arguments, _INT_POINTER, ctypes.c_int]
            library.pdeb_spatial_batch_forward.argtypes = batch_arguments
            library.pdeb_spatial_batch_forward.restype = ctypes.c_int
            library.pdeb_spatial_batch_adjoint.argtypes = batch_arguments
            library.pdeb_spatial_batch_adjoint.restype = ctypes.c_int
            covariance_arguments = [
                _DOUBLE_POINTER,
                _DOUBLE_POINTER,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
                _DOUBLE_POINTER,
                _DOUBLE_POINTER,
                ctypes.c_int,
            ]
            library.pdeb_covariance_forward.argtypes = covariance_arguments
            library.pdeb_covariance_forward.restype = ctypes.c_int
            library.pdeb_covariance_adjoint.argtypes = covariance_arguments
            library.pdeb_covariance_adjoint.restype = ctypes.c_int
            if abi_version >= 6:
                covariance_batch_arguments = [
                    *covariance_arguments,
                    ctypes.c_int,
                ]
                library.pdeb_covariance_batch_forward.argtypes = (
                    covariance_batch_arguments)
                library.pdeb_covariance_batch_forward.restype = ctypes.c_int
                library.pdeb_covariance_batch_adjoint.argtypes = (
                    covariance_batch_arguments)
                library.pdeb_covariance_batch_adjoint.restype = ctypes.c_int
            library._personal_deblurrer_abi_version = abi_version
            _LIBRARY = library
            return library
        except (AttributeError, OSError):
            continue
    return None


def native_available() -> bool:
    """Return whether the verified ABI can be loaded without building it."""
    return _load_library() is not None


class NativeReflectedPathPlan:
    """Own contiguous plan arrays for the flat C gather/scatter ABI."""

    def __init__(
        self,
        source_indices: np.ndarray,
        weights: np.ndarray,
        shape: tuple[int, int],
    ) -> None:
        library = _load_library()
        if library is None:
            raise RuntimeError("native reflected-path ABI is unavailable")
        self._library = library
        self._indices = np.ascontiguousarray(source_indices, dtype=np.int64)
        self._weights = np.ascontiguousarray(weights, dtype=np.float64)
        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]
        if self._indices.shape != (len(self._weights), self.pixels):
            raise ValueError("native path plan has inconsistent dimensions")
        label = library.pdeb_path_operator_backend()
        self.backend = label.decode("ascii")

    def _apply(self, name: str, image: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(image, dtype=np.float64)
        if value.shape[:2] != self.shape or value.ndim not in (2, 3):
            raise ValueError("native path operator image shape mismatch")
        channels = 1 if value.ndim == 2 else int(value.shape[2])
        output = np.empty_like(value)
        function = getattr(self._library, name)
        status = function(
            value.ctypes.data_as(_DOUBLE_POINTER),
            output.ctypes.data_as(_DOUBLE_POINTER),
            self.pixels,
            channels,
            self._indices.ctypes.data_as(_INT64_POINTER),
            self._weights.ctypes.data_as(_DOUBLE_POINTER),
            len(self._weights),
        )
        if status != 0:
            raise RuntimeError(f"native path operator failed with status {status}")
        return output

    def forward(self, image: np.ndarray) -> np.ndarray:
        return self._apply("pdeb_path_forward", image)

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        return self._apply("pdeb_path_adjoint", image)


class NativeSpatialExposurePlan:
    """Own a spatially weighted bilinear gather/scatter plan for ABI v6."""

    def __init__(
        self,
        source_indices: np.ndarray,
        coefficients: np.ndarray,
        shape: tuple[int, int],
    ) -> None:
        library = _load_library()
        if library is None:
            raise RuntimeError("native spatial-exposure ABI is unavailable")
        self._library = library
        self._indices = np.ascontiguousarray(source_indices, dtype=np.int64)
        self._coefficients = np.ascontiguousarray(
            coefficients, dtype=np.float64)
        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]
        if self._indices.shape != self._coefficients.shape:
            raise ValueError("native spatial indices and coefficients must match")
        if self._indices.ndim != 2 or self._indices.shape[1] != self.pixels:
            raise ValueError("native spatial plan has inconsistent dimensions")
        label = library.pdeb_path_operator_backend()
        self.backend = label.decode("ascii")

    def _apply(self, name: str, image: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(image, dtype=np.float64)
        if value.shape[:2] != self.shape or value.ndim not in (2, 3):
            raise ValueError("native spatial operator image shape mismatch")
        channels = 1 if value.ndim == 2 else int(value.shape[2])
        output = np.empty_like(value)
        function = getattr(self._library, name)
        status = function(
            value.ctypes.data_as(_DOUBLE_POINTER),
            output.ctypes.data_as(_DOUBLE_POINTER),
            self.pixels,
            channels,
            self._indices.ctypes.data_as(_INT64_POINTER),
            self._coefficients.ctypes.data_as(_DOUBLE_POINTER),
            self._indices.shape[0],
        )
        if status != 0:
            raise RuntimeError(
                f"native spatial operator failed with status {status}")
        return output

    def forward(self, image: np.ndarray) -> np.ndarray:
        return self._apply("pdeb_spatial_forward", image)

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        return self._apply("pdeb_spatial_adjoint", image)


class NativeSpatialExposureBatchPlan:
    """Apply exchangeable spatial plans in one ABI v6 crossing."""

    def __init__(
        self,
        source_indices: np.ndarray,
        coefficients: np.ndarray,
        shape: tuple[int, int],
        contribution_counts: np.ndarray | None = None,
    ) -> None:
        library = _load_library()
        if library is None:
            raise RuntimeError("native spatial-exposure ABI is unavailable")
        self._library = library
        self._indices = np.ascontiguousarray(source_indices, dtype=np.int64)
        self._coefficients = np.ascontiguousarray(
            coefficients, dtype=np.float64)
        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]
        if self._indices.shape != self._coefficients.shape:
            raise ValueError("native batch indices and coefficients must match")
        if self._indices.ndim != 3 or self._indices.shape[2] != self.pixels:
            raise ValueError("native batch plan has inconsistent dimensions")
        self.plans = int(self._indices.shape[0])
        self.contributions = int(self._indices.shape[1])
        self._contribution_counts = np.ascontiguousarray(
            (np.full(self.plans, self.contributions, dtype=np.int32)
             if contribution_counts is None else contribution_counts),
            dtype=np.int32,
        )
        if (
            self._contribution_counts.shape != (self.plans,)
            or np.any(self._contribution_counts <= 0)
            or np.any(self._contribution_counts > self.contributions)
        ):
            raise ValueError("native batch contribution counts are invalid")
        label = library.pdeb_path_operator_backend()
        self.backend = f"{label.decode('ascii')}_batch"

    def _apply(self, name: str, images: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(images, dtype=np.float64)
        if value.shape[0] != self.plans or value.shape[1:3] != self.shape:
            raise ValueError("native spatial batch image shape mismatch")
        if value.ndim not in (3, 4):
            raise ValueError("native spatial batch expects SxHxW or SxHxWxC")
        channels = 1 if value.ndim == 3 else int(value.shape[3])
        output = np.empty_like(value)
        function = getattr(self._library, name)
        status = function(
            value.ctypes.data_as(_DOUBLE_POINTER),
            output.ctypes.data_as(_DOUBLE_POINTER),
            self.pixels,
            channels,
            self._indices.ctypes.data_as(_INT64_POINTER),
            self._coefficients.ctypes.data_as(_DOUBLE_POINTER),
            self.contributions,
            self._contribution_counts.ctypes.data_as(_INT_POINTER),
            self.plans,
        )
        if status != 0:
            raise RuntimeError(
                f"native spatial batch operator failed with status {status}")
        return output

    def forward(self, images: np.ndarray) -> np.ndarray:
        return self._apply("pdeb_spatial_batch_forward", images)

    def adjoint(self, images: np.ndarray) -> np.ndarray:
        return self._apply("pdeb_spatial_batch_adjoint", images)


class NativeCovarianceExposurePlan:
    """Generate a positive nine-atom covariance/shape measure inside ABI v6."""

    def __init__(
        self,
        covariance_axes: np.ndarray,
        shape: tuple[int, int],
        axis_side_weights: np.ndarray | None = None,
    ) -> None:
        library = _load_library()
        if library is None:
            raise RuntimeError("native covariance-exposure ABI is unavailable")
        self._library = library
        self.shape = (int(shape[0]), int(shape[1]))
        self._axes = np.ascontiguousarray(
            covariance_axes, dtype=np.float64)
        if self._axes.shape != (*self.shape, 4):
            raise ValueError("native covariance axes must have shape HxWx4")
        self._side_weights = np.ascontiguousarray(
            (np.full(2, 1.0 / 6.0)
             if axis_side_weights is None else axis_side_weights),
            dtype=np.float64,
        )
        if (
            self._side_weights.shape not in ((2,), (*self.shape, 2))
            or np.any(self._side_weights <= 0.0)
            or np.any(self._side_weights >= 0.5)
        ):
            raise ValueError("native covariance side weights must be in (0,1/2)")
        self._spatial_side_weights = int(self._side_weights.ndim == 3)
        label = library.pdeb_path_operator_backend()
        self.backend = f"{label.decode('ascii')}_covariance_generated"

    def _apply(self, name: str, image: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(image, dtype=np.float64)
        if value.shape[:2] != self.shape or value.ndim not in (2, 3):
            raise ValueError("native covariance operator image shape mismatch")
        channels = 1 if value.ndim == 2 else int(value.shape[2])
        output = np.empty_like(value)
        function = getattr(self._library, name)
        status = function(
            value.ctypes.data_as(_DOUBLE_POINTER),
            output.ctypes.data_as(_DOUBLE_POINTER),
            self.shape[0],
            self.shape[1],
            channels,
            self._axes.ctypes.data_as(_DOUBLE_POINTER),
            self._side_weights.ctypes.data_as(_DOUBLE_POINTER),
            self._spatial_side_weights,
        )
        if status != 0:
            raise RuntimeError(
                f"native covariance operator failed with status {status}")
        return output

    def forward(self, image: np.ndarray) -> np.ndarray:
        return self._apply("pdeb_covariance_forward", image)

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        return self._apply("pdeb_covariance_adjoint", image)


class NativeCovarianceExposureBatchPlan:
    """Run exchangeable generated covariance plans in parallel in ABI v6."""

    def __init__(
        self,
        covariance_axes: np.ndarray,
        shape: tuple[int, int],
        axis_side_weights: np.ndarray,
    ) -> None:
        library = _load_library()
        if (
            library is None
            or getattr(library, "_personal_deblurrer_abi_version", 0) < 6
        ):
            raise RuntimeError("native covariance batch ABI v6 is unavailable")
        self._library = library
        self.shape = (int(shape[0]), int(shape[1]))
        self._axes = np.ascontiguousarray(covariance_axes, dtype=np.float64)
        if self._axes.ndim != 4 or self._axes.shape[1:] != (*self.shape, 4):
            raise ValueError("native covariance batch axes must be NxHxWx4")
        self.plans = int(self._axes.shape[0])
        self._side_weights = np.ascontiguousarray(
            axis_side_weights, dtype=np.float64)
        if self._side_weights.shape == (self.plans, 2):
            self._spatial_side_weights = 0
        elif self._side_weights.shape == (self.plans, *self.shape, 2):
            self._spatial_side_weights = 1
        else:
            raise ValueError(
                "native covariance batch weights must be Nx2 or NxHxWx2")
        if (
            np.any(self._side_weights <= 0.0)
            or np.any(self._side_weights >= 0.5)
        ):
            raise ValueError("native covariance batch weights must be in (0,1/2)")
        label = library.pdeb_path_operator_backend()
        self.backend = f"{label.decode('ascii')}_covariance_generated_batch"

    def _apply(self, name: str, images: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(images, dtype=np.float64)
        if (
            value.shape[0] != self.plans
            or value.shape[1:3] != self.shape
            or value.ndim not in (3, 4)
        ):
            raise ValueError("native covariance batch image shape mismatch")
        channels = 1 if value.ndim == 3 else int(value.shape[3])
        output = np.empty_like(value)
        function = getattr(self._library, name)
        status = function(
            value.ctypes.data_as(_DOUBLE_POINTER),
            output.ctypes.data_as(_DOUBLE_POINTER),
            self.shape[0],
            self.shape[1],
            channels,
            self._axes.ctypes.data_as(_DOUBLE_POINTER),
            self._side_weights.ctypes.data_as(_DOUBLE_POINTER),
            self._spatial_side_weights,
            self.plans,
        )
        if status != 0:
            raise RuntimeError(
                f"native covariance batch failed with status {status}")
        return output

    def forward(self, images: np.ndarray) -> np.ndarray:
        return self._apply("pdeb_covariance_batch_forward", images)

    def adjoint(self, images: np.ndarray) -> np.ndarray:
        return self._apply("pdeb_covariance_batch_adjoint", images)
