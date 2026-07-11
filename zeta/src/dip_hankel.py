"""DIP-backed linear convolution and uniform-grid Hankel products."""
from __future__ import annotations

import ctypes
import json
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libzeta_dip.dylib"


def build_binding(force: bool = False) -> Path:
    LIB.parent.mkdir(parents=True, exist_ok=True)
    source = ROOT / "src" / "dip_binding.cpp"
    header = ROOT.parent / "src" / "detail" / "bruun_dip_kernel.hpp"
    if force or not LIB.exists() or LIB.stat().st_mtime < max(source.stat().st_mtime, header.stat().st_mtime):
        subprocess.run([
            "clang++", "-std=c++17", "-O3", "-fPIC", "-dynamiclib",
            str(source), "-o", str(LIB)
        ], check=True, cwd=ROOT)
    return LIB


def _library():
    lib = ctypes.CDLL(str(build_binding()))
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.zeta_dip_convolve.argtypes = [ptr, ctypes.c_int, ptr, ctypes.c_int, ptr]
    lib.zeta_dip_convolve.restype = ctypes.c_int
    lib.zeta_dip_backend.restype = ctypes.c_char_p
    return lib


def dip_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    out = np.empty(a.size + b.size - 1, dtype=np.float64)
    ptr = ctypes.POINTER(ctypes.c_double)
    rc = _library().zeta_dip_convolve(
        a.ctypes.data_as(ptr), a.size, b.ctypes.data_as(ptr), b.size,
        out.ctypes.data_as(ptr))
    if rc < 0:
        raise RuntimeError(f"DIP convolution failed ({rc})")
    return out


def hankel_dense(kernel_samples: np.ndarray, x: np.ndarray, du: float) -> np.ndarray:
    """H[i,j] = du*kernel_samples[i+j], samples have length 2N-1."""
    x = np.asarray(x, dtype=float)
    idx = np.add.outer(np.arange(x.size), np.arange(x.size))
    return du * np.asarray(kernel_samples)[idx] @ x


def hankel_dip(kernel_samples: np.ndarray, x: np.ndarray, du: float) -> np.ndarray:
    """Reflection + zero-padded DIP convolution + exact valid crop."""
    x = np.asarray(x, dtype=float)
    n = x.size
    k = np.asarray(kernel_samples, dtype=float)
    if k.size != 2 * n - 1:
        raise ValueError("kernel_samples must have length 2N-1")
    # conv(k, reverse(x))[n-1+i] = sum_j k[i+j] x[j].
    return du * dip_convolve(k, x[::-1])[n - 1:2 * n - 1]


def compare_dense_dip(kernel_samples, du, sizes=(16, 32, 64, 128), trials=4, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    full_n = (len(kernel_samples) + 1) // 2
    for n in sizes:
        if n != full_n:
            grid = np.linspace(0, 2 * full_n - 2, 2 * n - 1)
            k = np.interp(grid, np.arange(2 * full_n - 1), kernel_samples)
        else:
            k = np.asarray(kernel_samples)
        for trial in range(trials):
            x = rng.standard_normal(n)
            yd = hankel_dense(k, x, du * full_n / n)
            yf = hankel_dip(k, x, du * full_n / n)
            rel = np.linalg.norm(yf - yd) / max(np.linalg.norm(yd), np.finfo(float).tiny)
            rows.append({"N": n, "trial": trial, "relative_error": rel,
                         "passed": bool(rel < 1e-10)})
    return rows


def write_operator_observable(path: Path, **record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
