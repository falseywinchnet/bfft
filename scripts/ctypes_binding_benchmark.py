#!/usr/bin/env python3
"""Measure Python ctypes pointer-conversion overhead for BFFT calls.

This benchmark intentionally avoids NumPy so it can run in restricted
environments where Python wheels are unavailable. It compares the former
typed-pointer calling convention against the current raw-address calling
convention while executing the same native ``bfft_forward`` kernel.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import time
from pathlib import Path


class Complex(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


DoublePointer = ctypes.POINTER(ctypes.c_double)
ComplexPointer = ctypes.POINTER(Complex)
PlanPointer = ctypes.c_void_p


def configure_library(lib: ctypes.CDLL, use_raw_addresses: bool) -> None:
    lib.bfft_plan_create.restype = ctypes.c_int
    lib.bfft_plan_create.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(PlanPointer),
    ]
    lib.bfft_plan_bins.restype = ctypes.c_size_t
    lib.bfft_plan_bins.argtypes = [PlanPointer]
    lib.bfft_plan_work_size.restype = ctypes.c_size_t
    lib.bfft_plan_work_size.argtypes = [PlanPointer]
    lib.bfft_plan_native_scratch_size.restype = ctypes.c_size_t
    lib.bfft_plan_native_scratch_size.argtypes = [PlanPointer]
    lib.bfft_forward.restype = ctypes.c_int

    if use_raw_addresses:
        lib.bfft_forward.argtypes = [
            PlanPointer,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
    else:
        lib.bfft_forward.argtypes = [
            PlanPointer,
            DoublePointer,
            ComplexPointer,
            DoublePointer,
            ComplexPointer,
        ]


def make_plan(lib: ctypes.CDLL, n: int) -> PlanPointer:
    plan = PlanPointer()
    status = lib.bfft_plan_create(n, ctypes.byref(plan))
    if status != 0:
        raise RuntimeError(f"bfft_plan_create failed with status {status}")
    return plan


def median_call_time_us(run, iters: int, rounds: int) -> float:
    for _ in range(20):
        run()

    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            run()
        elapsed = time.perf_counter_ns() - t0
        samples.append(elapsed / iters / 1000.0)

    return statistics.median(samples)


def bench_mode(lib_path: Path, n: int, mode: str, iters: int, rounds: int) -> float:
    use_raw_addresses = mode.endswith("_new")
    public_wrapper_style = mode.startswith("public_")

    lib = ctypes.CDLL(str(lib_path))
    configure_library(lib, use_raw_addresses)

    plan = make_plan(lib, n)
    bins = lib.bfft_plan_bins(plan)
    work_size = lib.bfft_plan_work_size(plan)
    scratch_size = lib.bfft_plan_native_scratch_size(plan)

    input_buffer = (ctypes.c_double * n)()
    output_buffer = (Complex * bins)()
    work_buffer = (ctypes.c_double * work_size)()
    scratch_buffer = (Complex * scratch_size)()

    for i in range(n):
        input_buffer[i] = (i * 17 % 251) / 251.0

    if use_raw_addresses:
        input_arg = ctypes.addressof(input_buffer)
        output_arg = ctypes.addressof(output_buffer)
        work_arg = ctypes.addressof(work_buffer)
        scratch_arg = ctypes.addressof(scratch_buffer)

        def run_public_style():
            return lib.bfft_forward(
                plan,
                ctypes.addressof(input_buffer),
                ctypes.addressof(output_buffer),
                ctypes.addressof(work_buffer),
                ctypes.addressof(scratch_buffer),
            )

        def run_low_level_style():
            return lib.bfft_forward(
                plan,
                input_arg,
                output_arg,
                work_arg,
                scratch_arg,
            )
    else:
        output_arg = ctypes.cast(output_buffer, ComplexPointer)
        work_arg = ctypes.cast(work_buffer, DoublePointer)
        scratch_arg = ctypes.cast(scratch_buffer, ComplexPointer)

        def run_public_style():
            return lib.bfft_forward(
                plan,
                ctypes.cast(input_buffer, DoublePointer),
                ctypes.cast(output_buffer, ComplexPointer),
                ctypes.cast(work_buffer, DoublePointer),
                ctypes.cast(scratch_buffer, ComplexPointer),
            )

        def run_low_level_style():
            return lib.bfft_forward(
                plan,
                ctypes.cast(input_buffer, DoublePointer),
                output_arg,
                work_arg,
                scratch_arg,
            )

    run = run_public_style
    if not public_wrapper_style:
        run = run_low_level_style

    return median_call_time_us(run, iters, rounds)


def default_iters(n: int) -> int:
    if n <= 1024:
        return 10000
    return 4000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lib",
        default="build/libbfft.so",
        help="Path to the BFFT shared library.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=7,
        help="Measurement rounds per mode.",
    )
    parser.add_argument(
        "sizes",
        nargs="*",
        type=int,
        default=[512, 1024, 4096],
        help="FFT sizes to measure.",
    )
    args = parser.parse_args()

    lib_path = Path(args.lib).resolve()
    for n in args.sizes:
        iters = default_iters(n)
        old_public = bench_mode(lib_path, n, "public_old", iters, args.rounds)
        new_public = bench_mode(lib_path, n, "public_new", iters, args.rounds)
        old_low = bench_mode(lib_path, n, "low_old", iters, args.rounds)
        new_low = bench_mode(lib_path, n, "low_new", iters, args.rounds)

        print(
            f"N={n:5d} "
            f"public_old={old_public:8.3f} "
            f"public_new={new_public:8.3f} "
            f"speedup={old_public / new_public:5.2f}x "
            f"low_old={old_low:8.3f} "
            f"low_new={new_low:8.3f} "
            f"speedup={old_low / new_low:5.2f}x us"
        )


if __name__ == "__main__":
    main()
