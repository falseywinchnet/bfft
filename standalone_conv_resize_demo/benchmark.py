"""Small reproducible throughput screen for the standalone backends."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from backend import (
    backend_description, conv_resize, easu_resize, lanczos3_resize,
    linear_resize, polyphase_fir_resize,
)


METHODS = {
    "convstar": conv_resize,
    "polyphase_fir8": lambda x,s: polyphase_fir_resize(x,s,radius=8),
    "lanczos3": lanczos3_resize,
    "bilinear": linear_resize,
    "easu": easu_resize,
}


def median_time(operation, source, shape, repeats: int) -> float:
    operation(source, shape)
    values = []
    for _ in range(repeats):
        started = time.perf_counter(); operation(source, shape)
        values.append(time.perf_counter()-started)
    return float(np.median(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="128,256,512")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.threads is not None:
        os.environ["CONV_NATIVE_THREADS"] = str(max(1,args.threads))
    rng = np.random.default_rng(20260827)
    records = []
    for side in (int(x) for x in args.sizes.split(",")):
        source = rng.random((side,side,3), dtype=np.float32)
        target = (2*side,2*side)
        for name, operation in METHODS.items():
            elapsed = median_time(operation,source,target,args.repeats)
            records.append({
                "method": name, "source_side": side,
                "target_side": 2*side, "median_ms": 1000.0*elapsed,
                "target_megapixels_per_second": (4*side*side)/elapsed/1.0e6,
            })
    print(json.dumps({"backend":backend_description(),"records":records},indent=2))


if __name__ == "__main__":
    main()
