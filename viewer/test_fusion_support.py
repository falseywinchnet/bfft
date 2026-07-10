#!/usr/bin/env python3
"""Geometry regression for cross-frame short-support routing."""

from __future__ import annotations

import numpy as np

import dip_stream


class ZeroSource:
    num_samples = 1_000_000
    is_complex = True

    @staticmethod
    def read(start, count):
        del start
        return np.zeros(int(count), np.complex64)


def main():
    src = ZeroSource()
    configurations = [
        (512, 64), (512, 128), (512, 256),
        (1024, 64), (1024, 128), (1024, 256), (1024, 512),
        (2048, 512), (4096, 512),
    ]
    for nb, ns in configurations:
        stream = dip_stream.FusionStream(src, nb=nb, ns=ns, workers=1)
        try:
            owner = np.arange(stream.owner_frames)[:, None]
            global_desired = owner * stream.owned + stream.desired_delta
            global_routed = ((owner + stream.parent_shift) * stream.owned +
                             stream.local_delta)
            assert np.array_equal(global_desired, global_routed)
            assert stream.local_delta.min() >= 0
            assert stream.local_delta.max() <= nb // 32 - ns // 32
            assert stream.stride == stream.rows_per_tile * stream.HS
            if (nb, ns) == (1024, 512):
                assert stream.desired_delta.tolist() == [14, 15, 16, 17]
                assert stream.parent_shift.tolist() == [0, 0, 0, 1]
                assert stream.local_delta.tolist() == [14, 15, 16, 13]
            print(f"PASS NB={nb} NS={ns}: shifts "
                  f"{stream.parent_shift.tolist()}")
        finally:
            stream.close()


if __name__ == "__main__":
    main()
