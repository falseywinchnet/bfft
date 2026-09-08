"""Reproducible correctness/storage/representation-cost screen, not a DIF race."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np

from twos import (ShiftPlan, compact_forward, compact_inverse, compact_bin,
                  polynomial_bin, diagnostic_complex, pack_orbits,
                  unpack_orbits, phase_shift_orbits, orbit_lane_bits,
                  pack_samples, fold_packed)


def timed(function, repeats=5):
    result = function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function()
        samples.append(time.perf_counter_ns() - start)
    return result, statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    rng = random.Random(2051)
    rows = []
    for n in (8, 16, 32, 64, 128, 256, 512, 1024):
        bound = 32768
        x = [rng.randint(-bound, bound) for _ in range(n)]
        p = ShiftPlan(n, bound)
        expanded, full_ns = timed(lambda: p.forward(x))
        packets, compact_ns = timed(lambda: compact_forward(x))
        orbit_bits = orbit_lane_bits(n, bound)
        input_word, input_pack_ns = timed(lambda: pack_samples(x, p))
        folded, folded_ns = timed(lambda: fold_packed(input_word, p))
        assert folded == pack_orbits(packets, p.lane_bits)
        packed, pack_ns = timed(lambda: pack_orbits(packets, orbit_bits))
        shifted, phase_ns = timed(lambda: phase_shift_orbits(packed, orbit_bits, 7))
        assert compact_inverse(packets) == x
        assert compact_inverse(unpack_orbits(shifted, orbit_bits)) == x[-7:] + x[:-7]
        assert p.fft(expanded, inverse=True) == [p.reduce(v) for v in x]
        # Every symbolic bin, not just sampled bins.
        decoded, decode_ns = timed(lambda: [p.coefficients(word) for word in expanded], 3)
        for k in range(n):
            assert decoded[k] == polynomial_bin(x, k) == compact_bin(packets, n, k)
        if n <= 256:
            numeric, projection_ns = timed(lambda: [diagnostic_complex(c, n) for c in decoded], 3)
            reference = np.fft.fft(x)
            error = float(np.max(np.abs(np.array(numeric) - reference)))
            relative = error / max(1, float(np.max(np.abs(reference))))
            assert relative < 1e-12
        else:
            projection_ns = error = relative = None
        rows.append(dict(N=n, bound=bound, lane_bits=p.lane_bits,
                         bits_per_expanded_residue=p.width + 1,
                         expanded_storage_bits=n * (p.width + 1),
                         compact_coefficient_count=sum(map(len, packets.values())),
                         compact_modular_payload_bits=sum(v.bit_length() for v in packed.values()),
                         compact_guarded_capacity_bits=sum((m >> 1) * b + 1 for m, b in orbit_bits.items())
                         + (n * bound).bit_length() + 1,
                         compact_orbit_lane_bits=orbit_bits,
                         packed_input_encode_ns=input_pack_ns,
                         packed_fold_ns=folded_ns,
                         packed_fold_exact=True,
                         expanded_fft_ns=full_ns, compact_encode_ns=compact_ns,
                         compact_pack_ns=pack_ns, compact_delay_ns=phase_ns,
                         expanded_decode_ns=decode_ns, numeric_projection_ns=projection_ns,
                         projection_max_abs_error=error, projection_max_relative_error=relative,
                         exact_bins_checked=n, exact_inverse=True, exact_compact_delay=True))
    x = list(range(1, 9))
    p = ShiftPlan(8, 8)
    example = dict(input=x, lane_bits=p.lane_bits, modulus=str(p.modulus),
                   expanded_words=[str(v) for v in p.forward(x)],
                   bin_one_coefficients=p.coefficients(p.forward(x)[1]),
                   compact_packets=compact_forward(x))
    here = Path(__file__).parent
    result = dict(platform=platform.platform(), processor=platform.processor(),
                  python=platform.python_version(), numpy=np.__version__,
                  contract='Exact algebraic Fourier encodings; ordinary complex projection is paid separately.',
                  timings='Python reference costs; no native BFFT speed comparison or speedup claim.',
                  sources={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in here.glob('*.py')},
                  rows=rows, example=example)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(out=args.out, exact_bins=sum(r['exact_bins_checked'] for r in rows),
                          worst_numeric_relative_error=max(r['projection_max_relative_error'] or 0 for r in rows))))


if __name__ == '__main__':
    main()
