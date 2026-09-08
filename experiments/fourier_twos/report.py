"""Render saved representation measurements without rerunning the experiment."""
import hashlib
import json
from pathlib import Path


def main():
    here = Path(__file__).parent
    result = json.loads((here / 'm4_results/fourier_twos_results.json').read_text())
    for name, expected in result['sources'].items():
        actual = hashlib.sha256((here / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('measurement source changed: ' + name)
    error = max(row['projection_max_relative_error'] or 0 for row in result['rows'])
    text = [
        '# Measured Fourier-by-twos representation costs', '',
        '2026-09-05, Apple M4, Python ' + result['python'] + '. These are reference-Python timings, '
        'not a native BFFT speed comparison. Construction/plan metadata and Python object overhead '
        'are excluded from packed-bit capacity figures. All three executors return exact algebraic '
        'encodings; ordinary numeric projection is separate.', '',
        'All 2,040 bins across the eight sizes below passed exact integer coefficient comparison '
        'to a direct polynomial DFT. All inverse, packed-fold and compact phase-delay checks passed. '
        'Numeric projections through N=256 had worst maximum absolute error / maximum reference '
        'magnitude ' + str(error) + '. Eleven focused unit tests passed.', '',
        '| N | Expanded bits/bin | Expanded capacity (KiB) | Compact capacity (KiB) | Expanded FFT (ms) | Compact encode (us) | Packed phase delay (us) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for v in result['rows']:
        text.append('| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(
            v['N'], v['bits_per_expanded_residue'], v['expanded_storage_bits'] / 8192,
            v['compact_guarded_capacity_bits'] / 8192, v['expanded_fft_ns'] / 1e6,
            v['compact_encode_ns'] / 1e3, v['compact_delay_ns'] / 1e3))
    text += ['',
        'Input bound is |x[j]| <= 32,768. Times are medians of five warmed calls. '
        'The compact delay starts with packed packets and excludes encoding, packing and inverse '
        'reconstruction. It acts on all frequency orbits. The shifts move long integers; this is '
        'not constant-time delay or a faster alternative to changing a time-domain read index.', '',
        'The direct packed-fold executor retains uniform guard widths. Its payload is larger '
        'than the per-orbit-width compact capacity above. Initial packing is measured separately:', '',
        '| N | Input packing (us) | Packed fold (us) |',
        '|---:|---:|---:|']
    for v in result['rows']:
        text.append('| {} | {:.3f} | {:.3f} |'.format(v['N'], v['packed_input_encode_ns'] / 1e3,
                                                    v['packed_fold_ns'] / 1e3))
    text += ['',
        'Python scheduling, allocation and host load make these short timings noisy. They establish '
        'a cost ledger, not a promotion gate. Horner input packing repeatedly grows an integer and '
        'can dominate the fold itself. No timing omits numeric projection while claiming an '
        'end-to-end numeric FFT speedup.', '',
        'At N=1024 the expanded representation is 1.69 MiB, the variable-width compact representation '
        'has about 2.38 KiB of guarded payload capacity, and the compact state has N coefficient '
        'lanes. Resolving independent numeric complex bins remains required. No production kernel '
        'was changed.', '',
        'Raw data and source hashes: [fourier_twos_results.json](m4_results/fourier_twos_results.json). '
        'Test log: [fourier_twos_tests.txt](m4_results/fourier_twos_tests.txt).']
    (here / 'RESULTS.md').write_text('\n'.join(text) + '\n')


if __name__ == '__main__':
    main()
