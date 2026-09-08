"""Summarize actual complete-transform medians; never select minima as wins."""
from pathlib import Path
import json
import math

here = Path(__file__).resolve().parent
parts = ['# Measured packet-gauge results', '',
         'Times include natural input, arithmetic, and natural-order complex output. '
         'Initialization is excluded; all work and output buffers are reused. '
         'Ratios below are baseline time / candidate time. '
         'The Mini was heavily contended: these are exploratory observations, not a promotion gate.', '']
for name in ('final_m4', 'final_a18', 'selective_m4'):
    path = here / (name + '.jsonl')
    if not path.exists():
        continue
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    table = {}
    for row in rows:
        table.setdefault(row['n'], {})[row['method']] = row
    parts += [f'## {name}', '', '| N | DIP µs | Gauge core µs | Hybrid µs | Unfolded µs | DIF µs | DIT µs |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for n, methods in table.items():
        if n < 256:
            continue
        times = [methods[k]['ns'] / 1000 for k in ('dip', 'gauge_core', 'gauge_hybrid', 'unfolded_packet', 'dif', 'dit')]
        parts.append(f'| {n} | ' + ' | '.join(f'{t:.3f}' for t in times) + ' |')
    parts += ['', f"Maximum reported relative spectral error: {max(r['relative_error'] for r in rows):.3g}; "
              f"round-trip error: {max(r['roundtrip_error'] for r in rows):.3g}.", '',
              'Geometric mean speed ratios over N ≥ 256 (each size has equal weight):', '',
              '| Candidate | vs DIP | vs faster DIF/DIT |', '|---|---:|---:|']
    for k in ('dip_clone','fused8','gauge8','gauge16','gauge32','gauge_core','gauge_hybrid','unfolded_packet','gauge_selective'):
        if k not in next(iter(table.values())):
            continue
        selected = [d for n,d in table.items() if n>=256]
        relative = math.exp(sum(math.log(d['dip']['ns']/d[k]['ns']) for d in selected)/len(selected))
        baseline = math.exp(sum(math.log(min(d['dif']['ns'],d['dit']['ns'])/d[k]['ns']) for d in selected)/len(selected))
        parts.append(f'| {k} | {relative:.3f}× | {baseline:.3f}× |')
    if 'gauge_selective' in next(iter(table.values())):
        parts += ['', '| N | DIP µs | Selective µs | Faster DIF/DIT µs | DIP / selective |', '|---:|---:|---:|---:|---:|']
        for n,d in table.items():
            if n<256: continue
            a,b=d['dip']['ns'],d['gauge_selective']['ns']
            best=min(d['dif']['ns'],d['dit']['ns'])
            parts.append(f'| {n} | {a/1000:.3f} | {b/1000:.3f} | {best/1000:.3f} | {a/b:.3f}× |')
    parts += ['', f'Raw data: [{path.name}]({path.name}).', '']
if (here/'cell_m4.jsonl').exists():
    parts += ['## Isolated core, M4', '', 'These timings include the same input-reset memcpy in both paths. '
              'The gauge coefficients are precomputed in this microbenchmark; the full transform includes per-node dispatch costs.', '',
              '| q (columns per stream) | Existing ns | Gauge ns | Ratio |','|---:|---:|---:|---:|']
    for s in (here/'cell_m4.jsonl').read_text().splitlines():
        r=json.loads(s)
        parts.append(f"| {r['q']} | {r['old_ns']:.3f} | {r['gauge_ns']:.3f} | {r['speedup']:.3f}× |")
(here/'RESULTS.md').write_text('\n'.join(parts)+'\n')
print('\n'.join(parts))
