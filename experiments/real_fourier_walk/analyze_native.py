import collections
import json
import pathlib
import statistics

root=pathlib.Path(__file__).resolve().parent
folder=root/'native_m4'
rows=json.loads((folder/'results.json').read_text())
groups=collections.defaultdict(list)
for row in rows:groups[row['N'],row['policy']].append(row)
summary=[]
for (n,policy),group in sorted(groups.items()):
    summary.append({'N':n,'policy':policy,'runs':len(group),
                    **{field:statistics.median(row[field] for row in group)
                       for field in ('dif_ns','walk_ns','walk_over_dif')},
                    'minimum_run_ratio':min(row['walk_over_dif'] for row in group),
                    'maximum_run_ratio':max(row['walk_over_dif'] for row in group),
                    'relative_l2_error':max(row['relative_l2_error'] for row in group),
                    'executed_crossings':group[0]['executed_crossings']})
(folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lookup={(r['N'],r['policy']):r for r in summary}
lines=['# Normalized quartic NEON measurements','',
       'Three process runs per size, seven alternating-order chunks per policy per run. '
       'Table entries are medians across runs; times are microseconds. Lower walk/DIF is better. '
       'Both complete real transforms include standard-order output.','',
       '| N | DIF us (paired with separated) | Separated us | Walk/DIF | Sibling/DIF | Random/DIF | Unfused/DIF |',
       '|---:|---:|---:|---:|---:|---:|---:|']
for n in sorted({r['N'] for r in summary}):
    r=lookup[n,'separated']
    lines.append(f"| {n} | {r['dif_ns']/1000:.3f} | {r['walk_ns']/1000:.3f} | {r['walk_over_dif']:.3f} | "
                 f"{lookup[n,'sibling']['walk_over_dif']:.3f} | {lookup[n,'random']['walk_over_dif']:.3f} | "
                 f"{lookup[n,'separated_unfused']['walk_over_dif']:.3f} |")
lines.extend(['','No tested separated-plan size beats production DIF in the median.' if all(r['walk_over_dif']>=1 for r in summary if r['policy']=='separated')
              else 'Inspect the per-size ratios and run ranges before interpreting a crossover.',
              '', 'See `NATIVE_FACTORIZATION.md` for the algebra, scope, and benchmark contract. '
              'Raw results, complete benchmark output, host/compiler metadata, assembly, and checks are in `native_m4/`.',''])
(root/'NATIVE_RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines))
