"""Regenerate the measured general-geometry paper table from retained data."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
rows=json.loads((root/'experiments/krylov_bregman/results/general_geometry_v2.json').read_text())['rows']
lines=[r'\subsection{Controlled cross-geometry discovery experiment}',
 r'We use 24-dimensional Euclidean, quartic, entropy, and hypentropy mirrors,',
 r'with a fixed dense invertible mixing of primal coordinates, three seeds,',
 r'and three distinct rates. The scalar mirror gradients are $q$, $q+q^3$,',
 r'$\log q$, and $\operatorname{arsinh}q$, respectively. Eight ordinary mirror',
 r'transitions supply nine snapshots; no future reference or rates enter fitting.',
 r'The known mirror factorization supplies the $s$ coordinates. The supplied',
 r'dictionary and synthesized rational chart are evaluated at horizons 16, 64,',
 r'and 128 against independent ordinary execution. Reported errors are maximum',
 r'relative primal errors across all 12 geometry/seed combinations.',
 r'\begin{center}\small',r'\begin{tabular}{lrrrr}\toprule',
 r'Representation & Snapshot rank & $m=16$ & $m=64$ & $m=128$\\\midrule']
for name,label,rank in [('dual','Dual coordinate','8'),('log','Log dual coordinate','8'),('reciprocal','Reciprocal dual coordinate','4'),('synthesized_crossratio','Synthesized cross-ratio','---')]:
 values=[max(r['relative_primal_error'] for r in rows if r['observable']==name and r['horizon']==h) for h in (16,64,128)]
 lines.append(label+' & '+rank+' & '+' & '.join(f'{v:.2e}' for v in values)+r'\\')
lines += [r'\bottomrule\end{tabular}\end{center}',
 r'The SVD rank threshold is $10^{-10}$ relative to the largest singular value.',
 r'Cross-ratio synthesis fits a separate rational relation per transformed',
 r'coordinate and does not use a low-rank snapshot compression. Its rates are',
 r'learned, not supplied. The table measures a favorable constructed family;',
 r'it is not an acceptance-coverage study over arbitrary objectives. Numerical',
 r'errors reflect identification, decoding, and floating-point evolution, whereas',
 r'the exact closure statements follow from the algebra above. This experiment',
 r'records acquisition, fitting, transport, and ordinary execution time, but does',
 r'not perform a repeated performance gate; no wall-clock acceleration claim',
 r'is drawn from it. Ten new tests check mirror inversion, actual mirror updates,',
 r'convex Hessians, compatibility and its failure, recurrence recovery, chart',
 r'synthesis and rejection, directional defect cancellation, and polynomial',
 r'degree growth. The complete research suite passes 51 tests on the M4 Mini.']
(root/'paper/krylov_bregman/general_geometry_results.tex').write_text('\n'.join(lines)+'\n')
