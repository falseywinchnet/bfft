# Matrix Transport: Frozen Unseen Battery

## Result first

The Ripple-developed optimizer was frozen, including its declared learning rate
of 0.003, and then run against AdamW on 23 other problems. Each arm used the
same self-context M-layer, width 24, initialization, minibatch order, exact
self-context backward pass, 500 updates, and three paired seeds. There were
0 numerical failures in 138 runs.

Matrix Transport is a faster learner on this battery, but it is not yet a
uniformly better generalizer:

| Measure | Frozen unseen result |
|---|---:|
| Mean validation acquisition-AUC delta | **+0.0040** |
| Task-clustered bootstrap 95% interval | [-0.0007, +0.0092] |
| Paired-seed AUC wins | **42 / 69** |
| Task-mean AUC wins | **13 / 23** |
| Mean held-out score delta | +0.0004 |
| Task-clustered bootstrap 95% interval | [-0.0171, +0.0223] |
| Paired held-out wins / ties | 26 / 6 of 69 |
| Task-mean held-out wins / ties | 6 / 1 of 23 |
| Mean tail-score delta (12 applicable tasks) | -0.0043 |
| Task-clustered tail 95% interval | [-0.0342, +0.0318] |
| Mean M4 runtime, AdamW → Matrix | 2.61s → 3.53s (1.35×) |

The acquisition result is distributed rather than being only a single showcase:
42 of 69 paired runs and 13 of 23 task means favor Matrix Transport. The
task-clustered interval still crosses zero, so this is encouraging evidence,
not a settled population effect. The final held-out result is less favorable:
its mean is almost exactly tied, its median task delta is
-0.0011, and extrapolation tails
lean toward AdamW. This rejects the stronger claim that the current geometry
is already a universal replacement for AdamW.

The largest new win is the high-rank 16-D spiral: mean held-out score rises
from 0.7040 to 0.8898 while AUC rises by 0.0282. The clearest regressions are
radial stripes, chirp continuation, and drifted-chirp continuation. Those are
not optimizer failures during fitting; they are cases where faster
interpolation does not select the same extrapolating function.

## Cached matrix roots

The Kronecker factors are still updated every step. Their inverse fourth roots
are recomputed only when the corrected covariance moves by at least 5% in
relative Frobenius norm, or when a hard ten-step age limit is reached. The
cache never consults the loss, validation score, task identity, or gradient
labels. It is therefore a numerical realization rule for the same metric, not
a task-conditioned optimizer.

Across this unseen battery, evaluation snapshots report a mean root-refresh
fraction of 0.252 and mean root age of
3.23 updates. On the five-seed Ripple control,
the cache reduced end-to-end M4 time from 6.65s to 6.20s and slightly increased
mean AUC from 0.78443 to 0.78532. Matrix Transport remains
1.35× AdamW wall time on this small CPU battery;
the residual is the matrix algebra and transport state, not repeated roots
alone.

## Every problem

The score is taken at the best validation checkpoint. AUC is the area under
the validation-learning curve, so it measures acquisition speed throughout
the fixed 500-update budget.

| Problem | AdamW AUC | Matrix AUC | Δ AUC | AdamW held-out | Matrix held-out | Δ held-out |
|---|---:|---:|---:|---:|---:|---:|
| checkerboard | 0.8258 | 0.8424 | +0.0166 | 0.4457 | 0.4221 | -0.0236 |
| chirp_1d | 0.8845 | 0.8969 | +0.0124 | 0.3797 | 0.3240 | -0.0558 |
| complex_spiral_3d | 0.8582 | 0.8866 | +0.0284 | 0.0374 | 0.0505 | +0.0131 |
| fourier_mix_1d | 0.9234 | 0.9284 | +0.0050 | 0.4274 | 0.4791 | +0.0517 |
| hyperchecker | 0.4988 | 0.5004 | +0.0015 | 0.5162 | 0.5148 | -0.0014 |
| hypercube_checker | 0.5017 | 0.5000 | -0.0017 | 0.5090 | 0.5068 | -0.0022 |
| localized_steps_1d | 0.9599 | 0.9668 | +0.0070 | 0.9546 | 0.9491 | -0.0055 |
| lorenz_lobes | 0.9854 | 0.9824 | -0.0030 | 0.9997 | 0.9989 | -0.0008 |
| multiscale_1d | 0.9028 | 0.9077 | +0.0049 | 0.4054 | 0.4043 | -0.0011 |
| nd_spiral_high_rank | 0.9267 | 0.9549 | +0.0282 | 0.7040 | 0.8898 | +0.1858 |
| nd_spiral_low_rank | 0.8985 | 0.9183 | +0.0198 | 0.3842 | 0.3913 | +0.0072 |
| periodic_nd | 0.5621 | 0.5609 | -0.0012 | 0.5750 | 0.5746 | -0.0003 |
| periodic_wells | 0.8744 | 0.8758 | +0.0014 | 0.9899 | 0.9893 | -0.0007 |
| pinwheel | 0.9647 | 0.9707 | +0.0059 | 1.0000 | 1.0000 | +0.0000 |
| poly_drifted_chirp_1d | 0.8284 | 0.8299 | +0.0015 | 0.2138 | 0.1658 | -0.0480 |
| radial_stripes | 0.5283 | 0.5135 | -0.0148 | 0.6910 | 0.5976 | -0.0935 |
| ring_sdf | 0.9824 | 0.9703 | -0.0121 | 0.9999 | 1.0000 | +0.0001 |
| sinusoid_bounds | 0.9546 | 0.9455 | -0.0091 | 0.9899 | 0.9837 | -0.0062 |
| sparse_sine_1d | 0.6379 | 0.6279 | -0.0100 | 0.6164 | 0.5831 | -0.0333 |
| spiral | 0.9054 | 0.9284 | +0.0231 | 0.3443 | 0.3735 | +0.0292 |
| swiss_cheese | 0.9258 | 0.9198 | -0.0060 | 0.9833 | 0.9798 | -0.0035 |
| two_moons | 0.9761 | 0.9760 | -0.0001 | 0.9997 | 0.9995 | -0.0003 |
| xor_quads | 0.9741 | 0.9693 | -0.0049 | 0.9964 | 0.9943 | -0.0020 |

## Interpretation

The result supports the narrow mechanism proposed on Ripple: a full matrix
second moment plus confidence-earned transport can improve how quickly a
self-context model acquires structure without using an elementwise adaptive
rate. It does not support the stronger proposition that learning faster on
observed support necessarily improves continuation. The tail regressions are
especially useful: they locate the next mathematical problem in the
optimizer's scalar coherence gain or checkpoint path, not in the cached
eigendecomposition.

No task-specific learning rates, geometry branches, loss probes, or post-hoc
optimizer changes were used. Ripple is excluded from all counts above.
