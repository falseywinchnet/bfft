# Bin-driven CONV* FIR design

`convstar_filter_design.py` treats filter design as projection of requested
frequency bins onto a realizable real-FIR space. It supports lengths 16--512,
arbitrary complex targets, nonuniform frequencies, bin weights, exact complex
bin constraints, and even or odd linear phase.

For response matrix (B), desired bins (d), and nonnegative diagonal bin
metric (Q), the weighted design is

\[
 h^\star=\arg\min_{h\in\mathcal H}\|Bh-d\|_Q^2.
\]

The implementation forms

\[
 W=\operatorname{Re}(B^*QB),\qquad
 g=\operatorname{Re}(B^*Qd),
\]

reduces exact constraints to an affine nullspace, and solves the resulting SPD
projection. The Toeplitz identity

\[
 W_{nm}=\sum_k q_k\cos(2\pi f_k(n-m))
\]

avoids materializing the complete complex design matrix. Iterative refinement
of the Cholesky solution supplies the reported stationarity certificate.

`design_minimax_from_bins` separately solves the discrete weighted
\(L_\infty\) problem for real linear-phase amplitude targets with a HiGHS
linear program. Its result is globally optimal on the supplied bin set. A
continuous-frequency equiripple claim additionally requires an inter-bin peak
certificate or an extremum-exchange loop.

## Measured result

The final M4 Mini audit is stored in `convstar_filter_design_m4.json`. For
weighted \(L_2\) designs with 1,025--4,097 requested bins, wall time rises from
0.56 ms at 16 taps to 57.9 ms at 512 taps. The nonsymmetric arbitrary-complex
512-tap stress case also takes 57.9 ms and has relative KKT stationarity
residual \(1.09\times10^{-15}\).

For a classical 63-tap two-band low-pass, the discrete minimax LP and a dense
Remez--McClellan solve converge to filters differing by at most
\(7.01\times10^{-7}\) per tap. Their dense-grid peak errors are respectively
\(5.28565\times10^{-4}\) and \(5.28658\times10^{-4}\). Specialized Remez is
much faster on this exact classical problem: 0.36 ms versus 71.5 ms for the
generic LP. Thus the demonstrated advantage is declarative arbitrary complex
bin fitting, weighting, and exact constraints—not replacement of Remez's
special-case runtime.

## Minimal use

```python
import numpy as np

from experiments.convstar_filter_design import design_from_bins

frequencies = np.linspace(0.0, 0.5, 4097)
desired = np.exp(-20.0 * frequencies**2) * np.exp(-12j * np.pi * frequencies)
design = design_from_bins(
    desired,
    length=128,
    frequencies=frequencies,
    exact_frequencies=[0.0],
    exact_values=[1.0],
)

taps = design.taps
print(design.certificate)
```

Run the focused tests with:

```sh
.venv-jpeg/bin/python -m unittest experiments.test_convstar_filter_design
```

Run the length and timing audit with:

```sh
.venv-jpeg/bin/python -m experiments.benchmark_convstar_filter_design \
  --out /tmp/convstar_filter_design.json
```
