# Four-child moment atlas: exact manuscript edit notes

These notes describe a future edit.  They do **not** authorize a change to
the composed manuscript.  The fused native implementation evaluates the
two-dimensional zero-detail operator already defined in Section IX; it does
not define another CONV variant and does not require an appendix.

## 1. Canonical composition in Section IX

**Location.** In `Restriction and Matched Multiresolution Geometry`,
subsection `Two-dimensional moment geometry`, immediately after
Eq. `blockinverse` and before the sentence beginning `Thus the four child
cells are replaced without redundancy.`

**Insert.**

```tex
Let \(\Pcal_2z=(p_x,p_y,p_{xy})\) denote the three proposals obtained by
the factor construction below, let \(\Acal_z^{(2)}\) denote their joint
face-current admission, and let \(\Scal_2\) denote the inverse
\eqref{eq:blockinverse}.  The complete zero-detail four-child atlas is
therefore the single map
\begin{equation}
 \Fcal_2(z)=\Scal_2\!\left(z,\Acal_z^{(2)}\Pcal_2z\right).
 \label{eq:fourchildatlas}
\end{equation}
This notation composes the proposal, admission, and synthesis already
specified; it introduces no additional reconstruction rule.
```

**Preamble obligation.** If `\Fcal` is not already declared, add it beside
the existing calligraphic operator declarations:

```tex
\newcommand{\Fcal}{\mathcal F}
```

Do not add a new method name, extension, or theorem at this location.

## 2. State precisely what zero-detail admission controls

**Location.** Immediately after the proof of `Conservative CONV lifting` and
before the paragraph beginning `The transform consequently distinguishes
exact recovery from discarded-detail reconstruction.`

**Insert.**

```tex
Consequently, zero-detail synthesis preserves every parent-cell mass and
does not increase the ordered sign variation of any corresponding factor
line.  This is a topological statement about transported variation.  Child amplitudes are
governed by \eqref{eq:admittedmomentbound} and, across levels, by
\eqref{eq:multilevelstability}; face-current admissibility is not an
assertion that every child lies in the global range of the coarse raster.
```

This sentence is required because the four-child state may have nonzero
internal range excursion without acquiring an unobserved face-current sign
pair.  Do not call that internal excursion `ringing`; the proved ringing
criterion in this paper is ordered-current variation.

## 3. CONV* fused realization

**Location.** In `Can CONV Be Made Efficient Using Convolution?`, subsection
`Polyphase synthesis`, immediately after the paragraph ending `the same
expression yields a fixed polyphase bank` and before `Equivalence audit`.

**Insert as a new unnumbered paragraph, not a subsection.**

```tex
\paragraph*{Fused four-child realization.}
For conservative twofold synthesis, \eqref{eq:fourchildatlas} need not be
materialized as three independent analysis products followed by a separate
array synthesis.  One source traversal forms the horizontal, vertical, and
mixed half-cell proposals; joint admission may overwrite those proposal
channels once their orthant moments have been formed; and
\eqref{eq:blockinverse} writes four disjoint child cells for every parent.
The horizontal factor requires one contiguous line-bank transpose, whereas
the vertical and mixed factors already have the native raster line-bank
layout.  A single reusable workspace therefore evaluates exactly the same
CONV* map without an additional image pass.
```

Do not include the C symbol, buffer offsets, allocation counts, compiler
flags, or SIMD intrinsics in the manuscript.  Those belong to supplementary
source.

## 4. Finite-precision implementation audit

**Location.** `Synthetic Evaluation`, subsection `Native CPU cost`, after
the existing timing table.  Add one compact table or one paragraph only
after the benchmark is rerun on the paper's declared machine and worker
count.

**Table schema.**

```tex
\begin{table}[t]
\centering
\caption{Fused four-child CONV* implementation audit}
\label{tab:fourchildfusion}
\begin{tabular}{lrrr}
\hline
Parent raster & fused (ms) & decomposed (ms) & ratio \\
\hline
% Insert paired medians from the archival benchmark artifact.
\hline
\end{tabular}
\end{table}
```

The accompanying sentence should report:

```tex
The fused and decomposed binary32 realizations were bitwise identical on
every audited scalar and three-channel field.  The maximum recovered
parent-mean residual was [ARCHIVAL VALUE].  The maximum measured horizontal
and vertical factorwise sign surpluses were [ARCHIVAL H] and [ARCHIVAL V],
respectively.  Timing ratios compare two realizations of the same operator,
not two interpolation methods.
```

The current local screening value for the maximum parent-mean residual is
`1.1920928955078125e-7`.  It must be replaced by the value from the final
archival run rather than copied blindly.

The native binary32 implementation must use one arithmetic path for every
line-bank lane, including the scalar tail.  Coarse faces that vanish in exact
arithmetic are classified by the forward-error test

```tex
 |a+b|\leq\gamma_8(|a|+|b|),\qquad
 \gamma_8=\frac{8\epsilon}{1-8\epsilon},
```

not by the floating-point predicate `a+b==0`.  This is an implementation
obligation, not an alteration of the exact admission map.  The manuscript
should state it in its finite-precision paragraph only if binary32 recursive
conformance data are reported.

## 5. High-frequency evidence placement

Do not put Fourier-mode gains in Sections IX or XII.  After the enlarged
analytic census is frozen, add them only to `Synthetic Evaluation`, following
`Technical synthetic fields`, under a short subheading such as
`Cell-average detail retention`.

The evidence must separately report:

1. internal atlas excursion beyond the source-cell sample range;
2. final excursion beyond the source-cell sample range;
3. final excursion beyond the exact target-cell range;
4. exact-target MSE and signed mode gain;
5. the maximum horizontal and vertical factorwise sign surplus;
6. results below, near, and above the target Nyquist boundary;
7. phase, orientation, grid-size, and affine-geometry aggregates; and
8. every worst case, including cases in which atlas refinement loses.

The exact target must be obtained analytically from the declared continuous
synthetic.  No competing interpolator may serve as truth.  Internal atlas
excursion and final output excursion must never be merged into one statistic.

## 6. Material that must remain outside the paper

- The Python/C binding name and line-bank packing code.
- Development timings that are not from the declared archival environment.
- The three-mode pilot as though it were a population result.
- A claim that positive footprint integration always removes internal atlas
  excursion.  The present observations establish this only for the tested
  population.
- A claim of real-arithmetic identity from float32 conformance.  The exact
  implementation statement is bitwise equality to the decomposed float32
  realization; the mathematical map is identified independently by
  Eq. `fourchildatlas`.

## 7. Scope determined by the expanded census

The four-child atlas is the conservative zero-detail scale state.  It is not
to replace direct CONV synthesis for arbitrary enlargement.  Positive target
footprints strongly suppress internal child excursion during reduction, but
enlargement footprints expose that excursion increasingly as the output grid
approaches the child atlas resolution.  Consequently:

- retain direct CONV as the interpolation/enlargement operator;
- use the moment atlas for conservative restriction, lifting, and
  detail-retaining reduction;
- compare one and two atlas levels explicitly in the evidence insert;
- never infer final range containment from parent-mass conservation; and
- report reduction, identity, and enlargement as separate regimes.

The identity-scale result is exact at the cell-average level: restriction of
either atlas back onto the unchanged source cells returns the parent means.
That statement follows from the block inverse and is not empirical.

## 8. Exact theorem versus recursive binary32 audit

The factorwise sign-variation theorem is an exact-arithmetic statement.  A
recursive binary32 realization can retain a near-zero first-level face which
is subsequently magnified by the second admission.  The expanded census
therefore records both the exact theorem and the measured binary32 surplus;
it must not silently replace one by the other.  The archival implementation
paragraph may claim zero measured surplus only if the final implementation
either carries sufficient precision/state to certify the zero class or its
entire published census actually attains zero.  Otherwise report the measured
maximum and identify it as a finite-precision conformance residual.
