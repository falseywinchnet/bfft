# The Physically Realizable FFT
### Positive-cone transport, the mass-potential gauge, and Fourier analysis as a flow law

Companions: `experiments/cone_fft_gauged.py` (the cured mass-substrate machine
and every certificate quoted below), `experiments/cone_dip_delayline.py` (the
field-substrate machine), reference lineage
`bruun512_positive_forward_inverse.py`. The signed algebra and walk geometry
inherited here — the Bruun residue factorization, the normalized cells, the
DIF segment recursion, the DIP diagonal walk and its angle law — are the BFFT
project's kernels and notes (`bruun_dif_dit_kernel_audit.md`,
`dip_phase_packet_design.md`, `dip_twisted_walk.md`).

---

## 1. The premise

Every FFT ever run has been a *simulation* of Fourier analysis: signed,
complex arithmetic executed by a general-purpose machine that fabricates
negation and multiplication out of switched logic. The question asked here is
older and stranger: **what would it take for the Fourier transform to be a
law obeyed by a material system** — a network a signal *flows through*, whose
steady output ports simply *are* the spectrum?

The obstruction is signs. Physical quantities that flow — mass, power,
charge packets, light intensity, probability — are nonnegative. A medium can
split them, route them, merge them, delay them, absorb them. It cannot
subtract them, and subtraction is the soul of the FFT: every butterfly is a
difference, and Parseval hides an enormous ledger of cancellation inside
"destructive interference."

The positive-cone FFT resolves the obstruction by **lifting** rather than
simulating. A signed value `x` becomes a pair of nonnegative sheet values
`(x⁺, x⁻)` with `x = x⁺ − x⁻`. A signed weight becomes *routing*: positive
weights carry each sheet straight; negative weights carry each sheet to the
opposite sheet (a **sheet swap**). Formally, a signed cell `M` lifts to the
nonnegative block matrix

```
L(M) = [ M⁺  M⁻ ]        M = M⁺ − M⁻ ,  M⁺, M⁻ ≥ 0 entrywise
       [ M⁻  M⁺ ]
```

and the lift is a monoid homomorphism — `L(A)·L(B) = L(A·B)` — with the
projection `π(p, n) = p − n` intertwining everything: `π∘L(M) = M∘π`. The
whole signed circuit rides above the nonnegative circuit as its shadow. The
primitive set changes character completely:

| signed arithmetic | cone transport |
|---|---|
| multiply by w > 0 | splitter / attenuator (ratio w) |
| multiply by w < 0 | splitter + **sheet swap** |
| add | junction / coupler |
| subtract | junction after a sheet swap |
| cancellation | **annihilation**: `(p, n) → (p − min, n − min)` at an explicit absorber |
| register | delay line |

Annihilation commutes with π (it removes equal mass from both sheets), so it
is *optional and free of algebraic consequence*: a policy, not an operation.
That single fact is what makes the machine physical — cancellation can happen
wherever the substrate finds convenient (or nowhere), and the answer is
unchanged. **The payoff is not speed. It is realizability**: an FFT whose
run-time physics is passive transport, whose only irreversible act is
explicitly sited dissipation.

The invariants any candidate must preserve (the design law of this program):

1. **Nonnegativity** — every lifted coefficient stays ≥ 0; signs exist only
   as sheet swaps or phase/delay geometry.
2. **Locality** — cells touch a bounded neighborhood; no global gather, no
   dense mixing, no table-sized rescue permutation.
3. **Measured common mode** — sheet mass is the physical energy/dynamic-range
   cost; its growth must be accounted locally, not hidden.
4. **Explicit annihilation** — traceable, local, optional.
5. **Cone-geometric inverse** — the inverse runs the same transport geometry
   backward, not a foreign reconstruction.

## 2. The mass-potential gauge theorem

The reference implementation held its sheets bounded with a global scalar,
`stage_scale = 0.5`, compensated at the end. That is a stabilizer, not an
invariant: the Bruun circuit's binomial wires double their mass per stage
while its rotation wires grow only by `c + s ∈ (1, √2]`, so one constant
over-damps every rotation rail, and the terminal compensation `2⁸` re-inflates
whatever error the damping created. The missing object is a **positive
diagonal gauge** — a per-wire calibration — under which transport itself
conserves mass.

**Theorem (mass-potential gauge).** *Let a feed-forward circuit compute the
FFT as a composition of local signed cells `M`. Define a potential `g_w > 0`
on every wire by the backward recursion*

```
g_w = Σ over cell outputs i fed by wire w of  g_i · |M_iw| ,
```

*with boundary `g ≡ 1` on the output wires — one backward sweep of the
all-ones covector through the absolute-value circuit. Re-express every cell
in gauged units `u_w = g_w · x_w`, i.e. replace each weight by
`M̃_iw = g_i · M_iw / g_w`. Then:*

1. *(Exact mass preservation)* every lifted cell `L(M̃)` is exactly
   column-stochastic: `Σᵢ |M̃_iw| = 1` for every input wire. Total sheet mass
   is invariant under every transport cell, and is decremented only by
   annihilation, giving the exact ledger
   **mass_in = mass_out + Σ annihilated**.
2. *(Exact projection)* with the input presented in gauged units (`g·x`,
   lifted), the projected output *is* the FFT — the gauges telescope, the
   output boundary is 1, and **no compensation factor exists**.
3. *(Uniqueness)* `g` is the unique positive gauge achieving (1) given the
   boundary normalization: it is the Perron left-vector of the absolute
   circuit — physically, each wire's calibrated "mass amplification to
   output."

*Proof.* Column sums of `|L(M̃)|` equal column sums of `|M̃|` (the block
structure duplicates columns across sheets), and the recursion sets
`Σᵢ g_i |M_iw| / g_w = 1` identically. Stochasticity of every cell makes
`1ᵀ(p; n)` invariant under transport; annihilation subtracts `2·min(p,n)`.
For projection, the gauged composition is `D_out (ΠM) D_in⁻¹` with
`D_out = I`; pre-scaling the input by `D_in` cancels the remaining factor.
Uniqueness: a positive diagonal solving `Σᵢ g_i |M_iw| = g_w` on a connected
feed-forward DAG is determined wire-by-wire by backward induction. ∎

The gauge is not arithmetic — it is **calibration geometry**: splitter ratios
and transducer gains set once at build time. In `cone_fft_gauged.py` the
theorem is certified numerically at n = 512:

- column-stochasticity defect over every cell, both directions: **2.2·10⁻¹⁶**;
- per-stage transport mass defect: **≤ 2.6·10⁻¹⁶** (relative);
- projected forward vs `numpy.fft.rfft`: ~10⁻¹⁴; cone roundtrip: ~10⁻¹⁵;
- ledger `mass_in = mass_out + Σ burned`: exact to accumulation rounding;
- with annihilation disabled: final mass / input mass = **1.000000**.

The potential field itself is physically informative: at n = 512 the input
potentials span `[256, 1324]` — a 5.2× calibration range, the exact spread
that `stage_scale` was smearing into one number. Under the crude scalar the
signal decays by 2⁻⁸ toward the noise floor and is re-inflated afterward;
under the gauge, wire mass never grows, and *decreases* monotonically as
annihilation burns interference — the machine gets quieter as it resolves
the spectrum.

## 3. The dissipation ledger: interference as heat

Because gauged transport conserves mass exactly, the trace of annihilated
mass per stage is a physically meaningful account of the FFT's hidden
cancellation — Parseval's interference, itemized:

- Gaussian input, n = 512: **97.4 % of input mass is annihilated** on the way
  to the spectrum, burned in a smooth ~30 %-per-stage schedule.
- The impulse — whose spectrum is flat, with nothing to cancel — burns
  **exactly zero**: the machine is lossless for it.
- The all-ones / alternating inputs (single-bin spectra) burn ~99.8 %:
  maximal concentration requires maximal interference.

This is the deep physical reading of the transform: **an FFT is a device for
converting almost all of a generic signal's transport mass into heat, in
exchange for concentrating what remains onto the right wires.** Sparse
spectra are cheap; dense information is thermally expensive. A physical FFT
therefore has a signal-dependent energy signature — itself measurable, itself
information (Section 7).

## 4. The mass-substrate machine: the cured cone-DIF

The DIF geometry is the natural mass machine. Its circuit is the Bruun
factor tree: a binomial DC spine (pure splitter/combiner pairs) and Bruun
rotation nodes (4-port cells with weights `1, c, s`) — segment-local,
feed-forward, no interior permutation. In cone form:

- **two rails per wire** (the sheets), all coefficients nonnegative;
- each cell = a handful of fixed-ratio splitters and junctions, with sheet
  swaps as rail crossovers;
- the gauge = the build-time splitter/transformer ratios (the 5.2× range
  above — comfortably within passive component tolerance);
- absorbers wherever the substrate wants them, or nowhere;
- the inverse is the same network read backward with its own backward-swept
  gauge (certified stochastic to 2.2·10⁻¹⁶).

Realizations, in increasing exoticism: **two-rail analog electronics**
(differential pairs where the rails are literal conductors; sheet swap = a
crossover; annihilation = a differential clamp); **RLC / microwave networks**
(power dividers and 180° hybrids; the gauge absorbed into impedance ratios);
**mechanical flexure lattices** (force/displacement as the flow; couplers as
lever ratios; the machine computes modal decompositions of whatever shakes
it); **microfluidic or granular transport** (literal mass; annihilation as a
reaction chamber consuming matched flows). The DIT is deliberately excluded:
its gather-heavy, schedule-driven form is programmed switching — already a
computer — and so fails the spirit of the premise.

## 5. The field-substrate machine: the delay-line DIP

Where the substrate carries a **field with a carrier** — voltage waves,
light, acoustics — physics supplies signs natively: a path difference of
half a carrier period multiplies an envelope by −1. The two sheets collapse
onto the two phases of one line, annihilation happens for free by
superposition at any junction, and a subtler prize appears, specific to the
DIP walk:

**The DIP packet is a physical signal.** The Bruun pair `(a, b) = (Re, −Im)`
of a diagonal residue is exactly the I/Q envelope of one narrowband wave
`z = a + jb`, and the Bruun cell's rotation is

```
R + jI  =  e^{jθ} · (oa + j·ob)        θ = π·d/e
```

— a pure phase shift, i.e. **a delay line of length θ·λ/2π**. The complete
DIP cell is `lo = z_e + Wz_o`, `hi = conj(z_e − Wz_o)`: one θ-stub, one sum
junction, one difference junction (a half-period stub into a junction), one
conjugation (a half-period stub on the Q rail in two-rail form). Element
census per cell: ~2 junctions + 4 stubs. **The machine contains no
multiplier and no active element anywhere; every twiddle is geometry.**
`cone_dip_delayline.py` runs the machine on element semantics alone and
matches `numpy.fft.rfft` to 10⁻¹⁴ at n up to 4096.

The DIP earns its place over the DIF here through its angle law. Because
θ depends only on the span's diagonal (the isoclinic property), a serial,
time-multiplexed machine — one physical section per level, each a feedback
delay of `w/2` slots, one coupler, one phase pad, the classic
single-path-delay-feedback pipeline that SAW and CCD devices used — needs
only a **staircase** of θ settings: 3, 6, 12, 24, 48, 96 values per stage at
n = 512, changing once per span, quasi-static. A serial DIT needs an agile
per-slot twiddle stream and a twiddle memory — a modulator, i.e. a computer
again. The DIP's risk is coherence: the θ-stubs assume a stable carrier, so
dispersion, drift, and timing skew are the engineering battle (Section 6).
The payoff if won: an n-point Fourier transform as **log n couplers and some
carefully cut lengths of line**.

The two machines are complementary, not competing: the cone-DIF is the
*incoherent* (intensity/mass) realization, robust and slow; the delay-line
DIP is the *coherent* (field) realization, fast and delicate. Both project
to the same algebra; both inverses run the same geometry backward; and the
gauge theorem applies verbatim to any cone-lifted portion of either.

## 6. Engineering the machine

**Calibration is the gauge.** The theorem converts the stabilization problem
into a manufacturing spec: build splitter ratios `g_i|M_iw|/g_w`, and mass
conservation — hence bounded dynamic range — is a structural property, not a
control loop. Tolerance analysis is linear: a ratio error ε on one cell
perturbs column sums by ε, so mass drifts by at most (1+ε) per stage —
graceful, and trimmable per cell because the invariant is *local*.

**Loss is gauge-absorbable.** Uniform per-stage loss is a scalar gauge and
disappears into calibration. Differential loss is the real enemy (it is an
uncommanded weight change); the two-rail layout keeps the rails of one wire
physically adjacent so loss stays common-mode — the same discipline
differential electronics has used for a century.

**Noise enters at absorbers and junctions.** Annihilation sites are the
machine's thermodynamic ports; deferring annihilation (legal — it commutes
with π) trades dynamic range for noise isolation. The trace machinery
computes exactly this tradeoff, per stage, per policy.

**Dynamic range** is the cancellation ratio: sheet mass over projected mass.
Ungauged it explodes as 2^stages; gauged it is monotone non-increasing.
What remains is signal-dependent (Section 3) and sets the substrate's
required linear range — measured, not guessed.

**Bandwidth and scale.** The spatial machine is O(n log n) passive elements
with O(log n) propagation depth — a transform at the substrate's group
velocity, effectively latency-only. The serial DIP machine is O(log n)
sections with O(n) slots — hardware so small it can be printed at the sensor.
Crossings, the classic integrated-photonics killer, are confined by the DIP's
span locality to the seed and boundary combs (the same transport theorems
that shaped the CPU kernel shape the waveguide lay