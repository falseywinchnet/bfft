# The physically realizable FFT: a positive-cone transport theory

Companion code (all claims certified to machine precision):
`experiments/cone_fft_gauged.py` (the cured cone-DIF: gauged mass-preserving
transport + dissipation ledger + projection proof),
`experiments/cone_dip_delayline.py` (the delay-line cone-DIP: a Fourier
machine with zero multipliers and zero active elements),
`experiments/dip_twisted_walk.py` and the DIF/DIT/DIP kernels for the
digital lineage this grew out of.

Research candidate. This is a theory-and-reference deliverable, not a compute
kernel. The goal is *realizability*, not speed.

---

## 0. The thesis

An FFT is normally *simulated arithmetic*: multiply, subtract, store. A
positive-cone FFT asks whether the same linear map can instead be a *physical
flow law* — routed nonnegative transport plus explicit local annihilation —
whose projection onto a difference coordinate is the Fourier transform. If it
can, the primitive set stops being {×, −, store} and becomes {split, couple,
route, sheet-swap, delay, absorb}: the vocabulary of passive circuits, optical
power paths, mechanical flexures, and delay lines. The transform becomes
something you *build*, not something you *run*.

Two substrates realize it, and they are duals:

- **Mass substrate (cone-DIF).** Two nonnegative rails carry every value;
  x = x⁺ − x⁻. Signs live in *which rail*. Cancellation is an explicit
  absorber that removes min(x⁺, x⁻) from both rails. §§1–4.
- **Field substrate (cone-DIP).** One narrowband carrier per value; the
  envelope is complex, negation is a half-period *delay*, rotation is a
  *delay*, cancellation is *interference*. Multipliers vanish entirely. §5.

---

## 1. The lift and what breaks

Lift a real vector to the nonnegative cone by two sheets, projected by
π(p, n) = p − n. A signed linear cell M lifts to

    L(M) = [[M⁺, M⁻], [M⁻, M⁺]],   M⁺ = max(M,0), M⁻ = max(−M,0),

which is nonnegative and satisfies π ∘ L(M) = M ∘ π exactly: **the lifted
machine projects to the signed machine, always.** A negative weight −|w| is
realized by routing to the *opposite* sheet (a "sheet swap", `flip` in code).
Subtraction is gone.

The catch is dynamic range. Without cancellation the two sheets accumulate a
growing *common mode*: the same large quantity on both rails, carrying no
projected information but consuming physical energy / headroom. The Bruun
circuit makes this acute — its binomial (sum/difference) wires **double**
their sheet mass every stage, while its rotation wires grow only by
c + s ∈ (1, √2]. The reference code (`bruun512_positive_forward_inverse.py`)
capped the blow-up with a single global scalar `stage_scale = 0.5` and undid
it with a compensation factor at the end. That is an anti-pathological hack:
it over-damps the rotation rails, it is not an invariant of the circuit, and
the end compensation re-inflates the floating-point floor (visible as the
ramp case's 1e−12 error in the reference).

## 2. The theorem: the mass-potential gauge

**Claim.** There is a unique positive per-wire gauge that makes *every* lifted
cell exactly mass-preserving while keeping the projection exact.

Let the forward circuit be a feed-forward composition of local cells. Define a
positive potential g_w on every wire by the **backward recursion**

    g_w = Σ_{cell outputs i fed by w} g_i · |M_{iw}|,     g = 1 on outputs.

This is one backward sweep of the all-ones covector through the
*absolute-value* circuit |M|. Re-express each wire in gauged units u_w = g_w·x_w;
equivalently replace each weight by M̃_{iw} = g_i · M_{iw} / g_w. Then:

1. **Exact mass preservation.** Every gauged cell is column-stochastic in
   absolute value: Σ_i |M̃_{iw}| = 1 for every input wire w. Total sheet mass
   is invariant under every transport cell; it changes *only* at annihilators.
   The machine therefore keeps an exact **dissipation ledger**
   `mass_in = mass_out + Σ_stages annihilated`.
2. **Exact projection.** Present the input in gauged units; the gauges
   telescope through the composition and the output boundary is g = 1, so the
   projected output equals the FFT with **no compensation factor at all**.
3. **Uniqueness.** g is the Perron left-vector of the absolute circuit — the
   physically meaningful "mass amplification from this wire to the output".
   It is unique up to the boundary normalization.

*Proof sketch.* Column sums of |L(M̃)| equal column sums of |M̃| (the lift
duplicates each column into two nonnegative halves whose absolute sums add to
the original), which the recursion pins to 1. π intertwines L(M̃) and M̃;
annihilation subtracts equal mass from both rails so it commutes with π.
Composition telescopes g_i/g_w to 1 except at the two boundaries. ∎

**Certified** in `cone_fft_gauged.py`: cell-stochasticity defect 2.2e−16
(forward and inverse), forward-vs-numpy 1e−14, roundtrip 1e−15, transport
defect per stage 2e−16, and the no-annihilation variant conserves sheet mass
to `mass_out/mass_in = 1.000000` exactly (pure transport is lossless; only the
absorbers dissipate). The input gauge spans g ∈ [256, 1324] for n = 512 —
structured, not a runaway — with ratio 5.17.

## 3. The dissipation ledger (the physical cost meter)

The gauge turns "numerical cancellation" into a measurable physical quantity:
the mass sent to absorbers each stage. For a normal-random input, n = 512:
**97.4% of the input sheet mass is annihilated** across the transform — the
FFT is, physically, a near-total cancellation engine, and the ledger says so
stage by stage (75110 → 62675 → 43758 → … burned; see the trace). This is the
honest dynamic-range / energy budget of a physical build: the absorbers must
handle ~40× the projected signal mass. It is the number a crude global scalar
was hiding.

Design consequence: **where** you annihilate is a real engineering knob. Early
annihilation (each stage) minimizes peak wire mass — the gauged machine's peak
falls monotonically 2646 → 48 across stages. Deferred annihilation (transport
only, absorb once at the end) is exactly lossless in transport but needs rails
sized for the full undamped mass. The gauge makes both exact; the choice is
thermal, not numerical.

## 4. The cured cone-DIF, as hardware

The DIF is the intuitive physical form: a feed-forward tree of binomial
splitters and Bruun rotation couplers, no feedback. Element dictionary:

| algebra | mass-substrate element |
|---|---|
| x = x⁺ − x⁻ | two rails (two-rail circuit / two power paths / ± flexure modes) |
| + | Y-combiner / directional coupler (sum port) |
| − (sign) | **sheet swap**: cross the two rails (a braid / crossover) |
| × c, × s (c,s ≥ 0 after gauge) | attenuating tap / coupling ratio |
| cancellation | **absorber** at min(p,n): matched load / dump port |
| gauge g_w | fixed per-wire scaling = coupler ratios chosen once |

Because the gauge makes every cell column-stochastic, the coupler ratios are
*passive-realizable* (they sum to ≤ 1 in magnitude — no gain needed anywhere;
a purely passive network suffices, with the absorbers as the only lossy
elements). Candidate media: two-rail RLC ladders (differential lines); nanophotonic
directional-coupler meshes (the Bruun tree is a binary mesh of 2×2 and 4×4
couplers with fixed splitting ratios + fixed phase pads); MEMS flexure networks
where the ± sheets are ± displacement modes. The transform is the steady-state
power distribution of the mesh; the spectrum is read as the differential
(rail-A minus rail-B) amplitude at the output ports.

## 5. Toward the cone-DIP: the delay-line Fourier machine

The DIP carries complex phase, which looks *harder* to physicalize — until you
put it on a carrier, where the field itself supplies the algebra for free.
Realized and certified in `cone_dip_delayline.py`:

- **A value is a narrowband envelope** z = a + j·b on a carrier of period T.
  The DIP packet (a, b) = (Re, −Im) of a residue diagonal is exactly this I/Q
  envelope.
- **Negation is a T/2 delay.** A half-period path difference multiplies the
  envelope by −1. The cone's two sheets become the two carrier phases of one
  line; the sheet-swap becomes a half-period stub.
- **Rotation is a delay.** The Bruun rotation R + jI = e^{jθ}·z_o is a pure
  carrier phase shift θ = 2π·(path length)/λ. **Every twiddle is a length.**
  No multiplier exists in the machine.
- **The cell is two junctions.** lo = z_e + W·z_o, hi = conj(z_e − W·z_o) with
  W = e^{jθ}: one θ-stub, one sum junction, one difference (T/2 stub + sum),
  one conjugation (lower-sideband selection; in two-rail real form, one T/2
  stub on the b rail). **Census: 1 θ-stub + 3 half-period stubs + 2 junctions
  per cell; zero active elements.**
- **Annihilation is interference.** Destructive superposition at a junction
  *is* the min(p, n) removal — the absorber the DIF needs explicitly, the DIP
  gets from physics at every junction.

The DIP's isoclinic angle law θ(d) = πd/e (constant across a whole span) is the
decisive hardware advantage for a **serial, time-multiplexed** build: one
physical section per level (a feedback delay of w/2 slots + one coupler + one
θ pad), and the θ control is a per-stage **staircase** — quasi-static trombone
lines or thermo-optic pads suffice, no fast modulator. The serial census
(n = 512): 6 sections, feedback delays {32,16,8,4,2,1} slots, θ-settings per
stage {3,6,12,24,48,96}. A serial *DIT* by contrast needs a per-slot agile
twiddle stream (a fast modulator + twiddle memory) — the DIP's constant-per-
span law is what makes the passive delay-line realization plausible.

Risk register for the DIP machine: narrowband/dispersion tolerance (θ is
wavelength-exact only at the design carrier); conjugation requires a physical
sideband-select or phase-conjugate element (nonlinear or balanced-mixer, the
one place activity may sneak in); delay-length calibration accumulates over
depth. High risk, high payoff — a fully passive spectral engine.

## 6. Where a physical Fourier machine earns its keep

Functions that are awkward or impossible on a von-Neumann FFT but natural here:

- **Zero-latency, zero-clock spectra.** A passive coupler mesh (cone-DIF) or
  delay-line (cone-DIP) produces the transform at the speed of propagation with
  no sampling clock and no arithmetic units — a spectrum that exists as a
  steady state. Useful for ultra-low-power always-on sensing (the mesh draws
  power only through its absorbers), and for front-end RF/optical channelizers
  that must not spend a clock cycle.
- **In-line optical spectral filtering / correlation.** With the transform in
  the optical path, residue-domain filtering (our `apply_residue_filter`) is a
  set of fixed attenuators between the forward and inverse meshes — an all-
  optical convolution engine with no detector/DSP/re-emit round trip.
- **Physical cancellation as sensing.** The dissipation ledger (§3) is a
  measurable signal: the mass arriving at each absorber reports the spectral
  cancellation structure directly. A device could read *where energy cancels*
  as a native output — anomaly/coherence detection without computing the
  spectrum at all.
- **Mechanical / MEMS modal analyzers.** With ± flexure modes as the sheets,
  a micro-machined coupler network is a passive vibration spectrum analyzer:
  the structure's steady deflection pattern *is* the transform of the drive.
- **Analog training substrates.** The gauge is a fixed set of passive ratios;
  a reconfigurable coupler mesh with tunable ratios is a physically-realized
  linear layer whose canonical calibration (the Perron gauge) is known in
  closed form — a bridge between the FFT and analog neuromorphic hardware.

## 7. What is proven vs. what is conjectured

Proven (machine-precision certificates in the companion code): the gauge
exists, is unique, makes every cell exactly mass-preserving, keeps projection
exact with no compensation, and yields an exact dissipation ledger; the
delay-line DIP with only stubs/junctions/conjugation equals numpy's rfft.

Conjectured / engineering-open: passive-media realizations meet their loss and
tolerance budgets; the DIP conjugation element can be made passive-enough; the
ledger's ~97% dissipation is acceptable for target applications (it is a
headroom cost, not necessarily a power cost, since deferred annihilation is
lossless in transport). These are the white-paper's calls to the lab.
