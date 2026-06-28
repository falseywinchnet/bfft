# DFFT boundary map and dragon control plan

DFFT is an experimental fork of the BFFT tree kept in this repository as a
self-contained dragon workspace.  The first purpose is not to change public BFFT
semantics.  The purpose is to make every boundary in the double real-to-complex
and complex-to-real path explicit before performance work begins, so later
kernels are not forced to reinterpret choices that earlier stages left implicit.


## Design DAG node contract

The dragon DAG is not a compute DAG.  A node is a provisional commitment at the
point where one atomic code operation yields to the next.  Each node records:

- the commitment being made,
- the dominant future consumer that should inherit the decision,
- the atomic yield where control would otherwise escape,
- the control currently surrendered to the caller, compiler, allocator, hardware,
  or undocumented local convention,
- the control DFFT intends to take back,
- and the internal degrees of freedom that remain adjustable before the design
  hardens into code.

The practical question for every node is: what is mine that I have not been
granted because this code trusted something outside the DFFT boundary to provide
it?  Only after that question is answered should a node become a kernel, an
arena, a queue, a cursor, or a public API rule.

## Step 1: front-to-back boundary inventory

### Real-to-complex double pathway

1. **API intake boundary**: `bfft_forward` validates plan, input, output, work,
   and native scratch ownership before crossing into `bruun::RFFT`.
2. **Plan persistence boundary**: `bfft_plan` owns `bruun::RFFT`; the generated
   schedules, table pointers, transform size, standard-output policy, and work
   sizes become long-lived commitments.
3. **Workspace persistence boundary**: `bfft_workspace` owns aligned reusable
   work storage.  This is the first explicit place where DFFT can bind scratch
   lifetime to a plan instead of a single call.
4. **Input cadence boundary**: the real input is consumed as a transform-sized
   contiguous epoch.  DFFT treats that epoch as an intake event rather than as a
   pointer LLVM may freely rediscover.
5. **Residue/native production boundary**: the Bruun forward kernel converts the
   time epoch into native residue order using `work` as the transport surface.
6. **Leaf/block boundary**: small base cases choose whether arithmetic is
   scalar, lane-paired, or AVX2/FMA vectorized.  This is where vector lanes become
   claims about which samples deserve to live together.
7. **Twiddle phase boundary**: each recombination commits to phase update order,
   sign convention, and fused multiply-add shape.
8. **Native-output boundary**: native complex bins leave the scheduled Bruun
   order.  DFFT names this as a valid consumer language instead of a temporary.
9. **Standard-output boundary**: `forward_standard` either emits standard bins
   directly through a two-phase policy or uses `native_scratch` as an explicit
   order-conversion bridge.
10. **Magnitude/polar consumer boundary**: magnitude and mag/phase APIs change
    the dominant future consumer; the correct transport form is no longer raw
    complex storage.
11. **C ABI egress boundary**: `bfft_complex` layout is asserted to match the
    internal complex type.  That ABI match is a hard register/memory contract.

### Complex-to-real double pathway

1. **API intake boundary**: `bfft_inverse` validates the standard complex epoch
   and output epoch.
2. **Standard-to-native boundary**: standard FFT-order input must become the
   internal inverse language before scheduled inverse work can begin.
3. **Polar decode boundary**: `inverse_mag_phase` commits magnitude/phase back to
   rectangular complex values before inverse scheduling.
4. **Native inverse boundary**: `inverse_native` enters when the caller already
   speaks the internal order, forbidding redundant reinterpretation.
5. **Inverse schedule boundary**: the inverse schedule reverses the forward
   residue schedule and appends the final inverse binomial tail.
6. **Register-lane boundary**: inverse recombination owns lane order, conjugate
   twiddle signs, FMA shape, and store adjacency.
7. **Block-event boundary**: every inverse codelet consumes a parent block and
   emits child blocks with fixed stride promises.
8. **Final time-domain boundary**: output samples are written once in caller
   order, scaled according to the existing BFFT convention.
9. **Residue escape boundary**: `bfft_inverse_residues` exposes the residue
   ontology directly for probes that want to manage their own transport.

## Step 2: machine-control opportunities

- Replace pointer-only scratch with named arenas: intake, native, standard,
  twiddle, lane-pack, and egress.
- Promote schedule records from arithmetic descriptions to boundary events with
  input span, output span, lane claim, phase claim, and next-consumer claim.
- Use explicit block cursors instead of recomputing offsets inside hot kernels.
- Keep native order as a first-class output mode and only cross to standard order
  when the declared consumer needs it.
- Attach store-adjacency metadata to each codelet so the next stage inherits a
  layout instead of inferring one.
- Batch adjacent transforms by plan-owned epochs so load cadence is under DFFT
  control rather than under the caller/compiler boundary.
- Specialize queues for later consumers: complex, magnitude, polar, inverse, and
  residue consumers should not share an anonymous scratch vocabulary.

## Step 3: override plan

DFFT starts as a full copy of BFFT so experiments can be invasive without
contaminating the clean baseline.  The experimental world should be introduced in
small committed stages:

1. Add a `dragon` namespace with boundary records and a DAG printer.
2. Teach the double forward path to emit a boundary trace in debug builds.
3. Replace generic scratch handoff with a plan-owned arena descriptor.
4. Split standard-output conversion into named native-to-standard events.
5. Add inverse boundary records that mirror the forward event DAG.
6. Add block cursors to the double scheduled forward path.
7. Add block cursors to the double scheduled inverse path.
8. Add lane-pack descriptors for AVX2/FMA codelets.
9. Benchmark native, standard, magnitude, and inverse paths at every stage.
10. Keep only changes where the dominant future consumer receives a simpler
    language than before.
11. Promote stable dragon mechanisms back toward the root BFFT only after their
    boundary ownership proves itself.

## Steps 4-11 benchmark loop

Every dragon change must report:

- transform size and precision,
- API path (`forward`, `forward_native`, `forward_magnitude`, `inverse`, or
  `inverse_native`),
- boundary event changed,
- timing before and after,
- numerical correctness status,
- and the later-stage reinterpretation that was removed.

The dragon rule is simple: no local kernel win matters unless a later consumer
has less ambiguity, less transport work, or fewer layout choices to remake.

## Manual boundary register

The executable `dragon.hpp` experiment was removed because boundary ownership is
not proven by a compute test.  The live boundary work now starts in
`src/detail/dragon_boundary_register.md`, which manually records the exact yield
points in the double forward and inverse paths and names the specific control
that DFFT must take back before code changes are justified.
