# Dragon FFT boundary register

This register replaces the removed `dragon.hpp` experiment.  It is not a C++ API,
not a compute test, and not a model of the transform.  It is a manual register of
places where one atomic code operation yields to the next and where DFFT must
stop trusting the compiler, caller, allocator, or local convention to build the
machine around the FFT.

## Double real-to-complex path

### R2C-00: C API intake yields to `bruun::RFFT`

- Current site: `dfft/src/bfft.cpp`, `bfft_forward`.
- Current handoff: validates non-null plan/input/output/work and conditional
  native scratch, then calls `plan->impl.forward_standard(...)`.
- Surrendered control: pointer alignment, alias class, input cadence, work-span
  ownership, scratch-span ownership, and output store contract are not named.
- Dragon takeover: replace this anonymous call boundary with a call-local intake
  record held on the stack by `bfft_forward`, then pass that record into a DFFT
  execution entry point.  The record must name input epoch, output epoch, work
  epoch, native scratch epoch, alias relation, and required output language.

### R2C-01: Plan persistence yields to execution

- Current site: `dfft/src/detail/bruun_kernel.hpp`, `RFFT::init`,
  `work_size`, `native_scratch_size`, and `standard_output_policy_name`.
- Current handoff: plan initialization stores schedules and policy indirectly;
  execution rediscovers meaning through member functions and branches.
- Surrendered control: schedule identity, table adjacency, standard-output
  policy, work-span partition, and consumer priority are not carried as one
  execution object.
- Dragon takeover: add a plan-owned execution manifest beside `RFFT` containing
  size, bins, native order, standard policy, table spans, and named work spans.

### R2C-02: Work buffer yields to residue/native production

- Current site: `RFFT::forward_native_with_alignment`, `forward_recursive`,
  `forward_residues`, and `forward_residues_inplace`.
- Current handoff: a `double* work` span becomes many meanings: copied input,
  residue storage, stage state, and tail source.
- Surrendered control: the compiler sees a pointer; DFFT does not own a typed
  arena language for the buffer.
- Dragon takeover: split work into named arenas before entering residue code:
  intake-copy, residue-current, residue-next, phase/twiddle, and standard-egress.
  Every stage receives arena cursors rather than recomputed raw offsets.

### R2C-03: Load cadence yields to first residue writes

- Current site: `forward_residues` and recursive/scheduled leaf entry.
- Current handoff: input samples are loaded according to local loop form.
- Surrendered control: prefetch distance, batch stride, adjacent-transform
  cadence, and load grouping are outside the algorithm's vocabulary.
- Dragon takeover: introduce an intake cursor with sample stride, batch stride,
  load group size, and prefetch distance.  Leaf code consumes the cursor instead
  of the raw input pointer.

### R2C-04: Leaf/codelet yields to scheduled block frontier

- Current site: q-specific leaves, `forward_stage`, and AVX2/SIMD codelets in
  `bruun_kernel.hpp`.
- Current handoff: codelet-local arithmetic chooses live range, lane grouping,
  and store order.
- Surrendered control: register lifetime and lane ontology are left to local C++
  expressions and compiler allocation.
- Dragon takeover: define block-event records with input arena, output arena,
  lane pack, live range budget, store group, and downstream consumer.

### R2C-05: Twiddle phase update yields to native bins

- Current site: twiddle recombination code in forward stage/tail functions.
- Current handoff: phase signs, recurrence, and table loads are embedded in
  arithmetic expressions.
- Surrendered control: phase cursor, table locality, and FMA polarity are not
  owned as data.
- Dragon takeover: pass an explicit phase cursor into recombination codelets;
  table span and FMA polarity must be part of the block-event record.

### R2C-06: Native bins yield to standard output

- Current site: `forward_standard`, `forward_standard_two_phase`, and
  `native_to_standard_complex`.
- Current handoff: a policy branch decides direct two-phase output or native
  scratch conversion.
- Surrendered control: gather order, scratch bridge ownership, and output store
  adjacency are hidden behind the policy branch.
- Dragon takeover: represent native-to-standard as its own transport stage with
  gather cursor, scratch cursor, store cursor, and declared consumer language.

### R2C-07: Complex bins yield to magnitude/polar consumers

- Current site: `forward_magnitude` and `forward_mag_phase`.
- Current handoff: consumer-specific output is calculated after complex/native
  work has already shaped the machine.
- Surrendered control: the dominant future consumer does not influence earlier
  storage language soon enough.
- Dragon takeover: consumer language becomes part of the intake record.  If the
  consumer is magnitude or polar, standard complex materialization is optional,
  not assumed.

## Double complex-to-real path

### C2R-00: C API intake yields to inverse execution

- Current site: `dfft/src/bfft.cpp`, `bfft_inverse`.
- Current handoff: validates plan/input/output and calls `plan->impl.inverse`.
- Surrendered control: input order, output alias class, destination cadence, and
  inverse scratch ownership are unnamed.
- Dragon takeover: add inverse intake record naming standard/native input order,
  output epoch, alias relation, and destination stride before inverse decode.

### C2R-01: Standard bins yield to internal inverse language

- Current site: `RFFT::inverse` and native-to-standard inverse helpers.
- Current handoff: standard bins are decoded through internal indexing.
- Surrendered control: decode staging, symmetry reconstruction, and read
  adjacency are not visible as boundary decisions.
- Dragon takeover: create a standard-decode stage with read cursor, symmetry
  policy, scratch cursor, and next block frontier.

### C2R-02: Inverse schedule yields to register-lane recombination

- Current site: scheduled inverse operations and inverse codelets.
- Current handoff: parent blocks become child blocks through local codelet loops.
- Surrendered control: parent cursor, child cursor, queue order, lane order, and
  spill budget are not owned outside the compiler's register allocator.
- Dragon takeover: inverse block-event records must own parent span, child span,
  lane pack, store group, spill budget, and twiddle polarity.

### C2R-03: Child blocks yield to final real output

- Current site: final inverse binomial/tail and output writes.
- Current handoff: samples reach caller memory in the loop order chosen by the
  current inverse form.
- Surrendered control: write-combine grouping, scaling placement, and output
  cadence are not a final egress object.
- Dragon takeover: final egress stage owns output cursor, scale placement,
  write-combine group, tail policy, and caller-visible sample order.

## First implementation target

The first real dragon implementation should not add a public abstraction layer.
It should alter the double forward path internally:

1. Add a private intake record in `src/bfft.cpp` for `bfft_forward`.
2. Add a private execution manifest to `RFFT` for work-span names and standard
   policy.
3. Thread the intake record and manifest into `forward_standard` without changing
   the public ABI.
4. Replace at least one raw-offset stage boundary with an arena cursor consumed
   by the existing arithmetic.
5. Benchmark only after a raw pointer/recomputed offset boundary has actually
   disappeared.
