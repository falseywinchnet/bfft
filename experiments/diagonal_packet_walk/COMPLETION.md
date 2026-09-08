# Requirement-by-requirement completion audit

2026-09-04. The completed objective is the four-part investigation of a
diagonal packet family and its costs. It does not require demonstrating a
faster algorithm, and this experiment does not claim one.

| Requested item | Authoritative construction/evidence | Finding |
|---|---|---|
| Define a diagonal packet family at increasing sizes | README's H and unique (p,q,j,l) coordinates; `Packet.__init__`, `forward`, `inverse`; `test_packet.py` | Exact N=4^r family, including recovered N=64 case. Odd r has oblique subgroup; even r is rectangular. |
| Derive transition and carry rules explicitly | README's monomial P, natural C, invariant labels, Cp block formula, and cancelled transition; separate executable forms in `packet.py`; dense `first_sweep.json` | All mappings and phases are specified. No input/output-only diagonal factor is substituted for the entrywise kernel identity. |
| Determine packet complexity and economical recursion | a² invariant blocks of dimension b²; full-rank Vandermonde/separation-rank proof; exact factored and cancelled implementations | Blocks and fixed-partition separation rank grow. Exact application remains O(N log N) via Fourier factors. Cancellation reveals a mixed DIT/DIF schedule, not an independent bounded diagonal primitive. |
| Account for required movement against DIT, DIF, Stockham | README pass/index/coefficient/scratch ledger; `audit_counts.py`; scalar and NEON final JSON; `NEON_LEDGER.md`; generated-code inspection | Entry, exit, internal gathers/scatters, reversals, coefficients, index references, and explicit register lane exchanges are charged under a declared logical-access model. Packing has no boundary conversion. No vector stack accesses in the inspected uninstrumented kernel bodies. |

## Verification inspected at completion

- `test_packet.py`: three suites pass. Independent dense Q/F/G basis checks
  through N=256 at three slopes; vector patterns and cancelled transition
  through N=65536; invariant-label checks through N=4096.
- `first_sweep.json`: dense independently constructed matrices through N=1024.
  This establishes finite-size support observations alongside, rather than
  instead of, the all-size algebraic formulas.
- `neon_oracle_m4.json`: 48 explicit native input/output cases checked with
  NumPy FFT, covering six schedules and all eight sizes. Maximum absolute
  error 7.9824e-12; relative L2 errors also gated.
- `native_neon_final_m4.json` and `native_scalar_final_m4.json`: each has 48
  size/method records. Seven arithmetic/sample counters match independent
  formulas. Four NEON movement fields also match the specified representation.
- `neon_assembly_m4.json`: actual compiled assembly hash, primitive lowering,
  full static opcode inventories for inspected functions, and no vector stack
  accesses in the three uninstrumented kernel/dispatch bodies. The parser
  checks scaled and unscaled stack memory instructions and spill annotations.
- `provenance.json`: hashes current sources and the final evidence files.

## Limits of the conclusion

Logical accesses and intrinsic operations are not DRAM traffic or a dynamic
CPU instruction trace. Compiler setup/control/ABI instructions are represented
by the static assembly inventory, not falsely counted as complex data routing.
The NEON packing uses one complex double per register. Alternative SoA,
cross-butterfly, AVX, or production real-input BFFT implementations may have
different costs. These results prove the construction and account for the
specified implementations; they are not an optimality theorem over all FFTs.

No production kernel or website deployment was changed by this experiment.
