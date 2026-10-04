# BFFT experiment recovery

Recovery began October 3, 2026 CDT and completed its validation October 4.
The isolated branch is `codex/experiment-recovery-20261003`, based on main
`6e64386f2da658a90493733c08b01dea6425ecdd`. The authoritative main checkout and
all original research/consumer checkouts remain untouched. This is an integration
and preservation branch, not a promotion of these experiments into production.

## Recovery index

| Original checkout / branch / host | Committed destination | Status and resolution |
| --- | --- | --- |
| Neo `bfft`, uncommitted main | `experiments/conv_dual_pair`, `conv_fast_aa`, additions under `conv_warp`, and CONV paper sources | Recovered mathematical/prototype work and compact receipts, including rejected pairing, ringing, visibility, and cost results. |
| Neo `bfft`, uncommitted main | `experiments/krylov_em`, `tropical_transport`, `equilateral_pentagon_reptiles` | Experimental/negative findings preserved. No new acceleration or novelty claim. |
| Neo `bfft`, uncommitted main | `experiments/scene_pattern`, plus the native CONV nested-dispatch fix | Reusable source and tests recovered. The 120-capture reconstruction objective remains incomplete. Private photographs and collection-specific results are outside Git. |
| Neo `bfft`, uncommitted main | `realtime_vector_fx` entropy extension, `mosaic_fx`, `mac_audio_normalizer` | Source, build integration, reference tests, and compact public receipts recovered. No plugin installation or audio-device change. |
| Neo `wrench-engine-study/bfft`, `codex/wrench-packing-study` at `1ff367f` | `experiments/wrench_transport`, `paper/wrench-transport` | Native baseline and existing history integrated by merge `c4aced6`; uncommitted packing fixtures, protocols, scorers and findings recovered separately. |
| Neo `obligation-dynamics/bfft`, `codex/obligation-dynamics` at `1ff367f` | `experiments/obligation_dynamics`, `experiments/tape_dynamics` | Stable source snapshots, with fixtures and compact diagnostics. Tape was active and had no completed validated final report at cutoff; see its `RECOVERY_STATUS.md`. |
| Neo `krylov-bregman-research/bfft`, `codex/krylov-bregman-research` at `181ea97` | `experiments/krylov_bregman`, `entropic_transport_closure`, `paper/krylov_bregman` | Existing history integrated by merge `b5c73d3`; primitive/frame/multiplicative follow-ups and manuscript edits recovered. Rejected potential-optimizer direction remains labeled rejected. |
| Same Krylov worktree | `experiments/library_acceleration` | Real-library adapters, analytical-twin direction, tests and negative/default-comparator results recovered. |
| Mini `/Users/joshuahkuttenkuler/code/bfft`, main at `7e41072` | `experiments/mini_legacy_recovery` | Untracked RFHT, sin/cos and STFT diagnostics. Four duplicate records point to existing equivalent files instead of adding duplicate copies. The old committed HEAD is already in main's ancestry. |
| Mini `.codex/worktrees/77b6/bfft`, detached at `7e41072` | `research/conv_reshape` | Independent September source-provenance review, script, and publication snapshot. Historical dated verification, not a new claim about current publication status. |
| Mini build mirrors `bfft-6b3e7ffa7539`, `bfft-7852c0dd62d1`, `bfft-daa90db3ea8e` | `recovery/2026-10-03/variants` | Eight divergent mirror files retained as historical variants. Local sources are canonical: mirror packing lacks later demo support, and mirror analytical-twin guidance predates the user's later direction. |
| Mini targeted `/private/tmp` diagnostics | `recovery/2026-10-03/evidence/mini-scratch` | 28 unique compact receipts/fixtures recovered. These are historical scratch results with incomplete run provenance, not a fresh accepted study. No additional unique source was found there. |
| Neo `/Users/ultimussecundai/Downloads/cleanup2`, vendor extension | `experiments/cleanup_ring_transfer` | Lane-valued Bruun DIT extension, complex codelet, independent ring probe, dependencies and compact evidence. Production BFFT kernels unchanged. |

The exact snapshot commit is recorded in `COMMITS.md` after the snapshot commit
is created. Merge commits preserve original research commit history. Per-file
origins, branch/HEAD, destination, byte count, SHA-256 and copy time are in
`manifest.json`, `remote-manifest.json`, `consumer-manifest.json`, and
`scratch-manifest.json`. Variant paths contain original bytes, not silently
blended algorithms. The root AGENTS.md combines applicable experiment commands;
original instruction snapshots are retained for comparison.

## Evidence storage and exclusions

Large physics trajectories and the rejected-solver source archive are stored
by SHA-256 under:

`/Users/ultimussecundai/Documents/CodexRecovery/bfft-20261003/evidence/`

The manifest records every original relative path and its hash-addressed storage
location. Compact scores, scene fixtures, validation receipts and failed runs
remain in Git. `restore_evidence.py` verifies full hashes and reconstructs large
evidence into a separate directory without replacing divergent files:

```sh
python3 recovery/2026-10-03/restore_evidence.py \
  --destination /tmp/bfft-restored-evidence --verify-only
python3 recovery/2026-10-03/restore_evidence.py \
  --only experiments/wrench_transport/packing/ \
  --destination /tmp/bfft-restored-evidence
```

All 505 external-evidence records passed checksum verification; duplicate
records share stored content. One 41.9 MB archive was restored and independently
matched to its manifest hash. The archive remains a rejected historical
direction; it was not extracted or added blindly to the source tree.

The private scene collection and outputs were preserved with APFS clonefile:
**24,126 files, 4,103,659,579 logical bytes**, under the same local recovery
directory's `private-scene-pattern/`. Its private per-file checksum manifest
is `private-scene-manifest.json`; the public aggregate and manifest digest are
in `inventory.json`. This is a same-disk snapshot, not off-device backup.
Neither the images nor collection-specific raw receipts or full private file
inventory were put in Git. Historical source documentation retains its image
identifiers where needed to explain the experiments. Original scene files remain in place.

Compiler executables, WASM binaries, environments, caches, local tool settings,
and host installation receipts were excluded from the public additions.
Meaningful ignored evidence was reviewed and curated instead of discarded.
The two WASM modules used for validation were rebuilt from retained C sources.
No source copy was deleted, reset, cleaned, archived, or modified. Existing
committed branch artifacts retain their history; this recovery does not rewrite
past commits to remove their previously committed paper/measurement assets.

The deleted main-checkout `output/pdf/conv_paper_fused.tex` was not deleted in
this branch: the existing committed paper is retained alongside the recovered
new composition. Other old local backup/publish branches have no commits beyond
main. A stale Mini baseline worktree registration points to a missing directory;
no Git pruning was performed. The consumer website exhibit stays in its owning
repository. The Mini's recent removal of checksum-identical mirror files was
not treated as lost source. Active Tape outputs after the documented cutoff
belong to the continuing physics task and are not silently folded into this
snapshot.

## Validation

Validation ran against the recovery checkout's own Mini mirror
`bfft-675dacf2c222`, with build outputs in `/tmp/bfft-recovery-20261003`.
The harness deferred when it detected the parent's physics benchmark. Builds
used one job and reduced scheduling priority. No physics benchmark was run
as part of recovery, and none of these test durations is a performance claim.

* **402 Python tests passed** across CONV, entropic/Krylov/library transport,
  scene-pattern, and native CONV backend regressions. The initial system-Python
  run had two import errors because Matplotlib was absent. The ten tests in
  those two modules passed using the already-installed Mini Miniforge runtime;
  no packages were installed.
* **10 CTest targets passed**: Tape and its Wrench acceptance/numerical checks,
  Obligation, entropy-stretch core/chains/reference equivalence, and Mosaic core.
* **Seven JavaScript test/check scripts passed**, covering Wrench prototype
  behavior, three independent scoring copies, and the 108-case line-profile
  check. Its initial missing-input error was resolved by restoring the retained
  public line-coverage receipt; the implementation was unchanged.
* **120 rebuilt-WASM cases passed**, including 3,000,816 exactly matching
  transient controls and zero differing encoded bytes. Maximum fused pixel
  difference was `6.062042645638144e-9`. This is finite validation, not a universal
  bit-identity guarantee for fused arithmetic.
* The recovered Cleanup ring probe passed whole-field, direct-DFT, peak and
  competitor checks against the current BFFT library. Its incidental timing
  output is not used as a new performance result.
* TV Normalizer built successfully into an isolated `/tmp` app. It was not
  installed or launched; live audio routing and OBS/Metal host behavior were not
  requalified in this recovery.

Full commands, output and failed-initial-attempt receipts are under `validation/`.
`verify.py` reproduces the focused native/Python/JavaScript phases through
`m4build`; it uses existing optional dependency sources from the old Krylov
mirror without installing software. Follow each experiment README for full
benchmarks and bulk-output reproduction. The CONV freestanding reference can
be rebuilt with `build_wasm_reference.sh` using the Neo's existing WASI SDK.

The source/path review found no credential-pattern or compiled-executable
alerts in candidate additions. The remote was verified PUBLIC before preparing
publication, which is why private datasets, local installation records and the
uninspected rejected-source archive remain outside the public branch. This is
a bounded content review, not a formal secret-detection guarantee.
