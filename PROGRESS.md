# BFFT progress

Agent loop status: active

## Current snapshot

- Public C and C++ headers exist under `include/bfft/`.
- The implementation is split between `src/bfft.cpp` and
  `src/detail/bruun_kernel.hpp`.
- The Makefile defines build, test, install, uninstall, examples, and clean
  targets.
- C and C++ examples exist under `examples/`.
- Correctness and C API tests exist under `tests/` and are wired into
  `make test`.
- Public API, architecture, maintainer, and release docs exist under `docs/`.
- The unattended agent loop state files and runner have been seeded.

## Completion checklist

- [x] Public C header exists.
- [x] Public C++ wrapper exists.
- [x] Source implementation exists.
- [x] Examples exist.
- [x] Tests are wired into `make test`.
- [x] Public docs exist.
- [x] Clean `make clean`, `make`, and `make test` result recorded.
- [x] Staged install validated with `DESTDIR`.
- [x] Downstream staged-install smoke program validated.
- [x] API, examples, and docs audited against current headers.
- [ ] Release checklist updated with final validation evidence.
- [ ] Placeholder TODO scan completed.

## Round log

- Initial scaffold: added durable state files for unattended Codex rounds.
  Validation for the library itself has not been run by this scaffold task.
- Baseline validation: fixed the Makefile AVX2/FMA compiler probe so GNU make
  can parse it, then ran `make clean`, `make`, and `make test`.
  Result: all passed. Test output reported `backend=neon-128` and standard
  policy `fused-scatter-plus-layout-convert`.
- Staged install validation: ran `make clean`, `make`, `make test`, then
  `make install DESTDIR=/tmp/bfft-stage.IkgTTn PREFIX=/usr`.
  Result: all passed. The staged files were:
  `/usr/include/bfft/bfft.h`, `/usr/include/bfft/bfft.hpp`,
  `/usr/lib/libbfft.a`, and `/usr/lib/libbfft.so`. Modes were `0644` for
  headers/static library and `0755` for the shared library. No install fixes
  were needed.
- Downstream staged-install smoke: ran `make clean`, `make`, `make test`, then
  `make install DESTDIR=/tmp/bfft-stage.downstream.6Dg0BM PREFIX=/usr`.
  Built an ignored C++ smoke program against the staged install with
  `c++ -std=c++17 -I/tmp/bfft-stage.downstream.6Dg0BM/usr/include build/downstream_smoke.cpp /tmp/bfft-stage.downstream.6Dg0BM/usr/lib/libbfft.a -lm -o build/downstream_smoke`.
  Ran `build/downstream_smoke`.
  Result: all passed. The smoke output reported `backend=neon-128` and standard
  policy `fused-scatter-plus-layout-convert`.
- API/docs audit: documented every public C declaration in `include/bfft/bfft.h`,
  added matching C++ wrapper comments in `include/bfft/bfft.hpp`, expanded
  `docs/api.md` to cover version/backend helpers, types, plan sizes, transform
  buffers, filtering buffers, and error handling, and clarified backend policy
  wording in `README.md`.
  Validation: `make clean` passed. A first plain `make` failed while compiling
  `examples/benchmark.cpp` because the default macOS compiler temp directory
  reported `No space left on device`. Removed prior BFFT staged-install temp
  artifacts under `/private/tmp`, then ran `make clean`,
  `TMPDIR=/private/tmp make`, and `TMPDIR=/private/tmp make test`.
  Result: all passed. Test output reported `backend=neon-128` and standard
  policy `fused-scatter-plus-layout-convert`.
- Example audit: added `examples/cpp_api_demo.cpp`, wired it into the
  `examples` and `all` builds, updated README example output lists and snippets
  to show plan-derived work and native scratch sizes, and clarified the examples
  role in `docs/architecture.md`.
  Validation: ran `make clean`, `TMPDIR=/private/tmp make`,
  `TMPDIR=/private/tmp make test`, `build/examples/c_api_demo`,
  `build/examples/cpp_api_demo`, and `build/examples/benchmark 64 1`.
  Result: all passed. Test and demo output reported `backend=neon-128` and
  standard policy `fused-scatter-plus-layout-convert`.
- Release checklist evidence attempt: read the release checklist, public
  headers, Makefile, tests, examples, and release docs, then attempted to gather
  fresh final validation evidence for `docs/release-checklist.md`.
  Validation: `make clean` passed. The first `TMPDIR=/private/tmp make` failed
  before compilation because `mkdir` reported `No space left on device` while
  creating `build/`. The repository measured 584K and the filesystem reported
  124Mi available at 100% capacity. No unrelated temp files were removed. A
  retry of `TMPDIR=/private/tmp make` passed. `TMPDIR=/private/tmp make test`
  then failed compiling `tests/correctness.cpp` with
  `clang++: error: unable to make temporary file: No space left on device`.
  The generated build tree measured 300K and the filesystem reported 117Mi
  available at 100% capacity. `git diff --check` passed after documenting the
  blocker.
  Result: blocked by host filesystem space. The release checklist evidence task
  remains unchecked, and `docs/release-checklist.md` was not changed.

  human note: i manually cleared git caches. this turns out to have eaten 16GB.
  i think theres a bug somewhere in the codex client side runtime.
  remember to check your git caches periodically- idk what the frick its doing.
  after that, i manually merged everything so some notes need checking and editing before proceeding.
  i then manually wrote a new benchmark.cpp and ran it. looking good so far! lets keep going.
- Repository refresh and workflow check: fast-forwarded `main` from
  `5b80049` to `424a615` and fetched tag `a-0.1`. The default local workflow
  passed after the cache cleanup: `make clean`, `make`, and `make test`.
  Test output reported `backend=neon-128` and policy
  `fused-scatter-plus-layout-convert`.
  Workflow notes: `.github/FUNDING.yml` exists, but there is no
  `.github/workflows/` CI configuration yet. New FFTW/SFDR probe sources now
  live under `tests/`, but `make test` still builds only `tests/correctness.cpp`
  and `tests/api_c.c`. Added a next implementation task for dedicated float32
  internals and float32 testing.
- Dedicated float32 internals and tests: added a single-precision complex type,
  float32 work-size query, standard/native float32 forward and inverse calls,
  float32 native/standard conversion helpers, and matching C++ wrappers. The
  float32 path uses float work buffers and float complex storage internally,
  with plan layout maps shared from the existing plan.
  Tests: extended `tests/correctness.cpp` to compare float32 standard spectra
  against a naive reference, check standard and native roundtrips, check native
  conversion, and exercise the C++ vector wrapper. Extended `tests/api_c.c` to
  cover float32 C forward/inverse, native conversion, and invalid-argument
  handling. Updated `README.md`, `docs/api.md`, and `docs/architecture.md`.
  Validation: `make clean`, `make`, and `make test` all passed on 2026-06-11.
  Test output reported `backend=neon-128` and policy
  `fused-scatter-plus-layout-convert`.
- Float32 SIMD helper structure: factored float32 work-buffer setup, inverse
  scaling, spectrum pack/unpack, and real-output copy into internal helpers in
  `src/detail/bruun_kernel.hpp`. The helper loops use the existing backend
  level for AVX-512, AVX2, SSE2, NEON, or scalar tails where applicable and do
  not expose any transform-selection compile flags. Updated
  `docs/architecture.md` and marked the selected task complete in `TASKS.md`.
  Validation: `make clean`, `make`, `make test`, and `git diff --check` all
  passed on 2026-06-11. Test output reported `backend=neon-128` and policy
  `fused-scatter-plus-layout-convert`.
- Float32 BH7 probe mode: extended `tests/bfft_fftw_sfdr_bh7_probe.cpp` with
  BFFT modes `f64-standard`, `f64-native`, `f32-standard`, and `f32-native`;
  native modes execute native forward and convert to standard order before SFDR
  measurement, keeping FFTW as the double-precision reference. Added `make probes` targets
  for the tracked probe sources under `tests/` and documented the BH7
  `f32-native` invocation in `README.md`.
  Validation: `make probes`, `make test`, and `git diff --check` passed on
  2026-06-11. A smoke run of
  `build/tests/bfft_fftw_sfdr_bh7_probe 8 2 4 bh7 f32-native` completed and
  printed `bfft_mode=f32-native`.

- Float32 BH7 folded-spur fix: the f32 complex FFT now keeps stage twiddles,
  twiddle recurrence, and butterfly products in double precision while storing
  public/work buffers as float32. This removes the deterministic folded-bin
  leakage seen at bins such as `N/2 - 7` in native f32 BH7 runs. The BH7 SFDR
  probe now loads FFTWf dynamically for f32 modes when available, adds an
  `fftw_precision` CSV column, and falls back to double FFTW explicitly when
  FFTWf is absent. Added a `make test` regression at `N=32768`, `k=7`, native
  f32 BH7, with a 144 dB SFDR floor after native-to-standard conversion.
  Validation: `make probes`; `build/tests/bfft_fftw_sfdr_bh7_probe 15 2 8 bh7
  f32-native`; `build/tests/bfft_fftw_sfdr_bh7_probe 22 8 22 bh7 f32-native`
  produced `fftw_precision=f32` and `bfft_sfdr_db=149.99262537` at 4M. Full
  `make test` validation is pending this change set.
