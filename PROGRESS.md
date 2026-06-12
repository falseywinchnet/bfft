# BFFT progress

Agent loop status: active

## Current snapshot

- Public C and C++ headers exist under `include/bfft/`.
- Implementation details live in `src/bfft.cpp` and
  `src/detail/bruun_kernel.hpp`.
- Makefile and CMake both build the library, examples, and core tests.
- `make test` and CTest cover correctness and the C API.
- Optional FFTW/SFDR/library comparison probes live under `tests/` and are built
  by `make probes` or the CMake probe option.
- Package metadata now exists for `pkg-config` and CMake package consumers.
- A first GitHub Actions workflow exists for Ubuntu Makefile and CMake builds.
- `TODO_HUMAN.md` explains the GitHub-side CI and branch-protection steps for a
  repository operator who has not used GitHub Actions before.

## Completion checklist

- [x] Public C header exists.
- [x] Public C++ wrapper exists.
- [x] Source implementation exists.
- [x] Examples exist.
- [x] Tests are wired into `make test`.
- [x] Tests are wired into CTest for the CMake build.
- [x] Public docs exist.
- [x] CMake build support exists.
- [x] First-pass package metadata exists.
- [x] First-pass CI workflow exists.
- [x] Human CI setup manual exists.
- [ ] First hosted GitHub Actions run verified green.
- [x] Package-discovery downstream smoke tests passed for staged installs.
- [ ] Release checklist updated with final beta validation evidence.
- [ ] Placeholder TODO scan completed.

## Recent commit audit

Recent commits show that the repository has moved beyond the old first-release
plan:

- CMake build and external FFT comparison probe support were added.
- Forward segment locality was improved.
- Magnitude-only forward APIs were added.
- Float32 SIMD and rounded-twiddle work landed before the current cleanup.
- Deleted probe files were later restored as tracked probe sources under
  `tests/`.

The roadmap and task queue have been updated to treat CI, package discovery,
and beta validation as the next work instead of listing CMake and package
metadata as future-only items.

## Round log

- Baseline release preparation established the public headers, implementation,
  examples, Makefile, tests, public docs, release notes, and maintainer notes.
- Staged install and downstream smoke validation previously passed after local
  disk-cache cleanup. The disk-space blocker was environmental, not a repository
  build issue.
- Float32 development added dedicated public float32 entry points, internal
  float32 work-buffer helpers, SIMD-aware helper structure, FFTW/BH7 probes, and
  rounded plan-owned float32 twiddle tables for the native BH7 target.
- CMake support was added with library, examples, CTest, optional probes,
  optional PFFFT benchmark comparison, optional IPP comparison, and install
  rules.
- This cleanup round added Ubuntu GitHub Actions CI, package metadata install
  rules, a human CI manual, refreshed progress/roadmap/task/Codex files, removed
  stale `.DS_Store` files, documented the current beta direction, and validated staged package discovery with both `pkg-config` and CMake `find_package`.

## Remaining beta work

1. Push the branch and verify the hosted GitHub Actions run.
2. Refresh `docs/release-checklist.md` with the hosted CI URL and any final
   release-machine validation evidence.
3. Complete a placeholder TODO scan and move any real post-beta work into
   `ROADMAP.md`.
