# BFFT task queue

Use this queue for small reviewable rounds. Keep three to six unchecked tasks
ready while beta work remains.

## Completed

- [x] Run baseline validation with `make clean`, `make`, and `make test`.
- [x] Validate staged installation with `DESTDIR` and `PREFIX`.
- [x] Build a tiny downstream smoke program against the staged install.
- [x] Audit public headers, README, and API docs for API mismatches.
- [x] Audit examples for copy-paste quality and current scratch-size guidance.
- [x] Add dedicated float32 internals and tests.
- [x] Add internal float32 SIMD helper structure.
- [x] Extend FFTW/BH7 probes for float32 native runs.
- [x] Fix the float32 native BH7 folded-spur regression.
- [x] Add CMake build, test, install, and optional comparison probe support.
- [x] Add first-pass GitHub Actions CI for Makefile and CMake Ubuntu builds.
- [x] Add first-pass package metadata for `pkg-config` and CMake consumers.

## Next

- [ ] After this branch is pushed, verify the GitHub Actions **ci** workflow on
  GitHub and record the first green run in `PROGRESS.md`.
- [x] Smoke-test staged installs with `pkg-config --cflags --libs bfft` and a
  downstream `find_package(BFFT CONFIG REQUIRED)` CMake project.
- [ ] Update `docs/release-checklist.md` with final beta validation evidence,
  including the hosted GitHub Actions run URL after CI is enabled on GitHub.
- [ ] Run a placeholder TODO scan across source, public docs, examples, tests,
  and CI; move genuine post-beta work into this roadmap.
- [ ] Run a final `IDEA.md` completion pass and mark `PROGRESS.md` complete only
  if every first-release checklist item is satisfied.
