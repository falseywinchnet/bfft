# BFFT task queue

Each unattended round should complete exactly one unchecked task. If the queue
gets stale, replace or refine tasks before doing implementation work.

- [x] Run baseline validation with `make clean`, `make`, and `make test`;
  fix only small failures if found, then record the result in `PROGRESS.md`.
- [x] Validate staged installation with `DESTDIR` and `PREFIX`, then record the
  installed file list and any fixes in `PROGRESS.md`.
- [x] Build a tiny downstream smoke program against the staged install and
  document the exact compile/link command.
- [x] Audit `include/bfft/bfft.h`, `include/bfft/bfft.hpp`, `README.md`, and
  `docs/api.md` for API mismatches.
- [x] Audit examples for copy-paste quality and current scratch-size guidance.
- [x] Develop dedicated float32 internals and tests: add a real single-precision
  internal transform path, expose only policy-neutral public API changes if
  needed, and add focused float32 correctness/API coverage through `make test`.
- [ ] Update `docs/release-checklist.md` with final validation evidence for
  the first `0.1.0` release.
  The previous disk-space blocker has been cleared locally; `make clean`,
  `make`, and `make test` passed on 2026-06-11. Finish this task by rerunning
  the full release validation set, including staged install and downstream
  smoke, and recording the evidence.
- [ ] Run a placeholder TODO scan across source, public docs, examples, and
  tests; remove stale markers or record any post-release items in the roadmap.
- [ ] Run a final `IDEA.md` completion pass and mark `PROGRESS.md` complete
  only if every first-release checklist item is satisfied.
