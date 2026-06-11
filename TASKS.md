# BFFT task queue

Each unattended round should complete exactly one unchecked task. If the queue
gets stale, replace or refine tasks before doing implementation work.

- [x] Run baseline validation with `make clean`, `make`, and `make test`;
  fix only small failures if found, then record the result in `PROGRESS.md`.
- [x] Validate staged installation with `DESTDIR` and `PREFIX`, then record the
  installed file list and any fixes in `PROGRESS.md`.
- [ ] Build a tiny downstream smoke program against the staged install and
  document the exact compile/link command.
- [ ] Audit `include/bfft/bfft.h`, `include/bfft/bfft.hpp`, `README.md`, and
  `docs/api.md` for API mismatches.
- [ ] Audit examples for copy-paste quality and current scratch-size guidance.
- [ ] Update `docs/release-checklist.md` with final validation evidence for
  the first `0.1.0` release.
