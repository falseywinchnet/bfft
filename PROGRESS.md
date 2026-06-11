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
- [ ] Staged install validated with `DESTDIR`.
- [ ] Downstream staged-install smoke program validated.
- [ ] API, examples, and docs audited against current headers.
- [ ] Release checklist updated with final validation evidence.
- [ ] Placeholder TODO scan completed.

## Round log

- Initial scaffold: added durable state files for unattended Codex rounds.
  Validation for the library itself has not been run by this scaffold task.
- Baseline validation: fixed the Makefile AVX2/FMA compiler probe so GNU make
  can parse it, then ran `make clean`, `make`, and `make test`.
  Result: all passed. Test output reported `backend=neon-128` and standard
  policy `fused-scatter-plus-layout-convert`.
