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
