# BFFT roadmap

Current target: prepare BFFT for a beta-quality `0.1.x` release.

## Completed foundation

- Public C and C++ APIs live in `include/bfft/`.
- The implementation is split between `src/bfft.cpp` and
  `src/detail/bruun_kernel.hpp`.
- Makefile build, test, install, uninstall, examples, and optional probe targets
  exist.
- CMake build, test, install, optional probe, optional PFFFT benchmark, and
  optional IPP comparison paths exist.
- C and C++ examples are tracked under `examples/`.
- Correctness and C API tests are wired into both `make test` and CTest.
- Double and float32 standard, native, inverse, magnitude, residue, and filter
  API paths are documented.
- FFTW/SFDR support probes are tracked under `tests/` and can be built with
  `make probes`.

## Beta preparation

1. Land the first CI workflow and get a green GitHub Actions run on `main`.
2. Teach the human repository operator how to inspect CI failures and require
   checks before merging.
3. Polish package metadata so installed builds are discoverable through
   `pkg-config` and CMake package config files.
4. Keep downstream package-discovery smoke tests repeatable for staged installs.
5. Refresh `docs/release-checklist.md` with hosted CI evidence before tagging.

## Next development round

- Verify the new GitHub Actions workflow on the hosted repository.
- Promote the local `pkg-config` and `find_package(BFFT CONFIG REQUIRED)`
  smoke checks into scripted or CI-visible checks.
- Decide whether the beta should be numbered `0.1.0-beta.1`, `0.1.1-beta.1`,
  or kept as an untagged beta branch until CI history is stable.

## Later work

- Expand CI to macOS and Windows after the Makefile/CMake platform details are
  validated there.
- Add scalar-forced and AVX-class CI variants without exposing user-facing
  transform-selection flags.
- Add aarch64 NEON CI when hosted runners or cross-runner support are practical.
- Add release archive checks and optional package manager recipes after the beta
  workflow is repeatable.
