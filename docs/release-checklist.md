# BFFT beta/release checklist

## Completed in this repository setup

- Split the prototype into public headers, source, examples, tests, and docs.
- Removed the embedded benchmark `main` from the library kernel.
- Added `make`, `make test`, and `make install` workflows.
- Added CMake configure, build, test, and install workflows.
- Added C and C++ APIs.
- Added double and float32 transform paths.
- Added correctness and API tests.
- Added optional comparison probes for FFTW/SFDR/library checks.
- Added automatic backend and packing policy.
- Added repository instructions for future coding agents.
- Added maintainer notes for GitHub-only repository settings.
- Added first-pass GitHub Actions CI for Ubuntu Makefile and CMake builds.
- Added first-pass package metadata for `pkg-config` and CMake package users.

## Before tagging a beta

- Verify the hosted GitHub Actions workflow is green on the default branch.
- Run `make clean && make && make test` locally.
- Run CMake configure, build, and CTest locally.
- Run a staged Makefile install with `DESTDIR`.
- Run a staged CMake install.
- Build and run a small downstream program against the staged install.
- Smoke-test `pkg-config --cflags --libs bfft` against a staged install.
- Smoke-test `find_package(BFFT CONFIG REQUIRED)` against a staged CMake
  install.
- Review `docs/maintainer-notes.md` and apply GitHub settings that are possible
  before beta.
- Create release notes that call out Linux as the validated platform for the
  first beta unless additional hosted CI is added and green.

## Evidence to record before tagging

Record the date, platform, compiler, and command output summary for:

- Makefile build and tests.
- CMake build and tests.
- Makefile staged install file list.
- CMake staged install file list.
- Downstream direct-link smoke program.
- Downstream `pkg-config` smoke program.
- Downstream CMake `find_package` smoke program.
- Hosted GitHub Actions run URL.

## Current local evidence from 2026-06-12

Validated on Linux with GCC 13.3.0:

- `make clean && make && make test` passed.
- `make install DESTDIR=/tmp/bfft-stage.VKGfSR PREFIX=/usr` staged headers,
  static library, shared library, and `lib/pkgconfig/bfft.pc`.
- `cmake -S . -B build-cmake -DBFFT_BUILD_PROBES=OFF` passed.
- `cmake --build build-cmake --parallel` passed.
- `ctest --test-dir build-cmake --output-on-failure` passed.
- `cmake --install build-cmake --prefix /tmp/bfft-cmake-stage/usr` staged
  headers, static library, shared library, `bfft.pc`, `BFFTConfig.cmake`,
  `BFFTConfigVersion.cmake`, `BFFTTargets.cmake`, and
  `BFFTTargets-noconfig.cmake`.
- A C smoke program compiled and ran with staged `pkg-config` metadata from
  `/tmp/bfft-stage.VKGfSR/usr/lib/pkgconfig`.
- A CMake downstream project configured, built, linked `bfft::static`, and ran
  with `CMAKE_PREFIX_PATH=/tmp/bfft-cmake-stage/usr`.

Still needed before tagging: hosted GitHub Actions run URL and any final
release-machine validation pass the maintainer wants to treat as canonical.
