# BFFT completion idea

BFFT is complete for the first public library release when it is a small,
installable Bruun real FFT library with a coherent C ABI, a lightweight C++
wrapper, tests, examples, docs, and a validated Makefile workflow.

## Product shape

- Power-of-two real transforms for `N >= 4`.
- Standard FFT-order real-to-complex output for everyday users.
- Native spectrum order for users who want the internal heap-optimized layout.
- Residue-domain transform and filtering helpers for pipelines that can avoid
  spectrum permutation.
- Automatic backend and packing policy derived from the build target and plan
  size.
- No public transform-selection compile flags.

## Completion checklist

- Public C declarations in `include/bfft/bfft.h` are documented and tested.
- Public C++ wrappers in `include/bfft/bfft.hpp` are documented and tested.
- Build outputs include static and shared libraries.
- `make clean`, `make`, and `make test` pass from a clean tree.
- `make install DESTDIR=... PREFIX=...` stages the documented files.
- A downstream smoke program can build against the staged install.
- Examples cover the C API, C++ API, and benchmark path without becoming large.
- `README.md`, `docs/api.md`, and `docs/architecture.md` match the shipped API
  and policy.
- Release notes or a release checklist identifies validated platforms and known
  limits.
- GitHub-only repository settings are recorded in `docs/maintainer-notes.md`.
- No placeholder TODO markers remain in source, public docs, examples, or tests
  unless they describe post-release work in the roadmap.

## Out of scope for 0.1.0

- CMake packaging.
- Windows DLL export work.
- macOS install-name tuning.
- Prebuilt release artifacts.
- Multi-platform CI beyond notes or a lightweight first workflow.
