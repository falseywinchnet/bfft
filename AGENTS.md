# AGENTS.md

This repository is the source tree for BFFT, a small Bruun real FFT library.

## Scope
These instructions apply to the whole repository unless a more specific `AGENTS.md` appears in a subdirectory.

## Project rules
- Keep public API declarations in `include/bfft/` and implementation details in `src/` or `src/detail/`.
- Do not reintroduce user-facing transform-selection compile flags. Backend and packing policy should be automatic from the build and from plan/output size.
- Prefer explicit control flow and descriptive names. Do not use ternary operators in new code when ordinary `if`/`else` is clearer.
- Keep examples small and suitable for copy/paste into user applications.
- Integrate new tests into `make test`.
- Update documentation when behavior, install paths, APIs, or policies change.
- If GitHub repository appearance/settings cannot be changed from the working tree, record the needed action in `docs/maintainer-notes.md`.

## Validation
Before committing code changes, run at minimum:

```sh
make clean
make
make test
```

If install behavior changes, also validate staged installation with `DESTDIR`.
