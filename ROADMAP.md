# BFFT roadmap

Current target: prepare BFFT for a first `0.1.0` public release.

## Phase 1: baseline validation

- Run the clean build and test workflow.
- Fix small build or test failures if found.
- Record the exact validation result in `PROGRESS.md`.

## Phase 2: install and downstream smoke

- Validate staged installation with `DESTDIR` and `PREFIX`.
- Build a tiny downstream C or C++ program against the staged install.
- Fix mismatches between installed paths and documentation.

## Phase 3: API and docs audit

- Compare public headers with `docs/api.md` and `README.md`.
- Confirm examples use current allocation sizes and scratch buffers.
- Add focused tests only where public behavior is not covered.

## Phase 4: release readiness

- Update `docs/release-checklist.md` with final validation evidence.
- Confirm `docs/maintainer-notes.md` covers GitHub-only setup.
- Mark `PROGRESS.md` complete only after the checklist in `IDEA.md` is true.

## Later work

- Add CMake support.
- Add CI for Linux, macOS, Windows, scalar fallback, AVX-class builds, and NEON.
- Add packaging metadata after the initial source release is stable.
