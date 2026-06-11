# BFFT first-release checklist

## Completed in this repository setup

- Split the prototype into public headers, source, examples, tests, and docs.
- Removed the embedded benchmark `main` from the library kernel.
- Added `make`, `make test`, and `make install` workflows.
- Added C and C++ APIs.
- Added correctness and API tests.
- Added automatic backend and packing policy.
- Added repository instructions for future coding agents.
- Added maintainer notes for GitHub-only repository settings.

## Before tagging 0.1.0

- Run `make clean && make && make test`.
- Run a staged install with `DESTDIR`.
- Build and run a small downstream program against the staged install.
- Review `docs/maintainer-notes.md` and apply GitHub settings.
- Create release notes that call out Linux as the validated platform for this
  first release.
