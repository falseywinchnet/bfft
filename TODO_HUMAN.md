# TODO for the human repository operator

This file is a small manual for the next beta-prep round. It assumes you have
little or no GitHub Actions experience.

## 1. Turn on and inspect CI

1. Push this branch to GitHub and open a pull request.
2. Open the repository on GitHub.
3. Click the **Actions** tab.
4. If GitHub asks whether to enable workflows, click the button to enable them.
5. Click the workflow named **ci**.
6. Confirm two jobs run on Ubuntu:
   - **make (ubuntu-latest)**: runs `make clean && make`, `make test`, and a
     staged Makefile install.
   - **cmake (ubuntu-latest)**: configures, builds, tests, and stages a CMake
     install with optional probes disabled.
7. If either job fails, open the failed job, expand the red step, and copy the
   last 50 to 100 log lines into the next development issue or prompt.

## 2. Use CI before merging

For each pull request before beta:

1. Wait for the **ci** workflow to finish.
2. Merge only if both jobs are green.
3. If you intentionally merge a red run, record why in the release notes or in
   `docs/maintainer-notes.md` so future you knows it was not accidental.

## 3. Add basic branch protection after the first green CI run

GitHub only allows you to require checks after those checks have run at least
once on the repository.

1. Go to **Settings** -> **Branches**.
2. Add a branch protection rule for `main`.
3. Enable **Require a pull request before merging**.
4. Enable **Require status checks to pass before merging**.
5. Select these required checks:
   - `make (ubuntu-latest)`
   - `cmake (ubuntu-latest)`
6. Optional but recommended: enable **Require branches to be up to date before
   merging** once the project has more frequent contributors.

## 4. Add a README badge after CI is green

After the workflow has run on GitHub, add this near the top of `README.md` if
you want a status badge:

```md
[![ci](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
```

Replace `OWNER/REPO` with the real GitHub owner and repository name.

## 5. Package metadata smoke checks

The repository now installs package metadata for two common consumers:

- `pkg-config`: `bfft.pc`
- CMake packages: `BFFTConfig.cmake`, `BFFTConfigVersion.cmake`, and
  `BFFTTargets.cmake`

For the next beta round, ask the coding agent to add downstream smoke tests for
both of these paths. The human goal is simple: users should be able to find BFFT
with either `pkg-config --libs bfft` or `find_package(BFFT CONFIG REQUIRED)`
after installation.

## 6. Beta release gate

Before tagging a beta, make sure all of these are true:

- GitHub CI is green on `main`.
- `docs/release-checklist.md` has fresh local validation evidence.
- `docs/maintainer-notes.md` no longer lists CI as future-only work.
- A downstream smoke program has been tested against a staged install.
- Known unsupported platforms are called out plainly in the release notes.
