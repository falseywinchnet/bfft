# Codex round rules

This file is the local work guide for coding-agent rounds in this repository.
Follow `AGENTS.md` first. If these rules and `AGENTS.md` appear to disagree,
use the stricter rule.

## Round contract

Each round should complete one coherent, reviewable task, then stop.

At the start of each round:

1. Read `AGENTS.md`, `CODEX.md`, `IDEA.md`, `ROADMAP.md`, `PROGRESS.md`,
   `TASKS.md`, the public headers, build files, tests, examples, and docs that
   are relevant to the task.
2. Check recent commits with `git log --oneline -n 20` so status files are not
   updated from stale assumptions.
3. Select one unchecked task from `TASKS.md`, or refresh the queue if it is
   stale.
4. Keep the task small enough to validate in the same round.

During the round:

1. Keep public declarations in `include/bfft/`.
2. Keep implementation details in `src/` or `src/detail/`.
3. Keep examples short and copy-paste friendly.
4. Update docs and tests when behavior, API, install paths, examples, package
   metadata, or policy changes.
5. Do not rewrite unrelated working code.
6. Do not introduce transform-selection compile flags in the public API.
7. For CI changes, keep the first required workflow understandable to a new
   GitHub Actions user before adding a large platform matrix.

Validation:

1. For code, tests, public API, build, install, CI, or package metadata changes,
   run:

   ```sh
   make clean
   make
   make test
   ```

2. If install behavior changes, also run a staged install with `DESTDIR`.
3. If CMake behavior changes, run CMake configure, build, and CTest.
4. For docs-only or planning-only changes, run the smallest meaningful check and
   record what was not run.
5. If validation fails and the fix is obvious, make one focused fix attempt.
6. If validation still fails, document the blocker and stop.

At the end of each round:

1. Mark the selected task complete in `TASKS.md`, or leave it unchecked with a
   blocker note.
2. Keep three to six sensible future tasks in `TASKS.md` until beta work is
   complete.
3. Update `PROGRESS.md` with what changed, what validation ran, and what
   remains.
4. Update `ROADMAP.md` only when the actual plan changes.
5. Commit the completed changes when the session instructions require it.

## Completion signal

Keep this exact line in `PROGRESS.md` while work remains:

```text
Agent loop status: active
```

Change it to this exact line only when the completion checklist in `IDEA.md` is
satisfied:

```text
Agent loop status: complete
```
