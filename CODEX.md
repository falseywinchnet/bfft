# Codex unattended round rules

This file controls unattended rounds started by `scripts/agent-loop.sh`.
Follow `AGENTS.md` first. If these rules and `AGENTS.md` appear to disagree,
use the stricter rule.

## Round contract

Each Codex invocation is one round. Complete exactly one useful task, then stop.

At the start of each round:

1. Read `AGENTS.md`, `CODEX.md`, `IDEA.md`, `ROADMAP.md`, `PROGRESS.md`,
   `TASKS.md`, the public headers, the Makefile, tests, examples, and docs that
   are relevant to the next task.
2. Evaluate the repository against `IDEA.md`.
3. Select one unchecked task from `TASKS.md`, or add one new task if the queue
   is empty or stale.
4. Keep the task small enough to finish and validate in this round.

During the round:

1. Implement only the selected task.
2. Keep public declarations in `include/bfft/`.
3. Keep implementation details in `src/` or `src/detail/`.
4. Keep examples short and copy-paste friendly.
5. Update docs and tests when behavior, API, install paths, examples, or policy
   changes.
6. Do not rewrite unrelated working code.
7. Do not introduce transform-selection compile flags in the public API.

Validation:

1. For code, tests, public API, build, or install changes, run:

   ```sh
   make clean
   make
   make test
   ```

2. If install behavior changes, also run a staged install with `DESTDIR`.
3. For docs-only or planning-only changes, run the smallest meaningful
   validation command and record what was not run.
4. If validation fails and the fix is obvious, make one focused fix attempt.
5. If validation still fails, document the blocker and stop.

At the end of each round:

1. Mark the selected task complete in `TASKS.md`, or leave it unchecked with a
   blocker note.
2. Keep three to six sensible future tasks in `TASKS.md` until the project is
   complete.
3. Update `PROGRESS.md` with what changed, what validation ran, and what
   remains.
4. Update `ROADMAP.md` only when the actual plan changes.
5. Commit all round changes with a concise message after validation passes.
6. Stop after that single task. Do not begin another task.

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
