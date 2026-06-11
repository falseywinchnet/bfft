#!/usr/bin/env bash
set -euo pipefail

MAX_ROUNDS="${1:-25}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
CODEX_BIN="${CODEX_BIN:-codex}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

case "$MAX_ROUNDS" in
    ''|*[!0-9]*)
        echo "MAX_ROUNDS must be a positive integer."
        exit 2
        ;;
esac

if [ "$MAX_ROUNDS" -lt 1 ]; then
    echo "No rounds requested."
    exit 0
fi

GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ "$GIT_ROOT" != "$REPO_ROOT" ]; then
    echo "Refusing to run outside the standalone bfft Git repository."
    echo "Expected: $REPO_ROOT"
    echo "Actual: ${GIT_ROOT:-not a Git worktree}"
    exit 1
fi

if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
    echo "Could not find Codex executable: $CODEX_BIN"
    exit 1
fi

for required_file in AGENTS.md CODEX.md IDEA.md ROADMAP.md PROGRESS.md TASKS.md; do
    if [ ! -f "$required_file" ]; then
        echo "Missing required loop file: $required_file"
        exit 1
    fi
done

PROMPT='Read AGENTS.md, CODEX.md, IDEA.md, ROADMAP.md, PROGRESS.md, and TASKS.md. Operate for exactly one unattended round.

Goal:
Advance this repository toward the complete BFFT library described in IDEA.md.

Round procedure:
1. Inspect the planning files, public headers, Makefile, tests, examples, and docs relevant to the next task.
2. Evaluate progress against IDEA.md.
3. Select or create exactly one next useful task.
4. Complete that task.
5. Run relevant validation, following AGENTS.md and CODEX.md.
6. Update TASKS.md, PROGRESS.md, and ROADMAP.md as needed.
7. Leave the worktree changes unstaged. The supervisor script will commit them.
8. Stop.

Do not wait for user input.
Do not ask questions.
Do not perform a second task in this round.
Do not run git add or git commit.
If blocked, document the blocker in PROGRESS.md and stop.'

CODEX_ARGS=(--ask-for-approval never --sandbox workspace-write --cd "$REPO_ROOT")
if [ -n "${CODEX_MODEL:-}" ]; then
    CODEX_ARGS=(-m "$CODEX_MODEL" "${CODEX_ARGS[@]}")
fi

for round in $(seq 1 "$MAX_ROUNDS"); do
    echo "=== Agent round $round / $MAX_ROUNDS ==="

    if grep -qi '^Agent loop status:[[:space:]]*complete[[:space:]]*$' PROGRESS.md; then
        echo "Progress file reports completion. Stopping."
        exit 0
    fi

    if [ "$ALLOW_DIRTY" != "1" ] && [ -n "$(git status --porcelain)" ]; then
        echo "Worktree has uncommitted changes. Commit or stash them before running unattended rounds."
        git status --short
        exit 1
    fi

    before="$(git rev-parse HEAD)"

    "$CODEX_BIN" "${CODEX_ARGS[@]}" exec "$PROMPT"

    if [ -n "$(git status --porcelain)" ]; then
        git diff --check
        git add -A
        git commit -m "Advance BFFT release readiness"
    fi

    after="$(git rev-parse HEAD)"

    if [ "$before" = "$after" ]; then
        echo "No commit was created this round. Stopping."
        git status --short
        exit 0
    fi

    if grep -qi '^Agent loop status:[[:space:]]*complete[[:space:]]*$' PROGRESS.md; then
        echo "Progress file reports completion. Stopping."
        exit 0
    fi

    sleep "$SLEEP_SECONDS"
done
