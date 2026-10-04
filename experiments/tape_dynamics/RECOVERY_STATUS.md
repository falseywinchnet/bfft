# Recovery snapshot, October 3, 2026

This is a snapshot of the active, uncommitted Tape experiment, taken from
`codex/obligation-dynamics` at base `1ff367fb821b4b98d33be888babaf60b55826f8b`.
Source and fixture copies were checked against SHA-256 before and after copying
between **2026-10-04 04:52:09 and 04:52:13 UTC** (October 3, 11:52 p.m. CDT).
The parent physics task continues independently. No algorithm was altered for
this recovery, and the active originals and their Mini mirror were untouched.

At this cutoff the README referred to `FINDINGS.md`, but that file did not yet
exist. There was no completed, validated final benchmark report. The saved
development scores include rejected approaches, and one archived trajectory
was truncated by disk exhaustion. They are not release qualification or a
claim that the requested real-time/accuracy objective has been met. Some
`results/final` files were being collected; the directory name does not certify
a completed study. Recovery verification is recorded separately in the recovery
index and does not convert development timings into a new performance result.

Compact scores and the exact 1,000-body fixture remain here. Large trajectories
are preserved in checksum-addressed local storage, with every original path,
size, and hash in `recovery/2026-10-03/manifest.json`. To restore them into a
separate directory:

```sh
python3 recovery/2026-10-03/restore_evidence.py \
  --only experiments/tape_dynamics/ --destination /tmp/tape-recovered-evidence
```

The original benchmark commands remain in README. Use distinct build/output
paths from the active physics task, and never run timed comparisons concurrently.
