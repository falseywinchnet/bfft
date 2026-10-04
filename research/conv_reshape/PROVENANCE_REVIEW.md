# Independent CONV* source-version review

Verified September 17, 2026. **No source-version discrepancy was found in the six named references.** Rainstar Paint's documented CONV* research snapshot matches the specified MacBook Neo archive and the local research mirror. Its composed paper is also byte-identical to the currently published `papers_please/main` file. This is an independent provenance review; the separate implementation review owns mathematical and C++ correctness findings.

The parent task narrowed the original research assignment after Rainstar Paint 0.1.2 shipped. No alternative sampler, new C++ module, application edit, release change, remote write, or paid-model call was made here. Only this isolated task's audit artifacts were written.

The application checkout inspected was `/Users/joshuahkuttenkuler/Developer/Projects/rainstar-paint`, at commit `9ed0c775d242398798e678087f0fd47df38b507b`. I read `docs/WARP_MATH.md` and `docs/CONV_PROVENANCE.md`, and separately fetched the published `docs/WARP_MATH.md`. The published document contains the same paper commit and all four hashes that it explicitly names below.

The current `papers_please/main` head was independently read from GitHub:

- Commit: `a43e178f053b9f968c18e8f14d8489a949d04120`.
- Commit timestamp: `2026-09-14T23:44:05Z`.
- Tree: `17992b672bb873b0a693e440dfe37867871ddc5f`.
- Paper: [conv_paper_composed.tex at the verified commit](https://github.com/falseywinchnet/papers_please/blob/a43e178f053b9f968c18e8f14d8489a949d04120/conv_paper_composed.tex).
- Paper size: **166,102 bytes**.
- Paper SHA-256: `5b89a42e9f6ac5a7cb8b6ca46ef223f3a2f2a9afa3d408ac60bf5ac9244baed4`.
- Paper Git blob SHA-1: `7e2b7d3b1821ba5f5a60e92fcb4d49d0e2926232`, independently recomputed with the Git blob header and matched to the published tree entry.

The head commit removes the separately published `convstar_warp_addendum.tex`; this is not evidence that the addendum's mathematics is absent. The current composed paper contains the finite-bank warped-pixel section, including the geometry-selected positive projective quadrature construction. The composed paper is the publication artifact to use for this source-version comparison. Do not retrieve the deleted standalone file from an older commit and treat that older revision as the newest addendum.

For all rows in the table, the mirror base is:

`/Users/joshuahkuttenkuler/Developer/CodexBuilds/bfft-6b3e7ffa7539/`

The archive is:

`/Users/joshuahkuttenkuler/CodexEmergencyBackups/20260916T015732Z-MacBook-Neo/home-bfft.tar.gz`

Each archive member is `Users/ultimussecundai/bfft/` followed by the relative path below. The audit streams the gzip tar and reads only these named file payloads; it does not extract or modify the backup. Equality is tested on complete byte strings in addition to hashing.

| Relative source path | Bytes | SHA-256 in both archive and mirror |
| --- | ---: | --- |
| `output/pdf/conv_paper_composed.tex` | 166102 | `5b89a42e9f6ac5a7cb8b6ca46ef223f3a2f2a9afa3d408ac60bf5ac9244baed4` |
| `output/pdf/convstar_warp_addendum.tex` | 24607 | `87e6cf5ad6fbf11cbcfbb2b626478929de279054a0c1e027194c3ded988cfdd2` |
| `output/pdf/convstar_warp_measurements.tex` | 2736 | `ae1230a9a87a35e9b6342708c69ddf08a0914713608d1a303502a6d844891d96` |
| `experiments/conv_warp/compact_measurement/reference_source.c` | 19923 | `de1f14b64495da678354cba7717e4c7c729ed7b075877619ed16f2ea3dd84b53` |
| `experiments/conv_warp/joint_reference.py` | 47624 | `88f034a06b00939679dbecf1cc440b90e0952300c13829186a175dfdfca64860` |
| `experiments/conv_warp/PRELIMINARY_THEORY.md` | 7532 | `838a497037657cc700679328b80b5bdb8ec0e81d9da1034e44ca2fe25956affe` |

All six archive/mirror comparisons passed. The public paper also passed an exact comparison with the mirror; therefore it matches the archive transitively and by identical SHA-256. No newer published main-branch revision was found. This verdict is bounded to the named archive and current published main branch; it does not assert the absence of unpublished work elsewhere.

The documentation correctly separates the newer `warp.cpp` provenance in `WARP_MATH.md` from the earlier website-derived resize module in `CONV_PROVENANCE.md`. The latter is not the sole source description for the lasso sampler. The former explicitly identifies compact β* rather than the global metric β variant, and discloses bounded Gauss pixel integration rather than the paper's exact clipped-cell/certified projective moments. Those are documented implementation choices, not a source-version mismatch. This review does not independently validate their C++ realization.

The machine-readable evidence is `source_receipt.json`. `published_head.json` records the independently fetched publication head. `published_main.tex` retains the exact published paper bytes, whose Git blob identity is verified by the audit script. Reproduce the archive/mirror/paper comparison from this worktree with:

```sh
python3 research/conv_reshape/audit_sources.py
```

That command verifies the saved publication snapshot and the current local inputs; it does not refresh GitHub. A future current-version claim requires fetching GitHub head/tree/file again. The script asserts all six files are present, all archive/mirror byte comparisons pass, and the saved publication file matches both the mirror and the verified Git blob identity. It writes only `source_receipt.json` next to itself.
