# TTNN Direct Phase 13 Autotune Surface And Evidence Diet

## Status

```text
PHASE_13_STATUS=ACCEPTED
PHASE_14_STARTED=false
```

Phase 13 reduces the autotune package-root facade and one duplicate artifact
hash helper. Campaign, search, measurement, ranking, winner, diagnostics,
runtime, compiler, codegen, precision, and execution behavior are unchanged.

## Public Surface

`autotune/__init__.py` fell from 579 to 14 lines and from 266 exports to five:

```text
PrecisionContract
atomic_write_json
build_final_campaign_report
dry_run_template
list_template_definitions
```

The before inventory classified the removed exports as 152 `TEST_ONLY` and
109 `LEGACY_PHASE_API`; two documented and three reproducibility-required
exports remain. No `UNKNOWN` export existed or was removed. All 25 checked
source/test package-root imports were migrated to their real owner modules;
the after count is zero. A compact import-boundary test freezes this surface.

`final_campaign.py` is `KEEP_REQUIRED`. The executable ignored report script
`autotune/doc7_phase10_final/prepare_final_report.py` still imports
`build_final_campaign_report`, `PrecisionContract`, and `atomic_write_json` to
rebuild `phase10_acceptance.json` and `PAPER_SUMMARY.md`. Its dedicated test
therefore remains and imports the real owner directly. Other ignored phase
scripts are historical campaign artifacts, not checked current APIs.

## Artifact Plumbing

`artifact.py` removed its private SHA-256 implementation and now consumes
`autotune.measurement.file_sha256`. No third hash implementation and no
`artifact_support.py` were added. Baseline and paper JSON readers, atomic
writers, manifests, and verification errors remain local: their serialization
or exception/cleanup contracts are not fully equivalent.

Fresh rebuilds used the same inputs and separate after directories. Baseline
is 6/6 byte-identical. Paper output is 8/8 byte-identical, including
`artifact_manifest.json` and `reproduce.sh`; CLI module paths and reproduce
commands are unchanged. The historical paper bundle still lacks the
pre-existing source `p150a_goal7_correctness_performance_evidence_20260714.json`,
so a complete fixed fixture provides the direct byte comparison.

## Frozen Product

The canonical autotune dry-run preserves schema, proposal enumeration,
pipeline stages/status, precision hash, execution contract, search structure,
fingerprint, and `run_identity=semantic-3c098a1f252b9c356465`. All 17 frozen
campaign/search files and every recorded Phase 12.1 profiler, Phase 10.1
provenance, Phase 11 depth, runtime, compiler, and codegen file hash match.

A fresh build from the canonical fused-winner config produced seven artifacts
byte-identical to Phase 12.1. `run_decode.py` remains 140 lines, compiles, and
dry-run validation passes. No campaign ran and no winner was promoted.
Therefore hardware was not rerun and the inherited formal evidence remains:

```text
35.8723317007 tokens/s/user; p50 28.0515274952 ms; CV 0.0211981%
top1/top5 0.910/0.982; minimum greedy agreement 0.964
Phase 12.1 fused profiler: 708 ops; ratio 0.9952410885; p50 28.121555 ms
```

## Regression And Size

```text
Product:     283 passed, 718 subtests
Diagnostics: 175 passed, 46 subtests
Focused:     258 passed, 686 subtests
```

Before canonical LOC was 80,392. Source changes add 12/remove 585 (`-573`);
test changes add 175/remove 92 (`+83`). This report adds 90 lines, making the
final canonical LOC 79,992 and the total Phase 13 net deletion 400 lines.
Both KEEP_REQUIRED gates are met: source deletion is at least 350 and total
deletion is at least 400. Cumulative reduction is 39,185 lines from 119,177
(`32.880%`), leaving 44,239 lines to the 35,753 target.

Machine evidence is under `build-tenstorrent/models/llama31_ttnn_direct/
ttnn_direct_refactor/phase13_before` and `phase13_after`. Phase 14 was not
started.
