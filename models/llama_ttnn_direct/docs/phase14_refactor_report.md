# TTNN Direct Phase 14 Refactor Report

```text
PHASE_14_STATUS=ACCEPTED
PHASE_15_STARTED=false
branch=ttnn-direct-phase1
phase13_base=80b827b
```

Phase 14 retires unreachable autotune phase-report plumbing while preserving
the canonical campaign, supported standalone algorithms, reproducibility
outputs, and the frozen product winner. No device campaign or winner promotion
was run.

## Accounting Reconciliation

The Phase 12.1 report's `80,318` total and the replayed `80,392` total differ
only in report accounting. The committed Phase 12.1 report is 221 lines. The
old total implies a 147-line report because its code-only subtotal was 80,171:

```text
221 - 147 = 74
```

The clean tree replay is therefore:

```text
Phase 12.1 commit cf830231: 80,392
Phase 13 before / commit 80b827b: 80,392
Phase 13 HEAD: 79,992
```

The frozen Phase 1 baseline remains 119,177 lines. Machine evidence is in
`build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase14_before/`.

## Test Count Reconciliation

The `284/176` values in the Phase 14 input document are stale summaries. Clean
collection with the exact command below produced:

```text
Phase 12.1 product: 282 nodes
Phase 13 product:   283 nodes
Phase 14 product:   274 nodes
Phase 12.1 diagnostics: 175 nodes
Phase 13 diagnostics:   175 nodes
Phase 14 diagnostics:   175 nodes
```

Phase 13 added exactly one product node, the public autotune-root boundary
test. Phase 14 removes eight layout-campaign test nodes, two prefetch
phase-promotion nodes, and one SDPA weighted-report node, while adding two
replacement algorithm-contract nodes. Thus `283 - 11 + 2 = 274`.

Unified collection/regression command:

```bash
python -m pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests -q
python -m pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests_diagnostics -q
```

The actual Phase 14 regression used
`/wafer/zhuxinye/miniconda3/envs/buddy-ttmlir/bin/python` and passed:

```text
product:     274 passed, 718 subtests
diagnostics: 175 passed, 46 subtests
focused:     246 passed, 686 subtests
```

The focused selection covers campaign/search/confirmation, measurement and
workers, retained autotune modules, artifact/baseline, product validation,
and import boundaries. Compileall and `git diff --check` also passed.

## Ignored Reproducibility Script

The active ignored producer is:

```text
build-tenstorrent/models/llama31_ttnn_direct/autotune/doc7_phase10_final/prepare_final_report.py
```

It is untracked and ignored, with SHA-256:

```text
fd689292bbd7d7fa072b01beafe60e378091a672264809149783fe0ee6af7428
```

It is manually maintained under the ignored Phase 10 build tree; no tracked
generator produces it. It consumes frozen evidence and produces:

```text
phase10_acceptance.json
PAPER_SUMMARY.md
```

Its Buddy imports now target the real owners directly:

```python
models.llama_ttnn_direct.buddy_ttnn_direct.autotune.final_campaign
models.llama_ttnn_direct.buddy_ttnn_direct.autotune.schema
models.llama_ttnn_direct.buddy_ttnn_direct.autotune.search
```

It has no package-root imports. The script remains a real `REPRO_ONLY` caller,
so `final_campaign.py` and its dedicated test are retained.

## Autotune Root Surface

The root surface changed from five exports to two:

```text
before:
  PrecisionContract
  atomic_write_json
  build_final_campaign_report
  dry_run_template
  list_template_definitions

after:
  dry_run_template
  list_template_definitions
```

The three aliases used only by the ignored producer were removed after its
direct-import migration. No `UNKNOWN` or documented export was removed.

## Reachability Decisions

| Module | Status | Decision |
| --- | --- | --- |
| `final_campaign` | `REPRO_ONLY` | keep report producer |
| `generalization` | `STANDALONE_SUPPORTED` | keep |
| `fused_attention` | `STANDALONE_SUPPORTED` | keep algorithm, remove phase report |
| `packed_mlp` | `STANDALONE_SUPPORTED` | keep algorithm, remove phase report |
| `sdpa_buckets` | `STANDALONE_SUPPORTED` | keep algorithm, remove phase report |
| `layout_campaign` | `RETIRED_PHASE_ONLY` | delete whole module |
| `prefetch` | `STANDALONE_SUPPORTED` | keep algorithm, remove phase report |
| `ranking` | `STANDALONE_SUPPORTED` | keep algorithm, remove phase report |
| `transfer` | `CURRENT_PIPELINE` | keep |

`layout_campaign.py` had no current source, CLI, paper, repro, documented, or
unknown callers. Its remaining references were dedicated Phase 7 tests,
historical doc7 artifacts, and phase-report surfaces. The supported layout
owner remains `layout_graph.py`.

Mixed modules retain enumeration, legality, ranking, selection, application,
measurement/audit, and standalone report contracts. Removed phase-only
symbols are:

```text
fused_attention.build_fused_attention_phase_report
packed_mlp.FULL_MODEL_PROMOTION_GAIN
packed_mlp.build_packed_gate_up_phase_report
sdpa_buckets.SDPA_BUCKET_PROMOTION_GAIN
sdpa_buckets.build_sdpa_bucket_phase_report
prefetch.build_prefetch_phase_report
ranking.build_ranking_phase_report
layout_campaign.build_phase7_acceptance_report
```

The two private weighted-latency helpers were removed only with their retired
report owners. No generic report framework or compatibility facade was added.

## Frozen Product And Capability Evidence

The following identities are byte-identical to Phase 13:

```text
campaign/search/confirmation/model evaluator
schema/measurement/microbench/hardware workers
legality/space/templates/matmul/sdpa/layout graph
runtime/compiler/codegen
artifact/baseline and their CLIs
Phase 10.1 provenance
Phase 11 depth
Phase 12.1 profiler
```

The canonical autotune dry-run preserves proposal counts, candidate IDs,
pipeline stages/status, precision and execution contracts, search structure,
and `run_identity=semantic-3c098a1f252b9c356465`.

The fresh frozen-template build was:

```bash
/wafer/zhuxinye/miniconda3/envs/buddy-ttmlir/bin/python \
  -m models.llama_ttnn_direct.buddy_ttnn_direct.cli build \
  --model-path /wafer/share/models/Llama-3.1-8B-Instruct \
  --config build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase8_after/canonical_template_config.json \
  --out-dir build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase14_after/program
```

The effective template configuration is the fused context SDPA winner recorded
under the doc7 candidate tree. The seven generated files are byte-identical:

```text
model.py
config.json
semantic_graph.json
execution_plan.json
weights_manifest.json
run_decode.py
README.md
```

`run_decode.py` remains 140 lines and compiles. `winner_promoted=false` and
`autotune_campaign_run=false`. The formal inherited hardware result is:

```text
35.8723317007 tokens/s/user
decode p50: 28.0515274952 ms
CV: 0.0211981%
top1/top5: 0.910/0.982
minimum greedy agreement: 0.964
```

Because campaign/search/measurement/workers, product bundle, profiler, and
winner identities are unchanged, no P150A rerun is required for this phase.

## LOC Accounting

Before the Phase 14 report itself, canonical tracked LOC is 78,399:

```text
Phase 13 -> Phase 14 code/test change: -1,593 lines
source changes: +1 / -1,182
test changes: +12 / -424
report: +254 / -0
final Phase 14 net: -1,339 lines
final canonical LOC: 78,653
source net: -1,181 lines
test net: -412 lines
documentation net: +254 lines
```

Against the frozen 119,177-line baseline, this is a cumulative reduction of
40,524 lines (`34.003%`). The remaining distance to the 35,753-line target is
42,900 lines. The mixed-module safe threshold is met: source deletion is over
800 lines and total tracked deletion is over 1,000 lines.

No supported algorithm was removed to chase LOC.

## Stop Decision

All required Phase 14 evidence is under:

```text
build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase14_before/
build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase14_after/
```

No unknown, documented, current, paper, or reproducibility capability was
deleted. Phase 15 is explicitly not started.
