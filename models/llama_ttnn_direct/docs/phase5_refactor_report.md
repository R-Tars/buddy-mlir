# TTNN Direct Phase 5 Refactor Report

Status: canonical semantic autotune ownership convergence and all regression
gates passed.

## Scope

Phase 5 started from commit
`ffbfdcc58d98feeadea9c828ffb9c3c3db1b191b`. It makes
`buddy_ttnn_direct/autotune` the only owner of tuning semantics, changes
`diagnose --stage autotune` into a lazy dispatch to that owner, and removes the
second tuner from `diagnostics/autotune`. It does not change model math,
precision recipes, the product runtime hot path, the frozen best config, or
generated model behavior. No tuning campaign or new winner was produced, and
Phase 6 has not started. The unrelated dirty `llvm` path is excluded.

## Before Audit

The removed diagnostics tuner contained four files and 1,085 tracked lines:

| File | Lines |
| --- | ---: |
| `diagnostics/autotune/candidate.py` | 432 |
| `diagnostics/autotune/runner.py` | 499 |
| `diagnostics/autotune/selection.py` | 143 |
| `diagnostics/autotune/__init__.py` | 11 |

Its production reachability was limited to the lazy autotune branch in
`diagnostics/cli.py`. The remaining consumers were its own package exports,
the old diagnostics autotune test, and documentation. `check_device_ownership`
had no external consumer; the selection helpers were private to the package
and its test; and `candidate.py` had no product or canonical-autotune importer.

The duplicate stack hard-coded three progressive axes: LM-head split 8/16,
official L1 versus LM-head DRAM concat, and official program config versus an
SDPA 8x4 grid. It also independently implemented candidate identity,
subprocess profiling, resume, winner selection, and promotion. These semantics
already had richer canonical implementations under `autotune`.

## Canonical Entry Point

The new `autotune/campaign.py` exposes `run_autotune_campaign`. It is a
449-line composition layer, not a new search algorithm. It directly composes:

- schema-v2 `SearchSpaceConfig` and canonical candidate fingerprints;
- frozen `PrecisionContract`, `ExecutionContract`, and `MeasurementContract`;
- existing template, MatMul, and SDPA enumeration and legality;
- `run_active_measurement_scheduler` and successive halving;
- `run_hierarchical_search` with the six canonical pipeline stages;
- `ConfirmationPolicy` and matched A/B confirmation.

`autotune/model_evaluator.py` is a 247-line generic whole-model evaluator. It
materializes a schema-v2 candidate, writes a generated program bundle, invokes
the product `profile --mode decode-steady` command in an isolated subprocess,
loads its report, classifies failures, and resumes by canonical fingerprint.
It does not contain legacy candidate slugs, the three old axes, selection
logic, promotion math, or another search loop.

The final dispatch is:

```text
product cli.py
  -> diagnostics/cli.py (stage registration and argument mapping)
  -> autotune/campaign.py (all tuning orchestration and semantics)
```

The import is lazy inside the `stage == "autotune"` branch. Product `cli.py`
still imports only diagnostics registration/dispatch helpers and has no direct
autotune dependency.

## Removal And Boundary

All four `diagnostics/autotune` files and all 1,085 lines were deleted; the
directory no longer exists. The old `AUTOTUNE_LEVELS`, `run_layered_autotune`,
`_select_winner`, and `_confirmation_promotion_decision` implementations are
absent from Python source. The old three-level tests were replaced with tests
for canonical dry-run identity, schema-v2 candidate identity, frozen
contracts, active and hierarchical composition, canonical confirmation,
failure persistence, resume, and CLI preflight.

The ownership guard now requires:

```text
diagnostics/autotune directory: absent
autotune -> diagnostics imports: 0
diagnostics -> autotune imports: diagnostics/cli.py -> autotune/campaign.py only
retired layered-tuner symbols in Python source: 0
```

The stale Phase 4 `reports/autotune.py` architecture reference was removed in
the first Phase 5 commit. Architecture and command documentation now state the
same ownership and dispatch contract.

## Size

Canonical autotune grew from 26 files / 22,315 lines to 28 files / 23,011
lines. The two new files add 696 lines, while deleting the 1,085-line duplicate
tuner yields a net implementation reduction of 389 lines. This satisfies the
Phase 5 implementation target while keeping the campaign at or below 450 lines
and the optional evaluator at or below 250 lines. No existing canonical
algorithm file changed.

Including this report, the full tracked model tree is 227 files / 108,242
lines, compared with 228 files / 108,285 lines at Phase 5 start: one file and
43 lines removed overall. Tests, composition, and ownership documentation add
coverage around the larger implementation deletion. Against the Phase 1
baseline of 253 files / 119,177 lines, the repository has cumulatively removed
26 files and 10,935 tracked lines.

## CLI And Bundle Validation

`diagnose --stage autotune --dry-run` is prompt-free and device-free. Its
fresh report records schema v2, `hierarchical_constrained_beam_search`, all six
pipeline stages, frozen precision/execution contracts, canonical search
budgets and confirmation policy, and `cartesian_exhaustive_search=false`. A
non-dry-run invocation without a prompt fails during preflight before opening
a device.

The six root commands remain `build`, `generate`, `profile`, `validate`,
`inspect`, and `diagnose`. The fresh generated bundle contains all seven
required artifacts. Its `run_decode.py` is 140 lines, compiles, and passes
generate, profile, validate, inspect, diagnose, and generated-runner dry-runs.

## Software Regression

| Check | Result |
| --- | --- |
| `compileall -q models/llama_ttnn_direct` | Pass |
| Product tests | 277 passed, 712 subtests passed |
| Diagnostics tests | 227 passed, 27 subtests passed |
| Canonical autotune tests | 143 passed, 677 subtests passed |
| Import-boundary tests | 15 passed, 35 subtests passed |
| `git diff --check` | Pass |

## Best Config Freeze

The frozen winner remains
`autotune/doc7_phase10_final/isolated_linears/compatible_gate_up_down_lm/program`.
Its before/after identities are identical:

```text
config:             1ed82e5b0f2f2ce8444fa881ce725f62a37d6f556fa2cee493cda6dc7e5fde5d
precision contract: 4c8d2ca23c0d6dc7fd1f07a9b96680ee534dc3e4a30b37c9fe1fe5501e41ff11
execution contract: b1cbda955efffc534afead92b7267c4875fbcedba45e8db0ae5a5c9faae96b8f
template choices:   8ee3b05ec01ef8c0854b60119c1a02f71a448379821e90aef4bb010e98416cc9
program configs:    c87dbd92cd9a8d9877dac9ff2c338f837375186211ec2e6a5dffa16716cbd5d8
memory configs:     5ccad7b894b69f4a3ba8f4973d89b798700b64326a94e24e6434bab1d0337c22
SDPA config:        ba9c0498d380fcc70bc97b2417c39bf5ed26cafc617d6545eb45cd4d08d776c2
```

Precision, trace execution, persistent inputs, after-prefill timing, fixed
page tables, and force-argmax sampling are unchanged.

## Correctness

Fresh full-depth generation passed with 32 layers, prompt-conditioned prefill,
prefill-populated KV cache, generated text `!\nWe`, and user-zero token IDs
`[4999, 1687]`.

Fresh All-BF16 validation passed 99 comparisons and 98 numeric PCC checks.
Minimum PCC was `0.992598` against the `0.99` threshold, and observed/reference
top token 4999 matched. The fresh 500-token performance-recipe run also passed:
official top-1/top-5 was `0.91/0.98`, Buddy aggregate and minimum-user
top-1/top-5 were `0.91/0.982`, and minimum-user greedy agreement was `0.964`.

## P150A Regression

The fresh runs use the same official-repro batch32 prompt corpus and contract
as Phase 4: prefill 256, cache 1024, eager prefill, five warmups, 100 measured
iterations, three independent repetitions, full decode trace, persistent
inputs, after-prefill timing, and force argmax.

| Metric | Phase 4 | Phase 5 | Change |
| --- | ---: | ---: | ---: |
| Median tokens/s/user | 35.560901 | 35.583117 | +0.06247% |
| Median p50 decode latency | 28.079519 ms | 28.054182 ms | -0.025337 ms |
| Throughput CV | 0.05616% | 0.01825% | Pass |

Raw throughput was `35.583117`, `35.575845`, and `35.591729` tokens/s/user.
Every run retained four trace captures, 105 replays, seven persistent inputs,
and zero program compilations after capture. The <=1% regression and <=1.5%
CV gates passed.

Two earlier full-depth attempts coincided with an external NoC/DRAM benchmark
and failed with TT-UMD bus errors. Their logs are preserved separately. After
the competing process released the device, `tt-smi` showed healthy P150A DRAM,
PCIe Gen5 x16, firmware 19.5.0, and no uncorrected GDDR errors; a fresh third
full-depth run and every subsequent correctness/performance run passed.

Machine-readable before/after inventories, identities, ownership audits, CLI
contracts, raw correctness reports, and raw performance reports are under
`build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase5_*`.
