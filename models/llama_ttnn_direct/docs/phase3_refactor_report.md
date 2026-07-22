# TTNN Direct Phase 3 Refactor Report

Status: Phase 3 software and P150A gates passed.

Baseline: 238 tracked files / 113132 lines. Final: 236 files / 111930 lines,
net deletion 1202 lines. Implementation/test delta before this report is
+596 / -1845 lines; no `llvm` change is included.

Deleted pure facades: `generate.py`, `runtime_inputs.py`, `validation.py`,
`codegen/python_ttnn.py`. Deleted the 989-line `decode_loop.py` and 431-line
`profile_template.py` implementations.

Canonical imports now target `runtime.generate`, `runtime.profile`,
`runtime.prefill_profile`, `runtime.inputs`, `runtime.tokenizer`,
`reports.validation`, `compiler.codegen`, `compiler.config`, and
`compiler.source_templates`.

`decode-loop-legacy` is an 83-line adapter over `runtime.generate.run_generate`;
it preserves prompt-conditioned prefill/decode and the required legacy fields.
`template-profile` is a 119-line adapter over `run_smoke_mlp`,
`MeasurementContract`, and `summarize_samples`; it does not duplicate device,
trace, statistics, or correctness helpers.

Two diagnostic stages (`decode-shell`, `decode-step-profile`) preserve isolated
workflow capabilities. Internal subprocesses use only the six product commands.
The CMake package target now calls the canonical package API instead of the
deleted `package-program` root command.

Ownership AST test: pass; no unexpected product -> diagnostics/autotune,
product -> correctness, CLI -> implementation, or autotune -> diagnostics edges.
The five documented historical dependency debts remain frozen and unchanged.

Report field migration is recorded in `legacy_decode_loop_migration.json`;
redundant legacy top-level shape/timing fields were removed, while canonical
nested runtime and plan evidence is retained. No formal product runtime math,
precision recipe, config, trace path, or protected compiler template changed.

Software verification: compileall pass; product 274 passed / 712 subtests;
diagnostics 240 passed / 17 subtests; CMake `llama31_ttnn_direct_program` pass;
fresh seven-artifact bundle, all product dry-runs, required diagnostics dry-runs,
and generated-runner compatibility dry-runs pass.

P150A 5x100x3 median is 35.577902 t/s/u with 0.0340% CV and 0.0549%
regression; all trace/persistent-input checks pass. Full-depth generation,
99-check all-BF16 PCC (minimum 0.992598), and 500-token quality pass. Evidence
is under `phase3_after`. Next phase: physically separate autotune core
and diagnostics after manual review; no Phase 4 work was started.
