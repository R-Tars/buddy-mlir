# TTNN Direct Evidence

Only the compact current summary is tracked in this directory. Runtime and
build code must not import documentation evidence.

Raw reports, latency samples, correctness arrays, generated programs, and
autotune candidates belong under the ignored build tree:

```text
${BUDDY_BUILD}/models/llama31_ttnn_direct/evidence_archive/
```

To update evidence, run the matched correctness and P150A performance gates,
retain their raw outputs in that build directory, then update
`latest_summary.json` with only the final metrics, hashes, source identities,
and raw-artifact location. Do not copy per-run arrays or historical campaign
records back into the source tree.
