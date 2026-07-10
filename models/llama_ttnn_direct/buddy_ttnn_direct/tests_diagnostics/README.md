# TTNN Direct diagnostics tests

This directory contains legacy validation, smoke, sweep, search, and
component-level bring-up tests. They are intentionally excluded from the
default product test command:

```bash
pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests -q
```

Run the diagnostics coverage explicitly when changing bring-up tooling or
compatibility paths:

```bash
pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests_diagnostics -q
```

The tests use fake or injected TTNN modules and do not claim device-level
correctness or performance.
