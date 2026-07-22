"""Compatibility report adapter for the retired decode-only loop diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..runtime.generate import run_generate
from ..runtime.reports import write_report


def run_prompt_decode_loop(
    *,
    out: str | Path,
    program_dir: str | Path,
    model_path: str | Path | None = None,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    decode_steps: int = 2,
    layers: int = 1,
    device: str,
    device_id: int = 0,
    batch_size: int | None = None,
    cache_len: int | None = None,
    dtype_seed: str = "bf16",
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
    tokenizer_module: Any | None = None,
) -> dict[str, Any]:
    """Run the legacy stage through the canonical generate/session pipeline."""

    step_count = int(decode_steps)
    if step_count <= 0:
        raise ValueError("decode_steps must be positive")
    generate = run_generate(
        out=out,
        program_dir=program_dir,
        model_path=model_path,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        max_new_tokens=step_count + 1,
        layers=layers,
        prefill_len=1,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=dry_run,
        ttnn_module=ttnn_module,
        torch_module=torch_module,
        tokenizer_module=tokenizer_module,
        report_level="full",
        runtime_input_mode="recreate",
        execution_mode="eager",
        prefill_execution_mode="eager",
    )
    report = dict(generate)
    report.update(
        command="prompt-decode-loop",
        template="prompt_decode_loop",
        decode_steps=step_count,
        legacy_requested_decode_steps=step_count,
        runtime_owner="TTNNDirectRuntimeContext",
        runtime_session_owner="runtime.generate.run_generate",
        legacy_adapter={
            "replaced": "buddy_ttnn_direct.decode_loop",
            "runtime_api": "runtime.generate.run_generate",
            "execution_mode": "eager/recreate",
            "semantics": "prompt_conditioned_prefill_decode",
            "token_budget_mapping": (
                "canonical max_new_tokens = legacy decode_steps + 1 because "
                "prefill materializes the first generated token"
            ),
        },
    )
    report.setdefault("generated_token_ids", [])
    report.setdefault("generated_text", "")
    report.setdefault("step_reports", [])
    report.setdefault("throughput_summary", {})
    write_report(out, report)
    return report
