from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..smoke_decode_shell import _to_namespace
from ..smoke_single_layer_decode import _load_generated_model
from .state import build_generate_state


def build_runtime_session(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    decode_plan: dict[str, Any],
    prefill_plan: dict[str, Any],
    program_dir: Path,
    model_path: Path,
    prompt: str,
    tokenizer_path: str | Path,
    tokenizer_module: Any | None,
    config: dict[str, Any],
    layer_count: int,
    batch_size: int,
    cache_len: int,
    prefill_len: int,
    observer: Any | None = None,
) -> SimpleNamespace:
    """Materialize one reusable model/runtime context for prefill and decode."""

    context = build_generate_state(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
        decode_plan=decode_plan,
        prefill_plan=prefill_plan,
        program_dir=program_dir,
        model_path=model_path,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        tokenizer_module=tokenizer_module,
    )
    generated = _load_generated_model(program_dir / "model.py", ttnn)
    runtime_config = dict(config)
    runtime_config["num_layers"] = layer_count
    runtime_config["batch_size"] = batch_size
    runtime_config["max_cache_len"] = cache_len
    runtime_config["seq_len"] = 1
    runtime_config["prefill"] = dict(runtime_config.get("prefill") or {})
    runtime_config["prefill"]["seq_len"] = prefill_len
    model = generated.BuddyLlama31TTNN(
        device=device,
        parameters=context.parameters,
        config=_to_namespace(runtime_config),
        observer=observer,
    )
    context.install_generated_model(
        generated_module=generated,
        generated_model=model,
    )
    return SimpleNamespace(
        context=context,
        generated_module=generated,
        generated_model=model,
        config=runtime_config,
    )
