from __future__ import annotations

import gc
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

from ..runtime.tokenizer import tokenize_prompt_for_prefill
from .artifacts import kv_cache_snapshot, last_token_vector, tensor_snapshot


def capture_hf_reference(
    *,
    model_path: str | Path,
    tokenizer_path: str | Path | None,
    prompt: str,
    layers: int,
    prefill_len: int,
    dtype: str = "bfloat16",
    torch_module: Any | None = None,
    transformers_module: Any | None = None,
) -> dict[str, Any]:
    if layers <= 0:
        raise ValueError("layers must be positive")
    torch = torch_module or importlib.import_module("torch")
    transformers = transformers_module or importlib.import_module(
        "transformers"
    )
    model_root = Path(model_path)
    tokenization = tokenize_prompt_for_prefill(
        prompt=prompt,
        batch_size=1,
        prefill_len=prefill_len,
        tokenizer_path=tokenizer_path or model_root,
        tokenizer_module=transformers,
    )
    effective_ids = tokenization.token_ids[0][
        : tokenization.effective_token_count
    ]
    input_ids = torch.tensor([effective_ids], dtype=torch.long)

    config = transformers.AutoConfig.from_pretrained(
        str(model_root),
        local_files_only=True,
    )
    available_layers = int(config.num_hidden_layers)
    if layers > available_layers:
        raise ValueError(
            f"layers must be <= HF config num_hidden_layers ({available_layers})"
        )
    config.num_hidden_layers = int(layers)
    torch_dtype = _torch_dtype(torch, dtype)
    model = _load_model(
        transformers=transformers,
        model_root=model_root,
        config=config,
        torch_dtype=torch_dtype,
    )
    model.eval()

    layer_outputs: dict[int, Any] = {}
    hooks = []
    for layer_id, layer in enumerate(model.model.layers):
        hooks.append(
            layer.register_forward_hook(
                _layer_capture_hook(layer_outputs, layer_id)
            )
        )
    final_hidden: dict[str, Any] = {}
    hooks.append(
        model.model.norm.register_forward_hook(
            _final_capture_hook(final_hidden)
        )
    )
    try:
        with torch.inference_mode():
            outputs = _forward_model(model, input_ids)
        snapshots: dict[str, dict[str, Any]] = {}
        for layer_id in range(layers):
            snapshots[f"prefill.layer.{layer_id}.hidden"] = tensor_snapshot(
                f"prefill.layer.{layer_id}.hidden",
                last_token_vector(layer_outputs[layer_id]),
                logical_shape=_shape(layer_outputs[layer_id]),
                sample_policy={
                    "kind": "last_token_vector",
                    "batch_id": 0,
                    "position": len(effective_ids) - 1,
                },
            )
        snapshots["prefill.final_hidden"] = tensor_snapshot(
            "prefill.final_hidden",
            last_token_vector(final_hidden["tensor"]),
            logical_shape=_shape(final_hidden["tensor"]),
            sample_policy={
                "kind": "last_token_vector",
                "batch_id": 0,
                "position": len(effective_ids) - 1,
            },
        )
        logits = last_token_vector(outputs.logits)
        snapshots["prefill.logits"] = tensor_snapshot(
            "prefill.logits",
            logits,
            logical_shape=_shape(outputs.logits),
            sample_policy={
                "kind": "last_token_vector",
                "batch_id": 0,
                "position": len(effective_ids) - 1,
            },
        )
        cache_layers = _cache_layers(outputs.past_key_values)
        for layer_id in range(layers):
            key, value = cache_layers[layer_id]
            snapshots[f"prefill.layer.{layer_id}.key_cache"] = (
                kv_cache_snapshot(
                    f"prefill.layer.{layer_id}.key_cache",
                    key,
                )
            )
            snapshots[f"prefill.layer.{layer_id}.value_cache"] = (
                kv_cache_snapshot(
                    f"prefill.layer.{layer_id}.value_cache",
                    value,
                )
            )
        top_token = int(torch.argmax(logits, dim=-1).item())
        return {
            "schema_version": 1,
            "kind": "hf_llama_correctness_reference",
            "status": "captured",
            "passed": True,
            "model_path": str(model_root),
            "model_config_sha256": _file_sha256(model_root / "config.json"),
            "tokenizer_path": str(tokenizer_path or model_root),
            "prompt_sha256": tokenization.prompt_sha256,
            "layers": int(layers),
            "prefill_len": int(prefill_len),
            "effective_token_count": len(effective_ids),
            "input_token_ids": effective_ids,
            "dtype": dtype,
            "torch_version": str(getattr(torch, "__version__", "unknown")),
            "transformers_version": str(
                getattr(transformers, "__version__", "unknown")
            ),
            "reference_semantics": "truncated_layers_then_final_norm_lm_head",
            "top_token": top_token,
            "checkpoints": snapshots,
        }
    finally:
        for hook in hooks:
            hook.remove()
        del model
        gc.collect()


def write_hf_reference(path: str | Path, report: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


def load_hf_reference(
    path: str | Path,
    *,
    model_path: str | Path,
    prompt: str,
    layers: int,
    prefill_len: int,
    dtype: str,
) -> dict[str, Any]:
    source = Path(path)
    report = json.loads(source.read_text())
    errors: list[str] = []
    if report.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if report.get("kind") != "hf_llama_correctness_reference":
        errors.append("kind must be hf_llama_correctness_reference")
    if report.get("status") != "captured" or report.get("passed") is not True:
        errors.append("reference status must be captured and passed")
    expected_config_digest = _file_sha256(Path(model_path) / "config.json")
    if report.get("model_config_sha256") != expected_config_digest:
        errors.append("model config digest does not match")
    expected_prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if report.get("prompt_sha256") != expected_prompt_digest:
        errors.append("prompt digest does not match")
    if int(report.get("layers", -1)) != int(layers):
        errors.append("layer count does not match")
    if int(report.get("prefill_len", -1)) != int(prefill_len):
        errors.append("prefill length does not match")
    if report.get("dtype") != dtype:
        errors.append("reference dtype does not match")
    checkpoints = report.get("checkpoints")
    expected_names = _expected_checkpoint_names(int(layers))
    if not isinstance(checkpoints, dict):
        errors.append("checkpoints must be a mapping")
    elif set(checkpoints) != expected_names:
        errors.append(
            "checkpoint names do not match the requested layer set"
        )
    else:
        for name, snapshot in checkpoints.items():
            if not _valid_snapshot(snapshot):
                errors.append(f"checkpoint {name} has an invalid snapshot")
    if not isinstance(report.get("input_token_ids"), list) or not report.get(
        "input_token_ids"
    ):
        errors.append("input_token_ids must be non-empty")
    if errors:
        raise ValueError(
            f"invalid HF correctness reference {source}:\n- "
            + "\n- ".join(errors)
        )
    return report


def _load_model(
    *,
    transformers: Any,
    model_root: Path,
    config: Any,
    torch_dtype: Any,
) -> Any:
    kwargs = {
        "config": config,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            str(model_root),
            dtype=torch_dtype,
            **kwargs,
        )
    except TypeError:
        return transformers.AutoModelForCausalLM.from_pretrained(
            str(model_root),
            torch_dtype=torch_dtype,
            **kwargs,
        )


def _forward_model(model: Any, input_ids: Any) -> Any:
    kwargs = {
        "input_ids": input_ids,
        "use_cache": True,
        "return_dict": True,
    }
    try:
        return model(logits_to_keep=1, **kwargs)
    except TypeError:
        return model(**kwargs)


def _layer_capture_hook(storage: dict[int, Any], layer_id: int) -> Any:
    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        storage[layer_id] = output[0] if isinstance(output, tuple) else output

    return capture


def _final_capture_hook(storage: dict[str, Any]) -> Any:
    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        storage["tensor"] = output

    return capture


def _cache_layers(cache: Any) -> list[tuple[Any, Any]]:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return [
            (
                getattr(layer, "keys", getattr(layer, "key", None)),
                getattr(layer, "values", getattr(layer, "value", None)),
            )
            for layer in layers
        ]
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        return list(zip(key_cache, value_cache, strict=True))
    if isinstance(cache, (tuple, list)):
        return [(layer[0], layer[1]) for layer in cache]
    raise TypeError("unsupported Hugging Face KV cache representation")


def _torch_dtype(torch: Any, dtype: str) -> Any:
    aliases = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float32": "float32",
        "fp32": "float32",
    }
    attr = aliases.get(dtype)
    if attr is None or not hasattr(torch, attr):
        raise ValueError(f"unsupported HF reference dtype: {dtype}")
    return getattr(torch, attr)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shape(tensor: Any) -> list[int]:
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    return [int(dim) for dim in getattr(tensor, "shape", ())]


def _expected_checkpoint_names(layers: int) -> set[str]:
    names = {"prefill.final_hidden", "prefill.logits"}
    for layer_id in range(layers):
        names.update(
            {
                f"prefill.layer.{layer_id}.hidden",
                f"prefill.layer.{layer_id}.key_cache",
                f"prefill.layer.{layer_id}.value_cache",
            }
        )
    return names


def _valid_snapshot(snapshot: Any) -> bool:
    if not isinstance(snapshot, dict):
        return False
    values = snapshot.get("values")
    try:
        sample_count = int(snapshot.get("sample_count"))
    except (TypeError, ValueError):
        return False
    return (
        isinstance(values, list)
        and sample_count == len(values)
        and isinstance(snapshot.get("sha256"), str)
        and len(snapshot["sha256"]) == 64
    )
