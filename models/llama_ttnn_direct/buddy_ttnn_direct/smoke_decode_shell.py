from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .codegen.parameters import (
    ParameterMaterializationError,
    load_llama_parameters_from_manifests,
)
from .codegen.ttnn_tensorizer import (
    TTNNTensorizationError,
    load_parameter_config_from_program,
)
from .smoke_mlp import (
    NO_TTNN_DEVICE_MESSAGE,
    NoTTNNDeviceError,
    _managed_ttnn_device,
)


DECODE_SHELL_OPS = [
    "embedding",
    "rms_norm.mlp",
    "linear.mlp_gate",
    "linear.mlp_up",
    "mul.silu",
    "linear.mlp_down",
    "residual_add",
    "rms_norm.final",
    "split_lm_head",
    "argmax_or_sampling",
]


def run_smoke_decode_shell(
    *,
    out: str | Path,
    program_dir: str | Path,
    layers: int,
    disable_attention: bool,
    device: str,
    device_id: int = 0,
    model_path: str | Path | None = None,
    dry_run: bool = False,
    ttnn_module: Any | None = None,
    parameters: Any | None = None,
    token_ids: Any | None = None,
) -> dict[str, Any]:
    if layers <= 0:
        raise ValueError("layers must be positive")
    if not disable_attention:
        raise ValueError(
            "smoke-decode-shell PR-D supports --disable-attention only"
        )

    program_root = Path(program_dir)
    config = json.loads((program_root / "config.json").read_text())
    layer_count = min(layers, int(config["num_layers"]))
    if dry_run:
        report = _base_report(
            program_dir=program_root,
            device=device,
            device_id=device_id,
            layers=layer_count,
            disable_attention=disable_attention,
            dry_run=True,
        )
        report.update(
            {
                "passed": True,
                "status": "dry_run",
                "latency_ms": 0.0,
                "message": "Dry run only; TTNN device is not required.",
            }
        )
        _write_report(out, report)
        return report

    try:
        ttnn = (
            ttnn_module
            if ttnn_module is not None
            else importlib.import_module("ttnn")
        )
    except ImportError as err:
        report = _no_device_report(
            program_dir=program_root,
            device=device,
            device_id=device_id,
            layers=layer_count,
            disable_attention=disable_attention,
            detail=str(err),
        )
        _write_report(out, report)
        return report

    try:
        with _maybe_managed_device(ttnn, device_id, ttnn_module) as ttnn_device:
            shell_params = parameters
            if shell_params is None:
                if model_path is None:
                    raise ValueError(
                        "--model-path is required for non-dry-run decode shell"
                    )
                host_params = load_llama_parameters_from_manifests(
                    model_path=model_path,
                    weights_manifest=program_root / "weights_manifest.json",
                    config=program_root / "config.json",
                    tensor_backend="torch",
                    layers=range(layer_count),
                )
                shell_params = _tensorize_shell_parameters(
                    ttnn,
                    host_params,
                    load_parameter_config_from_program(program_root),
                    ttnn_device,
                    layer_count,
                )
            report = _run_generated_decode_shell(
                ttnn=ttnn,
                program_dir=program_root,
                parameters=shell_params,
                config=config,
                device=ttnn_device,
                layer_count=layer_count,
                token_ids=token_ids,
            )
            report.update(
                {
                    **_base_report(
                        program_dir=program_root,
                        device=device,
                        device_id=device_id,
                        layers=layer_count,
                        disable_attention=disable_attention,
                        dry_run=False,
                    ),
                    **report,
                }
            )
    except NoTTNNDeviceError as err:
        report = _no_device_report(
            program_dir=program_root,
            device=device,
            device_id=device_id,
            layers=layer_count,
            disable_attention=disable_attention,
            detail=str(err),
        )
    except (
        ParameterMaterializationError,
        TTNNTensorizationError,
        ValueError,
    ) as err:
        report = _failed_report(
            program_dir=program_root,
            device=device,
            device_id=device_id,
            layers=layer_count,
            disable_attention=disable_attention,
            status="configuration_error",
            message=str(err),
        )
    except Exception as err:
        report = _failed_report(
            program_dir=program_root,
            device=device,
            device_id=device_id,
            layers=layer_count,
            disable_attention=disable_attention,
            status="runtime_error",
            message=f"{type(err).__name__}: {err}",
        )

    _write_report(out, report)
    return report


def _run_generated_decode_shell(
    *,
    ttnn: Any,
    program_dir: Path,
    parameters: Any,
    config: dict[str, Any],
    device: Any,
    layer_count: int,
    token_ids: Any | None,
) -> dict[str, Any]:
    generated = _load_generated_model(program_dir / "model.py", ttnn)
    model = generated.BuddyLlama31TTNN(
        device=device,
        parameters=parameters,
        config=_to_namespace(config),
    )
    token_ids = "token_ids" if token_ids is None else token_ids

    start = time.perf_counter()
    hidden = model.embed(token_ids)
    layer_reports = []
    for layer_id in range(layer_count):
        residual = hidden
        normalized = model.rmsnorm(hidden, layer_id, kind="mlp")
        mlp_out = model.mlp_decode(layer_id, normalized)
        hidden = model.ops.add(residual, mlp_out)
        layer_reports.append(
            {
                "layer_id": layer_id,
                "attention": "disabled",
                "output_shape": _shape(hidden),
                "output_dtype": _dtype(hidden),
            }
        )
    hidden = model.final_norm(hidden)
    token = model.lm_head_argmax(hidden)
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        "passed": True,
        "status": "passed",
        "latency_ms": latency_ms,
        "layers": layer_reports,
        "output": {
            "shape": _shape(token),
            "dtype": _dtype(token),
            "repr": repr(token),
        },
        "reference": {
            "status": "not_run",
            "reason": "PR-D validates generated shell execution and shapes only.",
        },
    }


def _tensorize_shell_parameters(
    ttnn: Any,
    host_params: Any,
    parameter_config: dict[str, Any],
    device: Any,
    layer_count: int,
) -> SimpleNamespace:
    weights = parameter_config["weights"]

    def convert(tensor: Any, role: str, layer_id: int | None) -> Any:
        entry = _find_config_entry(weights, role, layer_id)
        return ttnn.from_torch(
            tensor,
            device=device,
            dtype=_resolve_ttnn_dtype(ttnn, str(entry["target_dtype"])),
            layout=_resolve_ttnn_layout(ttnn, str(entry["layout"])),
        )

    layers = [None] * len(host_params.layers)
    for layer_id in range(layer_count):
        layer = host_params.layers[layer_id]
        layers[layer_id] = SimpleNamespace(
            input_norm=SimpleNamespace(
                weight=convert(layer.input_norm.weight, "input_norm", layer_id)
            ),
            post_attention_norm=SimpleNamespace(
                weight=convert(
                    layer.post_attention_norm.weight,
                    "post_attention_norm",
                    layer_id,
                )
            ),
            mlp=SimpleNamespace(
                gate_proj=SimpleNamespace(
                    weight=convert(layer.mlp.gate_proj.weight, "mlp_gate", layer_id)
                ),
                up_proj=SimpleNamespace(
                    weight=convert(layer.mlp.up_proj.weight, "mlp_up", layer_id)
                ),
                down_proj=SimpleNamespace(
                    weight=convert(layer.mlp.down_proj.weight, "mlp_down", layer_id)
                ),
            ),
        )

    lm_entry = _find_config_entry(weights, "lm_head", None)
    lm_splits = []
    for split in host_params.lm_head.splits:
        lm_splits.append(
            SimpleNamespace(
                shard_id=split.shard_id,
                vocab_start=split.vocab_start,
                vocab_end=split.vocab_end,
                weight=ttnn.from_torch(
                    split.weight,
                    device=device,
                    dtype=_resolve_ttnn_dtype(ttnn, str(lm_entry["target_dtype"])),
                    layout=_resolve_ttnn_layout(ttnn, str(lm_entry["layout"])),
                ),
            )
        )

    return SimpleNamespace(
        embedding=SimpleNamespace(
            weight=convert(host_params.embedding.weight, "embedding", None)
        ),
        layers=layers,
        final_norm=SimpleNamespace(
            weight=convert(host_params.final_norm.weight, "final_norm", None)
        ),
        lm_head=SimpleNamespace(splits=lm_splits),
    )


def _find_config_entry(
    weights: dict[str, Any],
    role: str,
    layer_id: int | None,
) -> dict[str, Any]:
    matches = [
        entry
        for entry in weights.values()
        if isinstance(entry, dict)
        and entry.get("role") == role
        and entry.get("layer_id") == layer_id
    ]
    if len(matches) != 1:
        raise TTNNTensorizationError(
            f"expected one config entry for role={role}, layer_id={layer_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


@contextmanager
def _maybe_managed_device(ttnn: Any, device_id: int, injected_ttnn: Any | None):
    if injected_ttnn is not None:
        yield f"fake_device:{device_id}"
        return
    with _managed_ttnn_device(ttnn, device_id) as device:
        yield device


def _load_generated_model(path: Path, ttnn: Any):
    module_name = "generated_buddy_ttnn_decode_shell"
    old_ttnn = sys.modules.get("ttnn")
    sys.modules["ttnn"] = ttnn
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load generated model: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
        return module
    finally:
        if old_ttnn is None:
            sys.modules.pop("ttnn", None)
        else:
            sys.modules["ttnn"] = old_ttnn


def _resolve_ttnn_dtype(ttnn: Any, target_dtype: str) -> Any:
    return getattr(ttnn, target_dtype, target_dtype)


def _resolve_ttnn_layout(ttnn: Any, layout: str) -> Any:
    if layout == "tile":
        return getattr(ttnn, "TILE_LAYOUT", layout)
    if layout == "row_major":
        return getattr(ttnn, "ROW_MAJOR_LAYOUT", layout)
    return getattr(ttnn, layout, layout)


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _shape(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _dtype(tensor: Any) -> str | None:
    dtype = getattr(tensor, "dtype", None)
    return str(dtype) if dtype is not None else None


def _base_report(
    *,
    program_dir: Path,
    device: str,
    device_id: int,
    layers: int,
    disable_attention: bool,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "template": "decode_shell",
        "program_dir": str(program_dir),
        "device": device,
        "device_id": device_id,
        "layers_requested": layers,
        "disable_attention": disable_attention,
        "dry_run": dry_run,
        "ttnn_ops": list(DECODE_SHELL_OPS),
    }


def _no_device_report(
    *,
    program_dir: Path,
    device: str,
    device_id: int,
    layers: int,
    disable_attention: bool,
    detail: str,
) -> dict[str, Any]:
    report = _base_report(
        program_dir=program_dir,
        device=device,
        device_id=device_id,
        layers=layers,
        disable_attention=disable_attention,
        dry_run=False,
    )
    report.update(
        {
            "passed": False,
            "status": "no_device",
            "error": NO_TTNN_DEVICE_MESSAGE,
            "detail": detail,
            "latency_ms": None,
        }
    )
    return report


def _failed_report(
    *,
    program_dir: Path,
    device: str,
    device_id: int,
    layers: int,
    disable_attention: bool,
    status: str,
    message: str,
) -> dict[str, Any]:
    report = _base_report(
        program_dir=program_dir,
        device=device,
        device_id=device_id,
        layers=layers,
        disable_attention=disable_attention,
        dry_run=False,
    )
    report.update(
        {
            "passed": False,
            "status": status,
            "error": message,
            "latency_ms": None,
        }
    )
    return report


def _write_report(out: str | Path, report: dict[str, Any]) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
