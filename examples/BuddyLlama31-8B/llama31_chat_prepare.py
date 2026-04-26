# ===- deepseek_chat_prepare.py -----------------------------------------------
#
# Prepare the host-side artifacts required by ``deepseek_chat_run.py`` (the
# interactive 1024-token-context chat runner that drives P150A with both
# ``deepseek_prefill_static.ttnn`` and ``deepseek_decode_static.ttnn``
# flatbuffers).
#
# We re-run the Buddy ``DynamoCompiler`` for prefill and decode under the same
# configuration as ``buddy-deepseek-r1-lower-ttir.py --static-cache
# --max-cache-len 1024``, then snapshot for each phase:
#
#   - ``slot_roles.json``: per-``@subgraph0``-input role tag.  Roles include
#     ``"weight"`` (host-side tensor baked into the model), ``"input_ids"``,
#     ``"cache_position"``, ``"past_K"``/``"past_V"``, ``"inv_freq"``.
#   - ``weights.npz``: one array per weight slot, keyed ``"w_%04d" % slot``.
#   - ``inv_freq.npy``: F32 rotary base (if present).
#   - ``shapes.json`` / ``dtypes.json``: per-arg metadata for sanity-check.
#
# The chat runner only needs to load these ``.npz`` / ``.json`` artifacts and
# swap in the runtime tensors (tokens, cache position, past KV) per step.
#
# Usage::
#
#   python deepseek_chat_prepare.py -o chat_artifacts --max-cache-len 1024
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import PlaceholderOp
from buddy.compiler.graph.transform import (
    simply_fuse,
    flash_attention_prefill,
    gqa_attention_fusion,
)
from buddy.compiler.ops import tosa


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Prepare prefill+decode artifacts for the interactive DeepSeek chat runner."
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get(
            "LLAMA31_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct"
        ),
    )
    p.add_argument(
        "--max-cache-len",
        type=int,
        default=1024,
        help="Static cache length (must match lowering).",
    )
    p.add_argument(
        "-o",
        "--artifacts",
        type=Path,
        default=here / "chat_artifacts",
    )
    p.add_argument("--use-proxy", action="store_true")
    p.add_argument(
        "--phases",
        default="prefill,decode",
        help="Comma-separated list of phases to prepare (default prefill,decode).",
    )
    return p.parse_args()


def _proxy_env():
    u = "http://192.168.15.159:7890"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.setdefault(k, u)


def _patch_static_cache_for_buddy() -> None:
    """Same monkey-patch as the lowering script.

    Replaces ``StaticLayer.update`` with a ``where``-based scatter so we do not
    hit ``aten.index_copy_`` (unregistered in the Buddy TOSA op map).
    """
    import transformers.cache_utils as cu

    def update(self, key_states, value_states, cache_kwargs=None):
        if self.keys is None:
            self.lazy_initialization(key_states)
        cache_position = (
            cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        )
        if cache_position is None:
            cache_position = torch.arange(key_states.shape[-2], device=self.device)

        L = self.keys.shape[-2]
        S = key_states.shape[-2]

        if S == L:
            self.keys = key_states
            self.values = value_states
            return self.keys, self.values

        idx = torch.arange(L, device=self.keys.device).view(1, L)
        pos2d = cache_position.view(-1, 1)
        mask = idx == pos2d
        mask = mask.view(1, 1, L, 1)
        B, H, _, D = self.keys.shape
        ks_exp = key_states.expand(B, H, L, D)
        vs_exp = value_states.expand(B, H, L, D)
        self.keys = torch.where(mask, ks_exp, self.keys)
        self.values = torch.where(mask, vs_exp, self.values)
        return self.keys, self.values

    cu.StaticLayer.update = update


def _load_model(model_id: str):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    m.eval()
    return m


def _torch_dtype_str(t: torch.Tensor) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float16: "float16",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.bool: "bool",
    }.get(t.dtype, str(t.dtype))


def _tensor_to_numpy_and_dtype(t: torch.Tensor) -> tuple[np.ndarray, str]:
    """Serialize tensor to numpy in a reversible way.  bf16 is stored as uint16
    bits so numpy can own the buffer; the runner view-casts back."""
    tcpu = t.detach().cpu().contiguous()
    if tcpu.dtype == torch.bfloat16:
        return tcpu.view(torch.uint16).numpy().astype(np.uint16), "bfloat16"
    if tcpu.dtype == torch.float32:
        return tcpu.numpy().astype(np.float32), "float32"
    if tcpu.dtype == torch.float16:
        return tcpu.view(torch.uint16).numpy().astype(np.uint16), "float16"
    if tcpu.dtype == torch.int64:
        return tcpu.numpy().astype(np.int64), "int64"
    if tcpu.dtype == torch.int32:
        return tcpu.numpy().astype(np.int32), "int32"
    if tcpu.dtype == torch.bool:
        return tcpu.numpy().astype(np.uint8), "bool"
    raise ValueError(f"unsupported dtype {tcpu.dtype}")


def _fuse_list():
    return [simply_fuse, gqa_attention_fusion, flash_attention_prefill]


def _kv_shape_from_config(model, max_cache_len: int) -> list[int]:
    """Static past_K / past_V placeholder shape = [1, n_kv, L, head_dim].

    DeepSeek-R1-Distill-Qwen-1.5B has n_kv=2 / head_dim=128; Llama-3.1-8B has
    n_kv=8 / head_dim=128. Reading them off the HF config keeps this script
    model-agnostic.
    """
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )
    return [1, int(n_kv), int(max_cache_len), int(head_dim)]


def _prepare_phase(
    phase: str,
    model,
    out_dir: Path,
    max_cache_len: int,
) -> None:
    from transformers import StaticCache

    L = int(max_cache_len)
    device = next(model.parameters()).device
    dynamo_compiler = DynamoCompiler(primary_registry=tosa.ops_registry)
    kv_shape = _kv_shape_from_config(model, L)

    if phase == "prefill":
        input_ids = torch.zeros(1, L, dtype=torch.long, device=device)
        with torch.no_grad():
            graphs = dynamo_compiler.importer(
                model,
                input_ids=input_ids,
                use_cache=True,
                cache_implementation="static",
            )
    elif phase == "decode":
        past_kv = StaticCache(config=model.config, max_cache_len=L)
        decode_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        cache_position = torch.tensor([200], dtype=torch.long, device=device)
        with torch.no_grad():
            model(
                input_ids=decode_ids,
                past_key_values=past_kv,
                use_cache=True,
                cache_implementation="static",
            )
            graphs = dynamo_compiler.importer(
                model,
                input_ids=decode_ids,
                use_cache=True,
                cache_position=cache_position,
                past_key_values=past_kv,
                cache_implementation="static",
            )
    else:
        raise ValueError(f"unknown phase {phase!r}")

    if len(graphs) != 1:
        raise RuntimeError(f"expected 1 graph, got {len(graphs)} for {phase}")
    g = graphs[0]

    params_ref = getattr(g, "_params_ref", None)
    runtime_ref = getattr(g, "_runtime_inputs_ref", None)
    if params_ref is None:
        raise RuntimeError("graph missing _params_ref (Buddy import)")
    if runtime_ref is None:
        raise RuntimeError("graph missing _runtime_inputs_ref (update Buddy frontend)")

    g.fuse_ops(_fuse_list())
    driver = GraphDriver(g)
    if len(driver.subgraphs) != 1:
        raise RuntimeError(
            f"expected 1 subgraph for {phase}, got {len(driver.subgraphs)}"
        )
    sg_name = list(driver._subgraphs.keys())[0]
    sg_input_names = driver._subgraphs_inputs[sg_name]

    fake_ix_list = list(g._fake_params)
    inp_ix_list = list(g._inputs)
    fake_ix_set = set(int(x) for x in fake_ix_list)
    inp_ix_set = set(int(x) for x in inp_ix_list)

    name_to_tensor: dict[str, torch.Tensor] = {}
    name_to_is_input: dict[str, bool] = {}
    for fi, pidx in enumerate(fake_ix_list):
        ph_op = g.body[pidx]
        if fi < len(params_ref):
            name_to_tensor[ph_op.name] = params_ref[fi]
            name_to_is_input[ph_op.name] = False
    for ii, iidx in enumerate(inp_ix_list):
        ph_op = g.body[iidx]
        if ii < len(runtime_ref):
            name_to_tensor[ph_op.name] = runtime_ref[ii]
            name_to_is_input[ph_op.name] = True

    slot_roles: list[dict] = []
    weights: dict[str, np.ndarray] = {}
    shapes: list[list[int]] = []
    dtypes: list[str] = []

    kv_seen = 0
    inv_freq_arr: np.ndarray | None = None

    for slot, ph_name in enumerate(sg_input_names):
        if ph_name not in name_to_tensor:
            raise RuntimeError(
                f"{phase}: subgraph input slot {slot} ({ph_name}) has no tensor"
            )
        t = name_to_tensor[ph_name]
        is_input = name_to_is_input[ph_name]
        shp = list(t.shape)
        dt = _torch_dtype_str(t)
        shapes.append(shp)
        dtypes.append(dt)

        role: str = "unknown"
        if not is_input:
            role = "weight"
            arr, _ = _tensor_to_numpy_and_dtype(t)
            weights[f"w_{slot:04d}"] = arr
        else:
            if dt == "int64" and shp == [1, L]:
                role = "input_ids"
            elif dt == "int64" and shp == [1, 1]:
                role = "input_ids"
            elif dt == "int64" and shp == [1]:
                role = "cache_position"
            elif dt == "bfloat16" and shp == kv_shape:
                if kv_seen % 2 == 0:
                    role = f"past_K_{kv_seen // 2:02d}"
                else:
                    role = f"past_V_{kv_seen // 2:02d}"
                kv_seen += 1
            elif dt == "float32" and shp == [64]:
                role = "inv_freq"
                arr, _ = _tensor_to_numpy_and_dtype(t)
                inv_freq_arr = arr.astype(np.float32)
            else:
                role = f"runtime:{dt}:{shp}"

        slot_roles.append(
            {
                "slot": slot,
                "placeholder": ph_name,
                "role": role,
                "shape": shp,
                "dtype": dt,
            }
        )

    phase_dir = out_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "slot_roles.json").write_text(
        json.dumps(slot_roles, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (phase_dir / "shapes.json").write_text(json.dumps(shapes), encoding="utf-8")
    (phase_dir / "dtypes.json").write_text(json.dumps(dtypes), encoding="utf-8")
    if weights:
        np.savez(phase_dir / "weights.npz", **weights)
    if inv_freq_arr is not None:
        np.save(phase_dir / "inv_freq.npy", inv_freq_arr)
    summary = {
        "phase": phase,
        "max_cache_len": L,
        "num_slots": len(slot_roles),
        "num_weight_slots": sum(1 for r in slot_roles if r["role"] == "weight"),
        "num_runtime_slots": sum(1 for r in slot_roles if r["role"] != "weight"),
        "kv_seen": kv_seen,
        "has_inv_freq": inv_freq_arr is not None,
    }
    (phase_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[OK] {phase}: {summary}")


def main() -> int:
    args = _parse_args()
    if args.use_proxy:
        _proxy_env()
    args.artifacts.mkdir(parents=True, exist_ok=True)

    _patch_static_cache_for_buddy()
    try:
        model = _load_model(args.model)
    except Exception as e:
        print(f"error: load model: {e}", file=sys.stderr)
        return 1

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for phase in phases:
        _prepare_phase(phase, model, args.artifacts, args.max_cache_len)

    print(f"[OK] chat artifacts -> {args.artifacts.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
