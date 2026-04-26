# ===- buddy-deepseek-r1-lower-ttir.py ------------------------------------------
#
# DeepSeek-R1-Distill-Qwen-1.5B (or similar) → Buddy Graph → ``lower_to_ttir()``
# (bf16 TTIR). Requires ``ttmlir`` on ``PYTHONPATH`` and a HuggingFace model
# checkout (or set ``DEEPSEEK_MODEL_PATH`` to a local directory).
#
# Prefill vs decode:
#   - ``--mode prefill`` traces one forward with a short prompt (causal mask).
#   - ``--mode decode`` runs a prefill pass to build ``past_key_values``, then
#     traces **one decode step** with ``past_key_values`` + ``use_cache=True``
#     (same pattern as ``export_golden_logits.py``). Fusion order is
#     ``gqa_attention_fusion`` **before** ``flash_attention_prefill`` so GQA can
#     match SDPA+KV-cache. ``GQAAttentionFusedOp`` still needs
#     ``kwargs['cur_pos_tensor']`` → use ``--cur-pos-placeholder`` or the
#     heuristic (name contains ``pos`` / ``cache`` / ``cache_position``).
#
# Usage::
#
#   export PYTHONPATH=$BUDDY_BUILD/python_packages:$PYTHONPATH
#   export PYTHONPATH=$TTMLIR_BUILD/python_packages:$PYTHONPATH
#   python buddy-deepseek-r1-lower-ttir.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#   python buddy-deepseek-r1-lower-ttir.py --mode decode --seq 32
#   python export_golden_logits.py --out-dir golden_out
#   python compare_golden_bundle.py --golden-dir golden_out --device-dir ttrt_out
#
# 网络访问 HuggingFace 失败时：在 shell 中执行 ``proxy_on``（见 ``~/.bashrc``），
# 或运行本脚本时加 ``--use-proxy``（与 ``proxy_on`` 使用相同代理地址）。
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import GQAAttentionFusedOp, PlaceholderOp
from buddy.compiler.graph.ttir_import import (
    append_ttir_forward_bf16_f32_packed_i64_runtime,
)
from buddy.compiler.graph.transform import (
    simply_fuse,
    flash_attention_prefill,
    gqa_attention_fusion,
)
from buddy.compiler.ops import tosa


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Lower DeepSeek-R1-Distill-Qwen subgraph to TTIR MLIR (bf16)."
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("DEEPSEEK_MODEL_PATH", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        help="HF model id or local path.",
    )
    p.add_argument(
        "--mode",
        choices=("prefill", "decode"),
        default="prefill",
        help="Graph to trace: prefill (prompt) or single decode step.",
    )
    p.add_argument(
        "--seq",
        type=int,
        default=32,
        help="Sequence length for prefill trace (default 32).",
    )
    p.add_argument(
        "--cur-pos-placeholder",
        type=str,
        default=None,
        help=(
            "For --mode decode: Buddy name of the cache position input "
            "(e.g. arg for current index). Patched into GQAAttentionFusedOp.kwargs['cur_pos_tensor']."
        ),
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=here / "ttir_out",
        help="Output directory for MLIR.",
    )
    p.add_argument(
        "--element-dtype",
        choices=("bf16", "f32"),
        default="bf16",
        help="TTIR element type for lower_to_ttir (default bf16).",
    )
    p.add_argument(
        "--ttmlir-opt",
        type=str,
        default=None,
        help="Path to ttmlir-opt for a quick parse check.",
    )
    p.add_argument(
        "--use-proxy",
        action="store_true",
        help=(
            "Set http(s)_proxy to the same URL as interactive ``proxy_on`` "
            "(http://192.168.15.159:7890) for HuggingFace downloads."
        ),
    )
    p.add_argument(
        "--packed-forward",
        action="store_true",
        help=(
            "After lowering, append ``@forward`` that packs bf16/f32 weights into "
            "two 1-D tensors and passes i64 inputs (e.g. token ids) through, then "
            "calls ``@subgraph0`` (same role as LeNet ``--packed-forward``)."
        ),
    )
    p.add_argument(
        "--prefill-use-cache",
        action="store_true",
        help=(
            "For --mode prefill, trace the model with use_cache=True so the "
            "flatbuffer also returns past_key_values (makes prefill chainable "
            "into the decode flatbuffer)."
        ),
    )
    p.add_argument(
        "--static-cache",
        action="store_true",
        help=(
            "Use HuggingFace StaticCache with fixed max_cache_len (see --max-cache-len). "
            "Prefill traces input_ids.shape=[1,max_cache_len] (pad short prompts at host "
            "side), decode traces past_key_values=[1,2,max_cache_len,head_dim] + "
            "cache_position. Mirrors the reference f32 example (import-deepseek-r1.py)."
        ),
    )
    p.add_argument(
        "--max-cache-len",
        type=int,
        default=1024,
        help="Static cache length (default 1024, used together with --static-cache).",
    )
    return p.parse_args()


def _patch_static_cache_for_buddy() -> None:
    """Replace ``transformers.cache_utils.StaticLayer.update`` so it does not
    emit ``aten.index_copy_`` (which the Buddy TOSA op map does not register).

    Equivalent scatter-via-``where`` implementation:

    - **Prefill** (``key_states.shape[-2] == max_cache_len``): directly assign
      key/value states as the new cache contents; attention uses the full 1024
      in any case.
    - **Decode** (``key_states.shape[-2] == 1``): build a boolean mask
      ``arange(max_cache_len) == cache_position[0]`` and ``torch.where`` the
      new 1-token state into the corresponding slot.
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


def _load_model_and_tokenizer(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    m.eval()
    return m, tok


def _patch_gqa_cur_pos(graph, name: str) -> None:
    if not name:
        return
    for op in graph.body:
        if isinstance(op, GQAAttentionFusedOp):
            op.kwargs["cur_pos_tensor"] = name


def _decode_forward_kwargs(model, past, prefill_len: int, decode_ids: torch.Tensor) -> dict:
    """Keyword arguments for a single decode forward (align with golden export)."""
    device = decode_ids.device
    kw: dict = {
        "past_key_values": past,
        "use_cache": True,
    }
    try:
        sig = inspect.signature(model.forward)
        if "cache_position" in sig.parameters:
            kw["cache_position"] = torch.tensor(
                [prefill_len], dtype=torch.long, device=device
            )
    except (TypeError, ValueError):
        pass
    return kw


def _fusion_list():
    # GQA must run before flash_attention_prefill so SDPA+KV-cache is still
    # ScaledDotProductFlashAttentionForCpuOp when gqa_attention_fusion runs.
    return [simply_fuse, gqa_attention_fusion, flash_attention_prefill]


def main() -> int:
    args = _parse_args()
    if args.use_proxy:
        u = "http://192.168.15.159:7890"
        os.environ.setdefault("http_proxy", u)
        os.environ.setdefault("https_proxy", u)
        os.environ.setdefault("HTTP_PROXY", u)
        os.environ.setdefault("HTTPS_PROXY", u)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, tokenizer = _load_model_and_tokenizer(args.model)
    except Exception as e:
        print(
            f"error: could not load model {args.model!r} ({e}). "
            "Install transformers and set DEEPSEEK_MODEL_PATH or --model.",
            file=sys.stderr,
        )
        return 1

    device = next(model.parameters()).device
    dynamo_compiler = DynamoCompiler(primary_registry=tosa.ops_registry)

    if args.static_cache:
        _patch_static_cache_for_buddy()
        from transformers import StaticCache

        L = int(args.max_cache_len)
        if args.mode == "prefill":
            input_ids = torch.zeros(1, L, dtype=torch.long, device=device)
            with torch.no_grad():
                graphs = dynamo_compiler.importer(
                    model,
                    input_ids=input_ids,
                    use_cache=True,
                    cache_implementation="static",
                )
        else:
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
    elif args.mode == "prefill":
        input_ids = torch.ones(1, args.seq, dtype=torch.long, device=device)
        with torch.no_grad():
            if args.prefill_use_cache:
                graphs = dynamo_compiler.importer(model, input_ids, use_cache=True)
            else:
                graphs = dynamo_compiler.importer(model, input_ids)
    else:
        prefill_ids = torch.ones(1, args.seq, dtype=torch.long, device=device)
        decode_ids = torch.ones(1, 1, dtype=torch.long, device=device)
        with torch.no_grad():
            out0 = model(input_ids=prefill_ids, use_cache=True)
            past = out0.past_key_values
        dec_kw = _decode_forward_kwargs(model, past, args.seq, decode_ids)
        with torch.no_grad():
            graphs = dynamo_compiler.importer(model, decode_ids, **dec_kw)

    if len(graphs) != 1:
        print(f"error: expected one graph, got {len(graphs)}.", file=sys.stderr)
        return 1

    g = graphs[0]
    g.fuse_ops(_fusion_list())

    if args.mode == "decode":
        ph_names = [str(op.name) for op in g.body if isinstance(op, PlaceholderOp)]
        cur = args.cur_pos_placeholder
        if cur is None:
            for op in g.body:
                if not isinstance(op, PlaceholderOp):
                    continue
                meta = op.tensor_meta or {}
                shape = meta.get("shape")
                dtype = meta.get("dtype")
                dt_str = str(dtype).lower()
                is_int = "int" in dt_str and "64" in dt_str
                if is_int and shape is not None and list(shape) == [1]:
                    cur = str(op.name)
                    break
            if cur is None:
                for n in ph_names:
                    nl = n.lower()
                    if (
                        "pos" in nl
                        or "cache" in nl
                        or "cache_position" in nl
                        or nl.endswith("_pos")
                    ):
                        cur = n
                        break
        if cur:
            _patch_gqa_cur_pos(g, cur)
        else:
            print(
                "Placeholders:",
                ", ".join(ph_names),
                file=sys.stderr,
            )
            print(
                "warning: decode mode without a matching --cur-pos-placeholder; "
                "GQAAttentionFusedOp lowering may fail until kwargs['cur_pos_tensor'] is set.",
                file=sys.stderr,
            )

    driver = GraphDriver(g)
    sg = driver.subgraphs[0]
    try:
        sg.lower_to_ttir(element_dtype=args.element_dtype)
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        print(
            f"error: missing TTIR lowering for an op: {e}. "
            "Extend buddy.compiler.ops.ttir_llm.",
            file=sys.stderr,
        )
        return 1

    mod = sg.ttir_module
    if args.packed_forward:
        try:
            append_ttir_forward_bf16_f32_packed_i64_runtime(
                mod,
                subgraph_func_name="subgraph0",
                forward_func_name="forward",
            )
        except Exception as e:
            print(
                f"error: --packed-forward failed ({e}). "
                "Decode graphs with multiple integer inputs may need a different entry.",
                file=sys.stderr,
            )
            return 1

    out_stem = f"deepseek_r1_ttir_{args.mode}"
    out = args.output_dir / (
        f"{out_stem}_module.mlir" if args.packed_forward else f"{out_stem}.mlir"
    )
    out.write_text(str(mod).strip() + "\n", encoding="utf-8")
    print(f"Wrote {out.resolve()}")

    opt = args.ttmlir_opt or shutil.which("ttmlir-opt")
    if opt:
        cmd = [opt, str(out), "-o", os.devnull]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"warning: ttmlir-opt failed: {e}", file=sys.stderr)
            return 1
        print("ttmlir-opt: OK")
    else:
        print("Skip ttmlir-opt (not on PATH).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
