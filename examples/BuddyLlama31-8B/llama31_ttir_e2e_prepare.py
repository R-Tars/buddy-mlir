# ===- llama31_ttir_e2e_prepare.py --------------------------------------------
#
# 为 ``buddy-llama31-lower-ttir.py`` 产出的 ``@subgraph0`` 准备
# **packed_bf16 / packed_f32 / input_ids** 与 **PyTorch golden logits**
# （仅最终 logits），供 ``llama31_ttrt_run_subgraph.py`` 在 P150A 上对比。
#
# 需能访问 HF 模型（或 ``LLAMA31_MODEL_PATH`` 本地目录）；与 lowering 使用相同
# ``--seq``、``torch.ones`` 占位输入以与 TTIR 对齐。
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.operation import PlaceholderOp
from buddy.compiler.ops import tosa


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Export packed weights + input_ids + golden logits for Llama-3.1 TTIR @subgraph0."
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("LLAMA31_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct"),
    )
    p.add_argument("--seq", type=int, default=32, help="Must match lowering (prefill seq length).")
    p.add_argument(
        "-o",
        "--artifacts",
        type=Path,
        default=here / "ttir_e2e_artifacts",
    )
    p.add_argument("--use-proxy", action="store_true")
    p.add_argument(
        "--use-cache",
        action="store_true",
        help=(
            "Trace and evaluate the prefill with use_cache=True so that golden "
            "past_key_values are saved alongside the logits (for chaining into decode)."
        ),
    )
    return p.parse_args()


def _proxy_env():
    u = "http://192.168.15.159:7890"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.setdefault(k, u)


def _bf16_flat_u16(t: torch.Tensor) -> np.ndarray:
    """1-D uint16 view of bf16 tensor (same bits as torch bf16)."""
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.detach().cpu().view(torch.uint16).numpy().reshape(-1).astype(np.uint16)


def main() -> int:
    args = _parse_args()
    if args.use_proxy:
        _proxy_env()

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    args.artifacts.mkdir(parents=True, exist_ok=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"error: load model: {e}", file=sys.stderr)
        return 1
    model.eval()

    device = next(model.parameters()).device
    input_ids = torch.ones(1, args.seq, dtype=torch.long, device=device)

    dynamo_compiler = DynamoCompiler(primary_registry=tosa.ops_registry)
    with torch.no_grad():
        if args.use_cache:
            graphs = dynamo_compiler.importer(model, input_ids, use_cache=True)
        else:
            graphs = dynamo_compiler.importer(model, input_ids)
    if len(graphs) != 1:
        print(f"error: expected one graph, got {len(graphs)}", file=sys.stderr)
        return 1

    g = graphs[0]
    ref = getattr(g, "_params_ref", None)
    if not ref:
        print("error: graph has no _params_ref (Buddy import issue).", file=sys.stderr)
        return 1

    # Golden logits (last timestep), same as export_golden_logits.py
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=args.use_cache)
        logits = out.logits[:, -1, :].float().cpu().numpy()
    np.save(args.artifacts / "golden_prefill_last_logits.npy", logits.astype(np.float32))

    # Optionally also save golden past_key_values so the E2E loop runner can
    # validate KV-cache outputs from the prefill flatbuffer against PyTorch.
    if args.use_cache:
        past = getattr(out, "past_key_values", None)
        if past is not None:
            try:
                layers = list(past)
            except TypeError:
                layers = [past[i] for i in range(len(past))]
            kv_dict: dict[str, np.ndarray] = {}
            for li, kv in enumerate(layers):
                k, v = kv[0], kv[1]
                kv_dict[f"k_{li:02d}"] = k.detach().float().cpu().numpy()
                kv_dict[f"v_{li:02d}"] = v.detach().float().cpu().numpy()
            np.savez(args.artifacts / "golden_prefill_kv.npz", **kv_dict)

    ph_order = [op for op in g.body if isinstance(op, PlaceholderOp)]
    fake_ix = set(int(x) for x in g._fake_params)
    inp_ix = set(int(x) for x in g._inputs)

    # Stream bf16 weights straight to disk. Concatenating all of them into a
    # single numpy array first (the original DeepSeek-1.5B path) needs an extra
    # full-size allocation (~16 GiB for Llama-3.1-8B) on top of the model,
    # which blows up host RAM. Writing per-chunk avoids that peak.
    bf16_sizes_list: list[int] = []
    bf16_count = 0
    f32_chunks: list[np.ndarray] = []
    int_tensors: list[torch.Tensor] = []

    bf16_path = args.artifacts / "packed_bf16.data"
    bf16_fh = open(bf16_path, "wb")

    fi = 0
    try:
        for ph in ph_order:
            ix = g.body.index(ph)
            if ix in inp_ix:
                int_tensors.append(input_ids.detach().cpu())
                continue

            if ix in fake_ix:
                if fi >= len(ref):
                    print(
                        f"error: ran out of _params_ref at {ph.name}",
                        file=sys.stderr,
                    )
                    return 1
                t = ref[fi]
                fi += 1
                if t.dtype == torch.bfloat16:
                    chunk = _bf16_flat_u16(t)
                    chunk.tofile(bf16_fh)
                    bf16_sizes_list.append(int(chunk.size))
                    bf16_count += 1
                    del chunk
                elif t.dtype == torch.float32:
                    f32_chunks.append(
                        t.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                        .reshape(-1)
                    )
                else:
                    print(
                        f"warning: skip param dtype {t.dtype} at {ph.name}",
                        file=sys.stderr,
                    )
                continue

            print(
                f"error: placeholder {ph.name} neither input nor fake param",
                file=sys.stderr,
            )
            return 1
    finally:
        bf16_fh.close()

    if bf16_count == 0:
        print("error: no bf16 chunks collected.", file=sys.stderr)
        return 1

    packed_bf16_elems = sum(bf16_sizes_list)
    bf16_sizes = np.array(bf16_sizes_list, dtype=np.int64)
    np.save(args.artifacts / "bf16_sizes.npy", bf16_sizes)

    if f32_chunks:
        packed_f32 = np.concatenate(f32_chunks).astype(np.float32)
        np.save(args.artifacts / "packed_f32.npy", packed_f32)
    else:
        (args.artifacts / "packed_f32.npy").write_bytes(b"")

    if len(int_tensors) != 1:
        print(
            f"warning: expected 1 integer runtime input, got {len(int_tensors)}; "
            "decode / other graphs may need a different runner.",
            file=sys.stderr,
        )
    if int_tensors:
        np.save(args.artifacts / "input_ids_i64.npy", int_tensors[0].numpy().astype(np.int64))

    meta = [
        f"seq={args.seq}",
        f"packed_bf16_elems={packed_bf16_elems}",
        f"bf16_count={bf16_count}",
        f"packed_f32_elems={sum(x.size for x in f32_chunks)}",
        f"f32_count={len(f32_chunks)}",
        f"golden_shape={tuple(logits.shape)}",
    ]
    (args.artifacts / "manifest.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")

    print(f"[OK] artifacts -> {args.artifacts.resolve()}")
    print("     packed_bf16.data (uint16 bits), packed_f32.npy, input_ids_i64.npy,")
    print("     golden_prefill_last_logits.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
