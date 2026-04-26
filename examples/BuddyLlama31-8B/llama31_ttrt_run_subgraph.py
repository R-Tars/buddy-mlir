# ===- deepseek_ttrt_run_subgraph.py -----------------------------------------
#
# Device runner for the DeepSeek prefill flatbuffer produced **without**
# ``--packed-forward``. The public program is ``@subgraph0`` and takes one
# tensor per weight (plus the integer ``input_ids``). Passing weights
# individually avoids the giant ``slice_static`` + ``reshape`` chain that a
# single 1.7B-element packed buffer would require in TTNN / tt-metal.
#
# Artifacts expected in ``--e2e-dir`` (created by ``deepseek_ttir_e2e_prepare.py``):
#   - ``packed_bf16.data``  linear uint16 bits of all bf16 weights in subgraph0
#                           placeholder order (one weight after another).
#   - ``bf16_sizes.npy``    int64 numel per bf16 weight (runner slice plan).
#   - ``packed_f32.npy``    linear float32 weights in subgraph0 order.
#   - ``input_ids_i64.npy`` integer input ids.
#   - ``golden_prefill_last_logits.npy`` logits for last time step.
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _ensure_ttrt():
    try:
        import ttrt  # noqa: F401
        from ttrt.common.api import API

        API.initialize_apis()
        import ttrt.runtime  # noqa: F401
        from ttrt.common.query import Query
        from ttrt.common.util import (
            Binary,
            FileManager,
            Logger,
            convert_input_layouts,
            convert_runtime_to_torch_tensor,
            create_tensor,
        )
    except ImportError as e:
        print(f"error: ttrt import failed: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    return (
        Query,
        Binary,
        FileManager,
        Logger,
        convert_input_layouts,
        convert_runtime_to_torch_tensor,
        create_tensor,
    )


def _shards_for_create_tensor(entry):
    if isinstance(entry, torch.Tensor):
        return [entry]
    return entry


def _is_int_dtype(td) -> bool:
    return td in (torch.int64, torch.int32, torch.int16, torch.int8, torch.bool)


def cmd_run(args: argparse.Namespace) -> int:
    (
        Query,
        Binary,
        FileManager,
        Logger,
        convert_input_layouts,
        convert_runtime_to_torch_tensor,
        create_tensor,
    ) = _ensure_ttrt()
    import ttrt.runtime

    ttnn_path = Path(args.ttnn).resolve()
    e2e = Path(args.e2e_dir).resolve()
    golden = np.load(e2e / "golden_prefill_last_logits.npy").astype(np.float32)

    # Memory-map the bf16 blob (~15 GiB for Llama-3.1-8B). Eagerly
    # ``np.fromfile`` would RSS the whole thing up front on top of the
    # per-weight copies that create_tensor needs, which does not fit.
    bf16_blob_u16 = np.memmap(
        e2e / "packed_bf16.data", dtype=np.uint16, mode="r"
    )
    bf16_sizes = np.load(e2e / "bf16_sizes.npy").astype(np.int64)
    if bf16_sizes.sum() != bf16_blob_u16.size:
        print(
            f"error: bf16 sizes sum {bf16_sizes.sum()} != blob elems "
            f"{bf16_blob_u16.size}",
            file=sys.stderr,
        )
        return 1
    bf16_offsets = np.concatenate(([0], np.cumsum(bf16_sizes)))

    f32_path = e2e / "packed_f32.npy"
    f32_blob = None
    f32_offset = 0
    if f32_path.is_file() and f32_path.stat().st_size > 0:
        f32_blob = np.load(f32_path).astype(np.float32).reshape(-1)

    input_ids = np.load(e2e / "input_ids_i64.npy").astype(np.int64)

    logger = Logger("")
    file_manager = FileManager(logger)
    binary = Binary(logger, file_manager, str(ttnn_path))

    query = Query({"--quiet": True}, logger)
    query.preprocess()
    query.check_constraints()
    query.execute()
    query.postprocess()
    if query.test_result != "pass" or query.device_ids is None:
        print("error: ttrt query failed", file=sys.stderr)
        return 1

    if not args.ignore_system_desc:
        try:
            binary.check_system_desc(query, ignore=False)
        except Exception as e:
            print(f"error: system desc: {e}", file=sys.stderr)
            return 1

    ttrt.runtime.set_compatible_device_runtime(binary.fbb)
    program = binary.get_program(int(args.program_index))
    if program.is_private():
        print("error: private program", file=sys.stderr)
        return 1

    inputs_list: list[torch.Tensor] = []
    bf16_i = 0
    for idx, inp in enumerate(program.inputs):
        dt = inp["desc"]["layout"]["memory_desc"]["data_type"]
        tdt = Binary.Program.from_data_type(dt)
        shp = tuple(inp["desc"]["shape"])
        n = int(np.prod(shp))

        if _is_int_dtype(tdt):
            t = torch.from_numpy(input_ids.copy()).to(tdt)
            if tuple(t.shape) != shp:
                t = t.reshape(shp)
            inputs_list.append(t)
        elif tdt == torch.bfloat16:
            if bf16_i >= len(bf16_sizes):
                print(
                    f"error: more bf16 inputs than weights (idx={idx}, bf16_i={bf16_i})",
                    file=sys.stderr,
                )
                return 1
            exp_n = int(bf16_sizes[bf16_i])
            if exp_n != n:
                print(
                    f"error: bf16 input {idx} shape numel {n} but weight {bf16_i} size {exp_n}",
                    file=sys.stderr,
                )
                return 1
            beg = int(bf16_offsets[bf16_i])
            end = int(bf16_offsets[bf16_i + 1])
            chunk_u16 = bf16_blob_u16[beg:end].copy()
            t = torch.from_numpy(chunk_u16).view(torch.bfloat16).reshape(shp)
            inputs_list.append(t)
            bf16_i += 1
        elif tdt == torch.float32:
            if f32_blob is None:
                print(f"error: f32 input {idx} but no packed_f32.npy provided", file=sys.stderr)
                return 1
            if f32_offset + n > f32_blob.size:
                print(
                    f"error: ran out of f32 blob at input {idx} (need {n}, have {f32_blob.size - f32_offset})",
                    file=sys.stderr,
                )
                return 1
            chunk = f32_blob[f32_offset:f32_offset + n].copy()
            t = torch.from_numpy(chunk).reshape(shp)
            inputs_list.append(t)
            f32_offset += n
        else:
            print(f"error: unsupported input dtype {tdt} at idx {idx}", file=sys.stderr)
            return 1

    if bf16_i != len(bf16_sizes):
        print(
            f"warning: only consumed {bf16_i}/{len(bf16_sizes)} bf16 weights",
            file=sys.stderr,
        )

    program.populate_inputs(torch.zeros, inputs_list)
    program.populate_outputs(lambda shape, dtype=None: torch.zeros(shape, dtype=dtype))

    fb_mesh_shape = program.mesh_shape
    mesh_options = ttrt.runtime.MeshDeviceOptions()
    mesh_options.mesh_shape = fb_mesh_shape
    device = ttrt.runtime.open_mesh_device(mesh_options)

    inputs_rt = []
    for entry in program.input_tensors:
        inputs_rt.append(create_tensor(_shards_for_create_tensor(entry), fb_mesh_shape))
    inputs_rt = convert_input_layouts(device, inputs_rt, binary.fbb, int(args.program_index))

    try:
        runtime_outputs = ttrt.runtime.submit(
            device, binary.fbb, int(args.program_index), inputs_rt
        )
        ttrt.runtime.wait(runtime_outputs)
        if len(runtime_outputs) < 1:
            print("error: no outputs", file=sys.stderr)
            return 1
        out_idx = int(args.logits_output_index)
        if out_idx < 0:
            out_idx = len(runtime_outputs) + out_idx
        output_host = ttrt.runtime.to_host(runtime_outputs[out_idx], untilize=True)
        mesh = fb_mesh_shape if len(output_host) > 1 else (1, 1)
        out_torch = ttrt.runtime.create_multi_device_host_tensor_from_shards(
            output_host, {}, mesh
        )
        out_tensor = convert_runtime_to_torch_tensor(out_torch)
        ttrt.runtime.deallocate_tensor(runtime_outputs[out_idx], force=True)
    finally:
        ttrt.runtime.close_mesh_device(device)

    logits = out_tensor.detach().float().cpu().numpy()
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    elif logits.ndim == 2:
        pass
    else:
        logits = logits.reshape(1, -1)

    g = torch.from_numpy(golden.reshape(-1))
    d = torch.from_numpy(logits.reshape(-1))
    diff = (g - d).abs()
    from ttrt.common.util import get_atol_rtol_pcc as gap

    _, _, pcc, pcc_msg = gap(g, d, args.atol, args.rtol, None)
    metrics = {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "pcc": float(pcc),
        "pcc_msg": pcc_msg,
        "golden_shape": list(golden.shape),
        "device_shape": list(logits.shape),
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.save_device_logits:
        np.save(args.save_device_logits, logits.astype(np.float32))
    ok = float(pcc) >= args.min_pcc
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="DeepSeek prefill (no packed-forward): ttrt run subgraph0 + logits compare"
    )
    p.add_argument("--ttnn", type=Path, required=True)
    p.add_argument("--e2e-dir", type=Path, required=True)
    p.add_argument(
        "--program-index",
        type=int,
        default=0,
        help="Index of the public program (subgraph0). Default 0.",
    )
    p.add_argument(
        "--logits-output-index",
        type=int,
        default=-1,
        help="Which output tensor is logits (-1 = last).",
    )
    p.add_argument("--ignore-system-desc", action="store_true")
    p.add_argument("--atol", type=float, default=0.25)
    p.add_argument("--rtol", type=float, default=0.25)
    p.add_argument("--min-pcc", type=float, default=0.95)
    p.add_argument("--save-device-logits", type=Path, default=None)
    args = p.parse_args()
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
