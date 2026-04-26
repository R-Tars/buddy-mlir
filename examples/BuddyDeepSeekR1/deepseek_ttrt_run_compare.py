# ===- deepseek_ttrt_run_compare.py --------------------------------------------
#
# 在 P150A 上运行 ``deepseek*.ttnn``（由 ``@forward`` 入口编译得到），加载
# ``deepseek_ttir_e2e_prepare.py`` 产出的 packed 权重与 ``input_ids``（线性
# ``packed_bf16.data`` 不变；若 flatbuffer 中首参为 ``[32, N/32]``，则按描述 reshape）。
# 将 **最后一个输出张量**（logits）与 ``golden_prefill_last_logits.npy`` 对比。
#
# 依赖与 LeNet 相同：``ttrt``（tt-mlir ``python_packages``）、设备在线。
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


def _load_packed_bf16(path: Path, num_elems: int, torch_dtype) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != num_elems:
        raise ValueError(f"packed_bf16: file has {raw.size} u16, expected {num_elems}")
    t = torch.from_numpy(raw.copy()).view(torch.bfloat16)
    return t.to(torch_dtype)


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

    packed_bf16_path = e2e / "packed_bf16.data"
    packed_f32_path = e2e / "packed_f32.npy"
    input_ids_path = e2e / "input_ids_i64.npy"
    if not packed_bf16_path.is_file():
        print(f"missing {packed_bf16_path}", file=sys.stderr)
        return 1
    if not input_ids_path.is_file():
        print(f"missing {input_ids_path}", file=sys.stderr)
        return 1

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

    dt0 = program.inputs[0]["desc"]["layout"]["memory_desc"]["data_type"]
    torch_dtype0 = Binary.Program.from_data_type(dt0)

    n0 = int(np.prod(program.inputs[0]["desc"]["shape"]))
    shp0 = tuple(program.inputs[0]["desc"]["shape"])
    pb = _load_packed_bf16(packed_bf16_path, n0, torch_dtype0).reshape(shp0)
    inputs_list: list[torch.Tensor] = [pb]

    j = 1
    if len(program.inputs) > 1:
        dt1 = program.inputs[1]["desc"]["layout"]["memory_desc"]["data_type"]
        td1 = Binary.Program.from_data_type(dt1)
        n1 = int(np.prod(program.inputs[1]["desc"]["shape"]))
        if td1 == torch.float32 and packed_f32_path.is_file() and packed_f32_path.stat().st_size > 0:
            pf = np.load(packed_f32_path).astype(np.float32).reshape(-1)
            if pf.size != n1:
                print(f"error: packed_f32 size {pf.size} vs fb {n1}", file=sys.stderr)
                return 1
            inputs_list.append(torch.from_numpy(pf.copy()).to(td1))
            j = 2
        elif "int" in str(td1).lower() or td1 in (torch.int64, torch.int32):
            arr = np.load(input_ids_path).astype(np.int64)
            t = torch.from_numpy(arr.copy()).to(td1)
            shp = tuple(program.inputs[1]["desc"]["shape"])
            if tuple(t.shape) != shp:
                t = t.reshape(shp)
            inputs_list.append(t)
            j = 2

    while j < len(program.inputs):
        dt = program.inputs[j]["desc"]["layout"]["memory_desc"]["data_type"]
        tdt = Binary.Program.from_data_type(dt)
        shp = tuple(program.inputs[j]["desc"]["shape"])
        arr = np.load(input_ids_path).astype(np.int64)
        t = torch.from_numpy(arr.copy()).to(tdt)
        if tuple(t.shape) != shp:
            t = t.reshape(shp)
        inputs_list.append(t)
        j += 1

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
    p = argparse.ArgumentParser(description="DeepSeek TTIR @forward: ttrt run + logits compare")
    p.add_argument("--ttnn", type=Path, required=True)
    p.add_argument("--e2e-dir", type=Path, required=True)
    p.add_argument("--program-index", type=int, default=1, help="1 = @forward, 0 = @subgraph0")
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
