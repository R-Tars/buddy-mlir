# ===- lenet_ttrt_run_compare.py ------------------------------------------------
#
# 在 Tenstorrent 设备上运行编译得到的 ``.ttnn`` flatbuffer，使用与
# ``lenet_ttir_e2e_prepare.py`` 相同的 ``arg0.data`` + 输入图像，将设备输出与
# ``golden_logits_f32.npy`` / ``golden_class.txt`` 对比。
#
# 依赖：已安装的 ``ttrt``（含 ``ttrt.common`` / ``ttrt.runtime``），通常来自
# tt-mlir 构建目录下的 ``python_packages``，例如::
#
#   export PYTHONPATH=$TTMLIR_BUILD/python_packages:$PYTHONPATH
#
# 典型流程::
#
#   # 1) 准备权重、输入与 PyTorch golden
#   python lenet_ttir_e2e_prepare.py -o ./ttir_e2e_artifacts
#
#   # 2) 用本机 system desc 将 TTIR 编译为 .ttnn（命令依你环境而定）
#   #    ttmlir-opt lenet_ttir_module.mlir ... | ttmlir-translate --ttnn-to-flatbuffer -o lenet.ttnn
#
#   # 3) 本脚本：设备推理 + 对比
#   python lenet_ttrt_run_compare.py --ttnn ./lenet.ttnn \\
#     --e2e-dir ./ttir_e2e_artifacts --program-index 0
#
# 若仅已有 ``device_output_0.pt``（例如 ``ttrt run --save-artifacts`` 产出），可只做离线对比::
#
#   python lenet_ttrt_run_compare.py compare --device-output ./device_output_0.pt \\
#     --e2e-dir ./ttir_e2e_artifacts
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _ensure_ttrt_imports():
    try:
        # 必须先初始化 ttrt（library_tweaks）再 register APIs，否则 Query.registered_args 为空。
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
        print(
            "error: 无法导入 ttrt（需要 tt-mlir 构建的 python_packages 在 PYTHONPATH 中）。\n"
            f"  {e}",
            file=sys.stderr,
        )
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


def _load_torch_pt(path: Path) -> torch.Tensor:
    obj = torch.load(path, weights_only=True)
    if isinstance(obj, (list, tuple)):
        if len(obj) != 1:
            raise ValueError(f"期望 {path} 内为单个 tensor，得到 {len(obj)} 项")
        obj = obj[0]
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"期望 Tensor，得到 {type(obj)}")
    return obj


def _shards_for_create_tensor(entry) -> list:
    """ttrt ``populate_inputs`` 在 golden 路径下会 append 裸 Tensor；``create_tensor`` 需要 shard 列表。"""
    if isinstance(entry, torch.Tensor):
        return [entry]
    return entry


def _load_packed_and_image(
    e2e_dir: Path,
    program,
    torch_dtype: torch.dtype,
) -> list[torch.Tensor]:
    arg0_path = e2e_dir / "arg0.data"
    img_path = e2e_dir / "input_1x1x28x28_f32.npy"
    if not arg0_path.is_file():
        raise FileNotFoundError(arg0_path)
    if not img_path.is_file():
        raise FileNotFoundError(img_path)

    flat_f32 = np.fromfile(arg0_path, dtype=np.float32)
    expected0 = int(np.prod(program.inputs[0]["desc"]["shape"]))
    if flat_f32.size != expected0:
        raise ValueError(
            f"arg0.data 元素数 {flat_f32.size} 与 flatbuffer 输入0 期望 {expected0} 不一致"
        )

    packed = torch.from_numpy(flat_f32.copy()).to(torch_dtype)

    img_np = np.load(img_path)
    if img_np.shape != (1, 1, 28, 28):
        raise ValueError(f"输入图像形状应为 (1,1,28,28)，得到 {img_np.shape}")
    image = torch.from_numpy(img_np.astype(np.float32)).to(torch_dtype)

    return [packed, image]


def _compare_logits(
    golden_logits: np.ndarray,
    device_logits: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict:
    from ttrt.common.util import get_atol_rtol_pcc as gap

    g = torch.from_numpy(golden_logits).float().reshape(-1)
    d = device_logits.detach().float().cpu().reshape(-1)
    if g.numel() != d.numel():
        raise ValueError(f"形状不一致: golden {g.shape} vs device {d.shape}")
    diff = (g - d).abs()
    _, _, pcc, pcc_msg = gap(g, d, atol, rtol, None)
    return {
        "golden_argmax": int(torch.argmax(g).item()),
        "device_argmax": int(torch.argmax(d).item()),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "pcc": float(pcc),
        "pcc_msg": pcc_msg,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    _ensure_ttrt_imports()

    e2e = Path(args.e2e_dir)
    golden = np.load(e2e / "golden_logits_f32.npy").astype(np.float32)
    out = _load_torch_pt(Path(args.device_output))
    metrics = _compare_logits(golden, out, args.atol, args.rtol)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    match = metrics["golden_argmax"] == metrics["device_argmax"]
    if not match:
        print(
            f"warning: argmax 不一致 (golden={metrics['golden_argmax']}, device={metrics['device_argmax']})",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    (
        Query,
        Binary,
        FileManager,
        Logger,
        convert_input_layouts,
        convert_runtime_to_torch_tensor,
        create_tensor,
    ) = _ensure_ttrt_imports()

    import ttrt.runtime

    ttnn_path = Path(args.ttnn).resolve()
    if not ttnn_path.is_file():
        print(f"error: 找不到 flatbuffer: {ttnn_path}", file=sys.stderr)
        return 1

    e2e_dir = Path(args.e2e_dir).resolve()
    program_index = int(args.program_index)

    logger = Logger(args.log_file or "")
    file_manager = FileManager(logger)
    vb = getattr(args, "verbose", False)

    def _vb(msg: str) -> None:
        if vb:
            print(msg, file=sys.stderr, flush=True)

    _vb("[lenet_ttrt] loading flatbuffer …")
    binary = Binary(logger, file_manager, str(ttnn_path))
    _vb("[lenet_ttrt] flatbuffer loaded")

    query = Query(
        {"--quiet": True, "--disable-eth-dispatch": args.disable_eth_dispatch},
        logger,
    )
    query.preprocess()
    query.check_constraints()
    query.execute()
    query.postprocess()

    if query.test_result != "pass" or query.device_ids is None:
        print("error: ttrt query 未成功取得 system_desc / 设备。", file=sys.stderr)
        return 1

    _vb("[lenet_ttrt] query OK")

    # 注意：ttrt 的 check_system_desc(..., ignore=True) 仍会对两份巨大 JSON 做 difflib，mock 编译的
    # flatbuffer 在真机上可能卡住数分钟；完全跳过可避免该开销。
    if not args.ignore_system_desc:
        try:
            binary.check_system_desc(query, ignore=False)
        except Exception as e:
            print(f"error: system desc 校验失败: {e}", file=sys.stderr)
            print(
                "提示: 若编译所用 system desc 与当前机器不一致，可尝试 --ignore-system-desc",
                file=sys.stderr,
            )
            return 1

    _vb("[lenet_ttrt] set_compatible_device_runtime …")
    ttrt.runtime.set_compatible_device_runtime(binary.fbb)
    _vb("[lenet_ttrt] get_program …")
    program = binary.get_program(program_index)
    if program.is_private():
        print(f"error: program {program_index} 为 private，请换 --program-index", file=sys.stderr)
        return 1

    dt0 = program.inputs[0]["desc"]["layout"]["memory_desc"]["data_type"]
    torch_dtype = Binary.Program.from_data_type(dt0)

    _vb("[lenet_ttrt] load arg0 + image tensors …")
    golden_inputs = _load_packed_and_image(e2e_dir, program, torch_dtype)

    _vb("[lenet_ttrt] populate_inputs/outputs …")
    program.populate_inputs(torch.zeros, golden_inputs)
    program.populate_outputs(lambda shape, dtype=None: torch.zeros(shape, dtype=dtype))

    fb_mesh_shape = program.mesh_shape
    num_mesh_devices = 1
    for d in fb_mesh_shape:
        num_mesh_devices *= d
    if num_mesh_devices > len(query.device_ids):
        print(
            f"error: 需要 {num_mesh_devices} 颗设备，当前仅 {len(query.device_ids)}",
            file=sys.stderr,
        )
        return 1

    mesh_options = ttrt.runtime.MeshDeviceOptions()
    mesh_options.mesh_shape = fb_mesh_shape
    _vb("[lenet_ttrt] open_mesh_device …")
    device = ttrt.runtime.open_mesh_device(mesh_options)

    inputs_rt = []
    _vb("[lenet_ttrt] create_tensor(host) …")
    for entry in program.input_tensors:
        inputs_rt.append(create_tensor(_shards_for_create_tensor(entry), fb_mesh_shape))

    _vb("[lenet_ttrt] convert_input_layouts …")
    inputs_rt = convert_input_layouts(device, inputs_rt, binary.fbb, program_index)

    try:
        _vb("[lenet_ttrt] submit (首次可能编译内核，耗时数分钟) …")
        runtime_outputs = ttrt.runtime.submit(
            device, binary.fbb, program_index, inputs_rt
        )
        ttrt.runtime.wait(runtime_outputs)
        _vb("[lenet_ttrt] submit done")

        if len(runtime_outputs) < 1:
            print("error: 无输出张量", file=sys.stderr)
            return 1

        output_host = ttrt.runtime.to_host(runtime_outputs[0], untilize=True)
        if binary.extension != ".ttm":
            mesh = fb_mesh_shape if len(output_host) > 1 else (1, 1)
            out_torch = ttrt.runtime.create_multi_device_host_tensor_from_shards(
                output_host, {}, mesh
            )
            out_tensor = convert_runtime_to_torch_tensor(out_torch)
        else:
            raise RuntimeError("当前脚本仅针对 .ttnn flatbuffer 验证过")

        ttrt.runtime.deallocate_tensor(runtime_outputs[0], force=True)
    finally:
        ttrt.runtime.close_mesh_device(device)

    golden_path = e2e_dir / "golden_logits_f32.npy"
    golden = np.load(golden_path).astype(np.float32)
    metrics = _compare_logits(golden, out_tensor, args.atol, args.rtol)

    if args.save_device_output:
        outp = Path(args.save_device_output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_tensor, outp)
        metrics["saved_device_output"] = str(outp.resolve())

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    gold_cls_path = e2e_dir / "golden_class.txt"
    if gold_cls_path.is_file():
        gold_cls = int(gold_cls_path.read_text().strip())
        metrics["golden_class_file"] = gold_cls

    ok_argmax = metrics["golden_argmax"] == metrics["device_argmax"]
    if args.require_argmax_match and not ok_argmax:
        return 1
    if metrics["pcc"] < args.min_pcc:
        print(
            f"warning: PCC {metrics['pcc']:.6f} 低于阈值 {args.min_pcc}",
            file=sys.stderr,
        )
        return 1 if args.strict else 0
    return 0 if ok_argmax or not args.require_argmax_match else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="LeNet: Tenstorrent 设备推理并与 PyTorch golden 对比"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="在设备上运行 .ttnn 并与 golden 对比")
    pr.add_argument("--ttnn", type=Path, required=True, help="ttnn flatbuffer 路径")
    pr.add_argument(
        "--e2e-dir",
        type=Path,
        required=True,
        help="lenet_ttir_e2e_prepare.py 输出目录（含 arg0.data、input_*.npy、golden_*.）",
    )
    pr.add_argument(
        "--program-index",
        type=int,
        default=1,
        help=(
            "flatbuffer 内 program 下标；含 @forward 的 LeNet 打包入口通常为 1（0 为 subgraph0）。"
        ),
    )
    pr.add_argument(
        "--ignore-system-desc",
        action="store_true",
        help="跳过 flatbuffer 与当前机器的 system desc 严格匹配（仅当你清楚风险时）",
    )
    pr.add_argument(
        "--disable-eth-dispatch",
        action="store_true",
        help="对应 ttrt query 的 dispatch 选项",
    )
    pr.add_argument("--atol", type=float, default=1e-5)
    pr.add_argument("--rtol", type=float, default=1e-5)
    pr.add_argument("--min-pcc", type=float, default=0.99)
    pr.add_argument(
        "--strict",
        action="store_true",
        help="PCC 低于 --min-pcc 时返回非零",
    )
    pr.add_argument(
        "--require-argmax-match",
        action="store_true",
        help="要求 argmax 与 golden 一致，否则非零退出",
    )
    pr.add_argument(
        "--save-device-output",
        type=Path,
        default=None,
        help="将设备 logits 另存为 .pt",
    )
    pr.add_argument("--log-file", type=str, default="")
    pr.add_argument(
        "--verbose",
        action="store_true",
        help="在 stderr 打印各阶段进度（便于定位卡住位置）",
    )
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("compare", help="仅对比已有 device_output.pt 与 golden")
    pc.add_argument("--device-output", type=Path, required=True)
    pc.add_argument("--e2e-dir", type=Path, required=True)
    pc.add_argument("--atol", type=float, default=1e-5)
    pc.add_argument("--rtol", type=float, default=1e-5)
    pc.set_defaults(func=cmd_compare)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
