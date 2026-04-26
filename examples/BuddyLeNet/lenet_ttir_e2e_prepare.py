# ===- lenet_ttir_e2e_prepare.py ------------------------------------------------
#
# 为用 ``--packed-forward`` 生成的 ``lenet_ttir_module.mlir``（``@forward``）准备运行时输入
# 与 **PyTorch 参考结果**（Golden），便于与 ``ttrt run`` 或后续设备输出对齐校验。
#
# 写出（默认 ``./ttir_e2e_artifacts/``）::
#
#   - ``arg0.data`` — 与 ``buddy-lenet-import.py`` 相同布局的 f32 打包权重
#   - ``input_1x1x28x28_f32.npy`` — NCHW，与训练时 ``Normalize(0.5,0.5)`` 一致
#   - ``image_path.txt`` — 使用的图像路径
#   - ``golden_class.txt`` — PyTorch ``argmax`` 类别
#   - ``golden_logits_f32.npy`` — 形状 ``(1,10)`` 的 logits
#
# 与 TTIR ``@forward(%packed, %image)`` 对应：先做本脚本，再按你本机 ``ttmlir-opt`` / ``ttrt``
# 文档把二进制权重与输入送入编译产物（Tenstorrent 侧输入布局依 system desc 可能需再转换）。
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import LeNet


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Export arg0.data + input tensor + PyTorch golden for TTIR @forward."
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="lenet-model.pth (default: tests/Models/BuddyLeNet/lenet-model.pth)",
    )
    p.add_argument(
        "--image",
        type=Path,
        default=here / "images" / "1.png",
        help="Grayscale/classification test image (default: images/1.png).",
    )
    p.add_argument(
        "-o",
        "--artifacts",
        type=Path,
        default=here / "ttir_e2e_artifacts",
        help="Output directory.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    here = Path(__file__).resolve().parent
    tests_lenet = here.parent.parent / "tests" / "Models" / "BuddyLeNet"
    model_path = args.model or (tests_lenet / "lenet-model.pth")
    if not model_path.is_file():
        print(f"error: model not found: {model_path}", file=sys.stderr)
        return 1
    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 1

    args.artifacts.mkdir(parents=True, exist_ok=True)

    # Same preprocessing as pytorch-lenet-inference.py
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    img = Image.open(args.image).convert("L")
    x = transform(img).unsqueeze(0)

    model = LeNet()
    model = torch.load(model_path, weights_only=False)
    model.eval()
    with torch.no_grad():
        logits = model(x).numpy().astype(np.float32)
    pred = int(np.argmax(logits[0]))

    np.save(args.artifacts / "input_1x1x28x28_f32.npy", x.numpy().astype(np.float32))
    np.save(args.artifacts / "golden_logits_f32.npy", logits)
    (args.artifacts / "golden_class.txt").write_text(str(pred) + "\n", encoding="utf-8")
    (args.artifacts / "image_path.txt").write_text(str(args.image.resolve()) + "\n")

    # Buddy-packed weights via existing importer (f32 blob)
    env = {**os.environ, "LENET_MODEL_PATH": str(model_path.resolve())}
    subprocess.run(
        [
            sys.executable,
            str(here / "buddy-lenet-import.py"),
            "--output-dir",
            str(args.artifacts),
        ],
        check=True,
        env=env,
    )

    print(f"[OK] artifacts in {args.artifacts.resolve()}")
    print(f"     golden class (PyTorch): {pred}")
    print("     files: arg0.data, input_1x1x28x28_f32.npy, golden_logits_f32.npy, ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
