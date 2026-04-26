# ===- buddy-lenet-lower-ttir.py -----------------------------------------------
#
# LeNet → Buddy Graph → ``Graph.lower_to_ttir()`` (bf16 TTIR) + optional
# ``ttmlir-opt`` sanity check. Use ``--packed-forward`` to also emit ``forward``
# that unpacks a 1-D weight buffer (same semantics as native
# ``buddy-lenet-import.py``) and ``func.call``\ s ``subgraph0``.
#
# Usage (from build tree with ``buddy.compiler`` on PYTHONPATH)::
#
#   export PYTHONPATH=$BUDDY_BUILD/python_packages:$PYTHONPATH
#   export PYTHONPATH=$TTMLIR_BUILD/python_packages:$PYTHONPATH   # for ttmlir
#   python buddy-lenet-lower-ttir.py --ttmlir-opt $(which ttmlir-opt)
#
# Default output directory is ``examples/BuddyLeNet/ttir_out/`` (override with ``-o``).
#
# Or use random-init weights (no ``.pth``)::
#
#   python buddy-lenet-lower-ttir.py --random-init
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.ttir_import import append_ttir_forward_with_packed_weights
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa


def _parse_args() -> argparse.Namespace:
    _script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Lower LeNet subgraph0 to TTIR MLIR (bf16) and optional ttmlir-opt check."
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=_script_dir / "ttir_out",
        help=(
            "Directory for lenet_ttir_subgraph0.mlir "
            f"(default: {_script_dir / 'ttir_out'} — under this example folder)."
        ),
    )
    p.add_argument(
        "--random-init",
        action="store_true",
        help="Use an uninitialized LeNet() (same op graph as trained weights).",
    )
    p.add_argument(
        "--element-dtype",
        choices=("bf16", "f32"),
        default="bf16",
        help='TTIR tensor element type for ``lower_to_ttir`` (default: bf16; use f32 to match arg0.data / PyTorch f32).',
    )
    p.add_argument(
        "--ttmlir-opt",
        type=str,
        default=None,
        help="Path to ttmlir-opt; if set, run: OPT <mlir> -o /dev/null",
    )
    p.add_argument(
        "--packed-forward",
        action="store_true",
        help=(
            "Append func @forward with packed 1-D weights + call @subgraph0 "
            "(native buddy-lenet-import style)."
        ),
    )
    p.add_argument(
        "--ttnn-pipeline-check",
        action="store_true",
        help=(
            "If --ttmlir-opt is set, also run "
            "--ttir-to-ttnn-backend-pipeline=mock-system-desc-arch=wormhole_b0 "
            "(pipeline smoke test; needs matching ttmlir-opt build)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    _here = Path(__file__).resolve().parent
    _tests_lenet = _here.parent.parent / "tests" / "Models" / "BuddyLeNet"
    if _tests_lenet.is_dir():
        sys.path.insert(0, str(_tests_lenet))
    from model import LeNet

    model = LeNet()
    if not args.random_init:
        model_path = os.environ.get("LENET_MODEL_PATH")
        if model_path is None:
            model_path = _tests_lenet / "lenet-model.pth"
        model_path = Path(model_path)
        if not model_path.is_file():
            print(
                f"error: no weights at {model_path} (set LENET_MODEL_PATH or use --random-init).",
                file=sys.stderr,
            )
            return 1
        model = torch.load(model_path, weights_only=False)
    model = model.eval()

    dynamo_compiler = DynamoCompiler(primary_registry=tosa.ops_registry)
    data = torch.randn([1, 1, 28, 28])
    with torch.no_grad():
        graphs = dynamo_compiler.importer(model, data)

    if len(graphs) != 1:
        print(f"error: expected one graph, got {len(graphs)}.", file=sys.stderr)
        return 1

    graphs[0].fuse_ops([simply_fuse])
    driver = GraphDriver(graphs[0])
    sg = driver.subgraphs[0]
    try:
        sg.lower_to_ttir(element_dtype=args.element_dtype)
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    mod = sg.ttir_module

    if args.packed_forward:
        append_ttir_forward_with_packed_weights(
            mod,
            subgraph_func_name="subgraph0",
            forward_func_name="forward",
        )

    mlir_path = out / "lenet_ttir_subgraph0.mlir"
    if args.packed_forward:
        mlir_path = out / "lenet_ttir_module.mlir"
    text = str(mod).strip() + "\n"
    mlir_path.write_text(text, encoding="utf-8")
    print(f"Wrote TTIR module: {mlir_path.resolve()}")
    print(mod)

    opt = args.ttmlir_opt or shutil.which("ttmlir-opt")
    if opt:
        cmd = [opt, str(mlir_path), "-o", os.devnull]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"warning: ttmlir-opt failed: {e}", file=sys.stderr)
            return 1
        print("ttmlir-opt: OK")
        if args.ttnn_pipeline_check:
            pipe = (
                "--ttir-to-ttnn-backend-pipeline="
                "mock-system-desc-arch=wormhole_b0"
            )
            cmd2 = [opt, str(mlir_path), pipe, "-o", os.devnull]
            print(f"Running: {' '.join(cmd2)}")
            try:
                subprocess.run(cmd2, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(
                    f"warning: TTNN backend pipeline failed: {e}",
                    file=sys.stderr,
                )
                return 1
            print("ttmlir-opt TTNN backend pipeline: OK")
    else:
        print(
            "Skip ttmlir-opt (pass --ttmlir-opt PATH or put ttmlir-opt on PATH).",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
