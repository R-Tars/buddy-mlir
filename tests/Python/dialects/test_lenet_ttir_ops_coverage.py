# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# Step 5: every LeNet (post-fuse) compute op type must appear in
# ``buddy.compiler.ops.ttir.ops_registry``. Torch / model changes may add op
# names — extend ``ops/ttir.py`` and re-run.

import sys
from pathlib import Path

import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.operation import (
    FuncOp,
    GetItemOp,
    OutputOp,
    PlaceholderOp,
)
from buddy.compiler.ops import tosa
from buddy.compiler.ops.ttir import ops_registry

_repo_root = Path(__file__).resolve().parents[3]
_lenet_dir = _repo_root / "tests" / "Models" / "BuddyLeNet"
if _lenet_dir.is_dir():
    sys.path.insert(0, str(_lenet_dir))
else:
    raise SystemExit(f"Expected LeNet test dir at {_lenet_dir}")

from model import LeNet  # noqa: E402

_SKIP = frozenset({PlaceholderOp, OutputOp, GetItemOp, FuncOp})

model = LeNet()
model.eval()
dynamo_compiler = DynamoCompiler(primary_registry=tosa.ops_registry)
data = torch.randn([1, 1, 28, 28])
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)
assert len(graphs) == 1
graph = graphs[0]
graph.fuse_ops([simply_fuse])

needed = {
    type(n).__name__
    for n in graph.body
    if type(n) not in _SKIP
}
registered = set(ops_registry.keys())
missing = sorted(needed - registered)
if missing:
    raise AssertionError(
        "LeNet uses Buddy ops not yet in ttir.ops_registry: " + ", ".join(missing)
    )

for k in sorted(needed):
    print("present", k)

print("LENET_TTIR_COVERAGE_OK")

# CHECK: LENET_TTIR_COVERAGE_OK
# CHECK-DAG: present AddMMOp
# CHECK-DAG: present AddOp
# CHECK-DAG: present Conv2dOp
# CHECK-DAG: present MaxPool2dOp
# CHECK-DAG: present PermuteOp
# CHECK-DAG: present ReluOp
# CHECK-DAG: present TOp
# CHECK-DAG: present TransposeOp
# CHECK-DAG: present ViewOp
