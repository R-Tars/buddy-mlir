# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# Stage-0 helper: dump sorted Buddy op class names for LeNet (f32, random init)
# after ``simply_fuse``. Excludes graph plumbing (placeholders, output, etc.).

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

names = sorted(
    {
        type(n).__name__
        for n in graph.body
        if type(n) not in _SKIP
    }
)
for name in names:
    print(name)

# Keep in sync with the torch / buddy frontend for this model.
# CHECK-DAG: AddMMOp
# CHECK-DAG: AddOp
# CHECK-DAG: Conv2dOp
# CHECK-DAG: MaxPool2dOp
# CHECK-DAG: PermuteOp
# CHECK-DAG: ReluOp
# CHECK-DAG: TOp
# CHECK-DAG: TransposeOp
# CHECK-DAG: ViewOp
