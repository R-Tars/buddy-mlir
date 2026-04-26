# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# TTIR lowering registry: expected Buddy op class names for the LeNet path
# (incremental coverage). No ttmlir import required.

from buddy.compiler.ops.ttir import ops_registry

for key in sorted(ops_registry.keys()):
    print(key)

# CHECK: AddMMOp
# CHECK: AddOp
# CHECK: Conv2dOp
# CHECK: MaxPool2dOp
# CHECK: PermuteOp
# CHECK: ReluOp
# CHECK: ReshapeOp
# CHECK: TOp
# CHECK: TransposeOp
# CHECK: ViewOp
