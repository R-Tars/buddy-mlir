"""Compatibility facade for the TTNN Direct compiler modules."""

from ..compiler.codegen import dry_run_report, write_python_ttnn_skeleton
from ..compiler.config import (
    build_codegen_config,
    validate_execution_plan_for_codegen,
)
from ..compiler.source_templates import (
    render_codegen_readme,
    render_python_ttnn_model,
)

__all__ = [
    "build_codegen_config",
    "dry_run_report",
    "render_codegen_readme",
    "render_python_ttnn_model",
    "validate_execution_plan_for_codegen",
    "write_python_ttnn_skeleton",
]
