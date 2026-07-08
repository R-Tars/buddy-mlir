from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .codegen.artifacts import prepare_offline_artifacts
from .codegen.config_emit import (
    SEED_RECIPE,
    dump_parameter_config,
    emit_parameter_config,
    parameter_config_dry_run_report,
)
from .codegen.config_diff import (
    default_official_config_path,
    diff_official_config,
    dump_config_diff,
)
from .codegen.package import (
    package_dry_run_report,
    package_ttnn_direct_program,
)
from .codegen.parameters import (
    ParameterMaterializationError,
    load_llama_parameters_from_manifests,
    materialize_parameters_from_program,
    parse_layer_ids,
)
from .codegen.python_ttnn import dry_run_report, write_python_ttnn_skeleton
from .codegen.program import write_decode_program_bundle
from .codegen.ttnn_tensorizer import (
    TTNNTensorizationError,
    load_parameter_config_from_program,
    parse_role_groups,
    tensorize_parameters_from_program_dry_run,
    to_ttnn_parameters,
)
from .semantic.dump import dump_graph_json, load_graph_json
from .semantic.importer_hf_llama import import_hf_llama
from .profile_template import profile_template
from .search.report import dump_search_report
from .search.decode_step_autotune import run_decode_step_autotune
from .search.decode_depth_sweep import run_decode_depth_sweep
from .search.generate_depth_sweep import run_generate_depth_sweep
from .search.runner import run_lm_head_search
from .search.space import load_search_space
from .decode_loop import run_prompt_decode_loop
from .generate import run_generate, run_profile_generate
from .smoke_mlp import NO_TTNN_DEVICE_MESSAGE, run_smoke_mlp
from .smoke_decode_shell import run_smoke_decode_shell
from .smoke_attention_primitive import (
    ATTENTION_PRIMITIVES,
    run_smoke_attention_primitive,
)
from .smoke_attention_layer import run_smoke_attention_layer
from .smoke_single_layer_decode import (
    profile_decode_step,
    run_smoke_decode_step,
    run_smoke_single_layer_decode,
)
from .smoke_prefill import run_smoke_prefill
from .templates.registry import (
    build_execution_plan,
    dump_execution_plan,
    load_execution_plan,
    load_template_config,
)
from .templates.diff import (
    diff_plan_against_official,
    dump_plan_diff,
    load_official_template,
)
from .validation import (
    default_decode_step_search_space_path,
    default_official_template_path,
    default_performance_baselines_path,
    default_search_space_path,
    preflight_real_decode,
    recover_real_decode_process_failure,
    recover_real_decode_process_timeout,
    validate_direct,
    validate_real_decode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="buddy-ttnn-direct",
        description="Buddy-TTNN Direct model tooling.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_prompt_runtime_args(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--prompt",
            default=None,
            help=(
                "Prompt text used to build decode token_ids through an HF "
                "tokenizer instead of smoke-synthesized token tensors."
            ),
        )
        command.add_argument(
            "--tokenizer-path",
            type=Path,
            default=None,
            help=(
                "Optional tokenizer directory/model id. Defaults to "
                "--model-path when a prompt is provided."
            ),
        )

    import_llama = subparsers.add_parser(
        "import-llama",
        help="Import an HF Llama config into a semantic graph JSON.",
    )
    import_llama.add_argument(
        "--model-path",
        required=True,
        help="Local HF model directory or model id.",
    )
    import_llama.add_argument(
        "--mode",
        choices=("prefill", "decode"),
        default="decode",
        help="Semantic graph mode to emit.",
    )
    import_llama.add_argument("--batch-size", type=int, required=True)
    import_llama.add_argument("--seq-len", type=int, required=True)
    import_llama.add_argument("--max-cache-len", type=int, required=True)
    import_llama.add_argument(
        "--generation-mode",
        choices=("greedy", "sampling"),
        default="greedy",
    )
    import_llama.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Emit graph JSON from config and metadata only. Phase 1 never "
            "loads full tensor payloads, so this mainly documents intent."
        ),
    )
    import_llama.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output semantic graph JSON path.",
    )
    import_llama.set_defaults(func=_cmd_import_llama)

    plan = subparsers.add_parser(
        "plan",
        help="Map a Llama semantic graph to official-like TTNN templates.",
    )
    plan.add_argument(
        "--semantic-json",
        type=Path,
        required=True,
        help="Input semantic graph JSON from import-llama.",
    )
    plan.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Template seed config JSON.",
    )
    plan.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output execution plan JSON path.",
    )
    plan.set_defaults(func=_cmd_plan)

    codegen_python = subparsers.add_parser(
        "codegen-python",
        help="Generate a Python TTNN skeleton from an execution plan.",
    )
    codegen_python.add_argument(
        "--plan-json",
        type=Path,
        required=True,
        help="Input execution plan JSON from plan.",
    )
    codegen_python.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for generated Python TTNN skeleton artifacts.",
    )
    codegen_python.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report generated artifact paths without writing.",
    )
    codegen_python.set_defaults(func=_cmd_codegen_python)

    emit_config = subparsers.add_parser(
        "emit-config",
        help="Emit parameter dtype/layout/packing metadata for a Llama graph.",
    )
    emit_config.add_argument(
        "--semantic-json",
        type=Path,
        required=True,
        help="Input semantic graph JSON from import-llama.",
    )
    emit_config.add_argument(
        "--recipe",
        default=SEED_RECIPE,
        help="Parameter metadata recipe name.",
    )
    emit_config.add_argument(
        "--lm-head-split-count",
        type=int,
        default=8,
        help="Number of vocab shards for split LM-head metadata.",
    )
    emit_config.add_argument(
        "--kv-page-block-size",
        type=int,
        default=32,
        help="Paged KV cache block size in tokens.",
    )
    emit_config.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output parameter metadata JSON path.",
    )
    emit_config.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report metadata summary without writing.",
    )
    emit_config.set_defaults(func=_cmd_emit_config)

    diff_plan = subparsers.add_parser(
        "diff-plan",
        help="Diff a TTNN Direct plan against an official-like decode template.",
    )
    diff_plan.add_argument(
        "--ours",
        type=Path,
        required=True,
        help="Input TTNN Direct execution plan JSON.",
    )
    diff_plan.add_argument(
        "--official-template",
        type=Path,
        required=True,
        help="Official-like reference decode template JSON.",
    )
    diff_plan.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output structural diff JSON path.",
    )
    diff_plan.set_defaults(func=_cmd_diff_plan)

    diff_official_config_parser = subparsers.add_parser(
        "diff-official-config",
        help=(
            "Diff generated TTNN Direct config metadata against an official "
            "or official-like config parity JSON."
        ),
    )
    diff_official_config_parser.add_argument(
        "--ours",
        type=Path,
        required=True,
        help="Generated TTNN Direct config.json or normalized parity JSON.",
    )
    diff_official_config_parser.add_argument(
        "--official",
        type=Path,
        default=default_official_config_path(),
        help="Official or official-like config parity JSON.",
    )
    diff_official_config_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output config parity diff JSON path.",
    )
    diff_official_config_parser.set_defaults(func=_cmd_diff_official_config)

    prepare_artifacts = subparsers.add_parser(
        "prepare-artifacts",
        help="Prepare offline TTNN Direct weight and cache manifests.",
    )
    prepare_artifacts.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local HF model directory containing safetensors metadata.",
    )
    prepare_artifacts.add_argument(
        "--semantic-json",
        type=Path,
        required=True,
        help="Input semantic graph JSON from import-llama.",
    )
    prepare_artifacts.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Parameter metadata JSON from emit-config.",
    )
    prepare_artifacts.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for offline manifest artifacts.",
    )
    prepare_artifacts.set_defaults(func=_cmd_prepare_artifacts)

    smoke_mlp = subparsers.add_parser(
        "smoke-mlp",
        help="Run or dry-run a small TTNN MLP template smoke test.",
    )
    smoke_mlp.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the smoke report.",
    )
    smoke_mlp.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to open when hardware is available.",
    )
    smoke_mlp.add_argument("--batch-size", type=int, required=True)
    smoke_mlp.add_argument("--hidden-size", type=int, required=True)
    smoke_mlp.add_argument("--intermediate-size", type=int, required=True)
    smoke_mlp.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    smoke_mlp.add_argument(
        "--pcc-threshold",
        type=float,
        default=0.99,
        help="Minimum PCC required for a hardware smoke pass.",
    )
    smoke_mlp.add_argument("--seed", type=int, default=0)
    smoke_mlp.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the smoke report schema without opening a TTNN device.",
    )
    smoke_mlp.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output smoke report JSON path.",
    )
    smoke_mlp.set_defaults(func=_cmd_smoke_mlp)

    smoke_decode_shell = subparsers.add_parser(
        "smoke-decode-shell",
        help=(
            "Run or dry-run a one-layer generated decode shell with "
            "attention disabled."
        ),
    )
    smoke_decode_shell.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    smoke_decode_shell.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local HF model directory. Required for non-dry-run execution.",
    )
    add_prompt_runtime_args(smoke_decode_shell)
    smoke_decode_shell.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of decoder layers to run in the shell.",
    )
    smoke_decode_shell.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override generated batch size for shell token inputs.",
    )
    smoke_decode_shell.add_argument(
        "--cache-len",
        type=int,
        default=None,
        help="Override generated cache length recorded for validation.",
    )
    smoke_decode_shell.add_argument(
        "--disable-attention",
        action="store_true",
        help="Required in PR-D; runs embedding/RMSNorm/MLP/LM-head only.",
    )
    smoke_decode_shell.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the smoke report.",
    )
    smoke_decode_shell.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to open when hardware is available.",
    )
    smoke_decode_shell.add_argument(
        "--pcc-threshold",
        type=float,
        default=0.99,
        help="Minimum final-hidden PCC required when torch reference runs.",
    )
    smoke_decode_shell.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the decode shell report schema without opening a device.",
    )
    smoke_decode_shell.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output decode shell smoke report JSON path.",
    )
    smoke_decode_shell.set_defaults(func=_cmd_smoke_decode_shell)

    smoke_attention_primitive = subparsers.add_parser(
        "smoke-attention-primitive",
        help=(
            "Run or dry-run one official attention decode TTNN primitive "
            "for API/shape validation."
        ),
    )
    smoke_attention_primitive.add_argument(
        "--primitive",
        choices=ATTENTION_PRIMITIVES,
        required=True,
        help="Attention primitive to smoke test.",
    )
    smoke_attention_primitive.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the smoke report.",
    )
    smoke_attention_primitive.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to open when hardware is available.",
    )
    smoke_attention_primitive.add_argument("--batch-size", type=int, required=True)
    smoke_attention_primitive.add_argument("--hidden-size", type=int, required=True)
    smoke_attention_primitive.add_argument("--num-heads", type=int, required=True)
    smoke_attention_primitive.add_argument("--num-kv-heads", type=int, required=True)
    smoke_attention_primitive.add_argument("--head-dim", type=int, required=True)
    smoke_attention_primitive.add_argument(
        "--max-cache-len",
        type=int,
        default=1024,
        help="Paged KV cache length used for cache primitive shapes.",
    )
    smoke_attention_primitive.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    smoke_attention_primitive.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the primitive report schema without opening a TTNN device.",
    )
    smoke_attention_primitive.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output attention primitive smoke report JSON path.",
    )
    smoke_attention_primitive.set_defaults(func=_cmd_smoke_attention_primitive)

    smoke_attention_layer = subparsers.add_parser(
        "smoke-attention-layer",
        help=(
            "Run or dry-run a one-layer attention decode smoke path by "
            "composing validated TTNN primitives."
        ),
    )
    smoke_attention_layer.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    smoke_attention_layer.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Decoder layer id to smoke test.",
    )
    smoke_attention_layer.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the smoke report.",
    )
    smoke_attention_layer.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to open when hardware is available.",
    )
    smoke_attention_layer.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override generated batch size for synthetic smoke tensors.",
    )
    smoke_attention_layer.add_argument(
        "--cache-len",
        type=int,
        default=None,
        help="Override generated max cache length for synthetic smoke tensors.",
    )
    smoke_attention_layer.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    smoke_attention_layer.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the attention layer report schema without opening a device.",
    )
    smoke_attention_layer.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output attention layer smoke report JSON path.",
    )
    smoke_attention_layer.set_defaults(func=_cmd_smoke_attention_layer)

    smoke_single_layer_decode = subparsers.add_parser(
        "smoke-single-layer-decode",
        help=(
            "Run or dry-run generated single-layer decode with attention, "
            "MLP, final norm, and LM-head."
        ),
    )
    smoke_single_layer_decode.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    smoke_single_layer_decode.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Optional local HF model directory. When provided in device mode, "
            "weights are materialized and tensorized instead of using synthetic "
            "parameters."
        ),
    )
    add_prompt_runtime_args(smoke_single_layer_decode)
    smoke_single_layer_decode.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the smoke report.",
    )
    smoke_single_layer_decode.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to open when hardware is available.",
    )
    smoke_single_layer_decode.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override generated batch size for synthetic decode tensors.",
    )
    smoke_single_layer_decode.add_argument(
        "--cache-len",
        type=int,
        default=None,
        help="Override generated max cache length for synthetic decode tensors.",
    )
    smoke_single_layer_decode.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    smoke_single_layer_decode.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the generated decode report schema without opening a device.",
    )
    smoke_single_layer_decode.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output generated single-layer decode smoke report JSON path.",
    )
    smoke_single_layer_decode.set_defaults(func=_cmd_smoke_single_layer_decode)

    smoke_prefill = subparsers.add_parser(
        "smoke-prefill",
        help=(
            "Run or dry-run generated prefill_prompt for a fixed prompt "
            "length and report KV cache population evidence."
        ),
    )
    smoke_prefill.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    smoke_prefill.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of generated decoder layers to execute.",
    )
    smoke_prefill.add_argument(
        "--prefill-len",
        type=int,
        default=None,
        help="Prompt length to prefill. Defaults to generated config.",
    )
    smoke_prefill.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the smoke report.",
    )
    smoke_prefill.add_argument("--device-id", type=int, default=0)
    smoke_prefill.add_argument("--batch-size", type=int, default=None)
    smoke_prefill.add_argument("--cache-len", type=int, default=None)
    smoke_prefill.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    smoke_prefill.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the prefill report schema without opening a device.",
    )
    smoke_prefill.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output generated prefill smoke report JSON path.",
    )
    smoke_prefill.set_defaults(func=_cmd_smoke_prefill)

    smoke_decode_step = subparsers.add_parser(
        "smoke-decode-step",
        help=(
            "Run or dry-run generated decode_step for a configurable "
            "number of decoder layers."
        ),
    )
    smoke_decode_step.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    smoke_decode_step.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Optional local HF model directory. When provided in device mode, "
            "weights are materialized and tensorized instead of using synthetic "
            "parameters."
        ),
    )
    add_prompt_runtime_args(smoke_decode_step)
    smoke_decode_step.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of generated decoder layers to execute.",
    )
    smoke_decode_step.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the smoke report.",
    )
    smoke_decode_step.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to open when hardware is available.",
    )
    smoke_decode_step.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override generated batch size for synthetic decode tensors.",
    )
    smoke_decode_step.add_argument(
        "--cache-len",
        type=int,
        default=None,
        help="Override generated max cache length for synthetic decode tensors.",
    )
    smoke_decode_step.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    smoke_decode_step.add_argument(
        "--trace",
        action="store_true",
        help="Capture generated decode_step once and execute the trace.",
    )
    smoke_decode_step.add_argument(
        "--trace-iterations",
        type=int,
        default=1,
        help="Number of execute_trace iterations after capture.",
    )
    smoke_decode_step.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the generated decode report schema without opening a device.",
    )
    smoke_decode_step.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output generated decode-step smoke report JSON path.",
    )
    smoke_decode_step.set_defaults(func=_cmd_smoke_decode_step)

    profile_decode_step_parser = subparsers.add_parser(
        "profile-decode-step",
        help=(
            "Profile generated decode_step by section for bottleneck "
            "attribution."
        ),
    )
    profile_decode_step_parser.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    profile_decode_step_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Optional local HF model directory. When provided in device mode, "
            "weights are materialized and tensorized instead of using synthetic "
            "parameters."
        ),
    )
    add_prompt_runtime_args(profile_decode_step_parser)
    profile_decode_step_parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of generated decoder layers to profile.",
    )
    profile_decode_step_parser.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the profile report.",
    )
    profile_decode_step_parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to open when hardware is available.",
    )
    profile_decode_step_parser.add_argument("--batch-size", type=int, default=None)
    profile_decode_step_parser.add_argument("--cache-len", type=int, default=None)
    profile_decode_step_parser.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    profile_decode_step_parser.add_argument(
        "--trace",
        action="store_true",
        help="Also capture/execute a generated decode_step trace.",
    )
    profile_decode_step_parser.add_argument(
        "--trace-iterations",
        type=int,
        default=1,
        help="Number of execute_trace iterations after capture.",
    )
    profile_decode_step_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the profile report schema without opening a device.",
    )
    profile_decode_step_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output generated decode-step profile report JSON path.",
    )
    profile_decode_step_parser.set_defaults(func=_cmd_profile_decode_step)

    prompt_decode_loop = subparsers.add_parser(
        "prompt-decode-loop",
        help=(
            "Run or dry-run a tokenizer-owned multi-step generated decode "
            "loop that owns token ids, decode metadata, rotary tensors, "
            "and paged KV cache."
        ),
    )
    prompt_decode_loop.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    prompt_decode_loop.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local HF model directory. Required for non-dry-run execution.",
    )
    add_prompt_runtime_args(prompt_decode_loop)
    prompt_decode_loop.add_argument(
        "--decode-steps",
        type=int,
        default=2,
        help="Number of generated decode steps to run in the loop.",
    )
    prompt_decode_loop.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Alias for --decode-steps used by decode-loop demos.",
    )
    prompt_decode_loop.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of generated decoder layers to execute per step.",
    )
    prompt_decode_loop.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the report.",
    )
    prompt_decode_loop.add_argument("--device-id", type=int, default=0)
    prompt_decode_loop.add_argument("--batch-size", type=int, default=None)
    prompt_decode_loop.add_argument("--cache-len", type=int, default=None)
    prompt_decode_loop.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    prompt_decode_loop.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the loop report schema without opening a device.",
    )
    prompt_decode_loop.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output prompt decode loop report JSON path.",
    )
    prompt_decode_loop.set_defaults(func=_cmd_prompt_decode_loop)

    generate = subparsers.add_parser(
        "generate",
        help=(
            "Run or dry-run prompt prefill followed by generated decode "
            "steps using the prefilled KV cache."
        ),
    )
    generate.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    generate.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local HF model directory. Required for non-dry-run execution.",
    )
    add_prompt_runtime_args(generate)
    generate.add_argument(
        "--max-new-tokens",
        type=int,
        default=2,
        help=(
            "Total generated tokens to report. The first token comes from "
            "prefill; remaining tokens come from decode steps."
        ),
    )
    generate.add_argument(
        "--prefill-len",
        type=int,
        default=None,
        help="Fixed prompt length for prefill. Defaults to generated config.",
    )
    generate.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of generated decoder layers to execute.",
    )
    generate.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the report.",
    )
    generate.add_argument("--device-id", type=int, default=0)
    generate.add_argument("--batch-size", type=int, default=None)
    generate.add_argument("--cache-len", type=int, default=None)
    generate.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    generate.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the generate report schema without opening a device.",
    )
    generate.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output generate report JSON path.",
    )
    generate.set_defaults(func=_cmd_generate)

    profile_generate = subparsers.add_parser(
        "profile-generate",
        help=(
            "Run the prefill+decode generate path and emit first-pass "
            "latency/throughput profile fields without claiming parity."
        ),
    )
    profile_generate.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    profile_generate.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local HF model directory. Required for non-dry-run execution.",
    )
    add_prompt_runtime_args(profile_generate)
    profile_generate.add_argument(
        "--max-new-tokens",
        type=int,
        default=2,
        help="Total generated tokens to profile.",
    )
    profile_generate.add_argument(
        "--prefill-len",
        type=int,
        default=None,
        help="Fixed prompt length for prefill. Defaults to generated config.",
    )
    profile_generate.add_argument("--layers", type=int, default=1)
    profile_generate.add_argument("--device", default="p150a")
    profile_generate.add_argument("--device-id", type=int, default=0)
    profile_generate.add_argument("--batch-size", type=int, default=None)
    profile_generate.add_argument("--cache-len", type=int, default=None)
    profile_generate.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    profile_generate.add_argument(
        "--dry-run",
        action="store_true",
        help="Write profile/generate schemas without opening a TTNN device.",
    )
    profile_generate.add_argument(
        "--generate-report",
        type=Path,
        default=None,
        help="Optional path for the underlying generate report.",
    )
    profile_generate.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output generate profile report JSON path.",
    )
    profile_generate.set_defaults(func=_cmd_profile_generate)

    decode_depth_sweep = subparsers.add_parser(
        "decode-depth-sweep",
        help=(
            "Run profile-decode-step across an increasing set of decoder "
            "depths for functional depth bring-up."
        ),
    )
    decode_depth_sweep.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    decode_depth_sweep.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Optional local HF model directory. When provided in device mode, "
            "each depth profile materializes and tensorizes real weights."
        ),
    )
    add_prompt_runtime_args(decode_depth_sweep)
    decode_depth_sweep.add_argument(
        "--depths",
        default=None,
        help=(
            "Comma-separated depth list such as 1,2,4,full. Defaults to "
            "1,2,4,full clipped to the generated layer count."
        ),
    )
    decode_depth_sweep.add_argument(
        "--profiles-dir",
        type=Path,
        default=None,
        help="Directory for per-depth profile-decode-step reports.",
    )
    decode_depth_sweep.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the sweep report.",
    )
    decode_depth_sweep.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to open when hardware is available.",
    )
    decode_depth_sweep.add_argument("--batch-size", type=int, default=None)
    decode_depth_sweep.add_argument("--cache-len", type=int, default=None)
    decode_depth_sweep.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    decode_depth_sweep.add_argument(
        "--trace",
        action="store_true",
        help="Also capture/execute a generated decode_step trace per depth.",
    )
    decode_depth_sweep.add_argument(
        "--trace-iterations",
        type=int,
        default=1,
        help="Number of execute_trace iterations after capture.",
    )
    decode_depth_sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="Write per-depth profile schemas without opening a device.",
    )
    decode_depth_sweep.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output decode depth sweep report JSON path.",
    )
    decode_depth_sweep.set_defaults(func=_cmd_decode_depth_sweep)

    generate_depth_sweep = subparsers.add_parser(
        "generate-depth-sweep",
        help=(
            "Run generate across increasing decoder depths while preserving "
            "per-depth prefill and generated text evidence."
        ),
    )
    generate_depth_sweep.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    generate_depth_sweep.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local HF model directory. Required for non-dry-run execution.",
    )
    add_prompt_runtime_args(generate_depth_sweep)
    generate_depth_sweep.add_argument(
        "--depths",
        default=None,
        help="Comma-separated depths such as 1,2,4,8,full.",
    )
    generate_depth_sweep.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory for per-depth generate reports.",
    )
    generate_depth_sweep.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Total generated tokens per depth.",
    )
    generate_depth_sweep.add_argument(
        "--prefill-len",
        type=int,
        default=None,
        help="Fixed prompt length for prefill. Defaults to generated config.",
    )
    generate_depth_sweep.add_argument("--batch-size", type=int, default=None)
    generate_depth_sweep.add_argument("--cache-len", type=int, default=None)
    generate_depth_sweep.add_argument("--device", default="p150a")
    generate_depth_sweep.add_argument("--device-id", type=int, default=0)
    generate_depth_sweep.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    generate_depth_sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="Write reports without opening a TTNN device.",
    )
    generate_depth_sweep.add_argument(
        "--require-full-depth",
        action="store_true",
        help="Require the full generated layer depth to pass.",
    )
    generate_depth_sweep.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output generate depth sweep summary report JSON.",
    )
    generate_depth_sweep.set_defaults(func=_cmd_generate_depth_sweep)

    profile = subparsers.add_parser(
        "profile-template",
        help="Profile or dry-run a generated TTNN template.",
    )
    profile.add_argument(
        "--template",
        choices=("mlp_decode",),
        required=True,
        help="Template to profile. Phase 10 supports mlp_decode only.",
    )
    profile.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Generated config.json or template config JSON.",
    )
    profile.add_argument("--warmup", type=int, default=0)
    profile.add_argument("--iterations", type=int, default=1)
    profile.add_argument(
        "--trace",
        action="store_true",
        help="Try TTNN trace capture/execute when hardware APIs exist.",
    )
    profile.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the profiling report schema without opening a device.",
    )
    profile.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to open when hardware is available.",
    )
    profile.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    profile.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output profiling report JSON path.",
    )
    profile.set_defaults(func=_cmd_profile_template)

    build_program = subparsers.add_parser(
        "build-program",
        help="Build a complete TTNN Direct decode program bundle.",
    )
    build_program.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local HF Llama model directory or model id.",
    )
    build_program.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Template seed config JSON.",
    )
    build_program.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for the decode program bundle.",
    )
    build_program.set_defaults(func=_cmd_build_program)

    package_program = subparsers.add_parser(
        "package-program",
        help="Package a generated TTNN Direct Python program directory.",
    )
    package_program.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    package_program.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output package directory.",
    )
    package_program.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print package manifest JSON without writing.",
    )
    package_program.set_defaults(func=_cmd_package_program)

    materialize_parameters = subparsers.add_parser(
        "materialize-parameters",
        help=(
            "Materialize host-side Llama parameters for a generated TTNN "
            "Direct program."
        ),
    )
    materialize_parameters.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local HF Llama model directory containing safetensors files.",
    )
    materialize_parameters.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    materialize_parameters.add_argument(
        "--backend",
        choices=("torch",),
        default="torch",
        help="Host tensor backend. PR-B supports torch only.",
    )
    materialize_parameters.add_argument(
        "--layers",
        default=None,
        help=(
            "Optional comma-separated decoder layer ids to materialize, "
            "for example --layers 0."
        ),
    )
    materialize_parameters.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output parameter materialization report JSON path.",
    )
    materialize_parameters.set_defaults(func=_cmd_materialize_parameters)

    tensorize_parameters = subparsers.add_parser(
        "tensorize-parameters",
        help=(
            "Convert materialized host-side parameters into TTNN tensors. "
            "Supports staged role groups for TTNN Direct bring-up."
        ),
    )
    tensorize_parameters.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    tensorize_parameters.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local HF model directory. Required unless --dry-run is used.",
    )
    tensorize_parameters.add_argument(
        "--roles",
        default="mlp,lm_head",
        help=(
            "Comma-separated role groups to tensorize: "
            "embedding,norm,attention,mlp,lm_head. Defaults to mlp,lm_head."
        ),
    )
    tensorize_parameters.add_argument(
        "--layers",
        default=None,
        help=(
            "Optional comma-separated decoder layer ids to tensorize, "
            "for example --layers 0."
        ),
    )
    tensorize_parameters.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in the tensorization report.",
    )
    tensorize_parameters.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to open when not using --dry-run.",
    )
    tensorize_parameters.add_argument(
        "--dry-run",
        action="store_true",
        help="Write planned dtype/layout conversions without importing TTNN.",
    )
    tensorize_parameters.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output tensorization report JSON path.",
    )
    tensorize_parameters.set_defaults(func=_cmd_tensorize_parameters)

    search = subparsers.add_parser(
        "search",
        help="Enumerate semantic-level TTNN Direct autotune candidates.",
    )
    search.add_argument(
        "--semantic-json",
        type=Path,
        required=True,
        help="Input semantic graph JSON from import-llama.",
    )
    search.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Base template config JSON.",
    )
    search.add_argument(
        "--space",
        type=Path,
        required=True,
        help="Search space JSON.",
    )
    search.add_argument(
        "--metric",
        default="latency_ms",
        help="Metric name to optimize. Phase 14 supports latency_ms only.",
    )
    search.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output search report JSON path.",
    )
    search.add_argument(
        "--candidates-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for candidate configs/plans/codegen. "
            "Defaults to <out-stem>_candidates next to --out."
        ),
    )
    search.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate candidate artifacts without running hardware metrics.",
    )
    search.set_defaults(func=_cmd_search)

    autotune_decode_step = subparsers.add_parser(
        "autotune-decode-step",
        help=(
            "Evaluate a minimal generated decode-step autotune space using "
            "profile-decode-step."
        ),
    )
    autotune_decode_step.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    autotune_decode_step.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Optional local HF model directory. When provided in device mode, "
            "candidate profiles materialize and tensorize real weights."
        ),
    )
    add_prompt_runtime_args(autotune_decode_step)
    autotune_decode_step.add_argument(
        "--space",
        type=Path,
        required=True,
        help="Decode-step autotune search space JSON.",
    )
    autotune_decode_step.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of generated decoder layers to profile per candidate.",
    )
    autotune_decode_step.add_argument("--batch-size", type=int, default=None)
    autotune_decode_step.add_argument("--cache-len", type=int, default=None)
    autotune_decode_step.add_argument(
        "--metric",
        default="latency_ms",
        help=(
            "Metric name. Supports latency_ms, "
            "tokens_per_second_per_user, and aggregate_tokens_per_second."
        ),
    )
    autotune_decode_step.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in reports.",
    )
    autotune_decode_step.add_argument("--device-id", type=int, default=0)
    autotune_decode_step.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    autotune_decode_step.add_argument(
        "--trace",
        action="store_true",
        help="Also capture/execute generated decode_step trace per candidate.",
    )
    autotune_decode_step.add_argument(
        "--trace-iterations",
        type=int,
        default=1,
        help="Number of execute_trace iterations after candidate capture.",
    )
    autotune_decode_step.add_argument(
        "--candidates-dir",
        type=Path,
        default=None,
        help="Optional output directory for candidate configs and reports.",
    )
    autotune_decode_step.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate candidate configs without running TTNN profiles.",
    )
    autotune_decode_step.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output decode-step autotune report JSON path.",
    )
    autotune_decode_step.set_defaults(func=_cmd_autotune_decode_step)

    validate = subparsers.add_parser(
        "validate-direct",
        help=(
            "Run all device-free TTNN Direct dry-run/codegen/package checks "
            "and write a JSON validation report."
        ),
    )
    validate.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local HF Llama model directory or model id.",
    )
    validate.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Template seed config JSON.",
    )
    validate.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for validation artifacts and report.",
    )
    validate.add_argument(
        "--official-template",
        type=Path,
        default=default_official_template_path(),
        help="Official-like reference decode template JSON.",
    )
    validate.add_argument(
        "--official-config",
        type=Path,
        default=default_official_config_path(),
        help="Official or official-like config parity JSON.",
    )
    validate.add_argument(
        "--search-space",
        type=Path,
        default=default_search_space_path(),
        help="Semantic autotune search space JSON.",
    )
    validate.add_argument(
        "--decode-step-search-space",
        type=Path,
        default=default_decode_step_search_space_path(),
        help="Decode-step autotune search space JSON.",
    )
    validate.add_argument(
        "--metric",
        default="latency_ms",
        help="Search metric name. Dry-run search gates support latency_ms.",
    )
    validate.set_defaults(func=_cmd_validate_direct)

    validate_real = subparsers.add_parser(
        "validate-real-decode",
        help=(
            "Run real-weight generated decode validation gates for an "
            "existing TTNN Direct program."
        ),
    )
    validate_real.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="Input directory from build-program.",
    )
    validate_real.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local HF Llama model directory containing safetensors files.",
    )
    add_prompt_runtime_args(validate_real)
    validate_real.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for real decode validation reports.",
    )
    validate_real.add_argument(
        "--decode-step-search-space",
        type=Path,
        default=default_decode_step_search_space_path(),
        help="Decode-step autotune search space JSON.",
    )
    validate_real.add_argument(
        "--official-config",
        type=Path,
        default=None,
        help=(
            "Official or seed TTNN config parity JSON used by the real decode "
            "validation diff gate."
        ),
    )
    validate_real.add_argument("--layers", type=int, default=1)
    validate_real.add_argument("--batch-size", type=int, default=None)
    validate_real.add_argument("--cache-len", type=int, default=None)
    validate_real.add_argument(
        "--max-new-tokens",
        type=int,
        default=2,
        help=(
            "Generated token budget for the prefill+decode generate evidence "
            "step. The first token is materialized from prefill output."
        ),
    )
    validate_real.add_argument(
        "--prefill-len",
        type=int,
        default=None,
        help=(
            "Prompt length used by the prefill+decode generate evidence step."
        ),
    )
    validate_real.add_argument(
        "--device",
        default="p150a",
        help="Target device label recorded in reports.",
    )
    validate_real.add_argument("--device-id", type=int, default=0)
    validate_real.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    validate_real.add_argument(
        "--trace",
        action="store_true",
        help="Capture/execute generated decode_step trace in smoke/profile.",
    )
    validate_real.add_argument(
        "--trace-iterations",
        type=int,
        default=1,
        help="Number of execute_trace iterations after capture.",
    )
    validate_real.add_argument(
        "--metric",
        default="latency_ms",
        help=(
            "Autotune metric name. Supports latency_ms, "
            "tokens_per_second_per_user, and aggregate_tokens_per_second."
        ),
    )
    validate_real.add_argument(
        "--skip-autotune",
        action="store_true",
        help="Run materialize/smoke/profile gates without candidate search.",
    )
    validate_real.add_argument(
        "--guard-device-busy",
        action="store_true",
        help=(
            "Check for external Tenstorrent reset/example processes before "
            "running real device gates and write device_busy evidence instead "
            "of racing a shared board."
        ),
    )
    validate_real.add_argument(
        "--guard-device-health",
        action="store_true",
        help=(
            "Run an isolated TTNN open/write/read health probe before real "
            "device gates and write device_unhealthy evidence if the shared "
            "board is visible but not runnable."
        ),
    )
    validate_real.add_argument(
        "--disable-device-isolation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    validate_real.add_argument(
        "--device-isolation-timeout-seconds",
        type=float,
        default=3600.0,
        help=(
            "Maximum wall-clock seconds for the isolated real validation "
            "subprocess. Use 0 to disable the timeout."
        ),
    )
    validate_real.add_argument(
        "--require-full-decode-step",
        action="store_true",
        help=(
            "Require final full decode-step acceptance scope. This enables "
            "trace capture, full generated depth, generated batch/cache "
            "shape, batch32 decode contract, and decode-shell numeric "
            "reference gates."
        ),
    )
    validate_real.add_argument(
        "--require-model-end-to-end",
        action="store_true",
        help=(
            "Require model end-to-end acceptance. This enables "
            "--require-full-decode-step and additionally fails while runtime "
            "inputs are supplied by smoke/profile synthetic tensors."
        ),
    )
    validate_real.add_argument(
        "--require-official-performance-parity",
        action="store_true",
        help=(
            "Require official performance-parity acceptance. This enables "
            "--require-full-decode-step and --require-official-config-match, "
            "and requires --baseline-reference plus --min-baseline-ratio."
        ),
    )
    validate_real.add_argument(
        "--require-trace",
        action="store_true",
        help=(
            "Require smoke/profile decode-step reports to capture and "
            "execute a TTNN trace before accepting the validation report."
        ),
    )
    validate_real.add_argument(
        "--require-official-config-match",
        action="store_true",
        help=(
            "Require the generated program config to exactly match the "
            "official parity config before accepting validation."
        ),
    )
    validate_real.add_argument(
        "--require-full-depth",
        action="store_true",
        help=(
            "Require --layers to equal the generated program num_layers "
            "before accepting validation."
        ),
    )
    validate_real.add_argument(
        "--require-program-runtime-shape",
        action="store_true",
        help=(
            "Require resolved batch/cache dimensions to match the generated "
            "program config before accepting validation."
        ),
    )
    validate_real.add_argument(
        "--require-batch32-decode-step",
        action="store_true",
        help=(
            "Require the generated decode-step validation contract to run "
            "with batch size 32 before accepting validation."
        ),
    )
    validate_real.add_argument(
        "--min-tokens-per-second-per-user",
        type=float,
        default=None,
        help=(
            "Minimum profile throughput gate for acceptance. The value is "
            "checked against profile_decode_step.throughput_summary."
        ),
    )
    validate_real.add_argument(
        "--baseline-tokens-per-second-per-user",
        type=float,
        default=None,
        help=(
            "Reference decode throughput used to record a performance ratio "
            "in validate-real-decode evidence."
        ),
    )
    validate_real.add_argument(
        "--performance-baselines",
        type=Path,
        default=default_performance_baselines_path(),
        help=(
            "Performance baseline reference JSON used when "
            "--baseline-reference is set."
        ),
    )
    validate_real.add_argument(
        "--baseline-reference",
        default=None,
        help=(
            "Baseline id from --performance-baselines. When set, "
            "validate-real-decode uses that entry's decode t/s/u as the "
            "baseline and records the entry in evidence."
        ),
    )
    validate_real.add_argument(
        "--min-baseline-ratio",
        type=float,
        default=None,
        help=(
            "Minimum observed/baseline tokens-per-second-per-user ratio "
            "required for validate-real-decode acceptance."
        ),
    )
    validate_real.add_argument(
        "--decode-shell-pcc-threshold",
        type=float,
        default=0.99,
        help=(
            "Minimum final-hidden PCC for the attention-disabled decode shell "
            "gate when a torch numeric reference can run."
        ),
    )
    validate_real.add_argument(
        "--require-decode-shell-numeric-reference",
        action="store_true",
        help=(
            "Require the attention-disabled decode shell gate to run and pass "
            "its torch numeric reference before accepting validation."
        ),
    )
    validate_real.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Write the real-decode validation report schema without loading "
            "safetensors or opening a TTNN device."
        ),
    )
    validate_real.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Write a real-decode preflight report and exit before loading "
            "weights, opening a TTNN device, or running runtime gates."
        ),
    )
    validate_real.set_defaults(func=_cmd_validate_real_decode)
    return parser


def _cmd_import_llama(args: argparse.Namespace) -> int:
    graph = import_hf_llama(
        args.model_path,
        mode=args.mode,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_cache_len=args.max_cache_len,
        generation_mode=args.generation_mode,
    )
    dump_graph_json(graph, args.out)
    suffix = " (dry-run)" if args.dry_run else ""
    print(f"wrote Llama semantic graph{suffix}: {args.out}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    graph = load_graph_json(args.semantic_json)
    config = load_template_config(args.config)
    plan = build_execution_plan(graph, config)
    dump_execution_plan(plan, args.out)
    print(f"wrote TTNN Direct execution plan: {args.out}")
    return 0


def _cmd_codegen_python(args: argparse.Namespace) -> int:
    plan = load_execution_plan(args.plan_json)
    if args.dry_run:
        print(json.dumps(dry_run_report(plan, args.out_dir), indent=2))
        return 0

    paths = write_python_ttnn_skeleton(plan, args.out_dir)
    print(f"wrote Python TTNN skeleton: {args.out_dir}")
    for name in sorted(paths):
        print(f"  {name}: {paths[name]}")
    return 0


def _cmd_emit_config(args: argparse.Namespace) -> int:
    graph = load_graph_json(args.semantic_json)
    if args.dry_run:
        print(
            json.dumps(
                parameter_config_dry_run_report(
                    graph,
                    recipe=args.recipe,
                    lm_head_split_count=args.lm_head_split_count,
                    kv_page_block_size=args.kv_page_block_size,
                ),
                indent=2,
            )
        )
        return 0

    config = emit_parameter_config(
        graph,
        recipe=args.recipe,
        lm_head_split_count=args.lm_head_split_count,
        kv_page_block_size=args.kv_page_block_size,
    )
    dump_parameter_config(config, args.out)
    print(f"wrote parameter metadata config: {args.out}")
    return 0


def _cmd_diff_plan(args: argparse.Namespace) -> int:
    plan = load_execution_plan(args.ours)
    official_template = load_official_template(args.official_template)
    diff = diff_plan_against_official(plan, official_template)
    dump_plan_diff(diff, args.out)
    print(f"wrote TTNN Direct plan diff: {args.out}")
    return 0


def _cmd_diff_official_config(args: argparse.Namespace) -> int:
    diff = diff_official_config(args.ours, args.official)
    dump_config_diff(diff, args.out)
    print(f"wrote TTNN Direct official config diff: {args.out}")
    print(f"  status: {diff['status']}")
    print(f"  issues: {diff['summary']['issue_count']}")
    return 0


def _cmd_prepare_artifacts(args: argparse.Namespace) -> int:
    graph = load_graph_json(args.semantic_json)
    config = json.loads(args.config.read_text())
    paths = prepare_offline_artifacts(
        args.model_path,
        graph,
        config,
        args.out_dir,
    )
    print(f"wrote TTNN Direct artifact manifests: {args.out_dir}")
    for name in sorted(paths):
        print(f"  {name}: {paths[name]}")
    return 0


def _cmd_smoke_mlp(args: argparse.Namespace) -> int:
    report = run_smoke_mlp(
        out=args.out,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        dtype_seed=args.dtype_seed,
        dry_run=args.dry_run,
        pcc_threshold=args.pcc_threshold,
        seed=args.seed,
    )
    print(f"wrote MLP smoke report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_smoke_decode_shell(args: argparse.Namespace) -> int:
    report = run_smoke_decode_shell(
        out=args.out,
        program_dir=args.program_dir,
        layers=args.layers,
        disable_attention=args.disable_attention,
        device=args.device,
        device_id=args.device_id,
        model_path=args.model_path,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        dry_run=args.dry_run,
        pcc_threshold=args.pcc_threshold,
    )
    print(f"wrote decode shell smoke report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_smoke_attention_primitive(args: argparse.Namespace) -> int:
    report = run_smoke_attention_primitive(
        out=args.out,
        primitive=args.primitive,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        max_cache_len=args.max_cache_len,
        dtype_seed=args.dtype_seed,
        dry_run=args.dry_run,
    )
    print(f"wrote attention primitive smoke report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_smoke_attention_layer(args: argparse.Namespace) -> int:
    report = run_smoke_attention_layer(
        out=args.out,
        program_dir=args.program_dir,
        layer=args.layer,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype_seed=args.dtype_seed,
        dry_run=args.dry_run,
    )
    print(f"wrote attention layer smoke report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_smoke_single_layer_decode(args: argparse.Namespace) -> int:
    report = run_smoke_single_layer_decode(
        out=args.out,
        program_dir=args.program_dir,
        model_path=args.model_path,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype_seed=args.dtype_seed,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        dry_run=args.dry_run,
    )
    print(f"wrote single-layer decode smoke report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_smoke_prefill(args: argparse.Namespace) -> int:
    report = run_smoke_prefill(
        out=args.out,
        program_dir=args.program_dir,
        layers=args.layers,
        prefill_len=args.prefill_len,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype_seed=args.dtype_seed,
        dry_run=args.dry_run,
    )
    print(f"wrote prefill smoke report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_smoke_decode_step(args: argparse.Namespace) -> int:
    report = run_smoke_decode_step(
        out=args.out,
        program_dir=args.program_dir,
        layers=args.layers,
        model_path=args.model_path,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype_seed=args.dtype_seed,
        trace=args.trace,
        trace_iterations=args.trace_iterations,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        dry_run=args.dry_run,
    )
    print(f"wrote decode-step smoke report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_profile_decode_step(args: argparse.Namespace) -> int:
    report = profile_decode_step(
        out=args.out,
        program_dir=args.program_dir,
        layers=args.layers,
        model_path=args.model_path,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype_seed=args.dtype_seed,
        trace=args.trace,
        trace_iterations=args.trace_iterations,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        dry_run=args.dry_run,
    )
    print(f"wrote decode-step profile report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_prompt_decode_loop(args: argparse.Namespace) -> int:
    report = run_prompt_decode_loop(
        out=args.out,
        program_dir=args.program_dir,
        model_path=args.model_path,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        decode_steps=(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else args.decode_steps
        ),
        layers=args.layers,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype_seed=args.dtype_seed,
        dry_run=args.dry_run,
    )
    print(f"wrote prompt decode loop report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_generate(args: argparse.Namespace) -> int:
    report = run_generate(
        out=args.out,
        program_dir=args.program_dir,
        model_path=args.model_path,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        max_new_tokens=args.max_new_tokens,
        layers=args.layers,
        prefill_len=args.prefill_len,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype_seed=args.dtype_seed,
        dry_run=args.dry_run,
    )
    print(f"wrote generate report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_profile_generate(args: argparse.Namespace) -> int:
    report = run_profile_generate(
        out=args.out,
        program_dir=args.program_dir,
        model_path=args.model_path,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        max_new_tokens=args.max_new_tokens,
        layers=args.layers,
        prefill_len=args.prefill_len,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype_seed=args.dtype_seed,
        dry_run=args.dry_run,
        generate_report=args.generate_report,
    )
    print(f"wrote generate profile report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_decode_depth_sweep(args: argparse.Namespace) -> int:
    report = run_decode_depth_sweep(
        out=args.out,
        program_dir=args.program_dir,
        depths=args.depths,
        model_path=args.model_path,
        profiles_dir=args.profiles_dir,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype_seed=args.dtype_seed,
        trace=args.trace,
        trace_iterations=args.trace_iterations,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        dry_run=args.dry_run,
    )
    print(f"wrote decode depth sweep report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_generate_depth_sweep(args: argparse.Namespace) -> int:
    report = run_generate_depth_sweep(
        out=args.out,
        program_dir=args.program_dir,
        depths=args.depths,
        model_path=args.model_path,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        reports_dir=args.reports_dir,
        max_new_tokens=args.max_new_tokens,
        prefill_len=args.prefill_len,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        device=args.device,
        device_id=args.device_id,
        dtype_seed=args.dtype_seed,
        dry_run=args.dry_run,
        require_full_depth=args.require_full_depth,
    )
    print(f"wrote generate depth sweep report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1


def _cmd_profile_template(args: argparse.Namespace) -> int:
    report = profile_template(
        template=args.template,
        config_path=args.config,
        out=args.out,
        warmup=args.warmup,
        iterations=args.iterations,
        trace=args.trace,
        dry_run=args.dry_run,
        device_id=args.device_id,
        dtype_seed=args.dtype_seed,
    )
    print(f"wrote template profiling report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("status") in {"dry_run", "profiled"} else 1


def _cmd_build_program(args: argparse.Namespace) -> int:
    config = load_template_config(args.config)
    graph = import_hf_llama(
        args.model_path,
        mode="decode",
        batch_size=int(config["batch_size"]),
        seq_len=int(config["decode_seq_len"]),
        max_cache_len=int(config["max_cache_len"]),
        generation_mode=(
            "greedy"
            if config["generation_template"] == "device_argmax_greedy"
            else "sampling"
        ),
    )
    plan = build_execution_plan(graph, config)
    paths = write_decode_program_bundle(
        graph=graph,
        plan=plan,
        template_config=config,
        model_path=args.model_path,
        out_dir=args.out_dir,
    )
    print(f"wrote TTNN Direct decode program: {args.out_dir}")
    for name in sorted(paths):
        print(f"  {name}: {paths[name]}")
    return 0


def _cmd_package_program(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            json.dumps(
                package_dry_run_report(args.program_dir, args.out_dir),
                indent=2,
            )
        )
        return 0

    paths = package_ttnn_direct_program(args.program_dir, args.out_dir)
    print(f"wrote TTNN Direct program package: {args.out_dir}")
    for name in sorted(paths):
        print(f"  {name}: {paths[name]}")
    return 0


def _cmd_materialize_parameters(args: argparse.Namespace) -> int:
    try:
        report = materialize_parameters_from_program(
            model_path=args.model_path,
            program_dir=args.program_dir,
            backend=args.backend,
            layers=parse_layer_ids(args.layers),
            out=args.out,
        )
    except ParameterMaterializationError as exc:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "status": "fail",
            "backend": args.backend,
            "model_path": str(args.model_path),
            "program_dir": str(args.program_dir),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote parameter materialization report: {args.out}")
        print(f"  status: {report['status']}")
        return 1

    print(f"wrote parameter materialization report: {args.out}")
    print(f"  status: {report['status']}")
    return 0


def _cmd_tensorize_parameters(args: argparse.Namespace) -> int:
    try:
        roles = parse_role_groups(args.roles)
        layers = parse_layer_ids(args.layers)
        if args.dry_run:
            report = tensorize_parameters_from_program_dry_run(
                program_dir=args.program_dir,
                roles=roles,
                layers=layers,
                device=args.device,
                out=args.out,
            )
            print(f"wrote TTNN tensorization report: {args.out}")
            print(f"  status: {report['status']}")
            return 0

        if args.model_path is None:
            raise TTNNTensorizationError(
                "--model-path is required unless --dry-run is used"
            )
        parameter_config = load_parameter_config_from_program(args.program_dir)
        torch_params = load_llama_parameters_from_manifests(
            model_path=args.model_path,
            weights_manifest=args.program_dir / "weights_manifest.json",
            config=args.program_dir / "config.json",
            tensor_backend="torch",
            layers=layers,
        )
        ttnn = __import__("ttnn")
        device = _open_ttnn_device(ttnn, args.device_id)
        try:
            result = to_ttnn_parameters(
                torch_params,
                device,
                parameter_config,
                roles=roles,
                layers=layers,
                ttnn_module=ttnn,
            )
        finally:
            _close_ttnn_device(ttnn, device)
        report = {
            **result.report,
            "program_dir": str(args.program_dir),
            "model_path": str(args.model_path),
            "device_label": args.device,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote TTNN tensorization report: {args.out}")
        print(f"  status: {report['status']}")
        return 0
    except (
        ModuleNotFoundError,
        ParameterMaterializationError,
        TTNNTensorizationError,
    ) as exc:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "status": "fail",
            "dry_run": bool(args.dry_run),
            "program_dir": str(args.program_dir),
            "model_path": str(args.model_path) if args.model_path else None,
            "device": args.device,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote TTNN tensorization report: {args.out}")
        print(f"  status: {report['status']}")
        return 1


def _cmd_search(args: argparse.Namespace) -> int:
    graph = load_graph_json(args.semantic_json)
    base_config = load_template_config(args.base_config)
    space = load_search_space(args.space)
    report = run_lm_head_search(
        graph=graph,
        base_config=base_config,
        space=space,
        metric=args.metric,
        out=args.out,
        candidates_dir=args.candidates_dir,
        dry_run=args.dry_run,
    )
    dump_search_report(report, args.out)
    print(f"wrote TTNN Direct search report: {args.out}")
    print(f"  candidates: {report['candidates_dir']}")
    return 0


def _cmd_autotune_decode_step(args: argparse.Namespace) -> int:
    space = load_search_space(args.space)
    report = run_decode_step_autotune(
        program_dir=args.program_dir,
        space=space,
        out=args.out,
        layers=args.layers,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        model_path=args.model_path,
        metric=args.metric,
        candidates_dir=args.candidates_dir,
        dry_run=args.dry_run,
        device=args.device,
        device_id=args.device_id,
        dtype_seed=args.dtype_seed,
        trace=args.trace,
        trace_iterations=args.trace_iterations,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
    )
    dump_search_report(report, args.out)
    print(f"wrote TTNN Direct decode-step autotune report: {args.out}")
    print(f"  candidates: {report['candidates_dir']}")
    return 0 if report.get("dry_run") or report.get("best") is not None else 1


def _open_ttnn_device(ttnn: object, device_id: int) -> object:
    open_device = getattr(ttnn, "open_device", None)
    if open_device is None:
        return device_id
    return open_device(device_id=device_id)


def _close_ttnn_device(ttnn: object, device: object) -> None:
    close_device = getattr(ttnn, "close_device", None)
    if close_device is not None:
        close_device(device)


def _cmd_validate_direct(args: argparse.Namespace) -> int:
    report = validate_direct(
        model_path=args.model_path,
        config_path=args.config,
        out_dir=args.out_dir,
        official_template_path=args.official_template,
        official_config_path=args.official_config,
        search_space_path=args.search_space,
        decode_step_search_space_path=args.decode_step_search_space,
        metric=args.metric,
    )
    print(
        "wrote TTNN Direct validation report: "
        f"{args.out_dir / 'validation_report.json'}"
    )
    print(f"  status: {report['status']}")
    return 0 if report["status"] == "pass" else 1


def _cmd_validate_real_decode(args: argparse.Namespace) -> int:
    if args.preflight_only:
        report = preflight_real_decode(
            program_dir=args.program_dir,
            model_path=args.model_path,
            out=args.out_dir / "real_decode_preflight_report.json",
            official_config_path=args.official_config,
            decode_step_search_space_path=args.decode_step_search_space,
            performance_baselines_path=args.performance_baselines,
            layers=args.layers,
            batch_size=args.batch_size,
            cache_len=args.cache_len,
            device=args.device,
            device_id=args.device_id,
            trace=args.trace,
            trace_iterations=args.trace_iterations,
            metric=args.metric,
            skip_autotune=args.skip_autotune,
            require_full_decode_step=args.require_full_decode_step,
            require_model_end_to_end=args.require_model_end_to_end,
            require_official_performance_parity=(
                args.require_official_performance_parity
            ),
            require_trace=args.require_trace,
            require_official_config_match=args.require_official_config_match,
            require_full_depth=args.require_full_depth,
            require_program_runtime_shape=args.require_program_runtime_shape,
            require_batch32_decode_step=args.require_batch32_decode_step,
            min_tokens_per_second_per_user=(
                args.min_tokens_per_second_per_user
            ),
            baseline_tokens_per_second_per_user=(
                args.baseline_tokens_per_second_per_user
            ),
            baseline_reference=args.baseline_reference,
            min_baseline_ratio=args.min_baseline_ratio,
            decode_shell_pcc_threshold=args.decode_shell_pcc_threshold,
            require_decode_shell_numeric_reference=(
                args.require_decode_shell_numeric_reference
            ),
            prompt=args.prompt,
            tokenizer_path=args.tokenizer_path,
            max_new_tokens=args.max_new_tokens,
            prefill_len=args.prefill_len,
            guard_device_busy=args.guard_device_busy,
            guard_device_health=args.guard_device_health,
        )
        report_path = args.out_dir / "real_decode_preflight_report.json"
        print(json.dumps({"status": report["status"], "report": str(report_path)}, indent=2))
        return 0 if report["status"] == "pass" else 1

    if (
        args.guard_device_health
        and not args.disable_device_isolation
        and not args.dry_run
    ):
        return _cmd_validate_real_decode_isolated(args)

    report = validate_real_decode(
        program_dir=args.program_dir,
        model_path=args.model_path,
        out_dir=args.out_dir,
        official_config_path=args.official_config,
        decode_step_search_space_path=args.decode_step_search_space,
        layers=args.layers,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        max_new_tokens=args.max_new_tokens,
        prefill_len=args.prefill_len,
        device=args.device,
        device_id=args.device_id,
        dtype_seed=args.dtype_seed,
        trace=args.trace,
        trace_iterations=args.trace_iterations,
        metric=args.metric,
        dry_run=args.dry_run,
        skip_autotune=args.skip_autotune,
        require_full_decode_step=args.require_full_decode_step,
        require_model_end_to_end=args.require_model_end_to_end,
        require_official_performance_parity=(
            args.require_official_performance_parity
        ),
        require_trace=args.require_trace,
        require_official_config_match=args.require_official_config_match,
        require_full_depth=args.require_full_depth,
        require_program_runtime_shape=args.require_program_runtime_shape,
        require_batch32_decode_step=args.require_batch32_decode_step,
        min_tokens_per_second_per_user=args.min_tokens_per_second_per_user,
        baseline_tokens_per_second_per_user=(
            args.baseline_tokens_per_second_per_user
        ),
        performance_baselines_path=args.performance_baselines,
        baseline_reference=args.baseline_reference,
        min_baseline_ratio=args.min_baseline_ratio,
        decode_shell_pcc_threshold=args.decode_shell_pcc_threshold,
        require_decode_shell_numeric_reference=(
            args.require_decode_shell_numeric_reference
        ),
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        guard_device_busy=args.guard_device_busy,
        guard_device_health=args.guard_device_health,
    )
    print(
        "wrote TTNN Direct real decode validation report: "
        f"{args.out_dir / 'real_decode_validation_report.json'}"
    )
    print(f"  status: {report['status']}")
    if report["status"] in {"pass", "dry_run"}:
        return 0
    if report["status"] == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 1


def _cmd_validate_real_decode_isolated(args: argparse.Namespace) -> int:
    raw_argv = list(getattr(args, "_raw_argv", []))
    if "--disable-device-isolation" not in raw_argv:
        raw_argv.append("--disable-device-isolation")
    command = [
        sys.executable,
        "-m",
        "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
        *raw_argv,
    ]
    timeout_seconds = (
        float(args.device_isolation_timeout_seconds)
        if args.device_isolation_timeout_seconds
        and float(args.device_isolation_timeout_seconds) > 0.0
        else None
    )
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = getattr(exc, "stdout", None) or getattr(exc, "output", None)
        stderr = getattr(exc, "stderr", None)
        stdout_text = _coerce_subprocess_text(stdout)
        stderr_text = _coerce_subprocess_text(stderr)
        if stdout_text:
            print(stdout_text, end="")
        if stderr_text:
            print(stderr_text, end="", file=sys.stderr)
        report = recover_real_decode_process_timeout(
            out_dir=args.out_dir,
            timeout_seconds=float(exc.timeout),
            stdout=stdout,
            stderr=stderr,
            command=command,
        )
        report_path = args.out_dir / "real_decode_validation_report.json"
        print(
            "wrote TTNN Direct real decode validation report: "
            f"{report_path}"
        )
        print(f"  status: {report['status']}")
        return 1
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode in {0, 1, 2}:
        return int(result.returncode)

    report = recover_real_decode_process_failure(
        out_dir=args.out_dir,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=command,
    )
    report_path = args.out_dir / "real_decode_validation_report.json"
    print(
        "wrote TTNN Direct real decode validation report: "
        f"{report_path}"
    )
    print(f"  status: {report['status']}")
    return 1


def _coerce_subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    args._raw_argv = raw_argv
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
