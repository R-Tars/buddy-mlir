from __future__ import annotations

import argparse
from pathlib import Path

from ..reports.contracts import ATTENTION_PRIMITIVES

DIAGNOSE_STAGES = (
    "mlp",
    "attention-primitive",
    "attention-layer",
    "prefill",
    "decode-shell",
    "decode-step",
    "decode-step-profile",
    "decode-loop-legacy",
    "depth-sweep",
    "generate-depth-sweep",
    "autotune",
    "autotune-profiler-audit",
    "benchmark-parity",
    "execution-graph-diff",
    "performance-correctness",
    "template-profile",
)


def add_diagnose_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", choices=DIAGNOSE_STAGES, required=True)
    parser.add_argument("--program-dir", type=Path, default=None)
    parser.add_argument("--buddy-program", type=Path, default=None)
    parser.add_argument("--official-tt-metal-root", type=Path, default=None)
    parser.add_argument("--official-python", type=Path, default=None)
    parser.add_argument("--official-release-root", type=Path, default=None)
    parser.add_argument("--official-release-python", type=Path, default=None)
    parser.add_argument("--official-release-runtime-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--template", choices=("mlp_decode",), default=None)
    parser.add_argument("--profiler-csv", type=Path, default=None)
    parser.add_argument("--profile-report", type=Path, default=None)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None)
    prompt_group.add_argument(
        "--input-prompts",
        "--input_prompts",
        type=Path,
        default=None,
    )
    parser.add_argument("--instruct", action="store_true")
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="p150a")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cache-len", type=int, default=None)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--prefill-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--profiles-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--primitive", choices=ATTENTION_PRIMITIVES)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--max-cache-len", type=int, default=1024)
    parser.add_argument("--dtype-seed", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--disable-attention", action="store_true")
    parser.add_argument("--pcc-threshold", type=float, default=0.99)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-iterations", type=int, default=1)
    parser.add_argument("--require-full-depth", action="store_true")
    parser.add_argument("--candidates-dir", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--accuracy-tokens", type=int, default=500)
    parser.add_argument("--page-block-size", type=int, default=32)
    parser.add_argument("--benchmark-timeout", type=float, default=3600.0)
    parser.add_argument("--address-space-limit-gb", type=float, default=95.0)
    parser.add_argument("--min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--no-resume", action="store_true")


def run_stage(args: argparse.Namespace) -> dict[str, object]:
    if args.stage == "template-profile":
        _require_args(args, "template", "config")
        from .template_profile import profile_template

        return profile_template(
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
    if args.stage == "autotune-profiler-audit":
        _require_args(
            args,
            "program_dir",
            "model_path",
            "input_prompts",
        )
        from .autotune_profiler_audit import run_autotune_profiler_audit

        return run_autotune_profiler_audit(
            out=args.out,
            program_dir=args.program_dir,
            model_path=args.model_path,
            input_prompts=args.input_prompts,
            tokenizer_path=args.tokenizer_path,
            instruct=args.instruct,
            device=args.device,
            device_id=args.device_id,
            timeout_seconds=args.benchmark_timeout,
            profiler_csv=args.profiler_csv,
            profile_report=args.profile_report,
        )
    if args.stage == "performance-correctness":
        _require_args(
            args,
            "buddy_program",
            "official_tt_metal_root",
            "model_path",
        )
        from .performance_correctness import run_performance_correctness

        return run_performance_correctness(
            out=args.out,
            buddy_program=args.buddy_program,
            official_tt_metal_root=args.official_tt_metal_root,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            official_python=args.official_python,
            token_count=args.accuracy_tokens,
            layers=args.layers,
            batch_size=args.batch_size or 32,
            prefill_len=args.prefill_len or 512,
            cache_len=args.cache_len or 1024,
            page_block_size=args.page_block_size,
            device=args.device,
            device_id=args.device_id,
            timeout_seconds=args.benchmark_timeout,
            address_space_limit_bytes=int(args.address_space_limit_gb * 1_000_000_000),
            dry_run=args.dry_run,
        )
    if args.stage == "execution-graph-diff":
        _require_args(
            args,
            "buddy_program",
            "official_tt_metal_root",
            "model_path",
        )
        from .execution_graph_diff import run_execution_graph_diff

        return run_execution_graph_diff(
            out=args.out,
            buddy_program=args.buddy_program,
            official_tt_metal_root=args.official_tt_metal_root,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            input_prompts=args.input_prompts,
            batch_size=args.batch_size or 32,
            prefill_len=args.prefill_len or 256,
            cache_len=args.cache_len or 1024,
            page_block_size=args.page_block_size,
            device=args.device,
            device_id=args.device_id,
            official_python=args.official_python,
            timeout_seconds=args.benchmark_timeout,
            address_space_limit_bytes=int(args.address_space_limit_gb * 1_000_000_000),
            dry_run=args.dry_run,
        )
    if args.stage == "benchmark-parity":
        _require_args(
            args,
            "buddy_program",
            "official_tt_metal_root",
            "model_path",
        )
        from .benchmark_parity import run_benchmark_parity

        return run_benchmark_parity(
            out=args.out,
            buddy_program=args.buddy_program,
            official_tt_metal_root=args.official_tt_metal_root,
            model_path=args.model_path,
            input_prompts=args.input_prompts,
            tokenizer_path=args.tokenizer_path,
            batch_size=args.batch_size or 32,
            prefill_len=args.prefill_len or 128,
            cache_len=args.cache_len or 1024,
            page_block_size=args.page_block_size,
            warmup=args.warmup,
            iterations=args.iterations,
            repetitions=args.repetitions,
            device=args.device,
            device_id=args.device_id,
            official_python=args.official_python,
            official_release_root=args.official_release_root,
            official_release_python=args.official_release_python,
            official_release_runtime_root=(args.official_release_runtime_root),
            timeout_seconds=args.benchmark_timeout,
            address_space_limit_bytes=int(args.address_space_limit_gb * 1_000_000_000),
            dry_run=args.dry_run,
        )
    if args.stage == "mlp":
        _require_args(
            args,
            "batch_size",
            "hidden_size",
            "intermediate_size",
        )
        from ..smoke_mlp import run_smoke_mlp

        return run_smoke_mlp(
            out=args.out,
            device=args.device,
            device_id=args.device_id,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            dtype_seed=args.dtype_seed,
            dry_run=args.dry_run,
        )
    if args.stage == "attention-primitive":
        _require_args(
            args,
            "primitive",
            "batch_size",
            "hidden_size",
            "num_heads",
            "num_kv_heads",
            "head_dim",
        )
        from ..smoke_attention_primitive import run_smoke_attention_primitive

        return run_smoke_attention_primitive(
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
    if args.stage == "attention-layer":
        _require_args(args, "program_dir")
        from ..smoke_attention_layer import run_smoke_attention_layer

        return run_smoke_attention_layer(
            out=args.out,
            program_dir=args.program_dir,
            layer=max(0, int(args.layers) - 1),
            device=args.device,
            device_id=args.device_id,
            batch_size=args.batch_size,
            cache_len=args.cache_len,
            dtype_seed=args.dtype_seed,
            dry_run=args.dry_run,
        )
    if args.stage == "prefill":
        _require_args(args, "program_dir")
        from ..smoke_prefill import run_smoke_prefill

        return run_smoke_prefill(
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
    if args.stage == "decode-step":
        _require_args(args, "program_dir")
        from ..smoke_single_layer_decode import run_smoke_decode_step

        return run_smoke_decode_step(
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
    if args.stage == "decode-shell":
        _require_args(args, "program_dir")
        from ..smoke_decode_shell import run_smoke_decode_shell

        return run_smoke_decode_shell(
            out=args.out,
            program_dir=args.program_dir,
            layers=args.layers,
            disable_attention=args.disable_attention,
            model_path=args.model_path,
            device=args.device,
            device_id=args.device_id,
            batch_size=args.batch_size,
            cache_len=args.cache_len,
            prompt=args.prompt,
            tokenizer_path=args.tokenizer_path,
            pcc_threshold=args.pcc_threshold,
            dry_run=args.dry_run,
        )
    if args.stage == "decode-step-profile":
        _require_args(args, "program_dir")
        from ..smoke_single_layer_decode import profile_decode_step

        return profile_decode_step(
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
    if args.stage == "decode-loop-legacy":
        _require_args(args, "program_dir")
        from .legacy_decode_loop import run_prompt_decode_loop

        return run_prompt_decode_loop(
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
    if args.stage == "depth-sweep":
        _require_args(args, "program_dir")
        from .decode_depth_sweep import run_decode_depth_sweep

        return run_decode_depth_sweep(
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
            isolate_depth_steps=not args.dry_run,
        )
    if args.stage == "generate-depth-sweep":
        _require_args(args, "program_dir")
        from .generate_depth_sweep import run_generate_depth_sweep

        return run_generate_depth_sweep(
            out=args.out,
            program_dir=args.program_dir,
            depths=args.depths,
            model_path=args.model_path,
            prompt=args.prompt,
            tokenizer_path=args.tokenizer_path,
            max_new_tokens=args.max_new_tokens,
            prefill_len=args.prefill_len,
            batch_size=args.batch_size,
            cache_len=args.cache_len,
            reports_dir=args.reports_dir,
            device=args.device,
            device_id=args.device_id,
            dtype_seed=args.dtype_seed,
            dry_run=args.dry_run,
            require_full_depth=args.require_full_depth,
            isolate_depth_steps=not args.dry_run,
        )
    if args.stage == "autotune":
        _require_args(args, "model_path", "config")
        candidate_gate_runner = None
        if not args.dry_run:
            _require_args(args, "prompt", "official_tt_metal_root")
            from .candidate_quality_gate import build_candidate_quality_gate

            candidates_root = args.candidates_dir or (
                args.out.parent / f"{args.out.stem}_candidates"
            )
            candidate_gate_runner = build_candidate_quality_gate(
                official_tt_metal_root=args.official_tt_metal_root,
                official_python=args.official_python,
                accuracy_tokens=args.accuracy_tokens,
                gate_root=candidates_root / "quality_gates",
            )
        from ..autotune.campaign import run_autotune_campaign

        return run_autotune_campaign(
            model_path=args.model_path,
            config_path=args.config,
            out=args.out,
            prompt=args.prompt,
            tokenizer_path=args.tokenizer_path,
            candidates_dir=args.candidates_dir,
            layers=args.layers,
            prefill_len=args.prefill_len,
            batch_size=args.batch_size,
            cache_len=args.cache_len,
            device=args.device,
            device_id=args.device_id,
            dtype_seed=args.dtype_seed,
            warmup=args.warmup,
            iterations=args.iterations,
            repetitions=args.repetitions,
            min_relative_improvement=args.min_relative_improvement,
            dry_run=args.dry_run,
            resume=not args.no_resume,
            candidate_gate_runner=candidate_gate_runner,
        )
    raise ValueError(f"unsupported diagnose stage: {args.stage}")


def _require_args(args: argparse.Namespace, *names: str) -> None:
    missing = [
        "--" + name.replace("_", "-")
        for name in names
        if getattr(args, name, None) is None
    ]
    if missing:
        raise ValueError(
            f"diagnose --stage {args.stage} requires " + ", ".join(missing)
        )
