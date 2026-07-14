from __future__ import annotations

import argparse


DIAGNOSE_STAGES = (
    "mlp",
    "attention-primitive",
    "attention-layer",
    "prefill",
    "decode-step",
    "decode-loop-legacy",
    "depth-sweep",
    "generate-depth-sweep",
    "autotune",
    "benchmark-parity",
    "execution-graph-diff",
)


def run_stage(args: argparse.Namespace) -> dict[str, object]:
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
            address_space_limit_bytes=int(
                args.address_space_limit_gb * 1_000_000_000
            ),
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
            timeout_seconds=args.benchmark_timeout,
            address_space_limit_bytes=int(
                args.address_space_limit_gb * 1_000_000_000
            ),
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
    if args.stage == "decode-loop-legacy":
        _require_args(args, "program_dir")
        from ..decode_loop import run_prompt_decode_loop

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
            device=args.device,
            device_id=args.device_id,
            dtype_seed=args.dtype_seed,
            dry_run=args.dry_run,
            require_full_depth=args.require_full_depth,
            isolate_depth_steps=not args.dry_run,
        )
    if args.stage == "autotune":
        _require_args(args, "model_path", "config")
        if not args.dry_run:
            _require_args(args, "prompt")
        from .autotune import run_layered_autotune

        return run_layered_autotune(
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
            confirm_warmup=args.confirm_warmup,
            confirm_iterations=args.confirm_iterations,
            min_relative_improvement=args.min_relative_improvement,
            dry_run=args.dry_run,
            resume=not args.no_resume,
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
