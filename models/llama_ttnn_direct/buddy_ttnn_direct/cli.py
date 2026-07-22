from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .codegen.program import write_decode_program_bundle
from .runtime.generate import run_generate
from .runtime.prefill_profile import run_profile_prefill_steady
from .runtime.profile import run_profile_decode_steady, run_profile_generate
from .reports.validation import (
    validate_device,
    validate_dryrun,
    validate_functional,
    validate_performance,
)
from .runtime.errors import NO_TTNN_DEVICE_MESSAGE
from .semantic.importer_hf_llama import import_hf_llama
from .templates.registry import build_execution_plan, load_template_config

PRODUCT_COMMANDS = (
    "build",
    "generate",
    "profile",
    "validate",
    "inspect",
    "diagnose",
)


def _add_prompt_runtime_args(
    command: argparse.ArgumentParser,
    *,
    batch_prompts: bool = False,
) -> None:
    prompt_group = command.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        default=None,
        help="Prompt text used by the HF tokenizer.",
    )
    if batch_prompts:
        prompt_group.add_argument(
            "--input-prompts",
            "--input_prompts",
            type=Path,
            default=None,
            help="JSON prompt corpus for batch inference.",
        )
        command.add_argument(
            "--instruct",
            action="store_true",
            help="Apply the tokenizer chat template.",
        )
    command.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer directory or model id.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="buddy-ttnn-direct",
        description="Buddy-TTNN Direct model tooling.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{" + ",".join(PRODUCT_COMMANDS) + "}",
    )

    build = subparsers.add_parser(
        "build",
        help="Build a TTNN Direct generated program bundle.",
    )
    build.add_argument("--model-path", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    build.set_defaults(func=_cmd_build_program)

    generate = subparsers.add_parser(
        "generate",
        help="Run prompt prefill followed by generated decode steps.",
    )
    generate.add_argument("--program-dir", type=Path, required=True)
    generate.add_argument("--model-path", type=Path, default=None)
    _add_prompt_runtime_args(generate, batch_prompts=True)
    generate.add_argument("--max-new-tokens", type=int, default=2)
    generate.add_argument("--prefill-len", type=int, default=None)
    generate.add_argument("--layers", type=int, default=1)
    generate.add_argument("--device", default="p150a")
    generate.add_argument("--device-id", type=int, default=0)
    generate.add_argument("--batch-size", type=int, default=None)
    generate.add_argument("--cache-len", type=int, default=None)
    generate.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    generate.add_argument(
        "--runtime-input-mode",
        choices=("recreate", "persistent"),
        default=None,
    )
    generate.add_argument(
        "--execution-mode",
        choices=("eager", "trace"),
        default="eager",
    )
    generate.add_argument(
        "--prefill-execution-mode",
        choices=("eager", "trace"),
        default="eager",
    )
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument(
        "--report-level",
        choices=("none", "summary", "full"),
        default=None,
    )
    generate.add_argument("--out", type=Path, default=None)
    generate.set_defaults(func=_cmd_generate)

    profile = subparsers.add_parser(
        "profile",
        help="Profile generate, prefill, or post-prefill steady decode.",
    )
    profile.add_argument("--program-dir", type=Path, required=True)
    profile.add_argument("--model-path", type=Path, default=None)
    _add_prompt_runtime_args(profile, batch_prompts=True)
    profile.add_argument(
        "--mode",
        choices=("generate", "decode-steady", "prefill-steady"),
        default="generate",
    )
    profile.add_argument("--max-new-tokens", type=int, default=2)
    profile.add_argument("--prefill-len", type=int, default=None)
    profile.add_argument("--layers", type=int, default=None)
    profile.add_argument("--device", default="p150a")
    profile.add_argument("--device-id", type=int, default=0)
    profile.add_argument("--batch-size", type=int, default=None)
    profile.add_argument("--cache-len", type=int, default=None)
    profile.add_argument("--warmup", type=int, default=5)
    profile.add_argument("--iterations", type=int, default=50)
    profile.add_argument("--after-prefill", action="store_true")
    profile.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    profile.add_argument(
        "--runtime-input-mode",
        choices=("recreate", "persistent"),
        default=None,
    )
    profile.add_argument(
        "--execution-mode",
        choices=("eager", "trace"),
        default="eager",
    )
    profile.add_argument(
        "--prefill-execution-mode",
        choices=("eager", "trace"),
        default="eager",
    )
    profile.add_argument("--dry-run", action="store_true")
    profile.add_argument("--generate-report", type=Path, default=None)
    profile.add_argument("--out", type=Path, required=True)
    profile.set_defaults(func=_cmd_profile_generate)

    validate = subparsers.add_parser(
        "validate",
        help="Validate the main TTNN Direct workflow.",
    )
    validate.add_argument(
        "--suite",
        choices=(
            "dryrun",
            "functional",
            "device",
            "performance",
            "correctness",
        ),
        default="dryrun",
    )
    validate.add_argument("--program-dir", type=Path, default=None)
    validate.add_argument("--model-path", type=Path, default=None)
    _add_prompt_runtime_args(validate)
    validate.add_argument("--out-dir", type=Path, required=True)
    validate.add_argument("--max-new-tokens", type=int, default=2)
    validate.add_argument("--prefill-len", type=int, default=None)
    validate.add_argument("--layers", type=int, default=1)
    validate.add_argument("--batch-size", type=int, default=None)
    validate.add_argument("--cache-len", type=int, default=None)
    validate.add_argument("--device", default="p150a")
    validate.add_argument("--device-id", type=int, default=0)
    validate.add_argument(
        "--dtype-seed",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    validate.add_argument(
        "--runtime-input-mode",
        choices=("recreate", "persistent"),
        default=None,
    )
    validate.add_argument("--require-full-depth", action="store_true")
    validate.add_argument("--check", action="append", default=None)
    validate.add_argument("--pcc-threshold", type=float, default=0.99)
    validate.add_argument(
        "--reference-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    validate.add_argument("--hf-reference", type=Path, default=None)
    validate.set_defaults(func=_cmd_validate)

    inspect = subparsers.add_parser(
        "inspect",
        help="Inspect a generated TTNN Direct program directory.",
    )
    inspect.add_argument("--program-dir", type=Path, required=True)
    inspect.add_argument(
        "--official-template",
        type=Path,
        default=None,
        help="Optional official operation template to compare with the plan.",
    )
    inspect.add_argument(
        "--official-config",
        type=Path,
        default=None,
        help="Optional official TTNN config to compare with config.json.",
    )
    inspect.add_argument("--out", type=Path, default=None)
    inspect.set_defaults(func=_cmd_inspect)

    diagnose = subparsers.add_parser(
        "diagnose",
        help="Run development diagnostics.",
    )
    from .diagnostics.cli import add_diagnose_arguments

    add_diagnose_arguments(diagnose)
    diagnose.set_defaults(func=_cmd_diagnose)
    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    if args.report_level == "none" and args.out is not None:
        print(
            "generate: --report-level none cannot be combined with --out",
            file=sys.stderr,
        )
        return 2
    if args.report_level in {"summary", "full"} and args.out is None:
        print(
            f"generate: --report-level {args.report_level} requires --out",
            file=sys.stderr,
        )
        return 2
    report = run_generate(
        out=args.out,
        program_dir=args.program_dir,
        model_path=args.model_path,
        prompt=args.prompt,
        input_prompts=args.input_prompts,
        instruct=args.instruct,
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
        report_level=args.report_level,
        runtime_input_mode=getattr(args, "runtime_input_mode", None),
        execution_mode=getattr(args, "execution_mode", "eager"),
        prefill_execution_mode=getattr(args, "prefill_execution_mode", "eager"),
    )
    _print_generated_text(report)
    if args.out is not None:
        print(f"wrote generate report: {args.out}")
        diagnostics = report.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics.get("decode_steps"):
            print("wrote generate step diagnostics: " f"{diagnostics['decode_steps']}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    if not report.get("passed"):
        error = report.get("error") or report.get("detail") or report.get("status")
        print(f"generate failed: {error}", file=sys.stderr)
    return 0 if report.get("passed") else 1


def _print_generated_text(report: dict[str, object]) -> None:
    text_by_user = report.get("generated_text_by_user")
    if not isinstance(text_by_user, list) or not text_by_user:
        if report.get("status") == "dry_run":
            print("generate dry-run passed")
        return
    if len(text_by_user) == 1:
        print(str(text_by_user[0]))
        return
    for user_index, text in enumerate(text_by_user):
        print(f"[user {user_index}]")
        print(str(text))


def _cmd_profile_generate(args: argparse.Namespace) -> int:
    if getattr(args, "mode", "generate") == "prefill-steady":
        report = run_profile_prefill_steady(
            out=args.out,
            program_dir=args.program_dir,
            model_path=args.model_path,
            prompt=args.prompt,
            input_prompts=args.input_prompts,
            instruct=args.instruct,
            tokenizer_path=args.tokenizer_path,
            layers=args.layers,
            prefill_len=args.prefill_len,
            device=args.device,
            device_id=args.device_id,
            batch_size=args.batch_size,
            cache_len=args.cache_len,
            dtype_seed=args.dtype_seed,
            warmup=args.warmup,
            iterations=args.iterations,
            dry_run=args.dry_run,
            prefill_execution_mode=getattr(args, "prefill_execution_mode", "eager"),
        )
        print(f"wrote steady prefill profile report: {args.out}")
        if report.get("status") == "no_device":
            print(NO_TTNN_DEVICE_MESSAGE)
            return 2
        return 0 if report.get("passed") else 1

    if getattr(args, "mode", "generate") == "decode-steady":
        report = run_profile_decode_steady(
            out=args.out,
            program_dir=args.program_dir,
            model_path=args.model_path,
            prompt=args.prompt,
            input_prompts=args.input_prompts,
            instruct=args.instruct,
            tokenizer_path=args.tokenizer_path,
            layers=args.layers,
            prefill_len=args.prefill_len,
            device=args.device,
            device_id=args.device_id,
            batch_size=args.batch_size,
            cache_len=args.cache_len,
            dtype_seed=args.dtype_seed,
            warmup=args.warmup,
            iterations=args.iterations,
            after_prefill=True,
            dry_run=args.dry_run,
            runtime_input_mode=getattr(args, "runtime_input_mode", None),
            execution_mode=getattr(args, "execution_mode", "eager"),
            prefill_execution_mode=getattr(args, "prefill_execution_mode", "eager"),
        )
        print(f"wrote steady decode profile report: {args.out}")
        if report.get("status") == "no_device":
            print(NO_TTNN_DEVICE_MESSAGE)
            return 2
        return 0 if report.get("passed") else 1

    report = run_profile_generate(
        out=args.out,
        program_dir=args.program_dir,
        model_path=args.model_path,
        prompt=args.prompt,
        input_prompts=args.input_prompts,
        instruct=args.instruct,
        tokenizer_path=args.tokenizer_path,
        max_new_tokens=args.max_new_tokens,
        layers=args.layers or 1,
        prefill_len=args.prefill_len,
        device=args.device,
        device_id=args.device_id,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype_seed=args.dtype_seed,
        dry_run=args.dry_run,
        generate_report=args.generate_report,
        runtime_input_mode=getattr(args, "runtime_input_mode", None),
        execution_mode=getattr(args, "execution_mode", "eager"),
        prefill_execution_mode=getattr(args, "prefill_execution_mode", "eager"),
    )
    print(f"wrote generate profile report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 0 if report.get("passed") else 1



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



def _cmd_validate(args: argparse.Namespace) -> int:
    if args.suite == "dryrun":
        if args.program_dir is None:
            print(
                "validate --suite dryrun requires --program-dir; run build " "first",
                file=sys.stderr,
            )
            return 1
        return _cmd_validate_program_dryrun(args)

    if args.program_dir is None or args.model_path is None:
        print(
            f"validate --suite {args.suite} requires --program-dir and " "--model-path",
            file=sys.stderr,
        )
        return 1
    return _cmd_validate_product_runtime(args)


def _cmd_validate_program_dryrun(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generate_report = args.out_dir / "generate_dryrun.json"
    profile_report = args.out_dir / "profile_dryrun.json"
    underlying_generate_report = (
        args.out_dir / "profile_underlying_generate_dryrun.json"
    )
    artifacts = _program_artifact_report(args.program_dir)
    generate = run_generate(
        out=generate_report,
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
        dry_run=True,
        runtime_input_mode=getattr(args, "runtime_input_mode", None),
    )
    profile = run_profile_generate(
        out=profile_report,
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
        dry_run=True,
        generate_report=underlying_generate_report,
        runtime_input_mode=getattr(args, "runtime_input_mode", None),
    )
    report = validate_dryrun(
        artifacts=artifacts,
        generate=generate,
        profile=profile,
    )
    report.update(
        {
            "program_dir": str(args.program_dir),
            "artifacts": artifacts,
            "reports": {
                "generate": str(generate_report),
                "profile": str(profile_report),
                "profile_underlying_generate": str(underlying_generate_report),
            },
        }
    )
    report_path = args.out_dir / "validation_report.json"
    _write_report_json(report_path, report)
    print(f"wrote TTNN Direct validation report: {report_path}")
    print(f"  status: {report['status']}")
    return 0 if report["passed"] else 1


def _cmd_validate_product_runtime(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _program_artifact_report(args.program_dir)
    if args.suite == "correctness":
        return _cmd_validate_correctness(args, artifacts)
    generate_report_path = args.out_dir / "generate.json"
    reports = {"generate": str(generate_report_path)}

    if args.suite == "performance":
        profile_report_path = args.out_dir / "profile.json"
        profile = run_profile_generate(
            out=profile_report_path,
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
            dry_run=False,
            generate_report=generate_report_path,
            runtime_input_mode=getattr(args, "runtime_input_mode", None),
        )
        report = validate_performance(
            artifacts=artifacts,
            profile=profile,
            require_full_depth=bool(args.require_full_depth),
        )
        reports["profile"] = str(profile_report_path)
        runtime_status = profile.get("generate_status")
    else:
        generate = run_generate(
            out=generate_report_path,
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
            dry_run=False,
            runtime_input_mode=getattr(args, "runtime_input_mode", None),
        )
        if args.suite == "functional":
            report = validate_functional(
                artifacts=artifacts,
                generate=generate,
            )
        else:
            report = validate_device(
                artifacts=artifacts,
                generate=generate,
                require_full_depth=bool(args.require_full_depth),
                expected_device=args.device,
                expected_device_id=args.device_id,
            )
        runtime_status = generate.get("status")

    report.update(
        {
            "program_dir": str(args.program_dir),
            "model_path": str(args.model_path),
            "artifacts": artifacts,
            "reports": reports,
        }
    )
    report_path = args.out_dir / "validation_report.json"
    _write_report_json(report_path, report)
    print(f"wrote TTNN Direct validation report: {report_path}")
    print(f"  status: {report['status']}")
    if report["passed"]:
        return 0
    if runtime_status == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 1


def _cmd_validate_correctness(
    args: argparse.Namespace,
    artifacts: dict[str, Any],
) -> int:
    from .correctness.run import DEFAULT_CHECKS, run_correctness

    if args.prompt is None:
        print(
            "validate --suite correctness requires --prompt",
            file=sys.stderr,
        )
        return 1
    config = json.loads((args.program_dir / "config.json").read_text())
    prefill_len = int(
        args.prefill_len
        or (config.get("prefill") or {}).get("seq_len")
        or config.get("seq_len", 1)
    )
    batch_size = int(args.batch_size or config["batch_size"])
    cache_len = int(args.cache_len or config["max_cache_len"])
    checks = _correctness_checks(args.check, default=DEFAULT_CHECKS)
    report = run_correctness(
        out_dir=args.out_dir,
        program_dir=args.program_dir,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path or args.model_path,
        prompt=args.prompt,
        layers=args.layers,
        prefill_len=prefill_len,
        batch_size=batch_size,
        cache_len=cache_len,
        device=args.device,
        device_id=args.device_id,
        dtype_seed=args.dtype_seed,
        reference_dtype=args.reference_dtype,
        hf_reference=args.hf_reference,
        checks=checks,
        pcc_threshold=args.pcc_threshold,
        runtime_input_mode=getattr(args, "runtime_input_mode", None),
    )
    report["artifacts"] = artifacts
    if not artifacts.get("passed"):
        report["passed"] = False
        report["status"] = "fail"
        report["failed_checks"] = [
            "program_artifacts",
            *report.get("failed_checks", []),
        ]
    report_path = args.out_dir / "validation_report.json"
    _write_report_json(report_path, report)
    print(f"wrote TTNN Direct correctness report: {report_path}")
    print(f"  status: {report['status']}")
    if report["passed"]:
        return 0
    if report.get("runtime_status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    return 1


def _correctness_checks(
    values: list[str] | None,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if not values:
        return default
    return tuple(
        check.strip() for value in values for check in value.split(",") if check.strip()
    )



def _cmd_inspect(args: argparse.Namespace) -> int:
    report = {
        "schema_version": 1,
        "command": "inspect",
        "program_dir": str(args.program_dir),
        "artifacts": _program_artifact_report(args.program_dir),
        "config": _read_json_if_present(args.program_dir / "config.json"),
        "execution_plan": _read_json_if_present(
            args.program_dir / "execution_plan.json"
        ),
    }
    if args.official_template is not None:
        from .templates.diff import (
            diff_plan_against_official,
            load_official_template,
        )

        report["plan_diff"] = diff_plan_against_official(
            report["execution_plan"],
            load_official_template(args.official_template),
        )
    if args.official_config is not None:
        from .codegen.config_diff import diff_official_config

        report["official_config_diff"] = diff_official_config(
            args.program_dir / "config.json",
            args.official_config,
        )
    report["status"] = "pass" if report["artifacts"]["passed"] else "fail"
    report["passed"] = report["status"] == "pass"
    if args.out is not None:
        _write_report_json(args.out, report)
        print(f"wrote TTNN Direct inspect report: {args.out}")
    else:
        print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def _cmd_diagnose(args: argparse.Namespace) -> int:
    from .diagnostics.cli import run_stage

    try:
        report = run_stage(args)
    except ValueError as exc:
        report = {
            "schema_version": 1,
            "command": "diagnose",
            "stage": args.stage,
            "status": "fail",
            "passed": False,
            "error": str(exc),
        }
        _write_report_json(args.out, report)
        print(f"wrote diagnostics report: {args.out}")
        print(f"  status: {report['status']}")
        return 1
    print(f"wrote diagnostics report: {args.out}")
    if report.get("status") == "no_device":
        print(NO_TTNN_DEVICE_MESSAGE)
        return 2
    successful_statuses = {"pass", "passed", "dry_run", "profiled"}
    return 0 if report.get("passed") or report.get("status") in successful_statuses else 1


def _program_artifact_report(program_dir: Path) -> dict[str, object]:
    required = [
        "README.md",
        "config.json",
        "execution_plan.json",
        "model.py",
        "run_decode.py",
        "semantic_graph.json",
        "weights_manifest.json",
    ]
    files = {
        name: {
            "path": str(program_dir / name),
            "exists": (program_dir / name).is_file(),
        }
        for name in required
    }
    return {
        "program_dir": str(program_dir),
        "files": files,
        "missing": [name for name, item in files.items() if not item["exists"]],
        "passed": all(item["exists"] for item in files.values()),
    }


def _read_json_if_present(path: Path) -> object | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _write_report_json(path: Path, report: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")



def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args._raw_argv = list(argv) if argv is not None else sys.argv[1:]
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
