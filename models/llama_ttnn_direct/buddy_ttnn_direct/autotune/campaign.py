from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..compiler.config import build_codegen_config
from ..semantic.importer_hf_llama import import_hf_llama
from ..templates.registry import build_execution_plan, load_template_config
from .confirmation import ConfirmationPolicy
from .legality import DeviceDescriptor, WorkloadSpec
from .matmul import MATMUL_OPERATORS, enumerate_matmul_programs, rank_matmul_measurement_candidates
from .measurement import file_sha256, prompt_corpus_sha256, resolve_runtime_commit
from .model_evaluator import CandidateGateRunner, ModelCandidateEvaluator
from .schema import ExecutionContract, PrecisionContract, sha256_json
from .sdpa import enumerate_sdpa_programs, rank_sdpa_measurement_candidates
from .search import (
    PIPELINE_STAGES,
    SearchBudget,
    SearchCallbacks,
    SearchProposalGroup,
    atomic_write_json,
    build_active_measurement_inputs,
    matmul_proposal_group,
    run_active_measurement_scheduler,
    run_hierarchical_search,
    sdpa_proposal_group,
    template_proposal_groups,
)
from .space import SearchSpaceConfig
from .templates import DEFAULT_TEMPLATE_SELECTION, list_template_definitions


def run_autotune_campaign(
    *,
    model_path: str | Path,
    config_path: str | Path,
    out: str | Path,
    prompt: str | None,
    tokenizer_path: str | Path | None = None,
    candidates_dir: str | Path | None = None,
    layers: int | None = None,
    prefill_len: int | None = None,
    batch_size: int | None = None,
    cache_len: int | None = None,
    device: str = "p150a",
    device_id: int = 0,
    dtype_seed: str = "bf16",
    warmup: int = 5,
    iterations: int = 10,
    repetitions: int = 3,
    min_relative_improvement: float = 0.01,
    dry_run: bool = False,
    resume: bool = True,
    budget: SearchBudget | None = None,
    confirmation_policy: ConfirmationPolicy | None = None,
    callbacks: SearchCallbacks | None = None,
    measurement_runner: Callable[..., Mapping[str, Any]] | None = None,
    active_full_model_runner: Callable[..., Mapping[str, Any]] | None = None,
    candidate_gate_runner: CandidateGateRunner | None = None,
) -> dict[str, Any]:
    """Compose canonical schema-v2 enumeration, measurement, and search APIs."""
    output = Path(out)
    if not dry_run and not prompt:
        raise ValueError("prompt is required for hardware autotune")
    if dtype_seed != "bf16":
        raise ValueError("semantic autotune precision is frozen to 'bf16'")
    if repetitions != 3:
        raise ValueError("canonical confirmation requires exactly 3 repetitions")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if min_relative_improvement < 0.0:
        raise ValueError("min_relative_improvement must be non-negative")
    policy = confirmation_policy or ConfirmationPolicy(
        minimum_relative_improvement=min_relative_improvement
    )
    selected_budget = budget or SearchBudget()
    try:
        context = _context(
            Path(model_path),
            Path(config_path),
            prompt,
            layers,
            prefill_len,
            batch_size,
            cache_len,
            device,
            device_id,
        )
        enumerations = _enumerate(context)
        identity = _identity(context, enumerations, prompt)
        root = (
            Path(candidates_dir) if candidates_dir else output.parent / f"{output.stem}_candidates"
        )
        root.mkdir(parents=True, exist_ok=True)
        report = _report(
            context, enumerations, identity, output, root, dry_run, selected_budget, policy
        )
        report["requested_measurement"] = {
            "warmup": int(warmup),
            "iterations": int(iterations),
            "repetitions": int(repetitions),
            "note": "active scheduler and final confirmation use canonical contracts",
        }
        groups = _analytical_groups(context, enumerations)
        if dry_run:
            search = _search(context, groups, root, selected_budget, policy, resume, True, identity)
            return _finish(
                report,
                search=search,
                passed=bool(search.get("passed")),
                status="dry_run" if search.get("passed") else "failed",
            )

        runtime_commit = resolve_runtime_commit()
        evaluator = ModelCandidateEvaluator(
            context={
                **context,
                "prompt": str(prompt),
                "tokenizer_path": Path(tokenizer_path) if tokenizer_path else None,
                "runtime_commit": runtime_commit,
            },
            output_root=root / "model_measurements",
            candidate_spaces=enumerations["candidate_spaces"],
            resume=resume,
            candidate_gate_runner=candidate_gate_runner,
        )
        active = run_active_measurement_scheduler(
            candidate_groups=enumerations["active_inputs"]["groups"],
            statically_rejected=enumerations["active_inputs"]["rejected"],
            measurement_runner=measurement_runner or evaluator.active_measurement,
            full_model_runner=active_full_model_runner or evaluator.active_full_model,
            out=root / "active_measurement_report.json",
            resume=resume,
        )
        report["active_measurement_report"] = str(root / "active_measurement_report.json")
        if not active.get("passed"):
            return _finish(report, active=active, passed=False, status="failed")
        measured = _measured_groups(context["base_space"], enumerations, active)
        search = _search(
            context,
            (*_template_groups(groups), *measured),
            root,
            selected_budget,
            policy,
            resume,
            False,
            identity,
            callbacks=_callbacks(callbacks, evaluator),
        )
        return _finish(
            report,
            active=active,
            search=search,
            passed=bool(search.get("passed")),
            status="passed" if search.get("passed") else "failed",
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "command": "diagnose",
            "stage": "autotune",
            "algorithm": "hierarchical_constrained_beam_search",
            "status": "failed",
            "passed": False,
            "dry_run": bool(dry_run),
            "failure_report_written": True,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        atomic_write_json(output, failure)
        return failure


def _context(
    model: Path,
    config: Path,
    prompt: str | None,
    layers: int | None,
    prefill: int | None,
    batch: int | None,
    cache: int | None,
    device: str,
    device_id: int,
) -> dict[str, Any]:
    seed = load_template_config(config)
    batch = int(batch or seed["batch_size"])
    cache = int(cache or seed["max_cache_len"])
    graph = import_hf_llama(
        model,
        mode="decode",
        batch_size=batch,
        seq_len=int(seed["decode_seq_len"]),
        max_cache_len=cache,
        generation_mode="greedy",
    )
    seed = copy.deepcopy(seed)
    seed.update(batch_size=batch, max_cache_len=cache)
    runtime = build_codegen_config(build_execution_plan(graph, seed))
    precision = PrecisionContract.from_template_config(seed)
    execution = ExecutionContract.from_template_config(
        seed, prompt_corpus_sha256=prompt_corpus_sha256(prompt)
    )
    depth = graph.num_layers if layers is None else int(layers)
    if not 0 < depth <= graph.num_layers:
        raise ValueError(f"layers must be in [1, {graph.num_layers}]")
    if device != "p150a":
        raise ValueError("canonical P150A campaign currently supports device='p150a'")
    return {
        "model_root": model,
        "config_path": config,
        "seed": seed,
        "graph": graph,
        "base_space": SearchSpaceConfig.from_runtime_config(runtime),
        "workload": WorkloadSpec.from_runtime_config(runtime),
        "device_descriptor": DeviceDescriptor.p150a(),
        "precision": precision,
        "execution": execution,
        "layers": depth,
        "prefill_len": int(prefill or seed["prefill_seq_len"]),
        "batch_size": batch,
        "cache_len": cache,
        "device": device,
        "device_id": int(device_id),
    }


def _enumerate(c: Mapping[str, Any]) -> dict[str, Any]:
    matmuls = tuple(
        enumerate_matmul_programs(
            operator_name=name,
            base_space=c["base_space"],
            workload=c["workload"],
            device=c["device_descriptor"],
            precision_contract=c["precision"],
            max_proposals=64,
        )
        for name in MATMUL_OPERATORS
    )
    sdpa = enumerate_sdpa_programs(
        base_space=c["base_space"],
        workload=c["workload"],
        device=c["device_descriptor"],
        precision_contract=c["precision"],
        max_proposals=128,
        active_context_len=c["cache_len"],
    )
    groups, rejected = build_active_measurement_inputs(matmul_results=matmuls, sdpa_result=sdpa)
    spaces = {x.candidate_id: x.search_space for r in matmuls for x in r.candidates}
    spaces.update({x.candidate_id: x.search_space for x in sdpa.candidates})
    return {
        "matmul": matmuls,
        "sdpa": sdpa,
        "active_inputs": {"groups": groups, "rejected": rejected},
        "candidate_spaces": spaces,
    }


def _analytical_groups(
    c: Mapping[str, Any], e: Mapping[str, Any]
) -> tuple[SearchProposalGroup, ...]:
    definitions = list_template_definitions()
    template_evidence = {
        item.name: {
            "status": "estimated",
            "passed": True,
            "latency_ms": 0.0 if item.is_default else float(item.launch_count),
            "estimated": True,
        }
        for item in definitions
    }
    result = list(
        template_proposal_groups(
            base_space=c["base_space"], measurements=template_evidence
        )
    )
    for enumeration in (*e["matmul"], e["sdpa"]):
        ranked = (
            rank_matmul_measurement_candidates(enumeration)
            if hasattr(enumeration, "operator_name")
            else rank_sdpa_measurement_candidates(enumeration)
        )
        evidence = {
            x.candidate_id: {
                "status": "passed",
                "passed": True,
                "latency_ms": max(0.001, x.analytical_score),
                "l1_bytes": x.l1_bytes,
                "estimated": True,
            }
            for x in ranked
        }
        if hasattr(enumeration, "operator_name"):
            result.append(
                matmul_proposal_group(
                    enumeration, base_space=c["base_space"], measurements=evidence
                )
            )
        else:
            result.append(
                sdpa_proposal_group(enumeration, base_space=c["base_space"], measurements=evidence)
            )
    return tuple(result)


def _measured_groups(
    base: SearchSpaceConfig, e: Mapping[str, Any], active: Mapping[str, Any]
) -> tuple[SearchProposalGroup, ...]:
    reports = active.get("operators")
    if not isinstance(reports, Mapping):
        raise ValueError("active measurement report has no operator reports")
    result = []
    for enumeration in e["matmul"]:
        evidence = (reports.get(enumeration.operator_name) or {}).get("proposal_measurements", {})
        result.append(matmul_proposal_group(enumeration, base_space=base, measurements=evidence))
    enumeration = e["sdpa"]
    evidence = (reports.get("attention.sdpa") or {}).get("proposal_measurements", {})
    result.append(sdpa_proposal_group(enumeration, base_space=base, measurements=evidence))
    return tuple(result)


def _template_groups(groups: Sequence[SearchProposalGroup]) -> tuple[SearchProposalGroup, ...]:
    return tuple(x for x in groups if x.stage == "template")


def _callbacks(
    value: SearchCallbacks | None, evaluator: ModelCandidateEvaluator
) -> SearchCallbacks:
    value = value or SearchCallbacks()
    return SearchCallbacks(
        value.candidate_evaluator,
        value.layer_evaluator or evaluator.layer_evaluator,
        value.full_model_evaluator or evaluator.full_model_evaluator,
        value.confirmation_runner or evaluator.confirmation_runner,
        value.candidate_config_factory or evaluator.candidate_config,
    )


def _search(
    c: Mapping[str, Any],
    groups: Sequence[SearchProposalGroup],
    root: Path,
    budget: SearchBudget,
    policy: ConfirmationPolicy,
    resume: bool,
    dry_run: bool,
    identity: Mapping[str, Any],
    callbacks: SearchCallbacks | None = None,
) -> dict[str, Any]:
    return run_hierarchical_search(
        base_space=c["base_space"],
        proposal_groups=groups,
        template_config=c["seed"],
        out_dir=root / "hierarchical",
        callbacks=callbacks,
        budget=budget,
        confirmation_policy=policy,
        resume=resume,
        dry_run=dry_run,
        run_identity=identity["run_id"],
    )


def _identity(c: Mapping[str, Any], e: Mapping[str, Any], prompt: str | None) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "model_config_sha256": file_sha256(c["model_root"] / "config.json"),
        "seed_config_sha256": file_sha256(c["config_path"]),
        "prompt_corpus_sha256": prompt_corpus_sha256(prompt),
        "base_space": c["base_space"].to_dict(),
        "precision_contract": c["precision"].to_dict(),
        "execution_contract": c["execution"].to_dict(),
        "target": {k: c[k] for k in ("batch_size", "prefill_len", "cache_len", "device_id")},
        "operator_count": len(e["matmul"]) + 1,
    }
    return {"run_id": f"semantic-{sha256_json(payload)[:20]}", "payload": payload}


def _report(
    c: Mapping[str, Any],
    e: Mapping[str, Any],
    identity: Mapping[str, Any],
    output: Path,
    root: Path,
    dry: bool,
    budget: SearchBudget,
    policy: ConfirmationPolicy,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "command": "diagnose",
        "stage": "autotune",
        "algorithm": "hierarchical_constrained_beam_search",
        "pipeline_stages": list(PIPELINE_STAGES),
        "search_space_schema_version": 2,
        "precision_contract": c["precision"].to_dict(),
        "execution_contract": c["execution"].to_dict(),
        "search_budget": budget.to_dict(),
        "confirmation_policy": policy.to_dict(),
        "cartesian_exhaustive_search": False,
        "dry_run": dry,
        "status": "running",
        "passed": False,
        "model_path": str(c["model_root"]),
        "config_path": str(c["config_path"]),
        "report_path": str(output),
        "candidates_dir": str(root),
        "resume_identity": identity,
        "proposal_enumeration_summary": _summary(e),
        "failure_report_written": True,
    }


def _summary(e: Mapping[str, Any]) -> dict[str, Any]:
    compact = lambda x: {
        "status": x.status,
        "legal_candidate_count": len(x.candidates),
        "rejected_candidate_count": len(x.rejected),
        "official_candidate_ids": [y.candidate_id for y in x.official_candidates],
    }
    return {
        "template_registry": {
            "definition_count": len(list_template_definitions()),
            "axes": {
                a: [x.name for x in list_template_definitions() if x.axis == a]
                for a in DEFAULT_TEMPLATE_SELECTION
            },
        },
        "matmul": {x.operator_name: compact(x) for x in e["matmul"]},
        "sdpa": compact(e["sdpa"]),
    }


def _finish(
    report: dict[str, Any],
    *,
    passed: bool,
    status: str,
    search: Mapping[str, Any] | None = None,
    active: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report.update(status=status, passed=bool(passed))
    if search is not None:
        report["search_report"] = search
        report["pipeline_stage_status"] = {
            str(stage.get("name")): str(stage.get("status", "unknown"))
            for stage in search.get("stages", [])
            if isinstance(stage, Mapping) and stage.get("name")
        }
        report["hierarchical_search_report"] = str(
            Path(report["candidates_dir"]) / "hierarchical" / "search_report.json"
        )
        report["resume"] = search.get("resume")
    if active is not None:
        report["active_measurement"] = active
    atomic_write_json(Path(report["report_path"]), report)
    return report
