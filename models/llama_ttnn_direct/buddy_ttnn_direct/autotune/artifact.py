from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import stat
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import sha256_json

PAPER_ARTIFACT_SCHEMA_VERSION = 1
REQUIRED_ABLATIONS = (
    "config_only",
    "template_only",
    "layout_only",
    "template_config",
    "template_config_layout",
    "representative_layer_transfer_on",
    "representative_layer_transfer_off",
    "analytical_pruning_on",
    "analytical_pruning_off",
)
_REQUIRED_SOURCES = (
    "template_acceptance",
    "microbench_acceptance",
    "matmul_acceptance",
    "sdpa_enumeration",
    "sdpa_acceptance",
    "layout_acceptance",
    "search_report",
    "generalization_report",
    "correctness_evidence",
)
_ENABLED_KEYS = (
    "config_search",
    "template_search",
    "layout_search",
    "representative_layer_transfer",
    "analytical_pruning",
)
_ABLATION_MODES = {"measured", "reused_equivalent", "analytical"}


class PaperArtifactError(ValueError):
    """Raised when paper evidence is incomplete or internally inconsistent."""


def build_paper_artifact(
    spec: Mapping[str, Any] | str | Path,
    *,
    out_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    failure_path = destination / "artifact_failure.json"
    try:
        raw_spec, spec_base = _load_spec(spec)
        normalized_spec = _normalize_spec(raw_spec, spec_base=spec_base)
        sources, source_manifest = _load_sources(normalized_spec)
        metrics = _extract_metrics(sources)
        ablations = _build_ablations(
            normalized_spec["ablations"],
            metrics=metrics,
            source_labels=set(source_manifest),
        )
        artifact = {
            "schema_version": PAPER_ARTIFACT_SCHEMA_VERSION,
            "stage": "semantic-autotune-paper-artifact",
            "status": "passed",
            "passed": True,
            "campaign": copy.deepcopy(normalized_spec["campaign"]),
            "campaign_fingerprint": sha256_json(
                {
                    "campaign": normalized_spec["campaign"],
                    "source_hashes": {
                        label: record["sha256"]
                        for label, record in sorted(source_manifest.items())
                    },
                    "ablations": normalized_spec["ablations"],
                }
            ),
            "metrics": metrics,
            "ablations": ablations,
            "required_ablations": list(REQUIRED_ABLATIONS),
            "source_manifest": "source_manifest.json",
            "artifact_manifest": "artifact_manifest.json",
            "reproducibility": {
                "spec": "spec.json",
                "compact_evidence": "evidence_summary.json",
                "verify_command": (
                    "python -m models.llama_ttnn_direct.buddy_ttnn_direct."
                    "autotune.artifact_cli verify --artifact-dir ."
                ),
                "rebuild_command": (
                    "python -m models.llama_ttnn_direct.buddy_ttnn_direct."
                    "autotune.artifact_cli build --spec spec.json "
                    "--out-dir rebuilt"
                ),
                "script": "reproduce.sh",
            },
        }
        _atomic_write_json(destination / "spec.json", normalized_spec)
        _atomic_write_json(destination / "source_manifest.json", source_manifest)
        _atomic_write_json(destination / "paper_artifact.json", artifact)
        _atomic_write_text(
            destination / "ablation.csv", _render_ablation_csv(ablations)
        )
        _atomic_write_text(destination / "RESULTS.md", _render_markdown(artifact))
        _atomic_write_json(
            destination / "evidence_summary.json",
            _compact_evidence(artifact, source_manifest),
        )
        _write_reproduce_script(destination / "reproduce.sh")
        artifact_manifest = _artifact_manifest(
            destination,
            filenames=(
                "spec.json",
                "source_manifest.json",
                "paper_artifact.json",
                "ablation.csv",
                "RESULTS.md",
                "evidence_summary.json",
                "reproduce.sh",
            ),
        )
        _atomic_write_json(destination / "artifact_manifest.json", artifact_manifest)
        if failure_path.exists():
            failure_path.unlink()
        verification = verify_paper_artifact(destination)
        if not verification["passed"]:
            raise PaperArtifactError(
                "freshly built artifact did not verify: "
                + ", ".join(verification["errors"])
            )
        return artifact
    except Exception as exc:
        failure = {
            "schema_version": PAPER_ARTIFACT_SCHEMA_VERSION,
            "stage": "semantic-autotune-paper-artifact",
            "status": "failed",
            "passed": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "failure_report_written": True,
        }
        _atomic_write_json(failure_path, failure)
        return failure


def verify_paper_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    root = Path(artifact_dir)
    errors: list[str] = []
    if (root / "artifact_failure.json").is_file():
        errors.append("artifact_failure.json is present")
    artifact = _read_json(root / "paper_artifact.json")
    source_manifest = _read_json(root / "source_manifest.json")
    artifact_manifest = _read_json(root / "artifact_manifest.json")
    if not isinstance(artifact, dict):
        errors.append("paper_artifact.json is missing or malformed")
    if not isinstance(source_manifest, dict):
        errors.append("source_manifest.json is missing or malformed")
    if not isinstance(artifact_manifest, dict):
        errors.append("artifact_manifest.json is missing or malformed")

    if isinstance(artifact, dict):
        if artifact.get("schema_version") != PAPER_ARTIFACT_SCHEMA_VERSION:
            errors.append("paper artifact schema version mismatch")
        if artifact.get("status") != "passed" or not artifact.get("passed"):
            errors.append("paper artifact is not passed")
        try:
            _validate_built_ablations(artifact.get("ablations"))
            _validate_metric_surface(artifact.get("metrics"))
        except PaperArtifactError as exc:
            errors.append(str(exc))

    if isinstance(artifact_manifest, dict):
        for filename, expected in (artifact_manifest.get("files") or {}).items():
            path = root / filename
            if not path.is_file():
                errors.append(f"artifact file is missing: {filename}")
                continue
            observed = _file_sha256(path)
            if observed != expected.get("sha256"):
                errors.append(f"artifact file hash mismatch: {filename}")
            if path.stat().st_size != expected.get("bytes"):
                errors.append(f"artifact file size mismatch: {filename}")

    if isinstance(source_manifest, dict):
        for label, record in source_manifest.items():
            path = Path(str(record.get("path", "")))
            if not path.is_file():
                errors.append(f"source evidence is missing: {label}")
                continue
            if _file_sha256(path) != record.get("sha256"):
                errors.append(f"source evidence hash mismatch: {label}")
            if path.stat().st_size != record.get("bytes"):
                errors.append(f"source evidence size mismatch: {label}")

    return {
        "schema_version": PAPER_ARTIFACT_SCHEMA_VERSION,
        "stage": "semantic-autotune-paper-artifact-verification",
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "artifact_dir": str(root),
        "errors": errors,
        "checked_artifact_file_count": (
            len((artifact_manifest or {}).get("files") or {})
            if isinstance(artifact_manifest, dict)
            else 0
        ),
        "checked_source_count": (
            len(source_manifest) if isinstance(source_manifest, dict) else 0
        ),
    }


def _load_spec(
    value: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], Path]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value)), Path.cwd()
    path = Path(value)
    decoded = _read_json(path)
    if not isinstance(decoded, dict):
        raise PaperArtifactError(f"artifact spec is missing or malformed: {path}")
    return decoded, path.resolve().parent


def _normalize_spec(spec: Mapping[str, Any], *, spec_base: Path) -> dict[str, Any]:
    if spec.get("schema_version") != PAPER_ARTIFACT_SCHEMA_VERSION:
        raise PaperArtifactError(
            f"spec schema_version must be {PAPER_ARTIFACT_SCHEMA_VERSION}"
        )
    campaign = spec.get("campaign")
    if not isinstance(campaign, Mapping):
        raise PaperArtifactError("spec campaign must be an object")
    required_campaign = ("name", "model", "device", "workload")
    missing_campaign = [key for key in required_campaign if not campaign.get(key)]
    if missing_campaign:
        raise PaperArtifactError(
            "spec campaign is missing: " + ", ".join(missing_campaign)
        )
    sources = spec.get("sources")
    if not isinstance(sources, Mapping):
        raise PaperArtifactError("spec sources must be an object")
    missing_sources = [key for key in _REQUIRED_SOURCES if key not in sources]
    if missing_sources:
        raise PaperArtifactError(
            "spec sources are missing: " + ", ".join(missing_sources)
        )
    layout_paths = sources.get("layout_search_reports")
    if not isinstance(layout_paths, Sequence) or isinstance(layout_paths, str):
        raise PaperArtifactError("layout_search_reports must be a non-empty list")
    if not layout_paths:
        raise PaperArtifactError("layout_search_reports must be a non-empty list")

    normalized_sources: dict[str, Any] = {}
    for label, source in sources.items():
        if label == "layout_search_reports":
            normalized_sources[label] = [
                str(_resolve_source_path(path, spec_base)) for path in source
            ]
        else:
            normalized_sources[label] = str(_resolve_source_path(source, spec_base))
    ablations = spec.get("ablations")
    if not isinstance(ablations, Sequence) or isinstance(ablations, str):
        raise PaperArtifactError("spec ablations must be a list")
    normalized_ablations = [copy.deepcopy(dict(row)) for row in ablations]
    _validate_ablation_specs(normalized_ablations)
    return {
        "schema_version": PAPER_ARTIFACT_SCHEMA_VERSION,
        "campaign": copy.deepcopy(dict(campaign)),
        "sources": normalized_sources,
        "ablations": normalized_ablations,
    }


def _resolve_source_path(value: Any, base: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise PaperArtifactError("source evidence paths must be non-empty")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_sources(
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sources: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    for label, value in spec["sources"].items():
        values = value if label == "layout_search_reports" else [value]
        for index, raw_path in enumerate(values):
            source_label = (
                f"{label}[{index}]" if label == "layout_search_reports" else label
            )
            path = Path(raw_path)
            decoded = _read_json(path)
            if not isinstance(decoded, dict):
                raise PaperArtifactError(
                    f"source evidence is missing or malformed: {source_label}"
                )
            _require_passed_source(source_label, decoded)
            sources[source_label] = decoded
            manifest[source_label] = _source_manifest_record(path, decoded)

    search_report = sources["search_report"]
    best_config = Path(
        str((search_report.get("artifacts") or {}).get("best_config", ""))
    )
    if not best_config.is_file():
        raise PaperArtifactError("search report best_config is missing")
    manifest["search_best_config"] = _source_manifest_record(best_config, None)
    sources["search_best_config"] = best_config
    return sources, manifest


def _require_passed_source(label: str, report: Mapping[str, Any]) -> None:
    if report.get("status") not in {"passed", "selected"}:
        raise PaperArtifactError(f"source evidence did not pass: {label}")
    if label in {"search_report", "generalization_report"} and not report.get("passed"):
        raise PaperArtifactError(f"source evidence did not pass: {label}")


def _source_manifest_record(
    path: Path, decoded: Mapping[str, Any] | None
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "schema_version": decoded.get("schema_version") if decoded else None,
        "status": decoded.get("status") if decoded else None,
    }


def _extract_metrics(sources: Mapping[str, Any]) -> dict[str, Any]:
    template = sources["template_acceptance"]
    microbench = sources["microbench_acceptance"]
    matmul = sources["matmul_acceptance"]
    sdpa_enumeration = sources["sdpa_enumeration"]
    sdpa_acceptance = sources["sdpa_acceptance"]
    layout_acceptance = sources["layout_acceptance"]
    search = sources["search_report"]
    generalization = sources["generalization_report"]
    correctness = sources["correctness_evidence"]

    template_count = len(template.get("templates") or [])
    template_legal = sum(
        1 for row in template.get("templates") or [] if row.get("status") == "passed"
    )
    candidate_counts = matmul.get("candidate_counts") or {}
    matmul_legal = sum(int(row.get("legal", 0)) for row in candidate_counts.values())
    matmul_rejected = sum(
        int(row.get("rejected", 0)) for row in candidate_counts.values()
    )
    sdpa_total = int(sdpa_enumeration.get("proposal_count", 0))
    sdpa_legal = int(sdpa_enumeration.get("legal_candidate_count", 0))
    sdpa_rejected = int(sdpa_enumeration.get("rejected_candidate_count", 0))
    if sdpa_legal + sdpa_rejected != sdpa_total:
        raise PaperArtifactError("SDPA enumeration counts are inconsistent")

    layout_reports = [
        value
        for label, value in sources.items()
        if label.startswith("layout_search_reports[")
    ]
    layout_states = sum(
        int((report.get("search_statistics") or {}).get("expanded_state_count", 0))
        for report in layout_reports
    )
    layout_rejected = sum(
        int(
            (report.get("legality") or {}).get(
                "illegal_or_unmeasured_transition_count", 0
            )
        )
        for report in layout_reports
    )
    program_total = matmul_legal + matmul_rejected + sdpa_total
    program_rejected = matmul_rejected + sdpa_rejected
    total_entities = template_count + program_total + layout_states
    aggregate_rejected = (
        (template_count - template_legal) + program_rejected + layout_rejected
    )

    matmul_microbench_count = len(
        ((matmul.get("microbenchmark_selection") or {}).get("ranked") or [])
    )
    sdpa_microbench_count = sum(
        1
        for value in (sdpa_acceptance.get("correctness") or {}).values()
        if isinstance(value, Mapping) and value.get("passed")
    )
    layout_microbench_count = int(bool(layout_acceptance.get("conversion_measurement")))

    full_model_stage = next(
        (
            stage
            for stage in search.get("stages") or []
            if stage.get("name") == "full_model_trace"
        ),
        {},
    )
    full_model_screen_count = len(full_model_stage.get("evaluations") or [])
    layer_confirmation_stage = next(
        (
            stage
            for stage in search.get("stages") or []
            if stage.get("name") == "layer_confirmation"
        ),
        {},
    )
    layer_confirmation_candidate_count = int(
        layer_confirmation_stage.get(
            "candidate_count",
            len(layer_confirmation_stage.get("evaluations") or []),
        )
    )
    if layer_confirmation_candidate_count <= 0:
        layer_confirmation_candidate_count = max(full_model_screen_count, 1)
    confirmation = search.get("matched_ab_confirmation") or {}
    confirmation_profile_count = len(confirmation.get("execution_order") or [])
    arms = confirmation.get("arms") or {}
    promotion = confirmation.get("promotion") or {}
    selected_arm = promotion.get("selected_arm")
    selected_metrics = arms.get(selected_arm) or {}
    best_tpsu = _positive_float(selected_metrics.get("median"), "best performance")

    official = correctness.get("final_decode_parity") or {}
    official_tpsu = _positive_float(
        official.get("official_release_median_tpsu"), "official performance"
    )
    published_tpsu = _positive_float(
        (correctness.get("target") or {}).get(
            "published_decode_tokens_per_second_per_user"
        ),
        "published performance",
    )
    performance_accuracy = correctness.get("performance_recipe_correctness") or {}
    compiler_accuracy = correctness.get("compiler_correctness") or {}
    accuracy_passed = bool(
        performance_accuracy.get("passed") and compiler_accuracy.get("passed")
    )
    if not accuracy_passed:
        raise PaperArtifactError("accuracy evidence did not pass")

    budget_usage = search.get("budget_usage") or {}
    device_seconds = float(budget_usage.get("device_seconds", 0.0))
    if device_seconds <= 0:
        raise PaperArtifactError("search report has no measured device time")

    representative_layers = (microbench.get("transfer_report") or {}).get(
        "representative_layers"
    ) or []
    layer_count = int((microbench.get("transfer_report") or {}).get("layer_count", 0))
    if not representative_layers or layer_count <= 0:
        raise PaperArtifactError("representative-layer transfer evidence is missing")

    generalization_acceptance = generalization.get("acceptance") or {}
    if not generalization_acceptance.get("passed"):
        raise PaperArtifactError("generalization acceptance did not pass")

    return {
        "search_time": {
            "device_process_seconds": device_seconds,
            "device_process_minutes": device_seconds / 60.0,
            "scope": "orchestrator-accounted candidate subprocess wall time",
            "evaluation_count": int(budget_usage.get("evaluation_count", 0)),
            "reused_evaluation_count": int(
                budget_usage.get("reused_evaluation_count", 0)
            ),
        },
        "enumeration": {
            "template_variants": {
                "enumerated": template_count,
                "legal": template_legal,
                "rejected": template_count - template_legal,
            },
            "matmul_program_candidates": {
                "enumerated": matmul_legal + matmul_rejected,
                "legal": matmul_legal,
                "rejected": matmul_rejected,
            },
            "sdpa_program_candidates": {
                "enumerated": sdpa_total,
                "legal": sdpa_legal,
                "rejected": sdpa_rejected,
            },
            "layout_states": {
                "enumerated": layout_states,
                "legal": layout_states - layout_rejected,
                "rejected": layout_rejected,
            },
            "program_candidate_total": program_total,
            "total_search_entities": total_entities,
        },
        "pruning": {
            "program_candidates_rejected": program_rejected,
            "program_candidate_pruning_ratio": (
                program_rejected / program_total if program_total else 0.0
            ),
            "aggregate_rejected": aggregate_rejected,
            "aggregate_pruning_ratio": (
                aggregate_rejected / total_entities if total_entities else 0.0
            ),
        },
        "microbench": {
            "matmul_candidate_measurements": matmul_microbench_count,
            "sdpa_op_or_region_measurements": sdpa_microbench_count,
            "layout_conversion_measurements": layout_microbench_count,
            "experiment_count": (
                matmul_microbench_count
                + sdpa_microbench_count
                + layout_microbench_count
            ),
        },
        "full_model": {
            "screen_profile_count": full_model_screen_count,
            "confirmation_profile_count": confirmation_profile_count,
            "profile_count": full_model_screen_count + confirmation_profile_count,
            "candidate_count": len(arms),
            "confirmation_contract": copy.deepcopy(
                confirmation.get("measurement_contract") or {}
            ),
        },
        "best_performance": {
            "metric": "tokens_per_second_per_user",
            "value": best_tpsu,
            "selected_arm": selected_arm,
            "selected_candidate_fingerprint": promotion.get(
                "selected_candidate_fingerprint"
            ),
            "official_release_value": official_tpsu,
            "official_ratio": best_tpsu / official_tpsu,
            "published_reference_value": published_tpsu,
            "published_reference_ratio": best_tpsu / published_tpsu,
            "cv": selected_metrics.get("cv"),
        },
        "accuracy": {
            "passed": accuracy_passed,
            "evaluated_tokens": performance_accuracy.get("evaluated_tokens"),
            "official_top1_accuracy": performance_accuracy.get(
                "official_top1_accuracy"
            ),
            "buddy_min_user_top1_accuracy": performance_accuracy.get(
                "buddy_min_user_top1_accuracy"
            ),
            "official_top5_accuracy": performance_accuracy.get(
                "official_top5_accuracy"
            ),
            "buddy_min_user_top5_accuracy": performance_accuracy.get(
                "buddy_min_user_top5_accuracy"
            ),
            "buddy_min_user_greedy_agreement_with_official": performance_accuracy.get(
                "buddy_min_user_greedy_agreement_with_official"
            ),
            "logits_pcc": compiler_accuracy.get("logits_pcc"),
            "minimum_hidden_pcc": compiler_accuracy.get("minimum_hidden_pcc"),
            "minimum_sampled_kv_pcc": compiler_accuracy.get("minimum_sampled_kv_pcc"),
        },
        "representative_layer_transfer": {
            "layer_count": layer_count,
            "representative_layers": list(representative_layers),
            "representative_layer_count": len(representative_layers),
            "candidate_count_at_layer_confirmation": (
                layer_confirmation_candidate_count
            ),
            "transferred_layer_count": sum(
                int(group.get("transfer_count", 0))
                for group in (microbench.get("transfer_report") or {}).get("groups", [])
            ),
        },
        "generalization": {
            "model_count": generalization_acceptance.get("model_count"),
            "workload_count": generalization_acceptance.get("workload_count"),
            "workloads_per_model": copy.deepcopy(
                generalization_acceptance.get("workloads_per_model") or {}
            ),
            "passed": True,
        },
        "selected_config_sha256": _file_sha256(sources["search_best_config"]),
    }


def _validate_ablation_specs(rows: Sequence[Mapping[str, Any]]) -> None:
    names = [str(row.get("name", "")) for row in rows]
    if len(set(names)) != len(names):
        raise PaperArtifactError("ablation names must be unique")
    missing = [name for name in REQUIRED_ABLATIONS if name not in names]
    extra = [name for name in names if name not in REQUIRED_ABLATIONS]
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if extra:
            parts.append("unexpected " + ", ".join(extra))
        raise PaperArtifactError("ablation matrix mismatch: " + "; ".join(parts))
    by_name = {str(row["name"]): row for row in rows}
    expected_axes = {
        "config_only": (True, False, False),
        "template_only": (False, True, False),
        "layout_only": (False, False, True),
        "template_config": (True, True, False),
        "template_config_layout": (True, True, True),
    }
    for name, axes in expected_axes.items():
        enabled = _validate_enabled(by_name[name].get("enabled"), name)
        observed = (
            enabled["config_search"],
            enabled["template_search"],
            enabled["layout_search"],
        )
        if observed != axes:
            raise PaperArtifactError(f"ablation {name} has incorrect search axes")
    for name in REQUIRED_ABLATIONS:
        row = by_name[name]
        mode = row.get("mode")
        if mode not in _ABLATION_MODES:
            raise PaperArtifactError(
                f"ablation {name} mode must be one of {sorted(_ABLATION_MODES)}"
            )
        enabled = _validate_enabled(row.get("enabled"), name)
        if mode != "measured" and not row.get("equivalent_to"):
            raise PaperArtifactError(
                f"ablation {name} requires equivalent_to for mode {mode}"
            )
        if not isinstance(row.get("evidence_labels"), list) or not row.get(
            "evidence_labels"
        ):
            raise PaperArtifactError(f"ablation {name} requires evidence_labels")
        if not isinstance(row.get("notes"), list) or not row.get("notes"):
            raise PaperArtifactError(f"ablation {name} requires notes")
        if (
            name.endswith("transfer_on")
            and not enabled["representative_layer_transfer"]
        ):
            raise PaperArtifactError("transfer_on ablation must enable transfer")
        if name.endswith("transfer_off") and enabled["representative_layer_transfer"]:
            raise PaperArtifactError("transfer_off ablation must disable transfer")
        if name.endswith("pruning_on") and not enabled["analytical_pruning"]:
            raise PaperArtifactError("pruning_on ablation must enable pruning")
        if name.endswith("pruning_off") and enabled["analytical_pruning"]:
            raise PaperArtifactError("pruning_off ablation must disable pruning")


def _validate_enabled(value: Any, name: str) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(_ENABLED_KEYS):
        raise PaperArtifactError(
            f"ablation {name} enabled must contain exactly {list(_ENABLED_KEYS)}"
        )
    if any(not isinstance(value[key], bool) for key in _ENABLED_KEYS):
        raise PaperArtifactError(f"ablation {name} enabled values must be boolean")
    return {key: bool(value[key]) for key in _ENABLED_KEYS}


def _build_ablations(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Mapping[str, Any],
    source_labels: set[str],
) -> list[dict[str, Any]]:
    domains = {
        "template_search": {
            **metrics["enumeration"]["template_variants"],
            "microbench": metrics["enumeration"]["template_variants"]["legal"],
        },
        "config_search": {
            "enumerated": metrics["enumeration"]["program_candidate_total"],
            "legal": (
                metrics["enumeration"]["matmul_program_candidates"]["legal"]
                + metrics["enumeration"]["sdpa_program_candidates"]["legal"]
            ),
            "rejected": metrics["pruning"]["program_candidates_rejected"],
            "microbench": (
                metrics["microbench"]["matmul_candidate_measurements"]
                + metrics["microbench"]["sdpa_op_or_region_measurements"]
            ),
        },
        "layout_search": {
            **metrics["enumeration"]["layout_states"],
            "microbench": metrics["microbench"]["layout_conversion_measurements"],
        },
    }
    built = []
    for row in rows:
        name = str(row["name"])
        enabled = _validate_enabled(row["enabled"], name)
        selected_domains = [domain for domain in domains if enabled.get(domain, False)]
        enumerated = sum(domains[name]["enumerated"] for name in selected_domains)
        possible_rejected = sum(domains[name]["rejected"] for name in selected_domains)
        pruned = possible_rejected if enabled["analytical_pruning"] else 0
        surviving = enumerated - pruned
        microbench_count = sum(domains[name]["microbench"] for name in selected_domains)
        transfer = metrics["representative_layer_transfer"]
        layer_multiplier = (
            transfer["representative_layer_count"]
            if enabled["representative_layer_transfer"]
            else transfer["layer_count"]
        )
        planned_layer_evaluations = (
            transfer["candidate_count_at_layer_confirmation"] * layer_multiplier
        )
        evidence_labels = [str(label) for label in row["evidence_labels"]]
        unknown = [label for label in evidence_labels if label not in source_labels]
        if unknown:
            raise PaperArtifactError(
                f"ablation {name} references unknown evidence: {', '.join(unknown)}"
            )
        full_model_count = int(
            row.get(
                "full_model_profile_count",
                metrics["full_model"]["profile_count"],
            )
        )
        if full_model_count < 0:
            raise PaperArtifactError(
                f"ablation {name} full_model_profile_count must be non-negative"
            )
        built.append(
            {
                "name": name,
                "mode": row["mode"],
                "enabled": enabled,
                "equivalent_to": row.get("equivalent_to"),
                "selection": str(row.get("selection", "retain_incumbent")),
                "enumerated_candidate_count": enumerated,
                "analytically_pruned_candidate_count": pruned,
                "surviving_candidate_count": surviving,
                "microbench_experiment_count": microbench_count,
                "planned_layer_evaluation_count": planned_layer_evaluations,
                "full_model_profile_count": full_model_count,
                "best_performance_tpsu": metrics["best_performance"]["value"],
                "official_ratio": metrics["best_performance"]["official_ratio"],
                "accuracy_passed": metrics["accuracy"]["passed"],
                "selected_config_sha256": metrics["selected_config_sha256"],
                "evidence_labels": evidence_labels,
                "notes": [str(note) for note in row["notes"]],
            }
        )
    _validate_built_ablations(built)
    return built


def _validate_built_ablations(value: Any) -> None:
    if not isinstance(value, list):
        raise PaperArtifactError("paper artifact ablations must be a list")
    names = {row.get("name") for row in value if isinstance(row, Mapping)}
    if names != set(REQUIRED_ABLATIONS):
        raise PaperArtifactError("paper artifact ablation matrix is incomplete")
    for row in value:
        if not isinstance(row, Mapping):
            raise PaperArtifactError("paper artifact ablation rows must be objects")
        if float(row.get("best_performance_tpsu", 0.0)) <= 0:
            raise PaperArtifactError(
                f"ablation {row.get('name')} has no best performance"
            )
        if float(row.get("official_ratio", 0.0)) <= 0:
            raise PaperArtifactError(
                f"ablation {row.get('name')} has no official ratio"
            )
        if not row.get("accuracy_passed"):
            raise PaperArtifactError(
                f"ablation {row.get('name')} did not preserve accuracy"
            )


def _validate_metric_surface(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise PaperArtifactError("paper artifact metrics must be an object")
    required = (
        "search_time",
        "enumeration",
        "pruning",
        "microbench",
        "full_model",
        "best_performance",
        "accuracy",
        "generalization",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise PaperArtifactError(
            "paper artifact metrics are missing: " + ", ".join(missing)
        )
    if float(value["search_time"].get("device_process_seconds", 0.0)) <= 0:
        raise PaperArtifactError("paper artifact search time is not positive")
    if int(value["enumeration"].get("total_search_entities", 0)) <= 0:
        raise PaperArtifactError("paper artifact enumerated no candidates")
    if not 0.0 <= float(value["pruning"].get("aggregate_pruning_ratio", -1.0)) <= 1.0:
        raise PaperArtifactError("paper artifact pruning ratio is invalid")
    if int(value["microbench"].get("experiment_count", 0)) <= 0:
        raise PaperArtifactError("paper artifact has no microbench count")
    if int(value["full_model"].get("profile_count", 0)) <= 0:
        raise PaperArtifactError("paper artifact has no full-model count")
    if float(value["best_performance"].get("official_ratio", 0.0)) <= 0:
        raise PaperArtifactError("paper artifact has no official ratio")
    if not value["accuracy"].get("passed"):
        raise PaperArtifactError("paper artifact accuracy did not pass")


def _render_ablation_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "name",
        "mode",
        "config_search",
        "template_search",
        "layout_search",
        "representative_layer_transfer",
        "analytical_pruning",
        "enumerated_candidate_count",
        "analytically_pruned_candidate_count",
        "surviving_candidate_count",
        "microbench_experiment_count",
        "planned_layer_evaluation_count",
        "full_model_profile_count",
        "best_performance_tpsu",
        "official_ratio",
        "accuracy_passed",
        "selected_config_sha256",
        "equivalent_to",
        "selection",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        enabled = row["enabled"]
        writer.writerow(
            {
                **{key: row.get(key) for key in fields},
                **{key: enabled[key] for key in _ENABLED_KEYS},
            }
        )
    return stream.getvalue()


def _compact_evidence(
    artifact: Mapping[str, Any], source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": PAPER_ARTIFACT_SCHEMA_VERSION,
        "phase": 10,
        "status": "passed",
        "title": "Reproducible semantic-autotune paper artifact",
        "campaign": copy.deepcopy(artifact["campaign"]),
        "campaign_fingerprint": artifact["campaign_fingerprint"],
        "metrics": copy.deepcopy(artifact["metrics"]),
        "ablations": [
            {
                key: copy.deepcopy(row[key])
                for key in (
                    "name",
                    "mode",
                    "enabled",
                    "equivalent_to",
                    "selection",
                    "enumerated_candidate_count",
                    "analytically_pruned_candidate_count",
                    "surviving_candidate_count",
                    "microbench_experiment_count",
                    "planned_layer_evaluation_count",
                    "full_model_profile_count",
                    "best_performance_tpsu",
                    "official_ratio",
                    "accuracy_passed",
                    "selected_config_sha256",
                )
            }
            for row in artifact["ablations"]
        ],
        "source_evidence": copy.deepcopy(dict(source_manifest)),
        "acceptance": {
            "all_required_metrics_present": True,
            "all_required_ablations_present": True,
            "source_hashes_recorded": True,
            "accuracy_passed": artifact["metrics"]["accuracy"]["passed"],
            "generalization_passed": artifact["metrics"]["generalization"]["passed"],
            "passed": True,
        },
    }


def _render_markdown(artifact: Mapping[str, Any]) -> str:
    metrics = artifact["metrics"]
    performance = metrics["best_performance"]
    accuracy = metrics["accuracy"]
    lines = [
        "# TTNN Direct Semantic Autotune Artifact",
        "",
        f"Campaign: `{artifact['campaign']['name']}`",
        "",
        "## Core Results",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Search device-process time | {metrics['search_time']['device_process_seconds']:.3f} s |",
        f"| Enumerated search entities | {metrics['enumeration']['total_search_entities']} |",
        f"| Program candidate pruning ratio | {metrics['pruning']['program_candidate_pruning_ratio']:.2%} |",
        f"| Microbench experiments | {metrics['microbench']['experiment_count']} |",
        f"| Full-model profiles | {metrics['full_model']['profile_count']} |",
        f"| Best performance | {performance['value']:.4f} t/s/u |",
        f"| Corresponding-release ratio | {performance['official_ratio']:.4%} |",
        f"| Fixed-corpus minimum top-1 | {accuracy['buddy_min_user_top1_accuracy']:.4f} |",
        f"| Logits PCC | {accuracy['logits_pcc']:.6f} |",
        "",
        "## Ablation",
        "",
        "| Variant | Mode | Enumerated | Pruned | Layer evals | Full model | t/s/u | Official ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in artifact["ablations"]:
        lines.append(
            "| {name} | {mode} | {enumerated_candidate_count} | "
            "{analytically_pruned_candidate_count} | "
            "{planned_layer_evaluation_count} | {full_model_profile_count} | "
            "{best_performance_tpsu:.4f} | {official_ratio:.4%} |".format(**row)
        )
    lines.extend(
        [
            "",
            "`reused_equivalent` rows reuse the strict matched measurement only "
            "when the selected config SHA256 is identical. `analytical` rows "
            "change search cost accounting but not the legal final config.",
            "",
            "Run `./reproduce.sh` to rebuild and verify the bundle against the "
            "recorded source hashes.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_reproduce_script(path: Path) -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
MODULE="models.llama_ttnn_direct.buddy_ttnn_direct.autotune.artifact_cli"

if [[ -n "${BUDDY_REPO_ROOT:-}" ]]; then
  REPO="$BUDDY_REPO_ROOT"
else
  REPO="$ROOT"
  while [[ "$REPO" != "/" && ! -f "$REPO/models/llama_ttnn_direct/buddy_ttnn_direct/__init__.py" ]]; do
    REPO="$(dirname "$REPO")"
  done
fi
if [[ ! -f "$REPO/models/llama_ttnn_direct/buddy_ttnn_direct/__init__.py" ]]; then
  printf 'unable to locate Buddy repository; set BUDDY_REPO_ROOT\n' >&2
  exit 2
fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

"$PYTHON" -m "$MODULE" verify --artifact-dir "$ROOT"
REBUILT="$ROOT/rebuilt-$(date +%Y%m%d_%H%M%S)-$$"
"$PYTHON" -m "$MODULE" build --spec "$ROOT/spec.json" --out-dir "$REBUILT"
"$PYTHON" -m "$MODULE" verify --artifact-dir "$REBUILT"
printf 'rebuilt artifact: %s\n' "$REBUILT"
"""
    _atomic_write_text(path, content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _artifact_manifest(root: Path, *, filenames: Sequence[str]) -> dict[str, Any]:
    files = {}
    for filename in filenames:
        path = root / filename
        files[filename] = {
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema_version": PAPER_ARTIFACT_SCHEMA_VERSION,
        "status": "passed",
        "files": files,
    }


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperArtifactError(f"{label} must be numeric") from exc
    if result <= 0:
        raise PaperArtifactError(f"{label} must be positive")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
