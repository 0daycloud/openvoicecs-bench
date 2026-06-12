"""Frozen run manifests for latency-cost-quality frontier releases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from src.evaluation.benchmark.changelog import DEFAULT_CHANGELOG_PATH
from src.evaluation.benchmark.openvoicecs import (
    DEFAULT_AUDIO_MANIFEST_PATH,
    DEFAULT_BASELINE_MANIFEST_PATH,
    DEFAULT_REVIEW_MANIFEST_PATH,
    DEFAULT_SCENARIO_PATH,
    build_release_audit,
    validate_report,
)
from src.evaluation.benchmark.pricing import DEFAULT_PRICING_MANIFEST_PATH
from src.evaluation.benchmark.pricing import resolve_report_pricing
from src.evaluation.benchmark.provenance import DEFAULT_PROVENANCE_MANIFEST_PATH
from src.evaluation.benchmark.splits import DEFAULT_SPLIT_MANIFEST_PATH


RUN_MANIFEST_VERSION = "0.1.0"


@dataclass(frozen=True)
class RunManifestIssue:
    """Structured run manifest validation issue."""

    path: str
    message: str


def build_run_manifest(
    report_paths: list[str | Path],
    *,
    scenario_path: str | Path = DEFAULT_SCENARIO_PATH,
    audio_manifest_path: str | Path | None = DEFAULT_AUDIO_MANIFEST_PATH,
    audio_asset_root: str | Path = ".",
    pricing_manifest_path: str | Path | None = DEFAULT_PRICING_MANIFEST_PATH,
    split_manifest_path: str | Path | None = DEFAULT_SPLIT_MANIFEST_PATH,
    provenance_manifest_path: str | Path | None = DEFAULT_PROVENANCE_MANIFEST_PATH,
    changelog_path: str | Path | None = DEFAULT_CHANGELOG_PATH,
    baseline_manifest_path: str | Path | None = DEFAULT_BASELINE_MANIFEST_PATH,
    review_manifest_path: str | Path | None = DEFAULT_REVIEW_MANIFEST_PATH,
    judge_model: str | None = None,
    judge_prompt_path: str | Path | None = None,
    seed: int = 0,
    region: str | None = None,
    network: str | None = None,
    hardware_profile: str | None = None,
    transport: str | None = None,
    concurrency_levels: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Build a frozen manifest for one frontier benchmark run/release."""
    reports = [_report_entry(Path(path)) for path in report_paths]
    release_audit = build_release_audit(
        scenario_path=scenario_path,
        audio_manifest_path=audio_manifest_path,
        audio_asset_root=audio_asset_root,
        pricing_manifest_path=pricing_manifest_path,
        split_manifest_path=split_manifest_path,
        provenance_manifest_path=provenance_manifest_path,
        changelog_path=changelog_path,
        baseline_manifest_path=baseline_manifest_path,
        review_manifest_path=review_manifest_path,
        sealed_queue_path=None,
        external_systems_path=None,
        claims_manifest_path=None,
        submission_intake_path=None,
    )
    judge_prompt = _prompt_entry(judge_prompt_path)
    pricing_manifest = _load_optional_json(pricing_manifest_path)
    systems = [_system_entry(entry["report"], pricing_manifest) for entry in reports]
    environment = _environment_entry(
        reports=[entry["report"] for entry in reports],
        region=region,
        network=network,
        hardware_profile=hardware_profile,
        transport=transport,
        concurrency_levels=concurrency_levels,
    )

    return {
        "benchmark": "Latency-Cost-Quality Frontier",
        "manifest_version": RUN_MANIFEST_VERSION,
        "generated_at": time.strftime("%Y-%m-%d"),
        "release_tuple": {
            "scenario_suite": release_audit["files"].get("scenario_suite"),
            "audio_manifest": release_audit["files"].get("audio_manifest"),
            "pricing_manifest": release_audit["files"].get("pricing_manifest"),
            "split_manifest": release_audit["files"].get("split_manifest"),
            "provenance_manifest": release_audit["files"].get("provenance_manifest"),
            "changelog": release_audit["files"].get("changelog"),
            "baseline_manifest": release_audit["files"].get("baseline_manifest"),
            "review_manifest": release_audit["files"].get("review_manifest"),
            "seed": seed,
            "judge": {
                "model": judge_model,
                "prompt": judge_prompt,
            },
            "environment": environment,
        },
        "release_audit": {
            "version": release_audit.get("version"),
            "release_stage": release_audit.get("release_stage"),
            "validation": release_audit.get("validation", {}),
            "release_gates": release_audit.get("release_gates", {}),
            "scenario_stats": release_audit.get("scenario_stats", {}),
            "audio_manifest_stats": release_audit.get("audio_manifest_stats", {}),
            "audio_asset_stats": release_audit.get("audio_asset_stats", {}),
            "pricing_manifest_stats": release_audit.get("pricing_manifest_stats", {}),
            "split_manifest_stats": release_audit.get("split_manifest_stats", {}),
            "provenance_stats": release_audit.get("provenance_stats", {}),
            "changelog_stats": release_audit.get("changelog_stats", {}),
            "baseline_stats": release_audit.get("baseline_stats", {}),
            "review_stats": release_audit.get("review_stats", {}),
        },
        "reports": [
            {
                key: value
                for key, value in entry.items()
                if key != "report"
            }
            for entry in reports
        ],
        "systems": systems,
    }


def validate_run_manifest_file(path: str | Path) -> list[RunManifestIssue]:
    """Load and validate a saved run manifest."""
    manifest_path = Path(path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    return validate_run_manifest(manifest, base_dir=manifest_path.parent, verify_files=True)


def validate_run_manifest(
    manifest: dict[str, Any],
    *,
    base_dir: str | Path = ".",
    verify_files: bool = False,
) -> list[RunManifestIssue]:
    """Validate a run manifest contract."""
    issues: list[RunManifestIssue] = []
    if not isinstance(manifest, dict):
        return [RunManifestIssue("<root>", "must be an object")]
    for field in ("benchmark", "manifest_version", "release_tuple", "reports", "systems"):
        if field not in manifest:
            issues.append(RunManifestIssue(field, "missing required field"))
    if issues:
        return issues

    release_tuple = manifest.get("release_tuple")
    if not isinstance(release_tuple, dict):
        issues.append(RunManifestIssue("release_tuple", "must be an object"))
    else:
        for field in (
            "scenario_suite",
            "pricing_manifest",
            "split_manifest",
            "provenance_manifest",
            "changelog",
            "baseline_manifest",
            "review_manifest",
            "seed",
            "judge",
            "environment",
        ):
            if field not in release_tuple:
                issues.append(RunManifestIssue(f"release_tuple.{field}", "missing required field"))
        for file_field in (
            "scenario_suite",
            "audio_manifest",
            "pricing_manifest",
            "split_manifest",
            "provenance_manifest",
            "changelog",
            "baseline_manifest",
            "review_manifest",
        ):
            item = release_tuple.get(file_field)
            if item is not None:
                _validate_file_entry(
                    issues,
                    f"release_tuple.{file_field}",
                    item,
                    base_dir=Path(base_dir),
                    verify_file=verify_files,
                )
        if not isinstance(release_tuple.get("seed"), int):
            issues.append(RunManifestIssue("release_tuple.seed", "must be an integer"))
        judge = release_tuple.get("judge")
        if not isinstance(judge, dict):
            issues.append(RunManifestIssue("release_tuple.judge", "must be an object"))
        environment = release_tuple.get("environment")
        if not isinstance(environment, dict):
            issues.append(RunManifestIssue("release_tuple.environment", "must be an object"))
        else:
            for field in ("region", "network", "transport"):
                if not _is_controlled_label(environment.get(field)):
                    issues.append(
                        RunManifestIssue(
                            f"release_tuple.environment.{field}",
                            "must be a non-empty controlled value",
                        )
                    )
            levels = environment.get("concurrency_levels")
            if (
                not isinstance(levels, list)
                or not levels
                or any(not isinstance(level, int) or level < 1 for level in levels)
            ):
                issues.append(
                    RunManifestIssue(
                        "release_tuple.environment.concurrency_levels",
                        "must be positive integers",
                    )
                )

    reports = manifest.get("reports")
    if not isinstance(reports, list) or not reports:
        issues.append(RunManifestIssue("reports", "must be a non-empty list"))
    else:
        for index, report in enumerate(reports):
            _validate_report_entry(
                issues,
                f"reports[{index}]",
                report,
                base_dir=Path(base_dir),
                verify_file=verify_files,
            )

    systems = manifest.get("systems")
    if not isinstance(systems, list) or not systems:
        issues.append(RunManifestIssue("systems", "must be a non-empty list"))
    else:
        for index, system in enumerate(systems):
            _validate_system_entry(issues, f"systems[{index}]", system)
    return issues


def _report_entry(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    validation_issues = validate_report(report)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "benchmark": report.get("benchmark"),
        "benchmark_version": report.get("benchmark_version"),
        "num_scenarios": report.get("num_scenarios"),
        "num_trials_per_scenario": report.get("num_trials_per_scenario"),
        "validation": {
            "passed": not validation_issues,
            "num_issues": len(validation_issues),
            "issues": [
                {
                    "scenario_id": issue.scenario_id,
                    "path": issue.path,
                    "message": issue.message,
                }
                for issue in validation_issues
            ],
        },
        "report": report,
    }


def _system_entry(
    report: dict[str, Any],
    pricing_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = report.get("model_metadata", {})
    embedded_pricing = report.get("pricing") or metadata.get("pricing") or {}
    resolved_pricing = resolve_report_pricing(report, pricing_manifest)
    return {
        "name": (
            metadata.get("display_name")
            or metadata.get("model_id")
            or metadata.get("agent")
            or "unknown"
        ),
        "provider": metadata.get("provider"),
        "model_id": metadata.get("model_id"),
        "pricing_profile_id": (
            metadata.get("pricing_profile_id")
            or metadata.get("pricing_profile")
            or metadata.get("profile_id")
            or resolved_pricing.get("profile_id")
        ),
        "pricing_snapshot_date": (
            metadata.get("pricing_snapshot_date")
            or report.get("pricing_snapshot_date")
            or resolved_pricing.get("snapshot_date")
        ),
        "pricing_source": (
            "profile"
            if resolved_pricing.get("profile_id") else
            ("embedded" if isinstance(embedded_pricing, dict) and embedded_pricing else None)
        ),
        "pipeline_type": (
            metadata.get("pipeline_type")
            or resolved_pricing.get("pipeline_type")
            or (
                embedded_pricing.get("pipeline_type")
                if isinstance(embedded_pricing, dict)
                else None
            )
        ),
        "submission_spec": metadata.get("submission_spec"),
        "transport": (
            metadata.get("transport")
            or report.get("reference_client", {}).get("transport")
        ),
        "reference_client": report.get("reference_client"),
        "judge_model": report.get("judge_model"),
        "benchmark": report.get("benchmark"),
    }


def _environment_entry(
    *,
    reports: list[dict[str, Any]],
    region: str | None,
    network: str | None,
    hardware_profile: str | None,
    transport: str | None,
    concurrency_levels: list[int] | tuple[int, ...] | None,
) -> dict[str, Any]:
    report_env = next(
        (
            report.get("environment") or report.get("reference_client")
            for report in reports
            if report.get("environment") or report.get("reference_client")
        ),
        {},
    )
    levels = concurrency_levels or report_env.get("concurrency_levels")
    return {
        "region": region or report_env.get("region"),
        "network": network or report_env.get("network"),
        "hardware_profile": hardware_profile or report_env.get("hardware_profile"),
        "transport": transport or report_env.get("transport"),
        "concurrency_levels": list(levels) if levels is not None else None,
    }


def _prompt_entry(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _is_controlled_label(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value.strip() != "unspecified"


def _validate_file_entry(
    issues: list[RunManifestIssue],
    path: str,
    item: Any,
    *,
    base_dir: Path | None = None,
    verify_file: bool = False,
) -> None:
    if not isinstance(item, dict):
        issues.append(RunManifestIssue(path, "must be an object"))
        return
    for field in ("path", "sha256", "bytes"):
        if field not in item:
            issues.append(RunManifestIssue(f"{path}.{field}", "missing required field"))
    sha = item.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        issues.append(RunManifestIssue(f"{path}.sha256", "must be a SHA-256 hex digest"))
    if not isinstance(item.get("bytes"), int) or item["bytes"] < 0:
        issues.append(RunManifestIssue(f"{path}.bytes", "must be a nonnegative integer"))
    if verify_file:
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            issues.append(RunManifestIssue(f"{path}.path", "must be a non-empty string"))
            return
        file_path = _resolve_manifest_file_path(raw_path, base_dir or Path("."))
        if not file_path.exists():
            issues.append(RunManifestIssue(f"{path}.path", "file does not exist"))
            return
        if isinstance(sha, str) and len(sha) == 64 and _sha256(file_path) != sha:
            issues.append(RunManifestIssue(f"{path}.sha256", "does not match file contents"))
        expected_bytes = item.get("bytes")
        if (
            isinstance(expected_bytes, int)
            and expected_bytes >= 0
            and file_path.stat().st_size != expected_bytes
        ):
            issues.append(RunManifestIssue(f"{path}.bytes", "does not match file size"))


def _validate_report_entry(
    issues: list[RunManifestIssue],
    path: str,
    item: Any,
    *,
    base_dir: Path | None = None,
    verify_file: bool = False,
) -> None:
    if not isinstance(item, dict):
        issues.append(RunManifestIssue(path, "must be an object"))
        return
    _validate_file_entry(
        issues,
        path,
        item,
        base_dir=base_dir,
        verify_file=verify_file,
    )
    validation = item.get("validation")
    if not isinstance(validation, dict):
        issues.append(RunManifestIssue(f"{path}.validation", "must be an object"))
    elif validation.get("passed") is not True:
        issues.append(RunManifestIssue(f"{path}.validation.passed", "must be true"))


def _validate_system_entry(
    issues: list[RunManifestIssue],
    path: str,
    item: Any,
) -> None:
    if not isinstance(item, dict):
        issues.append(RunManifestIssue(path, "must be an object"))
        return
    if not _is_controlled_label(item.get("name")) or item.get("name") == "unknown":
        issues.append(RunManifestIssue(f"{path}.name", "must be a non-empty system name"))
    if not _is_controlled_label(item.get("provider")):
        issues.append(RunManifestIssue(f"{path}.provider", "must be a non-empty provider"))
    if not (
        _is_controlled_label(item.get("model_id"))
        or _is_controlled_label(item.get("submission_spec"))
    ):
        issues.append(
            RunManifestIssue(
                f"{path}.model_id",
                "must include model_id or submission_spec",
            )
        )
    if not _is_controlled_label(item.get("pricing_snapshot_date")):
        issues.append(RunManifestIssue(f"{path}.pricing_snapshot_date", "must be pinned"))
    if item.get("pricing_source") not in {"profile", "embedded"}:
        issues.append(RunManifestIssue(f"{path}.pricing_source", "must be profile or embedded"))
    if not _is_controlled_label(item.get("pipeline_type")):
        issues.append(RunManifestIssue(f"{path}.pipeline_type", "must be pinned"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_manifest_file_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path
