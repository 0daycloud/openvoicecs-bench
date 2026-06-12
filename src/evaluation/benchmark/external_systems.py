"""External-system registry validation for OpenVoiceCS leaderboard evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_EXTERNAL_SYSTEMS_PATH = Path("data/openvoicecs/external_systems_v0.1.json")
SYSTEM_STATUSES = {"reference_fixture", "pending_external", "official", "rejected", "retired"}
SYSTEM_TYPES = {"reference", "external_voice_agent", "external_text_agent", "research_system"}
INPUT_MODALITIES = {"text", "audio", "multimodal"}
PIPELINE_TYPES = {"cascaded", "native_speech_to_speech", "unknown"}


@dataclass(frozen=True)
class ExternalSystemIssue:
    """Structured external-system registry validation issue."""

    item_id: str
    path: str
    message: str


def load_external_systems_registry(
    path: str | Path = DEFAULT_EXTERNAL_SYSTEMS_PATH,
) -> dict[str, Any]:
    """Load and validate the external-system registry."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        registry = json.load(f)
    issues = validate_external_systems_registry(registry)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS external systems validation failed:\n{formatted}")
    return registry


def validate_external_systems_registry_file(
    path: str | Path = DEFAULT_EXTERNAL_SYSTEMS_PATH,
    *,
    base_dir: str | Path = ".",
) -> list[ExternalSystemIssue]:
    """Validate a saved external-system registry JSON file."""
    with open(path, encoding="utf-8") as f:
        registry = json.load(f)
    return validate_external_systems_registry(registry, base_dir=base_dir)


def validate_external_systems_registry(
    registry: dict[str, Any],
    *,
    base_dir: str | Path = ".",
) -> list[ExternalSystemIssue]:
    """Return all external-system registry contract issues."""
    issues: list[ExternalSystemIssue] = []
    if not isinstance(registry, dict):
        return [ExternalSystemIssue("<external_systems>", "<root>", "must be an object")]
    for field in ("name", "version", "benchmark_version", "admission_policy", "systems"):
        if field not in registry:
            issues.append(ExternalSystemIssue("<external_systems>", field, "missing required field"))
    if issues:
        return issues
    if registry.get("name") != "OpenVoiceCS External Systems Registry":
        issues.append(
            ExternalSystemIssue(
                "<external_systems>",
                "name",
                "must be OpenVoiceCS External Systems Registry",
            )
        )
    for field in ("version", "benchmark_version"):
        if not _non_empty_string(registry.get(field)):
            issues.append(ExternalSystemIssue("<external_systems>", field, "must be a non-empty string"))

    _validate_admission_policy(issues, registry.get("admission_policy"))
    systems = registry.get("systems")
    if not isinstance(systems, list):
        issues.append(ExternalSystemIssue("<external_systems>", "systems", "must be a list"))
        return issues
    seen_ids: set[str] = set()
    for index, system in enumerate(systems):
        _validate_system_entry(
            issues,
            system,
            index=index,
            seen_ids=seen_ids,
            base_dir=Path(base_dir),
        )
    return issues


def external_systems_stats(registry: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize external-system registry release evidence."""
    if not isinstance(registry, dict):
        return {"present": False, "num_systems": 0}
    systems = registry.get("systems", [])
    systems = systems if isinstance(systems, list) else []
    by_status: dict[str, int] = {}
    for system in systems:
        if isinstance(system, dict):
            status = str(system.get("status") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
    return {
        "present": True,
        "version": registry.get("version"),
        "benchmark_version": registry.get("benchmark_version"),
        "num_systems": len(systems),
        "by_status": by_status,
        "official_systems": by_status.get("official", 0),
        "reference_fixtures": by_status.get("reference_fixture", 0),
        "pending_external": by_status.get("pending_external", 0),
    }


def _validate_admission_policy(
    issues: list[ExternalSystemIssue],
    policy: Any,
) -> None:
    if not isinstance(policy, dict):
        issues.append(ExternalSystemIssue("<external_systems>", "admission_policy", "must be an object"))
        return
    for field in (
        "official_requires_submission_card",
        "official_requires_run_manifest",
        "official_requires_judged_report",
        "official_requires_judge_annotation_package",
        "official_requires_pricing_profile",
        "official_requires_release_bundle",
    ):
        if policy.get(field) is not True:
            issues.append(ExternalSystemIssue("<external_systems>", f"admission_policy.{field}", "must be true"))
    min_trials = policy.get("minimum_trials_per_scenario")
    if isinstance(min_trials, bool) or not isinstance(min_trials, int) or min_trials < 1:
        issues.append(
            ExternalSystemIssue(
                "<external_systems>",
                "admission_policy.minimum_trials_per_scenario",
                "must be an integer >= 1",
            )
        )


def _validate_system_entry(
    issues: list[ExternalSystemIssue],
    system: Any,
    *,
    index: int,
    seen_ids: set[str],
    base_dir: Path,
) -> None:
    path = f"systems[{index}]"
    if not isinstance(system, dict):
        issues.append(ExternalSystemIssue(path, path, "must be an object"))
        return
    system_id = system.get("id")
    if not _non_empty_string(system_id):
        issues.append(ExternalSystemIssue(path, f"{path}.id", "must be a non-empty string"))
        system_id = path
    elif system_id in seen_ids:
        issues.append(ExternalSystemIssue(str(system_id), f"{path}.id", "duplicate system id"))
    seen_ids.add(str(system_id))

    status = system.get("status")
    if status not in SYSTEM_STATUSES:
        issues.append(
            ExternalSystemIssue(
                str(system_id),
                f"{path}.status",
                f"must be one of: {', '.join(sorted(SYSTEM_STATUSES))}",
            )
        )
    system_type = system.get("system_type")
    if system_type not in SYSTEM_TYPES:
        issues.append(
            ExternalSystemIssue(
                str(system_id),
                f"{path}.system_type",
                f"must be one of: {', '.join(sorted(SYSTEM_TYPES))}",
            )
        )
    for field in ("name", "provider"):
        if not _non_empty_string(system.get(field)):
            issues.append(ExternalSystemIssue(str(system_id), f"{path}.{field}", "must be a non-empty string"))
    if not _non_empty_string(system.get("model_id")) and not _non_empty_string(system.get("submission_spec")):
        issues.append(
            ExternalSystemIssue(
                str(system_id),
                f"{path}.model_id",
                "must include model_id or submission_spec",
            )
        )
    if system.get("input_modality") not in INPUT_MODALITIES:
        issues.append(
            ExternalSystemIssue(
                str(system_id),
                f"{path}.input_modality",
                f"must be one of: {', '.join(sorted(INPUT_MODALITIES))}",
            )
        )
    if system.get("pipeline_type") not in PIPELINE_TYPES:
        issues.append(
            ExternalSystemIssue(
                str(system_id),
                f"{path}.pipeline_type",
                f"must be one of: {', '.join(sorted(PIPELINE_TYPES))}",
            )
        )
    if not isinstance(system.get("official_leaderboard_eligible"), bool):
        issues.append(
            ExternalSystemIssue(
                str(system_id),
                f"{path}.official_leaderboard_eligible",
                "must be boolean",
            )
        )
    if status == "reference_fixture" and system.get("official_leaderboard_eligible") is not False:
        issues.append(
            ExternalSystemIssue(
                str(system_id),
                f"{path}.official_leaderboard_eligible",
                "reference fixtures cannot be official leaderboard entries",
            )
        )

    report_entry = system.get("report")
    if report_entry is not None:
        _validate_file_entry(
            issues,
            report_entry,
            f"{path}.report",
            str(system_id),
            base_dir=base_dir,
            kind="report",
        )
    card_entry = system.get("submission_card")
    if card_entry is not None:
        _validate_file_entry(
            issues,
            card_entry,
            f"{path}.submission_card",
            str(system_id),
            base_dir=base_dir,
            kind="submission_card",
        )
    for field in ("run_manifest", "release_bundle"):
        if system.get(field) is not None:
            _validate_file_entry(
                issues,
                system[field],
                f"{path}.{field}",
                str(system_id),
                base_dir=base_dir,
                kind=None,
            )
    if status == "official":
        _validate_official_system_requirements(issues, system, path, str(system_id))


def _validate_official_system_requirements(
    issues: list[ExternalSystemIssue],
    system: dict[str, Any],
    path: str,
    system_id: str,
) -> None:
    if system.get("official_leaderboard_eligible") is not True:
        issues.append(
            ExternalSystemIssue(
                system_id,
                f"{path}.official_leaderboard_eligible",
                "official systems must be leaderboard eligible",
            )
        )
    provider = str(system.get("provider") or "").strip().lower()
    if provider in {"reference", "openvoicecs_reference"}:
        issues.append(ExternalSystemIssue(system_id, f"{path}.provider", "official systems must not use reference provider"))
    for field in ("report", "submission_card", "run_manifest", "release_bundle", "judge_annotation_package"):
        if not isinstance(system.get(field), dict):
            issues.append(ExternalSystemIssue(system_id, f"{path}.{field}", "official systems must include this file entry"))
    for field in ("pricing_profile_id", "pricing_snapshot_date"):
        if not _non_empty_string(system.get(field)):
            issues.append(ExternalSystemIssue(system_id, f"{path}.{field}", "official systems must include this field"))
    evidence = system.get("judge_evidence")
    if not isinstance(evidence, dict):
        issues.append(ExternalSystemIssue(system_id, f"{path}.judge_evidence", "official systems must include judge evidence"))
        return
    if evidence.get("annotation_mode") == "reference_fixture":
        issues.append(
            ExternalSystemIssue(
                system_id,
                f"{path}.judge_evidence.annotation_mode",
                "official systems cannot use reference_fixture annotations",
            )
        )
    if evidence.get("minimum_raters_per_item", 0) < 2:
        issues.append(
            ExternalSystemIssue(
                system_id,
                f"{path}.judge_evidence.minimum_raters_per_item",
                "must be >= 2",
            )
        )


def _validate_file_entry(
    issues: list[ExternalSystemIssue],
    entry: Any,
    path: str,
    system_id: str,
    *,
    base_dir: Path,
    kind: str | None,
) -> None:
    if not isinstance(entry, dict):
        issues.append(ExternalSystemIssue(system_id, path, "must be an object"))
        return
    for field in ("path", "sha256", "bytes"):
        if field not in entry:
            issues.append(ExternalSystemIssue(system_id, f"{path}.{field}", "missing required field"))
    file_path = entry.get("path")
    if not _non_empty_string(file_path):
        issues.append(ExternalSystemIssue(system_id, f"{path}.path", "must be a non-empty string"))
        return
    resolved = Path(str(file_path))
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    if not resolved.exists():
        issues.append(ExternalSystemIssue(system_id, f"{path}.path", "file does not exist"))
        return
    data = resolved.read_bytes()
    expected_sha = hashlib.sha256(data).hexdigest()
    if entry.get("sha256") != expected_sha:
        issues.append(ExternalSystemIssue(system_id, f"{path}.sha256", "does not match file contents"))
    if entry.get("bytes") != len(data):
        issues.append(ExternalSystemIssue(system_id, f"{path}.bytes", "does not match file size"))
    if kind == "report":
        from src.evaluation.benchmark.openvoicecs import validate_report_file

        issues.extend(
            ExternalSystemIssue(system_id, f"{path}.{issue.path}", issue.message)
            for issue in validate_report_file(resolved)
        )
    elif kind == "submission_card":
        from src.evaluation.benchmark.submission import validate_submission_card_file

        issues.extend(
            ExternalSystemIssue(system_id, f"{path}.{issue.path}", issue.message)
            for issue in validate_submission_card_file(resolved)
        )


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
