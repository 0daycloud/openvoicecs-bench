"""Machine-readable datasheets for OpenVoiceCS benchmark releases."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.baselines import (
    DEFAULT_BASELINE_MANIFEST_PATH,
    reference_baseline_stats,
)
from src.evaluation.benchmark.changelog import DEFAULT_CHANGELOG_PATH
from src.evaluation.benchmark.openvoicecs import (
    DEFAULT_AUDIO_MANIFEST_PATH,
    DEFAULT_SCENARIO_PATH,
    build_release_audit,
)
from src.evaluation.benchmark.pricing import DEFAULT_PRICING_MANIFEST_PATH
from src.evaluation.benchmark.provenance import DEFAULT_PROVENANCE_MANIFEST_PATH
from src.evaluation.benchmark.reviews import (
    DEFAULT_REVIEW_MANIFEST_PATH,
    scenario_review_stats,
)
from src.evaluation.benchmark.splits import (
    DEFAULT_SPLIT_COMMITMENT_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
)

DEFAULT_DATASHEET_PATH = Path("data/openvoicecs/datasheet_v0.1.json")


@dataclass(frozen=True)
class DatasheetIssue:
    """Structured benchmark-datasheet validation issue."""

    path: str
    message: str


def build_benchmark_datasheet_file(
    *,
    scenario_path: str | Path = DEFAULT_SCENARIO_PATH,
    audio_manifest_path: str | Path | None = DEFAULT_AUDIO_MANIFEST_PATH,
    pricing_manifest_path: str | Path | None = DEFAULT_PRICING_MANIFEST_PATH,
    split_manifest_path: str | Path | None = DEFAULT_SPLIT_MANIFEST_PATH,
    provenance_manifest_path: str | Path | None = DEFAULT_PROVENANCE_MANIFEST_PATH,
    changelog_path: str | Path | None = DEFAULT_CHANGELOG_PATH,
    baseline_manifest_path: str | Path | None = DEFAULT_BASELINE_MANIFEST_PATH,
    review_manifest_path: str | Path | None = DEFAULT_REVIEW_MANIFEST_PATH,
    split_commitment_path: str | Path | None = DEFAULT_SPLIT_COMMITMENT_PATH,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and optionally write a benchmark datasheet from release files."""
    datasheet = build_benchmark_datasheet(
        scenario_path=scenario_path,
        audio_manifest_path=audio_manifest_path,
        pricing_manifest_path=pricing_manifest_path,
        split_manifest_path=split_manifest_path,
        provenance_manifest_path=provenance_manifest_path,
        changelog_path=changelog_path,
        baseline_manifest_path=baseline_manifest_path,
        review_manifest_path=review_manifest_path,
        split_commitment_path=split_commitment_path,
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(datasheet, indent=2) + "\n", encoding="utf-8")
    return datasheet


def build_benchmark_datasheet(
    *,
    scenario_path: str | Path = DEFAULT_SCENARIO_PATH,
    audio_manifest_path: str | Path | None = DEFAULT_AUDIO_MANIFEST_PATH,
    pricing_manifest_path: str | Path | None = DEFAULT_PRICING_MANIFEST_PATH,
    split_manifest_path: str | Path | None = DEFAULT_SPLIT_MANIFEST_PATH,
    provenance_manifest_path: str | Path | None = DEFAULT_PROVENANCE_MANIFEST_PATH,
    changelog_path: str | Path | None = DEFAULT_CHANGELOG_PATH,
    baseline_manifest_path: str | Path | None = DEFAULT_BASELINE_MANIFEST_PATH,
    review_manifest_path: str | Path | None = DEFAULT_REVIEW_MANIFEST_PATH,
    split_commitment_path: str | Path | None = DEFAULT_SPLIT_COMMITMENT_PATH,
) -> dict[str, Any]:
    """Build a datasheet-style release description for OpenVoiceCS-Bench."""
    audit = build_release_audit(
        scenario_path=scenario_path,
        audio_manifest_path=audio_manifest_path,
        pricing_manifest_path=pricing_manifest_path,
        split_manifest_path=split_manifest_path,
        provenance_manifest_path=provenance_manifest_path,
        changelog_path=changelog_path,
        baseline_manifest_path=baseline_manifest_path,
        review_manifest_path=review_manifest_path,
    )
    split_commitment = _load_optional_json(split_commitment_path)
    baseline_manifest = _load_optional_json(baseline_manifest_path)
    review_manifest = _load_optional_json(review_manifest_path)
    scenario_ids = _scenario_ids(scenario_path)
    return {
        "benchmark": "OpenVoiceCS-Bench Datasheet",
        "datasheet_version": "0.1.0",
        "generated_at": time.strftime("%Y-%m-%d"),
        "release": {
            "benchmark_version": audit.get("version"),
            "release_stage": audit.get("release_stage"),
            "scenario_file": _file_entry(scenario_path),
            "audio_manifest_file": _optional_file_entry(audio_manifest_path),
            "pricing_manifest_file": _optional_file_entry(pricing_manifest_path),
            "split_manifest_file": _optional_file_entry(split_manifest_path),
            "provenance_manifest_file": _optional_file_entry(provenance_manifest_path),
            "changelog_file": _optional_file_entry(changelog_path),
            "baseline_manifest_file": _optional_file_entry(baseline_manifest_path),
            "review_manifest_file": _optional_file_entry(review_manifest_path),
            "split_commitment_file": _optional_file_entry(split_commitment_path),
            "split_commitment_root_hash": (split_commitment or {}).get("root_hash"),
        },
        "intended_use": [
            "Evaluate voice AI and customer-service agents on task completion, "
            "policy compliance, tool use, safety, privacy, authentication, "
            "latency, and cost.",
            "Support reproducible local development, scientific comparison, "
            "and audited frontier reporting.",
        ],
        "out_of_scope_uses": [
            "Ranking general conversational intelligence outside customer-service workflows.",
            "Using deterministic benchmark success as a substitute for production safety review.",
            "Training on sealed-test scenarios, transcripts, tool oracles, "
            "expected states, or audio assets.",
        ],
        "data_summary": {
            "num_scenarios": audit.get("scenario_stats", {}).get("num_scenarios"),
            "domains": audit.get("scenario_stats", {}).get("domains", {}),
            "tracks": audit.get("scenario_stats", {}).get("tracks", {}),
            "difficulty": audit.get("scenario_stats", {}).get("difficulty", {}),
            "num_audio_variants": audit.get("audio_manifest_stats", {}).get("num_variants"),
            "audio_perturbations": audit.get("audio_manifest_stats", {}).get(
                "perturbation_types",
                {},
            ),
        },
        "split_policy": {
            "manifest_version": audit.get("split_manifest_stats", {}).get("version"),
            "scenario_coverage": audit.get("split_manifest_stats", {}).get("scenario_coverage"),
            "audio_variant_coverage": audit.get("split_manifest_stats", {}).get(
                "audio_variant_coverage",
            ),
            "splits": audit.get("split_manifest_stats", {}).get("splits", {}),
            "sealed_test_policy": (
                "Sealed items must not be published with full transcripts, "
                "tool oracles, expected states, or audio assets before evaluation."
            ),
            "commitment_root_hash": (split_commitment or {}).get("root_hash"),
            "sealed_ids_revealed": (split_commitment or {}).get("privacy", {}).get(
                "sealed_test_ids_revealed",
            ),
        },
        "provenance": audit.get("provenance_stats", {}),
        "changelog": audit.get("changelog_stats", {}),
        "baselines": reference_baseline_stats(baseline_manifest),
        "scenario_reviews": scenario_review_stats(review_manifest, scenario_ids=scenario_ids),
        "metrics": {
            "primary_quality": (
                "overall_score with deterministic state, tool, policy, grounding, "
                "privacy, auth, safety, and experience components"
            ),
            "reliability": ["pass@k", "pass^k", "mean_pass_rate", "confidence_intervals"],
            "operations": [
                "v2v_ttfb_ms",
                "v2v_last_byte_ms",
                "cost_usd_per_successful_conversation",
            ],
            "subjective": (
                "Optional judged conversation_experience_score, kept separate "
                "from deterministic task score."
            ),
        },
        "known_limitations": [
            "The seed release is small and intended for method review, not final industry ranking.",
            "The v0.1 audio assets are synthetic TTS fixtures; public beta should add broader consented speaker coverage.",
            "The seed release has an empty sealed_test split; public_beta and "
            "leaderboard_v1 require sealed scenarios.",
        ],
        "governance": {
            "scenario_ids_stable_after_publication": True,
            "requires_provenance_manifest": True,
            "requires_split_manifest": True,
            "requires_changelog": True,
            "requires_reference_baselines": True,
            "requires_scenario_reviews": True,
            "requires_split_commitments_for_public_releases": True,
            "requires_report_validation_before_leaderboard": True,
            "requires_submission_cards_for_official_leaderboards": True,
        },
        "release_validation": {
            "audit_passed": audit.get("validation", {}).get("passed"),
            "release_gates_passed": audit.get("release_gates", {}).get("passed"),
            "release_gates": audit.get("release_gates", {}),
        },
    }


def validate_benchmark_datasheet_file(path: str | Path) -> list[DatasheetIssue]:
    """Validate a saved benchmark datasheet JSON file."""
    with open(path, encoding="utf-8") as f:
        datasheet = json.load(f)
    return validate_benchmark_datasheet(datasheet)


def validate_benchmark_datasheet(datasheet: dict[str, Any]) -> list[DatasheetIssue]:
    """Validate benchmark datasheet structure and release-governance fields."""
    issues: list[DatasheetIssue] = []
    if not isinstance(datasheet, dict):
        return [DatasheetIssue("<root>", "must be an object")]
    required = {
        "benchmark",
        "datasheet_version",
        "release",
        "intended_use",
        "out_of_scope_uses",
        "data_summary",
        "split_policy",
        "provenance",
        "changelog",
        "baselines",
        "scenario_reviews",
        "metrics",
        "known_limitations",
        "governance",
        "release_validation",
    }
    for field in sorted(required - set(datasheet)):
        issues.append(DatasheetIssue(field, "missing required field"))
    if issues:
        return issues
    if datasheet.get("benchmark") != "OpenVoiceCS-Bench Datasheet":
        issues.append(DatasheetIssue("benchmark", "must be OpenVoiceCS-Bench Datasheet"))
    _validate_release(issues, datasheet.get("release"))
    _validate_non_empty_string_list(issues, datasheet, "intended_use")
    _validate_non_empty_string_list(issues, datasheet, "out_of_scope_uses")
    _validate_data_summary(issues, datasheet.get("data_summary"))
    _validate_split_policy(issues, datasheet.get("split_policy"))
    _validate_changelog_summary(issues, datasheet.get("changelog"))
    _validate_baseline_summary(issues, datasheet.get("baselines"))
    _validate_review_summary(issues, datasheet.get("scenario_reviews"))
    _validate_governance(issues, datasheet.get("governance"))
    _validate_release_validation(issues, datasheet.get("release_validation"))
    _validate_non_empty_string_list(issues, datasheet, "known_limitations")
    return issues


def _validate_release(issues: list[DatasheetIssue], release: Any) -> None:
    if not isinstance(release, dict):
        issues.append(DatasheetIssue("release", "must be an object"))
        return
    for field in (
        "benchmark_version",
        "release_stage",
        "scenario_file",
        "split_manifest_file",
        "provenance_manifest_file",
    ):
        if field not in release:
            issues.append(DatasheetIssue(f"release.{field}", "missing required field"))
    for field in (
        "scenario_file",
        "audio_manifest_file",
        "pricing_manifest_file",
        "split_manifest_file",
        "provenance_manifest_file",
        "changelog_file",
        "baseline_manifest_file",
        "review_manifest_file",
        "split_commitment_file",
    ):
        entry = release.get(field)
        if entry is not None:
            _validate_file_entry(issues, entry, f"release.{field}")
    root_hash = release.get("split_commitment_root_hash")
    if root_hash is not None and not _looks_sha256(root_hash):
        issues.append(
            DatasheetIssue(
                "release.split_commitment_root_hash",
                "must be a SHA-256 hex digest",
            )
        )


def _validate_data_summary(issues: list[DatasheetIssue], summary: Any) -> None:
    if not isinstance(summary, dict):
        issues.append(DatasheetIssue("data_summary", "must be an object"))
        return
    for field in ("num_scenarios", "num_audio_variants"):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            issues.append(DatasheetIssue(f"data_summary.{field}", "must be a nonnegative integer"))
    for field in ("domains", "tracks", "difficulty"):
        if not isinstance(summary.get(field), dict) or not summary[field]:
            issues.append(DatasheetIssue(f"data_summary.{field}", "must be a non-empty object"))


def _validate_split_policy(issues: list[DatasheetIssue], split_policy: Any) -> None:
    if not isinstance(split_policy, dict):
        issues.append(DatasheetIssue("split_policy", "must be an object"))
        return
    if not isinstance(split_policy.get("splits"), dict):
        issues.append(DatasheetIssue("split_policy.splits", "must be an object"))
    for field in ("scenario_coverage", "audio_variant_coverage"):
        value = split_policy.get(field)
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            issues.append(DatasheetIssue(f"split_policy.{field}", "must be between 0 and 1"))
    root_hash = split_policy.get("commitment_root_hash")
    if root_hash is not None and not _looks_sha256(root_hash):
        issues.append(
            DatasheetIssue(
                "split_policy.commitment_root_hash",
                "must be a SHA-256 hex digest",
            )
        )
    if split_policy.get("sealed_ids_revealed") not in {True, False, None}:
        issues.append(DatasheetIssue("split_policy.sealed_ids_revealed", "must be boolean or null"))


def _validate_changelog_summary(issues: list[DatasheetIssue], changelog: Any) -> None:
    if not isinstance(changelog, dict):
        issues.append(DatasheetIssue("changelog", "must be an object"))
        return
    if changelog.get("present") is not True:
        issues.append(DatasheetIssue("changelog.present", "must be true"))
    for field in ("num_entries", "num_open_errata"):
        value = changelog.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            issues.append(DatasheetIssue(f"changelog.{field}", "must be a nonnegative integer"))


def _validate_baseline_summary(issues: list[DatasheetIssue], baselines: Any) -> None:
    if not isinstance(baselines, dict):
        issues.append(DatasheetIssue("baselines", "must be an object"))
        return
    if baselines.get("present") is not True:
        issues.append(DatasheetIssue("baselines.present", "must be true"))
    count = baselines.get("num_baselines")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        issues.append(DatasheetIssue("baselines.num_baselines", "must be a positive integer"))
    if not isinstance(baselines.get("baselines"), dict) or not baselines["baselines"]:
        issues.append(DatasheetIssue("baselines.baselines", "must be a non-empty object"))


def _validate_review_summary(issues: list[DatasheetIssue], reviews: Any) -> None:
    if not isinstance(reviews, dict):
        issues.append(DatasheetIssue("scenario_reviews", "must be an object"))
        return
    if reviews.get("present") is not True:
        issues.append(DatasheetIssue("scenario_reviews.present", "must be true"))
    if reviews.get("scenario_approval_coverage") != 1.0:
        issues.append(
            DatasheetIssue(
                "scenario_reviews.scenario_approval_coverage",
                "must be 1.0",
            )
        )
    count = reviews.get("num_approved_scenarios")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        issues.append(
            DatasheetIssue(
                "scenario_reviews.num_approved_scenarios",
                "must be a positive integer",
            )
        )


def _validate_governance(issues: list[DatasheetIssue], governance: Any) -> None:
    if not isinstance(governance, dict):
        issues.append(DatasheetIssue("governance", "must be an object"))
        return
    required_true = (
        "scenario_ids_stable_after_publication",
        "requires_provenance_manifest",
        "requires_split_manifest",
        "requires_changelog",
        "requires_reference_baselines",
        "requires_scenario_reviews",
        "requires_report_validation_before_leaderboard",
    )
    for field in required_true:
        if governance.get(field) is not True:
            issues.append(DatasheetIssue(f"governance.{field}", "must be true"))


def _validate_release_validation(issues: list[DatasheetIssue], validation: Any) -> None:
    if not isinstance(validation, dict):
        issues.append(DatasheetIssue("release_validation", "must be an object"))
        return
    for field in ("audit_passed", "release_gates_passed", "release_gates"):
        if field not in validation:
            issues.append(DatasheetIssue(f"release_validation.{field}", "missing required field"))
    for field in ("audit_passed", "release_gates_passed"):
        if validation.get(field) is not True:
            issues.append(DatasheetIssue(f"release_validation.{field}", "must be true"))
    if not isinstance(validation.get("release_gates"), dict):
        issues.append(DatasheetIssue("release_validation.release_gates", "must be an object"))


def _validate_non_empty_string_list(
    issues: list[DatasheetIssue],
    data: dict[str, Any],
    field: str,
) -> None:
    value = data.get(field)
    if not isinstance(value, list) or not value:
        issues.append(DatasheetIssue(field, "must be a non-empty list"))
    elif not all(isinstance(item, str) and item.strip() for item in value):
        issues.append(DatasheetIssue(field, "must contain only non-empty strings"))


def _validate_file_entry(
    issues: list[DatasheetIssue],
    entry: Any,
    path: str,
) -> None:
    if not isinstance(entry, dict):
        issues.append(DatasheetIssue(path, "must be an object"))
        return
    for field in ("path", "sha256", "bytes"):
        if field not in entry:
            issues.append(DatasheetIssue(f"{path}.{field}", "missing required field"))
    if "sha256" in entry and not _looks_sha256(entry.get("sha256")):
        issues.append(DatasheetIssue(f"{path}.sha256", "must be a SHA-256 hex digest"))
    if "bytes" in entry and (
        isinstance(entry.get("bytes"), bool)
        or not isinstance(entry.get("bytes"), int)
        or entry["bytes"] < 0
    ):
        issues.append(DatasheetIssue(f"{path}.bytes", "must be a nonnegative integer"))


def _file_entry(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _optional_file_entry(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return _file_entry(path)


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _scenario_ids(path: str | Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        scenario.get("id")
        for scenario in data.get("scenarios", [])
        if isinstance(scenario, dict) and isinstance(scenario.get("id"), str)
    }


def _looks_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )
