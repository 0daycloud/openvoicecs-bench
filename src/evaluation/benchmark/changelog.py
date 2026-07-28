"""Release changelog and errata validation for OpenVoiceCS-Bench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.datapaths import data_path

DEFAULT_CHANGELOG_PATH = data_path("changelog_v0.1.json")
CHANGE_TYPES = {
    "release",
    "scenario_added",
    "scenario_modified",
    "scenario_deprecated",
    "audio_added",
    "audio_modified",
    "scoring_changed",
    "documentation_changed",
    "erratum",
}
COMPATIBILITY_CLASSES = {
    "initial_release",
    "backward_compatible",
    "scoring_affecting",
    "breaking",
    "deprecated",
    "documentation_only",
}
ERRATA_STATUS = {"open", "resolved", "wontfix"}
ERRATA_SEVERITY = {"low", "medium", "high", "critical"}


@dataclass(frozen=True)
class ChangelogIssue:
    """Structured release-changelog validation issue."""

    item_id: str
    path: str
    message: str


def load_changelog(path: str | Path = DEFAULT_CHANGELOG_PATH) -> dict[str, Any]:
    """Load a benchmark changelog JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_changelog_file(
    path: str | Path = DEFAULT_CHANGELOG_PATH,
    *,
    scenario_ids: set[str] | None = None,
    audio_variant_ids: set[str] | None = None,
    benchmark_version: str | None = None,
) -> list[ChangelogIssue]:
    """Validate a saved release changelog JSON file."""
    changelog = load_changelog(path)
    return validate_changelog(
        changelog,
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_variant_ids,
        benchmark_version=benchmark_version,
    )


def validate_changelog(
    changelog: dict[str, Any],
    *,
    scenario_ids: set[str] | None = None,
    audio_variant_ids: set[str] | None = None,
    benchmark_version: str | None = None,
) -> list[ChangelogIssue]:
    """Return all release-changelog contract issues."""
    issues: list[ChangelogIssue] = []
    if not isinstance(changelog, dict):
        return [ChangelogIssue("<changelog>", "<root>", "must be an object")]
    for field in ("name", "version", "benchmark_version", "entries", "errata"):
        if field not in changelog:
            issues.append(ChangelogIssue("<changelog>", field, "missing required field"))
    if issues:
        return issues

    if benchmark_version is not None and changelog.get("benchmark_version") != benchmark_version:
        issues.append(
            ChangelogIssue(
                "<changelog>",
                "benchmark_version",
                "must match scenario suite version",
            )
        )

    entries = changelog.get("entries")
    if not isinstance(entries, list) or not entries:
        issues.append(ChangelogIssue("<changelog>", "entries", "must be a non-empty list"))
        entries = []
    seen_entry_ids: set[str] = set()
    changed_scenarios: set[str] = set()
    changed_audio_variants: set[str] = set()
    for index, entry in enumerate(entries):
        _validate_entry(
            issues,
            entry,
            path=f"entries[{index}]",
            seen_ids=seen_entry_ids,
            scenario_ids=scenario_ids,
            audio_variant_ids=audio_variant_ids,
            changed_scenarios=changed_scenarios,
            changed_audio_variants=changed_audio_variants,
        )

    errata = changelog.get("errata")
    if not isinstance(errata, list):
        issues.append(ChangelogIssue("<changelog>", "errata", "must be a list"))
        errata = []
    seen_errata_ids: set[str] = set()
    for index, item in enumerate(errata):
        _validate_erratum(
            issues,
            item,
            path=f"errata[{index}]",
            seen_ids=seen_errata_ids,
            scenario_ids=scenario_ids,
            audio_variant_ids=audio_variant_ids,
        )
    return issues


def changelog_stats(
    changelog: dict[str, Any] | None,
    *,
    scenario_ids: set[str] | None = None,
    audio_variant_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Summarize release-changelog coverage and open errata."""
    if not isinstance(changelog, dict):
        return {
            "present": False,
            "num_entries": 0,
            "num_open_errata": 0,
            "scenario_change_coverage": 0.0,
            "audio_variant_change_coverage": 0.0,
        }
    entries = changelog.get("entries", [])
    errata = changelog.get("errata", [])
    entries = entries if isinstance(entries, list) else []
    errata = errata if isinstance(errata, list) else []
    changed_scenarios = _changed_ids(entries, "scenario_ids")
    changed_audio_variants = _changed_ids(entries, "audio_variant_ids")
    return {
        "present": True,
        "version": changelog.get("version"),
        "benchmark_version": changelog.get("benchmark_version"),
        "num_entries": len(entries),
        "entry_types": _count_field(entries, "type"),
        "compatibility": _count_field(entries, "compatibility"),
        "num_changed_scenarios": len(changed_scenarios),
        "num_changed_audio_variants": len(changed_audio_variants),
        "scenario_change_coverage": _coverage_rate(changed_scenarios, scenario_ids),
        "audio_variant_change_coverage": _coverage_rate(
            changed_audio_variants,
            audio_variant_ids,
        ),
        "num_errata": len(errata),
        "num_open_errata": sum(1 for item in errata if item.get("status") == "open"),
        "errata_severity": _count_field(errata, "severity"),
    }


def _validate_entry(
    issues: list[ChangelogIssue],
    entry: Any,
    *,
    path: str,
    seen_ids: set[str],
    scenario_ids: set[str] | None,
    audio_variant_ids: set[str] | None,
    changed_scenarios: set[str],
    changed_audio_variants: set[str],
) -> None:
    if not isinstance(entry, dict):
        issues.append(ChangelogIssue(path, path, "must be an object"))
        return
    item_id = _validate_identifier(issues, entry, path=path, seen_ids=seen_ids)
    _validate_non_empty_string(issues, item_id, entry.get("summary"), f"{path}.summary")
    _validate_non_empty_string(issues, item_id, entry.get("date"), f"{path}.date")
    if entry.get("type") not in CHANGE_TYPES:
        issues.append(ChangelogIssue(item_id, f"{path}.type", "must be a supported change type"))
    if entry.get("compatibility") not in COMPATIBILITY_CLASSES:
        issues.append(
            ChangelogIssue(
                item_id,
                f"{path}.compatibility",
                "must be a supported compatibility class",
            )
        )
    entry_scenarios = _validate_id_list(
        issues,
        item_id,
        entry.get("scenario_ids"),
        f"{path}.scenario_ids",
        known_ids=scenario_ids,
        item_label="scenario",
    )
    entry_audio = _validate_id_list(
        issues,
        item_id,
        entry.get("audio_variant_ids"),
        f"{path}.audio_variant_ids",
        known_ids=audio_variant_ids,
        item_label="audio variant",
    )
    changed_scenarios.update(entry_scenarios)
    changed_audio_variants.update(entry_audio)
    reviewers = entry.get("reviewed_by")
    if reviewers is not None:
        _validate_string_list(issues, item_id, reviewers, f"{path}.reviewed_by")


def _validate_erratum(
    issues: list[ChangelogIssue],
    item: Any,
    *,
    path: str,
    seen_ids: set[str],
    scenario_ids: set[str] | None,
    audio_variant_ids: set[str] | None,
) -> None:
    if not isinstance(item, dict):
        issues.append(ChangelogIssue(path, path, "must be an object"))
        return
    item_id = _validate_identifier(issues, item, path=path, seen_ids=seen_ids)
    _validate_non_empty_string(issues, item_id, item.get("summary"), f"{path}.summary")
    if item.get("status") not in ERRATA_STATUS:
        issues.append(
            ChangelogIssue(
                item_id,
                f"{path}.status",
                "must be open, resolved, or wontfix",
            )
        )
    if item.get("severity") not in ERRATA_SEVERITY:
        issues.append(
            ChangelogIssue(
                item_id,
                f"{path}.severity",
                "must be low, medium, high, or critical",
            )
        )
    _validate_id_list(
        issues,
        item_id,
        item.get("scenario_ids"),
        f"{path}.scenario_ids",
        known_ids=scenario_ids,
        item_label="scenario",
    )
    _validate_id_list(
        issues,
        item_id,
        item.get("audio_variant_ids"),
        f"{path}.audio_variant_ids",
        known_ids=audio_variant_ids,
        item_label="audio variant",
    )


def _validate_identifier(
    issues: list[ChangelogIssue],
    item: dict[str, Any],
    *,
    path: str,
    seen_ids: set[str],
) -> str:
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        issues.append(ChangelogIssue(path, f"{path}.id", "must be a non-empty string"))
        return path
    if item_id in seen_ids:
        issues.append(ChangelogIssue(item_id, f"{path}.id", "duplicate id"))
    seen_ids.add(item_id)
    return item_id


def _validate_non_empty_string(
    issues: list[ChangelogIssue],
    item_id: str,
    value: Any,
    path: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        issues.append(ChangelogIssue(item_id, path, "must be a non-empty string"))


def _validate_id_list(
    issues: list[ChangelogIssue],
    item_id: str,
    value: Any,
    path: str,
    *,
    known_ids: set[str] | None,
    item_label: str,
) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list):
        issues.append(ChangelogIssue(item_id, path, "must be a list"))
        return set()
    result: set[str] = set()
    for index, raw_id in enumerate(value):
        if not isinstance(raw_id, str) or not raw_id.strip():
            issues.append(ChangelogIssue(item_id, f"{path}[{index}]", "must be a non-empty string"))
            continue
        result.add(raw_id)
        if known_ids is not None and raw_id not in known_ids:
            issues.append(ChangelogIssue(raw_id, f"{path}[{index}]", f"unknown {item_label} id"))
    return result


def _validate_string_list(
    issues: list[ChangelogIssue],
    item_id: str,
    value: Any,
    path: str,
) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in value
    ):
        issues.append(ChangelogIssue(item_id, path, "must be a list of non-empty strings"))


def _changed_ids(entries: list[Any], field: str) -> set[str]:
    changed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get(field), list):
            continue
        changed.update(item for item in entry[field] if isinstance(item, str))
    return changed


def _coverage_rate(changed_ids: set[str], known_ids: set[str] | None) -> float:
    if known_ids is None:
        return 1.0 if changed_ids else 0.0
    if not known_ids:
        return 1.0
    return round(len(changed_ids & known_ids) / len(known_ids), 4)


def _count_field(items: list[Any], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        value = str(item.get(field, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
