"""Provenance and contamination controls for OpenVoiceCS benchmark releases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROVENANCE_MANIFEST_PATH = Path("data/openvoicecs/provenance_v0.1.json")
OPEN_LICENSES = {
    "CC-BY-4.0",
    "CC0-1.0",
    "MIT",
    "Apache-2.0",
    "Synthetic-Consent",
}
CONSENTED_AUDIO_SOURCES = {"synthetic", "consented_human"}


@dataclass(frozen=True)
class ProvenanceIssue:
    """Structured provenance-manifest validation issue."""

    item_id: str
    path: str
    message: str


def load_provenance_manifest(
    path: str | Path = DEFAULT_PROVENANCE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load a provenance manifest JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_provenance_manifest_file(
    path: str | Path = DEFAULT_PROVENANCE_MANIFEST_PATH,
    *,
    scenario_ids: set[str] | None = None,
    audio_variant_ids: set[str] | None = None,
) -> list[ProvenanceIssue]:
    """Validate a saved provenance manifest JSON file."""
    manifest = load_provenance_manifest(path)
    return validate_provenance_manifest(
        manifest,
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_variant_ids,
    )


def validate_provenance_manifest(
    manifest: dict[str, Any],
    *,
    scenario_ids: set[str] | None = None,
    audio_variant_ids: set[str] | None = None,
) -> list[ProvenanceIssue]:
    """Return all provenance-manifest contract issues."""
    issues: list[ProvenanceIssue] = []
    if not isinstance(manifest, dict):
        return [ProvenanceIssue("<provenance>", "<root>", "must be an object")]
    for field in ("name", "version", "scenarios", "audio_variants"):
        if field not in manifest:
            issues.append(ProvenanceIssue("<provenance>", field, "missing required field"))
    if issues:
        return issues

    scenarios = manifest.get("scenarios")
    audio_variants = manifest.get("audio_variants")
    if not isinstance(scenarios, dict):
        issues.append(ProvenanceIssue("<provenance>", "scenarios", "must be an object"))
        scenarios = {}
    if not isinstance(audio_variants, dict):
        issues.append(
            ProvenanceIssue("<provenance>", "audio_variants", "must be an object")
        )
        audio_variants = {}

    _validate_item_map(
        issues,
        scenarios,
        known_ids=scenario_ids,
        item_type="scenario",
        base_path="scenarios",
        required_fields=(
            "source_type",
            "license",
            "authoring_method",
            "contains_real_customer_data",
            "contamination_risk",
        ),
    )
    _validate_item_map(
        issues,
        audio_variants,
        known_ids=audio_variant_ids,
        item_type="audio variant",
        base_path="audio_variants",
        required_fields=(
            "source_type",
            "license",
            "speaker_consent",
            "voice_rights",
            "contains_real_customer_data",
            "contamination_risk",
        ),
    )

    if scenario_ids is not None:
        missing = scenario_ids - set(scenarios)
        for scenario_id in sorted(missing):
            issues.append(
                ProvenanceIssue(scenario_id, "scenarios", "missing scenario provenance")
            )
    if audio_variant_ids is not None:
        missing = audio_variant_ids - set(audio_variants)
        for variant_id in sorted(missing):
            issues.append(
                ProvenanceIssue(
                    variant_id,
                    "audio_variants",
                    "missing audio variant provenance",
                )
            )
    return issues


def provenance_stats(
    manifest: dict[str, Any] | None,
    *,
    scenario_ids: set[str] | None = None,
    audio_variant_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Summarize provenance, licensing, consent, and contamination coverage."""
    if not isinstance(manifest, dict):
        return {
            "present": False,
            "scenario_coverage": 0.0,
            "audio_variant_coverage": 0.0,
        }

    scenarios = manifest.get("scenarios", {})
    audio_variants = manifest.get("audio_variants", {})
    if not isinstance(scenarios, dict):
        scenarios = {}
    if not isinstance(audio_variants, dict):
        audio_variants = {}

    return {
        "present": True,
        "version": manifest.get("version"),
        "scenario_coverage": _coverage_rate(scenarios, scenario_ids),
        "audio_variant_coverage": _coverage_rate(audio_variants, audio_variant_ids),
        "scenario_license_open_rate": _item_rate(
            scenarios,
            lambda item: item.get("license") in OPEN_LICENSES,
        ),
        "audio_license_open_rate": _item_rate(
            audio_variants,
            lambda item: item.get("license") in OPEN_LICENSES,
        ),
        "audio_speaker_consent_rate": _item_rate(
            audio_variants,
            lambda item: item.get("speaker_consent") in {"documented", "synthetic"},
        ),
        "no_real_customer_data_rate": _item_rate(
            {**scenarios, **audio_variants},
            lambda item: item.get("contains_real_customer_data") is False,
        ),
        "low_contamination_risk_rate": _item_rate(
            {**scenarios, **audio_variants},
            lambda item: item.get("contamination_risk") in {"none", "low"},
        ),
        "source_types": _count_field({**scenarios, **audio_variants}, "source_type"),
    }


def _validate_item_map(
    issues: list[ProvenanceIssue],
    items: dict[str, Any],
    *,
    known_ids: set[str] | None,
    item_type: str,
    base_path: str,
    required_fields: tuple[str, ...],
) -> None:
    for item_id, item in items.items():
        path = f"{base_path}.{item_id}"
        if known_ids is not None and item_id not in known_ids:
            issues.append(ProvenanceIssue(item_id, path, f"unknown {item_type} id"))
        if not isinstance(item, dict):
            issues.append(ProvenanceIssue(item_id, path, "must be an object"))
            continue
        for field in required_fields:
            if field not in item:
                issues.append(ProvenanceIssue(item_id, f"{path}.{field}", "missing required field"))
        if item.get("license") is not None and not isinstance(item["license"], str):
            issues.append(ProvenanceIssue(item_id, f"{path}.license", "must be a string"))
        if item.get("contains_real_customer_data") not in {True, False, None}:
            issues.append(
                ProvenanceIssue(
                    item_id,
                    f"{path}.contains_real_customer_data",
                    "must be boolean",
                )
            )
        if base_path == "audio_variants":
            if item.get("speaker_consent") not in {"documented", "synthetic", "missing", None}:
                issues.append(
                    ProvenanceIssue(
                        item_id,
                        f"{path}.speaker_consent",
                        "must be documented, synthetic, or missing",
                    )
                )
            if item.get("source_type") not in CONSENTED_AUDIO_SOURCES:
                issues.append(
                    ProvenanceIssue(
                        item_id,
                        f"{path}.source_type",
                        "must be synthetic or consented_human",
                    )
                )


def _coverage_rate(items: dict[str, Any], known_ids: set[str] | None) -> float:
    if known_ids is None:
        return 1.0 if items else 0.0
    if not known_ids:
        return 1.0
    return round(len(set(items) & known_ids) / len(known_ids), 4)


def _item_rate(items: dict[str, Any], predicate) -> float:
    valid_items = [item for item in items.values() if isinstance(item, dict)]
    if not valid_items:
        return 0.0
    return round(sum(1 for item in valid_items if predicate(item)) / len(valid_items), 4)


def _count_field(items: dict[str, Any], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items.values():
        if not isinstance(item, dict):
            continue
        value = str(item.get(field, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
