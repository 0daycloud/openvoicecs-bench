"""Split manifest validation for OpenVoiceCS benchmark releases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SPLIT_MANIFEST_PATH = Path("data/openvoicecs/splits_v0.1.json")
DEFAULT_SPLIT_COMMITMENT_PATH = Path("data/openvoicecs/split_commitments_v0.1.json")
REQUIRED_SPLITS = {"public_dev", "sealed_test"}


@dataclass(frozen=True)
class SplitIssue:
    """Structured split-manifest validation issue."""

    item_id: str
    path: str
    message: str


def build_split_commitments_file(
    *,
    scenario_path: str | Path,
    split_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    audio_manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
    include_public_ids: bool = True,
    include_sealed_ids: bool = False,
) -> dict[str, Any]:
    """Build split commitments from release scenario/audio/split files."""
    suite = _load_json(scenario_path)
    split_manifest = load_split_manifest(split_path)
    audio_manifest = _load_json(audio_manifest_path) if audio_manifest_path else None
    commitments = build_split_commitments(
        suite=suite,
        split_manifest=split_manifest,
        audio_manifest=audio_manifest,
        include_public_ids=include_public_ids,
        include_sealed_ids=include_sealed_ids,
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(commitments, indent=2) + "\n", encoding="utf-8")
    return commitments


def build_split_commitments(
    *,
    suite: dict[str, Any],
    split_manifest: dict[str, Any],
    audio_manifest: dict[str, Any] | None = None,
    include_public_ids: bool = True,
    include_sealed_ids: bool = False,
) -> dict[str, Any]:
    """Build deterministic commitments for public and sealed benchmark splits."""
    scenarios = {
        scenario["id"]: scenario
        for scenario in suite.get("scenarios", [])
        if isinstance(scenario, dict) and isinstance(scenario.get("id"), str)
    }
    audio_variants = {
        variant["id"]: variant
        for variant in (audio_manifest or {}).get("variants", [])
        if isinstance(variant, dict) and isinstance(variant.get("id"), str)
    }
    splits = split_manifest.get("splits", {}) if isinstance(split_manifest, dict) else {}
    split_commitments = {}
    for split_name, split in sorted(splits.items()):
        split = split if isinstance(split, dict) else {}
        reveal_ids = include_sealed_ids if split_name == "sealed_test" else include_public_ids
        scenario_commitments = [
            _commitment_entry(
                item_id,
                scenarios[item_id],
                kind="scenario",
                reveal_id=reveal_ids,
            )
            for item_id in split.get("scenario_ids", [])
            if item_id in scenarios
        ]
        audio_commitments = [
            _commitment_entry(
                item_id,
                audio_variants[item_id],
                kind="audio_variant",
                reveal_id=reveal_ids,
            )
            for item_id in split.get("audio_variant_ids", [])
            if item_id in audio_variants
        ]
        split_commitments[str(split_name)] = {
            "num_scenarios": len(scenario_commitments),
            "num_audio_variants": len(audio_commitments),
            "scenario_commitments": scenario_commitments,
            "audio_variant_commitments": audio_commitments,
        }

    root_payload = {
        "scenario_suite_version": suite.get("version"),
        "split_manifest_version": split_manifest.get("version"),
        "audio_manifest_version": (audio_manifest or {}).get("version"),
        "splits": split_commitments,
    }
    return {
        "name": "OpenVoiceCS Split Commitments",
        "version": "0.1.0",
        "hash_algorithm": "sha256",
        "canonicalization": "json.dumps(sort_keys=True,separators=(',',':'))",
        "scenario_suite_version": suite.get("version"),
        "split_manifest_version": split_manifest.get("version"),
        "audio_manifest_version": (audio_manifest or {}).get("version"),
        "privacy": {
            "public_dev_ids_revealed": include_public_ids,
            "sealed_test_ids_revealed": include_sealed_ids,
        },
        "splits": split_commitments,
        "root_hash": _sha256_json(root_payload),
    }


def validate_split_commitments_file(
    commitment_path: str | Path = DEFAULT_SPLIT_COMMITMENT_PATH,
    *,
    scenario_path: str | Path,
    split_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    audio_manifest_path: str | Path | None = None,
) -> list[SplitIssue]:
    """Validate saved split commitments against release files."""
    commitments = _load_json(commitment_path)
    suite = _load_json(scenario_path)
    split_manifest = load_split_manifest(split_path)
    audio_manifest = _load_json(audio_manifest_path) if audio_manifest_path else None
    return validate_split_commitments(
        commitments,
        suite=suite,
        split_manifest=split_manifest,
        audio_manifest=audio_manifest,
    )


def validate_split_commitments(
    commitments: dict[str, Any],
    *,
    suite: dict[str, Any],
    split_manifest: dict[str, Any],
    audio_manifest: dict[str, Any] | None = None,
) -> list[SplitIssue]:
    """Validate split commitment hashes and counts."""
    issues: list[SplitIssue] = []
    if not isinstance(commitments, dict):
        return [SplitIssue("<split_commitments>", "<root>", "must be an object")]
    for field in ("name", "version", "hash_algorithm", "splits", "root_hash"):
        if field not in commitments:
            issues.append(SplitIssue("<split_commitments>", field, "missing required field"))
    if issues:
        return issues
    if commitments.get("hash_algorithm") != "sha256":
        issues.append(SplitIssue("<split_commitments>", "hash_algorithm", "must be sha256"))
    privacy = commitments.get("privacy", {})
    expected = build_split_commitments(
        suite=suite,
        split_manifest=split_manifest,
        audio_manifest=audio_manifest,
        include_public_ids=bool(privacy.get("public_dev_ids_revealed", True)),
        include_sealed_ids=bool(privacy.get("sealed_test_ids_revealed", False)),
    )
    _compare_commitment_field(issues, commitments, expected, "root_hash")
    for split_name, expected_split in expected["splits"].items():
        actual_split = commitments.get("splits", {}).get(split_name)
        if not isinstance(actual_split, dict):
            issues.append(SplitIssue(split_name, f"splits.{split_name}", "missing split"))
            continue
        for field in ("num_scenarios", "num_audio_variants"):
            _compare_commitment_field(
                issues,
                actual_split,
                expected_split,
                field,
                path=f"splits.{split_name}.{field}",
                item_id=split_name,
            )
        for field in ("scenario_commitments", "audio_variant_commitments"):
            if actual_split.get(field) != expected_split.get(field):
                issues.append(
                    SplitIssue(
                        split_name,
                        f"splits.{split_name}.{field}",
                        "commitments do not match release files",
                    )
                )
    return issues


def load_split_manifest(path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH) -> dict[str, Any]:
    """Load a split manifest JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_split_manifest_file(
    path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    *,
    scenario_ids: set[str] | None = None,
    audio_variant_ids: set[str] | None = None,
) -> list[SplitIssue]:
    """Validate a saved split manifest JSON file."""
    manifest = load_split_manifest(path)
    return validate_split_manifest(
        manifest,
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_variant_ids,
    )


def validate_split_manifest(
    manifest: dict[str, Any],
    *,
    scenario_ids: set[str] | None = None,
    audio_variant_ids: set[str] | None = None,
) -> list[SplitIssue]:
    """Return all split manifest contract issues."""
    issues: list[SplitIssue] = []
    if not isinstance(manifest, dict):
        return [SplitIssue("<split_manifest>", "<root>", "must be an object")]
    for field in ("name", "version", "splits"):
        if field not in manifest:
            issues.append(SplitIssue("<split_manifest>", field, "missing required field"))
    if issues:
        return issues

    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        return [SplitIssue("<split_manifest>", "splits", "must be an object")]
    missing_splits = REQUIRED_SPLITS - set(splits)
    for split_name in sorted(missing_splits):
        issues.append(SplitIssue(split_name, f"splits.{split_name}", "missing required split"))

    seen_scenarios: dict[str, str] = {}
    seen_audio: dict[str, str] = {}
    for split_name, split in splits.items():
        path = f"splits.{split_name}"
        if not isinstance(split, dict):
            issues.append(SplitIssue(str(split_name), path, "must be an object"))
            continue
        scenario_list = split.get("scenario_ids", [])
        audio_list = split.get("audio_variant_ids", [])
        if not isinstance(scenario_list, list):
            issues.append(
                SplitIssue(str(split_name), f"{path}.scenario_ids", "must be a list")
            )
            scenario_list = []
        if not isinstance(audio_list, list):
            issues.append(
                SplitIssue(str(split_name), f"{path}.audio_variant_ids", "must be a list")
            )
            audio_list = []
        _validate_id_list(
            issues,
            scenario_list,
            split_name=str(split_name),
            field_path=f"{path}.scenario_ids",
            known_ids=scenario_ids,
            seen=seen_scenarios,
            id_type="scenario",
        )
        _validate_id_list(
            issues,
            audio_list,
            split_name=str(split_name),
            field_path=f"{path}.audio_variant_ids",
            known_ids=audio_variant_ids,
            seen=seen_audio,
            id_type="audio variant",
        )
    return issues


def split_manifest_stats(
    manifest: dict[str, Any] | None,
    *,
    scenario_ids: set[str] | None = None,
    audio_variant_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Summarize split coverage for release audits."""
    if not isinstance(manifest, dict):
        return {
            "present": False,
            "splits": {},
            "scenario_coverage": 0.0,
            "audio_variant_coverage": 0.0,
        }
    splits = manifest.get("splits", {})
    stats = {}
    split_scenario_ids = set()
    split_audio_ids = set()
    if isinstance(splits, dict):
        for split_name, split in splits.items():
            if not isinstance(split, dict):
                continue
            scenarios = set(split.get("scenario_ids", []))
            audio_variants = set(split.get("audio_variant_ids", []))
            split_scenario_ids.update(scenarios)
            split_audio_ids.update(audio_variants)
            stats[str(split_name)] = {
                "num_scenarios": len(scenarios),
                "num_audio_variants": len(audio_variants),
            }
    scenario_total = len(scenario_ids or set())
    audio_total = len(audio_variant_ids or set())
    return {
        "present": True,
        "version": manifest.get("version"),
        "splits": dict(sorted(stats.items())),
        "scenario_coverage": round(
            len(split_scenario_ids & (scenario_ids or set())) / scenario_total,
            4,
        ) if scenario_total else None,
        "audio_variant_coverage": round(
            len(split_audio_ids & (audio_variant_ids or set())) / audio_total,
            4,
        ) if audio_total else None,
        "unassigned_scenario_ids": sorted((scenario_ids or set()) - split_scenario_ids),
        "unassigned_audio_variant_ids": sorted((audio_variant_ids or set()) - split_audio_ids),
    }


def _validate_id_list(
    issues: list[SplitIssue],
    ids: list[Any],
    *,
    split_name: str,
    field_path: str,
    known_ids: set[str] | None,
    seen: dict[str, str],
    id_type: str,
) -> None:
    local_seen = set()
    for index, raw_id in enumerate(ids):
        path = f"{field_path}[{index}]"
        if not isinstance(raw_id, str) or not raw_id:
            issues.append(SplitIssue(str(raw_id), path, "must be a non-empty string"))
            continue
        if raw_id in local_seen:
            issues.append(SplitIssue(raw_id, path, f"duplicate {id_type} id in split"))
        local_seen.add(raw_id)
        if known_ids is not None and raw_id not in known_ids:
            issues.append(SplitIssue(raw_id, path, f"unknown {id_type} id"))
        if raw_id in seen and seen[raw_id] != split_name:
            issues.append(
                SplitIssue(
                    raw_id,
                    path,
                    f"{id_type} id also assigned to split {seen[raw_id]}",
                )
            )
        seen[raw_id] = split_name


def _commitment_entry(
    item_id: str,
    payload: dict[str, Any],
    *,
    kind: str,
    reveal_id: bool,
) -> dict[str, Any]:
    entry = {
        "kind": kind,
        "sha256": _sha256_json({
            "kind": kind,
            "id": item_id,
            "payload": payload,
        }),
    }
    if reveal_id:
        entry["id"] = item_id
    return entry


def _compare_commitment_field(
    issues: list[SplitIssue],
    actual: dict[str, Any],
    expected: dict[str, Any],
    field: str,
    *,
    path: str | None = None,
    item_id: str = "<split_commitments>",
) -> None:
    if actual.get(field) != expected.get(field):
        issues.append(
            SplitIssue(
                item_id,
                path or field,
                "does not match release files",
            )
        )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data
