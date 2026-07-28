"""Leaderboard claim package validation for OpenVoiceCS releases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.datapaths import data_path

DEFAULT_CLAIMS_MANIFEST_PATH = data_path("claims", "leaderboard_claims_v0.1.json")
CLAIM_TYPES = {"reference_sanity", "candidate_beats_baseline", "frontier_membership", "slice_regression"}
CLAIM_STATUSES = {"reference_fixture", "pending_review", "official", "rejected", "retired"}
PROTECTED_SLICE_FIELDS = ("privacy", "auth_integrity", "safety")


@dataclass(frozen=True)
class ClaimIssue:
    """Structured leaderboard claim validation issue."""

    item_id: str
    path: str
    message: str


def load_claims_manifest(path: str | Path = DEFAULT_CLAIMS_MANIFEST_PATH) -> dict[str, Any]:
    """Load and validate a leaderboard claims manifest."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    issues = validate_claims_manifest(manifest)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS claims validation failed:\n{formatted}")
    return manifest


def validate_claims_manifest_file(
    path: str | Path = DEFAULT_CLAIMS_MANIFEST_PATH,
    *,
    base_dir: str | Path = ".",
) -> list[ClaimIssue]:
    """Validate a saved leaderboard claims manifest."""
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    return validate_claims_manifest(manifest, base_dir=base_dir)


def validate_claims_manifest(
    manifest: dict[str, Any],
    *,
    base_dir: str | Path = ".",
) -> list[ClaimIssue]:
    """Return all leaderboard claim manifest contract issues."""
    issues: list[ClaimIssue] = []
    if not isinstance(manifest, dict):
        return [ClaimIssue("<claims>", "<root>", "must be an object")]
    for field in ("name", "version", "benchmark_version", "claim_policy", "claims"):
        if field not in manifest:
            issues.append(ClaimIssue("<claims>", field, "missing required field"))
    if issues:
        return issues
    if manifest.get("name") != "OpenVoiceCS Leaderboard Claims":
        issues.append(ClaimIssue("<claims>", "name", "must be OpenVoiceCS Leaderboard Claims"))
    for field in ("version", "benchmark_version"):
        if not _non_empty_string(manifest.get(field)):
            issues.append(ClaimIssue("<claims>", field, "must be a non-empty string"))
    _validate_claim_policy(issues, manifest.get("claim_policy"))

    claims = manifest.get("claims")
    if not isinstance(claims, list):
        issues.append(ClaimIssue("<claims>", "claims", "must be a list"))
        return issues
    seen_ids: set[str] = set()
    for index, claim in enumerate(claims):
        _validate_claim_entry(
            issues,
            claim,
            index=index,
            seen_ids=seen_ids,
            base_dir=Path(base_dir),
        )
    return issues


def claims_stats(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize leaderboard claims evidence."""
    if not isinstance(manifest, dict):
        return {"present": False, "num_claims": 0}
    claims = manifest.get("claims", [])
    claims = claims if isinstance(claims, list) else []
    by_status: dict[str, int] = {}
    for claim in claims:
        if isinstance(claim, dict):
            status = str(claim.get("status") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
    return {
        "present": True,
        "version": manifest.get("version"),
        "benchmark_version": manifest.get("benchmark_version"),
        "num_claims": len(claims),
        "by_status": by_status,
        "official_claims": by_status.get("official", 0),
        "reference_fixture_claims": by_status.get("reference_fixture", 0),
    }


def _validate_claim_policy(issues: list[ClaimIssue], policy: Any) -> None:
    if not isinstance(policy, dict):
        issues.append(ClaimIssue("<claims>", "claim_policy", "must be an object"))
        return
    for field in (
        "official_requires_external_system_registry",
        "official_requires_release_bundle",
        "official_requires_pairwise_comparison",
        "official_requires_ci_excluding_zero",
        "official_requires_no_protected_slice_regressions",
        "official_requires_human_or_audited_judging",
    ):
        if policy.get(field) is not True:
            issues.append(ClaimIssue("<claims>", f"claim_policy.{field}", "must be true"))
    alpha = policy.get("maximum_mcnemar_p_value")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha <= 1:
        issues.append(ClaimIssue("<claims>", "claim_policy.maximum_mcnemar_p_value", "must be in (0, 1]"))


def _validate_claim_entry(
    issues: list[ClaimIssue],
    claim: Any,
    *,
    index: int,
    seen_ids: set[str],
    base_dir: Path,
) -> None:
    path = f"claims[{index}]"
    if not isinstance(claim, dict):
        issues.append(ClaimIssue(path, path, "must be an object"))
        return
    claim_id = claim.get("id")
    if not _non_empty_string(claim_id):
        issues.append(ClaimIssue(path, f"{path}.id", "must be a non-empty string"))
        claim_id = path
    elif claim_id in seen_ids:
        issues.append(ClaimIssue(str(claim_id), f"{path}.id", "duplicate claim id"))
    seen_ids.add(str(claim_id))

    if claim.get("claim_type") not in CLAIM_TYPES:
        issues.append(
            ClaimIssue(
                str(claim_id),
                f"{path}.claim_type",
                f"must be one of: {', '.join(sorted(CLAIM_TYPES))}",
            )
        )
    status = claim.get("status")
    if status not in CLAIM_STATUSES:
        issues.append(
            ClaimIssue(
                str(claim_id),
                f"{path}.status",
                f"must be one of: {', '.join(sorted(CLAIM_STATUSES))}",
            )
        )
    if not isinstance(claim.get("official_claim"), bool):
        issues.append(ClaimIssue(str(claim_id), f"{path}.official_claim", "must be boolean"))
    if status == "reference_fixture" and claim.get("official_claim") is not False:
        issues.append(ClaimIssue(str(claim_id), f"{path}.official_claim", "reference fixtures cannot be official claims"))

    comparison = _load_file_entry(
        issues,
        claim.get("comparison"),
        f"{path}.comparison",
        str(claim_id),
        base_dir,
    )
    for field in ("baseline_report", "candidate_report", "release_bundle", "external_systems_registry"):
        if claim.get(field) is not None:
            _load_file_entry(issues, claim[field], f"{path}.{field}", str(claim_id), base_dir, parse_json=False)
    if comparison is not None:
        _validate_comparison_evidence(issues, comparison, path, str(claim_id))
    if status == "official":
        _validate_official_claim(issues, claim, comparison, path, str(claim_id))


def _validate_official_claim(
    issues: list[ClaimIssue],
    claim: dict[str, Any],
    comparison: dict[str, Any] | None,
    path: str,
    claim_id: str,
) -> None:
    if claim.get("official_claim") is not True:
        issues.append(ClaimIssue(claim_id, f"{path}.official_claim", "official status requires official_claim true"))
    for field in ("comparison", "baseline_report", "candidate_report", "release_bundle", "external_systems_registry"):
        if not isinstance(claim.get(field), dict):
            issues.append(ClaimIssue(claim_id, f"{path}.{field}", "official claims must include this file entry"))
    judging = claim.get("judging_evidence")
    if not isinstance(judging, dict):
        issues.append(ClaimIssue(claim_id, f"{path}.judging_evidence", "official claims must include judging evidence"))
    elif judging.get("annotation_mode") == "reference_fixture":
        issues.append(ClaimIssue(claim_id, f"{path}.judging_evidence.annotation_mode", "official claims cannot use reference_fixture judging"))
    if comparison is None:
        return
    summary = comparison.get("summary", {})
    if summary.get("interpretation") != "candidate_higher_with_ci_excluding_zero":
        issues.append(ClaimIssue(claim_id, f"{path}.comparison.summary.interpretation", "official improvement claims require CI excluding zero"))
    p_value = summary.get("mcnemar_exact", {}).get("p_value")
    if isinstance(p_value, bool) or not isinstance(p_value, (int, float)) or p_value > 0.05:
        issues.append(ClaimIssue(claim_id, f"{path}.comparison.summary.mcnemar_exact.p_value", "must be <= 0.05"))
    for metric in PROTECTED_SLICE_FIELDS:
        delta = comparison.get("metric_deltas", {}).get(metric, {})
        if isinstance(delta, dict) and delta.get("ci_low", 0) < 0:
            issues.append(ClaimIssue(claim_id, f"{path}.comparison.metric_deltas.{metric}", "protected metric CI must not regress"))


def _validate_comparison_evidence(
    issues: list[ClaimIssue],
    comparison: dict[str, Any],
    path: str,
    claim_id: str,
) -> None:
    if comparison.get("benchmark") != "OpenVoiceCS-Bench Pairwise Comparison":
        issues.append(ClaimIssue(claim_id, f"{path}.comparison.benchmark", "must be OpenVoiceCS-Bench Pairwise Comparison"))
    matched = comparison.get("matched_scenarios", {})
    if not isinstance(matched, dict) or matched.get("count", 0) <= 0:
        issues.append(ClaimIssue(claim_id, f"{path}.comparison.matched_scenarios.count", "must be positive"))
    summary = comparison.get("summary")
    if not isinstance(summary, dict):
        issues.append(ClaimIssue(claim_id, f"{path}.comparison.summary", "must be an object"))
        return
    delta = summary.get("mean_paired_scenario_score_delta")
    if not isinstance(delta, dict):
        issues.append(ClaimIssue(claim_id, f"{path}.comparison.summary.mean_paired_scenario_score_delta", "must be an object"))
    mcnemar = summary.get("mcnemar_exact")
    if not isinstance(mcnemar, dict):
        issues.append(ClaimIssue(claim_id, f"{path}.comparison.summary.mcnemar_exact", "must be an object"))


def _load_file_entry(
    issues: list[ClaimIssue],
    entry: Any,
    path: str,
    item_id: str,
    base_dir: Path,
    *,
    parse_json: bool = True,
) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        issues.append(ClaimIssue(item_id, path, "must be an object"))
        return None
    for field in ("path", "sha256", "bytes"):
        if field not in entry:
            issues.append(ClaimIssue(item_id, f"{path}.{field}", "missing required field"))
    raw_path = entry.get("path")
    if not _non_empty_string(raw_path):
        issues.append(ClaimIssue(item_id, f"{path}.path", "must be a non-empty string"))
        return None
    resolved = Path(str(raw_path))
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    if not resolved.exists():
        issues.append(ClaimIssue(item_id, f"{path}.path", "file does not exist"))
        return None
    data = resolved.read_bytes()
    if entry.get("sha256") != hashlib.sha256(data).hexdigest():
        issues.append(ClaimIssue(item_id, f"{path}.sha256", "does not match file contents"))
    if entry.get("bytes") != len(data):
        issues.append(ClaimIssue(item_id, f"{path}.bytes", "does not match file size"))
    if not parse_json:
        return None
    try:
        loaded = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(ClaimIssue(item_id, path, f"invalid JSON: {exc.msg}"))
        return None
    return loaded if isinstance(loaded, dict) else None


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
