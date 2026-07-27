"""Scenario review manifests for OpenVoiceCS-Bench releases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_REVIEW_MANIFEST_PATH = Path("data/openvoicecs/scenario_reviews_v0.1.json")

SUPPORTED_REVIEW_STATUSES = {
    "approved",
    "changes_requested",
    "rejected",
    "draft",
}

DEFAULT_REQUIRED_CHECKS = (
    "realistic_customer_goal",
    "tool_contract_replayable",
    "oracle_expected_state_correct",
    "sop_policy_coverage",
    "privacy_coverage",
    "auth_integrity_coverage",
    "grounding_coverage",
    "forbidden_action_coverage",
    "provenance_reviewed",
    "contamination_reviewed",
)


@dataclass(frozen=True)
class ReviewIssue:
    """Structured scenario-review validation issue."""

    item_id: str
    path: str
    message: str


def load_review_manifest(
    path: str | Path = DEFAULT_REVIEW_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load a scenario-review manifest JSON file."""
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest if isinstance(manifest, dict) else {}


def validate_review_manifest_file(
    path: str | Path = DEFAULT_REVIEW_MANIFEST_PATH,
    *,
    scenario_ids: set[str] | None = None,
    benchmark_version: str | None = None,
) -> list[ReviewIssue]:
    """Validate a saved scenario-review manifest."""
    return validate_review_manifest(
        load_review_manifest(path),
        scenario_ids=scenario_ids,
        benchmark_version=benchmark_version,
    )


def validate_review_manifest(
    manifest: dict[str, Any],
    *,
    scenario_ids: set[str] | None = None,
    benchmark_version: str | None = None,
) -> list[ReviewIssue]:
    """Validate scenario-review coverage and approval metadata."""
    issues: list[ReviewIssue] = []
    if not isinstance(manifest, dict):
        return [ReviewIssue("<reviews>", "<root>", "must be an object")]
    for field in ("name", "version", "benchmark_version", "review_policy", "reviews"):
        if field not in manifest:
            issues.append(ReviewIssue("<reviews>", field, "missing required field"))
    if issues:
        return issues
    if manifest.get("name") != "OpenVoiceCS-Bench Scenario Reviews":
        issues.append(ReviewIssue("<reviews>", "name", "unsupported review manifest"))
    if benchmark_version is not None and manifest.get("benchmark_version") != benchmark_version:
        issues.append(
            ReviewIssue(
                "<reviews>",
                "benchmark_version",
                "must match scenario suite version",
            )
        )

    policy = _review_policy(manifest)
    required_checks = policy["required_checks"]
    reviews = manifest.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        issues.append(ReviewIssue("<reviews>", "reviews", "must be a non-empty list"))
        return issues

    seen: set[str] = set()
    reviewed_ids: set[str] = set()
    approved_ids: set[str] = set()
    for index, review in enumerate(reviews):
        scenario_id, approved = _validate_review_entry(
            issues,
            review,
            index=index,
            seen=seen,
            scenario_ids=scenario_ids,
            required_checks=required_checks,
            minimum_reviewers=policy["minimum_reviewers_per_scenario"],
        )
        if scenario_id is not None:
            reviewed_ids.add(scenario_id)
            if approved:
                approved_ids.add(scenario_id)

    if scenario_ids is not None:
        for scenario_id in sorted(scenario_ids - reviewed_ids):
            issues.append(ReviewIssue(scenario_id, "reviews", "missing scenario review"))
        for scenario_id in sorted(scenario_ids - approved_ids):
            issues.append(ReviewIssue(scenario_id, "reviews", "scenario not approved"))
    return issues


def scenario_review_stats(
    manifest: dict[str, Any] | None,
    *,
    scenario_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Summarize scenario-review coverage for release metadata."""
    if not isinstance(manifest, dict):
        return {"present": False, "num_reviews": 0}
    reviews = manifest.get("reviews", [])
    reviews = reviews if isinstance(reviews, list) else []
    policy = _review_policy(manifest)
    reviewed_ids = {
        review.get("scenario_id")
        for review in reviews
        if isinstance(review, dict) and isinstance(review.get("scenario_id"), str)
    }
    approved_ids = {
        review.get("scenario_id")
        for review in reviews
        if (
            isinstance(review, dict)
            and isinstance(review.get("scenario_id"), str)
            and review.get("status") == "approved"
            and _checks_pass(review.get("checks"), policy["required_checks"])
            and _reviewer_count(review.get("reviewers"))
            >= policy["minimum_reviewers_per_scenario"]
        )
    }
    total = len(scenario_ids or set())
    check_pass_counts = {
        check: sum(
            1
            for review in reviews
            if isinstance(review, dict)
            and isinstance(review.get("checks"), dict)
            and review["checks"].get(check) is True
        )
        for check in policy["required_checks"]
    }
    return {
        "present": True,
        "version": manifest.get("version"),
        "benchmark_version": manifest.get("benchmark_version"),
        "minimum_reviewers_per_scenario": policy["minimum_reviewers_per_scenario"],
        "required_checks": list(policy["required_checks"]),
        "num_reviews": len(reviews),
        "num_reviewed_scenarios": len(reviewed_ids),
        "num_approved_scenarios": len(approved_ids),
        "scenario_review_coverage": _coverage_rate(reviewed_ids, scenario_ids),
        "scenario_approval_coverage": _coverage_rate(approved_ids, scenario_ids),
        "unreviewed_scenario_ids": sorted((scenario_ids or set()) - reviewed_ids),
        "unapproved_scenario_ids": sorted((scenario_ids or set()) - approved_ids),
        "check_pass_counts": check_pass_counts,
        "check_pass_rates": {
            check: (round(count / total, 6) if total else None)
            for check, count in check_pass_counts.items()
        },
    }


def _validate_review_entry(
    issues: list[ReviewIssue],
    review: Any,
    *,
    index: int,
    seen: set[str],
    scenario_ids: set[str] | None,
    required_checks: tuple[str, ...],
    minimum_reviewers: int,
) -> tuple[str | None, bool]:
    path = f"reviews[{index}]"
    if not isinstance(review, dict):
        issues.append(ReviewIssue(path, path, "must be an object"))
        return None, False

    scenario_id = review.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        issues.append(ReviewIssue(path, f"{path}.scenario_id", "must be a non-empty string"))
        return None, False
    if scenario_id in seen:
        issues.append(ReviewIssue(scenario_id, f"{path}.scenario_id", "duplicate scenario id"))
    seen.add(scenario_id)
    if scenario_ids is not None and scenario_id not in scenario_ids:
        issues.append(ReviewIssue(scenario_id, f"{path}.scenario_id", "unknown scenario id"))

    status = review.get("status")
    if status not in SUPPORTED_REVIEW_STATUSES:
        issues.append(ReviewIssue(scenario_id, f"{path}.status", "must be a supported status"))
    reviewers = review.get("reviewers")
    reviewer_count = _reviewer_count(reviewers)
    if reviewer_count < minimum_reviewers:
        issues.append(
            ReviewIssue(
                scenario_id,
                f"{path}.reviewers",
                f"must include at least {minimum_reviewers} reviewers",
            )
        )
    if not isinstance(review.get("reviewed_at"), str) or not review["reviewed_at"].strip():
        issues.append(ReviewIssue(scenario_id, f"{path}.reviewed_at", "must be a non-empty string"))

    checks = review.get("checks")
    if not isinstance(checks, dict):
        issues.append(ReviewIssue(scenario_id, f"{path}.checks", "must be an object"))
        checks = {}
    for check in required_checks:
        if checks.get(check) is not True:
            issues.append(ReviewIssue(scenario_id, f"{path}.checks.{check}", "must be true"))
    notes = review.get("notes", [])
    if notes is not None and (
        not isinstance(notes, list)
        or any(not isinstance(note, str) or not note.strip() for note in notes)
    ):
        issues.append(ReviewIssue(scenario_id, f"{path}.notes", "must be a list of strings"))

    approved = (
        status == "approved"
        and reviewer_count >= minimum_reviewers
        and _checks_pass(checks, required_checks)
    )
    return scenario_id, approved


def _review_policy(manifest: dict[str, Any]) -> dict[str, Any]:
    policy = manifest.get("review_policy")
    if not isinstance(policy, dict):
        policy = {}
    minimum = policy.get("minimum_reviewers_per_scenario", 2)
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        minimum = 2
    raw_checks = policy.get("required_checks")
    if not isinstance(raw_checks, list) or not raw_checks:
        checks = DEFAULT_REQUIRED_CHECKS
    else:
        checks = tuple(
            check
            for check in raw_checks
            if isinstance(check, str) and check.strip()
        ) or DEFAULT_REQUIRED_CHECKS
    return {
        "minimum_reviewers_per_scenario": minimum,
        "required_checks": checks,
    }


def _checks_pass(checks: Any, required_checks: tuple[str, ...]) -> bool:
    return isinstance(checks, dict) and all(checks.get(check) is True for check in required_checks)


def _reviewer_count(reviewers: Any) -> int:
    if not isinstance(reviewers, list):
        return 0
    return len({
        reviewer
        for reviewer in reviewers
        if isinstance(reviewer, str) and reviewer.strip()
    })


def _coverage_rate(observed_ids: set[str], expected_ids: set[str] | None) -> float | None:
    if expected_ids is None:
        return None
    if not expected_ids:
        return 1.0
    return round(len(observed_ids & expected_ids) / len(expected_ids), 6)
