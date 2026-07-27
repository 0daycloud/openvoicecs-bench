"""Authoring utilities for adding OpenVoiceCS scenarios safely."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.coverage import (
    DEFAULT_COVERAGE_TARGET_PATH,
    build_coverage_plan,
)
from src.evaluation.benchmark.openvoicecs import (
    DEFAULT_SCENARIO_PATH,
    validate_audio_manifest_file,
    validate_scenarios,
)
from src.evaluation.benchmark.provenance import validate_provenance_manifest
from src.evaluation.benchmark.splits import validate_split_manifest


@dataclass(frozen=True)
class AuthoringIssue:
    """Structured scenario-authoring issue."""

    item_id: str
    path: str
    message: str


def add_scenarios_to_release_files(
    *,
    draft_path: str | Path,
    scenario_path: str | Path,
    split_path: str | Path,
    provenance_path: str | Path,
    output_scenario_path: str | Path,
    output_split_path: str | Path,
    output_provenance_path: str | Path,
    audio_manifest_path: str | Path | None = None,
    split: str = "public_dev",
    source_type: str = "hand_authored_synthetic",
    license_id: str = "CC-BY-4.0",
    authoring_method: str = "curated benchmark expansion",
    contamination_risk: str = "low",
    review_status: str = "draft",
) -> dict[str, Any]:
    """Append scenario drafts and write validated scenario/split/provenance files."""
    draft = _load_json(draft_path)
    suite = _load_json(scenario_path)
    split_manifest = _load_json(split_path)
    provenance = _load_json(provenance_path)
    audio_variant_ids = _load_audio_variant_ids(audio_manifest_path)

    new_scenarios = _extract_draft_scenarios(draft)
    added_ids = [scenario["id"] for scenario in new_scenarios if isinstance(scenario, dict)]
    expanded_suite = deepcopy(suite)
    expanded_split = deepcopy(split_manifest)
    expanded_provenance = deepcopy(provenance)
    expanded_suite.setdefault("scenarios", [])
    expanded_suite["scenarios"].extend(deepcopy(new_scenarios))
    _append_to_split(expanded_split, split, added_ids)
    _append_provenance(
        expanded_provenance,
        new_scenarios,
        source_type=source_type,
        license_id=license_id,
        authoring_method=authoring_method,
        contamination_risk=contamination_risk,
        review_status=review_status,
    )

    issues = validate_expanded_release(
        expanded_suite,
        expanded_split,
        expanded_provenance,
        audio_variant_ids=audio_variant_ids,
    )
    if issues:
        return {
            "added_ids": added_ids,
            "num_added": len(added_ids),
            "issues": issues,
            "outputs": {},
        }

    _write_json(output_scenario_path, expanded_suite)
    _write_json(output_split_path, expanded_split)
    _write_json(output_provenance_path, expanded_provenance)
    return {
        "added_ids": added_ids,
        "num_added": len(added_ids),
        "issues": [],
        "outputs": {
            "scenario_path": str(output_scenario_path),
            "split_path": str(output_split_path),
            "provenance_path": str(output_provenance_path),
        },
    }


def validate_expanded_release(
    suite: dict[str, Any],
    split_manifest: dict[str, Any],
    provenance_manifest: dict[str, Any],
    *,
    audio_variant_ids: set[str] | None = None,
) -> list[AuthoringIssue]:
    """Validate scenario suite, split assignments, and provenance together."""
    issues: list[AuthoringIssue] = []
    scenarios = suite.get("scenarios")
    if not isinstance(scenarios, list):
        return [AuthoringIssue("<suite>", "scenarios", "must be a list")]
    scenario_ids = {scenario.get("id") for scenario in scenarios if isinstance(scenario, dict)}
    for issue in validate_scenarios(scenarios):
        issues.append(AuthoringIssue(issue.scenario_id, issue.path, issue.message))
    for issue in validate_split_manifest(
        split_manifest,
        scenario_ids={item for item in scenario_ids if isinstance(item, str)},
        audio_variant_ids=audio_variant_ids or set(),
    ):
        issues.append(AuthoringIssue(issue.item_id, issue.path, issue.message))
    for issue in validate_provenance_manifest(
        provenance_manifest,
        scenario_ids={item for item in scenario_ids if isinstance(item, str)},
        audio_variant_ids=audio_variant_ids or set(),
    ):
        issues.append(AuthoringIssue(issue.item_id, issue.path, issue.message))
    return issues


def next_scenario_id(
    existing_ids: set[str],
    *,
    domain: str,
    slug: str,
) -> str:
    """Generate the next stable scenario ID for a domain/slug pair."""
    prefix = f"{_slug(domain)}-{_slug(slug)}"
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d{{3}})$")
    max_index = 0
    for scenario_id in existing_ids:
        match = pattern.match(scenario_id)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return f"{prefix}-{max_index + 1:03d}"


def scaffold_scenario_drafts(
    *,
    scenario_path: str | Path = DEFAULT_SCENARIO_PATH,
    split_path: str | Path | None = None,
    target_path: str | Path = DEFAULT_COVERAGE_TARGET_PATH,
    profile: str = "public_beta",
    count: int | None = None,
) -> dict[str, Any]:
    """Create incomplete scenario draft skeletons from coverage-plan gaps."""
    suite = _load_json(scenario_path)
    scenarios = suite.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("scenario suite must contain a scenarios list")

    plan = build_coverage_plan(
        scenario_path=scenario_path,
        split_path=split_path,
        target_path=target_path,
        profile=profile,
    )
    recommendations = plan["recommended_next_scenarios"]
    if count is not None:
        if count < 1:
            raise ValueError("count must be >= 1")
        recommendations = recommendations[:count]

    existing_ids = {
        scenario["id"]
        for scenario in scenarios
        if isinstance(scenario, dict) and isinstance(scenario.get("id"), str)
    }
    drafts = []
    for recommendation in recommendations:
        scenario_id = next_scenario_id(
            existing_ids,
            domain=recommendation["domain"],
            slug=_draft_slug(recommendation),
        )
        existing_ids.add(scenario_id)
        drafts.append(_scenario_skeleton(scenario_id, recommendation))

    return {
        "name": "OpenVoiceCS-Bench scenario drafts",
        "profile": profile,
        "source_scenario_path": str(scenario_path),
        "target_path": str(target_path),
        "num_scenarios": len(drafts),
        "draft_status": "incomplete_scaffold",
        "authoring_instructions": [
            "Replace every TODO value with synthetic, license-clean content.",
            "Define tool contracts whose expected calls replay to expected_state.",
            "Run add-scenarios and resolve all validation issues before release.",
        ],
        "coverage_gaps": plan["gaps"],
        "scenarios": drafts,
    }


def _extract_draft_scenarios(draft: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(draft.get("scenarios"), list):
        scenarios = draft["scenarios"]
    elif "id" in draft:
        scenarios = [draft]
    else:
        raise ValueError("draft must contain a scenario object or scenarios list")
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"draft scenarios[{index}] must be an object")
    return scenarios


def _append_to_split(
    split_manifest: dict[str, Any],
    split: str,
    scenario_ids: list[str],
) -> None:
    splits = split_manifest.setdefault("splits", {})
    target = splits.setdefault(split, {"scenario_ids": [], "audio_variant_ids": []})
    target.setdefault("scenario_ids", [])
    for scenario_id in scenario_ids:
        if scenario_id not in target["scenario_ids"]:
            target["scenario_ids"].append(scenario_id)


def _append_provenance(
    provenance_manifest: dict[str, Any],
    scenarios: list[dict[str, Any]],
    *,
    source_type: str,
    license_id: str,
    authoring_method: str,
    contamination_risk: str,
    review_status: str,
) -> None:
    scenario_provenance = provenance_manifest.setdefault("scenarios", {})
    for scenario in scenarios:
        scenario_id = scenario["id"]
        scenario_provenance[scenario_id] = {
            "source_type": source_type,
            "license": license_id,
            "authoring_method": authoring_method,
            "contains_real_customer_data": False,
            "contamination_risk": contamination_risk,
            "review_status": review_status,
        }


def _load_audio_variant_ids(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    issues = validate_audio_manifest_file(path)
    if issues:
        formatted = "; ".join(f"{issue.scenario_id}:{issue.message}" for issue in issues)
        raise ValueError(f"audio manifest validation failed: {formatted}")
    data = _load_json(path)
    return {
        variant["id"]
        for variant in data.get("variants", [])
        if isinstance(variant, dict) and variant.get("id")
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "scenario"


def _draft_slug(recommendation: dict[str, Any]) -> str:
    parts = [
        recommendation.get("track") or "track",
        recommendation.get("difficulty") or "difficulty",
        recommendation.get("split") or "split",
        "draft",
    ]
    return "-".join(str(part) for part in parts)


def _scenario_skeleton(scenario_id: str, recommendation: dict[str, Any]) -> dict[str, Any]:
    domain = recommendation["domain"]
    track = recommendation["track"]
    difficulty = recommendation["difficulty"]
    split = recommendation["split"]
    return {
        "id": scenario_id,
        "domain": domain,
        "track": track,
        "difficulty": difficulty,
        "customer_goal": f"TODO: define a synthetic {domain} customer goal for {track}.",
        "customer_profile": {
            "name": "TODO synthetic customer",
            "account_id": "acct_TODO",
        },
        "conversation": [
            {
                "role": "customer",
                "text": "TODO: write the customer opening turn, including any needed ambiguity.",
            }
        ],
        "initial_state": {
            "accounts": {},
            "orders": {},
            "cases": {},
        },
        "policy": {
            "summary": "TODO: write the applicable support policy and SOP constraints.",
            "required_events": ["TODO_required_policy_event"],
        },
        "tools": [],
        "experience": {
            "max_agent_words": 90,
            "max_latency_ms": 2500,
            "reference_latency_ms": 750,
        },
        "oracle": {
            "expected_tool_calls": [],
            "forbidden_tool_calls": [],
            "required_events": ["TODO_required_policy_event"],
            "forbidden_events": ["TODO_forbidden_policy_event"],
            "grounding": {
                "required_claims": [
                    {
                        "id": "TODO_required_claim",
                        "any_terms": ["TODO"],
                    }
                ],
                "forbidden_claims": [],
                "max_hallucinations_per_turn": 0,
            },
            "privacy": {
                "required_events": ["pii_minimization"],
                "forbidden_events": ["TODO_forbidden_privacy_event"],
                "forbidden_disclosures": [],
            },
            "auth": {
                "required_events": ["TODO_required_auth_event"],
                "verification_events": ["TODO_required_auth_event"],
                "protected_tools": [],
                "forbidden_events": ["TODO_forbidden_auth_event"],
            },
            "expected_state": {
                "TODO_replace_with_expected_final_state": True,
            },
            "reference_response": "TODO: write a concise, policy-grounded reference response.",
        },
        "tags": [domain, track, difficulty],
        "draft_metadata": {
            "coverage_split": split,
            "status": "incomplete_scaffold",
            "review_checklist": [
                "synthetic customer data only",
                "clear policy basis",
                "tool calls replay to expected_state",
                "privacy and auth gates covered",
                "grounding claims are observable in the final agent text",
            ],
        },
    }
