"""Tests for OpenVoiceCS scenario review manifests."""

from __future__ import annotations

from copy import deepcopy

from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench
from src.evaluation.benchmark.reviews import (
    load_review_manifest,
    scenario_review_stats,
    validate_review_manifest,
    validate_review_manifest_file,
)


def test_seed_review_manifest_approves_all_scenarios():
    bench = OpenVoiceCSBench.load()
    scenario_ids = {scenario["id"] for scenario in bench.scenarios}
    manifest = load_review_manifest()
    stats = scenario_review_stats(manifest, scenario_ids=scenario_ids)

    assert validate_review_manifest_file(
        scenario_ids=scenario_ids,
        benchmark_version=bench.version,
    ) == []
    assert stats["present"] is True
    assert stats["num_reviews"] == len(scenario_ids)
    assert stats["num_approved_scenarios"] == len(scenario_ids)
    assert stats["scenario_review_coverage"] == 1.0
    assert stats["scenario_approval_coverage"] == 1.0
    assert stats["unapproved_scenario_ids"] == []


def test_review_manifest_rejects_missing_approval_unknown_ids_and_checks():
    manifest = deepcopy(load_review_manifest())
    manifest["benchmark_version"] = "0.2.0"
    manifest["review_policy"]["minimum_reviewers_per_scenario"] = 2
    first = manifest["reviews"][0]
    first["scenario_id"] = "unknown-scenario"
    first["status"] = "changes_requested"
    first["reviewers"] = ["only_one"]
    first["checks"]["privacy_coverage"] = False

    messages = {
        (issue.item_id, issue.path, issue.message)
        for issue in validate_review_manifest(
            manifest,
            scenario_ids={"retail-refund-damaged-item-001"},
            benchmark_version="0.1.0",
        )
    }

    assert (
        "<reviews>",
        "benchmark_version",
        "must match scenario suite version",
    ) in messages
    assert ("unknown-scenario", "reviews[0].scenario_id", "unknown scenario id") in messages
    assert ("unknown-scenario", "reviews[0].reviewers", "must include at least 2 reviewers") in messages
    assert ("unknown-scenario", "reviews[0].checks.privacy_coverage", "must be true") in messages
    assert ("retail-refund-damaged-item-001", "reviews", "missing scenario review") in messages
    assert ("retail-refund-damaged-item-001", "reviews", "scenario not approved") in messages
