"""Tests for OpenVoiceCS benchmark datasheets."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from src.evaluation.benchmark.datasheet import (
    build_benchmark_datasheet,
    build_benchmark_datasheet_file,
    validate_benchmark_datasheet,
    validate_benchmark_datasheet_file,
)
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, load_audio_manifest


def test_benchmark_datasheet_builds_from_seed_release():
    with open("data/openvoicecs/split_commitments_v0.1.json", encoding="utf-8") as f:
        commitments = json.load(f)

    datasheet = build_benchmark_datasheet()

    assert validate_benchmark_datasheet(datasheet) == []
    assert datasheet["benchmark"] == "OpenVoiceCS-Bench Datasheet"
    assert datasheet["release"]["benchmark_version"] == "0.1.0"
    assert datasheet["release"]["split_commitment_root_hash"] == commitments["root_hash"]
    assert datasheet["data_summary"]["num_scenarios"] == len(OpenVoiceCSBench.load().scenarios)
    assert datasheet["data_summary"]["num_audio_variants"] == len(load_audio_manifest())
    assert datasheet["split_policy"]["scenario_coverage"] == 1.0
    assert datasheet["changelog"]["scenario_change_coverage"] == 1.0
    assert datasheet["baselines"]["num_baselines"] == 4
    assert datasheet["scenario_reviews"]["scenario_approval_coverage"] == 1.0
    assert datasheet["governance"]["requires_changelog"] is True
    assert datasheet["governance"]["requires_reference_baselines"] is True
    assert datasheet["governance"]["requires_scenario_reviews"] is True
    assert datasheet["governance"]["requires_submission_cards_for_official_leaderboards"]
    assert datasheet["release_validation"]["audit_passed"] is True


def test_benchmark_datasheet_file_round_trips_and_validates(tmp_path: Path):
    output_path = tmp_path / "datasheet.json"

    datasheet = build_benchmark_datasheet_file(output_path=output_path)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == datasheet
    assert validate_benchmark_datasheet_file(output_path) == []
    assert loaded["release"]["scenario_file"]["sha256"]
    assert loaded["release"]["changelog_file"]["sha256"]
    assert loaded["release"]["baseline_manifest_file"]["sha256"]
    assert loaded["release"]["review_manifest_file"]["sha256"]
    assert loaded["release"]["scenario_file"]["bytes"] > 0


def test_benchmark_datasheet_validation_rejects_tampering():
    datasheet = build_benchmark_datasheet()
    tampered = deepcopy(datasheet)
    tampered["release"]["split_commitment_root_hash"] = "not-a-hash"
    tampered["data_summary"]["num_scenarios"] = -1
    tampered["governance"]["requires_split_manifest"] = False
    tampered["governance"]["requires_changelog"] = False
    tampered["governance"]["requires_reference_baselines"] = False
    tampered["governance"]["requires_scenario_reviews"] = False
    tampered["baselines"]["num_baselines"] = 0
    tampered["scenario_reviews"]["scenario_approval_coverage"] = 0.5
    tampered["release_validation"]["release_gates_passed"] = False

    messages = {
        (issue.path, issue.message)
        for issue in validate_benchmark_datasheet(tampered)
    }

    assert (
        "release.split_commitment_root_hash",
        "must be a SHA-256 hex digest",
    ) in messages
    assert ("data_summary.num_scenarios", "must be a nonnegative integer") in messages
    assert ("governance.requires_split_manifest", "must be true") in messages
    assert ("governance.requires_changelog", "must be true") in messages
    assert ("governance.requires_reference_baselines", "must be true") in messages
    assert ("governance.requires_scenario_reviews", "must be true") in messages
    assert ("baselines.num_baselines", "must be a positive integer") in messages
    assert (
        "scenario_reviews.scenario_approval_coverage",
        "must be 1.0",
    ) in messages
    assert ("release_validation.release_gates_passed", "must be true") in messages


def test_benchmark_datasheet_validation_requires_policy_lists():
    datasheet = build_benchmark_datasheet()
    datasheet["intended_use"] = []
    datasheet["out_of_scope_uses"] = [" "]
    datasheet["known_limitations"] = "none"

    messages = {
        (issue.path, issue.message)
        for issue in validate_benchmark_datasheet(datasheet)
    }

    assert ("intended_use", "must be a non-empty list") in messages
    assert (
        "out_of_scope_uses",
        "must contain only non-empty strings",
    ) in messages
    assert ("known_limitations", "must be a non-empty list") in messages
