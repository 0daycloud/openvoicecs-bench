"""Tests for frozen frontier run manifests."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, load_audio_manifest, oracle_agent
from src.evaluation.benchmark.run_manifest import (
    build_run_manifest,
    validate_run_manifest,
    validate_run_manifest_file,
)


def _write_oracle_report(tmp_path):
    report = OpenVoiceCSBench.load().score_agent(
        oracle_agent,
        max_scenarios=1,
        trials=1,
        model_metadata={
            "display_name": "oracle",
            "provider": "reference",
            "model_id": "oracle-agent-v0.1",
            "pricing_profile_id": "reference-zero-v0.1",
            "pricing_snapshot_date": "2026-06-11",
        },
    )
    path = tmp_path / "oracle_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_build_run_manifest_freezes_files_reports_and_systems(tmp_path):
    report_path = _write_oracle_report(tmp_path)

    manifest = build_run_manifest(
        [report_path],
        judge_model="claude-opus-4-6",
        seed=42,
        region="local",
        network="loopback",
        hardware_profile="ci-macos-arm64",
        transport="in_process",
        concurrency_levels=[1, 10, 100],
    )

    assert manifest["manifest_version"] == "0.1.0"
    assert manifest["release_tuple"]["seed"] == 42
    assert manifest["release_tuple"]["judge"]["model"] == "claude-opus-4-6"
    assert manifest["release_tuple"]["environment"]["region"] == "local"
    assert manifest["release_tuple"]["environment"]["network"] == "loopback"
    assert manifest["release_tuple"]["environment"]["hardware_profile"] == "ci-macos-arm64"
    assert manifest["release_tuple"]["environment"]["transport"] == "in_process"
    assert manifest["release_tuple"]["environment"]["concurrency_levels"] == [1, 10, 100]
    assert len(manifest["release_tuple"]["scenario_suite"]["sha256"]) == 64
    assert len(manifest["release_tuple"]["pricing_manifest"]["sha256"]) == 64
    assert len(manifest["release_tuple"]["split_manifest"]["sha256"]) == 64
    assert len(manifest["release_tuple"]["provenance_manifest"]["sha256"]) == 64
    assert len(manifest["release_tuple"]["changelog"]["sha256"]) == 64
    assert len(manifest["release_tuple"]["baseline_manifest"]["sha256"]) == 64
    assert len(manifest["release_tuple"]["review_manifest"]["sha256"]) == 64
    assert manifest["release_audit"]["split_manifest_stats"]["scenario_coverage"] == 1.0
    assert manifest["release_audit"]["provenance_stats"]["scenario_coverage"] == 1.0
    assert manifest["release_audit"]["changelog_stats"]["scenario_change_coverage"] == 1.0
    assert manifest["release_audit"]["baseline_stats"]["num_baselines"] == 4
    assert manifest["release_audit"]["review_stats"]["scenario_approval_coverage"] == 1.0
    assert manifest["release_audit"]["audio_asset_stats"]["num_sha256_verified"] == len(load_audio_manifest())
    assert manifest["release_audit"]["audio_asset_stats"]["num_positive_duration_files"] == len(
        load_audio_manifest()
    )
    assert manifest["reports"][0]["path"].endswith("oracle_report.json")
    assert len(manifest["reports"][0]["sha256"]) == 64
    assert manifest["reports"][0]["validation"]["passed"] is True
    assert manifest["systems"][0]["name"] == "oracle"
    assert manifest["systems"][0]["provider"] == "reference"
    assert manifest["systems"][0]["model_id"] == "oracle-agent-v0.1"
    assert manifest["systems"][0]["pricing_profile_id"] == "reference-zero-v0.1"
    assert manifest["systems"][0]["pricing_source"] == "profile"
    assert manifest["systems"][0]["pipeline_type"] == "cascaded"
    assert validate_run_manifest(manifest) == []


def test_validate_run_manifest_file_round_trip(tmp_path):
    report_path = _write_oracle_report(tmp_path)
    manifest = build_run_manifest(
        [report_path],
        region="local",
        network="loopback",
        transport="in_process",
        concurrency_levels=[1],
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_run_manifest_file(manifest_path) == []


def test_validate_run_manifest_file_detects_tampered_report_file(tmp_path):
    report_path = _write_oracle_report(tmp_path)
    manifest = build_run_manifest(
        [report_path],
        region="local",
        network="loopback",
        transport="in_process",
        concurrency_levels=[1],
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report_path.write_text("{}", encoding="utf-8")

    messages = {(issue.path, issue.message) for issue in validate_run_manifest_file(manifest_path)}

    assert ("reports[0].sha256", "does not match file contents") in messages
    assert ("reports[0].bytes", "does not match file size") in messages


def test_validate_run_manifest_file_resolves_relative_file_entries(tmp_path):
    report_path = _write_oracle_report(tmp_path)
    manifest = build_run_manifest(
        [report_path],
        region="local",
        network="loopback",
        transport="in_process",
        concurrency_levels=[1],
    )
    scenario_source = Path(manifest["release_tuple"]["scenario_suite"]["path"])
    scenario_copy = tmp_path / "inputs" / "scenarios.json"
    scenario_copy.parent.mkdir(parents=True)
    scenario_copy.write_bytes(scenario_source.read_bytes())
    manifest["release_tuple"]["scenario_suite"]["path"] = "inputs/scenarios.json"
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_run_manifest_file(manifest_path) == []

    scenario_copy.write_text("{}", encoding="utf-8")
    messages = {(issue.path, issue.message) for issue in validate_run_manifest_file(manifest_path)}

    assert ("release_tuple.scenario_suite.sha256", "does not match file contents") in messages


def test_validate_run_manifest_file_prefers_manifest_relative_paths(tmp_path, monkeypatch):
    report_path = _write_oracle_report(tmp_path)
    manifest = build_run_manifest(
        [report_path],
        audio_manifest_path=None,
        pricing_manifest_path=None,
        split_manifest_path=None,
        provenance_manifest_path=None,
        changelog_path=None,
        baseline_manifest_path=None,
        review_manifest_path=None,
        region="local",
        network="loopback",
        transport="in_process",
        concurrency_levels=[1],
    )
    manifest["systems"][0]["pricing_source"] = "profile"
    manifest["systems"][0]["pipeline_type"] = "cascaded"
    scenario_source = Path(manifest["release_tuple"]["scenario_suite"]["path"])
    bundle_dir = tmp_path / "bundle"
    scenario_copy = bundle_dir / "inputs" / "scenarios.json"
    scenario_copy.parent.mkdir(parents=True)
    scenario_copy.write_bytes(scenario_source.read_bytes())
    manifest["release_tuple"]["scenario_suite"]["path"] = "inputs/scenarios.json"
    manifest_path = bundle_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cwd_conflict = tmp_path / "inputs" / "scenarios.json"
    cwd_conflict.parent.mkdir(parents=True)
    cwd_conflict.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert validate_run_manifest_file(manifest_path) == []


def test_validate_run_manifest_rejects_missing_hashes():
    issues = validate_run_manifest({
        "benchmark": "Latency-Cost-Quality Frontier",
        "manifest_version": "0.1.0",
        "release_tuple": {
            "scenario_suite": {"path": "x", "sha256": "bad", "bytes": 1},
            "pricing_manifest": {"path": "p", "sha256": "bad", "bytes": 1},
            "split_manifest": {"path": "s", "sha256": "bad", "bytes": 1},
            "seed": "0",
            "judge": {},
            "environment": {"concurrency_levels": [0]},
        },
        "reports": [
            {
                "path": "report.json",
                "sha256": "bad",
                "bytes": 1,
                "validation": {"passed": False},
            }
        ],
        "systems": [],
    })
    messages = {(issue.path, issue.message) for issue in issues}

    assert ("release_tuple.scenario_suite.sha256", "must be a SHA-256 hex digest") in messages
    assert ("release_tuple.split_manifest.sha256", "must be a SHA-256 hex digest") in messages
    assert ("release_tuple.seed", "must be an integer") in messages
    assert ("release_tuple.environment.region", "must be a non-empty controlled value") in messages
    assert ("release_tuple.environment.network", "must be a non-empty controlled value") in messages
    assert ("release_tuple.environment.transport", "must be a non-empty controlled value") in messages
    assert ("release_tuple.environment.concurrency_levels", "must be positive integers") in messages
    assert ("reports[0].validation.passed", "must be true") in messages
    assert ("systems", "must be a non-empty list") in messages


def test_validate_run_manifest_rejects_unspecified_environment(tmp_path):
    report_path = _write_oracle_report(tmp_path)
    manifest = build_run_manifest(
        [report_path],
        region="unspecified",
        network="unspecified",
        transport="unspecified",
        concurrency_levels=[],
    )

    messages = {(issue.path, issue.message) for issue in validate_run_manifest(manifest)}

    assert ("release_tuple.environment.region", "must be a non-empty controlled value") in messages
    assert ("release_tuple.environment.network", "must be a non-empty controlled value") in messages
    assert ("release_tuple.environment.transport", "must be a non-empty controlled value") in messages
    assert ("release_tuple.environment.concurrency_levels", "must be positive integers") in messages


def test_validate_run_manifest_rejects_unpinned_system_metadata():
    manifest = {
        "benchmark": "Latency-Cost-Quality Frontier",
        "manifest_version": "0.1.0",
        "release_tuple": {
            "scenario_suite": {"path": "x", "sha256": "a" * 64, "bytes": 1},
            "pricing_manifest": {"path": "p", "sha256": "b" * 64, "bytes": 1},
            "split_manifest": {"path": "s", "sha256": "c" * 64, "bytes": 1},
            "seed": 0,
            "judge": {},
            "environment": {
                "region": "local",
                "network": "loopback",
                "transport": "in_process",
                "concurrency_levels": [1],
            },
        },
        "reports": [
            {
                "path": "report.json",
                "sha256": "d" * 64,
                "bytes": 1,
                "validation": {"passed": True},
            }
        ],
        "systems": [
            {
                "name": "unknown",
                "provider": None,
                "model_id": None,
                "pricing_snapshot_date": None,
                "pricing_source": None,
                "pipeline_type": None,
            }
        ],
    }

    messages = {(issue.path, issue.message) for issue in validate_run_manifest(manifest)}

    assert ("systems[0].name", "must be a non-empty system name") in messages
    assert ("systems[0].provider", "must be a non-empty provider") in messages
    assert ("systems[0].model_id", "must include model_id or submission_spec") in messages
    assert ("systems[0].pricing_snapshot_date", "must be pinned") in messages
    assert ("systems[0].pricing_source", "must be profile or embedded") in messages
    assert ("systems[0].pipeline_type", "must be pinned") in messages
