"""Tests for latency-cost-quality release bundle assembly."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil

from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, oracle_agent
from src.evaluation.benchmark.release_bundle import (
    build_frontier_release_bundle,
    validate_frontier_release_bundle,
    validate_frontier_release_bundle_file,
)


def _metered_oracle_report() -> dict:
    bench = OpenVoiceCSBench.load()

    def metered_oracle(scenario, trial_index):
        trace = oracle_agent(scenario, trial_index)
        latency = scenario.get("experience", {}).get("reference_latency_ms", 700)
        trace["latency"] = {
            "v2v_ttfb_ms": latency,
            "v2v_last_byte_ms": latency + 350,
            "barge_in_stop_ms": 80,
            "interruption_recovery_ms": 200,
            "stage_latency_ms": {
                "asr_finalization_ms": 40,
                "llm_ttft_ms": 140,
                "tts_first_chunk_ms": 60,
            },
        }
        trace["cost_usd"] = 0.01
        return trace

    return bench.score_agent(
        metered_oracle,
        max_scenarios=1,
        trials=1,
        model_metadata={
            "display_name": "metered-oracle",
            "provider": "reference",
            "model_id": "oracle-agent-v0.1",
            "pricing_profile_id": "reference-zero-v0.1",
            "pricing_snapshot_date": "2026-06-11",
            "pipeline_type": "cascaded",
        },
    )


def test_build_frontier_release_bundle_writes_valid_artifacts(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_metered_oracle_report()), encoding="utf-8")

    bundle = build_frontier_release_bundle(
        [report_path],
        tmp_path / "bundle",
        region="local",
        network="loopback",
        transport="in_process",
        concurrency_levels=[1],
        pricing_snapshot_date="2026-06-11",
        readiness_profile="seed",
    )

    output = tmp_path / "bundle"
    assert bundle["validation"]["passed"] is True
    assert (output / "frontier_report.json").exists()
    assert (output / "run_manifest.json").exists()
    assert (output / "readiness.json").exists()
    assert (output / "release_bundle.json").exists()
    assert (output / "plots" / "frontier_plot_data.json").exists()
    assert (output / "plots" / "all_3d.svg").exists()
    assert (output / "scorecards" / "scorecards.json").exists()
    assert (output / "scorecards" / "scorecards.csv").exists()
    assert (output / "scorecards" / "scorecards.md").exists()
    assert (output / "inputs" / "reports" / "001_report.json").exists()
    assert (output / "inputs" / "manifests" / "scenarios.json").exists()
    assert (output / "inputs" / "manifests" / "changelog.json").exists()
    assert (output / "inputs" / "manifests" / "reference_baselines.json").exists()
    assert (output / "inputs" / "manifests" / "scenario_reviews.json").exists()
    assert bundle["artifacts"]["frontier_report"]["sha256"]
    assert not Path(bundle["artifacts"]["frontier_report"]["path"]).is_absolute()
    assert bundle["artifacts"]["scorecard:json"]["sha256"]
    assert bundle["artifacts"]["scorecard:csv"]["bytes"] > 0
    assert bundle["artifacts"]["run_manifest"]["bytes"] > 0
    assert bundle["input_files"]["reports"][0]["sha256"]
    assert not Path(bundle["input_files"]["reports"][0]["path"]).is_absolute()
    assert bundle["input_files"]["scenario"]["path"].endswith("inputs/manifests/scenarios.json")
    assert bundle["input_files"]["changelog"]["path"].endswith("inputs/manifests/changelog.json")
    assert bundle["input_files"]["baseline_manifest"]["path"].endswith(
        "inputs/manifests/reference_baselines.json"
    )
    assert bundle["input_files"]["review_manifest"]["path"].endswith(
        "inputs/manifests/scenario_reviews.json"
    )
    assert (
        bundle["input_files"]["scenario"]["source_path"]
        == "data/openvoicecs/scenarios_v0.1.json"
    )
    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["reports"][0]["path"] == "inputs/reports/001_report.json"
    assert (
        run_manifest["release_tuple"]["scenario_suite"]["path"]
        == "inputs/manifests/scenarios.json"
    )
    assert (
        run_manifest["release_tuple"]["baseline_manifest"]["path"]
        == "inputs/manifests/reference_baselines.json"
    )
    assert (
        run_manifest["release_tuple"]["review_manifest"]["path"]
        == "inputs/manifests/scenario_reviews.json"
    )
    assert validate_frontier_release_bundle(bundle, base_dir=output) == []
    assert validate_frontier_release_bundle_file(output / "release_bundle.json") == []

    copied = tmp_path / "copied_bundle"
    shutil.copytree(output, copied)
    report_path.write_text("{}", encoding="utf-8")
    assert validate_frontier_release_bundle_file(copied / "release_bundle.json") == []


def test_validate_frontier_release_bundle_prefers_bundle_relative_paths(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_metered_oracle_report()), encoding="utf-8")
    build_frontier_release_bundle(
        [report_path],
        tmp_path / "bundle",
        region="local",
        network="loopback",
        transport="in_process",
        concurrency_levels=[1],
        pricing_snapshot_date="2026-06-11",
        readiness_profile="seed",
    )
    (tmp_path / "frontier_report.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert validate_frontier_release_bundle_file(tmp_path / "bundle" / "release_bundle.json") == []


def test_build_frontier_release_bundle_reports_readiness_failures(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_metered_oracle_report()), encoding="utf-8")

    bundle = build_frontier_release_bundle(
        [report_path],
        tmp_path / "bundle",
        region="local",
        network="loopback",
        transport="in_process",
        concurrency_levels=[1],
        pricing_snapshot_date="2026-06-11",
        readiness_profile="public_beta",
    )

    assert bundle["validation"]["passed"] is False
    assert bundle["validation"]["frontier_report"]["passed"] is True
    assert bundle["validation"]["run_manifest"]["passed"] is True
    assert bundle["validation"]["readiness"]["passed"] is False


def test_build_frontier_release_bundle_rejects_invalid_input_report(tmp_path):
    report = _metered_oracle_report()
    report["pass_k"] = 0.0
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    try:
        build_frontier_release_bundle(
            [report_path],
            tmp_path / "bundle",
            region="local",
            network="loopback",
            transport="in_process",
            concurrency_levels=[1],
            pricing_snapshot_date="2026-06-11",
            readiness_profile="seed",
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected invalid report to be rejected")

    assert "Report validation failed:" in message
    assert "pass_k: must equal mean scenario pass^k" in message
    assert not (tmp_path / "bundle" / "release_bundle.json").exists()


def test_validate_frontier_release_bundle_detects_tampered_artifact(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_metered_oracle_report()), encoding="utf-8")
    build_frontier_release_bundle(
        [report_path],
        tmp_path / "bundle",
        region="local",
        network="loopback",
        transport="in_process",
        concurrency_levels=[1],
        pricing_snapshot_date="2026-06-11",
        readiness_profile="seed",
    )
    frontier_path = tmp_path / "bundle" / "frontier_report.json"
    frontier_path.write_text("{}", encoding="utf-8")

    issues = validate_frontier_release_bundle_file(tmp_path / "bundle" / "release_bundle.json")
    messages = {(issue.path, issue.message) for issue in issues}

    assert ("artifacts.frontier_report.sha256", "does not match file contents") in messages
    assert ("artifacts.frontier_report.bytes", "does not match file size") in messages


def test_validate_frontier_release_bundle_detects_tampered_input_report(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_metered_oracle_report()), encoding="utf-8")
    build_frontier_release_bundle(
        [report_path],
        tmp_path / "bundle",
        region="local",
        network="loopback",
        transport="in_process",
        concurrency_levels=[1],
        pricing_snapshot_date="2026-06-11",
        readiness_profile="seed",
    )
    snapshotted_report = tmp_path / "bundle" / "inputs" / "reports" / "001_report.json"
    snapshotted_report.write_text("{}", encoding="utf-8")

    issues = validate_frontier_release_bundle_file(tmp_path / "bundle" / "release_bundle.json")
    messages = {(issue.path, issue.message) for issue in issues}

    assert ("input_files.reports[0].sha256", "does not match file contents") in messages
    assert ("input_files.reports[0].bytes", "does not match file size") in messages


def test_validate_frontier_release_bundle_rejects_stale_plot_data(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_metered_oracle_report()), encoding="utf-8")
    bundle = build_frontier_release_bundle(
        [report_path],
        tmp_path / "bundle",
        region="local",
        network="loopback",
        transport="in_process",
        concurrency_levels=[1],
        pricing_snapshot_date="2026-06-11",
        readiness_profile="seed",
    )
    plot_data_path = tmp_path / "bundle" / "plots" / "frontier_plot_data.json"
    plot_data = json.loads(plot_data_path.read_text(encoding="utf-8"))
    plot_data["domains"]["all"]["points"][0]["system"] = "stale-system"
    plot_data_path.write_text(json.dumps(plot_data), encoding="utf-8")
    bundle["artifacts"]["plot:plot_data"]["sha256"] = hashlib.sha256(
        plot_data_path.read_bytes()
    ).hexdigest()
    bundle["artifacts"]["plot:plot_data"]["bytes"] = plot_data_path.stat().st_size

    issues = validate_frontier_release_bundle(bundle, base_dir=tmp_path / "bundle")
    messages = {(issue.path, issue.message) for issue in issues}

    assert (
        "artifacts.plot:plot_data",
        "must match plot data regenerated from frontier_report",
    ) in messages
