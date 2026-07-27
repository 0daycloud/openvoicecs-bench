"""Tests for OpenVoiceCS release-readiness profiles."""

from __future__ import annotations

import json

import pytest

from src.evaluation.benchmark.frontier import build_frontier_report, write_frontier_artifacts
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, build_release_audit, oracle_agent
from src.evaluation.benchmark.readiness import evaluate_release_readiness
from src.evaluation.benchmark.run_manifest import build_run_manifest


def _proxy_frontier_artifacts(tmp_path):
    def metered_oracle(scenario, trial_index):
        trace = oracle_agent(scenario, trial_index)
        trace["cost_usd"] = 0.01
        return trace

    report = OpenVoiceCSBench.load().score_agent(
        metered_oracle,
        max_scenarios=1,
        model_metadata={
            "display_name": "oracle",
            "provider": "reference",
            "model_id": "oracle-agent-v0.1",
            "pricing_profile_id": "reference-zero-v0.1",
            "pricing_snapshot_date": "2026-06-11",
        },
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    frontier = build_frontier_report(
        [report],
        pricing_snapshot_date="2026-06-11",
        environment={
            "region": "local",
            "network": "loopback",
            "transport": "in_process",
            "concurrency_levels": [1],
        },
    )
    manifest = build_run_manifest(
        [report_path],
        region="local",
        network="loopback",
        transport="in_process",
        concurrency_levels=[1],
    )
    plot_dir = tmp_path / "plots"
    write_frontier_artifacts(frontier, plot_dir)
    return frontier, manifest, plot_dir


def test_seed_readiness_profile_passes_current_seed_suite():
    audit = build_release_audit()

    readiness = evaluate_release_readiness(audit, profile="seed")

    assert readiness["profile"] == "seed"
    assert readiness["passed"] is True
    assert readiness["num_issues"] == 0


def test_leaderboard_readiness_profile_requires_frontier_artifacts():
    audit = build_release_audit()

    readiness = evaluate_release_readiness(audit, profile="leaderboard_v1")

    assert readiness["passed"] is False
    issues = {issue["criterion"]: issue for issue in readiness["issues"]}
    assert "num_scenarios" not in issues
    assert "track.end_to_end_voice" not in issues
    assert "pricing_profiles" not in issues
    assert "split.sealed_test.scenarios" not in issues
    assert "split.sealed_test.audio_variants" not in issues
    assert issues["frontier_artifact"]["observed"] is False
    assert issues["run_manifest"]["observed"] is False


def test_readiness_checks_supplied_frontier_and_manifest_artifacts(tmp_path):
    frontier, manifest, plot_dir = _proxy_frontier_artifacts(tmp_path)
    readiness = evaluate_release_readiness(
        build_release_audit(),
        profile="seed",
        frontier_report=frontier,
        run_manifest=manifest,
        plot_dir=plot_dir,
    )

    assert readiness["passed"] is True
    assert readiness["artifacts"] == {
        "frontier_report": True,
        "run_manifest": True,
        "plot_dir": True,
    }


def test_readiness_can_verify_run_manifest_file_hashes(tmp_path):
    frontier, manifest, plot_dir = _proxy_frontier_artifacts(tmp_path)
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")

    readiness = evaluate_release_readiness(
        build_release_audit(),
        profile="seed",
        frontier_report=frontier,
        run_manifest=manifest,
        run_manifest_base_dir=tmp_path,
        verify_run_manifest_files=True,
        plot_dir=plot_dir,
    )
    issues = {issue["criterion"]: issue for issue in readiness["issues"]}

    assert readiness["passed"] is False
    assert "run_manifest.reports[0].sha256" in issues
    assert "run_manifest.reports[0].bytes" in issues


def test_public_release_readiness_requires_judged_experience_and_judge_pins(tmp_path):
    frontier, manifest, plot_dir = _proxy_frontier_artifacts(tmp_path)

    readiness = evaluate_release_readiness(
        build_release_audit(),
        profile="public_beta",
        frontier_report=frontier,
        run_manifest=manifest,
        plot_dir=plot_dir,
    )
    issues = {issue["criterion"]: issue for issue in readiness["issues"]}

    assert "frontier_artifact.scorecards.oracle.experience_score_source" in issues
    assert issues["frontier_artifact.scorecards.oracle.experience_score_source"]["observed"] == "proxy"
    assert "frontier_artifact.scorecards.oracle.experience_evidence.coverage" in issues
    assert "frontier_artifact.scorecards.oracle.experience_evidence.num_judged_trials" in issues
    assert "frontier_artifact.scorecards.oracle.experience_evidence.judge_counts" in issues
    assert "run_manifest.release_tuple.judge.model" in issues
    assert "run_manifest.release_tuple.judge.prompt" in issues
    assert "frontier_artifact.environment.concurrency_levels" in issues
    assert "frontier_artifact.environment.hardware_profile" in issues
    assert "run_manifest.release_tuple.environment.concurrency_levels" in issues
    assert "run_manifest.release_tuple.environment.hardware_profile" in issues
    assert "frontier_artifact.scorecards.oracle.latency_at_100_concurrency_p95_ms" in issues
    assert "frontier_artifact.scorecards.oracle.latency_load.levels.1" in issues
    assert "frontier_artifact.scorecards.oracle.latency_load.levels.10" in issues
    assert "frontier_artifact.scorecards.oracle.latency_load.levels.100" in issues
    assert "frontier_artifact.scorecards.oracle.latency_measurement.event_stream_samples" in issues
    assert "frontier_artifact.scorecards.oracle.latency_measurement.vad_origin_samples" in issues
    assert (
        "frontier_artifact.scorecards.oracle.latency_measurement.barge_in_stop_event_samples"
        in issues
    )
    assert (
        "frontier_artifact.scorecards.oracle.latency_measurement.interruption_recovery_event_samples"
        in issues
    )
    assert "frontier_artifact.scorecards.oracle.cost_provenance.fully_loaded_samples" in issues
    assert (
        "frontier_artifact.scorecards.oracle.axis_confidence_intervals.p95_v2v_ttfb_ms.n"
        in issues
    )
    assert (
        issues[
            "frontier_artifact.scorecards.oracle.axis_confidence_intervals.p95_v2v_ttfb_ms.n"
        ]["required"]
        == ">=3"
    )
    assert "run_manifest.reports[0].num_trials_per_scenario" in issues
    assert issues["run_manifest.reports[0].num_trials_per_scenario"]["required"] == ">=3"


def test_leaderboard_readiness_requires_more_repeated_trials(tmp_path):
    frontier, manifest, plot_dir = _proxy_frontier_artifacts(tmp_path)

    readiness = evaluate_release_readiness(
        build_release_audit(),
        profile="leaderboard_v1",
        frontier_report=frontier,
        run_manifest=manifest,
        plot_dir=plot_dir,
    )
    issues = {issue["criterion"]: issue for issue in readiness["issues"]}

    assert "run_manifest.reports[0].num_trials_per_scenario" in issues
    assert issues["run_manifest.reports[0].num_trials_per_scenario"]["observed"] == 1
    assert issues["run_manifest.reports[0].num_trials_per_scenario"]["required"] == ">=5"


def test_public_readiness_requires_saturated_load_samples(tmp_path):
    frontier, manifest, plot_dir = _proxy_frontier_artifacts(tmp_path)
    scorecard = frontier["scorecards"]["oracle"]
    scorecard["latency_at_100_concurrency_p95_ms"] = 250.0
    scorecard["latency_load"] = {
        "levels": {
            "1": {
                "target_concurrency": 1,
                "sample_count": 1,
                "saturated": True,
                "p95_v2v_ttfb_ms": 200.0,
                "requested_calls": 1,
                "completed_calls": 1,
                "error_calls": 0,
                "peak_active_calls": 1,
            },
            "10": {
                "target_concurrency": 10,
                "sample_count": 10,
                "saturated": True,
                "p95_v2v_ttfb_ms": 225.0,
                "requested_calls": 10,
                "completed_calls": 10,
                "error_calls": 0,
                "peak_active_calls": 10,
            },
            "100": {
                "target_concurrency": 100,
                "sample_count": 1,
                "saturated": False,
                "p95_v2v_ttfb_ms": 250.0,
                "requested_calls": 100,
                "completed_calls": 99,
                "error_calls": 1,
                "peak_active_calls": 99,
            },
        }
    }
    frontier["systems"]["oracle"]["scorecard"] = scorecard
    frontier["environment"]["concurrency_levels"] = [1, 10, 100]

    readiness = evaluate_release_readiness(
        build_release_audit(),
        profile="public_beta",
        frontier_report=frontier,
        run_manifest=manifest,
        plot_dir=plot_dir,
    )
    issues = {issue["criterion"]: issue for issue in readiness["issues"]}

    assert "frontier_artifact.scorecards.oracle.latency_load.levels.100.sample_count" in issues
    assert issues[
        "frontier_artifact.scorecards.oracle.latency_load.levels.100.sample_count"
    ]["required"] == ">=100"
    assert "frontier_artifact.scorecards.oracle.latency_load.levels.100.completed_calls" in issues
    assert "frontier_artifact.scorecards.oracle.latency_load.levels.100.peak_active_calls" in issues
    assert "frontier_artifact.scorecards.oracle.latency_load.levels.100.saturated" in issues
    assert "frontier_artifact.scorecards.oracle.latency_load.levels.100.error_calls" in issues


def test_public_readiness_requires_full_latency_scorecard_fields(tmp_path):
    frontier, manifest, plot_dir = _proxy_frontier_artifacts(tmp_path)
    scorecard = frontier["scorecards"]["oracle"]
    for field in (
        "p90_v2v_ttfb_ms",
        "p99_v2v_ttfb_ms",
        "p95_v2v_last_byte_ms",
        "barge_in_stop_p95_ms",
        "interruption_recovery_p95_ms",
    ):
        scorecard[field] = None
    scorecard["stage_latency_ms"]["llm_ttft_ms"]["p95"] = None
    frontier["systems"]["oracle"]["scorecard"] = scorecard

    readiness = evaluate_release_readiness(
        build_release_audit(),
        profile="public_beta",
        frontier_report=frontier,
        run_manifest=manifest,
        plot_dir=plot_dir,
    )
    issues = {issue["criterion"]: issue for issue in readiness["issues"]}

    assert "frontier_artifact.scorecards.oracle.p90_v2v_ttfb_ms" in issues
    assert "frontier_artifact.scorecards.oracle.p99_v2v_ttfb_ms" in issues
    assert "frontier_artifact.scorecards.oracle.p95_v2v_last_byte_ms" in issues
    assert "frontier_artifact.scorecards.oracle.barge_in_stop_p95_ms" in issues
    assert "frontier_artifact.scorecards.oracle.interruption_recovery_p95_ms" in issues
    assert "frontier_artifact.scorecards.oracle.stage_latency_ms.llm_ttft_ms.p95" in issues


def test_readiness_rejects_malformed_supplied_frontier():
    readiness = evaluate_release_readiness(
        build_release_audit(),
        profile="seed",
        frontier_report={
            "benchmark": "Latency-Cost-Quality Frontier",
            "systems": {},
            "scorecards": {},
            "environment": {"region": "unspecified"},
        },
    )
    issues = {issue["criterion"]: issue for issue in readiness["issues"]}

    assert readiness["passed"] is False
    assert "frontier_artifact.systems" in issues
    assert "frontier_artifact.environment.region" in issues


def test_readiness_uses_frontier_report_validator(tmp_path):
    frontier, _manifest, _plot_dir = _proxy_frontier_artifacts(tmp_path)
    frontier["scorecards"]["oracle"].pop("axis_confidence_intervals")

    readiness = evaluate_release_readiness(
        build_release_audit(),
        profile="seed",
        frontier_report=frontier,
    )
    issues = {issue["criterion"]: issue for issue in readiness["issues"]}

    assert readiness["passed"] is False
    assert "frontier_artifact.scorecards.oracle.axis_confidence_intervals" in issues


def test_readiness_rejects_malformed_plot_artifacts(tmp_path):
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()
    (plot_dir / "frontier_plot_data.json").write_text(
        json.dumps({"domains": {"all": {"points": []}}}),
        encoding="utf-8",
    )
    readiness = evaluate_release_readiness(
        build_release_audit(),
        profile="seed",
        plot_dir=plot_dir,
    )
    issues = {issue["criterion"]: issue for issue in readiness["issues"]}

    assert readiness["passed"] is False
    assert "frontier_plot_artifacts.all.3d" in issues


def test_readiness_rejects_stale_plot_data_for_supplied_frontier(tmp_path):
    frontier, _manifest, plot_dir = _proxy_frontier_artifacts(tmp_path)
    plot_data_path = plot_dir / "frontier_plot_data.json"
    plot_data = json.loads(plot_data_path.read_text(encoding="utf-8"))
    plot_data["domains"]["all"]["points"][0]["system"] = "stale-system"
    plot_data_path.write_text(json.dumps(plot_data), encoding="utf-8")

    readiness = evaluate_release_readiness(
        build_release_audit(),
        profile="seed",
        frontier_report=frontier,
        plot_dir=plot_dir,
    )
    issues = {issue["criterion"]: issue for issue in readiness["issues"]}

    assert readiness["passed"] is False
    assert "frontier_plot_artifacts.plot_data" in issues


def test_readiness_rejects_unknown_profile():
    audit = build_release_audit()

    with pytest.raises(ValueError, match="unknown readiness profile"):
        evaluate_release_readiness(audit, profile="unknown")
