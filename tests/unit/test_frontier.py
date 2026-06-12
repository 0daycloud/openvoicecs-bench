"""Tests for latency-cost-quality frontier reporting."""

from __future__ import annotations

import json

from src.evaluation.benchmark.frontier import (
    build_constrained_frontiers,
    build_frontier_plot_data,
    build_frontier_report,
    build_scorecard_rows,
    build_utility_view,
    render_frontier_svg,
    validate_frontier_report,
    validate_frontier_report_file,
    write_frontier_artifacts,
    write_scorecard_artifacts,
)
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, oracle_agent


def _report(
    name: str,
    *,
    task_success: float,
    experience: float = 0.9,
    costs: list[float] | None = None,
    latencies: list[float] | None = None,
    pricing_date: str = "2026-06-11",
    p95_at_100: float | None = None,
) -> dict:
    latencies = latencies or [100.0]
    costs = costs or [0.01]
    return {
        "benchmark": "OpenVoiceCS-Bench",
        "benchmark_version": "0.1.0",
        "model_metadata": {
            "display_name": name,
            "pricing_snapshot_date": pricing_date,
        },
        "metric_scores": {
            "task_success": task_success,
            "experience_proxy": experience,
        },
        "operational_metrics": {
            "latency_at_100_concurrency_p95_ms": p95_at_100,
        },
        "num_scenarios": len(latencies),
        "results": [
            {
                "id": f"{name}-{idx}",
                "trials": [
                    {
                        "passed": True,
                        "latency": {
                            "v2v_ttfb_ms": latency,
                            "v2v_last_byte_ms": latency + 200.0,
                            "barge_in_stop_ms": latency / 4,
                            "interruption_recovery_ms": latency / 2,
                            "stage_latency_ms": {
                                "asr_finalization_ms": 20.0,
                                "llm_ttft_ms": latency - 40.0,
                                "tts_first_chunk_ms": 20.0,
                            },
                        },
                        "cost_usd": costs[idx],
                    }
                ],
            }
            for idx, latency in enumerate(latencies)
        ],
    }


def test_frontier_marks_non_dominated_systems():
    report = build_frontier_report(
        [
            _report("fast-expensive", task_success=0.9, costs=[0.08, 0.08], latencies=[100, 120]),
            _report("slow-cheap", task_success=0.9, costs=[0.02, 0.02], latencies=[400, 450]),
            _report("dominated", task_success=0.8, costs=[0.09, 0.09], latencies=[500, 550]),
        ]
    )

    assert report["frontier"] == ["fast-expensive", "slow-cheap"]
    assert report["systems"]["dominated"]["frontier_eligible"] is True


def test_frontier_report_preserves_controlled_environment_metadata():
    report = build_frontier_report(
        [_report("env", task_success=1.0, costs=[0.01], latencies=[100])],
        environment={
            "region": "us-east-test",
            "network": "controlled-wired",
            "hardware_profile": "c7g.2xlarge-client",
            "transport": "webrtc",
            "concurrency_levels": [1, 10, 100],
        },
    )

    assert report["environment"] == {
        "region": "us-east-test",
        "network": "controlled-wired",
        "hardware_profile": "c7g.2xlarge-client",
        "transport": "webrtc",
        "concurrency_levels": [1, 10, 100],
    }


def test_constrained_frontiers_filter_budget_before_pareto():
    frontier = build_frontier_report(
        [
            _report("fast-expensive", task_success=0.9, costs=[0.08, 0.08], latencies=[100, 120]),
            _report("slow-cheap", task_success=0.9, costs=[0.02, 0.02], latencies=[400, 450]),
            _report("balanced", task_success=0.95, costs=[0.04, 0.04], latencies=[250, 260]),
        ],
        latency_targets_ms=[300],
        cost_targets_usd=[0.05],
    )

    entries = frontier["constrained_frontiers"]["entries"]

    assert len(entries) == 1
    assert entries[0]["constraint"] == {"latency_ms": 300.0, "cost_usd": 0.05}
    assert entries[0]["eligible_systems"] == ["balanced"]
    assert entries[0]["frontier"] == ["balanced"]


def test_build_constrained_frontiers_accepts_single_axis_targets():
    systems = list(build_frontier_report([
        _report("fast", task_success=1.0, costs=[0.09], latencies=[100]),
        _report("cheap", task_success=0.9, costs=[0.01], latencies=[400]),
    ])["systems"].values())

    constrained = build_constrained_frontiers(systems, latency_targets_ms=[150])

    assert constrained["entries"][0]["constraint"] == {"latency_ms": 150.0, "cost_usd": None}
    assert constrained["entries"][0]["frontier"] == ["fast"]


def test_success_normalized_cost_uses_task_success_rate():
    report = build_frontier_report([
        _report("partial", task_success=0.5, costs=[0.10, 0.10], latencies=[100, 100]),
    ])

    scorecard = report["scorecards"]["partial"]

    assert scorecard["cost_usd_per_conversation"] == 0.1
    assert scorecard["cost_usd_per_successful_conversation"] == 0.2


def test_direct_cost_samples_are_not_fully_loaded_without_components():
    report = build_frontier_report([
        _report("direct-only", task_success=1.0, costs=[0.01], latencies=[100]),
    ])

    provenance = report["scorecards"]["direct-only"]["cost_provenance"]

    assert provenance["sample_count"] == 1
    assert provenance["direct_cost_samples"] == 1
    assert provenance["derived_cost_samples"] == 0
    assert provenance["fully_loaded_samples"] == 0
    assert provenance["missing_cost_samples"] == 0
    assert provenance["required_components"] == ["asr", "llm", "tts", "telephony", "transport"]


def test_cost_can_be_derived_from_usage_and_component_pricing():
    report = {
        "benchmark": "OpenVoiceCS-Bench",
        "model_metadata": {
            "display_name": "priced-pipeline",
            "pricing_snapshot_date": "2026-06-11",
            "pricing": {
                "asr_per_minute": 0.006,
                "input_per_mtok": 1.0,
                "output_per_mtok": 2.0,
                "tts_per_1k_characters": 0.015,
                "telephony_per_minute": 0.01,
                "transport_per_minute": 0.002,
            },
        },
        "metric_scores": {"task_success": 0.5, "experience_proxy": 0.9},
        "results": [
            {
                "id": "scenario-1",
                "trials": [
                    {
                        "latency_ms": 100,
                        "usage": {
                            "asr_seconds": 60,
                            "input_tokens": 1000,
                            "output_tokens": 500,
                            "tts_characters": 1000,
                            "telephony_seconds": 60,
                            "transport_seconds": 60,
                        },
                    }
                ],
            }
        ],
    }

    frontier = build_frontier_report([report])
    scorecard = frontier["scorecards"]["priced-pipeline"]

    assert scorecard["cost_usd_per_conversation"] == 0.035
    assert scorecard["cost_usd_per_successful_conversation"] == 0.07
    assert scorecard["cost_provenance"]["sample_count"] == 1
    assert scorecard["cost_provenance"]["derived_cost_samples"] == 1
    assert scorecard["cost_provenance"]["fully_loaded_samples"] == 1
    assert scorecard["cost_provenance"]["component_sample_counts"]["asr"] == 1
    assert scorecard["avg_component_cost_usd"]["asr"] == 0.006
    assert scorecard["avg_component_cost_usd"]["llm"] == 0.002
    assert scorecard["avg_component_cost_usd"]["tts"] == 0.015
    assert scorecard["avg_component_cost_usd"]["telephony"] == 0.01
    assert scorecard["avg_component_cost_usd"]["transport"] == 0.002


def test_experience_gate_and_missing_pricing_date_exclude_from_frontier():
    report = build_frontier_report([
        _report(
            "bad-exp",
            task_success=1.0,
            experience=0.5,
            costs=[0.01],
            latencies=[100],
            pricing_date="",
        ),
    ])

    system = report["systems"]["bad-exp"]
    reason_types = {reason["type"] for reason in system["exclusion_reasons"]}

    assert system["frontier_eligible"] is False
    assert "below_experience_gate" in reason_types
    assert "missing_pricing_snapshot_date" in reason_types
    assert report["frontier"] == []


def test_frontier_prefers_judged_experience_over_proxy_for_gate():
    report = _report(
        "judged-low",
        task_success=1.0,
        experience=1.0,
        costs=[0.01],
        latencies=[100],
    )
    report["conversation_experience_score"] = 0.4
    report["conversation_experience"] = {
        "coverage": 1.0,
        "num_judged_trials": 1,
        "judge_counts": {"human-rater-01": 1},
    }

    frontier = build_frontier_report([report], experience_gate=0.6)
    system = frontier["systems"]["judged-low"]
    scorecard = frontier["scorecards"]["judged-low"]

    assert system["frontier_eligible"] is False
    assert scorecard["experience_score"] == 0.4
    assert scorecard["experience_score_source"] == "judged"
    assert scorecard["experience_evidence"] == {
        "coverage": 1.0,
        "num_judged_trials": 1,
        "num_trials": 1,
        "judge_counts": {"human-rater-01": 1},
    }
    assert system["exclusion_reasons"][0]["type"] == "below_experience_gate"


def test_latency_scorecard_includes_percentiles_stages_and_load():
    report = build_frontier_report([
        _report(
            "with-load",
            task_success=1.0,
            costs=[0.01, 0.01, 0.01],
            latencies=[100, 200, 300],
            p95_at_100=900,
        ),
    ])

    scorecard = report["scorecards"]["with-load"]

    assert scorecard["p50_v2v_ttfb_ms"] == 200.0
    assert scorecard["p95_v2v_ttfb_ms"] == 290.0
    assert scorecard["p99_v2v_ttfb_ms"] == 298.0
    assert scorecard["p95_v2v_last_byte_ms"] == 490.0
    assert scorecard["barge_in_stop_p95_ms"] == 72.5
    assert scorecard["interruption_recovery_p95_ms"] == 145.0
    assert scorecard["latency_at_100_concurrency_p95_ms"] == 900.0
    assert scorecard["stage_latency_ms"]["asr_finalization_ms"]["p50"] == 20.0
    assert scorecard["latency_measurement"]["sample_count"] == 3
    assert scorecard["latency_measurement"]["unknown_samples"] == 3


def test_latency_scorecard_summarizes_measurement_provenance():
    report = {
        "benchmark": "OpenVoiceCS-Bench",
        "model_metadata": {
            "display_name": "evented",
            "pricing_snapshot_date": "2026-06-11",
        },
        "metric_scores": {"task_success": 1.0, "experience_proxy": 0.9},
        "num_scenarios": 1,
        "results": [
            {
                "id": "scenario-1",
                "trials": [
                    {
                        "passed": True,
                        "cost_usd": 0.01,
                        "latency": {
                            "v2v_ttfb_ms": 220,
                            "v2v_last_byte_ms": 520,
                            "measurement": {
                                "source": "event_stream",
                                "origin_event": "user.end_speech",
                                "origin_t_ms": 0.0,
                                "first_audio_event": "tts.first_audio",
                                "last_audio_event": "agent.complete",
                                "barge_in_stop_event": "barge_in.stop",
                                "interruption_recovery_event": "barge_in.recovered",
                            },
                        },
                    }
                ],
            }
        ],
    }

    frontier = build_frontier_report([report])
    measurement = frontier["scorecards"]["evented"]["latency_measurement"]

    assert measurement["sample_count"] == 1
    assert measurement["event_stream_samples"] == 1
    assert measurement["vad_origin_samples"] == 1
    assert measurement["first_audio_event_samples"] == 1
    assert measurement["last_audio_event_samples"] == 1


def test_scorecard_includes_axis_confidence_intervals():
    report = {
        "benchmark": "OpenVoiceCS-Bench",
        "model_metadata": {
            "display_name": "uncertain",
            "pricing_snapshot_date": "2026-06-11",
        },
        "metric_scores": {"task_success": 0.5, "experience_proxy": 0.9},
        "results": [
            {
                "id": "scenario-1",
                "trials": [
                    {"passed": True, "latency_ms": 100, "cost_usd": 0.01},
                    {"passed": False, "latency_ms": 300, "cost_usd": 0.03},
                ],
            }
        ],
    }

    frontier = build_frontier_report([report])
    scorecard = frontier["scorecards"]["uncertain"]
    intervals = scorecard["axis_confidence_intervals"]

    assert intervals["p95_v2v_ttfb_ms"]["method"] == "bootstrap_percentile"
    assert intervals["p95_v2v_ttfb_ms"]["estimate"] == 290.0
    assert intervals["p95_v2v_ttfb_ms"]["n"] == 2
    assert intervals["cost_usd_per_conversation"]["estimate"] == 0.02
    assert intervals["cost_usd_per_successful_conversation"]["estimate"] == 0.04
    assert intervals["task_success_rate"]["method"] == "wilson"
    assert intervals["task_success_rate"]["estimate"] == 0.5
    assert frontier["systems"]["uncertain"]["confidence_intervals"]["frontier_axes"] == intervals


def test_optional_utility_view_is_not_required_for_frontier():
    report = build_frontier_report(
        [
            _report("a", task_success=1.0, costs=[0.01], latencies=[300]),
            _report("b", task_success=0.9, costs=[0.02], latencies=[100]),
        ],
        utility_weights={"quality": 1.0, "latency": 0.1, "cost": 0.1},
    )

    assert report["utility_view"]["note"] == "Secondary opt-in utility view; not a benchmark leaderboard."
    assert set(report["utility_view"]["ranking"]) == {"a", "b"}
    assert validate_frontier_report(report) == []


def test_build_utility_view_accepts_pre_normalized_systems():
    report = build_frontier_report([
        _report("a", task_success=1.0, costs=[0.01], latencies=[200]),
    ])

    utility = build_utility_view(list(report["systems"].values()), {"quality": 1.0})

    assert utility["ranking"] == ["a"]
    assert utility["scores"]["a"] == 1.0


def test_validate_frontier_report_rejects_stale_utility_view():
    frontier = build_frontier_report(
        [
            _report("a", task_success=1.0, costs=[0.01], latencies=[300]),
            _report("b", task_success=0.9, costs=[0.02], latencies=[100]),
        ],
        utility_weights={"quality": 1.0, "latency": 0.1, "cost": 0.1},
    )
    frontier["utility_view"]["ranking"] = list(reversed(frontier["utility_view"]["ranking"]))
    frontier["utility_view"]["scores"]["a"] = -1.0

    messages = {(issue.path, issue.message) for issue in validate_frontier_report(frontier)}

    assert (
        "utility_view.scores.a",
        "must be numeric",
    ) not in messages
    assert (
        "utility_view",
        "must match recomputed utility view for declared weights",
    ) in messages


def test_frontier_consumes_openvoicecs_report_with_cost_and_latency():
    bench = OpenVoiceCSBench.load()

    def metered_oracle(scenario, trial_index):
        trace = oracle_agent(scenario, trial_index)
        trace["cost_usd"] = 0.01
        trace["latency"] = {
            "v2v_ttfb_ms": 250,
            "v2v_last_byte_ms": 650,
            "stage_latency_ms": {
                "asr_finalization_ms": 40,
                "llm_ttft_ms": 150,
                "tts_first_chunk_ms": 60,
            },
        }
        return trace

    openvoicecs_report = bench.score_agent(
        metered_oracle,
        max_scenarios=2,
        model_metadata={"agent": "metered_oracle", "pricing_snapshot_date": "2026-06-11"},
    )

    frontier = build_frontier_report([openvoicecs_report])
    scorecard = frontier["scorecards"]["metered_oracle"]

    assert frontier["frontier"] == ["metered_oracle"]
    assert frontier["systems"]["metered_oracle"]["frontier_eligible"] is True
    assert scorecard["p95_v2v_ttfb_ms"] == 250.0
    assert scorecard["cost_usd_per_successful_conversation"] == 0.01
    assert scorecard["task_success_rate"] == 1.0


def test_plot_data_includes_overall_and_domain_frontiers():
    reports = [
        _report("fast", task_success=1.0, costs=[0.03, 0.03], latencies=[100, 120]),
        _report("cheap", task_success=0.9, costs=[0.01, 0.01], latencies=[300, 320]),
        _report("weak", task_success=0.7, costs=[0.05, 0.05], latencies=[500, 520]),
    ]
    for report in reports:
        for idx, result in enumerate(report["results"]):
            result["domain"] = "retail" if idx == 0 else "travel"

    frontier = build_frontier_report(reports)
    plot_data = build_frontier_plot_data(frontier)

    assert set(plot_data["domains"]) == {"all", "retail", "travel"}
    assert set(frontier["domain_frontiers"]) == {"retail", "travel"}
    assert plot_data["domains"]["all"]["frontier"] == ["cheap", "fast"]
    assert plot_data["domains"]["retail"]["frontier"] == ["cheap", "fast"]
    assert frontier["domain_frontiers"]["retail"]["frontier"] == ["fast", "cheap"]
    assert frontier["domain_frontiers"]["retail"]["projection_frontiers"]["cost_vs_quality"] == [
        "cheap",
    ]
    assert frontier["domain_frontiers"]["retail"]["num_eligible_systems"] == 3
    assert plot_data["domains"]["all"]["points"][0]["on_frontier"] is True


def test_render_frontier_svg_outputs_3d_and_2d_svg():
    frontier = build_frontier_report([
        _report("fast", task_success=1.0, costs=[0.03], latencies=[100]),
        _report("cheap", task_success=0.9, costs=[0.01], latencies=[300]),
    ])
    plot_data = build_frontier_plot_data(frontier)

    svg_3d = render_frontier_svg(plot_data, projection="3d")
    svg_2d = render_frontier_svg(plot_data, projection="latency_vs_quality")

    assert svg_3d.startswith("<svg")
    assert "fast" in svg_3d
    assert "quality" in svg_3d
    assert "p95 voice-to-voice TTFB ms" in svg_2d


def test_write_frontier_artifacts_creates_plot_json_and_svgs(tmp_path):
    frontier = build_frontier_report([
        _report("a", task_success=1.0, costs=[0.01], latencies=[200]),
    ])

    written = write_frontier_artifacts(frontier, tmp_path)

    assert (tmp_path / "frontier_plot_data.json").exists()
    assert (tmp_path / "all_3d.svg").exists()
    assert (tmp_path / "all_latency_vs_quality.svg").exists()
    assert (tmp_path / "all_cost_vs_quality.svg").exists()
    assert written["plot_data"].endswith("frontier_plot_data.json")


def test_write_scorecard_artifacts_creates_standardized_outputs(tmp_path):
    frontier = build_frontier_report([
        _report("a", task_success=1.0, costs=[0.01], latencies=[200]),
    ])

    rows = build_scorecard_rows(frontier)
    written = write_scorecard_artifacts(frontier, tmp_path)

    assert rows[0]["system"] == "a"
    assert rows[0]["p95_v2v_ttfb_ms"] == 200.0
    assert rows[0]["cost_sample_count"] == 1
    assert rows[0]["direct_cost_samples"] == 1
    assert rows[0]["fully_loaded_cost_samples"] == 0
    assert (tmp_path / "scorecards.json").exists()
    assert (tmp_path / "scorecards.csv").read_text(encoding="utf-8").startswith(
        "system,frontier_eligible,"
    )
    assert "# Latency-Cost-Quality Scorecards" in (tmp_path / "scorecards.md").read_text(
        encoding="utf-8"
    )
    assert written["json"].endswith("scorecards.json")


def test_validate_frontier_report_accepts_generated_artifact(tmp_path):
    frontier = build_frontier_report([
        _report("valid", task_success=1.0, costs=[0.01], latencies=[200]),
    ])

    path = tmp_path / "frontier.json"
    path.write_text(json.dumps(frontier), encoding="utf-8")

    assert validate_frontier_report(frontier) == []
    assert validate_frontier_report_file(path) == []


def test_validate_frontier_report_rejects_stale_frontier_sets():
    reports = [
        _report("fast-expensive", task_success=0.9, costs=[0.08, 0.08], latencies=[100, 120]),
        _report("slow-cheap", task_success=0.9, costs=[0.02, 0.02], latencies=[400, 450]),
        _report("balanced", task_success=0.95, costs=[0.04, 0.04], latencies=[250, 260]),
        _report("weak", task_success=0.7, costs=[0.05, 0.05], latencies=[500, 520]),
    ]
    for report in reports:
        for result in report["results"]:
            result["domain"] = "retail"
    frontier = build_frontier_report(
        reports,
        latency_targets_ms=[300],
        cost_targets_usd=[0.05],
    )
    frontier["frontier"] = ["weak"]
    frontier["projection_frontiers"]["latency_vs_quality"] = ["weak"]
    frontier["constrained_frontiers"]["entries"][0]["frontier"] = ["fast-expensive"]
    frontier["constrained_frontiers"]["entries"][0]["eligible_systems"] = ["fast-expensive"]
    frontier["domain_frontiers"]["retail"]["frontier"] = ["weak"]

    messages = {(issue.path, issue.message) for issue in validate_frontier_report(frontier)}

    assert ("frontier", "must match recomputed Pareto frontier") in messages
    assert (
        "projection_frontiers.latency_vs_quality",
        "must match recomputed projection frontier",
    ) in messages
    assert (
        "constrained_frontiers.entries[0].frontier",
        "must match recomputed constrained frontier",
    ) in messages
    assert (
        "constrained_frontiers.entries[0].eligible_systems",
        "must match recomputed constrained eligible systems",
    ) in messages
    assert ("domain_frontiers", "must match recomputed domain frontiers") in messages


def test_validate_frontier_report_rejects_broken_references_and_axes():
    frontier = build_frontier_report([
        _report("broken", task_success=1.0, costs=[0.01], latencies=[200]),
    ])
    frontier["frontier"] = ["missing"]
    frontier["projection_frontiers"]["latency_vs_quality"] = ["missing"]
    frontier["domain_frontiers"]["all"] = {
        "frontier": ["missing"],
        "projection_frontiers": {"latency_vs_quality": [], "cost_vs_quality": []},
        "eligible_systems": ["broken"],
        "represented_systems": ["broken"],
        "num_eligible_systems": 1,
    }
    frontier["systems"]["broken"]["axes"]["p95_v2v_ttfb_ms"] = None
    frontier["scorecards"]["broken"].pop("p90_v2v_ttfb_ms")
    frontier["scorecards"]["broken"].pop("axis_confidence_intervals")

    issues = validate_frontier_report(frontier)
    messages = {(issue.path, issue.message) for issue in issues}

    assert ("frontier[0]", "unknown system name") in messages
    assert ("projection_frontiers.latency_vs_quality[0]", "unknown system name") in messages
    assert ("domain_frontiers.all.frontier[0]", "unknown system name") in messages
    assert (
        "systems.broken.axes.p95_v2v_ttfb_ms",
        "required for frontier-eligible systems",
    ) in messages
    assert (
        "scorecards.broken.p90_v2v_ttfb_ms",
        "missing required field",
    ) in messages
    assert (
        "scorecards.broken.axis_confidence_intervals",
        "missing required field",
    ) in messages


def test_validate_frontier_report_rejects_incoherent_axis_intervals():
    frontier = build_frontier_report([
        _report("intervals", task_success=1.0, costs=[0.01, 0.02], latencies=[200, 300]),
    ])
    intervals = frontier["scorecards"]["intervals"]["axis_confidence_intervals"]
    intervals["p95_v2v_ttfb_ms"].pop("method")
    intervals["p95_v2v_ttfb_ms"]["low"] = 400.0
    intervals["p95_v2v_ttfb_ms"]["high"] = 300.0
    intervals["cost_usd_per_successful_conversation"]["estimate"] = -0.01
    intervals["cost_usd_per_successful_conversation"]["confidence"] = 1.0
    intervals["task_success_rate"]["high"] = 1.2

    messages = {(issue.path, issue.message) for issue in validate_frontier_report(frontier)}

    assert (
        "scorecards.intervals.axis_confidence_intervals.p95_v2v_ttfb_ms.method",
        "missing required field",
    ) in messages
    assert (
        "scorecards.intervals.axis_confidence_intervals.p95_v2v_ttfb_ms",
        "low must be less than or equal to high",
    ) in messages
    assert (
        "scorecards.intervals.axis_confidence_intervals.cost_usd_per_successful_conversation.estimate",
        "must be nonnegative",
    ) in messages
    assert (
        "scorecards.intervals.axis_confidence_intervals.cost_usd_per_successful_conversation.confidence",
        "must be greater than 0.0 and less than 1.0",
    ) in messages
    assert (
        "scorecards.intervals.axis_confidence_intervals.task_success_rate.high",
        "must be between 0.0 and 1.0",
    ) in messages


def test_validate_frontier_report_rejects_incoherent_load_counts():
    frontier = build_frontier_report([
        _report("load-counts", task_success=1.0, costs=[0.01], latencies=[200]),
    ])
    load = frontier["scorecards"]["load-counts"]["latency_load"]["levels"]
    load["10"] = {
        "target_concurrency": 10,
        "sample_count": 12,
        "saturated": True,
        "p95_v2v_ttfb_ms": 200.0,
        "requested_calls": 10,
        "completed_calls": 11,
        "error_calls": 12,
        "peak_active_calls": 11,
    }

    messages = {(issue.path, issue.message) for issue in validate_frontier_report(frontier)}

    assert (
        "scorecards.load-counts.latency_load.levels.10.completed_calls",
        "must not exceed requested_calls",
    ) in messages
    assert (
        "scorecards.load-counts.latency_load.levels.10.error_calls",
        "must not exceed requested_calls",
    ) in messages
    assert (
        "scorecards.load-counts.latency_load.levels.10.sample_count",
        "must not exceed completed_calls",
    ) in messages
    assert (
        "scorecards.load-counts.latency_load.levels.10.peak_active_calls",
        "must not exceed target_concurrency",
    ) in messages
