"""Tests for paired OpenVoiceCS report comparison."""

from __future__ import annotations

import pytest

from src.evaluation.benchmark.comparison import compare_reports
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, no_op_agent, oracle_agent


def test_compare_reports_uses_matched_scenarios_and_paired_ci():
    bench = OpenVoiceCSBench.load()
    baseline = bench.score_agent(
        no_op_agent,
        max_scenarios=3,
        trials=1,
        model_metadata={"agent": "noop"},
    )
    candidate = bench.score_agent(
        oracle_agent,
        max_scenarios=3,
        trials=1,
        model_metadata={"agent": "oracle"},
    )

    comparison = compare_reports(baseline, candidate, iterations=500, seed=7)
    summary = comparison["summary"]

    assert comparison["baseline"] == "noop"
    assert comparison["candidate"] == "oracle"
    assert comparison["matched_scenarios"]["count"] == 3
    assert summary["overall_score_delta"] > 0
    assert summary["mean_paired_scenario_score_delta"]["ci_low"] > 0
    assert summary["interpretation"] == "candidate_higher_with_ci_excluding_zero"
    assert summary["mcnemar_exact"]["candidate_wins"] == 3
    assert summary["mcnemar_exact"]["baseline_wins"] == 0
    assert summary["mcnemar_exact"]["p_value"] == 0.25
    assert comparison["metric_deltas"]["task_success"]["mean"] > 0
    assert comparison["stratified_deltas"]["domain"]["scenario_score_delta"]["n"] == 3
    assert comparison["stratified_deltas"]["domain"]["scenario_score_delta"]["num_strata"] == 3
    assert comparison["slice_deltas"]["domain"]["retail"]["count"] == 1
    assert comparison["slice_deltas"]["domain"]["retail"]["mcnemar_exact"]["candidate_wins"] == 1


def test_compare_reports_rejects_without_common_scenarios():
    baseline = {
        "model_metadata": {"agent": "a"},
        "results": [{"id": "only-a", "avg_scores": {}, "pass_k": True, "pass_rate": 1.0}],
    }
    candidate = {
        "model_metadata": {"agent": "b"},
        "results": [{"id": "only-b", "avg_scores": {}, "pass_k": True, "pass_rate": 1.0}],
    }

    with pytest.raises(ValueError, match="no scenario IDs in common"):
        compare_reports(baseline, candidate)


def test_compare_reports_flags_matched_scenario_metadata_mismatches():
    baseline = {
        "model_metadata": {"agent": "a"},
        "results": [
            {
                "id": "shared",
                "domain": "retail",
                "track": "text_to_action",
                "difficulty": "easy",
                "avg_scores": {},
                "pass_k": True,
                "pass_rate": 1.0,
            }
        ],
    }
    candidate = {
        "model_metadata": {"agent": "b"},
        "results": [
            {
                "id": "shared",
                "domain": "travel",
                "track": "text_to_action",
                "difficulty": "easy",
                "avg_scores": {},
                "pass_k": True,
                "pass_rate": 1.0,
            }
        ],
    }

    comparison = compare_reports(baseline, candidate, iterations=10)

    assert comparison["matched_scenarios"]["metadata_mismatches"] == [
        {
            "id": "shared",
            "field": "domain",
            "baseline": "retail",
            "candidate": "travel",
        }
    ]
