"""Tests for OpenVoiceCS subjective judge aggregation."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src.evaluation.benchmark.judging import (
    ModelJudgeSpec,
    apply_judge_report,
    apply_judge_report_from_files,
    build_judge_report,
    build_judge_report_from_files,
    generate_model_judge_annotations,
    iter_blinded_judge_items,
    judge_annotation_package_stats,
    judge_study_stats,
    load_judge_annotation_package,
    load_judge_protocol,
    load_judge_rubric,
    load_judge_study_manifest,
    parse_model_judge_spec,
    validate_judge_annotation_package,
    validate_judge_annotation_package_file,
    validate_judge_annotations,
    validate_judge_protocol,
    validate_judge_protocol_file,
    validate_judge_report,
    validate_judge_report_file,
    validate_judge_rubric,
    validate_judge_rubric_file,
    validate_judge_study_manifest,
    validate_judge_study_manifest_file,
)
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, oracle_agent, validate_report


def _annotations(scenario_id: str) -> list[dict]:
    return [
        {
            "item_id": scenario_id,
            "scenario_id": scenario_id,
            "rater_id": "rater-a",
            "scores": {
                "empathy": 4,
                "clarity": 5,
                "naturalness": 4,
                "professionalism": 5,
                "resolution_communication": 5,
                "channel_fit": 4,
            },
        },
        {
            "item_id": scenario_id,
            "scenario_id": scenario_id,
            "rater_id": "rater-b",
            "scores": {
                "empathy": 5,
                "clarity": 5,
                "naturalness": 4,
                "professionalism": 5,
                "resolution_communication": 4,
                "channel_fit": 4,
            },
        },
    ]


def _matching_annotations(scenario_id: str) -> list[dict]:
    rows = _annotations(scenario_id)
    rows[1]["scores"] = dict(rows[0]["scores"])
    return rows


def test_judge_rubric_validates():
    rubric = load_judge_rubric()

    assert validate_judge_rubric(rubric) == []
    assert validate_judge_rubric_file() == []
    # Production validation uses a 1e-6 tolerance (judging.py); exact equality here is
    # version-dependent because CPython 3.12+ uses compensated summation in sum().
    assert sum(dimension["weight"] for dimension in rubric["dimensions"]) == pytest.approx(
        1.0, abs=1e-6
    )


def test_judge_protocol_validates():
    protocol = load_judge_protocol()

    assert validate_judge_protocol(protocol) == []
    assert validate_judge_protocol_file() == []
    assert protocol["minimum_raters_per_item"] == 2
    assert protocol["annotation_mode"] == "reference_fixture"


def test_judge_protocol_rejects_weak_process_controls():
    protocol = load_judge_protocol()
    protocol["minimum_raters_per_item"] = 1
    protocol["minimum_alpha_for_release"] = 1.5
    protocol["blinding"]["hide_system_identity"] = False
    protocol["quality_controls"]["gold_items_fraction"] = 0.0

    messages = {
        (issue.path, issue.message)
        for issue in validate_judge_protocol(protocol)
    }

    assert ("minimum_raters_per_item", "must be an integer >= 2") in messages
    assert ("minimum_alpha_for_release", "must be a number between 0 and 1") in messages
    assert ("blinding.hide_system_identity", "must be true") in messages
    assert (
        "quality_controls.gold_items_fraction",
        "must be a number between 0.01 and 1.0",
    ) in messages


def test_judge_study_manifest_validates():
    study = load_judge_study_manifest()

    assert validate_judge_study_manifest(study) == []
    assert validate_judge_study_manifest_file() == []
    stats = judge_study_stats(study)
    assert stats["status"] == "reference_fixture"
    assert stats["official_judging_eligible"] is False
    assert stats["num_raters"] == 2
    assert stats["by_rater_type"] == {"reference_fixture": 2}


def test_judge_study_manifest_rejects_hash_mismatch_and_weak_design():
    study = load_judge_study_manifest()
    study["protocol"]["sha256"] = "0" * 64
    study["study_design"]["minimum_items_per_system"] = 10
    study["blinding"]["hide_expected_actions"] = False
    study["audit"]["retain_raw_annotations"] = False

    messages = {
        (issue.path, issue.message)
        for issue in validate_judge_study_manifest(study)
    }

    assert ("protocol.sha256", "does not match file contents") in messages
    assert (
        "study_design.minimum_items_per_system",
        "must be an integer >= 30",
    ) in messages
    assert ("blinding.hide_expected_actions", "must be true") in messages
    assert ("audit.retain_raw_annotations", "must be true") in messages


def test_judge_study_manifest_rejects_official_reference_fixture():
    study = deepcopy(load_judge_study_manifest())
    study["status"] = "completed"
    study["official_judging_eligible"] = True

    messages = {
        (issue.path, issue.message)
        for issue in validate_judge_study_manifest(study)
    }

    assert (
        "rater_pool.raters[0].type",
        "official judge studies cannot use reference_fixture raters",
    ) in messages
    assert (
        "annotation_package.official_judging",
        "official judge studies require official annotation package evidence",
    ) in messages
    assert (
        "annotation_package.annotation_mode",
        "official judge studies cannot use reference_fixture annotations",
    ) in messages


def test_judge_annotation_package_validates():
    package = load_judge_annotation_package()

    assert validate_judge_annotation_package(package) == []
    assert validate_judge_annotation_package_file() == []
    stats = judge_annotation_package_stats(package)
    assert stats["num_packages"] == 2
    assert stats["num_annotations"] == 5520
    assert stats["official_judging"] is False


def test_judge_annotation_package_rejects_official_fixture_and_hash_mismatch():
    package = load_judge_annotation_package()
    package["official_judging"] = True
    package["packages"][0]["annotations"]["sha256"] = "0" * 64
    package["packages"][0]["blinding"]["hide_system_identity"] = False

    messages = {
        (issue.path, issue.message)
        for issue in validate_judge_annotation_package(package)
    }

    assert (
        "annotation_mode",
        "official judging cannot use reference_fixture annotations",
    ) in messages
    assert ("packages[0].annotations.sha256", "does not match file contents") in messages
    assert ("packages[0].blinding.hide_system_identity", "must be true") in messages


def test_build_judge_report_aggregates_scores_and_agreement():
    bench = OpenVoiceCSBench.load()
    report = bench.score_agent(oracle_agent, max_scenarios=1, trials=1)
    scenario_id = report["results"][0]["id"]
    rubric = load_judge_rubric()

    judge_report = build_judge_report(report, _annotations(scenario_id), rubric)

    assert judge_report["num_annotations"] == 2
    assert judge_report["num_items"] == 1
    assert judge_report["coverage"]["items_meeting_minimum_raters"] == 1
    assert judge_report["overall_subjective_score"] > 0.8
    assert judge_report["dimension_scores"]["clarity"] == 5.0
    assert judge_report["agreement"]["method"] == "krippendorff_alpha_interval"
    assert judge_report["agreement"]["by_dimension"]["clarity"] == 1.0


def test_validate_judge_report_accepts_release_quality_report(tmp_path):
    bench = OpenVoiceCSBench.load()
    report = bench.score_agent(oracle_agent, max_scenarios=1, trials=1)
    scenario_id = report["results"][0]["id"]
    judge_report = build_judge_report(
        report,
        _matching_annotations(scenario_id),
        load_judge_rubric(),
    )
    judge_path = tmp_path / "judge_report.json"
    judge_path.write_text(json.dumps(judge_report), encoding="utf-8")

    assert validate_judge_report(judge_report) == []
    assert validate_judge_report_file(judge_path) == []


def test_validate_judge_report_rejects_low_coverage_and_agreement():
    bench = OpenVoiceCSBench.load()
    report = bench.score_agent(oracle_agent, max_scenarios=1, trials=1)
    scenario_id = report["results"][0]["id"]
    low_coverage = build_judge_report(
        report,
        _annotations(scenario_id)[:1],
        load_judge_rubric(),
    )
    low_agreement = build_judge_report(report, _annotations(scenario_id), load_judge_rubric())

    coverage_messages = {
        (issue.path, issue.message)
        for issue in validate_judge_report(low_coverage)
    }
    agreement_messages = {
        (issue.path, issue.message)
        for issue in validate_judge_report(low_agreement)
    }

    assert (
        "coverage.items_below_minimum_raters",
        "all judged items must meet minimum rater coverage for release",
    ) in coverage_messages
    assert ("agreement.overall", "below minimum_alpha_for_release") in agreement_messages


def test_judge_report_from_jsonl_files(tmp_path):
    report = OpenVoiceCSBench.load().score_agent(oracle_agent, max_scenarios=1, trials=1)
    scenario_id = report["results"][0]["id"]
    report_path = tmp_path / "report.json"
    annotations_path = tmp_path / "annotations.jsonl"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    annotations_path.write_text(
        "\n".join(json.dumps(row) for row in _annotations(scenario_id)),
        encoding="utf-8",
    )

    judge_report = build_judge_report_from_files(report_path, annotations_path)

    assert judge_report["num_raters"] == 2
    assert judge_report["items"][0]["scenario_id"] == scenario_id


def test_apply_judge_report_adds_frontier_ready_experience(tmp_path):
    report = OpenVoiceCSBench.load().score_agent(oracle_agent, max_scenarios=1, trials=1)
    scenario_id = report["results"][0]["id"]
    judge_report = build_judge_report(report, _annotations(scenario_id), load_judge_rubric())

    judged = apply_judge_report(report, judge_report)

    assert judged["conversation_experience_score"] == judge_report["overall_subjective_score"]
    assert judged["conversation_experience"]["coverage"] == 1.0
    trial = judged["results"][0]["trials"][0]
    assert trial["experience_judgment"]["judge"]["type"] == "offline_aggregate"
    assert trial["experience_judgment"]["dimensions"]["clarity"]["score"] == 5.0
    assert validate_report(judged) == []

    report_path = tmp_path / "report.json"
    judge_path = tmp_path / "judge_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    judge_path.write_text(json.dumps(judge_report), encoding="utf-8")
    loaded = apply_judge_report_from_files(report_path, judge_path)

    assert loaded["conversation_experience"]["num_judged_trials"] == 1
    assert validate_report(loaded) == []


def test_validate_judge_annotations_catches_bad_scores():
    report = OpenVoiceCSBench.load().score_agent(oracle_agent, max_scenarios=1, trials=1)
    rubric = load_judge_rubric()
    bad = [{
        "item_id": "bad-item",
        "scenario_id": "missing",
        "rater_id": "rater-a",
        "scores": {"empathy": 6},
    }]

    messages = {
        (issue.path, issue.message)
        for issue in validate_judge_annotations(bad, rubric, report)
    }

    assert ("annotations[0].scenario_id", "unknown scenario id") in messages
    assert ("annotations[0].scores.clarity", "missing score") in messages
    assert ("annotations[0].scores.empathy", "must be between 1 and 5") in messages


def test_parse_model_judge_spec_accepts_openrouter_model():
    spec = parse_model_judge_spec("openrouter:anthropic/claude-sonnet-4.6")

    assert spec.provider == "openrouter"
    assert spec.model_id == "anthropic/claude-sonnet-4.6"


def test_model_judge_annotations_blind_items_and_aggregate():
    report = OpenVoiceCSBench.load().score_agent(oracle_agent, max_scenarios=1, trials=1)
    rubric = load_judge_rubric()
    dimensions = [dimension["id"] for dimension in rubric["dimensions"]]
    seen_payloads = []

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, max_output_tokens, temperature, timeout_seconds
        payload = json.loads(messages[1]["content"])
        seen_payloads.append(payload)
        item = payload["blinded_item"]
        assert "model_metadata" not in item
        assert "scores" not in item
        assert "tool_calls" not in item
        assert "tool_check" not in item
        assert item["messages"]
        return json.dumps({
            "scores": {dimension: 5 for dimension in dimensions},
            "notes": "clear and natural",
        })

    annotations = generate_model_judge_annotations(
        report,
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        rubric=rubric,
        prompt="Judge this transcript.",
        caller=caller,
    )

    assert len(annotations) == 2
    assert {row["rater_id"] for row in annotations} == {
        "model-judge-openrouter-judge-a",
        "model-judge-openrouter-judge-b",
    }
    assert len(seen_payloads) == 2
    judge_report = build_judge_report(report, annotations, rubric)
    assert validate_judge_report(judge_report) == []


def test_model_judge_annotations_calls_adjudicator_on_large_disagreement():
    report = OpenVoiceCSBench.load().score_agent(oracle_agent, max_scenarios=1, trials=1)
    rubric = load_judge_rubric()
    dimensions = [dimension["id"] for dimension in rubric["dimensions"]]
    calls = []

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del messages, max_output_tokens, temperature, timeout_seconds
        calls.append(spec.model_id)
        score = {"judge-a": 1, "judge-b": 5}.get(spec.model_id, 3)
        return json.dumps({"scores": {dimension: score for dimension in dimensions}})

    annotations = generate_model_judge_annotations(
        report,
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        rubric=rubric,
        prompt="Judge this transcript.",
        adjudicator=ModelJudgeSpec(provider="openrouter", model_id="judge-c"),
        disagreement_threshold=2,
        caller=caller,
    )

    assert calls == ["judge-a", "judge-b", "judge-c"]
    assert len(annotations) == 3
    assert annotations[-1]["judge"]["adjudicator"] is True


def test_model_judge_annotations_rejects_bad_model_score():
    report = OpenVoiceCSBench.load().score_agent(oracle_agent, max_scenarios=1, trials=1)
    rubric = load_judge_rubric()
    dimensions = [dimension["id"] for dimension in rubric["dimensions"]]

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, messages, max_output_tokens, temperature, timeout_seconds
        return json.dumps({"scores": {dimension: 9 for dimension in dimensions}})

    with pytest.raises(ValueError, match="must be between 1 and 5"):
        generate_model_judge_annotations(
            report,
            judge_specs=[ModelJudgeSpec(provider="openrouter", model_id="judge-a")],
            rubric=rubric,
            prompt="Judge this transcript.",
            caller=caller,
        )


def test_iter_blinded_judge_items_uses_trial_item_ids():
    report = OpenVoiceCSBench.load().score_agent(oracle_agent, max_scenarios=1, trials=1)

    items = iter_blinded_judge_items(report)

    assert items[0]["item_id"] == f"{report['results'][0]['id']}:0"
    assert items[0]["scenario_id"] == report["results"][0]["id"]
    assert "messages" in items[0]
    assert "model_metadata" not in items[0]
