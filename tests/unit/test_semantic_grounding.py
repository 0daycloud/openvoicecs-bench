"""Tests for the opt-in semantic factual-grounding check."""

from __future__ import annotations

import json

import pytest

from src.evaluation.benchmark.judging import ModelJudgeSpec
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench
from src.evaluation.benchmark.semantic_grounding import (
    build_semantic_grounding_report,
    generate_semantic_grounding_annotations,
    iter_blinded_grounding_items,
    validate_semantic_grounding_report,
)


def _grounding_scenario() -> dict:
    return {
        "id": "grounding-test-001",
        "domain": "billing",
        "track": "text_to_action",
        "difficulty": "easy",
        "customer_goal": "Get the modem rental fee waived.",
        "conversation": [
            {"role": "customer", "text": "Can you waive the modem rental fee?"}
        ],
        "initial_state": {"accounts": {"acct_1": {}}},
        "tools": [],
        "oracle": {
            "expected_state": {},
            "grounding": {
                "required_claims": [
                    {
                        "id": "fee_waived",
                        "any_terms": ["no change fee", "no fee", "fee waiver"],
                    }
                ],
            },
        },
    }


def _agent_with_text(text: str):
    def agent_fn(scenario, trial_index):
        del scenario, trial_index
        return {"messages": [{"role": "agent", "text": text}]}

    return agent_fn


def test_iter_blinded_grounding_items_returns_one_item_per_required_claim():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text(
            "I'm sorry, I can't waive that fee myself, so I've escalated it to billing."
        ),
        trials=1,
    )

    items = iter_blinded_grounding_items(report, [scenario])

    assert len(items) == 1
    item = items[0]
    assert item["item_id"] == "grounding-test-001:0:fee_waived"
    assert item["scenario_id"] == "grounding-test-001"
    assert item["trial_index"] == 0
    assert item["claim_id"] == "fee_waived"
    assert item["claim_description"] == "no change fee / no fee / fee waiver"
    assert "escalated" in item["agent_text"]
    assert set(item) == {
        "item_id", "scenario_id", "trial_index", "claim_id",
        "claim_description", "agent_text",
    }


def test_iter_blinded_grounding_items_skips_scenarios_without_required_claims():
    scenario = _grounding_scenario()
    scenario["id"] = "no-claims-001"
    del scenario["oracle"]["grounding"]
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("Sure, done."), trials=1
    )

    assert iter_blinded_grounding_items(report, [scenario]) == []


def test_generate_semantic_grounding_annotations_blinds_payload_and_returns_verdict():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text(
            "I'm sorry, I can't waive that fee myself, so I've escalated it to billing."
        ),
        trials=1,
    )
    seen_payloads = []

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, max_output_tokens, temperature, timeout_seconds
        payload = json.loads(messages[1]["content"])
        seen_payloads.append(payload)
        assert set(payload) == {"claim", "agent_text"}
        return json.dumps({"verdict": "honest_alternative", "rationale": "truthful escalation"})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        caller=caller,
    )

    assert len(annotations) == 2
    assert len(seen_payloads) == 2
    assert {a["verdict"] for a in annotations} == {"honest_alternative"}
    assert {a["rater_id"] for a in annotations} == {
        "grounding-judge-openrouter-judge-a",
        "grounding-judge-openrouter-judge-b",
    }
    for annotation in annotations:
        assert annotation["judge"]["type"] == "audited_grounding_judge"
        assert annotation["judge"]["adjudicator"] is False
        assert annotation["claim_id"] == "fee_waived"


def test_generate_semantic_grounding_annotations_calls_adjudicator_on_disagreement():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("I've escalated this."), trials=1
    )
    calls = []

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del messages, max_output_tokens, temperature, timeout_seconds
        calls.append(spec.model_id)
        verdict = {"judge-a": "grounded", "judge-b": "not_grounded"}.get(
            spec.model_id, "honest_alternative"
        )
        return json.dumps({"verdict": verdict})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        adjudicator=ModelJudgeSpec(provider="openrouter", model_id="judge-c"),
        caller=caller,
    )

    assert calls == ["judge-a", "judge-b", "judge-c"]
    assert len(annotations) == 3
    assert annotations[-1]["judge"]["adjudicator"] is True


def test_generate_semantic_grounding_annotations_does_not_adjudicate_on_agreement():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("I've escalated this."), trials=1
    )
    calls = []

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del messages, max_output_tokens, temperature, timeout_seconds
        calls.append(spec.model_id)
        return json.dumps({"verdict": "honest_alternative"})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        adjudicator=ModelJudgeSpec(provider="openrouter", model_id="judge-c"),
        caller=caller,
    )

    assert calls == ["judge-a", "judge-b"]
    assert len(annotations) == 2


def test_generate_semantic_grounding_annotations_rejects_bad_verdict():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("Sure thing."), trials=1
    )

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, messages, max_output_tokens, temperature, timeout_seconds
        return json.dumps({"verdict": "maybe"})

    with pytest.raises(ValueError, match="must be one of"):
        generate_semantic_grounding_annotations(
            report,
            [scenario],
            judge_specs=[ModelJudgeSpec(provider="openrouter", model_id="judge-a")],
            caller=caller,
        )


def test_build_semantic_grounding_report_scores_honest_alternative_as_satisfied():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text(
            "I'm sorry, I can't waive that fee myself, so I've escalated it to billing."
        ),
        trials=1,
    )

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, messages, max_output_tokens, temperature, timeout_seconds
        return json.dumps({"verdict": "honest_alternative"})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        caller=caller,
    )
    grounding_report = build_semantic_grounding_report(report, annotations)

    assert validate_semantic_grounding_report(grounding_report) == []
    assert grounding_report["benchmark"] == "OpenVoiceCS-Bench Semantic Grounding Report"
    assert grounding_report["overall_semantic_grounding_score"] == 1.0
    assert grounding_report["verdict_breakdown"]["honest_alternative"] == 1
    assert grounding_report["verdict_breakdown"]["not_grounded"] == 0
    assert len(grounding_report["items"]) == 1
    assert grounding_report["items"][0]["required_score"] == 1.0


def test_build_semantic_grounding_report_scores_not_grounded_as_missing():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("Sure, I've updated your account."), trials=1
    )

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, messages, max_output_tokens, temperature, timeout_seconds
        return json.dumps({"verdict": "not_grounded", "rationale": "no mention of the fee"})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        caller=caller,
    )
    grounding_report = build_semantic_grounding_report(report, annotations)

    assert grounding_report["overall_semantic_grounding_score"] == 0.0
    assert grounding_report["items"][0]["required_score"] == 0.0


def test_build_semantic_grounding_report_uses_adjudicator_to_break_ties():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("I've escalated this."), trials=1
    )

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del messages, max_output_tokens, temperature, timeout_seconds
        verdict = {"judge-a": "grounded", "judge-b": "not_grounded"}.get(
            spec.model_id, "honest_alternative"
        )
        return json.dumps({"verdict": verdict})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        adjudicator=ModelJudgeSpec(provider="openrouter", model_id="judge-c"),
        caller=caller,
    )
    grounding_report = build_semantic_grounding_report(report, annotations)

    assert grounding_report["items"][0]["claims"][0]["verdict"] == "honest_alternative"
    assert grounding_report["items"][0]["required_score"] == 1.0
