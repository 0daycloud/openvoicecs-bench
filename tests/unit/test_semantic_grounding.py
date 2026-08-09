"""Tests for the opt-in semantic factual-grounding check."""

from __future__ import annotations

from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench
from src.evaluation.benchmark.semantic_grounding import iter_blinded_grounding_items


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
