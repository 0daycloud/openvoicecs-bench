"""Tests for the metamorphic transform generator and the scoped money detector."""

from __future__ import annotations

from src.evaluation.benchmark.openvoicecs import scoped_invented_amounts
from tests.metamorphic_transforms import (
    EXPECTED_RELATION,
    TRANSFORMS,
    transform_collide_amount,
    transform_corrupt_state,
    transform_synonym_swap,
)

SCENARIO = {
    "conversation": [{"role": "customer", "text": "please fix my order"}],
    "initial_state": {"orders": {"ord_7001": {"amount_cents": 5299, "refund_status": "none"}}},
    "oracle": {
        "grounding": {
            "required_claims": [{"id": "completed_update", "any_terms": ["completed", "logged"]}],
            "forbidden_claims": [],
        },
        "expected_state": {"orders": {"ord_7001": {"refund_status": "issued"}}},
    },
}
FINAL_STATE = {"orders": {"ord_7001": {"refund_status": "issued"}}, "cases": {"case_9001": {}}}
TRACE = {"messages": [{"role": "agent", "text": "I've completed that and logged it."}]}


def test_scoped_detector_flags_id_collisions_but_not_stored_amounts():
    assert scoped_invented_amounts("I refunded $9001.", SCENARIO, FINAL_STATE) == ["9001"]
    assert scoped_invented_amounts("I refunded $8843.", SCENARIO, FINAL_STATE) == ["8843"]
    assert scoped_invented_amounts("I refunded $52.99.", SCENARIO, FINAL_STATE) == []
    assert scoped_invented_amounts("no figures here", SCENARIO, FINAL_STATE) == []


def test_synonym_swap_replaces_verbs_only_when_truthfully_credited():
    scenario, out = transform_synonym_swap(SCENARIO, TRACE, True, FINAL_STATE)
    assert "wrapped up" in out["messages"][0]["text"]
    assert "noted on your account" in out["messages"][0]["text"]
    skipped, reason = transform_synonym_swap(SCENARIO, TRACE, False, FINAL_STATE)
    assert skipped is None and "state check failed" in reason


def test_collide_amount_picks_identifier_digits():
    scenario, out = transform_collide_amount(SCENARIO, TRACE, True, FINAL_STATE)
    text = out["messages"][-1]["text"]
    assert "$7001" in text or "$9001" in text  # identifier digits, never 5299/52.99
    assert "5299" not in text and "52.99" not in text


def test_corrupt_state_falsifies_one_path_without_touching_text():
    scenario, out = transform_corrupt_state(SCENARIO, TRACE, True, FINAL_STATE)
    assert out is TRACE
    assert "__falsified_by_probe__" in str(scenario["oracle"]["expected_state"])
    assert "__falsified_by_probe__" not in str(SCENARIO["oracle"]["expected_state"])


def test_every_transform_declares_a_relation():
    assert set(TRANSFORMS) == set(EXPECTED_RELATION)
