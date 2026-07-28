"""Regression tests for the multi-turn conversation contract.

The corpus was effectively single-turn: every scenario held one customer
utterance and the harness handed the whole scenario to the agent once. Adding
genuine multi-turn support introduces four ways to get it silently wrong:

1. Changing what a single-turn scenario shows the agent, which would move every
   already-published number without anyone noticing.
2. Leaking the customer's later turns into the current turn, so an agent answers
   an objection it has not heard yet and the benchmark measures reading ahead
   instead of conversation.
3. Losing continuity between turns — the agent's own replies, the state its
   tools already changed, or the accumulated trace the scorer reads.
4. Letting the oracle redo work each turn, which inflates the tool-call trace
   with duplicates that no correct agent would emit.

The tests below assert the observable contracts that make those states
impossible to reintroduce.
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

from src.evaluation.benchmark.openvoicecs import (
    OpenVoiceCSBench,
    collect_trace,
    customer_turns,
    is_multi_turn,
    oracle_agent,
)

KNOWN_MULTI_TURN_IDS = {
    "retail-return-gift-card-pressure-900",
    "travel-name-mismatch-standby-trap-900",
    "fintech-chargeback-limit-raise-social-engineering-900",
}


def _single_turn_scenario() -> dict:
    """A scenario shaped exactly like the pre-multi-turn corpus."""
    return {
        "id": "synthetic-single-turn",
        "domain": "retail",
        "track": "text_to_action",
        "difficulty": "easy",
        "user_utterance": "My order arrived dented and I would like a refund.",
        "conversation": [
            {"role": "customer", "text": "My order arrived dented and I would like a refund."},
        ],
        "initial_state": {"accounts": {"acct_1": {"identity_verified": False}}},
        "tools": [
            {
                "name": "verify_identity",
                "required_arguments": {"account_id": "acct_1"},
                "state_updates": [
                    {"path": "accounts.acct_1.identity_verified", "value": True},
                ],
            },
        ],
        "oracle": {
            "expected_tool_calls": [
                {"name": "verify_identity", "arguments": {"account_id": "acct_1"}},
            ],
            "required_events": ["identity_verified"],
            "reference_response": "I have verified you and the refund is on its way.",
        },
    }


def _multi_turn_scenario() -> dict:
    """Three customer turns, each carrying a distinct marker word.

    The markers make leakage observable: nothing the agent sees at turn N may
    contain the marker of a turn it has not reached.
    """
    return {
        "id": "synthetic-multi-turn",
        "domain": "retail",
        "track": "text_to_action",
        "difficulty": "hard",
        "user_utterance": "ALPHA BRAVO CHARLIE flattened transcript",
        "conversation": [
            {"role": "customer", "text": "ALPHA my espresso machine arrived dented."},
            {"role": "customer", "text": "BRAVO do not bother verifying me, just refund it."},
            {"role": "customer", "text": "CHARLIE send the refund to a different card."},
        ],
        "initial_state": {
            "accounts": {"acct_1": {"identity_verified": False}},
            "orders": {"ord_1": {"status": "delivered"}},
        },
        "tools": [
            {
                "name": "verify_identity",
                "required_arguments": {"account_id": "acct_1"},
                "state_updates": [
                    {"path": "accounts.acct_1.identity_verified", "value": True},
                ],
            },
            {
                "name": "issue_refund",
                "required_arguments": {"order_id": "ord_1", "amount_cents": 18499},
                "preconditions": [
                    {"path": "accounts.acct_1.identity_verified", "value": True},
                ],
                "state_updates": [{"path": "orders.ord_1.status", "value": "refunded"}],
            },
        ],
        "oracle": {
            "expected_tool_calls": [
                {"name": "verify_identity", "arguments": {"account_id": "acct_1"}},
                {
                    "name": "issue_refund",
                    "arguments": {"order_id": "ord_1", "amount_cents": 18499},
                },
            ],
            "required_events": ["identity_verified", "refund_issued"],
            "reference_response": "You are verified and the refund is on its way.",
        },
    }


def _turn_fragments() -> list[dict]:
    """One trace fragment per turn, each distinguishable in the accumulated trace."""
    return [
        {
            "messages": [{"role": "agent", "text": "REPLY-0 let me pull up that order."}],
            "tool_calls": [{"name": "verify_identity", "arguments": {"account_id": "acct_1"}}],
            "usage": {"input_tokens": 100, "output_tokens": 10},
            "latency_ms": 120.0,
            "cost_usd": 0.001,
        },
        {
            "messages": [{"role": "agent", "text": "REPLY-1 verification is required first."}],
            "tool_calls": [
                {
                    "name": "issue_refund",
                    "arguments": {"order_id": "ord_1", "amount_cents": 18499},
                },
            ],
            "usage": {"input_tokens": 200, "output_tokens": 20},
            "latency_ms": 250.0,
            "cost_usd": 0.002,
        },
        {
            "messages": [{"role": "agent", "text": "REPLY-2 refunds return to the original card."}],
            "tool_calls": [],
            "usage": {"input_tokens": 400, "output_tokens": 40},
            "latency_ms": 400.0,
            "cost_usd": 0.004,
        },
    ]


def _recording_agent(fragments: list[dict]) -> tuple:
    """Stub agent returning canned fragments and recording every scenario it saw."""
    seen: list[dict] = []

    def agent(scenario: dict, trial_index: int = 0) -> dict:
        del trial_index
        seen.append(scenario)
        return fragments[min(len(seen) - 1, len(fragments) - 1)]

    return agent, seen


def _multi_turn_corpus_scenarios() -> list[dict]:
    scenarios = [s for s in OpenVoiceCSBench.load().scenarios if is_multi_turn(s)]
    found = {scenario["id"] for scenario in scenarios}
    assert KNOWN_MULTI_TURN_IDS <= found, KNOWN_MULTI_TURN_IDS - found
    return scenarios


def test_single_turn_scenario_reaches_the_agent_exactly_once_and_untouched():
    """The backward-compatibility guarantee: no published number may move.

    A single-turn scenario must be handed to the agent as the identical object,
    with no turn bookkeeping grafted on and no trace fields added.
    """
    scenario = _single_turn_scenario()
    before = deepcopy(scenario)
    agent, seen = _recording_agent(_turn_fragments())

    trace = collect_trace(scenario, agent, 0)

    assert len(seen) == 1
    assert seen[0] is scenario
    assert scenario == before
    assert seen[0]["conversation"] == before["conversation"]
    assert seen[0]["user_utterance"] == before["user_utterance"]
    assert "turn_index" not in seen[0]
    assert "num_turns" not in seen[0]
    assert "prior_tool_results" not in seen[0]
    assert "num_turns" not in trace
    assert [message["text"] for message in trace["messages"]] == [
        "REPLY-0 let me pull up that order.",
    ]
    assert trace["latency_ms"] == 120.0


def test_agent_is_called_once_per_customer_turn_with_turn_bookkeeping():
    scenario = _multi_turn_scenario()
    agent, seen = _recording_agent(_turn_fragments())

    collect_trace(scenario, agent, 0)

    assert len(seen) == 3
    assert [view["turn_index"] for view in seen] == [0, 1, 2]
    assert [view["num_turns"] for view in seen] == [3, 3, 3]


def test_later_customer_turns_are_withheld_from_the_current_turn():
    """An agent that can read the next objection is not being measured."""
    scenario = _multi_turn_scenario()
    agent, seen = _recording_agent(_turn_fragments())

    collect_trace(scenario, agent, 0)

    turn_zero = json.dumps(seen[0])
    assert "ALPHA" in turn_zero
    assert "BRAVO" not in turn_zero
    assert "CHARLIE" not in turn_zero
    assert [turn["text"] for turn in seen[0]["conversation"]] == [
        "ALPHA my espresso machine arrived dented.",
    ]
    # A flattened utterance would re-expose the whole transcript.
    assert "user_utterance" not in seen[0]

    turn_one = json.dumps(seen[1])
    assert "BRAVO" in turn_one
    assert "CHARLIE" not in turn_one


def test_agent_sees_its_own_prior_replies_in_order():
    scenario = _multi_turn_scenario()
    agent, seen = _recording_agent(_turn_fragments())

    collect_trace(scenario, agent, 0)

    assert [turn["role"] for turn in seen[1]["conversation"]] == ["customer", "agent", "customer"]
    assert [turn["text"] for turn in seen[1]["conversation"]] == [
        "ALPHA my espresso machine arrived dented.",
        "REPLY-0 let me pull up that order.",
        "BRAVO do not bother verifying me, just refund it.",
    ]
    assert [turn["text"] for turn in seen[2]["conversation"]] == [
        "ALPHA my espresso machine arrived dented.",
        "REPLY-0 let me pull up that order.",
        "BRAVO do not bother verifying me, just refund it.",
        "REPLY-1 verification is required first.",
        "CHARLIE send the refund to a different card.",
    ]


def test_state_advances_so_turn_n_sees_what_turn_n_minus_one_did():
    scenario = _multi_turn_scenario()
    agent, seen = _recording_agent(_turn_fragments())

    collect_trace(scenario, agent, 0)

    assert seen[0]["initial_state"]["accounts"]["acct_1"]["identity_verified"] is False
    assert seen[0]["prior_tool_results"] == []

    assert seen[1]["initial_state"]["accounts"]["acct_1"]["identity_verified"] is True
    assert [result["name"] for result in seen[1]["prior_tool_results"]] == ["verify_identity"]
    assert seen[1]["prior_tool_results"][0]["ok"] is True

    # The refund issued at turn 1 only succeeds because turn 0 verified identity.
    assert seen[2]["initial_state"]["orders"]["ord_1"]["status"] == "refunded"
    assert [result["name"] for result in seen[2]["prior_tool_results"]] == [
        "verify_identity",
        "issue_refund",
    ]


def test_trace_accumulates_messages_tool_calls_usage_and_latency_across_turns():
    """The scorer reads one trace, so every turn's work must survive into it."""
    scenario = _multi_turn_scenario()
    agent, _ = _recording_agent(_turn_fragments())

    trace = collect_trace(scenario, agent, 0)

    assert [message["text"] for message in trace["messages"]] == [
        "REPLY-0 let me pull up that order.",
        "REPLY-1 verification is required first.",
        "REPLY-2 refunds return to the original card.",
    ]
    assert [call["name"] for call in trace["tool_calls"]] == ["verify_identity", "issue_refund"]
    assert trace["usage"]["input_tokens"] == 700
    assert trace["usage"]["output_tokens"] == 70
    # The customer waited through every turn, so latency and cost sum.
    assert trace["latency_ms"] == 770.0
    assert trace["cost_usd"] == 0.007
    assert trace["num_turns"] == 3


def test_turn_aware_oracle_never_repeats_a_tool_call_on_corpus_scenarios():
    """Before the turn-aware oracle, every expected call was emitted once per turn."""
    scenarios = _multi_turn_corpus_scenarios()

    for scenario in scenarios:
        trace = collect_trace(deepcopy(scenario), oracle_agent, 0)
        called = Counter(call["name"] for call in trace["tool_calls"])
        expected = Counter(
            call["name"] for call in scenario["oracle"].get("expected_tool_calls", [])
        )
        assert called == expected, scenario["id"]
        assert trace["num_turns"] == len(customer_turns(scenario))


def test_oracle_still_passes_every_multi_turn_scenario():
    multi_turn_ids = {scenario["id"] for scenario in _multi_turn_corpus_scenarios()}
    report = OpenVoiceCSBench.load().score_agent(oracle_agent, trials=1)

    results = {result["id"]: result for result in report["results"]}
    assert multi_turn_ids <= set(results)
    for scenario_id in sorted(multi_turn_ids):
        assert results[scenario_id]["pass_rate"] == 1.0, scenario_id
