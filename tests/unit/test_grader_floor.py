"""Score-floor gates.

A degenerate agent's score is the benchmark's price of admission. These pin the
mechanisms that let it climb, so a change that makes the corpus easier to game
fails here rather than showing up as an inflated leaderboard.
"""

from __future__ import annotations

import json

from src.evaluation.benchmark.grader_floor import (
    DEGENERATE_AGENTS,
    agent_call_every_tool,
    agent_parrot,
    model_visible_view,
)
from src.evaluation.benchmark.grader_probe import load_scenarios
from src.evaluation.benchmark.openvoicecs import score_trace

SCENARIO_ID = "retail-refund-damaged-item-001"


def test_degenerate_agents_see_only_the_model_view():
    """A floor agent must not be able to read anything the model cannot.

    The oracle's expected calls, expected state, and reference response are the
    answers; if any leaked into the prompt the floor would be meaningless.
    """
    scenario = load_scenarios(scenario_ids=[SCENARIO_ID])[0]
    view = model_visible_view(scenario)
    blob = json.dumps(view)
    oracle = scenario["oracle"]
    assert "oracle" not in view
    assert oracle["reference_response"] not in blob
    for call in oracle["expected_tool_calls"]:
        for value in (call.get("arguments") or {}).values():
            if isinstance(value, str) and value.startswith(("case_", "ticket_", "alert_")):
                assert value not in blob, f"system-assigned {value} leaked into the prompt"


def test_call_every_tool_needs_no_oracle_access():
    scenario = load_scenarios(scenario_ids=[SCENARIO_ID])[0]
    stripped = dict(scenario)
    stripped["oracle"] = {
        **scenario["oracle"], "expected_tool_calls": [], "reference_response": "",
    }
    calls = agent_call_every_tool(stripped)["tool_calls"]
    assert [c["name"] for c in calls] == [t["name"] for t in scenario["tools"]]
    assert any(c["arguments"] for c in calls), "arguments should be scraped from the prompt"


def test_parrot_response_contains_no_oracle_text():
    scenario = load_scenarios(scenario_ids=[SCENARIO_ID])[0]
    text = agent_parrot(scenario)["messages"][0]["text"]
    assert scenario["oracle"]["reference_response"] not in text


def test_corpus_offers_distractor_tools():
    """Every declared tool being part of the expected flow makes the tool list an
    answer key: "call everything" and "call the right things" coincide.
    """
    scenarios = load_scenarios()
    without = [
        scenario["id"] for scenario in scenarios
        if not (
            {tool["name"] for tool in scenario.get("tools") or []}
            - {c["name"] for c in scenario["oracle"].get("expected_tool_calls") or []}
            - {c["name"] for c in scenario["oracle"].get("forbidden_tool_calls") or [] if c.get("name")}
        )
    ]
    # 220/220 today. Tightening this bound is the corpus-side fix for the floor.
    assert len(without) <= 220, f"{len(without)} scenarios offer no distractor tool"


def test_no_degenerate_agent_beats_a_perfect_score():
    """Sanity bound: the floor must at least stay under the oracle."""
    scenario = load_scenarios(scenario_ids=[SCENARIO_ID])[0]
    for agent_fn in DEGENERATE_AGENTS.values():
        scored = score_trace(scenario, agent_fn(scenario))
        assert scored["scores"]["task_success"] <= 1.0


def test_privacy_is_not_earned_by_silence():
    """Minimization must not be inferred from having said nothing at all.

    An agent that produces no text discloses nothing, so the forbidden half of
    the privacy check passes trivially. Deriving the required
    `pii_minimization` event from that same fact made the metric unable to
    distinguish minimization from silence.
    """
    scenario = load_scenarios(scenario_ids=[SCENARIO_ID])[0]
    silent = {"messages": [{"role": "agent", "text": ""}], "tool_calls": [], "events": []}
    scored = score_trace(scenario, silent)
    assert "pii_minimization" not in scored["derived_events"]
    assert scored["scores"]["privacy"] < 1.0
