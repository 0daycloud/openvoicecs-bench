"""Example external submission adapter for OpenVoiceCS-Bench.

Run through the benchmark CLI:
    python scripts/run_openvoicecs.py submit examples/openvoicecs_submission_adapter.py:run \
      --name example_submission --trials 1
"""

from __future__ import annotations


def run(scenario: dict, trial_index: int) -> dict:
    """Return a benchmark trace for one scenario.

    This demo follows the scenario oracle so contributors can verify their
    adapter wiring before replacing the internals with a real agent.
    """
    del trial_index
    oracle = scenario["oracle"]
    events = []
    for event in oracle.get("required_events", []):
        if event not in events:
            events.append(event)
    for section in ("privacy", "auth"):
        for event in oracle.get(section, {}).get("required_events", []):
            if event not in events:
                events.append(event)

    return {
        "messages": [
            {
                "role": "agent",
                "text": oracle.get("reference_response", "I can help with that."),
            }
        ],
        "tool_calls": oracle.get("expected_tool_calls", []),
        "events": events,
        "latency": {
            "v2v_ttfb_ms": scenario.get("experience", {}).get("reference_latency_ms", 750),
            "v2v_last_byte_ms": scenario.get("experience", {}).get("reference_latency_ms", 750) + 400,
        },
        "cost_usd": 0.01,
    }
