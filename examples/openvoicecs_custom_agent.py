"""Minimal custom-agent integration for OpenVoiceCS-Bench.

Run:
    python examples/openvoicecs_custom_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark import OpenVoiceCSBench


def custom_agent(scenario: dict, trial_index: int) -> dict:
    """Replace this with a real voice/customer-service agent adapter."""
    del trial_index
    first_tool = scenario["oracle"]["expected_tool_calls"][0]
    return {
        "messages": [
            {
                "role": "agent",
                "text": "I can help with that. I am starting the required verification step.",
            }
        ],
        "tool_calls": [first_tool],
        "events": scenario["oracle"].get("required_events", [])[:1],
        "latency_ms": 850,
        "usage": {"input_tokens": 800, "output_tokens": 60},
    }


def main() -> None:
    bench = OpenVoiceCSBench.load()
    report = bench.score_agent(custom_agent, max_scenarios=2, trials=1)
    print(f"OpenVoiceCS score: {report['overall_score']:.2f} / 100")
    print(f"pass@k: {report['pass_at_k']:.1%}")


if __name__ == "__main__":
    main()
