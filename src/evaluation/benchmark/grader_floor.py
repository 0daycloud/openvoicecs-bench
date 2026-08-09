"""The score floor: what a agent with no understanding can already earn.

`grader_probe` asks whether the grader punishes deviations from a correct trace.
This asks the complementary and harsher question: how high can a system score
*without* understanding anything?

Every agent here is deliberately incapable. None reads the oracle; each is
handed the same prompt a real model receives and does something mechanical with
it — parrots it back, calls every tool it was offered, refuses everything. The
gap between the best of them and a real model is the benchmark's actual
discriminating power. A floor that approaches the leaderboard's top score means
the ranking is measuring something other than capability.

Keeping these agents honest is the whole point, so they consume
``build_json_action_prompt`` output and nothing else. If a probe here can only
score well by reading a field the model never sees, it proves nothing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.evaluation.benchmark.openvoicecs import METRIC_NAMES, OpenVoiceCSBench
from src.evaluation.benchmark.provider_adapters import build_json_action_prompt

CONFIRMATION = (
    "I have verified your identity and completed this for you; I will not share "
    "any sensitive details."
)
REFUSAL = (
    "I'm sorry, I can't help with that request. I cannot share or change account "
    "details, and I won't be able to process this."
)


def model_visible_view(scenario: dict[str, Any], trial_index: int = 0) -> dict[str, Any]:
    """Exactly what the stepwise adapter puts in front of the model, parsed back.

    Going through the real prompt builder rather than reading the scenario keeps
    these agents from accidentally cheating: anything absent from the prompt is
    absent here too.
    """
    _system, user = build_json_action_prompt(scenario, trial_index)
    return json.loads(user.split("Customer session:\n", 1)[1])


def _identifier_index(records: Any) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Index the customer records by leaf field name and by container name.

    Sandbox ids live in two shapes: as a field (``{"account_id": "acct_1"}``) and
    as the key of a collection (``{"orders": {"ord_9101": {...}}}``). Both are in
    the prompt, so both are fair game.
    """
    by_field: dict[str, Any] = {}
    by_container: dict[str, list[str]] = {}

    def walk(node: Any, path: list[str]) -> None:
        if not isinstance(node, dict):
            return
        child_keys = [key for key, value in node.items() if isinstance(value, dict)]
        if path and child_keys:
            by_container[path[-1]] = child_keys
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, [*path, str(key)])
            else:
                by_field.setdefault(str(key), value)

    walk(records, [])
    return by_field, by_container


def _scrape_arguments(spec: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    by_field, by_container = _identifier_index(view.get("customer_records"))
    utterance = str(view.get("customer_utterance") or "")
    arguments: dict[str, Any] = {}
    for argument, declared_type in (spec.get("parameters") or {}).items():
        if "assigned by the system" in str(declared_type):
            continue  # the prompt says this one may be omitted
        if argument in by_field:
            arguments[argument] = by_field[argument]
            continue
        stem = re.sub(r"_id$", "", argument)
        container = next((c for c in (f"{stem}s", stem) if by_container.get(c)), None)
        if container:
            arguments[argument] = by_container[container][0]
            continue
        match = re.search(r"\b[a-z]{2,6}_[a-z0-9_]+\b", utterance)
        if match:
            arguments[argument] = match.group(0)
    return arguments


def _prompt_text(view: dict[str, Any]) -> str:
    """Everything the prompt already said, echoed back at the customer."""
    turns = " ".join(
        str(turn.get("text") or "")
        for turn in view.get("conversation") or []
        if turn.get("role") != "agent"
    )
    return " ".join(
        part for part in (
            str(view.get("customer_utterance") or ""),
            turns,
            str(view.get("customer_goal") or ""),
        ) if part
    ).strip()


def agent_refuse_everything(scenario: dict[str, Any], trial_index: int = 0) -> dict[str, Any]:
    """Never acts. Measures what refusal alone is worth."""
    del scenario, trial_index
    return {"messages": [{"role": "agent", "text": REFUSAL}], "tool_calls": [],
            "events": [], "latency_ms": 100}


def agent_parrot(scenario: dict[str, Any], trial_index: int = 0) -> dict[str, Any]:
    """Never acts, but repeats the prompt's own vocabulary back.

    Isolates how much of `factual_grounding` is satisfied by restating the
    question rather than by reporting a real outcome.
    """
    view = model_visible_view(scenario, trial_index)
    return {"messages": [{"role": "agent", "text": f"{_prompt_text(view)} {CONFIRMATION}"}],
            "tool_calls": [], "events": [], "latency_ms": 100}


def agent_call_every_tool(scenario: dict[str, Any], trial_index: int = 0) -> dict[str, Any]:
    """Calls every offered tool, with arguments scraped out of the prompt.

    Isolates how much of tool selection is decided by the scenario's own tool
    list — with no distractor tools, "call everything" and "call the right
    things" are the same move.
    """
    view = model_visible_view(scenario, trial_index)
    calls = [
        {"name": spec.get("name"), "arguments": _scrape_arguments(spec, view)}
        for spec in view.get("available_tools") or []
    ]
    return {"messages": [{"role": "agent", "text": CONFIRMATION}], "tool_calls": calls,
            "events": [], "latency_ms": 100}


def agent_parrot_and_call_everything(
    scenario: dict[str, Any], trial_index: int = 0
) -> dict[str, Any]:
    """Both mechanical tricks at once — the strongest agent that understands nothing."""
    view = model_visible_view(scenario, trial_index)
    calls = [
        {"name": spec.get("name"), "arguments": _scrape_arguments(spec, view)}
        for spec in view.get("available_tools") or []
    ]
    return {"messages": [{"role": "agent", "text": f"{_prompt_text(view)} {CONFIRMATION}"}],
            "tool_calls": calls, "events": [], "latency_ms": 100}


DEGENERATE_AGENTS = {
    "refuse_everything": agent_refuse_everything,
    "parrot": agent_parrot,
    "call_every_tool": agent_call_every_tool,
    "parrot_and_call_everything": agent_parrot_and_call_everything,
}


def measure_grader_floor(
    *,
    scenario_path: str | None = None,
    track: str | None = None,
    agents: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score every degenerate agent and report the resulting floor."""
    bench = OpenVoiceCSBench.load(scenario_path) if scenario_path else OpenVoiceCSBench.load()
    agents = agents or DEGENERATE_AGENTS
    measured = {}
    for name, agent_fn in agents.items():
        report = bench.score_agent(agent_fn, trials=1, track=track)
        results = report["results"]
        measured[name] = {
            "overall_score": report["overall_score"],
            "num_passed": sum(1 for r in results if r.get("pass_k") in (1, 1.0, True)),
            "num_scenarios": len(results),
            "metric_means": {
                metric: round(
                    sum(r["avg_scores"][metric] for r in results) / len(results), 4
                )
                for metric in METRIC_NAMES
                if results and metric in results[0].get("avg_scores", {})
            },
            "passed_by_track": _passed_by_track(results),
        }
    best = max(measured, key=lambda name: measured[name]["overall_score"])
    return {
        "track": track,
        "agents": measured,
        "floor_agent": best,
        "floor_overall_score": measured[best]["overall_score"],
        "floor_num_passed": measured[best]["num_passed"],
        "num_scenarios": measured[best]["num_scenarios"],
        # Metrics no degenerate agent ever loses a point on are contributing
        # weight to the composite without contributing discrimination.
        "metrics_never_lost": sorted(
            metric
            for metric in METRIC_NAMES
            if all(
                entry["metric_means"].get(metric, 0.0) >= 1.0
                for entry in measured.values()
            )
        ),
    }


def _passed_by_track(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_track: dict[str, dict[str, int]] = {}
    for result in results:
        entry = by_track.setdefault(str(result.get("track")), {"passed": 0, "total": 0})
        entry["total"] += 1
        if result.get("pass_k") in (1, 1.0, True):
            entry["passed"] += 1
    return dict(sorted(by_track.items()))
