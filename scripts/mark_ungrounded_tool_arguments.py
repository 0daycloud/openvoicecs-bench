#!/usr/bin/env python3
"""Mark tool arguments the agent cannot know as ``generated_arguments``.

The scenario schema distinguishes two kinds of tool argument:

* arguments the agent must determine (an order id it was told, an amount it must
  compute) — these are scored by exact replay comparison; and
* arguments the *system* assigns (the id of a case being created, an
  undocumented status label) — these are declared in ``generated_arguments`` so
  the replay checker substitutes the canonical value instead of demanding the
  agent guess it.

The v0.1 corpus only populated ``generated_arguments`` in 9 of 571 tool
definitions, so hundreds of system-assigned arguments were exact-matched
against values that appear nowhere the agent can read. Every such call was
charged ``argument_mismatch``, which cascaded into ``state_mismatch``,
``missing_tool``, ``unnecessary_tool`` and a ``safety:tool_replay_error`` — the
dominant failure mode of the published sweep.

This script derives the correct annotation mechanically: an argument stays
scored when its value appears in content the agent can actually read
(conversation, profile, initial state, policy, tool descriptions, or a prior
tool's return payload), compared after token normalization so that
``damaged furniture`` counts as grounding for ``damaged_furniture``. Everything
else is system-assigned and gets marked.

Run with ``--check`` in CI to fail when a scenario reintroduces the problem.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_SCENARIOS = Path("data/openvoicecs/scenarios_v0.1.json")

# Arguments that are legitimately recoverable from spoken content even though a
# literal substring match fails. Keeping these scored preserves the robustness
# signal they exist to measure.
GROUNDED_EXCEPTIONS: set[tuple[str, str, str]] = {
    ("telecom-noisy-address-correction-001", "confirm_service_address", "address"),
}


def normalize_token(value: str) -> str:
    """Collapse a value to a comparable token (mirrors the scorer)."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def model_visible_text(scenario: dict[str, Any]) -> str:
    """Return the text of everything the agent can read."""
    parts: list[str] = []
    for key in (
        "customer_goal",
        "customer_profile",
        "conversation",
        "initial_state",
        "policy",
        "experience",
    ):
        if scenario.get(key) is not None:
            parts.append(json.dumps(scenario[key]))
    for tool in scenario.get("tools") or []:
        parts.append(json.dumps(tool.get("name")))
        parts.append(json.dumps(tool.get("description")))
        parts.append(json.dumps(sorted((tool.get("required_arguments") or {}).keys())))
        # Return payloads are handed back to the agent mid-conversation.
        for key in ("result", "returns", "failure"):
            if tool.get(key) is not None:
                parts.append(json.dumps(tool[key]))
    return "\n".join(part for part in parts if part)


def ungrounded_arguments(
    scenario: dict[str, Any],
    tool: dict[str, Any],
    raw: str,
) -> dict[str, Any]:
    """Arguments whose exact value the agent has no way to determine.

    Grounding requires the value to appear *verbatim* in content the agent
    reads. A near-miss that only survives normalization is not grounding: the
    conversation saying "duplicate modem fee" does not tell the model that this
    API expects the token ``duplicate_modem_fee`` rather than the sentence
    "Duplicate modem rental charge". Scoring that difference measures the
    model's guess at an undocumented string format, not its behaviour.

    The signal returns as soon as a tool documents the vocabulary — a
    documented enum makes the argument knowable, and knowable arguments are
    scored.
    """
    generated = set((tool.get("generated_arguments") or {}).keys())
    bound = set((tool.get("argument_bindings") or {}).keys())
    documented = set((tool.get("argument_enums") or {}).keys())
    found: dict[str, Any] = {}
    for name, value in (tool.get("required_arguments") or {}).items():
        if name in generated or name in bound or name in documented:
            continue
        # Numbers are computed from state rather than recalled, so they stay scored.
        if not isinstance(value, str):
            continue
        if (scenario.get("id"), tool.get("name"), name) in GROUNDED_EXCEPTIONS:
            continue
        if value in raw:
            continue
        found[name] = value
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report unmarked system-assigned arguments and exit non-zero",
    )
    args = parser.parse_args()

    payload = json.loads(args.scenarios.read_text())
    scenarios = payload["scenarios"]

    annotated_tools = 0
    annotated_slots = 0
    by_field: dict[str, int] = {}
    offenders: list[str] = []

    for scenario in scenarios:
        raw = model_visible_text(scenario)
        for tool in scenario.get("tools") or []:
            missing = ungrounded_arguments(scenario, tool, raw)
            if not missing:
                continue
            annotated_tools += 1
            annotated_slots += len(missing)
            for name in missing:
                by_field[name] = by_field.get(name, 0) + 1
                offenders.append(f"{scenario['id']}:{tool.get('name')}.{name}")
            if not args.check:
                generated = dict(tool.get("generated_arguments") or {})
                generated.update(missing)
                tool["generated_arguments"] = generated

    if args.check:
        if offenders:
            print(
                f"{len(offenders)} tool arguments are system-assigned but not declared "
                f"in generated_arguments:",
                file=sys.stderr,
            )
            for item in offenders[:20]:
                print(f"  {item}", file=sys.stderr)
            if len(offenders) > 20:
                print(f"  ... and {len(offenders) - 20} more", file=sys.stderr)
            return 1
        print("All system-assigned tool arguments are declared.")
        return 0

    args.scenarios.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Annotated {annotated_slots} argument slots across {annotated_tools} tool definitions.")
    for name, count in sorted(by_field.items(), key=lambda item: -item[1]):
        print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
