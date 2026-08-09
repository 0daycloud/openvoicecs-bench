#!/usr/bin/env python
"""Report which ungrounded tool arguments are closed vocabularies.

Known limitation 3 stopped scoring 582 argument slots because the corpus never
documented their vocabularies, and the priority list proposes declaring enums to
make classification accuracy measurable again. Whether that works depends on
something the list does not state: how many distinct values an argument actually
takes.

An argument whose oracle value is the same everywhere is a closed vocabulary,
but declaring it as an enum only tells the model to copy the single legal
option. An argument with a fresh value per scenario is an identifier and is
correctly unknowable. Neither measures classification accuracy.

Usage:
    python scripts/report_enum_candidates.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_SCENARIOS = "data/openvoicecs/scenarios_v0.1.json"


def collect(scenarios: list[dict]) -> dict[tuple[str, str], collections.Counter]:
    """Oracle values per (tool, argument), limited to ungrounded arguments."""
    values: dict[tuple[str, str], collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for scenario in scenarios:
        ungrounded = {
            tool["name"]: set((tool.get("generated_arguments") or {}).keys())
            for tool in scenario.get("tools", [])
        }
        for call in scenario["oracle"].get("expected_tool_calls", []):
            name = call.get("name")
            for argument, value in (call.get("arguments") or {}).items():
                if argument in ungrounded.get(name, ()) and isinstance(
                    value, (str, int, float, bool)
                ):
                    values[(name, argument)][str(value)] += 1
    return values


def looks_like_identifier(values: collections.Counter) -> bool:
    """Whether the values are minted per scenario rather than drawn from a set.

    Ratio alone does not separate these: `case_id` takes 51 values over 155
    uses and `reason` takes 41 over 155. Shape does. Identifiers in this corpus
    carry a numeric suffix (`case_fs_001`, `action_fs_001`); vocabulary labels
    do not (`SIM_replacement`, `address_update`).
    """
    if not values:
        return False
    numbered = sum(1 for value in values if re.search(r"\d\s*$", value))
    return numbered / len(values) >= 0.8


def classify(values: collections.Counter) -> str:
    """What declaring an enum would achieve for this argument."""
    distinct, uses = len(values), sum(values.values())
    if looks_like_identifier(values):
        return "identifier - correctly unknowable"
    if uses < 2:
        return "single use - too thin to judge"
    if distinct == 1:
        return "SINGLE-VALUE - enum would be trivially satisfiable"
    return f"VOCABULARY ({distinct} labels) - enum would measure a real choice"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    args = parser.parse_args()

    scenarios = json.load(open(args.scenarios))["scenarios"]
    values = collect(scenarios)
    if not values:
        print("no ungrounded arguments found", file=sys.stderr)
        return 1

    rows = sorted(
        ((tool, arg, counter, classify(counter)) for (tool, arg), counter in values.items()),
        key=lambda row: (not row[3].startswith("VOCAB"), -sum(row[2].values())),
    )
    print(f"{'distinct':>8} {'uses':>5}  {'tool.argument':44s} verdict")
    print("-" * 104)
    for tool, arg, counter, verdict in rows:
        print(f"{len(counter):>8} {sum(counter.values()):>5}  {tool + '.' + arg:44s} {verdict}")

    vocab = [r for r in rows if r[3].startswith("VOCABULARY")]
    trivial = [r for r in rows if r[3].startswith("SINGLE-VALUE")]
    print(f"\nungrounded (tool, argument) pairs : {len(rows)}")
    print(f"real vocabularies                 : {len(vocab)}"
          f"  ({sum(sum(r[2].values()) for r in vocab)} oracle uses)")
    print(f"trivially satisfiable if declared : {len(trivial)}"
          f"  ({sum(sum(r[2].values()) for r in trivial)} oracle uses)")
    for tool, arg, counter, _ in vocab:
        print(f"\n{tool}.{arg} labels ({len(counter)}):")
        print("  " + ", ".join(sorted(counter)[:12]) + (", ..." if len(counter) > 12 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
