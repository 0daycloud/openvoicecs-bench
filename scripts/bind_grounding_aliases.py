#!/usr/bin/env python3
"""Bind required-claim ids to a shared paraphrase group where one exists.

``oracle.grounding.required_claims[].any_terms`` is a hand-authored literal
phrase list (docs/known-limitations.md section 7): an agent that says
"rebooked you at no charge" is marked ungrounded against a claim whose
``any_terms`` is ``["no change fee", "no fee", "fee waiver"]``. This script
does not invent new matching logic; it wires each claim to
``data/openvoicecs/grounding_aliases_v0.1.json`` by id, the same mechanical,
id-matched style ``bind_forbidden_event_triggers.py`` uses for events. The
scorer (``check_factual_grounding``) unions ``any_terms`` with the bound
group's patterns at score time — a claim not bound here is scored exactly as
before.

Run with ``--check`` to fail CI when a claim's id matches a published alias
group but the claim was not bound (drift between the corpus and the alias
library).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_SCENARIOS = Path("data/openvoicecs/scenarios_v0.1.json")
DEFAULT_ALIASES = Path("data/openvoicecs/grounding_aliases_v0.1.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--check", action="store_true", help="fail on unbound-but-bindable claims")
    args = parser.parse_args()

    payload = json.loads(args.scenarios.read_text())
    scenarios = payload["scenarios"]
    alias_groups = json.loads(args.aliases.read_text())["aliases"]

    bound = 0
    already_bound = 0
    unbound: list[str] = []

    for scenario in scenarios:
        claims = (scenario.get("oracle") or {}).get("grounding", {}).get("required_claims", [])
        for claim in claims:
            claim_id = claim.get("id")
            if claim_id not in alias_groups:
                continue
            if claim.get("alias_group") == claim_id:
                already_bound += 1
                continue
            if args.check:
                unbound.append(f"{scenario['id']}:{claim_id}")
                continue
            claim["alias_group"] = claim_id
            bound += 1

    if args.check:
        if unbound:
            print(
                f"{len(unbound)} required claims match a published alias group "
                f"but are not bound:",
                file=sys.stderr,
            )
            for item in unbound[:20]:
                print(f"  {item}", file=sys.stderr)
            return 1
        print(f"All {already_bound} eligible required claims are bound to an alias group.")
        return 0

    args.scenarios.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Bound {bound} required claims to an alias group ({already_bound} already bound).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
