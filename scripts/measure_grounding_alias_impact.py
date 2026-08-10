#!/usr/bin/env python3
"""Measure how many required-claim misses a real sweep's alias binding recovers.

docs/known-limitations.md section 7: required_claims.any_terms is a literal
phrase list, so an agent saying "rebooked you at no charge" is marked
ungrounded against a claim whose any_terms is ["no fee", "fee waiver"].
scripts/bind_grounding_aliases.py binds eligible claims to a shared paraphrase
group; this script quantifies the effect on real recorded traces by
re-checking every required claim in a scored sweep with (a) only its own
any_terms (pre-patch behavior) and (b) any_terms plus its bound alias
group's patterns, then reports how many previously-missing claims recover
and dumps each recovery for manual true/false-positive review.

    python scripts/measure_grounding_alias_impact.py \\
        data/openvoicecs/runs/text_action_v02_merged/reports \\
        --out /tmp/alias_recoveries.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark.openvoicecs import (  # noqa: E402
    DEFAULT_SCENARIO_PATH,
    _agent_text,
    _load_grounding_aliases,
    _matched_patterns,
)


def _is_missing(agent_text: str, terms: list[str]) -> tuple[bool, list[str]]:
    matched = _matched_patterns(agent_text, terms)
    return not bool(matched), matched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    scenarios = {
        s["id"]: s for s in json.loads(Path(DEFAULT_SCENARIO_PATH).read_text())["scenarios"]
    }
    aliases = _load_grounding_aliases()

    recoveries = []
    total_claim_checks = 0
    total_old_missing = 0

    for report_path in sorted(args.reports_dir.glob("*.json")):
        report = json.loads(report_path.read_text())
        model = report.get("model") or report_path.stem
        for result in report.get("results") or []:
            scenario = scenarios.get(result.get("id"))
            if scenario is None:
                continue
            claims = scenario.get("oracle", {}).get("grounding", {}).get("required_claims", [])
            if not claims:
                continue
            for trial in result.get("trials") or []:
                if trial.get("error"):
                    continue
                agent_text = _agent_text({"messages": trial.get("messages") or []})
                if not agent_text:
                    continue
                for claim in claims:
                    total_claim_checks += 1
                    any_terms = claim.get("any_terms", [])
                    was_missing, _ = _is_missing(agent_text, any_terms)
                    if not was_missing:
                        continue
                    total_old_missing += 1
                    group_terms = aliases.get(claim.get("alias_group") or "", [])
                    expanded = any_terms + [t for t in group_terms if t not in any_terms]
                    is_missing_now, matched = _is_missing(agent_text, expanded)
                    if not is_missing_now:
                        recoveries.append({
                            "model": model,
                            "scenario_id": scenario["id"],
                            "trial_index": trial.get("trial_index"),
                            "claim_id": claim.get("id"),
                            "alias_group": claim.get("alias_group"),
                            "matched_alias_patterns": matched,
                            "agent_text": agent_text[:400],
                        })

    print(f"claim checks scanned:        {total_claim_checks}")
    print(f"missing under OLD grader:    {total_old_missing}")
    print(f"recovered under NEW grader:  {len(recoveries)}")
    args.out.write_text(json.dumps(recoveries, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
