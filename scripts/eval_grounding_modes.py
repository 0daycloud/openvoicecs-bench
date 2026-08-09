#!/usr/bin/env python
"""Compare legacy and evidence grounding on stored model reports.

The required-claim check cannot be validated against itself, so this replays
real stored traces and uses the replayed state check as the reference for
whether the work was actually done:

* the agent did the work but lost grounding credit -> false negative
* the agent did not do the work but kept full credit -> false positive

Usage:
    python scripts/eval_grounding_modes.py
    python scripts/eval_grounding_modes.py --reports 'data/openvoicecs/runs/**/reports/*.json'
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark.grounding import (  # noqa: E402
    EVIDENCE,
    LEGACY,
    score_required_claims,
)

DEFAULT_REPORTS = "data/openvoicecs/runs/text_action_v02_merged/reports/*.json"
DEFAULT_SCENARIOS = "data/openvoicecs/scenarios_v0.1.json"


def load_trials(reports: str, scenarios: str):
    index = {s["id"]: s for s in json.load(open(scenarios))["scenarios"]}
    for path in sorted(glob.glob(reports, recursive=True)):
        try:
            report = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        for result in report.get("results", []):
            scenario = index.get(result.get("id"))
            if not scenario:
                continue
            for trial in result.get("trials", []):
                if "required_passed" not in (trial.get("grounding_check") or {}):
                    continue
                text = " ".join(
                    m.get("text", "")
                    for m in trial.get("messages", [])
                    if m.get("role") == "agent"
                ).strip()
                if not text:
                    continue
                yield scenario, text, (trial.get("state_check") or {}).get("passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default=DEFAULT_REPORTS)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    args = parser.parse_args()

    stats: collections.Counter = collections.Counter()
    for scenario, text, did_work in load_trials(args.reports, args.scenarios):
        claims = scenario["oracle"]["grounding"]["required_claims"]
        scores = {
            mode: score_required_claims(
                agent_text=text,
                required_claims=claims,
                state_satisfied=did_work,
                mode=mode,
            )["score"]
            for mode in (LEGACY, EVIDENCE)
        }
        if did_work is True:
            stats["did_work"] += 1
            for mode, score in scores.items():
                stats[f"fn_{mode}"] += score < 1.0
        elif did_work is False:
            stats["no_work"] += 1
            for mode, score in scores.items():
                stats[f"fp_{mode}"] += score == 1.0

    did, none = stats["did_work"], stats["no_work"]
    if not did or not none:
        print("no comparable trials found", file=sys.stderr)
        return 1

    def pct(count: int, total: int) -> str:
        return f"{count:5d} / {total:<5d} ({100 * count / total:5.1f}%)"

    print(f"trials replayed: {did + none}\n")
    print("false negatives  agent did the work, credit denied")
    print(f"  legacy    {pct(stats['fn_legacy'], did)}")
    print(f"  evidence  {pct(stats['fn_evidence'], did)}\n")
    print("false positives  agent did not do the work, full credit kept")
    print(f"  legacy    {pct(stats['fp_legacy'], none)}")
    print(f"  evidence  {pct(stats['fp_evidence'], none)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
