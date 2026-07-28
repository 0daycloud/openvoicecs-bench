#!/usr/bin/env python3
"""Re-score a completed sweep from its recorded traces.

Published reports keep the full trace for every trial — messages, tool calls,
usage, and latency. Everything the scorer does after the provider responds is a
pure function of ``(scenario, trace)``, so a run can be re-scored against a
corrected scorer without spending a cent on new inference.

This exists because the v0.1 sweep was scored by an instrument with three
defects: undocumented tool arguments were exact-matched and cascaded into false
safety violations, forbidden events could never fire, and provider billing
failures were averaged in as model scores of 0.0. Re-scoring separates what the
models actually did from what the harness did to them.

Trials that recorded an error are replayed as errors so the scorer's own
infrastructure/model classification decides whether they count.

    python scripts/rescore_recorded_run.py \\
        data/openvoicecs/runs/<run>/reports \\
        --output-dir data/openvoicecs/runs/<run>/rescored \\
        --track text_to_action --trials 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark.leaderboard import (  # noqa: E402
    leaderboard_row,
    rank_rows,
    write_leaderboard,
)
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench  # noqa: E402


class RecordedTraceError(RuntimeError):
    """Raised when a report has no trace for a scenario/trial pair."""


def build_replay_agent(report: dict[str, Any]):
    """Return an agent callable that replays the traces stored in ``report``."""
    recorded: dict[tuple[str, int], dict[str, Any]] = {}
    for result in report.get("results") or []:
        scenario_id = result.get("id")
        for trial in result.get("trials") or []:
            recorded[(scenario_id, trial.get("trial_index", 0))] = trial

    def agent(scenario: dict[str, Any], trial_index: int) -> dict[str, Any]:
        trial = recorded.get((scenario["id"], trial_index))
        if trial is None:
            raise RecordedTraceError(f"no recorded trace for {scenario['id']}#{trial_index}")
        if trial.get("error"):
            raise RuntimeError(trial["error"])
        return {
            "messages": trial.get("messages") or [],
            "tool_calls": trial.get("tool_calls") or [],
            # Recorded `events` were themselves derived by the old engine; the
            # current scorer re-derives them, so replaying them would launder
            # stale derivations back into the result.
            "events": [],
            "usage": trial.get("usage") or {},
            "cost_usd": trial.get("cost_usd"),
            "latency": trial.get("latency"),
            "latency_ms": trial.get("latency_ms"),
        }

    return agent, len(recorded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track", default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="do not rank models whose scored-trial coverage falls below this fraction",
    )
    args = parser.parse_args()

    suite = OpenVoiceCSBench.load()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for path in sorted(args.reports_dir.glob("*.json")):
        report = json.loads(path.read_text())
        if not report.get("results"):
            continue
        model_id = (report.get("model_metadata") or {}).get("model_id") or path.stem
        agent, num_traces = build_replay_agent(report)
        if not num_traces:
            continue

        # Only scenarios the run actually recorded can be re-scored. Scoring the
        # whole current suite against an older run would invent failures for
        # scenarios that were authored after it, and charge them to the model.
        recorded_ids = {result.get("id") for result in report["results"]}
        scenarios = [s for s in suite.scenarios if s["id"] in recorded_ids]
        if not scenarios:
            print(f"  {model_id}: no recorded scenario is still in the suite, skipping")
            continue
        bench = OpenVoiceCSBench(scenarios=scenarios, version=suite.version)

        rescored = bench.score_agent(
            agent,
            trials=args.trials,
            track=args.track,
            model_metadata=report.get("model_metadata") or {"model_id": model_id},
        )
        rescored["rescored_scenarios"] = len(scenarios)
        (args.output_dir / path.name).write_text(json.dumps(rescored, indent=2) + "\n")
        rows.append(leaderboard_row(rescored, model_id=model_id))

    ranked, excluded = rank_rows(rows, min_coverage=args.min_coverage)
    summary = args.output_dir / "summary.csv"
    write_leaderboard(ranked, summary)

    print(f"Re-scored {len(rows)} models; ranked {len(ranked)} -> {summary}")
    if excluded:
        print(f"Unmeasured (below {args.min_coverage:.0%} trial coverage), not ranked:")
        for row in excluded:
            print(f"  {row['model_id']}: {row['trial_coverage']:.1%} trial coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
