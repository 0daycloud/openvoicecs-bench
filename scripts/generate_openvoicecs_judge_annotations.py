#!/usr/bin/env python3
"""Generate deterministic OpenVoiceCS reference judge annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SCORES = {
    "empathy": 5,
    "clarity": 5,
    "naturalness": 5,
    "professionalism": 5,
    "resolution_communication": 5,
    "channel_fit": 5,
}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _trial_index(trial: dict[str, Any], fallback: int) -> int:
    value = trial.get("trial_index", fallback)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"trial_index must be an integer, got {value!r}")
    return value


def generate_annotations(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return two-rater, trial-level reference annotations for every trial."""
    rows: list[dict[str, Any]] = []
    for result in report.get("results", []):
        scenario_id = result.get("id")
        if not scenario_id:
            raise ValueError("every result must include an id")
        trials = result.get("trials", [])
        if not isinstance(trials, list) or not trials:
            raise ValueError(f"{scenario_id} must include one or more trials")
        for fallback_index, trial in enumerate(trials):
            trial_index = _trial_index(trial, fallback_index)
            item_id = f"{scenario_id}:{trial_index}"
            for rater_id in ("reference-rater-a", "reference-rater-b"):
                rows.append({
                    "item_id": item_id,
                    "scenario_id": scenario_id,
                    "rater_id": rater_id,
                    "scores": dict(DEFAULT_SCORES),
                })
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic OpenVoiceCS reference judge annotations."
    )
    parser.add_argument("report", type=Path, help="Source OpenVoiceCS report JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    args = parser.parse_args()

    report = _load_json(args.report)
    rows = generate_annotations(report)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} annotations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
