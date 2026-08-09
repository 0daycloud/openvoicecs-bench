#!/usr/bin/env python
"""Rank stored reports by `overall_score` with bootstrap confidence intervals.

The leaderboard ranks on `overall_score`, but the published intervals cover
only the binary pass proportions, so adjacent models carry no stated
uncertainty. This prints the interval per model and how many adjacent ranks are
statistically indistinguishable.

Usage:
    python scripts/report_score_intervals.py
    python scripts/report_score_intervals.py --reports 'data/openvoicecs/runs/**/reports/*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark.openvoicecs import _overall_score_interval  # noqa: E402

DEFAULT_REPORTS = "data/openvoicecs/runs/text_action_v02_merged/reports/*.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default=DEFAULT_REPORTS)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(args.reports, recursive=True)):
        try:
            report = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        results = report.get("results") or []
        if not results:
            continue
        interval = _overall_score_interval(results)
        if interval["estimate"] is None:
            continue
        model = report.get("model_id") or os.path.basename(path)[:-5]
        rows.append((model, interval))

    if len(rows) < 2:
        print("need at least two reports to compare", file=sys.stderr)
        return 1

    rows.sort(key=lambda row: -row[1]["estimate"])
    print(f"{'rank':>4}  {'model':44s} {'score':>7}  {'95% CI':>16}")
    for rank, (model, ci) in enumerate(rows[: args.top], 1):
        print(f"{rank:>4}  {model[:44]:44s} {ci['estimate']:7.2f}  [{ci['low']:6.2f}, {ci['high']:6.2f}]")

    overlaps = sum(1 for i in range(len(rows) - 1) if rows[i][1]["low"] <= rows[i + 1][1]["high"])
    widths = sorted(ci["high"] - ci["low"] for _, ci in rows)
    print(f"\nmodels compared: {len(rows)}")
    print(f"adjacent ranks with overlapping 95% CIs: {overlaps} / {len(rows) - 1} "
          f"({100 * overlaps / (len(rows) - 1):.0f}%)")
    print(f"median CI width: {widths[len(widths) // 2]:.2f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
