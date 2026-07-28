#!/usr/bin/env python3
"""Build a leaderboard CSV from a directory of scored OpenVoiceCS reports.

Use this on a sweep produced by the current scorer. To publish a leaderboard
from an older sweep, re-score its recorded traces first with
``scripts/rescore_recorded_run.py``, which writes reports this can read.

Models below the coverage floor are listed as unmeasured rather than ranked —
see ``src/evaluation/benchmark/leaderboard.py`` for why that distinction is
load-bearing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark.leaderboard import (  # noqa: E402
    DEFAULT_MIN_COVERAGE,
    rank_rows,
    rows_from_reports,
    write_leaderboard,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    args = parser.parse_args()

    rows = rows_from_reports(args.reports_dir)
    if not rows:
        raise SystemExit(f"no scored reports found in {args.reports_dir}")

    ranked, excluded = rank_rows(rows, min_coverage=args.min_coverage)
    write_leaderboard(ranked, args.output)

    print(f"Ranked {len(ranked)} of {len(rows)} models -> {args.output}")
    for position, row in enumerate(ranked, 1):
        print(
            f"  {position:2}. {row['model_id']:<38} "
            f"{row['overall_score']:>6}  cov={row['trial_coverage']:.0%}"
        )
    if excluded:
        print(f"\nUnmeasured (below {args.min_coverage:.0%} trial coverage), not ranked:")
        for row in excluded:
            print(f"  {row['model_id']:<38} cov={row['trial_coverage']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
