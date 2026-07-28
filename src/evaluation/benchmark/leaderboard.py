"""Leaderboard assembly from scored OpenVoiceCS reports.

A leaderboard is only as honest as its inclusion rule. The v0.1 sweep ranked
every model that produced a file, including 26 whose every trial failed with
HTTP 402 `Insufficient credits` and were recorded as 0.0 on all metrics. A model
nobody paid to evaluate is *unmeasured*, which is a different statement from
"scored badly", and collapsing the two is how a ranking becomes fiction.

So every row here carries its measurement coverage, and ranking requires a
coverage floor. Excluded models are still reported, with the reason.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

METRIC_COLUMNS = (
    "task_success",
    "tool_correctness",
    "sop_compliance",
    "privacy",
    "auth_integrity",
    "safety",
    "factual_grounding",
    "experience_proxy",
)

LEADERBOARD_COLUMNS = (
    "model_id",
    "overall_score",
    *METRIC_COLUMNS,
    "pass_at_k",
    "mean_pass_rate",
    "trial_coverage",
    "scored_trials",
    "infrastructure_error_trials",
    "measured_scenarios",
    "median_latency_ms",
    "avg_tool_calls",
    "avg_tokens_to_completion",
    "avg_tokens_to_success",
    "avg_tokens_to_failure",
    "tokens_per_success",
    "total_cost_usd",
)

DEFAULT_MIN_COVERAGE = 0.9


def _coverage(report: dict[str, Any]) -> dict[str, Any]:
    """Measurement coverage, tolerating reports written before it existed."""
    coverage = report.get("measurement_coverage")
    if isinstance(coverage, dict):
        return coverage
    total = scored = 0
    for result in report.get("results") or []:
        for trial in result.get("trials") or []:
            total += 1
            scored += 0 if trial.get("error") else 1
    return {
        "total_trials": total,
        "scored_trials": scored,
        "infrastructure_error_trials": total - scored,
        "trial_coverage": round(scored / total, 4) if total else 0.0,
        "measured_scenarios": report.get("num_measured_scenarios", report.get("num_scenarios", 0)),
    }


def _token_metrics(report: dict[str, Any], operational: dict[str, Any]) -> dict[str, Any]:
    """Token-to-completion figures, recomputed for reports predating the metric."""
    keys = (
        "avg_tokens_to_completion",
        "avg_tokens_to_success",
        "avg_tokens_to_failure",
        "tokens_per_success",
    )
    if any(operational.get(key) is not None for key in keys):
        return {key: operational.get(key) for key in keys}

    total: list[int] = []
    success: list[int] = []
    failure: list[int] = []
    for result in report.get("results") or []:
        for trial in result.get("trials") or []:
            if trial.get("error"):
                continue
            usage = trial.get("usage") or {}
            parts = [usage.get("input_tokens"), usage.get("output_tokens")]
            values = [p for p in parts if isinstance(p, (int, float)) and not isinstance(p, bool)]
            if not values:
                continue
            spent = int(sum(values))
            total.append(spent)
            resolved = (trial.get("scores") or {}).get("task_success") == 1.0
            (success if resolved else failure).append(spent)

    def mean(values: list[int]) -> float | None:
        return round(sum(values) / len(values), 2) if values else None

    return {
        "avg_tokens_to_completion": mean(total),
        "avg_tokens_to_success": mean(success),
        "avg_tokens_to_failure": mean(failure),
        "tokens_per_success": round(sum(total) / len(success), 2) if success else None,
    }


def leaderboard_row(report: dict[str, Any], *, model_id: str | None = None) -> dict[str, Any]:
    """Flatten one scored report into a leaderboard row."""
    metrics = report.get("metric_scores") or {}
    operational = report.get("operational_metrics") or {}
    coverage = _coverage(report)
    resolved = model_id or (report.get("model_metadata") or {}).get("model_id") or "unknown"
    return {
        "model_id": resolved,
        "overall_score": report.get("overall_score"),
        **{name: metrics.get(name) for name in METRIC_COLUMNS},
        "pass_at_k": report.get("pass_at_k"),
        "mean_pass_rate": report.get("mean_pass_rate"),
        "trial_coverage": coverage.get("trial_coverage"),
        "scored_trials": coverage.get("scored_trials"),
        "infrastructure_error_trials": coverage.get("infrastructure_error_trials"),
        "measured_scenarios": coverage.get("measured_scenarios"),
        "median_latency_ms": operational.get("median_latency_ms"),
        "avg_tool_calls": operational.get("avg_tool_calls"),
        **_token_metrics(report, operational),
        "total_cost_usd": operational.get("total_cost_usd"),
    }


def rank_rows(
    rows: list[dict[str, Any]],
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into ranked and excluded, ordered by descending overall score."""
    ranked, excluded = [], []
    for row in rows:
        coverage = row.get("trial_coverage") or 0.0
        (ranked if coverage >= min_coverage else excluded).append(row)
    ranked.sort(key=lambda row: row.get("overall_score") or 0.0, reverse=True)
    excluded.sort(key=lambda row: row.get("trial_coverage") or 0.0, reverse=True)
    return ranked, excluded


def write_leaderboard(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEADERBOARD_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in LEADERBOARD_COLUMNS})


def rows_from_reports(reports_dir: Path) -> list[dict[str, Any]]:
    """Build leaderboard rows from every scored report in a directory."""
    rows = []
    for path in sorted(reports_dir.glob("*.json")):
        # macOS archives round-trip AppleDouble sidecars (`._name.json`); they are
        # not reports and are not valid UTF-8.
        if path.name.startswith("._"):
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("results"):
            continue
        rows.append(leaderboard_row(report, model_id=None))
    return rows
