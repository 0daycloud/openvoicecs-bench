"""Statistical comparison helpers for OpenVoiceCS reports.

The benchmark should not imply that tiny point-estimate differences are
meaningful. This module compares two saved reports over matched scenario IDs and
reports paired bootstrap confidence intervals plus an exact McNemar test for
scenario-level pass/fail changes.
"""

from __future__ import annotations

import random
from math import comb
from typing import Any

from src.evaluation.benchmark.openvoicecs import METRIC_NAMES, METRIC_WEIGHTS


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    iterations: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compare two benchmark reports on matched scenarios.

    Deltas are candidate minus baseline. Scenario-level CIs are paired bootstrap
    intervals over common scenario IDs, which keeps task mix fixed.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    baseline_results = _results_by_id(baseline)
    candidate_results = _results_by_id(candidate)
    common_ids = sorted(set(baseline_results) & set(candidate_results))
    if not common_ids:
        raise ValueError("reports have no scenario IDs in common")

    scenario_score_deltas = [
        _scenario_score(candidate_results[scenario_id])
        - _scenario_score(baseline_results[scenario_id])
        for scenario_id in common_ids
    ]
    pass_rate_deltas = [
        _number(candidate_results[scenario_id].get("pass_rate"))
        - _number(baseline_results[scenario_id].get("pass_rate"))
        for scenario_id in common_ids
    ]
    metric_deltas = {
        metric: [
            _metric_score(candidate_results[scenario_id], metric)
            - _metric_score(baseline_results[scenario_id], metric)
            for scenario_id in common_ids
        ]
        for metric in METRIC_NAMES
    }
    mcnemar = _mcnemar_exact(
        [
            (
                bool(baseline_results[scenario_id].get("pass_k")),
                bool(candidate_results[scenario_id].get("pass_k")),
            )
            for scenario_id in common_ids
        ]
    )
    delta_items = [
        _delta_item(
            scenario_id,
            baseline_results[scenario_id],
            candidate_results[scenario_id],
        )
        for scenario_id in common_ids
    ]

    summary = {
        "overall_score_delta": round(
            _number(candidate.get("overall_score")) - _number(baseline.get("overall_score")),
            6,
        ),
        "mean_paired_scenario_score_delta": _summary_with_ci(
            scenario_score_deltas,
            iterations=iterations,
            seed=seed,
            confidence=confidence,
        ),
        "mean_paired_pass_rate_delta": _summary_with_ci(
            pass_rate_deltas,
            iterations=iterations,
            seed=seed + 1,
            confidence=confidence,
        ),
        "pass_k_delta": round(
            _number(candidate.get("pass_k")) - _number(baseline.get("pass_k")),
            6,
        ),
        "mcnemar_exact": mcnemar,
    }

    return {
        "benchmark": "OpenVoiceCS-Bench Pairwise Comparison",
        "baseline": _report_name(baseline, fallback="baseline"),
        "candidate": _report_name(candidate, fallback="candidate"),
        "method": {
            "paired_unit": "scenario_id",
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
            "confidence": confidence,
            "mcnemar": "two-sided exact binomial test over discordant pass^k pairs",
        },
        "matched_scenarios": {
            "count": len(common_ids),
            "ids": common_ids,
            "baseline_only": sorted(set(baseline_results) - set(candidate_results)),
            "candidate_only": sorted(set(candidate_results) - set(baseline_results)),
            "metadata_mismatches": _metadata_mismatches(
                baseline_results,
                candidate_results,
                common_ids,
            ),
        },
        "summary": {
            **summary,
            "interpretation": _interpret_delta(summary["mean_paired_scenario_score_delta"]),
        },
        "metric_deltas": {
            metric: _summary_with_ci(
                values,
                iterations=iterations,
                seed=seed + 100 + index,
                confidence=confidence,
            )
            for index, (metric, values) in enumerate(metric_deltas.items())
        },
        "stratified_deltas": _stratified_deltas(
            delta_items,
            iterations=iterations,
            seed=seed + 1_000,
            confidence=confidence,
        ),
        "slice_deltas": _slice_deltas(
            delta_items,
            iterations=iterations,
            seed=seed + 2_000,
            confidence=confidence,
        ),
    }


def _results_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = {}
    for result in report.get("results", []):
        if isinstance(result, dict) and result.get("id"):
            results[str(result["id"])] = result
    return results


def _delta_item(
    scenario_id: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": scenario_id,
        "domain": str(baseline.get("domain") or candidate.get("domain") or "unknown"),
        "track": str(baseline.get("track") or candidate.get("track") or "unknown"),
        "difficulty": str(
            baseline.get("difficulty") or candidate.get("difficulty") or "unknown"
        ),
        "scenario_score_delta": _scenario_score(candidate) - _scenario_score(baseline),
        "pass_rate_delta": (
            _number(candidate.get("pass_rate")) - _number(baseline.get("pass_rate"))
        ),
        "baseline_pass_k": bool(baseline.get("pass_k")),
        "candidate_pass_k": bool(candidate.get("pass_k")),
    }


def _metadata_mismatches(
    baseline_results: dict[str, dict[str, Any]],
    candidate_results: dict[str, dict[str, Any]],
    common_ids: list[str],
) -> list[dict[str, str]]:
    mismatches = []
    for scenario_id in common_ids:
        for field in ("domain", "track", "difficulty"):
            baseline_value = baseline_results[scenario_id].get(field)
            candidate_value = candidate_results[scenario_id].get(field)
            if baseline_value != candidate_value:
                mismatches.append(
                    {
                        "id": scenario_id,
                        "field": field,
                        "baseline": str(baseline_value),
                        "candidate": str(candidate_value),
                    }
                )
    return mismatches


def _stratified_deltas(
    items: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    return {
        field: {
            "scenario_score_delta": _stratified_summary_with_ci(
                _values_by_stratum(items, field, "scenario_score_delta"),
                iterations=iterations,
                seed=seed + index,
                confidence=confidence,
            ),
            "pass_rate_delta": _stratified_summary_with_ci(
                _values_by_stratum(items, field, "pass_rate_delta"),
                iterations=iterations,
                seed=seed + 100 + index,
                confidence=confidence,
            ),
        }
        for index, field in enumerate(("domain", "track", "difficulty"))
    }


def _slice_deltas(
    items: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    slices = {}
    for field_index, field in enumerate(("domain", "track", "difficulty")):
        field_slices = {}
        for slice_index, value in enumerate(sorted({str(item[field]) for item in items})):
            slice_items = [item for item in items if item[field] == value]
            field_slices[value] = {
                "count": len(slice_items),
                "scenario_score_delta": _summary_with_ci(
                    [item["scenario_score_delta"] for item in slice_items],
                    iterations=iterations,
                    seed=seed + field_index * 1_000 + slice_index,
                    confidence=confidence,
                ),
                "pass_rate_delta": _summary_with_ci(
                    [item["pass_rate_delta"] for item in slice_items],
                    iterations=iterations,
                    seed=seed + field_index * 1_000 + 100 + slice_index,
                    confidence=confidence,
                ),
                "mcnemar_exact": _mcnemar_exact(
                    [
                        (item["baseline_pass_k"], item["candidate_pass_k"])
                        for item in slice_items
                    ]
                ),
            }
        slices[field] = field_slices
    return slices


def _scenario_score(result: dict[str, Any]) -> float:
    scores = result.get("avg_scores") if isinstance(result.get("avg_scores"), dict) else {}
    weighted = sum(
        _number(scores.get(metric)) * weight
        for metric, weight in METRIC_WEIGHTS.items()
    )
    return weighted * 100.0


def _metric_score(result: dict[str, Any], metric: str) -> float:
    scores = result.get("avg_scores") if isinstance(result.get("avg_scores"), dict) else {}
    return _number(scores.get(metric))


def _summary_with_ci(
    values: list[float],
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    mean = _mean(values)
    low, high = _paired_bootstrap_ci(
        values,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
    )
    return {
        "mean": round(mean, 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "n": len(values),
    }


def _stratified_summary_with_ci(
    values_by_stratum: dict[str, list[float]],
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    values = [
        value
        for stratum_values in values_by_stratum.values()
        for value in stratum_values
    ]
    low, high = _stratified_bootstrap_ci(
        values_by_stratum,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
    )
    return {
        "mean": round(_mean(values), 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "n": len(values),
        "num_strata": len(values_by_stratum),
        "strata": {
            key: len(value)
            for key, value in sorted(values_by_stratum.items())
        },
    }


def _paired_bootstrap_ci(
    values: list[float],
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(iterations):
        samples.append(_mean([values[rng.randrange(n)] for _ in range(n)]))
    alpha = 1.0 - confidence
    return _percentile(samples, alpha / 2), _percentile(samples, 1 - alpha / 2)


def _stratified_bootstrap_ci(
    values_by_stratum: dict[str, list[float]],
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    values = [
        value
        for stratum_values in values_by_stratum.values()
        for value in stratum_values
    ]
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    samples = []
    strata = [list(stratum_values) for stratum_values in values_by_stratum.values()]
    for _ in range(iterations):
        sampled = []
        for stratum_values in strata:
            n = len(stratum_values)
            sampled.extend(stratum_values[rng.randrange(n)] for _ in range(n))
        samples.append(_mean(sampled))
    alpha = 1.0 - confidence
    return _percentile(samples, alpha / 2), _percentile(samples, 1 - alpha / 2)


def _values_by_stratum(
    items: list[dict[str, Any]],
    stratum_field: str,
    value_field: str,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for item in items:
        grouped.setdefault(str(item[stratum_field]), []).append(float(item[value_field]))
    return grouped


def _mcnemar_exact(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    candidate_wins = sum(
        1
        for baseline_pass, candidate_pass in pairs
        if candidate_pass and not baseline_pass
    )
    baseline_wins = sum(
        1
        for baseline_pass, candidate_pass in pairs
        if baseline_pass and not candidate_pass
    )
    ties = len(pairs) - candidate_wins - baseline_wins
    discordant = candidate_wins + baseline_wins
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(candidate_wins, baseline_wins)
        tail = sum(comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "discordant_pairs": discordant,
        "p_value": round(p_value, 8),
    }


def _interpret_delta(score_delta: dict[str, Any]) -> str:
    low = score_delta.get("ci_low")
    high = score_delta.get("ci_high")
    if low is not None and low > 0:
        return "candidate_higher_with_ci_excluding_zero"
    if high is not None and high < 0:
        return "baseline_higher_with_ci_excluding_zero"
    return "difference_not_resolved_by_ci"


def _report_name(report: dict[str, Any], *, fallback: str) -> str:
    metadata = report.get("model_metadata", {})
    if isinstance(metadata, dict):
        return str(
            metadata.get("display_name")
            or metadata.get("model_id")
            or metadata.get("agent")
            or fallback
        )
    return fallback


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * quantile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _number(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
