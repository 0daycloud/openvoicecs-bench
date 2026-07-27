"""Latency-cost-quality frontier reports for voice-agent benchmarks.

The frontier layer is intentionally benchmark-agnostic. It consumes saved
benchmark reports, normalizes the three axes, and marks non-dominated systems
without replacing the primary benchmark score with a single leaderboard rank.
"""

from __future__ import annotations

import csv
import html
import json
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.pricing import PIPELINE_REQUIRED_COMPONENTS, resolve_report_pricing

DEFAULT_EXPERIENCE_GATE = 0.6
DEFAULT_LATENCY_AXIS = "p95_v2v_ttfb_ms"
DEFAULT_COST_AXIS = "cost_usd_per_successful_conversation"
DEFAULT_QUALITY_AXIS = "task_success_rate"
COST_COMPONENTS = (
    "asr",
    "llm",
    "tts",
    "speech_to_speech",
    "telephony",
    "transport",
)
SCORECARD_COLUMNS = (
    "system",
    "frontier_eligible",
    "p50_v2v_ttfb_ms",
    "p90_v2v_ttfb_ms",
    "p95_v2v_ttfb_ms",
    "p99_v2v_ttfb_ms",
    "p95_v2v_last_byte_ms",
    "barge_in_stop_p95_ms",
    "interruption_recovery_p95_ms",
    "latency_at_100_concurrency_p95_ms",
    "cost_usd_per_conversation",
    "cost_usd_per_successful_conversation",
    "task_success_rate",
    "experience_score",
    "experience_score_source",
    "experience_coverage",
    "experience_judged_trials",
    "pricing_snapshot_date",
    "latency_measurement_samples",
    "latency_event_stream_samples",
    "latency_vad_origin_samples",
    "latency_at_100_sample_count",
    "latency_at_100_saturated",
    "latency_at_100_requested_calls",
    "latency_at_100_completed_calls",
    "latency_at_100_error_calls",
    "latency_at_100_peak_active_calls",
    "cost_sample_count",
    "direct_cost_samples",
    "derived_cost_samples",
    "fully_loaded_cost_samples",
    "missing_cost_samples",
    "cost_required_components",
)


@dataclass(frozen=True)
class FrontierIssue:
    path: str
    message: str


def build_frontier_report(
    reports: list[dict[str, Any]],
    *,
    pricing_manifest: dict[str, Any] | None = None,
    pricing_snapshot_date: str | None = None,
    environment: dict[str, Any] | None = None,
    experience_gate: float = DEFAULT_EXPERIENCE_GATE,
    utility_weights: dict[str, float] | None = None,
    latency_targets_ms: list[float] | tuple[float, ...] | None = None,
    cost_targets_usd: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Build a Pareto frontier report from benchmark result JSON objects.

    Lower latency and cost are better; higher quality is better. Systems with
    missing axes or experience scores below the gate are excluded from frontier
    eligibility but still appear in the scorecard with explicit reasons.
    """
    systems = [
        _normalize_system_report(
            report,
            index=index,
            pricing_manifest=pricing_manifest,
            pricing_snapshot_date=pricing_snapshot_date,
            experience_gate=experience_gate,
        )
        for index, report in enumerate(reports)
    ]
    eligible = [system for system in systems if system["frontier_eligible"]]

    frontier_names = _pareto_frontier_names(
        eligible,
        latency_key=DEFAULT_LATENCY_AXIS,
        cost_key=DEFAULT_COST_AXIS,
        quality_key=DEFAULT_QUALITY_AXIS,
    )
    projection_frontiers = {
        "latency_vs_quality": _projection_frontier_names(
            eligible,
            lower_key=DEFAULT_LATENCY_AXIS,
            higher_key=DEFAULT_QUALITY_AXIS,
        ),
        "cost_vs_quality": _projection_frontier_names(
            eligible,
            lower_key=DEFAULT_COST_AXIS,
            higher_key=DEFAULT_QUALITY_AXIS,
        ),
    }

    utility_view = None
    if utility_weights is not None:
        utility_view = build_utility_view(systems, utility_weights)
    constrained_frontiers = build_constrained_frontiers(
        eligible,
        latency_targets_ms=latency_targets_ms,
        cost_targets_usd=cost_targets_usd,
    )
    domain_frontiers = build_domain_frontiers(systems)

    return {
        "benchmark": "Latency-Cost-Quality Frontier",
        "generated_at": time.strftime("%Y-%m-%d"),
        "axes": {
            "latency": {
                "primary": DEFAULT_LATENCY_AXIS,
                "direction": "lower_is_better",
                "definition": "User end-of-speech/VAD endpoint to first audio byte out.",
            },
            "cost": {
                "primary": DEFAULT_COST_AXIS,
                "direction": "lower_is_better",
                "definition": "Fully loaded cost divided by task success rate.",
            },
            "quality": {
                "primary": DEFAULT_QUALITY_AXIS,
                "direction": "higher_is_better",
                "definition": "Task completion/resolution rate on the fixed scenario set.",
            },
        },
        "experience_gate": experience_gate,
        "pricing_snapshot": _pricing_snapshot_metadata(pricing_manifest, pricing_snapshot_date),
        "environment": environment or {},
        "frontier": frontier_names,
        "projection_frontiers": projection_frontiers,
        "constrained_frontiers": constrained_frontiers,
        "domain_frontiers": domain_frontiers,
        "systems": {system["name"]: system for system in systems},
        "scorecards": {
            system["name"]: system["scorecard"]
            for system in systems
        },
        "utility_view": utility_view,
        "methodology_notes": [
            "Pareto dominance uses p95 voice-to-voice TTFB, cost per successful conversation, and task success.",
            "Reports missing any primary axis are not frontier-eligible.",
            "Conversation experience is a gate, not an optimization axis.",
            "Provider pricing should be pinned with a snapshot date for reproducible cost comparisons.",
            "Constrained frontiers, when requested, filter systems by latency/cost budgets before Pareto evaluation.",
        ],
    }


def build_domain_frontiers(systems: list[dict[str, Any]]) -> dict[str, Any]:
    """Build Pareto frontiers for each scenario domain represented in systems."""
    domains = sorted({
        domain
        for system in systems
        for domain in system.get("domain_axes", {})
    })
    frontiers = {}
    for domain in domains:
        domain_systems = []
        represented = []
        for system in systems:
            axes = system.get("domain_axes", {}).get(domain)
            if not isinstance(axes, dict):
                continue
            represented.append(system["name"])
            if system.get("frontier_eligible") and _axes_have_primary_values(axes):
                domain_systems.append({"name": system["name"], "axes": axes})
        frontiers[domain] = {
            "frontier": _pareto_frontier_names(
                domain_systems,
                latency_key=DEFAULT_LATENCY_AXIS,
                cost_key=DEFAULT_COST_AXIS,
                quality_key=DEFAULT_QUALITY_AXIS,
            ),
            "projection_frontiers": {
                "latency_vs_quality": _projection_frontier_names(
                    domain_systems,
                    lower_key=DEFAULT_LATENCY_AXIS,
                    higher_key=DEFAULT_QUALITY_AXIS,
                ),
                "cost_vs_quality": _projection_frontier_names(
                    domain_systems,
                    lower_key=DEFAULT_COST_AXIS,
                    higher_key=DEFAULT_QUALITY_AXIS,
                ),
            },
            "eligible_systems": [system["name"] for system in domain_systems],
            "represented_systems": sorted(represented),
            "num_eligible_systems": len(domain_systems),
        }
    return frontiers


def build_constrained_frontiers(
    systems: list[dict[str, Any]],
    *,
    latency_targets_ms: list[float] | tuple[float, ...] | None = None,
    cost_targets_usd: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Build Pareto frontiers after applying optional latency/cost budgets."""
    latency_targets = _clean_targets(latency_targets_ms)
    cost_targets = _clean_targets(cost_targets_usd)
    constraints = []
    if latency_targets and cost_targets:
        constraints.extend(
            {"latency_ms": latency, "cost_usd": cost}
            for latency in latency_targets
            for cost in cost_targets
        )
    elif latency_targets:
        constraints.extend({"latency_ms": latency, "cost_usd": None} for latency in latency_targets)
    elif cost_targets:
        constraints.extend({"latency_ms": None, "cost_usd": cost} for cost in cost_targets)

    entries = []
    for constraint in constraints:
        constrained = [
            system for system in systems
            if _within_constraint(system, constraint)
        ]
        names = _pareto_frontier_names(
            constrained,
            latency_key=DEFAULT_LATENCY_AXIS,
            cost_key=DEFAULT_COST_AXIS,
            quality_key=DEFAULT_QUALITY_AXIS,
        )
        entries.append({
            "constraint": constraint,
            "frontier": names,
            "eligible_systems": [system["name"] for system in constrained],
            "num_eligible_systems": len(constrained),
        })
    return {
        "latency_targets_ms": latency_targets,
        "cost_targets_usd": cost_targets,
        "entries": entries,
    }


def build_frontier_plot_data(frontier_report: dict[str, Any]) -> dict[str, Any]:
    """Build machine-readable 3D and 2D plot points from a frontier report."""
    domains = {"all", *_available_domains(frontier_report)}
    domain_plots = {}
    for domain in sorted(domains):
        points = []
        for system in frontier_report.get("systems", {}).values():
            axes = _axes_for_domain(system, domain)
            if axes is None:
                continue
            point = {
                "system": system["name"],
                "domain": domain,
                "frontier_eligible": system["frontier_eligible"],
                "exclusion_reasons": system["exclusion_reasons"],
                DEFAULT_LATENCY_AXIS: axes.get(DEFAULT_LATENCY_AXIS),
                DEFAULT_COST_AXIS: axes.get(DEFAULT_COST_AXIS),
                DEFAULT_QUALITY_AXIS: axes.get(DEFAULT_QUALITY_AXIS),
            }
            points.append(point)

        eligible_points = [
            point for point in points
            if point["frontier_eligible"] and _point_has_primary_axes(point)
        ]
        frontier_names = _pareto_frontier_point_names(eligible_points)
        domain_plots[domain] = {
            "frontier": frontier_names,
            "points": [
                {
                    **point,
                    "on_frontier": point["system"] in frontier_names,
                }
                for point in points
            ],
            "projections": {
                "latency_vs_quality": _projection_frontier_point_names(
                    eligible_points,
                    lower_key=DEFAULT_LATENCY_AXIS,
                    higher_key=DEFAULT_QUALITY_AXIS,
                ),
                "cost_vs_quality": _projection_frontier_point_names(
                    eligible_points,
                    lower_key=DEFAULT_COST_AXIS,
                    higher_key=DEFAULT_QUALITY_AXIS,
                ),
            },
        }

    return {
        "benchmark": frontier_report.get("benchmark"),
        "generated_at": frontier_report.get("generated_at"),
        "axes": frontier_report.get("axes", {}),
        "domains": domain_plots,
    }


def render_frontier_svg(
    plot_data: dict[str, Any],
    *,
    domain: str = "all",
    projection: str = "3d",
    width: int = 960,
    height: int = 640,
) -> str:
    """Render a dependency-free SVG plot for a frontier domain."""
    domain_data = plot_data["domains"][domain]
    points = domain_data["points"]
    title = f"{domain} frontier: {projection}"
    if projection == "3d":
        return _render_3d_svg(points, title=title, width=width, height=height)
    if projection == "latency_vs_quality":
        return _render_2d_svg(
            points,
            x_key=DEFAULT_LATENCY_AXIS,
            y_key=DEFAULT_QUALITY_AXIS,
            x_label="p95 voice-to-voice TTFB ms",
            y_label="task success rate",
            title=title,
            width=width,
            height=height,
        )
    if projection == "cost_vs_quality":
        return _render_2d_svg(
            points,
            x_key=DEFAULT_COST_AXIS,
            y_key=DEFAULT_QUALITY_AXIS,
            x_label="cost per successful conversation USD",
            y_label="task success rate",
            title=title,
            width=width,
            height=height,
        )
    raise ValueError(f"unsupported projection: {projection}")


def write_frontier_artifacts(
    frontier_report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write plot JSON plus per-domain 3D and 2D SVG artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_data = build_frontier_plot_data(frontier_report)
    written: dict[str, str] = {}

    plot_json = output / "frontier_plot_data.json"
    plot_json.write_text(json.dumps(plot_data, indent=2), encoding="utf-8")
    written["plot_data"] = str(plot_json)

    for domain in plot_data["domains"]:
        safe_domain = _slug(domain)
        for projection in ("3d", "latency_vs_quality", "cost_vs_quality"):
            path = output / f"{safe_domain}_{projection}.svg"
            path.write_text(
                render_frontier_svg(plot_data, domain=domain, projection=projection),
                encoding="utf-8",
            )
            written[f"{domain}:{projection}"] = str(path)
    return written


def build_scorecard_rows(frontier_report: dict[str, Any]) -> list[dict[str, Any]]:
    """Build standardized per-system scorecard rows for export."""
    rows = []
    scorecards = frontier_report.get("scorecards", {})
    systems = frontier_report.get("systems", {})
    for name in sorted(scorecards):
        scorecard = scorecards[name]
        system = systems.get(name, {}) if isinstance(systems, dict) else {}
        measurement = scorecard.get("latency_measurement", {})
        load = scorecard.get("latency_load", {})
        load_levels = load.get("levels", {}) if isinstance(load, dict) else {}
        load_100 = load_levels.get("100", {}) if isinstance(load_levels, dict) else {}
        cost_provenance = scorecard.get("cost_provenance", {})
        row = {
            "system": name,
            "frontier_eligible": system.get("frontier_eligible"),
            "p50_v2v_ttfb_ms": scorecard.get("p50_v2v_ttfb_ms"),
            "p90_v2v_ttfb_ms": scorecard.get("p90_v2v_ttfb_ms"),
            "p95_v2v_ttfb_ms": scorecard.get("p95_v2v_ttfb_ms"),
            "p99_v2v_ttfb_ms": scorecard.get("p99_v2v_ttfb_ms"),
            "p95_v2v_last_byte_ms": scorecard.get("p95_v2v_last_byte_ms"),
            "barge_in_stop_p95_ms": scorecard.get("barge_in_stop_p95_ms"),
            "interruption_recovery_p95_ms": scorecard.get("interruption_recovery_p95_ms"),
            "latency_at_100_concurrency_p95_ms": scorecard.get("latency_at_100_concurrency_p95_ms"),
            "cost_usd_per_conversation": scorecard.get("cost_usd_per_conversation"),
            "cost_usd_per_successful_conversation": scorecard.get("cost_usd_per_successful_conversation"),
            "task_success_rate": scorecard.get("task_success_rate"),
            "experience_score": scorecard.get("experience_score"),
            "experience_score_source": scorecard.get("experience_score_source"),
            "experience_coverage": (
                scorecard.get("experience_evidence", {}).get("coverage")
                if isinstance(scorecard.get("experience_evidence"), dict)
                else None
            ),
            "experience_judged_trials": (
                scorecard.get("experience_evidence", {}).get("num_judged_trials")
                if isinstance(scorecard.get("experience_evidence"), dict)
                else None
            ),
            "pricing_snapshot_date": scorecard.get("pricing_snapshot_date"),
            "latency_measurement_samples": measurement.get("sample_count") if isinstance(measurement, dict) else None,
            "latency_event_stream_samples": measurement.get("event_stream_samples") if isinstance(measurement, dict) else None,
            "latency_vad_origin_samples": measurement.get("vad_origin_samples") if isinstance(measurement, dict) else None,
            "latency_at_100_sample_count": load_100.get("sample_count") if isinstance(load_100, dict) else None,
            "latency_at_100_saturated": load_100.get("saturated") if isinstance(load_100, dict) else None,
            "latency_at_100_requested_calls": load_100.get("requested_calls") if isinstance(load_100, dict) else None,
            "latency_at_100_completed_calls": load_100.get("completed_calls") if isinstance(load_100, dict) else None,
            "latency_at_100_error_calls": load_100.get("error_calls") if isinstance(load_100, dict) else None,
            "latency_at_100_peak_active_calls": load_100.get("peak_active_calls") if isinstance(load_100, dict) else None,
            "cost_sample_count": cost_provenance.get("sample_count") if isinstance(cost_provenance, dict) else None,
            "direct_cost_samples": cost_provenance.get("direct_cost_samples") if isinstance(cost_provenance, dict) else None,
            "derived_cost_samples": cost_provenance.get("derived_cost_samples") if isinstance(cost_provenance, dict) else None,
            "fully_loaded_cost_samples": cost_provenance.get("fully_loaded_samples") if isinstance(cost_provenance, dict) else None,
            "missing_cost_samples": cost_provenance.get("missing_cost_samples") if isinstance(cost_provenance, dict) else None,
            "cost_required_components": ";".join(cost_provenance.get("required_components", []))
            if isinstance(cost_provenance, dict) and isinstance(cost_provenance.get("required_components"), list)
            else None,
        }
        rows.append(row)
    return rows


def write_scorecard_artifacts(
    frontier_report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write standardized scorecards as JSON, CSV, and Markdown artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = build_scorecard_rows(frontier_report)
    written: dict[str, str] = {}

    json_path = output / "scorecards.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    written["json"] = str(json_path)

    csv_path = output / "scorecards.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in SCORECARD_COLUMNS})
    written["csv"] = str(csv_path)

    markdown_path = output / "scorecards.md"
    markdown_path.write_text(_render_scorecards_markdown(rows), encoding="utf-8")
    written["markdown"] = str(markdown_path)
    return written


def _render_scorecards_markdown(rows: list[dict[str, Any]]) -> str:
    visible_columns = (
        "system",
        "frontier_eligible",
        "p95_v2v_ttfb_ms",
        "cost_usd_per_successful_conversation",
        "task_success_rate",
        "experience_score",
        "latency_at_100_concurrency_p95_ms",
    )
    header = "| " + " | ".join(visible_columns) + " |"
    separator = "| " + " | ".join("---" for _ in visible_columns) + " |"
    lines = [
        "# Latency-Cost-Quality Scorecards",
        "",
        header,
        separator,
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_markdown_cell(row.get(column)) for column in visible_columns)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def validate_frontier_report_file(path: str | Path) -> list[FrontierIssue]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return validate_frontier_report(data)


def validate_frontier_report(frontier_report: dict[str, Any]) -> list[FrontierIssue]:
    """Validate a saved latency-cost-quality frontier artifact."""
    issues: list[FrontierIssue] = []
    if not isinstance(frontier_report, dict):
        return [FrontierIssue("<root>", "must be an object")]

    required_top_level = (
        "benchmark",
        "axes",
        "experience_gate",
        "pricing_snapshot",
        "environment",
        "frontier",
        "projection_frontiers",
        "constrained_frontiers",
        "domain_frontiers",
        "systems",
        "scorecards",
    )
    for field in required_top_level:
        if field not in frontier_report:
            issues.append(FrontierIssue(field, "missing required field"))

    if frontier_report.get("benchmark") != "Latency-Cost-Quality Frontier":
        issues.append(FrontierIssue("benchmark", "must be Latency-Cost-Quality Frontier"))

    systems = frontier_report.get("systems")
    scorecards = frontier_report.get("scorecards")
    if not isinstance(systems, dict) or not systems:
        issues.append(FrontierIssue("systems", "must be a non-empty object"))
        systems = {}
    if not isinstance(scorecards, dict) or not scorecards:
        issues.append(FrontierIssue("scorecards", "must be a non-empty object"))
        scorecards = {}

    system_names = set(systems)
    scorecard_names = set(scorecards)
    if system_names and scorecard_names and system_names != scorecard_names:
        issues.append(FrontierIssue("scorecards", "must contain exactly the system names"))

    _validate_name_list(
        frontier_report.get("frontier"),
        path="frontier",
        known_names=system_names,
        issues=issues,
    )
    _validate_projection_frontiers(
        frontier_report.get("projection_frontiers"),
        known_names=system_names,
        issues=issues,
    )
    _validate_constrained_frontiers(
        frontier_report.get("constrained_frontiers"),
        known_names=system_names,
        issues=issues,
    )
    _validate_domain_frontiers(
        frontier_report.get("domain_frontiers"),
        known_names=system_names,
        issues=issues,
    )
    _validate_utility_view(
        frontier_report.get("utility_view"),
        known_names=system_names,
        issues=issues,
    )

    if not isinstance(frontier_report.get("axes"), dict):
        issues.append(FrontierIssue("axes", "must be an object"))
    if not isinstance(frontier_report.get("pricing_snapshot"), dict):
        issues.append(FrontierIssue("pricing_snapshot", "must be an object"))
    if not isinstance(frontier_report.get("environment"), dict):
        issues.append(FrontierIssue("environment", "must be an object"))
    _validate_range(
        frontier_report.get("experience_gate"),
        path="experience_gate",
        low=0.0,
        high=1.0,
        issues=issues,
    )

    for name, system in systems.items():
        _validate_frontier_system(name, system, scorecards.get(name), issues)
    for name in sorted(scorecard_names - system_names):
        _validate_scorecard(name, scorecards[name], path=f"scorecards.{name}", issues=issues)
    _validate_recomputed_frontiers(frontier_report, systems, issues)
    _validate_recomputed_utility_view(frontier_report.get("utility_view"), systems, issues)

    return issues


def _validate_frontier_system(
    name: str,
    system: Any,
    scorecard: Any,
    issues: list[FrontierIssue],
) -> None:
    path = f"systems.{name}"
    if not isinstance(system, dict):
        issues.append(FrontierIssue(path, "must be an object"))
        return
    if system.get("name") != name:
        issues.append(FrontierIssue(f"{path}.name", "must match systems key"))

    axes = system.get("axes")
    if not isinstance(axes, dict):
        issues.append(FrontierIssue(f"{path}.axes", "must be an object"))
        axes = {}

    eligible = system.get("frontier_eligible")
    if not isinstance(eligible, bool):
        issues.append(FrontierIssue(f"{path}.frontier_eligible", "must be boolean"))
        eligible = False

    exclusion_reasons = system.get("exclusion_reasons")
    if not isinstance(exclusion_reasons, list):
        issues.append(FrontierIssue(f"{path}.exclusion_reasons", "must be a list"))
    elif eligible is False and not exclusion_reasons:
        issues.append(FrontierIssue(f"{path}.exclusion_reasons", "must explain ineligibility"))

    for axis in (DEFAULT_LATENCY_AXIS, DEFAULT_COST_AXIS, DEFAULT_QUALITY_AXIS):
        _validate_optional_number(
            axes.get(axis),
            path=f"{path}.axes.{axis}",
            issues=issues,
        )
        if eligible and _coerce_number(axes.get(axis)) is None:
            issues.append(FrontierIssue(f"{path}.axes.{axis}", "required for frontier-eligible systems"))

    _validate_nonnegative(
        axes.get(DEFAULT_LATENCY_AXIS),
        path=f"{path}.axes.{DEFAULT_LATENCY_AXIS}",
        issues=issues,
    )
    _validate_nonnegative(
        axes.get(DEFAULT_COST_AXIS),
        path=f"{path}.axes.{DEFAULT_COST_AXIS}",
        issues=issues,
    )
    _validate_range(
        axes.get(DEFAULT_QUALITY_AXIS),
        path=f"{path}.axes.{DEFAULT_QUALITY_AXIS}",
        low=0.0,
        high=1.0,
        issues=issues,
        allow_none=True,
    )

    if not isinstance(system.get("scorecard"), dict):
        issues.append(FrontierIssue(f"{path}.scorecard", "must be an object"))
    elif isinstance(scorecard, dict) and system["scorecard"] != scorecard:
        issues.append(FrontierIssue(f"{path}.scorecard", "must match scorecards entry"))
    _validate_scorecard(name, scorecard, path=f"scorecards.{name}", issues=issues)


def _validate_recomputed_frontiers(
    frontier_report: dict[str, Any],
    systems: dict[str, Any],
    issues: list[FrontierIssue],
) -> None:
    semantic_systems = _semantic_frontier_systems(systems)
    eligible = [
        system for system in semantic_systems
        if system["frontier_eligible"] and _axes_have_primary_values(system["axes"])
    ]
    expected_frontier = _pareto_frontier_names(
        eligible,
        latency_key=DEFAULT_LATENCY_AXIS,
        cost_key=DEFAULT_COST_AXIS,
        quality_key=DEFAULT_QUALITY_AXIS,
    )
    _validate_exact_name_list(
        frontier_report.get("frontier"),
        expected_frontier,
        path="frontier",
        issues=issues,
        message="must match recomputed Pareto frontier",
    )

    projections = frontier_report.get("projection_frontiers")
    if isinstance(projections, dict):
        expected_projections = {
            "latency_vs_quality": _projection_frontier_names(
                eligible,
                lower_key=DEFAULT_LATENCY_AXIS,
                higher_key=DEFAULT_QUALITY_AXIS,
            ),
            "cost_vs_quality": _projection_frontier_names(
                eligible,
                lower_key=DEFAULT_COST_AXIS,
                higher_key=DEFAULT_QUALITY_AXIS,
            ),
        }
        for key, expected in expected_projections.items():
            _validate_exact_name_list(
                projections.get(key),
                expected,
                path=f"projection_frontiers.{key}",
                issues=issues,
                message="must match recomputed projection frontier",
            )

    _validate_recomputed_constrained_frontiers(
        frontier_report.get("constrained_frontiers"),
        eligible,
        issues,
    )
    _validate_recomputed_domain_frontiers(
        frontier_report.get("domain_frontiers"),
        semantic_systems,
        issues,
    )


def _semantic_frontier_systems(systems: dict[str, Any]) -> list[dict[str, Any]]:
    semantic = []
    for name, system in systems.items():
        if not isinstance(system, dict):
            continue
        axes = system.get("axes")
        domain_axes = system.get("domain_axes")
        semantic.append({
            "name": name,
            "axes": axes if isinstance(axes, dict) else {},
            "domain_axes": domain_axes if isinstance(domain_axes, dict) else {},
            "frontier_eligible": system.get("frontier_eligible") is True,
        })
    return semantic


def _validate_recomputed_constrained_frontiers(
    constrained_frontiers: Any,
    eligible: list[dict[str, Any]],
    issues: list[FrontierIssue],
) -> None:
    if not isinstance(constrained_frontiers, dict):
        return
    entries = constrained_frontiers.get("entries")
    if not isinstance(entries, list):
        return
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("constraint"), dict):
            continue
        path = f"constrained_frontiers.entries[{index}]"
        constraint = {
            "latency_ms": _coerce_number(entry["constraint"].get("latency_ms")),
            "cost_usd": _coerce_number(entry["constraint"].get("cost_usd")),
        }
        constrained = [
            system for system in eligible
            if _within_constraint(system, constraint)
        ]
        expected_frontier = _pareto_frontier_names(
            constrained,
            latency_key=DEFAULT_LATENCY_AXIS,
            cost_key=DEFAULT_COST_AXIS,
            quality_key=DEFAULT_QUALITY_AXIS,
        )
        expected_eligible = [system["name"] for system in constrained]
        _validate_exact_name_list(
            entry.get("frontier"),
            expected_frontier,
            path=f"{path}.frontier",
            issues=issues,
            message="must match recomputed constrained frontier",
        )
        _validate_exact_name_list(
            entry.get("eligible_systems"),
            expected_eligible,
            path=f"{path}.eligible_systems",
            issues=issues,
            message="must match recomputed constrained eligible systems",
        )


def _validate_recomputed_domain_frontiers(
    domain_frontiers: Any,
    systems: list[dict[str, Any]],
    issues: list[FrontierIssue],
) -> None:
    if not isinstance(domain_frontiers, dict):
        return
    expected = build_domain_frontiers(systems)
    if domain_frontiers != expected:
        issues.append(
            FrontierIssue(
                "domain_frontiers",
                "must match recomputed domain frontiers",
            )
        )


def _validate_recomputed_utility_view(
    utility_view: Any,
    systems: dict[str, Any],
    issues: list[FrontierIssue],
) -> None:
    if utility_view is None or not isinstance(utility_view, dict):
        return
    weights = utility_view.get("weights")
    if not isinstance(weights, dict):
        return
    weight_values = {
        key: _coerce_number(weights.get(key))
        for key in ("quality", "latency", "cost")
    }
    if any(value is None for value in weight_values.values()):
        return
    semantic_systems = _semantic_frontier_systems(systems)
    try:
        expected = build_utility_view(
            semantic_systems,
            {
                key: value
                for key, value in weight_values.items()
                if value is not None
            },
        )
    except (KeyError, TypeError):
        return
    if utility_view != expected:
        issues.append(
            FrontierIssue(
                "utility_view",
                "must match recomputed utility view for declared weights",
            )
        )


def _validate_exact_name_list(
    actual: Any,
    expected: list[str],
    *,
    path: str,
    issues: list[FrontierIssue],
    message: str,
) -> None:
    if isinstance(actual, list) and actual != expected:
        issues.append(FrontierIssue(path, message))


def _validate_scorecard(
    name: str,
    scorecard: Any,
    *,
    path: str,
    issues: list[FrontierIssue],
) -> None:
    del name
    if not isinstance(scorecard, dict):
        issues.append(FrontierIssue(path, "must be an object"))
        return

    required = (
        "p50_v2v_ttfb_ms",
        "p90_v2v_ttfb_ms",
        "p95_v2v_ttfb_ms",
        "p99_v2v_ttfb_ms",
        "p95_v2v_last_byte_ms",
        "barge_in_stop_p95_ms",
        "interruption_recovery_p95_ms",
        "latency_at_100_concurrency_p95_ms",
        "latency_measurement",
        "latency_load",
        "cost_provenance",
        "cost_usd_per_conversation",
        "cost_usd_per_successful_conversation",
        "task_success_rate",
        "experience_score",
        "experience_score_source",
        "experience_evidence",
        "axis_confidence_intervals",
    )
    for field in required:
        if field not in scorecard:
            issues.append(FrontierIssue(f"{path}.{field}", "missing required field"))

    for field in (
        "p50_v2v_ttfb_ms",
        "p90_v2v_ttfb_ms",
        "p95_v2v_ttfb_ms",
        "p99_v2v_ttfb_ms",
        "p95_v2v_last_byte_ms",
        "barge_in_stop_p95_ms",
        "interruption_recovery_p95_ms",
        "latency_at_100_concurrency_p95_ms",
        "cost_usd_per_conversation",
        "cost_usd_per_successful_conversation",
    ):
        _validate_optional_number(scorecard.get(field), path=f"{path}.{field}", issues=issues)
        _validate_nonnegative(scorecard.get(field), path=f"{path}.{field}", issues=issues)

    for field in ("task_success_rate", "experience_score", "overall_quality_score"):
        if field in scorecard:
            _validate_range(scorecard.get(field), path=f"{path}.{field}", low=0.0, high=1.0, issues=issues, allow_none=True)

    source = scorecard.get("experience_score_source")
    if source is not None and source not in {"judged", "proxy"}:
        issues.append(FrontierIssue(f"{path}.experience_score_source", "must be judged, proxy, or null"))
    _validate_experience_evidence(
        scorecard.get("experience_evidence"),
        path=f"{path}.experience_evidence",
        issues=issues,
    )

    _validate_latency_measurement(
        scorecard.get("latency_measurement"),
        path=f"{path}.latency_measurement",
        issues=issues,
    )
    _validate_latency_load(
        scorecard.get("latency_load"),
        path=f"{path}.latency_load",
        issues=issues,
    )
    _validate_cost_provenance(
        scorecard.get("cost_provenance"),
        path=f"{path}.cost_provenance",
        issues=issues,
    )
    if "axis_confidence_intervals" in scorecard:
        _validate_axis_intervals(
            scorecard.get("axis_confidence_intervals"),
            path=f"{path}.axis_confidence_intervals",
            issues=issues,
        )


def _validate_axis_intervals(
    intervals: Any,
    *,
    path: str,
    issues: list[FrontierIssue],
) -> None:
    if not isinstance(intervals, dict):
        issues.append(FrontierIssue(path, "must be an object"))
        return
    required = (
        DEFAULT_LATENCY_AXIS,
        "cost_usd_per_conversation",
        DEFAULT_COST_AXIS,
        DEFAULT_QUALITY_AXIS,
    )
    for axis in required:
        item = intervals.get(axis)
        item_path = f"{path}.{axis}"
        if axis not in intervals:
            issues.append(FrontierIssue(item_path, "missing required field"))
            continue
        if not isinstance(item, dict):
            issues.append(FrontierIssue(item_path, "must be an object"))
            continue
        if "method" not in item:
            issues.append(FrontierIssue(f"{item_path}.method", "missing required field"))
        elif not isinstance(item.get("method"), str) or not item["method"].strip():
            issues.append(FrontierIssue(f"{item_path}.method", "must be a non-empty string"))
        for field in ("estimate", "low", "high"):
            _validate_optional_number(item.get(field), path=f"{item_path}.{field}", issues=issues)
            _validate_nonnegative(item.get(field), path=f"{item_path}.{field}", issues=issues)
        _validate_confidence_level(
            item.get("confidence"),
            path=f"{item_path}.confidence",
            issues=issues,
        )
        _validate_interval_order(item, path=item_path, issues=issues)
        if axis == DEFAULT_QUALITY_AXIS:
            for field in ("estimate", "low", "high"):
                _validate_range(
                    item.get(field),
                    path=f"{item_path}.{field}",
                    low=0.0,
                    high=1.0,
                    issues=issues,
                    allow_none=True,
                )
        n = item.get("n")
        if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 0):
            issues.append(FrontierIssue(f"{item_path}.n", "must be a nonnegative integer"))


def _validate_interval_order(
    item: dict[str, Any],
    *,
    path: str,
    issues: list[FrontierIssue],
) -> None:
    low = _coerce_number(item.get("low"))
    estimate = _coerce_number(item.get("estimate"))
    high = _coerce_number(item.get("high"))
    if low is not None and high is not None and low > high:
        issues.append(FrontierIssue(path, "low must be less than or equal to high"))
    if low is not None and estimate is not None and estimate < low:
        issues.append(FrontierIssue(f"{path}.estimate", "must be greater than or equal to low"))
    if estimate is not None and high is not None and estimate > high:
        issues.append(FrontierIssue(f"{path}.estimate", "must be less than or equal to high"))


def _validate_confidence_level(
    value: Any,
    *,
    path: str,
    issues: list[FrontierIssue],
) -> None:
    number = _coerce_number(value)
    if number is None:
        issues.append(FrontierIssue(path, "must be numeric"))
    elif number <= 0.0 or number >= 1.0:
        issues.append(FrontierIssue(path, "must be greater than 0.0 and less than 1.0"))


def _validate_experience_evidence(
    evidence: Any,
    *,
    path: str,
    issues: list[FrontierIssue],
) -> None:
    if not isinstance(evidence, dict):
        issues.append(FrontierIssue(path, "must be an object"))
        return
    _validate_range(
        evidence.get("coverage"),
        path=f"{path}.coverage",
        low=0.0,
        high=1.0,
        issues=issues,
        allow_none=True,
    )
    for field in ("num_judged_trials", "num_trials"):
        value = evidence.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            issues.append(FrontierIssue(f"{path}.{field}", "must be a nonnegative integer"))
    judged = evidence.get("num_judged_trials")
    total = evidence.get("num_trials")
    if (
        isinstance(judged, int)
        and not isinstance(judged, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and judged > total
    ):
        issues.append(FrontierIssue(f"{path}.num_judged_trials", "must not exceed num_trials"))
    judge_counts = evidence.get("judge_counts")
    if not isinstance(judge_counts, dict):
        issues.append(FrontierIssue(f"{path}.judge_counts", "must be an object"))
    else:
        for judge, count in judge_counts.items():
            if not isinstance(judge, str) or not judge:
                issues.append(FrontierIssue(f"{path}.judge_counts", "judge keys must be non-empty strings"))
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                issues.append(FrontierIssue(f"{path}.judge_counts.{judge}", "must be a nonnegative integer"))


def _validate_latency_measurement(
    measurement: Any,
    *,
    path: str,
    issues: list[FrontierIssue],
) -> None:
    if not isinstance(measurement, dict):
        issues.append(FrontierIssue(path, "must be an object"))
        return
    integer_fields = (
        "sample_count",
        "event_stream_samples",
        "reported_latency_samples",
        "runtime_fallback_samples",
        "unknown_samples",
        "vad_origin_samples",
        "first_audio_event_samples",
        "last_audio_event_samples",
        "barge_in_stop_event_samples",
        "interruption_recovery_event_samples",
    )
    for field in integer_fields:
        value = measurement.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            issues.append(FrontierIssue(f"{path}.{field}", "must be a nonnegative integer"))
    source_counts = measurement.get("source_counts")
    if not isinstance(source_counts, dict):
        issues.append(FrontierIssue(f"{path}.source_counts", "must be an object"))
    else:
        for source in ("event_stream", "reported_latency", "runtime_fallback", "unknown"):
            value = source_counts.get(source)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                issues.append(FrontierIssue(f"{path}.source_counts.{source}", "must be a nonnegative integer"))


def _validate_latency_load(
    load: Any,
    *,
    path: str,
    issues: list[FrontierIssue],
) -> None:
    if not isinstance(load, dict):
        issues.append(FrontierIssue(path, "must be an object"))
        return
    levels = load.get("levels")
    if not isinstance(levels, dict):
        issues.append(FrontierIssue(f"{path}.levels", "must be an object"))
        return
    for level, item in levels.items():
        item_path = f"{path}.levels.{level}"
        if not isinstance(item, dict):
            issues.append(FrontierIssue(item_path, "must be an object"))
            continue
        target = item.get("target_concurrency")
        if not isinstance(target, int) or isinstance(target, bool) or target < 1:
            issues.append(FrontierIssue(f"{item_path}.target_concurrency", "must be a positive integer"))
        sample_count = item.get("sample_count")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 0:
            issues.append(FrontierIssue(f"{item_path}.sample_count", "must be a nonnegative integer"))
        saturated = item.get("saturated")
        if not isinstance(saturated, bool):
            issues.append(FrontierIssue(f"{item_path}.saturated", "must be boolean"))
        _validate_optional_number(
            item.get("p95_v2v_ttfb_ms"),
            path=f"{item_path}.p95_v2v_ttfb_ms",
            issues=issues,
        )
        _validate_nonnegative(
            item.get("p95_v2v_ttfb_ms"),
            path=f"{item_path}.p95_v2v_ttfb_ms",
            issues=issues,
        )
        for field in (
            "requested_calls",
            "completed_calls",
            "error_calls",
            "peak_active_calls",
        ):
            if field in item and item[field] is not None:
                value = item[field]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    issues.append(FrontierIssue(f"{item_path}.{field}", "must be a nonnegative integer"))
        for field in ("error_rate", "wall_seconds"):
            if field in item and item[field] is not None:
                _validate_nonnegative(item[field], path=f"{item_path}.{field}", issues=issues)
        requested = item.get("requested_calls")
        completed = item.get("completed_calls")
        errors = item.get("error_calls")
        peak = item.get("peak_active_calls")
        if (
            isinstance(requested, int)
            and not isinstance(requested, bool)
            and isinstance(completed, int)
            and not isinstance(completed, bool)
            and completed > requested
        ):
            issues.append(FrontierIssue(f"{item_path}.completed_calls", "must not exceed requested_calls"))
        if (
            isinstance(requested, int)
            and not isinstance(requested, bool)
            and isinstance(errors, int)
            and not isinstance(errors, bool)
            and errors > requested
        ):
            issues.append(FrontierIssue(f"{item_path}.error_calls", "must not exceed requested_calls"))
        if (
            isinstance(completed, int)
            and not isinstance(completed, bool)
            and isinstance(sample_count, int)
            and not isinstance(sample_count, bool)
            and sample_count > completed
        ):
            issues.append(FrontierIssue(f"{item_path}.sample_count", "must not exceed completed_calls"))
        if (
            isinstance(target, int)
            and not isinstance(target, bool)
            and isinstance(peak, int)
            and not isinstance(peak, bool)
            and peak > target
        ):
            issues.append(FrontierIssue(f"{item_path}.peak_active_calls", "must not exceed target_concurrency"))


def _validate_cost_provenance(
    provenance: Any,
    *,
    path: str,
    issues: list[FrontierIssue],
) -> None:
    if not isinstance(provenance, dict):
        issues.append(FrontierIssue(path, "must be an object"))
        return
    integer_fields = (
        "sample_count",
        "direct_cost_samples",
        "derived_cost_samples",
        "fully_loaded_samples",
        "missing_cost_samples",
    )
    for field in integer_fields:
        value = provenance.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            issues.append(FrontierIssue(f"{path}.{field}", "must be a nonnegative integer"))
    required_components = provenance.get("required_components")
    if not isinstance(required_components, list) or any(
        component not in COST_COMPONENTS for component in required_components
    ):
        issues.append(FrontierIssue(f"{path}.required_components", "must be a list of supported components"))
    component_counts = provenance.get("component_sample_counts")
    if not isinstance(component_counts, dict):
        issues.append(FrontierIssue(f"{path}.component_sample_counts", "must be an object"))
    else:
        for component in COST_COMPONENTS:
            value = component_counts.get(component)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                issues.append(FrontierIssue(f"{path}.component_sample_counts.{component}", "must be a nonnegative integer"))
    if provenance.get("pricing_source") not in {"profile", "embedded", "none"}:
        issues.append(FrontierIssue(f"{path}.pricing_source", "must be profile, embedded, or none"))
    pipeline_type = provenance.get("pipeline_type")
    if pipeline_type is not None and pipeline_type not in PIPELINE_REQUIRED_COMPONENTS:
        issues.append(FrontierIssue(f"{path}.pipeline_type", "must be a supported pipeline type or null"))


def _validate_projection_frontiers(
    projection_frontiers: Any,
    *,
    known_names: set[str],
    issues: list[FrontierIssue],
    path: str = "projection_frontiers",
) -> None:
    if not isinstance(projection_frontiers, dict):
        issues.append(FrontierIssue(path, "must be an object"))
        return
    for field in ("latency_vs_quality", "cost_vs_quality"):
        _validate_name_list(
            projection_frontiers.get(field),
            path=f"{path}.{field}",
            known_names=known_names,
            issues=issues,
        )


def _validate_constrained_frontiers(
    constrained_frontiers: Any,
    *,
    known_names: set[str],
    issues: list[FrontierIssue],
) -> None:
    if not isinstance(constrained_frontiers, dict):
        issues.append(FrontierIssue("constrained_frontiers", "must be an object"))
        return
    entries = constrained_frontiers.get("entries")
    if not isinstance(entries, list):
        issues.append(FrontierIssue("constrained_frontiers.entries", "must be a list"))
        return
    for index, entry in enumerate(entries):
        path = f"constrained_frontiers.entries[{index}]"
        if not isinstance(entry, dict):
            issues.append(FrontierIssue(path, "must be an object"))
            continue
        constraint = entry.get("constraint")
        if not isinstance(constraint, dict):
            issues.append(FrontierIssue(f"{path}.constraint", "must be an object"))
        else:
            _validate_optional_number(constraint.get("latency_ms"), path=f"{path}.constraint.latency_ms", issues=issues)
            _validate_optional_number(constraint.get("cost_usd"), path=f"{path}.constraint.cost_usd", issues=issues)
        _validate_name_list(
            entry.get("frontier"),
            path=f"{path}.frontier",
            known_names=known_names,
            issues=issues,
        )
        _validate_name_list(
            entry.get("eligible_systems"),
            path=f"{path}.eligible_systems",
            known_names=known_names,
            issues=issues,
        )


def _validate_domain_frontiers(
    domain_frontiers: Any,
    *,
    known_names: set[str],
    issues: list[FrontierIssue],
) -> None:
    if not isinstance(domain_frontiers, dict):
        issues.append(FrontierIssue("domain_frontiers", "must be an object"))
        return
    for domain, entry in domain_frontiers.items():
        if not isinstance(domain, str) or not domain:
            issues.append(FrontierIssue("domain_frontiers", "domain names must be non-empty strings"))
            continue
        path = f"domain_frontiers.{domain}"
        if not isinstance(entry, dict):
            issues.append(FrontierIssue(path, "must be an object"))
            continue
        _validate_name_list(
            entry.get("frontier"),
            path=f"{path}.frontier",
            known_names=known_names,
            issues=issues,
        )
        _validate_projection_frontiers(
            entry.get("projection_frontiers"),
            known_names=known_names,
            issues=issues,
            path=f"{path}.projection_frontiers",
        )
        _validate_name_list(
            entry.get("eligible_systems"),
            path=f"{path}.eligible_systems",
            known_names=known_names,
            issues=issues,
        )
        _validate_name_list(
            entry.get("represented_systems"),
            path=f"{path}.represented_systems",
            known_names=known_names,
            issues=issues,
        )
        count = entry.get("num_eligible_systems")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            issues.append(
                FrontierIssue(f"{path}.num_eligible_systems", "must be a nonnegative integer")
            )


def _validate_utility_view(
    utility_view: Any,
    *,
    known_names: set[str],
    issues: list[FrontierIssue],
) -> None:
    if utility_view is None:
        return
    if not isinstance(utility_view, dict):
        issues.append(FrontierIssue("utility_view", "must be an object or null"))
        return
    weights = utility_view.get("weights")
    if not isinstance(weights, dict):
        issues.append(FrontierIssue("utility_view.weights", "must be an object"))
    else:
        for field in ("quality", "latency", "cost"):
            _validate_optional_number(
                weights.get(field),
                path=f"utility_view.weights.{field}",
                issues=issues,
            )
    _validate_name_list(
        utility_view.get("ranking"),
        path="utility_view.ranking",
        known_names=known_names,
        issues=issues,
    )
    scores = utility_view.get("scores")
    if not isinstance(scores, dict):
        issues.append(FrontierIssue("utility_view.scores", "must be an object"))
    else:
        for name, score in scores.items():
            if not isinstance(name, str) or not name:
                issues.append(FrontierIssue("utility_view.scores", "system names must be non-empty strings"))
                continue
            if known_names and name not in known_names:
                issues.append(FrontierIssue(f"utility_view.scores.{name}", "unknown system name"))
            _validate_optional_number(
                score,
                path=f"utility_view.scores.{name}",
                issues=issues,
            )
    note = utility_view.get("note")
    if not isinstance(note, str) or "not a benchmark leaderboard" not in note:
        issues.append(FrontierIssue("utility_view.note", "must identify utility as secondary"))


def _validate_name_list(
    value: Any,
    *,
    path: str,
    known_names: set[str],
    issues: list[FrontierIssue],
) -> None:
    if not isinstance(value, list):
        issues.append(FrontierIssue(path, "must be a list"))
        return
    seen = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, str) or not item:
            issues.append(FrontierIssue(item_path, "must be a non-empty string"))
            continue
        if item in seen:
            issues.append(FrontierIssue(item_path, "duplicate system name"))
        seen.add(item)
        if known_names and item not in known_names:
            issues.append(FrontierIssue(item_path, "unknown system name"))


def _validate_optional_number(value: Any, *, path: str, issues: list[FrontierIssue]) -> None:
    if value is not None and _coerce_number(value) is None:
        issues.append(FrontierIssue(path, "must be numeric"))


def _validate_nonnegative(value: Any, *, path: str, issues: list[FrontierIssue]) -> None:
    number = _coerce_number(value)
    if number is not None and number < 0:
        issues.append(FrontierIssue(path, "must be nonnegative"))


def _validate_range(
    value: Any,
    *,
    path: str,
    low: float,
    high: float,
    issues: list[FrontierIssue],
    allow_none: bool = False,
) -> None:
    number = _coerce_number(value)
    if number is None:
        if value is None and allow_none:
            return
        issues.append(FrontierIssue(path, "must be numeric"))
        return
    if number < low or number > high:
        issues.append(FrontierIssue(path, f"must be between {low} and {high}"))


def build_utility_view(
    systems: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Build an optional user-weighted utility view.

    Utility is intentionally opt-in: quality - lambda*latency - mu*cost after
    min-max normalization of present eligible values.
    """
    quality_weight = float(weights.get("quality", 1.0))
    latency_weight = float(weights.get("latency", 0.0))
    cost_weight = float(weights.get("cost", 0.0))
    eligible = [system for system in systems if system["frontier_eligible"]]
    latency_scores = _inverse_minmax(
        {system["name"]: system["axes"][DEFAULT_LATENCY_AXIS] for system in eligible}
    )
    cost_scores = _inverse_minmax(
        {system["name"]: system["axes"][DEFAULT_COST_AXIS] for system in eligible}
    )
    quality_scores = {
        system["name"]: system["axes"][DEFAULT_QUALITY_AXIS]
        for system in eligible
    }

    utility = {}
    for system in eligible:
        name = system["name"]
        score = (
            quality_weight * quality_scores[name]
            + latency_weight * (latency_scores[name] or 0.0)
            + cost_weight * (cost_scores[name] or 0.0)
        )
        utility[name] = round(score, 6)

    ranking = sorted(utility, key=lambda name: (-utility[name], name))
    return {
        "weights": {
            "quality": quality_weight,
            "latency": latency_weight,
            "cost": cost_weight,
        },
        "ranking": ranking,
        "scores": utility,
        "note": "Secondary opt-in utility view; not a benchmark leaderboard.",
    }


def _normalize_system_report(
    report: dict[str, Any],
    *,
    index: int,
    pricing_manifest: dict[str, Any] | None,
    pricing_snapshot_date: str | None,
    experience_gate: float,
) -> dict[str, Any]:
    name = _system_name(report, index)
    quality = _quality_metrics(report)
    latency = _latency_metrics(report)
    resolved_pricing = resolve_report_pricing(report, pricing_manifest)
    cost = _cost_metrics(report, quality["task_success_rate"], resolved_pricing)
    axis_intervals = _axis_confidence_intervals(
        report,
        task_success_rate=quality["task_success_rate"],
        resolved_pricing=resolved_pricing,
    )
    pricing_date = (
        _pricing_snapshot_date(report)
        or resolved_pricing.get("snapshot_date")
        or pricing_snapshot_date
    )
    domain_axes = _domain_axes(report, resolved_pricing)

    axes = {
        DEFAULT_LATENCY_AXIS: latency["p95_v2v_ttfb_ms"],
        DEFAULT_COST_AXIS: cost["cost_usd_per_successful_conversation"],
        DEFAULT_QUALITY_AXIS: quality["task_success_rate"],
    }
    missing_axes = [
        key for key, value in axes.items()
        if value is None or (isinstance(value, float) and not isfinite(value))
    ]
    exclusion_reasons = []
    if missing_axes:
        exclusion_reasons.append({"type": "missing_primary_axis", "axes": missing_axes})
    if quality["experience_score"] is None:
        exclusion_reasons.append({"type": "missing_experience_gate_score"})
    elif quality["experience_score"] < experience_gate:
        exclusion_reasons.append({
            "type": "below_experience_gate",
            "score": quality["experience_score"],
            "gate": experience_gate,
        })
    if not pricing_date:
        exclusion_reasons.append({"type": "missing_pricing_snapshot_date"})

    scorecard = {
        "p50_v2v_ttfb_ms": latency["p50_v2v_ttfb_ms"],
        "p90_v2v_ttfb_ms": latency["p90_v2v_ttfb_ms"],
        "p95_v2v_ttfb_ms": latency["p95_v2v_ttfb_ms"],
        "p99_v2v_ttfb_ms": latency["p99_v2v_ttfb_ms"],
        "p50_v2v_last_byte_ms": latency["p50_v2v_last_byte_ms"],
        "p95_v2v_last_byte_ms": latency["p95_v2v_last_byte_ms"],
        "barge_in_stop_p95_ms": latency["barge_in_stop_p95_ms"],
        "interruption_recovery_p95_ms": latency["interruption_recovery_p95_ms"],
        "latency_at_100_concurrency_p95_ms": latency["latency_at_100_concurrency_p95_ms"],
        "stage_latency_ms": latency["stage_latency_ms"],
        "latency_measurement": latency["latency_measurement"],
        "latency_load": latency["latency_load"],
        "cost_usd_per_conversation": cost["cost_usd_per_conversation"],
        "cost_usd_per_successful_conversation": cost["cost_usd_per_successful_conversation"],
        "avg_component_cost_usd": cost["avg_component_cost_usd"],
        "cost_provenance": cost["cost_provenance"],
        "task_success_rate": quality["task_success_rate"],
        "experience_score": quality["experience_score"],
        "experience_score_source": quality["experience_score_source"],
        "experience_evidence": _experience_evidence_summary(report),
        "overall_quality_score": quality["overall_quality_score"],
        "pricing_snapshot_date": pricing_date,
        "num_scenarios": report.get("num_scenarios") or report.get("num_scored"),
        "axis_confidence_intervals": axis_intervals,
    }

    return {
        "name": name,
        "benchmark": report.get("benchmark", report.get("benchmark_name", "unknown")),
        "benchmark_version": report.get("benchmark_version"),
        "model_metadata": report.get("model_metadata", {}),
        "axes": axes,
        "latency": latency,
        "cost": cost,
        "quality": quality,
        "resolved_pricing": _public_pricing_metadata(resolved_pricing),
        "domain_axes": domain_axes,
        "scorecard": scorecard,
        "frontier_eligible": not exclusion_reasons,
        "exclusion_reasons": exclusion_reasons,
        "confidence_intervals": {
            **(report.get("confidence_intervals", {}) or {}),
            "frontier_axes": axis_intervals,
        },
    }


def _quality_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    metrics = report.get("metric_scores", {})
    task_success = _first_number(
        metrics.get("task_success"),
        report.get("mean_pass_rate"),
        report.get("pass_rate"),
    )
    judged_experience = _first_number(report.get("conversation_experience_score"))
    proxy_experience = _first_number(
        metrics.get("experience_proxy"),
        _dimension_score(report, "natural_tone", divisor=10.0),
    )
    experience = _first_number(judged_experience, proxy_experience)
    source = "judged" if judged_experience is not None else "proxy"
    overall = _first_number(report.get("overall_score"))
    if overall is not None and overall > 1.0:
        overall = overall / 100.0 if overall > 10.0 else overall / 10.0
    return {
        "task_success_rate": _round_optional(task_success, 6),
        "experience_score": _round_optional(experience, 6),
        "experience_score_source": source if experience is not None else None,
        "overall_quality_score": _round_optional(overall, 6),
    }


def _experience_evidence_summary(report: dict[str, Any]) -> dict[str, Any]:
    evidence = report.get("conversation_experience")
    if not isinstance(evidence, dict):
        return {
            "coverage": 0.0,
            "num_judged_trials": 0,
            "num_trials": _report_trial_count(report),
            "judge_counts": {},
        }
    return {
        "coverage": _round_optional(_first_number(evidence.get("coverage")), 6),
        "num_judged_trials": _nonnegative_int_or_none(evidence.get("num_judged_trials")) or 0,
        "num_trials": _report_trial_count(report),
        "judge_counts": (
            dict(evidence.get("judge_counts"))
            if isinstance(evidence.get("judge_counts"), dict)
            else {}
        ),
    }


def _latency_metrics(report: dict[str, Any]) -> dict[str, Any]:
    samples = _collect_latency_samples(report)
    ttfb = _metric_samples(samples, "v2v_ttfb_ms")
    last_byte = _metric_samples(samples, "v2v_last_byte_ms")
    barge_in_stop = _metric_samples(samples, "barge_in_stop_ms")
    interruption = _metric_samples(samples, "interruption_recovery_ms")
    operational = report.get("operational_metrics", {})

    p50_ttfb = _first_number(_percentile(ttfb, 50), operational.get("median_latency_ms"))
    p95_ttfb = _first_number(_percentile(ttfb, 95), operational.get("p95_latency_ms"))
    p90_ttfb = _percentile(ttfb, 90)
    p99_ttfb = _percentile(ttfb, 99)
    load = _latency_at_concurrency(report, 100)

    return {
        "p50_v2v_ttfb_ms": _round_optional(p50_ttfb, 3),
        "p90_v2v_ttfb_ms": _round_optional(p90_ttfb, 3),
        "p95_v2v_ttfb_ms": _round_optional(p95_ttfb, 3),
        "p99_v2v_ttfb_ms": _round_optional(p99_ttfb, 3),
        "p50_v2v_last_byte_ms": _round_optional(_percentile(last_byte, 50), 3),
        "p95_v2v_last_byte_ms": _round_optional(_percentile(last_byte, 95), 3),
        "barge_in_stop_p95_ms": _round_optional(_percentile(barge_in_stop, 95), 3),
        "interruption_recovery_p95_ms": _round_optional(_percentile(interruption, 95), 3),
        "latency_at_100_concurrency_p95_ms": _round_optional(load, 3),
        "stage_latency_ms": _stage_latency_summary(samples),
        "latency_measurement": _latency_measurement_summary(samples),
        "latency_load": _latency_load_summary(report),
        "sample_count": len(ttfb),
    }


def _cost_metrics(
    report: dict[str, Any],
    task_success_rate: float | None,
    resolved_pricing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operational = report.get("operational_metrics", {})
    samples, component_samples = _collect_cost_samples(report, resolved_pricing)
    cost_provenance = _cost_provenance_summary(report, resolved_pricing)
    avg_cost = _first_number(
        _mean(samples),
        operational.get("avg_cost_usd"),
    )
    if avg_cost is None and operational.get("total_cost_usd") is not None:
        denominator = report.get("num_scored") or report.get("num_scenarios")
        if denominator:
            avg_cost = operational["total_cost_usd"] / denominator

    success_cost = None
    if avg_cost is not None and task_success_rate is not None and task_success_rate > 0:
        success_cost = avg_cost / task_success_rate

    return {
        "cost_usd_per_conversation": _round_optional(avg_cost, 6),
        "cost_usd_per_successful_conversation": _round_optional(success_cost, 6),
        "avg_component_cost_usd": {
            component: _round_optional(_mean(values), 6)
            for component, values in component_samples.items()
        },
        "cost_provenance": cost_provenance,
        "sample_count": len(samples),
    }


def _axis_confidence_intervals(
    report: dict[str, Any],
    *,
    task_success_rate: float | None,
    resolved_pricing: dict[str, Any],
) -> dict[str, Any]:
    latency_samples = _metric_samples(_collect_latency_samples(report), "v2v_ttfb_ms")
    cost_samples, _ = _collect_cost_samples(report, resolved_pricing)
    return {
        DEFAULT_LATENCY_AXIS: _bootstrap_interval(
            latency_samples,
            statistic=lambda values: _percentile(values, 95),
            ndigits=3,
        ),
        "cost_usd_per_conversation": _bootstrap_interval(
            cost_samples,
            statistic=_mean,
            ndigits=6,
        ),
        DEFAULT_COST_AXIS: _bootstrap_interval(
            cost_samples,
            statistic=(
                (lambda values: (_mean(values) / task_success_rate))
                if task_success_rate and task_success_rate > 0
                else (lambda _values: None)
            ),
            ndigits=6,
        ),
        DEFAULT_QUALITY_AXIS: _task_success_interval(report),
    }


def _domain_axes(
    report: dict[str, Any],
    resolved_pricing: dict[str, Any] | None,
) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in report.get("results", []):
        domain = result.get("domain") or result.get("category")
        if not domain:
            continue
        grouped.setdefault(str(domain), []).append(result)

    axes = {}
    for domain, results in grouped.items():
        subreport = {**report, "results": results, "num_scenarios": len(results)}
        quality = _domain_quality(results)
        latency = _latency_metrics(subreport)
        cost = _cost_metrics(subreport, quality, resolved_pricing)
        axes[domain] = {
            DEFAULT_LATENCY_AXIS: latency["p95_v2v_ttfb_ms"],
            DEFAULT_COST_AXIS: cost["cost_usd_per_successful_conversation"],
            DEFAULT_QUALITY_AXIS: quality,
        }
    return axes


def _domain_quality(results: list[dict[str, Any]]) -> float | None:
    values = []
    for result in results:
        score = _first_number(
            result.get("avg_scores", {}).get("task_success"),
            result.get("pass_rate"),
            1.0 if result.get("pass_k") is True else None,
        )
        if score is None and isinstance(result.get("trials"), list):
            passes = [
                1.0 if trial.get("passed") else 0.0
                for trial in result["trials"]
                if "error" not in trial
            ]
            score = _mean(passes)
        if score is not None:
            values.append(score)
    return _round_optional(_mean(values), 6)


def _collect_latency_samples(report: dict[str, Any]) -> list[dict[str, Any]]:
    samples = []
    for result in report.get("results", []):
        trials = result.get("trials")
        if isinstance(trials, list):
            rows = trials
        else:
            rows = [result]
        for row in rows:
            if "error" in row:
                continue
            sample = _latency_sample_from_row(row)
            if sample:
                samples.append(sample)
    return samples


def _latency_sample_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    latency = row.get("latency") or row.get("latency_ms") or {}
    if isinstance(latency, (int, float)):
        sample = {"v2v_ttfb_ms": float(latency)}
    elif isinstance(latency, dict):
        sample = {
            "v2v_ttfb_ms": _first_number(
                latency.get("v2v_ttfb_ms"),
                latency.get("voice_to_voice_ttfb_ms"),
                latency.get("ttfb_ms"),
                row.get("latency_ms"),
            ),
            "v2v_last_byte_ms": _first_number(
                latency.get("v2v_last_byte_ms"),
                latency.get("voice_to_voice_last_byte_ms"),
                latency.get("last_byte_ms"),
            ),
            "interruption_recovery_ms": _first_number(
                latency.get("interruption_recovery_ms"),
                latency.get("barge_in_recovery_ms"),
            ),
            "barge_in_stop_ms": _first_number(
                latency.get("barge_in_stop_ms"),
                latency.get("interruption_stop_ms"),
                latency.get("stop_speaking_ms"),
            ),
        }
        stage = latency.get("stage_latency_ms") or latency.get("pipeline_ms") or {}
        if isinstance(stage, dict):
            sample["stage_latency_ms"] = stage
        measurement = latency.get("measurement") or latency.get("measurement_metadata")
        if isinstance(measurement, dict):
            sample["measurement"] = measurement
    else:
        sample = {}

    breakdown = row.get("latency_breakdown_ms") or row.get("stage_latency_ms")
    if isinstance(breakdown, dict):
        sample["stage_latency_ms"] = breakdown
    return sample if any(value is not None for value in sample.values()) else None


def _latency_measurement_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = {
        "event_stream": 0,
        "reported_latency": 0,
        "runtime_fallback": 0,
        "unknown": 0,
    }
    vad_origin_samples = 0
    first_audio_event_samples = 0
    last_audio_event_samples = 0
    barge_in_stop_event_samples = 0
    interruption_recovery_event_samples = 0
    for sample in samples:
        measurement = sample.get("measurement")
        if not isinstance(measurement, dict):
            source_counts["unknown"] += 1
            continue
        source = measurement.get("source")
        if source not in source_counts:
            source = "unknown"
        source_counts[source] += 1
        if (
            measurement.get("origin_event") == "user.end_speech"
            and _first_number(measurement.get("origin_t_ms")) == 0.0
        ):
            vad_origin_samples += 1
        if measurement.get("first_audio_event") == "tts.first_audio":
            first_audio_event_samples += 1
        if measurement.get("last_audio_event") == "agent.complete":
            last_audio_event_samples += 1
        if measurement.get("barge_in_stop_event") == "barge_in.stop":
            barge_in_stop_event_samples += 1
        if measurement.get("interruption_recovery_event") == "barge_in.recovered":
            interruption_recovery_event_samples += 1
    return {
        "sample_count": len(samples),
        "source_counts": source_counts,
        "event_stream_samples": source_counts["event_stream"],
        "reported_latency_samples": source_counts["reported_latency"],
        "runtime_fallback_samples": source_counts["runtime_fallback"],
        "unknown_samples": source_counts["unknown"],
        "vad_origin_samples": vad_origin_samples,
        "first_audio_event_samples": first_audio_event_samples,
        "last_audio_event_samples": last_audio_event_samples,
        "barge_in_stop_event_samples": barge_in_stop_event_samples,
        "interruption_recovery_event_samples": interruption_recovery_event_samples,
    }


def _stage_latency_summary(samples: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    stages = {
        "asr_finalization_ms": [],
        "llm_ttft_ms": [],
        "tts_first_chunk_ms": [],
    }
    aliases = {
        "asr_finalization_ms": ("asr_finalization_ms", "asr_ms"),
        "llm_ttft_ms": ("llm_ttft_ms", "llm_ms"),
        "tts_first_chunk_ms": ("tts_first_chunk_ms", "tts_ms"),
    }
    for sample in samples:
        stage = sample.get("stage_latency_ms")
        if not isinstance(stage, dict):
            continue
        for canonical, names in aliases.items():
            value = _first_number(*(stage.get(name) for name in names))
            if value is not None:
                stages[canonical].append(value)

    return {
        stage: {
            "p50": _round_optional(_percentile(values, 50), 3),
            "p95": _round_optional(_percentile(values, 95), 3),
        }
        for stage, values in stages.items()
    }


def _latency_at_concurrency(report: dict[str, Any], concurrency: int) -> float | None:
    operational = report.get("operational_metrics", {})
    direct = _first_number(
        operational.get(f"latency_at_{concurrency}_concurrency_p95_ms"),
        operational.get(f"p95_latency_ms_at_{concurrency}_concurrency"),
    )
    if direct is not None:
        return direct
    load = operational.get("load") or operational.get("concurrency")
    if isinstance(load, dict):
        item = load.get(str(concurrency)) or load.get(concurrency)
        if isinstance(item, dict):
            return _first_number(item.get("p95_latency_ms"), item.get("p95_v2v_ttfb_ms"), item.get("p95"))
    return None


def _latency_load_summary(report: dict[str, Any]) -> dict[str, Any]:
    operational = report.get("operational_metrics", {})
    load = operational.get("load") or operational.get("concurrency") or {}
    levels: dict[str, Any] = {}
    if isinstance(load, dict):
        for raw_level, item in load.items():
            try:
                target = int(raw_level)
            except (TypeError, ValueError):
                continue
            if target < 1 or not isinstance(item, dict):
                continue
            sample_count = item.get("count", item.get("sample_count", 0))
            if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 0:
                sample_count = 0
            levels[str(target)] = {
                "target_concurrency": target,
                "sample_count": sample_count,
                "saturated": (
                    item["saturated"]
                    if isinstance(item.get("saturated"), bool)
                    else sample_count >= target
                ),
                "p95_v2v_ttfb_ms": _round_optional(
                    _first_number(
                        item.get("p95_v2v_ttfb_ms"),
                        item.get("p95_latency_ms"),
                        item.get("p95"),
                    ),
                    3,
                ),
                "requested_calls": _nonnegative_int_or_none(item.get("requested_calls")),
                "completed_calls": _nonnegative_int_or_none(item.get("completed_calls")),
                "error_calls": _nonnegative_int_or_none(item.get("error_calls")),
                "error_rate": _round_optional(_first_number(item.get("error_rate")), 6),
                "peak_active_calls": _nonnegative_int_or_none(item.get("peak_active_calls")),
                "wall_seconds": _round_optional(_first_number(item.get("wall_seconds")), 3),
            }
    return {"levels": dict(sorted(levels.items(), key=lambda item: int(item[0])))}


def _collect_cost_samples(
    report: dict[str, Any],
    resolved_pricing: dict[str, Any] | None = None,
) -> tuple[list[float], dict[str, list[float]]]:
    pricing = resolved_pricing or report.get("pricing") or report.get("model_metadata", {}).get("pricing") or {}
    samples = []
    component_samples: dict[str, list[float]] = {
        "asr": [],
        "llm": [],
        "tts": [],
        "speech_to_speech": [],
        "telephony": [],
        "transport": [],
    }
    for result in report.get("results", []):
        trials = result.get("trials")
        rows = trials if isinstance(trials, list) else [result]
        for row in rows:
            direct = _coerce_number(row.get("cost_usd"))
            row_pricing = row.get("pricing") or pricing
            derived = _derive_cost_usd(row.get("usage") or {}, row_pricing)
            if direct is not None:
                samples.append(direct)
            elif derived["total"] is not None:
                samples.append(derived["total"])
            for component, value in derived["components"].items():
                if value is not None:
                    component_samples[component].append(value)
    return samples, component_samples


def _cost_provenance_summary(
    report: dict[str, Any],
    resolved_pricing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pricing = resolved_pricing or report.get("pricing") or report.get("model_metadata", {}).get("pricing") or {}
    required_components = _required_cost_components(pricing)
    component_counts = {component: 0 for component in COST_COMPONENTS}
    sample_count = 0
    direct_cost_samples = 0
    derived_cost_samples = 0
    fully_loaded_samples = 0
    missing_cost_samples = 0

    for result in report.get("results", []):
        trials = result.get("trials")
        rows = trials if isinstance(trials, list) else [result]
        for row in rows:
            if "error" in row:
                continue
            sample_count += 1
            direct = _coerce_number(row.get("cost_usd"))
            row_pricing = row.get("pricing") or pricing
            derived = _derive_cost_usd(row.get("usage") or {}, row_pricing)
            if direct is not None:
                direct_cost_samples += 1
            if derived["total"] is not None:
                derived_cost_samples += 1
            if direct is None and derived["total"] is None:
                missing_cost_samples += 1

            present_components = {
                component for component, value in derived["components"].items()
                if value is not None
            }
            for component in present_components:
                component_counts[component] += 1
            if required_components and set(required_components).issubset(present_components):
                fully_loaded_samples += 1

    return {
        "sample_count": sample_count,
        "direct_cost_samples": direct_cost_samples,
        "derived_cost_samples": derived_cost_samples,
        "fully_loaded_samples": fully_loaded_samples,
        "missing_cost_samples": missing_cost_samples,
        "required_components": list(required_components),
        "component_sample_counts": component_counts,
        "pricing_source": _pricing_source(pricing),
        "pipeline_type": _pipeline_type(pricing),
    }


def _required_cost_components(pricing: dict[str, Any]) -> tuple[str, ...]:
    pipeline_type = _pipeline_type(pricing) or "cascaded"
    required = PIPELINE_REQUIRED_COMPONENTS.get(pipeline_type, PIPELINE_REQUIRED_COMPONENTS["cascaded"])
    return tuple(component for component in COST_COMPONENTS if component in required)


def _pricing_source(pricing: dict[str, Any]) -> str:
    if not isinstance(pricing, dict) or not pricing:
        return "none"
    if pricing.get("profile_id"):
        return "profile"
    return "embedded"


def _pipeline_type(pricing: dict[str, Any]) -> str | None:
    if not isinstance(pricing, dict):
        return None
    pipeline_type = pricing.get("pipeline_type")
    if pipeline_type in PIPELINE_REQUIRED_COMPONENTS:
        return str(pipeline_type)
    if pricing:
        return "cascaded"
    return None


def _derive_cost_usd(
    usage: dict[str, Any],
    pricing: dict[str, Any],
) -> dict[str, Any]:
    if not usage or not pricing:
        return {"total": None, "components": {}}
    components = {
        "asr": _asr_cost(usage, pricing),
        "llm": _llm_cost(usage, pricing),
        "tts": _tts_cost(usage, pricing),
        "speech_to_speech": _speech_to_speech_cost(usage, pricing),
        "telephony": _duration_cost(
            usage,
            pricing,
            usage_minute_keys=("telephony_minutes", "call_minutes"),
            usage_second_keys=("telephony_seconds", "call_duration_seconds"),
            pricing_keys=("telephony_per_minute", "phone_per_minute"),
        ),
        "transport": _duration_cost(
            usage,
            pricing,
            usage_minute_keys=("transport_minutes", "webrtc_minutes"),
            usage_second_keys=("transport_seconds",),
            pricing_keys=("transport_per_minute", "webrtc_per_minute"),
        ),
    }
    present = [value for value in components.values() if value is not None]
    return {
        "total": round(sum(present), 6) if present else None,
        "components": components,
    }


def _asr_cost(usage: dict[str, Any], pricing: dict[str, Any]) -> float | None:
    minutes = _first_number(
        usage.get("asr_minutes"),
        _seconds_to_minutes(
            _first_number(
                usage.get("asr_seconds"),
                usage.get("asr_audio_seconds"),
                usage.get("input_audio_seconds"),
            )
        ),
    )
    per_minute = _first_number(pricing.get("asr_per_minute"), pricing.get("stt_per_minute"))
    if minutes is not None and per_minute is not None:
        return minutes * per_minute
    hours = _seconds_to_hours(
        _first_number(
            usage.get("asr_seconds"),
            usage.get("asr_audio_seconds"),
            usage.get("input_audio_seconds"),
        )
    )
    per_hour = _first_number(pricing.get("asr_per_hour"), pricing.get("stt_per_hour"))
    if hours is not None and per_hour is not None:
        return hours * per_hour
    return None


def _llm_cost(usage: dict[str, Any], pricing: dict[str, Any]) -> float | None:
    input_tokens = _first_number(usage.get("input_tokens"), usage.get("prompt_tokens"))
    output_tokens = _first_number(usage.get("output_tokens"), usage.get("completion_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    cached_input_tokens = _first_number(usage.get("cached_input_tokens")) or 0.0
    uncached_input_tokens = max(0.0, input_tokens - cached_input_tokens)
    input_per_mtok = _first_number(pricing.get("input_per_mtok"), pricing.get("llm_input_per_mtok"))
    output_per_mtok = _first_number(pricing.get("output_per_mtok"), pricing.get("llm_output_per_mtok"))
    cached_per_mtok = _first_number(pricing.get("cached_input_per_mtok"), input_per_mtok)
    if input_per_mtok is None or output_per_mtok is None:
        return None
    return (
        (uncached_input_tokens / 1_000_000) * input_per_mtok
        + (cached_input_tokens / 1_000_000) * cached_per_mtok
        + (output_tokens / 1_000_000) * output_per_mtok
    )


def _tts_cost(usage: dict[str, Any], pricing: dict[str, Any]) -> float | None:
    characters = _first_number(
        usage.get("tts_characters"),
        usage.get("output_characters"),
        usage.get("characters"),
    )
    per_million_chars = _first_number(
        pricing.get("tts_per_million_characters"),
        pricing.get("tts_per_million_chars"),
    )
    if characters is not None and per_million_chars is not None:
        return (characters / 1_000_000) * per_million_chars
    per_1k_chars = _first_number(pricing.get("tts_per_1k_characters"), pricing.get("tts_per_1k_chars"))
    if characters is not None and per_1k_chars is not None:
        return (characters / 1_000) * per_1k_chars
    return _duration_cost(
        usage,
        pricing,
        usage_minute_keys=("tts_minutes", "output_audio_minutes"),
        usage_second_keys=("tts_seconds", "output_audio_seconds"),
        pricing_keys=("tts_per_minute",),
    )


def _speech_to_speech_cost(usage: dict[str, Any], pricing: dict[str, Any]) -> float | None:
    parts = [
        _duration_cost(
            usage,
            pricing,
            usage_minute_keys=(
                "speech_to_speech_minutes",
                "s2s_minutes",
                "realtime_minutes",
                "call_minutes",
            ),
            usage_second_keys=(
                "speech_to_speech_seconds",
                "s2s_seconds",
                "realtime_seconds",
                "call_duration_seconds",
            ),
            pricing_keys=(
                "speech_to_speech_per_minute",
                "s2s_per_minute",
                "realtime_per_minute",
            ),
        ),
        _duration_cost(
            usage,
            pricing,
            usage_minute_keys=("input_audio_minutes", "speech_input_minutes"),
            usage_second_keys=("input_audio_seconds", "speech_input_seconds"),
            pricing_keys=("input_audio_per_minute", "speech_input_per_minute"),
        ),
        _duration_cost(
            usage,
            pricing,
            usage_minute_keys=("output_audio_minutes", "speech_output_minutes"),
            usage_second_keys=("output_audio_seconds", "speech_output_seconds"),
            pricing_keys=("output_audio_per_minute", "speech_output_per_minute"),
        ),
        _audio_token_cost(usage, pricing),
    ]
    present = [part for part in parts if part is not None]
    return sum(present) if present else None


def _audio_token_cost(usage: dict[str, Any], pricing: dict[str, Any]) -> float | None:
    input_tokens = _first_number(usage.get("input_audio_tokens"), usage.get("audio_input_tokens"))
    output_tokens = _first_number(usage.get("output_audio_tokens"), usage.get("audio_output_tokens"))
    input_per_mtok = _first_number(
        pricing.get("input_audio_per_mtok"),
        pricing.get("audio_input_per_mtok"),
    )
    output_per_mtok = _first_number(
        pricing.get("output_audio_per_mtok"),
        pricing.get("audio_output_per_mtok"),
    )
    parts = []
    if input_tokens is not None and input_per_mtok is not None:
        parts.append((input_tokens / 1_000_000) * input_per_mtok)
    if output_tokens is not None and output_per_mtok is not None:
        parts.append((output_tokens / 1_000_000) * output_per_mtok)
    return sum(parts) if parts else None


def _duration_cost(
    usage: dict[str, Any],
    pricing: dict[str, Any],
    *,
    usage_minute_keys: tuple[str, ...],
    usage_second_keys: tuple[str, ...],
    pricing_keys: tuple[str, ...],
) -> float | None:
    minutes = _first_number(
        *(usage.get(key) for key in usage_minute_keys),
        _seconds_to_minutes(_first_number(*(usage.get(key) for key in usage_second_keys))),
    )
    per_minute = _first_number(*(pricing.get(key) for key in pricing_keys))
    if minutes is None or per_minute is None:
        return None
    return minutes * per_minute


def _seconds_to_minutes(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return seconds / 60.0


def _seconds_to_hours(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return seconds / 3600.0


def _metric_samples(samples: list[dict[str, Any]], key: str) -> list[float]:
    return [
        value for value in (_coerce_number(sample.get(key)) for sample in samples)
        if value is not None
    ]


def _report_trial_count(report: dict[str, Any]) -> int:
    total = 0
    for result in report.get("results", []):
        if isinstance(result, dict) and isinstance(result.get("trials"), list):
            total += len(result["trials"])
    return total


def _nonnegative_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _system_name(report: dict[str, Any], index: int) -> str:
    metadata = report.get("model_metadata", {})
    return (
        metadata.get("display_name")
        or metadata.get("model_id")
        or metadata.get("agent")
        or report.get("name")
        or f"submission_{index + 1}"
    )


def _pricing_snapshot_date(report: dict[str, Any]) -> str | None:
    metadata = report.get("model_metadata", {})
    pricing = metadata.get("pricing") if isinstance(metadata, dict) else None
    if isinstance(pricing, dict):
        value = pricing.get("snapshot_date") or pricing.get("date")
        if value:
            return str(value)
    return (
        report.get("pricing_snapshot_date")
        or metadata.get("pricing_snapshot_date")
        or metadata.get("price_snapshot_date")
    )


def _pricing_snapshot_metadata(
    pricing_manifest: dict[str, Any] | None,
    fallback_date: str | None,
) -> dict[str, Any]:
    if not pricing_manifest:
        return {
            "snapshot_date": fallback_date,
            "source": "fallback_argument" if fallback_date else None,
        }
    return {
        "name": pricing_manifest.get("name"),
        "version": pricing_manifest.get("version"),
        "snapshot_date": pricing_manifest.get("snapshot_date") or fallback_date,
        "currency": pricing_manifest.get("currency"),
        "num_entries": len(pricing_manifest.get("entries", [])),
        "num_profiles": len(pricing_manifest.get("profiles", [])),
    }


def _public_pricing_metadata(pricing: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "snapshot_date",
        "currency",
        "profile_id",
        "pipeline_type",
        "component_entry_ids",
    )
    return {key: pricing[key] for key in keys if key in pricing}


def _dimension_score(report: dict[str, Any], dimension: str, *, divisor: float) -> float | None:
    score = report.get("dimension_scores", {}).get(dimension)
    if score is None:
        return None
    return _coerce_number(score) / divisor


def _available_domains(frontier_report: dict[str, Any]) -> set[str]:
    domains = set()
    for system in frontier_report.get("systems", {}).values():
        domains.update(system.get("domain_axes", {}).keys())
    return domains


def _axes_for_domain(system: dict[str, Any], domain: str) -> dict[str, float | None] | None:
    if domain == "all":
        return system.get("axes")
    return system.get("domain_axes", {}).get(domain)


def _axes_have_primary_values(axes: dict[str, Any]) -> bool:
    for key in (DEFAULT_LATENCY_AXIS, DEFAULT_COST_AXIS, DEFAULT_QUALITY_AXIS):
        value = axes.get(key)
        if value is None or _coerce_number(value) is None:
            return False
    return True


def _point_has_primary_axes(point: dict[str, Any]) -> bool:
    return _axes_have_primary_values(point)


def _pareto_frontier_point_names(points: list[dict[str, Any]]) -> list[str]:
    frontier = []
    for candidate in points:
        dominated = False
        for challenger in points:
            if challenger is candidate:
                continue
            if _dominates(
                challenger,
                candidate,
                lower_keys=(DEFAULT_LATENCY_AXIS, DEFAULT_COST_AXIS),
                higher_keys=(DEFAULT_QUALITY_AXIS,),
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["system"])
    return sorted(frontier)


def _projection_frontier_point_names(
    points: list[dict[str, Any]],
    *,
    lower_key: str,
    higher_key: str,
) -> list[str]:
    frontier = []
    for candidate in points:
        dominated = False
        for challenger in points:
            if challenger is candidate:
                continue
            if _dominates(
                challenger,
                candidate,
                lower_keys=(lower_key,),
                higher_keys=(higher_key,),
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["system"])
    return sorted(frontier)


def _render_2d_svg(
    points: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    width: int,
    height: int,
) -> str:
    margin = {"left": 96, "right": 40, "top": 72, "bottom": 88}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    valid = [point for point in points if _has_numbers(point, x_key, y_key)]
    x_min, x_max = _extent([point[x_key] for point in valid])
    y_min, y_max = _extent([point[y_key] for point in valid])

    elements = [_svg_header(width, height, title)]
    elements.append(_axis_frame(margin, plot_w, plot_h, x_label, y_label))
    for point in valid:
        x = margin["left"] + _scale(point[x_key], x_min, x_max) * plot_w
        y = margin["top"] + (1 - _scale(point[y_key], y_min, y_max)) * plot_h
        elements.append(_point_svg(point, x, y))
    elements.append("</svg>")
    return "\n".join(elements)


def _render_3d_svg(
    points: list[dict[str, Any]],
    *,
    title: str,
    width: int,
    height: int,
) -> str:
    margin = {"left": 96, "right": 72, "top": 88, "bottom": 96}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    valid = [
        point for point in points
        if _has_numbers(point, DEFAULT_LATENCY_AXIS, DEFAULT_COST_AXIS, DEFAULT_QUALITY_AXIS)
    ]
    latency_min, latency_max = _extent([point[DEFAULT_LATENCY_AXIS] for point in valid])
    cost_min, cost_max = _extent([point[DEFAULT_COST_AXIS] for point in valid])
    quality_min, quality_max = _extent([point[DEFAULT_QUALITY_AXIS] for point in valid])

    elements = [_svg_header(width, height, title)]
    origin_x = margin["left"] + plot_w * 0.28
    origin_y = margin["top"] + plot_h * 0.78
    x_axis = (plot_w * 0.58, plot_h * 0.20)
    y_axis = (plot_w * 0.34, -plot_h * 0.42)
    z_axis = (0.0, -plot_h * 0.58)
    elements.extend([
        _line(origin_x, origin_y, origin_x + x_axis[0], origin_y + x_axis[1], "#475569"),
        _line(origin_x, origin_y, origin_x + y_axis[0], origin_y + y_axis[1], "#475569"),
        _line(origin_x, origin_y, origin_x + z_axis[0], origin_y + z_axis[1], "#475569"),
        _text(origin_x + x_axis[0] + 8, origin_y + x_axis[1] + 16, "latency", 13),
        _text(origin_x + y_axis[0] + 6, origin_y + y_axis[1] - 8, "cost", 13),
        _text(origin_x + z_axis[0] - 38, origin_y + z_axis[1] - 8, "quality", 13),
    ])
    for point in valid:
        lx = _scale(point[DEFAULT_LATENCY_AXIS], latency_min, latency_max)
        cy = _scale(point[DEFAULT_COST_AXIS], cost_min, cost_max)
        qz = _scale(point[DEFAULT_QUALITY_AXIS], quality_min, quality_max)
        x = origin_x + lx * x_axis[0] + cy * y_axis[0] + qz * z_axis[0]
        y = origin_y + lx * x_axis[1] + cy * y_axis[1] + qz * z_axis[1]
        elements.append(_point_svg(point, x, y))
    elements.append("</svg>")
    return "\n".join(elements)


def _svg_header(width: int, height: int, title: str) -> str:
    safe_title = html.escape(title)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{safe_title}">\n'
        "<style>"
        ".label{font:13px sans-serif;fill:#0f172a}"
        ".title{font:700 20px sans-serif;fill:#0f172a}"
        ".muted{font:12px sans-serif;fill:#64748b}"
        "</style>\n"
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n'
        f'<text class="title" x="32" y="38">{safe_title}</text>'
    )


def _axis_frame(
    margin: dict[str, int],
    plot_w: int,
    plot_h: int,
    x_label: str,
    y_label: str,
) -> str:
    left = margin["left"]
    top = margin["top"]
    bottom = top + plot_h
    right = left + plot_w
    return "\n".join([
        _line(left, bottom, right, bottom, "#334155"),
        _line(left, top, left, bottom, "#334155"),
        _text(left + plot_w / 2 - 90, bottom + 52, x_label, 13),
        _text(24, top + plot_h / 2, y_label, 13, rotate=-90),
    ])


def _point_svg(point: dict[str, Any], x: float, y: float) -> str:
    color = "#0284c7" if point.get("on_frontier") else "#94a3b8"
    if not point.get("frontier_eligible"):
        color = "#cbd5e1"
    radius = 7 if point.get("on_frontier") else 5
    label = html.escape(str(point["system"]))
    return "\n".join([
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" stroke="#0f172a" stroke-width="1"/>',
        f'<text class="muted" x="{x + 9:.2f}" y="{y - 7:.2f}">{label}</text>',
    ])


def _line(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="1.5"/>'
    )


def _text(x: float, y: float, text: str, size: int, *, rotate: int | None = None) -> str:
    safe = html.escape(text)
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return f'<text class="label" x="{x:.2f}" y="{y:.2f}" font-size="{size}"{transform}>{safe}</text>'


def _has_numbers(point: dict[str, Any], *keys: str) -> bool:
    return all(_coerce_number(point.get(key)) is not None for key in keys)


def _extent(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        padding = max(abs(low) * 0.1, 1.0)
        return low - padding, high + padding
    padding = (high - low) * 0.08
    return low - padding, high + padding


def _scale(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.5
    return (value - low) / (high - low)


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return slug or "all"


def _pareto_frontier_names(
    systems: list[dict[str, Any]],
    *,
    latency_key: str,
    cost_key: str,
    quality_key: str,
) -> list[str]:
    frontier = []
    for candidate in systems:
        dominated = False
        for challenger in systems:
            if challenger is candidate:
                continue
            if _dominates(
                challenger["axes"],
                candidate["axes"],
                lower_keys=(latency_key, cost_key),
                higher_keys=(quality_key,),
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["name"])
    return sorted(
        frontier,
        key=lambda name: (
            next(system["axes"][latency_key] for system in systems if system["name"] == name),
            next(system["axes"][cost_key] for system in systems if system["name"] == name),
            name,
        ),
    )


def _projection_frontier_names(
    systems: list[dict[str, Any]],
    *,
    lower_key: str,
    higher_key: str,
) -> list[str]:
    frontier = []
    for candidate in systems:
        dominated = False
        for challenger in systems:
            if challenger is candidate:
                continue
            if _dominates(
                challenger["axes"],
                candidate["axes"],
                lower_keys=(lower_key,),
                higher_keys=(higher_key,),
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["name"])
    return sorted(frontier)


def _clean_targets(values: list[float] | tuple[float, ...] | None) -> list[float]:
    if not values:
        return []
    return sorted({
        float(value)
        for value in values
        if not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(float(value))
        and float(value) >= 0
    })


def _within_constraint(system: dict[str, Any], constraint: dict[str, float | None]) -> bool:
    axes = system.get("axes", {})
    latency_target = constraint.get("latency_ms")
    cost_target = constraint.get("cost_usd")
    latency = axes.get(DEFAULT_LATENCY_AXIS)
    cost = axes.get(DEFAULT_COST_AXIS)
    if latency_target is not None and (latency is None or latency > latency_target):
        return False
    if cost_target is not None and (cost is None or cost > cost_target):
        return False
    return True


def _dominates(
    challenger: dict[str, float],
    candidate: dict[str, float],
    *,
    lower_keys: Iterable[str],
    higher_keys: Iterable[str],
) -> bool:
    at_least_as_good = []
    strictly_better = []
    for key in lower_keys:
        left = challenger[key]
        right = candidate[key]
        at_least_as_good.append(left <= right)
        strictly_better.append(left < right)
    for key in higher_keys:
        left = challenger[key]
        right = candidate[key]
        at_least_as_good.append(left >= right)
        strictly_better.append(left > right)
    return all(at_least_as_good) and any(strictly_better)


def _inverse_minmax(values: dict[str, float]) -> dict[str, float | None]:
    present = list(values.values())
    if not present:
        return {name: None for name in values}
    low = min(present)
    high = max(present)
    if high == low:
        return {name: 1.0 for name in values}
    return {
        name: (high - value) / (high - low)
        for name, value in values.items()
    }


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _coerce_number(value)
        if number is not None:
            return number
    return None


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if isfinite(number) else None
    return None


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float | int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _round_optional(value: float | None, ndigits: int) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)


def _bootstrap_interval(
    values: list[float | int],
    *,
    statistic,
    ndigits: int,
    confidence: float = 0.95,
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    estimate = statistic(values) if values else None
    if not values:
        return {
            "method": "bootstrap_percentile",
            "confidence": confidence,
            "estimate": None,
            "low": None,
            "high": None,
            "n": 0,
        }
    if len(values) == 1:
        rounded = _round_optional(estimate, ndigits)
        return {
            "method": "bootstrap_percentile",
            "confidence": confidence,
            "estimate": rounded,
            "low": rounded,
            "high": rounded,
            "n": 1,
        }

    rng = random.Random(seed)
    replicates = []
    count = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        sample_stat = statistic(sample)
        if sample_stat is not None:
            replicates.append(sample_stat)

    alpha = 1.0 - confidence
    return {
        "method": "bootstrap_percentile",
        "confidence": confidence,
        "estimate": _round_optional(estimate, ndigits),
        "low": _round_optional(_percentile(replicates, (alpha / 2.0) * 100), ndigits),
        "high": _round_optional(_percentile(replicates, (1.0 - alpha / 2.0) * 100), ndigits),
        "n": len(values),
        "iterations": iterations,
        "seed": seed,
    }


def _task_success_interval(report: dict[str, Any]) -> dict[str, Any]:
    ci = report.get("confidence_intervals", {})
    if isinstance(ci, dict):
        direct = ci.get("trial_pass_rate") or ci.get("task_success_rate")
        if isinstance(direct, dict):
            return {
                "method": direct.get("method", "source_report"),
                "confidence": direct.get("confidence", 0.95),
                "estimate": _round_optional(_first_number(direct.get("estimate")), 6),
                "low": _round_optional(_first_number(direct.get("low")), 6),
                "high": _round_optional(_first_number(direct.get("high")), 6),
                "n": direct.get("n", direct.get("total")),
            }

    samples = _collect_success_samples(report)
    successes = sum(samples)
    return _wilson_interval(successes, len(samples))


def _collect_success_samples(report: dict[str, Any]) -> list[int]:
    samples = []
    for result in report.get("results", []):
        trials = result.get("trials")
        if isinstance(trials, list):
            for trial in trials:
                if "error" not in trial and trial.get("passed") is not None:
                    samples.append(1 if trial.get("passed") else 0)
        elif result.get("passed") is not None:
            samples.append(1 if result.get("passed") else 0)
    return samples


def _wilson_interval(successes: int, total: int, confidence: float = 0.95) -> dict[str, Any]:
    if total <= 0:
        return {
            "method": "wilson",
            "confidence": confidence,
            "estimate": None,
            "low": None,
            "high": None,
            "n": 0,
        }
    z = 1.959963984540054 if confidence == 0.95 else 1.959963984540054
    phat = successes / total
    denominator = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    margin = z * ((phat * (1 - phat) + z * z / (4 * total)) / total) ** 0.5 / denominator
    return {
        "method": "wilson",
        "confidence": confidence,
        "estimate": round(phat, 6),
        "low": round(max(0.0, center - margin), 6),
        "high": round(min(1.0, center + margin), 6),
        "n": total,
    }
