"""Release-readiness profiles for OpenVoiceCS benchmark data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.frontier import (
    DEFAULT_COST_AXIS,
    DEFAULT_LATENCY_AXIS,
    DEFAULT_QUALITY_AXIS,
    build_frontier_plot_data,
    validate_frontier_report,
)
from src.evaluation.benchmark.run_manifest import validate_run_manifest

RELEASE_PROFILES: dict[str, dict[str, Any]] = {
    "seed": {
        "description": "Valid seed package for local development and method review.",
        "min_scenarios": 10,
        "min_domains": 5,
        "required_tracks": {
            "text_to_action": 1,
            "adversarial_compliance": 1,
        },
        "min_public_dev_scenarios": 10,
        "min_sealed_test_scenarios": 0,
        "min_audio_variants": 1,
        "min_public_dev_audio_variants": 1,
        "min_sealed_test_audio_variants": 0,
        "min_pricing_profiles": 2,
        "require_audio_asset_integrity": False,
        "require_provenance_integrity": True,
        "require_open_licenses": False,
        "require_audio_consent": False,
        "require_no_real_customer_data": True,
        "require_low_contamination_risk": True,
        "required_oracle_coverage": 1.0,
        "require_contract_valid": True,
        "require_release_gates": True,
        "require_frontier_artifact": False,
        "require_run_manifest": False,
        "require_plot_artifacts": False,
        "require_judged_experience": False,
        "require_judge_manifest_pinning": False,
        "required_concurrency_levels": [],
        "require_latency_at_100": False,
        "require_latency_event_evidence": False,
        "require_fully_loaded_cost_evidence": False,
        "require_comparable_pricing_profiles": False,
        "require_hardware_profile": False,
        "min_trials_per_scenario": 1,
    },
    "public_beta": {
        "description": "Open public-development benchmark candidate.",
        "min_scenarios": 50,
        "min_domains": 5,
        "required_tracks": {
            "text_to_action": 20,
            "adversarial_compliance": 10,
            "audio_to_action": 5,
            "robustness": 5,
        },
        "min_public_dev_scenarios": 30,
        "min_sealed_test_scenarios": 20,
        "min_audio_variants": 25,
        "min_public_dev_audio_variants": 15,
        "min_sealed_test_audio_variants": 10,
        "min_pricing_profiles": 3,
        "require_audio_asset_integrity": True,
        "require_provenance_integrity": True,
        "require_open_licenses": True,
        "require_audio_consent": True,
        "require_no_real_customer_data": True,
        "require_low_contamination_risk": True,
        "required_oracle_coverage": 1.0,
        "require_contract_valid": True,
        "require_release_gates": True,
        "require_frontier_artifact": True,
        "require_run_manifest": True,
        "require_plot_artifacts": True,
        "require_judged_experience": True,
        "require_judge_manifest_pinning": True,
        "required_concurrency_levels": [1, 10, 100],
        "require_latency_at_100": True,
        "require_latency_event_evidence": True,
        "require_fully_loaded_cost_evidence": True,
        "require_comparable_pricing_profiles": True,
        "require_hardware_profile": True,
        "min_trials_per_scenario": 3,
    },
    "leaderboard_v1": {
        "description": "Scientific leaderboard release with a non-public test set.",
        "min_scenarios": 200,
        "min_domains": 6,
        "required_tracks": {
            "text_to_action": 60,
            "adversarial_compliance": 40,
            "audio_to_action": 30,
            "robustness": 40,
            "end_to_end_voice": 20,
        },
        "min_public_dev_scenarios": 80,
        "min_sealed_test_scenarios": 120,
        "min_audio_variants": 120,
        "min_public_dev_audio_variants": 40,
        "min_sealed_test_audio_variants": 80,
        "min_pricing_profiles": 5,
        "require_audio_asset_integrity": True,
        "require_provenance_integrity": True,
        "require_open_licenses": True,
        "require_audio_consent": True,
        "require_no_real_customer_data": True,
        "require_low_contamination_risk": True,
        "required_oracle_coverage": 1.0,
        "require_contract_valid": True,
        "require_release_gates": True,
        "require_frontier_artifact": True,
        "require_run_manifest": True,
        "require_plot_artifacts": True,
        "require_judged_experience": True,
        "require_judge_manifest_pinning": True,
        "required_concurrency_levels": [1, 10, 100],
        "require_latency_at_100": True,
        "require_latency_event_evidence": True,
        "require_fully_loaded_cost_evidence": True,
        "require_comparable_pricing_profiles": True,
        "require_hardware_profile": True,
        "min_trials_per_scenario": 5,
    },
}


@dataclass(frozen=True)
class ReadinessIssue:
    """Structured release-readiness issue."""

    criterion: str
    observed: Any
    required: Any
    message: str


def evaluate_release_readiness(
    audit: dict[str, Any],
    profile: str = "seed",
    *,
    frontier_report: dict[str, Any] | None = None,
    run_manifest: dict[str, Any] | None = None,
    run_manifest_base_dir: str | Path = ".",
    verify_run_manifest_files: bool = False,
    plot_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a release audit against a named readiness profile."""
    if profile not in RELEASE_PROFILES:
        known = ", ".join(sorted(RELEASE_PROFILES))
        raise ValueError(f"unknown readiness profile {profile!r}; expected one of: {known}")

    criteria = RELEASE_PROFILES[profile]
    issues: list[ReadinessIssue] = []
    _check_contract_and_gates(issues, audit, criteria)
    _check_scenario_coverage(issues, audit, criteria)
    _check_audio_coverage(issues, audit, criteria)
    _check_audio_asset_integrity(issues, audit, criteria)
    _check_split_coverage(issues, audit, criteria)
    _check_provenance_integrity(issues, audit, criteria)
    _check_pricing_coverage(issues, audit, criteria)
    _check_oracle_coverage(issues, audit, criteria)
    _check_frontier_artifact(issues, frontier_report, criteria)
    _check_run_manifest_artifact(
        issues,
        run_manifest,
        criteria,
        base_dir=Path(run_manifest_base_dir),
        verify_files=verify_run_manifest_files,
    )
    _check_plot_artifacts(issues, plot_dir, criteria, frontier_report=frontier_report)

    return {
        "benchmark": audit.get("benchmark", "OpenVoiceCS-Bench"),
        "version": audit.get("version"),
        "profile": profile,
        "profile_description": criteria["description"],
        "passed": not issues,
        "num_issues": len(issues),
        "criteria": criteria,
        "artifacts": {
            "frontier_report": frontier_report is not None,
            "run_manifest": run_manifest is not None,
            "plot_dir": plot_dir is not None,
        },
        "issues": [
            {
                "criterion": issue.criterion,
                "observed": issue.observed,
                "required": issue.required,
                "message": issue.message,
            }
            for issue in issues
        ],
    }


def _check_contract_and_gates(
    issues: list[ReadinessIssue],
    audit: dict[str, Any],
    criteria: dict[str, Any],
) -> None:
    validation_passed = audit.get("validation", {}).get("passed") is True
    if criteria["require_contract_valid"] and not validation_passed:
        issues.append(
            ReadinessIssue(
                "valid_json_contract",
                validation_passed,
                True,
                "scenario, audio, pricing, and split manifests must validate",
            )
        )

    gates_passed = audit.get("release_gates", {}).get("passed") is True
    if criteria["require_release_gates"] and not gates_passed:
        issues.append(
            ReadinessIssue(
                "release_gates",
                gates_passed,
                True,
                "base release gates must pass before readiness can pass",
            )
        )


def _check_scenario_coverage(
    issues: list[ReadinessIssue],
    audit: dict[str, Any],
    criteria: dict[str, Any],
) -> None:
    stats = audit.get("scenario_stats", {})
    _require_min(
        issues,
        "num_scenarios",
        stats.get("num_scenarios", 0),
        criteria["min_scenarios"],
        "release does not contain enough scenarios",
    )
    _require_min(
        issues,
        "num_domains",
        len(stats.get("domains", {})),
        criteria["min_domains"],
        "release does not cover enough domains",
    )

    tracks = stats.get("tracks", {})
    for track, required in criteria["required_tracks"].items():
        _require_min(
            issues,
            f"track.{track}",
            tracks.get(track, 0),
            required,
            f"track {track!r} has insufficient scenario coverage",
        )


def _check_audio_coverage(
    issues: list[ReadinessIssue],
    audit: dict[str, Any],
    criteria: dict[str, Any],
) -> None:
    stats = audit.get("audio_manifest_stats", {})
    _require_min(
        issues,
        "num_audio_variants",
        stats.get("num_variants", 0),
        criteria["min_audio_variants"],
        "release does not contain enough audio or robustness variants",
    )


def _check_audio_asset_integrity(
    issues: list[ReadinessIssue],
    audit: dict[str, Any],
    criteria: dict[str, Any],
) -> None:
    if not criteria["require_audio_asset_integrity"]:
        return
    stats = audit.get("audio_asset_stats", {})
    num_variants = stats.get("num_variants", 0)
    checks = {
        "audio_assets.existing_files": stats.get("num_existing_files", 0),
        "audio_assets.sha256_verified": stats.get("num_sha256_verified", 0),
        "audio_assets.sample_rate_verified": stats.get("num_sample_rate_verified", 0),
        "audio_assets.duration_verified": stats.get("num_duration_verified", 0),
        "audio_assets.positive_duration": stats.get("num_positive_duration_files", 0),
    }
    for criterion, observed in checks.items():
        _require_min(
            issues,
            criterion,
            observed,
            num_variants,
            "every audio variant must have a verified physical asset",
        )


def _check_split_coverage(
    issues: list[ReadinessIssue],
    audit: dict[str, Any],
    criteria: dict[str, Any],
) -> None:
    split_stats = audit.get("split_manifest_stats", {})
    splits = split_stats.get("splits", {})
    public_dev = splits.get("public_dev", {})
    sealed_test = splits.get("sealed_test", {})
    _require_min(
        issues,
        "split.public_dev.scenarios",
        public_dev.get("num_scenarios", 0),
        criteria["min_public_dev_scenarios"],
        "public development split is too small",
    )
    _require_min(
        issues,
        "split.sealed_test.scenarios",
        sealed_test.get("num_scenarios", 0),
        criteria["min_sealed_test_scenarios"],
        "sealed test split is too small",
    )
    _require_min(
        issues,
        "split.public_dev.audio_variants",
        public_dev.get("num_audio_variants", 0),
        criteria["min_public_dev_audio_variants"],
        "public development split has too few audio variants",
    )
    _require_min(
        issues,
        "split.sealed_test.audio_variants",
        sealed_test.get("num_audio_variants", 0),
        criteria["min_sealed_test_audio_variants"],
        "sealed test split has too few audio variants",
    )


def _check_provenance_integrity(
    issues: list[ReadinessIssue],
    audit: dict[str, Any],
    criteria: dict[str, Any],
) -> None:
    if not criteria["require_provenance_integrity"]:
        return
    stats = audit.get("provenance_stats", {})
    _require_min_rate(
        issues,
        "provenance.scenario_coverage",
        stats.get("scenario_coverage", 0.0),
        1.0,
        "every scenario must have provenance metadata",
    )
    _require_min_rate(
        issues,
        "provenance.audio_variant_coverage",
        stats.get("audio_variant_coverage", 0.0),
        1.0,
        "every audio variant must have provenance metadata",
    )
    if criteria["require_open_licenses"]:
        _require_min_rate(
            issues,
            "provenance.scenario_license_open_rate",
            stats.get("scenario_license_open_rate", 0.0),
            1.0,
            "every scenario must have an approved open license",
        )
        _require_min_rate(
            issues,
            "provenance.audio_license_open_rate",
            stats.get("audio_license_open_rate", 0.0),
            1.0,
            "every audio variant must have an approved open license",
        )
    if criteria["require_audio_consent"]:
        _require_min_rate(
            issues,
            "provenance.audio_speaker_consent_rate",
            stats.get("audio_speaker_consent_rate", 0.0),
            1.0,
            "every audio variant must have consent or synthetic voice provenance",
        )
    if criteria["require_no_real_customer_data"]:
        _require_min_rate(
            issues,
            "provenance.no_real_customer_data_rate",
            stats.get("no_real_customer_data_rate", 0.0),
            1.0,
            "real customer data is not allowed in release profiles",
        )
    if criteria["require_low_contamination_risk"]:
        _require_min_rate(
            issues,
            "provenance.low_contamination_risk_rate",
            stats.get("low_contamination_risk_rate", 0.0),
            1.0,
            "items must be declared low or no contamination risk",
        )


def _check_pricing_coverage(
    issues: list[ReadinessIssue],
    audit: dict[str, Any],
    criteria: dict[str, Any],
) -> None:
    stats = audit.get("pricing_manifest_stats", {})
    _require_min(
        issues,
        "pricing_profiles",
        (
            stats.get("num_comparable_profiles", 0)
            if criteria.get("require_comparable_pricing_profiles")
            else stats.get("num_profiles", 0)
        ),
        criteria["min_pricing_profiles"],
        (
            "pricing manifest does not contain enough non-reference comparable pricing profiles"
            if criteria.get("require_comparable_pricing_profiles")
            else "pricing manifest does not contain enough pricing profiles"
        ),
    )


def _check_oracle_coverage(
    issues: list[ReadinessIssue],
    audit: dict[str, Any],
    criteria: dict[str, Any],
) -> None:
    required_rate = criteria["required_oracle_coverage"]
    coverage = audit.get("oracle_coverage", {})
    for oracle_field, item in coverage.items():
        observed = item.get("rate", 0.0) if isinstance(item, dict) else 0.0
        if observed < required_rate:
            issues.append(
                ReadinessIssue(
                    f"oracle_coverage.{oracle_field}",
                    observed,
                    required_rate,
                    f"oracle field {oracle_field!r} is not fully covered",
                )
            )


def _check_frontier_artifact(
    issues: list[ReadinessIssue],
    frontier_report: dict[str, Any] | None,
    criteria: dict[str, Any],
) -> None:
    if frontier_report is None:
        if criteria["require_frontier_artifact"]:
            issues.append(
                ReadinessIssue(
                    "frontier_artifact",
                    False,
                    True,
                    "frontier release profiles require a generated frontier report",
                )
            )
        return

    for issue in validate_frontier_report(frontier_report):
        issues.append(
            ReadinessIssue(
                f"frontier_artifact.{issue.path}",
                "invalid",
                "valid frontier report",
                issue.message,
            )
        )

    systems = frontier_report.get("systems")
    scorecards = frontier_report.get("scorecards")
    environment = frontier_report.get("environment")
    if frontier_report.get("benchmark") != "Latency-Cost-Quality Frontier":
        issues.append(
            ReadinessIssue(
                "frontier_artifact.benchmark",
                frontier_report.get("benchmark"),
                "Latency-Cost-Quality Frontier",
                "frontier report has the wrong benchmark name",
            )
        )
    if not isinstance(systems, dict) or not systems:
        issues.append(
            ReadinessIssue(
                "frontier_artifact.systems",
                0,
                ">0",
                "frontier report must contain systems",
            )
        )
        systems = {}
    if not isinstance(scorecards, dict) or not scorecards:
        issues.append(
            ReadinessIssue(
                "frontier_artifact.scorecards",
                0,
                ">0",
                "frontier report must contain scorecards",
            )
        )
        scorecards = {}
    if set(systems) != set(scorecards):
        issues.append(
            ReadinessIssue(
                "frontier_artifact.scorecard_alignment",
                sorted(set(scorecards)),
                sorted(set(systems)),
                "scorecards must align one-to-one with systems",
            )
        )
    for field in ("latency_vs_quality", "cost_vs_quality"):
        projections = frontier_report.get("projection_frontiers", {})
        if not isinstance(projections, dict) or field not in projections:
            issues.append(
                ReadinessIssue(
                    f"frontier_artifact.projection_frontiers.{field}",
                    False,
                    True,
                    "frontier report must include required 2D projection frontiers",
                )
            )
    if not isinstance(environment, dict):
        issues.append(
            ReadinessIssue(
                "frontier_artifact.environment",
                False,
                True,
                "frontier report must include environment metadata",
            )
        )
    else:
        for field in ("region", "network", "transport"):
            if not _is_controlled_label(environment.get(field)):
                issues.append(
                    ReadinessIssue(
                        f"frontier_artifact.environment.{field}",
                        environment.get(field),
                        "controlled value",
                        "frontier report must pin controlled environment metadata",
                    )
                )
        if criteria.get("require_hardware_profile") and not _is_controlled_label(
            environment.get("hardware_profile")
        ):
            issues.append(
                ReadinessIssue(
                    "frontier_artifact.environment.hardware_profile",
                    environment.get("hardware_profile"),
                    "controlled value",
                    "official frontier releases must pin the client hardware profile",
                )
            )
        levels = environment.get("concurrency_levels")
        if (
            not isinstance(levels, list)
            or not levels
            or any(not isinstance(level, int) or level < 1 for level in levels)
        ):
            issues.append(
                ReadinessIssue(
                    "frontier_artifact.environment.concurrency_levels",
                    levels,
                    "positive integer list",
                    "frontier report must pin represented concurrency levels",
                )
            )
        else:
            missing_levels = sorted(set(criteria["required_concurrency_levels"]) - set(levels))
            if missing_levels:
                issues.append(
                    ReadinessIssue(
                        "frontier_artifact.environment.concurrency_levels",
                        levels,
                        criteria["required_concurrency_levels"],
                        f"frontier release must include concurrency levels: {missing_levels}",
                    )
                )
    required_scorecard_fields = {
        "p50_v2v_ttfb_ms",
        "p90_v2v_ttfb_ms",
        "p95_v2v_ttfb_ms",
        "p99_v2v_ttfb_ms",
        "p95_v2v_last_byte_ms",
        "barge_in_stop_p95_ms",
        "interruption_recovery_p95_ms",
        "stage_latency_ms",
        "cost_usd_per_successful_conversation",
        "task_success_rate",
        "experience_score",
        "latency_measurement",
        "latency_load",
        "cost_provenance",
        "axis_confidence_intervals",
    }
    for name, scorecard in scorecards.items():
        if not isinstance(scorecard, dict):
            issues.append(
                ReadinessIssue(
                    f"frontier_artifact.scorecards.{name}",
                    False,
                    True,
                    "scorecard must be an object",
                )
            )
            continue
        missing = sorted(required_scorecard_fields - set(scorecard))
        if missing:
            issues.append(
                ReadinessIssue(
                    f"frontier_artifact.scorecards.{name}",
                    missing,
                    [],
                    "scorecard is missing required frontier fields",
                )
            )
        if (
            criteria["require_judged_experience"]
            and scorecard.get("experience_score_source") != "judged"
        ):
            issues.append(
                ReadinessIssue(
                    f"frontier_artifact.scorecards.{name}.experience_score_source",
                    scorecard.get("experience_score_source"),
                    "judged",
                    "official frontier releases must use judged conversation-experience scores",
                )
            )
        if criteria["require_judged_experience"]:
            _check_judged_experience_evidence(issues, name, scorecard)
        _check_axis_interval_evidence(issues, name, scorecard, criteria)
        if criteria["require_latency_at_100"] and scorecard.get("latency_at_100_concurrency_p95_ms") is None:
            issues.append(
                ReadinessIssue(
                    f"frontier_artifact.scorecards.{name}.latency_at_100_concurrency_p95_ms",
                    None,
                    "numeric p95 latency at 100 concurrency",
                    "official frontier releases must report p95 latency at 100 concurrent calls",
                )
        )
        if criteria.get("require_latency_event_evidence"):
            _check_latency_scorecard_fields(issues, name, scorecard)
            _check_latency_event_evidence(issues, name, scorecard)
        if criteria.get("require_latency_at_100"):
            _check_latency_load_evidence(issues, name, scorecard, criteria)
        if criteria.get("require_fully_loaded_cost_evidence"):
            _check_fully_loaded_cost_evidence(issues, name, scorecard)


def _check_axis_interval_evidence(
    issues: list[ReadinessIssue],
    system_name: str,
    scorecard: dict[str, Any],
    criteria: dict[str, Any],
) -> None:
    intervals = scorecard.get("axis_confidence_intervals")
    if not isinstance(intervals, dict):
        return
    min_trials = criteria.get("min_trials_per_scenario", 1)
    axes = (
        DEFAULT_LATENCY_AXIS,
        "cost_usd_per_conversation",
        DEFAULT_COST_AXIS,
        DEFAULT_QUALITY_AXIS,
    )
    for axis in axes:
        item = intervals.get(axis)
        if not isinstance(item, dict):
            continue
        n = item.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < min_trials:
            issues.append(
                ReadinessIssue(
                    f"frontier_artifact.scorecards.{system_name}.axis_confidence_intervals.{axis}.n",
                    n,
                    f">={min_trials}",
                    "frontier confidence intervals must be based on enough repeated samples",
                )
            )


def _check_judged_experience_evidence(
    issues: list[ReadinessIssue],
    system_name: str,
    scorecard: dict[str, Any],
) -> None:
    evidence = scorecard.get("experience_evidence")
    path = f"frontier_artifact.scorecards.{system_name}.experience_evidence"
    if not isinstance(evidence, dict):
        issues.append(
            ReadinessIssue(
                path,
                False,
                True,
                "official judged releases must include conversation-experience evidence",
            )
        )
        return
    sample_count = scorecard.get("sample_count")
    if not isinstance(sample_count, int):
        sample_count = evidence.get("num_trials")
    coverage = evidence.get("coverage")
    if coverage != 1.0:
        issues.append(
            ReadinessIssue(
                f"{path}.coverage",
                coverage,
                1.0,
                "official judged releases must judge every scored trial",
            )
        )
    judged_trials = evidence.get("num_judged_trials")
    if isinstance(sample_count, int) and judged_trials != sample_count:
        issues.append(
            ReadinessIssue(
                f"{path}.num_judged_trials",
                judged_trials,
                sample_count,
                "judged trial count must match scored latency sample count",
            )
        )
    judge_counts = evidence.get("judge_counts")
    if not isinstance(judge_counts, dict) or not judge_counts:
        issues.append(
            ReadinessIssue(
                f"{path}.judge_counts",
                judge_counts,
                "non-empty judge counts",
                "official judged releases must disclose judge coverage",
            )
        )


def _check_latency_scorecard_fields(
    issues: list[ReadinessIssue],
    system_name: str,
    scorecard: dict[str, Any],
) -> None:
    required_numbers = {
        "p50_v2v_ttfb_ms": "official frontier releases must report p50 voice-to-voice TTFB",
        "p90_v2v_ttfb_ms": "official frontier releases must report p90 voice-to-voice TTFB",
        "p95_v2v_ttfb_ms": "official frontier releases must report p95 voice-to-voice TTFB",
        "p99_v2v_ttfb_ms": "official frontier releases must report p99 voice-to-voice TTFB",
        "p95_v2v_last_byte_ms": "official frontier releases must report last-byte voice-to-voice latency",
        "barge_in_stop_p95_ms": "official frontier releases must report barge-in stop latency",
        "interruption_recovery_p95_ms": "official frontier releases must report interruption recovery latency",
    }
    for field, message in required_numbers.items():
        value = scorecard.get(field)
        if not _is_nonnegative_number(value):
            issues.append(
                ReadinessIssue(
                    f"frontier_artifact.scorecards.{system_name}.{field}",
                    value,
                    "nonnegative number",
                    message,
                )
            )

    stage_latency = scorecard.get("stage_latency_ms")
    stage_path = f"frontier_artifact.scorecards.{system_name}.stage_latency_ms"
    if not isinstance(stage_latency, dict):
        issues.append(
            ReadinessIssue(
                stage_path,
                stage_latency,
                "stage latency summary",
                "official frontier releases must decompose ASR, LLM, and TTS latency",
            )
        )
        return
    required_stages = {
        "asr_finalization_ms": "ASR finalization latency must be reported",
        "llm_ttft_ms": "LLM time-to-first-token latency must be reported",
        "tts_first_chunk_ms": "TTS first-chunk latency must be reported",
    }
    for stage, message in required_stages.items():
        summary = stage_latency.get(stage)
        if not isinstance(summary, dict):
            issues.append(
                ReadinessIssue(
                    f"{stage_path}.{stage}",
                    summary,
                    "p50/p95 summary",
                    message,
                )
            )
            continue
        for percentile in ("p50", "p95"):
            value = summary.get(percentile)
            if not _is_nonnegative_number(value):
                issues.append(
                    ReadinessIssue(
                        f"{stage_path}.{stage}.{percentile}",
                        value,
                        "nonnegative number",
                        message,
                    )
                )


def _check_latency_event_evidence(
    issues: list[ReadinessIssue],
    system_name: str,
    scorecard: dict[str, Any],
) -> None:
    measurement = scorecard.get("latency_measurement")
    path = f"frontier_artifact.scorecards.{system_name}.latency_measurement"
    if not isinstance(measurement, dict):
        issues.append(
            ReadinessIssue(
                path,
                False,
                True,
                "official frontier releases must include latency measurement provenance",
            )
        )
        return
    sample_count = measurement.get("sample_count", 0)
    if not isinstance(sample_count, int) or sample_count <= 0:
        issues.append(
            ReadinessIssue(
                f"{path}.sample_count",
                sample_count,
                ">0",
                "latency measurement provenance must cover scored samples",
            )
        )
        return
    required_counts = {
        "event_stream_samples": "latency must be measured from canonical realtime events",
        "vad_origin_samples": "latency must use user.end_speech VAD origin",
        "first_audio_event_samples": "TTFB must be backed by tts.first_audio events",
        "last_audio_event_samples": "last-byte latency must be backed by agent.complete events",
        "barge_in_stop_event_samples": "barge-in stop latency must be backed by barge_in.stop events",
        "interruption_recovery_event_samples": "interruption recovery must be backed by barge_in.recovered events",
    }
    for field, message in required_counts.items():
        observed = measurement.get(field)
        if observed != sample_count:
            issues.append(
                ReadinessIssue(
                    f"{path}.{field}",
                    observed,
                    sample_count,
                    message,
                )
            )


def _check_latency_load_evidence(
    issues: list[ReadinessIssue],
    system_name: str,
    scorecard: dict[str, Any],
    criteria: dict[str, Any],
) -> None:
    load = scorecard.get("latency_load")
    path = f"frontier_artifact.scorecards.{system_name}.latency_load"
    if not isinstance(load, dict):
        issues.append(
            ReadinessIssue(
                path,
                False,
                True,
                "official frontier releases must include load measurement evidence",
            )
        )
        return
    levels = load.get("levels")
    if not isinstance(levels, dict):
        issues.append(
            ReadinessIssue(
                f"{path}.levels",
                False,
                True,
                "load measurement evidence must include per-concurrency levels",
            )
        )
        return
    for level in criteria["required_concurrency_levels"]:
        item = levels.get(str(level))
        item_path = f"{path}.levels.{level}"
        if not isinstance(item, dict):
            issues.append(
                ReadinessIssue(
                    item_path,
                    False,
                    True,
                    f"official frontier releases must include load evidence for concurrency {level}",
                )
            )
            continue
        sample_count = item.get("sample_count")
        if not isinstance(sample_count, int) or sample_count < level:
            issues.append(
                ReadinessIssue(
                    f"{item_path}.sample_count",
                    sample_count,
                    f">={level}",
                    f"load evidence must include enough samples to exercise concurrency {level}",
                )
            )
        target = item.get("target_concurrency")
        if target != level:
            issues.append(
                ReadinessIssue(
                    f"{item_path}.target_concurrency",
                    target,
                    level,
                    f"load evidence target must match concurrency {level}",
                )
            )
        requested_calls = item.get("requested_calls")
        if not isinstance(requested_calls, int) or isinstance(requested_calls, bool) or requested_calls < level:
            issues.append(
                ReadinessIssue(
                    f"{item_path}.requested_calls",
                    requested_calls,
                    f">={level}",
                    f"load evidence must request at least {level} calls for concurrency {level}",
                )
            )
        completed_calls = item.get("completed_calls")
        if not isinstance(completed_calls, int) or isinstance(completed_calls, bool) or completed_calls < level:
            issues.append(
                ReadinessIssue(
                    f"{item_path}.completed_calls",
                    completed_calls,
                    f">={level}",
                    f"load evidence must complete at least {level} calls for concurrency {level}",
                )
            )
        peak_active_calls = item.get("peak_active_calls")
        if not isinstance(peak_active_calls, int) or isinstance(peak_active_calls, bool) or peak_active_calls < level:
            issues.append(
                ReadinessIssue(
                    f"{item_path}.peak_active_calls",
                    peak_active_calls,
                    f">={level}",
                    f"load evidence must reach {level} active calls",
                )
            )
        if item.get("saturated") is not True:
            issues.append(
                ReadinessIssue(
                    f"{item_path}.saturated",
                    item.get("saturated"),
                    True,
                    f"load evidence must saturate requested concurrency {level}",
                )
            )
        if item.get("p95_v2v_ttfb_ms") is None:
            issues.append(
                ReadinessIssue(
                    f"{item_path}.p95_v2v_ttfb_ms",
                    None,
                    "numeric p95 latency",
                    f"load evidence must report p95 TTFB for concurrency {level}",
                )
            )
        error_calls = item.get("error_calls")
        if error_calls != 0:
            issues.append(
                ReadinessIssue(
                    f"{item_path}.error_calls",
                    error_calls,
                    0,
                    f"official load evidence must not have failed calls at concurrency {level}",
                )
            )


def _check_fully_loaded_cost_evidence(
    issues: list[ReadinessIssue],
    system_name: str,
    scorecard: dict[str, Any],
) -> None:
    provenance = scorecard.get("cost_provenance")
    path = f"frontier_artifact.scorecards.{system_name}.cost_provenance"
    if not isinstance(provenance, dict):
        issues.append(
            ReadinessIssue(
                path,
                False,
                True,
                "official frontier releases must include cost provenance",
            )
        )
        return

    sample_count = provenance.get("sample_count", 0)
    if not isinstance(sample_count, int) or sample_count <= 0:
        issues.append(
            ReadinessIssue(
                f"{path}.sample_count",
                sample_count,
                ">0",
                "cost provenance must cover scored samples",
            )
        )
        return

    fully_loaded = provenance.get("fully_loaded_samples")
    if fully_loaded != sample_count:
        issues.append(
            ReadinessIssue(
                f"{path}.fully_loaded_samples",
                fully_loaded,
                sample_count,
                "official cost results must be derived from every required component",
            )
        )

    missing = provenance.get("missing_cost_samples")
    if missing != 0:
        issues.append(
            ReadinessIssue(
                f"{path}.missing_cost_samples",
                missing,
                0,
                "official cost results must not have unpriced scored samples",
            )
        )

    required_components = provenance.get("required_components")
    if not isinstance(required_components, list) or not required_components:
        issues.append(
            ReadinessIssue(
                f"{path}.required_components",
                required_components,
                "non-empty component list",
                "official cost results must declare required pricing components",
            )
        )
        return

    component_counts = provenance.get("component_sample_counts")
    if not isinstance(component_counts, dict):
        issues.append(
            ReadinessIssue(
                f"{path}.component_sample_counts",
                component_counts,
                "component coverage counts",
                "official cost results must include per-component coverage",
            )
        )
        return
    for component in required_components:
        observed = component_counts.get(component)
        if observed != sample_count:
            issues.append(
                ReadinessIssue(
                    f"{path}.component_sample_counts.{component}",
                    observed,
                    sample_count,
                    f"cost component {component!r} must cover every scored sample",
                )
            )


def _check_run_manifest_artifact(
    issues: list[ReadinessIssue],
    run_manifest: dict[str, Any] | None,
    criteria: dict[str, Any],
    *,
    base_dir: Path,
    verify_files: bool,
) -> None:
    if run_manifest is None:
        if criteria["require_run_manifest"]:
            issues.append(
                ReadinessIssue(
                    "run_manifest",
                    False,
                    True,
                    "frontier release profiles require a frozen run manifest",
                )
            )
        return
    manifest_issues = validate_run_manifest(
        run_manifest,
        base_dir=base_dir,
        verify_files=verify_files,
    )
    for issue in manifest_issues:
        issues.append(
            ReadinessIssue(
                f"run_manifest.{issue.path}",
                issue.message,
                "valid",
                "run manifest must validate",
            )
        )
    environment = run_manifest.get("release_tuple", {}).get("environment")
    if isinstance(environment, dict):
        if criteria.get("require_hardware_profile") and not _is_controlled_label(
            environment.get("hardware_profile")
        ):
            issues.append(
                ReadinessIssue(
                    "run_manifest.release_tuple.environment.hardware_profile",
                    environment.get("hardware_profile"),
                    "controlled value",
                    "run manifest must pin the client hardware profile",
                )
            )
        levels = environment.get("concurrency_levels")
        if isinstance(levels, list):
            missing_levels = sorted(set(criteria["required_concurrency_levels"]) - set(levels))
            if missing_levels:
                issues.append(
                    ReadinessIssue(
                        "run_manifest.release_tuple.environment.concurrency_levels",
                        levels,
                        criteria["required_concurrency_levels"],
                        f"run manifest must pin concurrency levels: {missing_levels}",
                    )
                )
    min_trials = criteria.get("min_trials_per_scenario", 1)
    reports = run_manifest.get("reports")
    if isinstance(reports, list):
        for index, report in enumerate(reports):
            if not isinstance(report, dict):
                continue
            observed_trials = report.get("num_trials_per_scenario")
            if not isinstance(observed_trials, int) or observed_trials < min_trials:
                issues.append(
                    ReadinessIssue(
                        f"run_manifest.reports[{index}].num_trials_per_scenario",
                        observed_trials,
                        f">={min_trials}",
                        "official frontier releases must include repeated trials for statistical reporting",
                    )
                )
    if criteria["require_judge_manifest_pinning"]:
        judge = run_manifest.get("release_tuple", {}).get("judge")
        if not isinstance(judge, dict):
            issues.append(
                ReadinessIssue(
                    "run_manifest.release_tuple.judge",
                    False,
                    True,
                    "run manifest must pin judge metadata",
                )
            )
            return
        if not _is_controlled_label(judge.get("model")):
            issues.append(
                ReadinessIssue(
                    "run_manifest.release_tuple.judge.model",
                    judge.get("model"),
                    "controlled value",
                    "official judged releases must pin the judge model",
                )
            )
        prompt = judge.get("prompt")
        if not isinstance(prompt, dict) or not _looks_sha256(prompt.get("sha256")):
            issues.append(
                ReadinessIssue(
                    "run_manifest.release_tuple.judge.prompt",
                    prompt,
                    "prompt file with SHA-256",
                    "official judged releases must pin the judge prompt hash",
                )
            )


def _check_plot_artifacts(
    issues: list[ReadinessIssue],
    plot_dir: str | Path | None,
    criteria: dict[str, Any],
    *,
    frontier_report: dict[str, Any] | None = None,
) -> None:
    if plot_dir is None:
        if criteria["require_plot_artifacts"]:
            issues.append(
                ReadinessIssue(
                    "frontier_plot_artifacts",
                    False,
                    True,
                    "frontier release profiles require generated plot artifacts",
                )
            )
        return

    plot_path = Path(plot_dir)
    plot_json = plot_path / "frontier_plot_data.json"
    if not plot_json.exists():
        issues.append(
            ReadinessIssue(
                "frontier_plot_artifacts.plot_data",
                False,
                True,
                "frontier_plot_data.json must exist",
            )
        )
        return
    try:
        plot_data = json.loads(plot_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(
            ReadinessIssue(
                "frontier_plot_artifacts.plot_data",
                str(exc),
                "valid JSON",
                "frontier plot data must be valid JSON",
            )
        )
        return

    domains = plot_data.get("domains")
    if not isinstance(domains, dict) or not domains:
        issues.append(
            ReadinessIssue(
                "frontier_plot_artifacts.domains",
                0,
                ">0",
                "frontier plot data must contain domain plots",
            )
        )
        return

    if frontier_report is not None:
        expected_plot_data = build_frontier_plot_data(frontier_report)
        if plot_data != expected_plot_data:
            issues.append(
                ReadinessIssue(
                    "frontier_plot_artifacts.plot_data",
                    "does not match supplied frontier report",
                    "regenerated plot data from frontier report",
                    "frontier plot data must be generated from the supplied frontier report",
                )
            )

    for domain in domains:
        safe_domain = _slug(str(domain))
        for projection in ("3d", "latency_vs_quality", "cost_vs_quality"):
            svg_path = plot_path / f"{safe_domain}_{projection}.svg"
            if not svg_path.exists():
                issues.append(
                    ReadinessIssue(
                        f"frontier_plot_artifacts.{domain}.{projection}",
                        False,
                        True,
                        "required SVG plot artifact is missing",
                    )
                )
                continue
            text = svg_path.read_text(encoding="utf-8")
            if "<svg" not in text or "</svg>" not in text:
                issues.append(
                    ReadinessIssue(
                        f"frontier_plot_artifacts.{domain}.{projection}",
                        "invalid",
                        "svg",
                        "plot artifact must be an SVG file",
                    )
                )


def _require_min(
    issues: list[ReadinessIssue],
    criterion: str,
    observed: Any,
    required: int,
    message: str,
) -> None:
    if not isinstance(observed, int):
        observed = 0
    if observed < required:
        issues.append(ReadinessIssue(criterion, observed, required, message))


def _require_min_rate(
    issues: list[ReadinessIssue],
    criterion: str,
    observed: Any,
    required: float,
    message: str,
) -> None:
    if not isinstance(observed, (int, float)):
        observed = 0.0
    if observed < required:
        issues.append(ReadinessIssue(criterion, observed, required, message))


def _is_controlled_label(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value.strip() != "unspecified"


def _looks_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def _is_nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0.0


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return slug or "all"
