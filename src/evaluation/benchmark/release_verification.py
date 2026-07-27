"""End-to-end release verification for OpenVoiceCS-Bench artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.baselines import (
    DEFAULT_BASELINE_MANIFEST_PATH,
    validate_reference_baselines_file,
)
from src.evaluation.benchmark.changelog import (
    DEFAULT_CHANGELOG_PATH,
    validate_changelog_file,
)
from src.evaluation.benchmark.claims import (
    DEFAULT_CLAIMS_MANIFEST_PATH,
    validate_claims_manifest_file,
)
from src.evaluation.benchmark.datasheet import (
    DEFAULT_DATASHEET_PATH,
    validate_benchmark_datasheet_file,
)
from src.evaluation.benchmark.external_endpoint import (
    DEFAULT_EXTERNAL_ENDPOINT_CONTRACT_PATH,
    validate_external_endpoint_contract_file,
)
from src.evaluation.benchmark.external_systems import (
    DEFAULT_EXTERNAL_SYSTEMS_PATH,
    validate_external_systems_registry_file,
)
from src.evaluation.benchmark.judging import (
    DEFAULT_JUDGE_ANNOTATION_PACKAGE_PATH,
    DEFAULT_JUDGE_PROTOCOL_PATH,
    DEFAULT_JUDGE_STUDY_PATH,
    validate_judge_annotation_package_file,
    validate_judge_protocol_file,
    validate_judge_study_manifest_file,
)
from src.evaluation.benchmark.openvoicecs import (
    DEFAULT_AUDIO_MANIFEST_PATH,
    DEFAULT_SCENARIO_PATH,
    DEFAULT_SUBMISSION_INTAKE_PATH,
    build_release_audit,
    validate_audio_assets_file,
    validate_audio_manifest_file,
    validate_suite_file,
)
from src.evaluation.benchmark.pricing import (
    DEFAULT_PRICING_MANIFEST_PATH,
    validate_pricing_manifest_file,
)
from src.evaluation.benchmark.provenance import (
    DEFAULT_PROVENANCE_MANIFEST_PATH,
    validate_provenance_manifest_file,
)
from src.evaluation.benchmark.readiness import evaluate_release_readiness
from src.evaluation.benchmark.reviews import (
    DEFAULT_REVIEW_MANIFEST_PATH,
    validate_review_manifest_file,
)
from src.evaluation.benchmark.sealed import (
    DEFAULT_SEALED_OPS_PATH,
    DEFAULT_SEALED_QUEUE_PATH,
    validate_sealed_ops_manifest_file,
    validate_sealed_queue_manifest_file,
)
from src.evaluation.benchmark.splits import (
    DEFAULT_SPLIT_COMMITMENT_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    validate_split_commitments_file,
    validate_split_manifest_file,
)

DEFAULT_RELEASE_AUDIT_PATH = Path("data/openvoicecs/release_audit.json")


@dataclass(frozen=True)
class ReleaseVerificationIssue:
    """Structured release verification issue."""

    check: str
    item_id: str
    path: str
    message: str


def verify_openvoicecs_release(
    *,
    scenario_path: str | Path = DEFAULT_SCENARIO_PATH,
    audio_manifest_path: str | Path | None = DEFAULT_AUDIO_MANIFEST_PATH,
    audio_asset_root: str | Path = ".",
    require_audio_assets: bool = False,
    pricing_manifest_path: str | Path | None = DEFAULT_PRICING_MANIFEST_PATH,
    split_manifest_path: str | Path | None = DEFAULT_SPLIT_MANIFEST_PATH,
    split_commitment_path: str | Path | None = DEFAULT_SPLIT_COMMITMENT_PATH,
    provenance_manifest_path: str | Path | None = DEFAULT_PROVENANCE_MANIFEST_PATH,
    changelog_path: str | Path | None = DEFAULT_CHANGELOG_PATH,
    baseline_manifest_path: str | Path | None = DEFAULT_BASELINE_MANIFEST_PATH,
    review_manifest_path: str | Path | None = DEFAULT_REVIEW_MANIFEST_PATH,
    datasheet_path: str | Path | None = DEFAULT_DATASHEET_PATH,
    judge_protocol_path: str | Path | None = DEFAULT_JUDGE_PROTOCOL_PATH,
    judge_study_path: str | Path | None = DEFAULT_JUDGE_STUDY_PATH,
    judge_annotation_package_path: str | Path | None = DEFAULT_JUDGE_ANNOTATION_PACKAGE_PATH,
    sealed_ops_path: str | Path | None = DEFAULT_SEALED_OPS_PATH,
    sealed_queue_path: str | Path | None = DEFAULT_SEALED_QUEUE_PATH,
    external_endpoint_contract_path: str | Path | None = DEFAULT_EXTERNAL_ENDPOINT_CONTRACT_PATH,
    external_systems_path: str | Path | None = DEFAULT_EXTERNAL_SYSTEMS_PATH,
    claims_manifest_path: str | Path | None = DEFAULT_CLAIMS_MANIFEST_PATH,
    submission_intake_path: str | Path | None = DEFAULT_SUBMISSION_INTAKE_PATH,
    release_audit_path: str | Path | None = DEFAULT_RELEASE_AUDIT_PATH,
    readiness_profile: str = "seed",
    frontier_report_path: str | Path | None = None,
    run_manifest_path: str | Path | None = None,
    plot_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify that the versioned benchmark release artifacts are coherent."""
    scenario_path = Path(scenario_path)
    suite = _load_json(scenario_path)
    scenarios = suite.get("scenarios", []) if isinstance(suite, dict) else []
    scenario_ids = {
        scenario.get("id")
        for scenario in scenarios
        if isinstance(scenario, dict) and scenario.get("id")
    }
    audio_variant_ids = _audio_variant_ids(audio_manifest_path)

    checks: list[dict[str, Any]] = []
    issues: list[ReleaseVerificationIssue] = []

    _run_check(
        checks,
        issues,
        "scenario_suite",
        lambda: validate_suite_file(scenario_path),
    )
    if audio_manifest_path is not None:
        _run_check(
            checks,
            issues,
            "audio_manifest",
            lambda: validate_audio_manifest_file(
                audio_manifest_path,
                scenario_ids=scenario_ids,
            ),
        )
        if require_audio_assets:
            _run_check(
                checks,
                issues,
                "audio_assets",
                lambda: validate_audio_assets_file(
                    audio_manifest_path,
                    root_dir=audio_asset_root,
                    scenario_ids=scenario_ids,
                ),
            )
    if pricing_manifest_path is not None:
        _run_check(
            checks,
            issues,
            "pricing_manifest",
            lambda: validate_pricing_manifest_file(pricing_manifest_path),
        )
    if split_manifest_path is not None:
        _run_check(
            checks,
            issues,
            "split_manifest",
            lambda: validate_split_manifest_file(
                split_manifest_path,
                scenario_ids=scenario_ids,
                audio_variant_ids=audio_variant_ids,
            ),
        )
    if split_commitment_path is not None:
        _run_check(
            checks,
            issues,
            "split_commitments",
            lambda: validate_split_commitments_file(
                split_commitment_path,
                scenario_path=scenario_path,
                split_path=split_manifest_path,
                audio_manifest_path=audio_manifest_path,
            ),
        )
    if provenance_manifest_path is not None:
        _run_check(
            checks,
            issues,
            "provenance_manifest",
            lambda: validate_provenance_manifest_file(
                provenance_manifest_path,
                scenario_ids=scenario_ids,
                audio_variant_ids=audio_variant_ids,
            ),
        )
    if changelog_path is not None:
        _run_check(
            checks,
            issues,
            "changelog",
            lambda: validate_changelog_file(
                changelog_path,
                scenario_ids=scenario_ids,
                audio_variant_ids=audio_variant_ids,
                benchmark_version=suite.get("version") if isinstance(suite, dict) else None,
            ),
        )
    if baseline_manifest_path is not None:
        _run_check(
            checks,
            issues,
            "reference_baselines",
            lambda: validate_reference_baselines_file(baseline_manifest_path),
        )
    if review_manifest_path is not None:
        _run_check(
            checks,
            issues,
            "scenario_reviews",
            lambda: validate_review_manifest_file(
                review_manifest_path,
                scenario_ids=scenario_ids,
                benchmark_version=suite.get("version") if isinstance(suite, dict) else None,
            ),
        )
    if datasheet_path is not None:
        _run_check(
            checks,
            issues,
            "datasheet",
            lambda: validate_benchmark_datasheet_file(datasheet_path),
        )
    if judge_protocol_path is not None:
        _run_check(
            checks,
            issues,
            "judge_protocol",
            lambda: validate_judge_protocol_file(judge_protocol_path),
        )
    if judge_study_path is not None:
        _run_check(
            checks,
            issues,
            "judge_study",
            lambda: validate_judge_study_manifest_file(judge_study_path),
        )
    if judge_annotation_package_path is not None:
        _run_check(
            checks,
            issues,
            "judge_annotation_package",
            lambda: validate_judge_annotation_package_file(judge_annotation_package_path),
        )
    if sealed_ops_path is not None:
        _run_check(
            checks,
            issues,
            "sealed_ops",
            lambda: validate_sealed_ops_manifest_file(
                sealed_ops_path,
                split_manifest_path=split_manifest_path,
                split_commitment_path=split_commitment_path,
            ),
        )
    if sealed_queue_path is not None:
        _run_check(
            checks,
            issues,
            "sealed_queue",
            lambda: validate_sealed_queue_manifest_file(
                sealed_queue_path,
                sealed_ops_path=sealed_ops_path,
                split_commitment_path=split_commitment_path,
            ),
        )
    if external_systems_path is not None:
        _run_check(
            checks,
            issues,
            "external_systems",
            lambda: validate_external_systems_registry_file(external_systems_path),
        )
    if external_endpoint_contract_path is not None:
        _run_check(
            checks,
            issues,
            "external_endpoint_contract",
            lambda: validate_external_endpoint_contract_file(external_endpoint_contract_path),
        )
    if claims_manifest_path is not None:
        _run_check(
            checks,
            issues,
            "leaderboard_claims",
            lambda: validate_claims_manifest_file(claims_manifest_path),
        )
    if submission_intake_path is not None:
        from src.evaluation.benchmark.submission import validate_submission_intake_file

        _run_check(
            checks,
            issues,
            "submission_intake",
            lambda: validate_submission_intake_file(submission_intake_path),
        )

    audit = build_release_audit(
        scenario_path=scenario_path,
        audio_manifest_path=audio_manifest_path,
        audio_asset_root=audio_asset_root,
        pricing_manifest_path=pricing_manifest_path,
        split_manifest_path=split_manifest_path,
        split_commitment_path=split_commitment_path,
        provenance_manifest_path=provenance_manifest_path,
        changelog_path=changelog_path,
        baseline_manifest_path=baseline_manifest_path,
        review_manifest_path=review_manifest_path,
        judge_protocol_path=judge_protocol_path,
        judge_study_path=judge_study_path,
        judge_annotation_package_path=judge_annotation_package_path,
        sealed_ops_path=sealed_ops_path,
        sealed_queue_path=sealed_queue_path,
        external_endpoint_contract_path=external_endpoint_contract_path,
        external_systems_path=external_systems_path,
        claims_manifest_path=claims_manifest_path,
        submission_intake_path=submission_intake_path,
    )
    _record_audit_checks(checks, issues, audit)
    if release_audit_path is not None:
        _record_saved_audit_check(checks, issues, audit, Path(release_audit_path))
    _record_readiness_check(
        checks,
        issues,
        audit,
        profile=readiness_profile,
        frontier_report_path=frontier_report_path,
        run_manifest_path=run_manifest_path,
        plot_dir=plot_dir,
    )

    return {
        "benchmark": "OpenVoiceCS-Bench",
        "verification_version": "0.1.0",
        "passed": not issues,
        "num_checks": len(checks),
        "num_issues": len(issues),
        "checks": checks,
        "issues": [
            {
                "check": issue.check,
                "item_id": issue.item_id,
                "path": issue.path,
                "message": issue.message,
            }
            for issue in issues
        ],
        "release_gates": audit.get("release_gates", {}),
        "readiness_profile": readiness_profile,
    }


def _run_check(
    checks: list[dict[str, Any]],
    issues: list[ReleaseVerificationIssue],
    name: str,
    fn: Callable[[], list[Any]],
) -> None:
    check_issues = [
        _normalize_issue(name, issue)
        for issue in fn()
    ]
    issues.extend(check_issues)
    checks.append({
        "name": name,
        "passed": not check_issues,
        "num_issues": len(check_issues),
    })


def _record_audit_checks(
    checks: list[dict[str, Any]],
    issues: list[ReleaseVerificationIssue],
    audit: dict[str, Any],
) -> None:
    validation = audit.get("validation", {})
    validation_issues = []
    if validation.get("passed") is not True:
        validation_issues.append(
            ReleaseVerificationIssue(
                "release_audit",
                "<audit>",
                "validation",
                "release audit validation must pass",
            )
        )
    gates = audit.get("release_gates", {})
    gate_issues = [
        ReleaseVerificationIssue(
            "release_gates",
            gate,
            f"release_gates.{gate}",
            "release gate must pass",
        )
        for gate, passed in gates.items()
        if gate != "passed" and passed is not True
    ]
    if gates.get("passed") is not True:
        gate_issues.append(
            ReleaseVerificationIssue(
                "release_gates",
                "<gates>",
                "release_gates.passed",
                "all release gates must pass",
            )
        )

    issues.extend(validation_issues)
    issues.extend(gate_issues)
    checks.append({
        "name": "release_audit",
        "passed": not validation_issues,
        "num_issues": len(validation_issues),
    })
    checks.append({
        "name": "release_gates",
        "passed": not gate_issues,
        "num_issues": len(gate_issues),
    })


def _record_saved_audit_check(
    checks: list[dict[str, Any]],
    issues: list[ReleaseVerificationIssue],
    computed_audit: dict[str, Any],
    release_audit_path: Path,
) -> None:
    check_issues = []
    if not release_audit_path.exists():
        check_issues.append(
            ReleaseVerificationIssue(
                "saved_release_audit",
                str(release_audit_path),
                "path",
                "file does not exist",
            )
        )
    else:
        saved_audit = _load_json(release_audit_path)
        for field in ("benchmark", "version", "release_stage", "files", "release_gates"):
            if saved_audit.get(field) != computed_audit.get(field):
                check_issues.append(
                    ReleaseVerificationIssue(
                        "saved_release_audit",
                        field,
                        field,
                        "does not match freshly computed release audit",
                    )
                )
        if saved_audit.get("validation", {}).get("passed") != computed_audit.get(
            "validation", {}
        ).get("passed"):
            check_issues.append(
                ReleaseVerificationIssue(
                    "saved_release_audit",
                    "validation",
                    "validation.passed",
                    "does not match freshly computed release audit",
                )
            )
    issues.extend(check_issues)
    checks.append({
        "name": "saved_release_audit",
        "passed": not check_issues,
        "num_issues": len(check_issues),
    })


def _record_readiness_check(
    checks: list[dict[str, Any]],
    issues: list[ReleaseVerificationIssue],
    audit: dict[str, Any],
    *,
    profile: str,
    frontier_report_path: str | Path | None,
    run_manifest_path: str | Path | None,
    plot_dir: str | Path | None,
) -> None:
    check_name = f"readiness:{profile}"
    check_issues: list[ReleaseVerificationIssue] = []
    frontier_report = None
    if frontier_report_path is not None:
        path = Path(frontier_report_path)
        if path.exists():
            frontier_report = _load_json(path)
        else:
            check_issues.append(
                ReleaseVerificationIssue(check_name, str(path), "frontier_report", "file does not exist")
            )
    run_manifest = None
    run_manifest_base_dir: str | Path = "."
    verify_run_manifest_files = False
    if run_manifest_path is not None:
        path = Path(run_manifest_path)
        run_manifest_base_dir = path.parent
        verify_run_manifest_files = True
        if path.exists():
            run_manifest = _load_json(path)
        else:
            check_issues.append(
                ReleaseVerificationIssue(check_name, str(path), "run_manifest", "file does not exist")
            )

    readiness = evaluate_release_readiness(
        audit,
        profile=profile,
        frontier_report=frontier_report,
        run_manifest=run_manifest,
        run_manifest_base_dir=run_manifest_base_dir,
        verify_run_manifest_files=verify_run_manifest_files,
        plot_dir=plot_dir,
    )
    check_issues.extend(
        ReleaseVerificationIssue(
            check_name,
            issue.get("criterion", "<readiness>"),
            issue.get("criterion", "<readiness>"),
            issue.get("message", "readiness criterion failed"),
        )
        for issue in readiness.get("issues", [])
    )
    issues.extend(check_issues)
    checks.append({
        "name": check_name,
        "passed": not check_issues,
        "num_issues": len(check_issues),
    })


def _normalize_issue(check: str, issue: Any) -> ReleaseVerificationIssue:
    return ReleaseVerificationIssue(
        check=check,
        item_id=str(
            getattr(issue, "scenario_id", None)
            or getattr(issue, "item_id", None)
            or getattr(issue, "path", None)
            or "<release>"
        ),
        path=str(getattr(issue, "path", "<root>")),
        message=str(getattr(issue, "message", issue)),
    )


def _audio_variant_ids(audio_manifest_path: str | Path | None) -> set[str]:
    if audio_manifest_path is None:
        return set()
    manifest = _load_json(audio_manifest_path)
    variants = manifest.get("variants", []) if isinstance(manifest, dict) else []
    return {
        variant.get("id")
        for variant in variants
        if isinstance(variant, dict) and variant.get("id")
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}
