"""Offline judge-rubric aggregation for OpenVoiceCS subjective quality.

Deterministic state oracles should remain the primary score. This module adds a
separate, auditable path for human or model-judge annotations of conversation
quality dimensions such as empathy, clarity, and voice-channel fit.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.datapaths import data_path
from src.evaluation.benchmark.provider_adapters import (
    OPENAI_COMPATIBLE_BASE_URLS,
    PROVIDER_ENV_KEYS,
    get_provider_api_key,
    load_workspace_env,
)

DEFAULT_JUDGE_RUBRIC_PATH = data_path("judge_rubric_v0.1.json")
DEFAULT_JUDGE_PROTOCOL_PATH = data_path("judging", "judge_protocol_v0.1.json")
DEFAULT_JUDGE_ANNOTATION_PACKAGE_PATH = Path(
    "data/openvoicecs/judging/judge_annotation_package_v0.1.json"
)
DEFAULT_JUDGE_STUDY_PATH = data_path("judging", "judge_study_v0.1.json")
JUDGE_STUDY_STATUSES = {"reference_fixture", "planned", "active", "completed", "retired"}


@dataclass(frozen=True)
class JudgeIssue:
    """Structured judge-rubric or annotation validation issue."""

    item_id: str
    path: str
    message: str


@dataclass(frozen=True)
class ModelJudgeSpec:
    """One OpenAI-compatible model used as an audited model judge."""

    provider: str
    model_id: str
    rater_id: str | None = None
    api_key: str | None = None
    base_url: str | None = None


ModelJudgeCaller = Callable[
    [ModelJudgeSpec, list[dict[str, str]], int, float, float],
    str,
]


def load_judge_rubric(path: str | Path = DEFAULT_JUDGE_RUBRIC_PATH) -> dict[str, Any]:
    """Load and validate the judge rubric."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        rubric = json.load(f)
    issues = validate_judge_rubric(rubric)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS judge rubric validation failed:\n{formatted}")
    return rubric


def load_judge_protocol(
    path: str | Path = DEFAULT_JUDGE_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Load and validate the judge protocol."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        protocol = json.load(f)
    issues = validate_judge_protocol(protocol)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS judge protocol validation failed:\n{formatted}")
    return protocol


def load_judge_annotation_package(
    path: str | Path = DEFAULT_JUDGE_ANNOTATION_PACKAGE_PATH,
) -> dict[str, Any]:
    """Load and validate the judge annotation package."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        package = json.load(f)
    issues = validate_judge_annotation_package(package)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(
            f"OpenVoiceCS judge annotation package validation failed:\n{formatted}"
        )
    return package


def load_judge_study_manifest(
    path: str | Path = DEFAULT_JUDGE_STUDY_PATH,
) -> dict[str, Any]:
    """Load and validate the judge study manifest."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        study = json.load(f)
    issues = validate_judge_study_manifest(study)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS judge study validation failed:\n{formatted}")
    return study


def validate_judge_study_manifest_file(
    path: str | Path = DEFAULT_JUDGE_STUDY_PATH,
    *,
    base_dir: str | Path = ".",
) -> list[JudgeIssue]:
    """Validate a saved judge study manifest JSON file."""
    with open(path, encoding="utf-8") as f:
        study = json.load(f)
    return validate_judge_study_manifest(study, base_dir=base_dir)


def validate_judge_study_manifest(
    study: dict[str, Any],
    *,
    base_dir: str | Path = ".",
) -> list[JudgeIssue]:
    """Return all judge-study manifest contract issues."""
    issues: list[JudgeIssue] = []
    if not isinstance(study, dict):
        return [JudgeIssue("<judge_study>", "<root>", "must be an object")]
    required = (
        "name",
        "version",
        "benchmark_version",
        "status",
        "official_judging_eligible",
        "protocol",
        "rubric",
        "prompt",
        "annotation_package",
        "study_design",
        "rater_pool",
        "calibration",
        "blinding",
        "adjudication",
        "audit",
    )
    for field in required:
        if field not in study:
            issues.append(JudgeIssue("<judge_study>", field, "missing required field"))
    if issues:
        return issues
    if study.get("name") != "OpenVoiceCS Judging Study":
        issues.append(JudgeIssue("<judge_study>", "name", "must be OpenVoiceCS Judging Study"))
    for field in ("version", "benchmark_version"):
        if not _non_empty_string(study.get(field)):
            issues.append(JudgeIssue("<judge_study>", field, "must be a non-empty string"))
    if study.get("status") not in JUDGE_STUDY_STATUSES:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "status",
                f"must be one of {sorted(JUDGE_STUDY_STATUSES)}",
            )
        )
    official = study.get("official_judging_eligible")
    if not isinstance(official, bool):
        issues.append(JudgeIssue("<judge_study>", "official_judging_eligible", "must be boolean"))
        official = False
    if study.get("status") == "reference_fixture" and official is not False:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "official_judging_eligible",
                "reference fixtures cannot be official judging evidence",
            )
        )

    base = Path(base_dir)
    protocol = _load_package_file(issues, study.get("protocol"), "protocol", "<judge_study>", base)
    rubric = _load_package_file(issues, study.get("rubric"), "rubric", "<judge_study>", base)
    _load_package_file(
        issues,
        study.get("prompt"),
        "prompt",
        "<judge_study>",
        base,
        parse_json=False,
    )
    annotation_package = _load_package_file(
        issues,
        study.get("annotation_package"),
        "annotation_package",
        "<judge_study>",
        base,
    )
    if isinstance(protocol, dict):
        issues.extend(validate_judge_protocol(protocol))
    if isinstance(rubric, dict):
        issues.extend(validate_judge_rubric(rubric))
    if isinstance(annotation_package, dict):
        issues.extend(
            JudgeIssue("<judge_study>", f"annotation_package.{issue.path}", issue.message)
            for issue in validate_judge_annotation_package(annotation_package)
        )

    _validate_judge_study_design(issues, study.get("study_design"), protocol=protocol)
    _validate_judge_study_rater_pool(
        issues,
        study.get("rater_pool"),
        official_judging_eligible=official is True,
    )
    _validate_judge_study_calibration(issues, study.get("calibration"), protocol=protocol)
    _validate_judge_study_blinding(issues, study.get("blinding"))
    _validate_judge_study_adjudication(issues, study.get("adjudication"), protocol=protocol)
    _validate_judge_study_audit(issues, study.get("audit"))
    if official is True:
        _validate_official_judge_study(issues, study, annotation_package)
    return issues


def judge_study_stats(study: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize judge study governance evidence."""
    if not isinstance(study, dict):
        return {"present": False, "num_raters": 0}
    raters = study.get("rater_pool", {}).get("raters", [])
    raters = raters if isinstance(raters, list) else []
    by_type: dict[str, int] = {}
    for rater in raters:
        if isinstance(rater, dict):
            rater_type = str(rater.get("type") or "unknown")
            by_type[rater_type] = by_type.get(rater_type, 0) + 1
    return {
        "present": True,
        "version": study.get("version"),
        "benchmark_version": study.get("benchmark_version"),
        "status": study.get("status"),
        "official_judging_eligible": study.get("official_judging_eligible"),
        "num_raters": len(raters),
        "by_rater_type": by_type,
        "minimum_raters_per_item": study.get("study_design", {}).get("minimum_raters_per_item"),
        "calibration_items": study.get("calibration", {}).get("minimum_training_items"),
    }


def validate_judge_annotation_package_file(
    path: str | Path = DEFAULT_JUDGE_ANNOTATION_PACKAGE_PATH,
    *,
    base_dir: str | Path = ".",
) -> list[JudgeIssue]:
    """Validate a saved judge annotation package JSON file."""
    with open(path, encoding="utf-8") as f:
        package = json.load(f)
    return validate_judge_annotation_package(package, base_dir=base_dir)


def validate_judge_annotation_package(
    package: dict[str, Any],
    *,
    base_dir: str | Path = ".",
) -> list[JudgeIssue]:
    """Return all judge-annotation package contract issues."""
    issues: list[JudgeIssue] = []
    if not isinstance(package, dict):
        return [JudgeIssue("<judge_annotation_package>", "<root>", "must be an object")]
    for field in (
        "name",
        "version",
        "benchmark_version",
        "annotation_mode",
        "official_judging",
        "protocol",
        "rubric",
        "prompt",
        "rater_manifest",
        "packages",
    ):
        if field not in package:
            issues.append(
                JudgeIssue("<judge_annotation_package>", field, "missing required field")
            )
    if issues:
        return issues
    if package.get("name") != "OpenVoiceCS Judge Annotation Package":
        issues.append(
            JudgeIssue(
                "<judge_annotation_package>",
                "name",
                "must be OpenVoiceCS Judge Annotation Package",
            )
        )
    for field in ("version", "benchmark_version"):
        if not _non_empty_string(package.get(field)):
            issues.append(
                JudgeIssue(
                    "<judge_annotation_package>",
                    field,
                    "must be a non-empty string",
                )
            )
    annotation_mode = package.get("annotation_mode")
    allowed_modes = {"reference_fixture", "human", "model", "human_and_model"}
    if annotation_mode not in allowed_modes:
        issues.append(
            JudgeIssue(
                "<judge_annotation_package>",
                "annotation_mode",
                f"must be one of {sorted(allowed_modes)}",
            )
        )
    official = package.get("official_judging")
    if not isinstance(official, bool):
        issues.append(
            JudgeIssue(
                "<judge_annotation_package>",
                "official_judging",
                "must be boolean",
            )
        )
    elif official and annotation_mode == "reference_fixture":
        issues.append(
            JudgeIssue(
                "<judge_annotation_package>",
                "annotation_mode",
                "official judging cannot use reference_fixture annotations",
            )
        )

    base = Path(base_dir)
    protocol = _load_package_file(
        issues,
        package.get("protocol"),
        "protocol",
        "<judge_annotation_package>",
        base,
    )
    rubric = _load_package_file(
        issues,
        package.get("rubric"),
        "rubric",
        "<judge_annotation_package>",
        base,
    )
    _load_package_file(
        issues,
        package.get("prompt"),
        "prompt",
        "<judge_annotation_package>",
        base,
        parse_json=False,
    )
    if isinstance(protocol, dict):
        issues.extend(validate_judge_protocol(protocol))
    if isinstance(rubric, dict):
        issues.extend(validate_judge_rubric(rubric))
    else:
        rubric = {}

    _validate_rater_manifest(
        issues,
        package.get("rater_manifest"),
        annotation_mode=annotation_mode,
    )
    packages = package.get("packages")
    if not isinstance(packages, list) or not packages:
        issues.append(
            JudgeIssue(
                "<judge_annotation_package>",
                "packages",
                "must be a non-empty list",
            )
        )
        return issues
    seen_ids: set[str] = set()
    for index, entry in enumerate(packages):
        _validate_annotation_package_entry(
            issues,
            entry,
            index=index,
            seen_ids=seen_ids,
            rubric=rubric,
            base_dir=base,
        )
    return issues


def judge_annotation_package_stats(package: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize judge annotation package evidence."""
    if not isinstance(package, dict):
        return {"present": False, "num_packages": 0}
    packages = package.get("packages", [])
    packages = packages if isinstance(packages, list) else []
    return {
        "present": True,
        "version": package.get("version"),
        "benchmark_version": package.get("benchmark_version"),
        "annotation_mode": package.get("annotation_mode"),
        "official_judging": package.get("official_judging"),
        "num_packages": len(packages),
        "num_annotations": sum(
            entry.get("num_annotations", 0)
            for entry in packages
            if isinstance(entry, dict) and isinstance(entry.get("num_annotations"), int)
        ),
        "num_items": sum(
            entry.get("num_items", 0)
            for entry in packages
            if isinstance(entry, dict) and isinstance(entry.get("num_items"), int)
        ),
    }


def _validate_judge_study_design(
    issues: list[JudgeIssue],
    design: Any,
    *,
    protocol: Any,
) -> None:
    if not isinstance(design, dict):
        issues.append(JudgeIssue("<judge_study>", "study_design", "must be an object"))
        return
    target_systems = design.get("target_systems")
    if not isinstance(target_systems, list) or not target_systems:
        issues.append(JudgeIssue("<judge_study>", "study_design.target_systems", "must be a non-empty list"))
    elif any(not _non_empty_string(system_id) for system_id in target_systems):
        issues.append(JudgeIssue("<judge_study>", "study_design.target_systems", "all entries must be non-empty strings"))
    sampling_unit = design.get("sampling_unit")
    if sampling_unit not in {"scenario", "trial", "conversation"}:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "study_design.sampling_unit",
                "must be scenario, trial, or conversation",
            )
        )
    minimum_items = design.get("minimum_items_per_system")
    if isinstance(minimum_items, bool) or not isinstance(minimum_items, int) or minimum_items < 30:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "study_design.minimum_items_per_system",
                "must be an integer >= 30",
            )
        )
    minimum_raters = design.get("minimum_raters_per_item")
    if isinstance(minimum_raters, bool) or not isinstance(minimum_raters, int) or minimum_raters < 2:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "study_design.minimum_raters_per_item",
                "must be an integer >= 2",
            )
        )
    protocol_raters = protocol.get("minimum_raters_per_item") if isinstance(protocol, dict) else None
    if isinstance(minimum_raters, int) and isinstance(protocol_raters, int) and minimum_raters < protocol_raters:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "study_design.minimum_raters_per_item",
                "must be >= protocol minimum_raters_per_item",
            )
        )
    stratification = design.get("stratification_fields")
    if not isinstance(stratification, list) or not stratification:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "study_design.stratification_fields",
                "must be a non-empty list",
            )
        )
    elif not {"domain", "track", "difficulty"}.issubset(set(stratification)):
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "study_design.stratification_fields",
                "must include domain, track, and difficulty",
            )
        )
    if not _non_empty_string(design.get("sampling_policy")):
        issues.append(JudgeIssue("<judge_study>", "study_design.sampling_policy", "must be a non-empty string"))


def _validate_judge_study_rater_pool(
    issues: list[JudgeIssue],
    pool: Any,
    *,
    official_judging_eligible: bool,
) -> None:
    if not isinstance(pool, dict):
        issues.append(JudgeIssue("<judge_study>", "rater_pool", "must be an object"))
        return
    raters = pool.get("raters")
    if not isinstance(raters, list) or len(raters) < 2:
        issues.append(JudgeIssue("<judge_study>", "rater_pool.raters", "must contain at least two raters"))
        return
    seen: set[str] = set()
    for index, rater in enumerate(raters):
        path = f"rater_pool.raters[{index}]"
        if not isinstance(rater, dict):
            issues.append(JudgeIssue("<judge_study>", path, "must be an object"))
            continue
        rater_id = rater.get("id")
        if not _non_empty_string(rater_id):
            issues.append(JudgeIssue("<judge_study>", f"{path}.id", "must be a non-empty string"))
            rater_id = path
        elif rater_id in seen:
            issues.append(JudgeIssue(str(rater_id), f"{path}.id", "duplicate rater id"))
        seen.add(str(rater_id))
        if rater.get("type") not in {"reference_fixture", "trained_human", "audited_model_judge"}:
            issues.append(
                JudgeIssue(
                    str(rater_id),
                    f"{path}.type",
                    "must be reference_fixture, trained_human, or audited_model_judge",
                )
            )
        for field in ("qualified", "conflict_attested", "domain_policy_trained", "calibration_passed"):
            if rater.get(field) is not True:
                issues.append(JudgeIssue(str(rater_id), f"{path}.{field}", "must be true"))
        if official_judging_eligible and rater.get("type") == "reference_fixture":
            issues.append(
                JudgeIssue(
                    str(rater_id),
                    f"{path}.type",
                    "official judge studies cannot use reference_fixture raters",
                )
            )


def _validate_judge_study_calibration(
    issues: list[JudgeIssue],
    calibration: Any,
    *,
    protocol: Any,
) -> None:
    if not isinstance(calibration, dict):
        issues.append(JudgeIssue("<judge_study>", "calibration", "must be an object"))
        return
    training_items = calibration.get("minimum_training_items")
    if isinstance(training_items, bool) or not isinstance(training_items, int) or training_items < 3:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "calibration.minimum_training_items",
                "must be an integer >= 3",
            )
        )
    protocol_training = None
    if isinstance(protocol, dict):
        protocol_training = protocol.get("rater_qualification", {}).get("minimum_training_items")
    if isinstance(training_items, int) and isinstance(protocol_training, int) and training_items < protocol_training:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "calibration.minimum_training_items",
                "must be >= protocol rater_qualification.minimum_training_items",
            )
        )
    _study_fraction_field(issues, calibration, "gold_items_fraction", min_value=0.01)
    _study_fraction_field(issues, calibration, "duplicate_items_fraction", min_value=0.01)
    threshold = calibration.get("calibration_pass_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.5 <= threshold <= 1.0:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "calibration.calibration_pass_threshold",
                "must be a number between 0.5 and 1.0",
            )
        )
    if calibration.get("rater_drift_review") not in {"per_batch", "per_submission_batch", "continuous"}:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "calibration.rater_drift_review",
                "must be per_batch, per_submission_batch, or continuous",
            )
        )


def _validate_judge_study_blinding(
    issues: list[JudgeIssue],
    blinding: Any,
) -> None:
    if not isinstance(blinding, dict):
        issues.append(JudgeIssue("<judge_study>", "blinding", "must be an object"))
        return
    for field in ("hide_system_identity", "shuffle_item_order", "hide_expected_actions"):
        if blinding.get(field) is not True:
            issues.append(JudgeIssue("<judge_study>", f"blinding.{field}", "must be true"))
    if blinding.get("mask_submission_metadata") is not True:
        issues.append(JudgeIssue("<judge_study>", "blinding.mask_submission_metadata", "must be true"))


def _validate_judge_study_adjudication(
    issues: list[JudgeIssue],
    adjudication: Any,
    *,
    protocol: Any,
) -> None:
    if not isinstance(adjudication, dict):
        issues.append(JudgeIssue("<judge_study>", "adjudication", "must be an object"))
        return
    if adjudication.get("required_for_disagreement") is not True:
        issues.append(JudgeIssue("<judge_study>", "adjudication.required_for_disagreement", "must be true"))
    threshold = adjudication.get("disagreement_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold <= 0:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "adjudication.disagreement_threshold",
                "must be a positive number",
            )
        )
    protocol_threshold = protocol.get("adjudication", {}).get("disagreement_threshold") if isinstance(protocol, dict) else None
    if (
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and isinstance(protocol_threshold, (int, float))
        and threshold > protocol_threshold
    ):
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "adjudication.disagreement_threshold",
                "must be <= protocol disagreement_threshold",
            )
        )
    if adjudication.get("adjudicator_must_be_independent") is not True:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "adjudication.adjudicator_must_be_independent",
                "must be true",
            )
        )


def _validate_judge_study_audit(
    issues: list[JudgeIssue],
    audit: Any,
) -> None:
    if not isinstance(audit, dict):
        issues.append(JudgeIssue("<judge_study>", "audit", "must be an object"))
        return
    for field in (
        "audit_log_required",
        "conflict_attestation_required",
        "publish_agreement_summary",
        "retain_raw_annotations",
    ):
        if audit.get(field) is not True:
            issues.append(JudgeIssue("<judge_study>", f"audit.{field}", "must be true"))
    if not _non_empty_string(audit.get("retention")):
        issues.append(JudgeIssue("<judge_study>", "audit.retention", "must be a non-empty string"))


def _validate_official_judge_study(
    issues: list[JudgeIssue],
    study: dict[str, Any],
    annotation_package: Any,
) -> None:
    if study.get("status") != "completed":
        issues.append(JudgeIssue("<judge_study>", "status", "official judge studies must be completed"))
    if not isinstance(annotation_package, dict):
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "annotation_package",
                "official judge studies must include a valid annotation package",
            )
        )
        return
    if annotation_package.get("official_judging") is not True:
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "annotation_package.official_judging",
                "official judge studies require official annotation package evidence",
            )
        )
    if annotation_package.get("annotation_mode") == "reference_fixture":
        issues.append(
            JudgeIssue(
                "<judge_study>",
                "annotation_package.annotation_mode",
                "official judge studies cannot use reference_fixture annotations",
            )
        )


def _study_fraction_field(
    issues: list[JudgeIssue],
    obj: dict[str, Any],
    field: str,
    *,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> None:
    value = obj.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not min_value <= value <= max_value
    ):
        issues.append(
            JudgeIssue(
                "<judge_study>",
                f"calibration.{field}",
                f"must be a number between {min_value} and {max_value}",
            )
        )


def validate_judge_protocol_file(
    path: str | Path = DEFAULT_JUDGE_PROTOCOL_PATH,
    *,
    base_dir: str | Path = ".",
) -> list[JudgeIssue]:
    """Validate a saved judge protocol JSON file."""
    with open(path, encoding="utf-8") as f:
        protocol = json.load(f)
    return validate_judge_protocol(protocol, base_dir=base_dir)


def validate_judge_protocol(
    protocol: dict[str, Any],
    *,
    base_dir: str | Path = ".",
) -> list[JudgeIssue]:
    """Return all judge-protocol contract issues."""
    issues: list[JudgeIssue] = []
    if not isinstance(protocol, dict):
        return [JudgeIssue("<judge_protocol>", "<root>", "must be an object")]

    required = (
        "name",
        "version",
        "benchmark_version",
        "rubric_path",
        "prompt_path",
        "annotation_mode",
        "minimum_raters_per_item",
        "minimum_alpha_for_release",
        "blinding",
        "adjudication",
        "rater_qualification",
        "quality_controls",
        "published_artifacts",
    )
    for field in required:
        if field not in protocol:
            issues.append(JudgeIssue("<judge_protocol>", field, "missing required field"))
    if issues:
        return issues

    for field in ("name", "version", "benchmark_version"):
        if not isinstance(protocol.get(field), str) or not protocol[field].strip():
            issues.append(JudgeIssue("<judge_protocol>", field, "must be a non-empty string"))

    _validate_protocol_path(issues, protocol, "rubric_path", base_dir)
    _validate_protocol_path(issues, protocol, "prompt_path", base_dir)

    allowed_modes = {"reference_fixture", "human", "model", "human_and_model"}
    if protocol.get("annotation_mode") not in allowed_modes:
        issues.append(
            JudgeIssue(
                "<judge_protocol>",
                "annotation_mode",
                f"must be one of {sorted(allowed_modes)}",
            )
        )

    minimum_raters = protocol.get("minimum_raters_per_item")
    if (
        isinstance(minimum_raters, bool)
        or not isinstance(minimum_raters, int)
        or minimum_raters < 2
    ):
        issues.append(
            JudgeIssue(
                "<judge_protocol>",
                "minimum_raters_per_item",
                "must be an integer >= 2",
            )
        )
    alpha = protocol.get("minimum_alpha_for_release")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
        issues.append(
            JudgeIssue(
                "<judge_protocol>",
                "minimum_alpha_for_release",
                "must be a number between 0 and 1",
            )
        )

    blinding = _object_field(issues, protocol, "blinding")
    if blinding is not None:
        _require_true(issues, blinding, "hide_system_identity", "blinding.hide_system_identity")
        _require_true(issues, blinding, "shuffle_item_order", "blinding.shuffle_item_order")
        _require_true(issues, blinding, "hide_expected_actions", "blinding.hide_expected_actions")

    adjudication = _object_field(issues, protocol, "adjudication")
    if adjudication is not None:
        _require_true(issues, adjudication, "required_for_disagreement", "adjudication.required_for_disagreement")
        threshold = adjudication.get("disagreement_threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or threshold <= 0
        ):
            issues.append(
                JudgeIssue(
                    "<judge_protocol>",
                    "adjudication.disagreement_threshold",
                    "must be a positive number",
                )
            )

    qualification = _object_field(issues, protocol, "rater_qualification")
    if qualification is not None:
        training_items = qualification.get("minimum_training_items")
        if (
            isinstance(training_items, bool)
            or not isinstance(training_items, int)
            or training_items < 3
        ):
            issues.append(
                JudgeIssue(
                    "<judge_protocol>",
                    "rater_qualification.minimum_training_items",
                    "must be an integer >= 3",
                )
            )
        _require_true(
            issues,
            qualification,
            "requires_domain_policy_review",
            "rater_qualification.requires_domain_policy_review",
        )
        _require_true(
            issues,
            qualification,
            "requires_conflict_attestation",
            "rater_qualification.requires_conflict_attestation",
        )

    controls = _object_field(issues, protocol, "quality_controls")
    if controls is not None:
        _fraction_field(issues, controls, "gold_items_fraction", min_value=0.01)
        _fraction_field(issues, controls, "duplicate_items_fraction", min_value=0.01)
        _fraction_field(issues, controls, "maximum_missing_rate", max_value=0.05)
        _require_true(issues, controls, "audit_log_required", "quality_controls.audit_log_required")

    artifacts = protocol.get("published_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        issues.append(
            JudgeIssue(
                "<judge_protocol>",
                "published_artifacts",
                "must be a non-empty list",
            )
        )
    elif any(not isinstance(item, str) or not item.strip() for item in artifacts):
        issues.append(
            JudgeIssue(
                "<judge_protocol>",
                "published_artifacts",
                "all entries must be non-empty strings",
            )
        )
    return issues


def validate_judge_rubric_file(
    path: str | Path = DEFAULT_JUDGE_RUBRIC_PATH,
) -> list[JudgeIssue]:
    """Validate a saved judge rubric JSON file."""
    with open(path, encoding="utf-8") as f:
        rubric = json.load(f)
    return validate_judge_rubric(rubric)


def validate_judge_rubric(rubric: dict[str, Any]) -> list[JudgeIssue]:
    """Return all judge-rubric contract issues."""
    issues: list[JudgeIssue] = []
    if not isinstance(rubric, dict):
        return [JudgeIssue("<rubric>", "<root>", "must be an object")]
    for field in ("name", "version", "scale", "dimensions"):
        if field not in rubric:
            issues.append(JudgeIssue("<rubric>", field, "missing required field"))
    if issues:
        return issues

    scale = rubric.get("scale")
    if not isinstance(scale, dict):
        issues.append(JudgeIssue("<rubric>", "scale", "must be an object"))
        min_score, max_score = 1, 5
    else:
        min_score = scale.get("min")
        max_score = scale.get("max")
        if not isinstance(min_score, int) or not isinstance(max_score, int):
            issues.append(JudgeIssue("<rubric>", "scale", "min/max must be integers"))
            min_score, max_score = 1, 5
        elif min_score >= max_score:
            issues.append(JudgeIssue("<rubric>", "scale", "min must be less than max"))

    dimensions = rubric.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        issues.append(JudgeIssue("<rubric>", "dimensions", "must be a non-empty list"))
        return issues

    seen_ids = set()
    weight_sum = 0.0
    for index, dimension in enumerate(dimensions):
        path = f"dimensions[{index}]"
        if not isinstance(dimension, dict):
            issues.append(JudgeIssue("<rubric>", path, "must be an object"))
            continue
        dimension_id = dimension.get("id")
        if not dimension_id:
            issues.append(JudgeIssue("<rubric>", f"{path}.id", "missing required field"))
        elif dimension_id in seen_ids:
            issues.append(JudgeIssue(str(dimension_id), f"{path}.id", "duplicate dimension id"))
        seen_ids.add(dimension_id)
        weight = dimension.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or weight <= 0
        ):
            issues.append(
                JudgeIssue(
                    str(dimension_id),
                    f"{path}.weight",
                    "must be a positive number",
                )
            )
        else:
            weight_sum += float(weight)
        for field in ("name", "description"):
            if not dimension.get(field):
                issues.append(
                    JudgeIssue(
                        str(dimension_id),
                        f"{path}.{field}",
                        "missing required field",
                    )
                )

    if abs(weight_sum - 1.0) > 1e-6:
        issues.append(JudgeIssue("<rubric>", "dimensions.weight", "weights must sum to 1.0"))
    return issues


def _validate_protocol_path(
    issues: list[JudgeIssue],
    protocol: dict[str, Any],
    field: str,
    base_dir: str | Path,
) -> None:
    value = protocol.get(field)
    if not isinstance(value, str) or not value.strip():
        issues.append(JudgeIssue("<judge_protocol>", field, "must be a non-empty string"))
        return
    path = Path(value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    if not path.exists():
        issues.append(JudgeIssue("<judge_protocol>", field, "referenced file does not exist"))


def _load_package_file(
    issues: list[JudgeIssue],
    entry: Any,
    path: str,
    item_id: str,
    base_dir: Path,
    *,
    parse_json: bool = True,
) -> Any:
    if not isinstance(entry, dict):
        issues.append(JudgeIssue(item_id, path, "must be an object"))
        return None
    resolved = _validate_package_file_entry(issues, entry, path, item_id, base_dir)
    if resolved is None or not parse_json:
        return None
    try:
        with open(resolved, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        issues.append(JudgeIssue(item_id, path, f"invalid JSON: {exc.msg}"))
        return None


def _validate_package_file_entry(
    issues: list[JudgeIssue],
    entry: Any,
    path: str,
    item_id: str,
    base_dir: Path,
) -> Path | None:
    if not isinstance(entry, dict):
        issues.append(JudgeIssue(item_id, path, "must be an object"))
        return None
    for field in ("path", "sha256", "bytes"):
        if field not in entry:
            issues.append(JudgeIssue(item_id, f"{path}.{field}", "missing required field"))
    raw_path = entry.get("path")
    if not _non_empty_string(raw_path):
        issues.append(JudgeIssue(item_id, f"{path}.path", "must be a non-empty string"))
        return None
    resolved = Path(str(raw_path))
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    if not resolved.exists():
        issues.append(JudgeIssue(item_id, f"{path}.path", "file does not exist"))
        return None
    data = resolved.read_bytes()
    if entry.get("sha256") != hashlib.sha256(data).hexdigest():
        issues.append(JudgeIssue(item_id, f"{path}.sha256", "does not match file contents"))
    if entry.get("bytes") != len(data):
        issues.append(JudgeIssue(item_id, f"{path}.bytes", "does not match file size"))
    return resolved


def _validate_rater_manifest(
    issues: list[JudgeIssue],
    manifest: Any,
    *,
    annotation_mode: Any,
) -> None:
    if not isinstance(manifest, dict):
        issues.append(JudgeIssue("<judge_annotation_package>", "rater_manifest", "must be an object"))
        return
    minimum = manifest.get("minimum_raters_per_item")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        issues.append(
            JudgeIssue(
                "<judge_annotation_package>",
                "rater_manifest.minimum_raters_per_item",
                "must be an integer >= 2",
            )
        )
    raters = manifest.get("raters")
    if not isinstance(raters, list) or len(raters) < 2:
        issues.append(
            JudgeIssue(
                "<judge_annotation_package>",
                "rater_manifest.raters",
                "must contain at least two raters",
            )
        )
        return
    seen: set[str] = set()
    for index, rater in enumerate(raters):
        rater_path = f"rater_manifest.raters[{index}]"
        if not isinstance(rater, dict):
            issues.append(JudgeIssue("<judge_annotation_package>", rater_path, "must be an object"))
            continue
        rater_id = rater.get("id")
        if not _non_empty_string(rater_id):
            issues.append(
                JudgeIssue("<judge_annotation_package>", f"{rater_path}.id", "must be a non-empty string")
            )
            rater_id = rater_path
        elif rater_id in seen:
            issues.append(JudgeIssue(str(rater_id), f"{rater_path}.id", "duplicate rater id"))
        seen.add(str(rater_id))
        if rater.get("type") not in {"reference_fixture", "trained_human", "audited_model_judge"}:
            issues.append(
                JudgeIssue(
                    str(rater_id),
                    f"{rater_path}.type",
                    "must be reference_fixture, trained_human, or audited_model_judge",
                )
            )
        if rater.get("qualified") is not True:
            issues.append(JudgeIssue(str(rater_id), f"{rater_path}.qualified", "must be true"))
        if rater.get("conflict_attested") is not True:
            issues.append(JudgeIssue(str(rater_id), f"{rater_path}.conflict_attested", "must be true"))
        if annotation_mode != "reference_fixture" and rater.get("type") == "reference_fixture":
            issues.append(
                JudgeIssue(
                    str(rater_id),
                    f"{rater_path}.type",
                    "non-fixture annotation packages cannot use reference_fixture raters",
                )
            )


def _validate_annotation_package_entry(
    issues: list[JudgeIssue],
    entry: Any,
    *,
    index: int,
    seen_ids: set[str],
    rubric: dict[str, Any],
    base_dir: Path,
) -> None:
    path = f"packages[{index}]"
    if not isinstance(entry, dict):
        issues.append(JudgeIssue(path, path, "must be an object"))
        return
    package_id = entry.get("id")
    if not _non_empty_string(package_id):
        issues.append(JudgeIssue(path, f"{path}.id", "must be a non-empty string"))
        package_id = path
    elif package_id in seen_ids:
        issues.append(JudgeIssue(str(package_id), f"{path}.id", "duplicate package id"))
    seen_ids.add(str(package_id))
    if not _non_empty_string(entry.get("system_id")):
        issues.append(JudgeIssue(str(package_id), f"{path}.system_id", "must be a non-empty string"))

    source_report_path = _validate_package_file_entry(
        issues,
        entry.get("source_report"),
        f"{path}.source_report",
        str(package_id),
        base_dir,
    )
    annotations_path = _validate_package_file_entry(
        issues,
        entry.get("annotations"),
        f"{path}.annotations",
        str(package_id),
        base_dir,
    )
    judge_report_path = _validate_package_file_entry(
        issues,
        entry.get("judge_report"),
        f"{path}.judge_report",
        str(package_id),
        base_dir,
    )

    report = _load_json_if_present(issues, source_report_path, str(package_id), f"{path}.source_report")
    judge_report = _load_json_if_present(issues, judge_report_path, str(package_id), f"{path}.judge_report")
    annotations: list[dict[str, Any]] = []
    if annotations_path is not None:
        try:
            annotations = load_judge_annotations(annotations_path)
        except (json.JSONDecodeError, ValueError) as exc:
            issues.append(JudgeIssue(str(package_id), f"{path}.annotations", str(exc)))

    if isinstance(report, dict) and isinstance(rubric, dict) and annotations:
        issues.extend(
            JudgeIssue(str(package_id), f"{path}.{issue.path}", issue.message)
            for issue in validate_judge_annotations(annotations, rubric, report)
        )
    if isinstance(judge_report, dict):
        issues.extend(
            JudgeIssue(str(package_id), f"{path}.{issue.path}", issue.message)
            for issue in validate_judge_report(judge_report)
        )
        for field in ("num_annotations", "num_items", "num_raters"):
            _compare_declared_count(
                issues,
                entry,
                field,
                judge_report.get(field),
                path,
                str(package_id),
            )
    if annotations:
        _compare_declared_count(
            issues,
            entry,
            "num_annotations",
            len(annotations),
            path,
            str(package_id),
        )
        _compare_declared_count(
            issues,
            entry,
            "num_items",
            len({annotation.get("item_id") for annotation in annotations if isinstance(annotation, dict)}),
            path,
            str(package_id),
        )
        _compare_declared_count(
            issues,
            entry,
            "num_raters",
            len({annotation.get("rater_id") for annotation in annotations if isinstance(annotation, dict)}),
            path,
            str(package_id),
        )
    blinding = entry.get("blinding")
    if not isinstance(blinding, dict):
        issues.append(JudgeIssue(str(package_id), f"{path}.blinding", "must be an object"))
    else:
        for field in ("hide_system_identity", "shuffle_item_order", "hide_expected_actions"):
            if blinding.get(field) is not True:
                issues.append(JudgeIssue(str(package_id), f"{path}.blinding.{field}", "must be true"))
    adjudication = entry.get("adjudication")
    if not isinstance(adjudication, dict):
        issues.append(JudgeIssue(str(package_id), f"{path}.adjudication", "must be an object"))
    else:
        if "required_for_disagreement" not in adjudication:
            issues.append(
                JudgeIssue(
                    str(package_id),
                    f"{path}.adjudication.required_for_disagreement",
                    "missing required field",
                )
            )
        if not isinstance(adjudication.get("num_items_adjudicated"), int):
            issues.append(
                JudgeIssue(
                    str(package_id),
                    f"{path}.adjudication.num_items_adjudicated",
                    "must be an integer",
                )
            )


def _load_json_if_present(
    issues: list[JudgeIssue],
    path: Path | None,
    item_id: str,
    issue_path: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        issues.append(JudgeIssue(item_id, issue_path, f"invalid JSON: {exc.msg}"))
        return None
    return data if isinstance(data, dict) else None


def _compare_declared_count(
    issues: list[JudgeIssue],
    declared: dict[str, Any],
    field: str,
    actual: Any,
    path: str,
    item_id: str,
) -> None:
    if declared.get(field) != actual:
        issues.append(JudgeIssue(item_id, f"{path}.{field}", "does not match annotation evidence"))


def _object_field(
    issues: list[JudgeIssue],
    obj: dict[str, Any],
    field: str,
) -> dict[str, Any] | None:
    value = obj.get(field)
    if not isinstance(value, dict):
        issues.append(JudgeIssue("<judge_protocol>", field, "must be an object"))
        return None
    return value


def _require_true(
    issues: list[JudgeIssue],
    obj: dict[str, Any],
    field: str,
    path: str,
) -> None:
    if obj.get(field) is not True:
        issues.append(JudgeIssue("<judge_protocol>", path, "must be true"))


def _fraction_field(
    issues: list[JudgeIssue],
    obj: dict[str, Any],
    field: str,
    *,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> None:
    value = obj.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not min_value <= value <= max_value
    ):
        issues.append(
            JudgeIssue(
                "<judge_protocol>",
                f"quality_controls.{field}",
                f"must be a number between {min_value} and {max_value}",
            )
        )


def load_judge_annotations(path: str | Path) -> list[dict[str, Any]]:
    """Load judge annotations from JSONL or a JSON list/object wrapper."""
    path = Path(path)
    if path.suffix == ".jsonl":
        annotations = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line))
        return annotations
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("annotations"), list):
        return data["annotations"]
    raise ValueError(
        "judge annotations must be JSONL, a JSON list, or an object with annotations"
    )


def parse_model_judge_spec(value: str) -> ModelJudgeSpec:
    """Parse ``provider:model`` judge specs such as ``openrouter:anthropic/claude``."""
    if ":" not in value:
        raise ValueError("judge spec must use provider:model_id, for example openrouter:...")
    provider, model_id = value.split(":", 1)
    provider = _normalize_model_judge_provider(provider)
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("judge spec model_id must be non-empty")
    if provider not in _openai_compatible_model_judge_providers():
        allowed = ", ".join(sorted(_openai_compatible_model_judge_providers()))
        raise ValueError(f"model judging currently supports OpenAI-compatible providers: {allowed}")
    return ModelJudgeSpec(provider=provider, model_id=model_id)


def generate_model_judge_annotations(
    report: dict[str, Any],
    *,
    judge_specs: list[ModelJudgeSpec],
    rubric: dict[str, Any],
    prompt: str,
    adjudicator: ModelJudgeSpec | None = None,
    disagreement_threshold: float | None = None,
    caller: ModelJudgeCaller | None = None,
    max_output_tokens: int = 700,
    temperature: float = 0.0,
    timeout_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    """Generate model-judge annotations for every trial in a benchmark report."""
    if not judge_specs:
        raise ValueError("at least one judge spec is required")
    issues = validate_judge_rubric(rubric)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS judge rubric validation failed:\n{formatted}")

    threshold = (
        float(disagreement_threshold)
        if disagreement_threshold is not None
        else _rubric_disagreement_threshold(rubric)
    )
    call = caller or call_openai_compatible_model_judge
    annotations: list[dict[str, Any]] = []
    rater_ids = _model_judge_rater_ids(judge_specs)
    adjudicator_id = (
        _model_judge_rater_id(adjudicator, prefix="adjudicator")
        if adjudicator
        else None
    )

    for item in iter_blinded_judge_items(report):
        item_annotations = []
        for spec, rater_id in zip(judge_specs, rater_ids, strict=True):
            annotation = _score_blinded_item_with_model_judge(
                item,
                spec=spec,
                rater_id=rater_id,
                rubric=rubric,
                prompt=prompt,
                caller=call,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
            item_annotations.append(annotation)
        if (
            adjudicator is not None
            and len(item_annotations) >= 2
            and _requires_adjudication(item_annotations[:2], threshold=threshold)
        ):
            item_annotations.append(
                _score_blinded_item_with_model_judge(
                    item,
                    spec=adjudicator,
                    rater_id=adjudicator_id or "adjudicator",
                    rubric=rubric,
                    prompt=prompt,
                    caller=call,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    adjudication=True,
                )
            )
        annotations.extend(item_annotations)
    return annotations


def generate_model_judge_annotations_from_files(
    report_path: str | Path,
    *,
    judge_specs: list[ModelJudgeSpec],
    rubric_path: str | Path = DEFAULT_JUDGE_RUBRIC_PATH,
    prompt_path: str | Path = "data/openvoicecs/judging/judge_prompt_v0.1.md",
    adjudicator: ModelJudgeSpec | None = None,
    disagreement_threshold: float | None = None,
    caller: ModelJudgeCaller | None = None,
    max_output_tokens: int = 700,
    temperature: float = 0.0,
    timeout_seconds: float = 60.0,
    env_path: str | Path = ".env",
) -> list[dict[str, Any]]:
    """Load files and generate model-judge annotations."""
    load_workspace_env(env_path)
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    rubric = load_judge_rubric(rubric_path)
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    return generate_model_judge_annotations(
        report,
        judge_specs=judge_specs,
        rubric=rubric,
        prompt=prompt,
        adjudicator=adjudicator,
        disagreement_threshold=disagreement_threshold,
        caller=caller,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )


def iter_blinded_judge_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return trial-level items with model identity, oracle, and scoring data removed."""
    items: list[dict[str, Any]] = []
    for result in report.get("results", []):
        if not isinstance(result, dict):
            continue
        scenario_id = str(result.get("id") or "")
        if not scenario_id:
            continue
        trials = result.get("trials") if isinstance(result.get("trials"), list) else []
        for fallback_index, trial in enumerate(trials):
            if not isinstance(trial, dict):
                continue
            trial_index = _trial_index_from_report(trial, fallback_index)
            item_id = f"{scenario_id}:{trial_index}"
            items.append({
                "item_id": item_id,
                "scenario_id": scenario_id,
                "trial_index": trial_index,
                "domain": result.get("domain"),
                "track": result.get("track"),
                "difficulty": result.get("difficulty"),
                "customer_goal": result.get("customer_goal"),
                "input_modality": result.get("input_modality"),
                "audio_variant": _blinded_audio_variant(result.get("audio_variant")),
                "messages": _blinded_messages(trial.get("messages")),
            })
    return items


def write_judge_annotations_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write judge annotations as sorted-key JSONL."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def call_openai_compatible_model_judge(
    spec: ModelJudgeSpec,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> str:
    """Call an OpenAI-compatible chat-completions endpoint for model judging."""
    import httpx

    api_key = get_provider_api_key(spec.provider, spec.api_key)
    if not api_key:
        names = "/".join(PROVIDER_ENV_KEYS[spec.provider])
        raise ValueError(f"{names} is required for provider={spec.provider}")
    base_url = spec.base_url or OPENAI_COMPATIBLE_BASE_URLS.get(spec.provider)
    if not base_url and spec.provider == "openai":
        base_url = "https://api.openai.com/v1"
    if not base_url:
        raise ValueError(f"provider={spec.provider} does not have an OpenAI-compatible base URL")
    request: dict[str, Any] = {
        "model": spec.model_id,
        "messages": messages,
        "temperature": temperature,
    }
    if spec.provider == "openai":
        request["max_completion_tokens"] = max_output_tokens
    else:
        request["max_tokens"] = max_output_tokens
    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=request,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("model judge response did not include choices[0].message.content") from exc
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    return str(content)


def _score_blinded_item_with_model_judge(
    item: dict[str, Any],
    *,
    spec: ModelJudgeSpec,
    rater_id: str,
    rubric: dict[str, Any],
    prompt: str,
    caller: ModelJudgeCaller,
    max_output_tokens: int,
    temperature: float,
    timeout_seconds: float,
    adjudication: bool = False,
) -> dict[str, Any]:
    messages = _build_model_judge_messages(
        item,
        rubric=rubric,
        prompt=prompt,
        adjudication=adjudication,
    )
    response_text = caller(spec, messages, max_output_tokens, temperature, timeout_seconds)
    parsed = _parse_model_judge_response(response_text, rubric)
    annotation: dict[str, Any] = {
        "item_id": item["item_id"],
        "scenario_id": item["scenario_id"],
        "rater_id": rater_id,
        "scores": parsed["scores"],
        "judge": {
            "type": "audited_model_judge",
            "provider": spec.provider,
            "model_id": spec.model_id,
            "adjudicator": adjudication,
        },
    }
    if parsed.get("notes"):
        annotation["notes"] = parsed["notes"]
    return annotation


def _build_model_judge_messages(
    item: dict[str, Any],
    *,
    rubric: dict[str, Any],
    prompt: str,
    adjudication: bool,
) -> list[dict[str, str]]:
    dimensions = [dimension["id"] for dimension in rubric.get("dimensions", [])]
    scale = rubric.get("scale", {})
    system = (
        prompt.strip()
        + "\n\nReturn only JSON. The JSON object must have this shape:\n"
        + json.dumps(
            {
                "scores": {dimension: scale.get("max", 5) for dimension in dimensions},
                "notes": "short rationale",
            },
            sort_keys=True,
        )
        + "\nScores must be integers for every rubric dimension. Do not include "
        "markdown, extra keys, deterministic benchmark scores, or tool/oracle analysis."
    )
    if adjudication:
        system += (
            "\nYou are adjudicating a disagreement. Score independently from the "
            "customer-facing transcript and rubric only."
        )
    rubric_view = {
        "name": rubric.get("name"),
        "version": rubric.get("version"),
        "scale": rubric.get("scale"),
        "dimensions": [
            {
                "id": dimension.get("id"),
                "name": dimension.get("name"),
                "description": dimension.get("description"),
            }
            for dimension in rubric.get("dimensions", [])
        ],
    }
    user = {
        "blinded_item": item,
        "rubric": rubric_view,
        "instruction": (
            "Score only the agent's customer-facing behavior. The item is blinded: "
            "system identity, hidden oracle, pass/fail status, and tool checks are omitted."
        ),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True, sort_keys=True)},
    ]


def _parse_model_judge_response(text: str, rubric: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(_extract_json_object(text))
    if not isinstance(payload, dict):
        raise ValueError("model judge response JSON must be an object")
    dimensions = [dimension["id"] for dimension in rubric.get("dimensions", [])]
    source_scores = payload.get("scores")
    if not isinstance(source_scores, dict):
        source_scores = {
            dimension: payload.get(dimension)
            for dimension in dimensions
            if dimension in payload
        }
    scores: dict[str, int] = {}
    min_score = rubric.get("scale", {}).get("min", 1)
    max_score = rubric.get("scale", {}).get("max", 5)
    for dimension in dimensions:
        if dimension not in source_scores:
            raise ValueError(f"model judge response missing score for {dimension}")
        value = source_scores[dimension]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"model judge score for {dimension} must be numeric")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"model judge score for {dimension} must be an integer")
        int_value = int(value)
        if int_value < min_score or int_value > max_score:
            raise ValueError(
                f"model judge score for {dimension} must be between {min_score} and {max_score}"
            )
        scores[dimension] = int_value
    extra = set(source_scores) - set(dimensions)
    if extra:
        raise ValueError(f"model judge response included unknown dimensions: {sorted(extra)}")
    notes = payload.get("notes") or payload.get("rationale")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("model judge notes must be a string")
    return {"scores": scores, "notes": notes}


def _requires_adjudication(
    annotations: list[dict[str, Any]],
    *,
    threshold: float,
) -> bool:
    if len(annotations) < 2:
        return False
    first, second = annotations[0], annotations[1]
    first_scores = first.get("scores", {})
    second_scores = second.get("scores", {})
    for dimension, value in first_scores.items():
        other = second_scores.get(dimension)
        if isinstance(value, (int, float)) and isinstance(other, (int, float)):
            if abs(float(value) - float(other)) >= threshold:
                return True
    return False


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("model judge response did not contain a JSON object")
    candidate = text[start : end + 1]
    json.loads(candidate)
    return candidate


def _normalize_model_judge_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("_", "-")
    aliases = {
        "open-router": "openrouter",
        "dashscope": "alibaba",
        "aliyun": "alibaba",
        "alibaba-cloud": "alibaba",
        "moonshot": "kimi",
        "moonshotai": "kimi",
        "mini-max": "minimax",
        "grok": "xai",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PROVIDER_ENV_KEYS:
        raise ValueError(f"provider must be one of: {', '.join(sorted(PROVIDER_ENV_KEYS))}")
    return normalized


def _openai_compatible_model_judge_providers() -> set[str]:
    return set(OPENAI_COMPATIBLE_BASE_URLS) | {"openai"}


def _model_judge_rater_ids(specs: list[ModelJudgeSpec]) -> list[str]:
    counts: dict[str, int] = {}
    rater_ids = []
    for spec in specs:
        base = _model_judge_rater_id(spec)
        counts[base] = counts.get(base, 0) + 1
        rater_ids.append(base if counts[base] == 1 else f"{base}-{counts[base]}")
    return rater_ids


def _model_judge_rater_id(
    spec: ModelJudgeSpec | None,
    *,
    prefix: str = "model-judge",
) -> str:
    if spec is None:
        return prefix
    if spec.rater_id:
        return spec.rater_id
    value = f"{prefix}-{spec.provider}-{spec.model_id}"
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()
    return value or prefix


def _rubric_disagreement_threshold(rubric: dict[str, Any]) -> float:
    # The judge protocol uses two scale points for v0.1; the rubric alone does not
    # define an adjudication trigger, so keep that release default here.
    scale_min = rubric.get("scale", {}).get("min", 1)
    scale_max = rubric.get("scale", {}).get("max", 5)
    if isinstance(scale_min, int) and isinstance(scale_max, int) and scale_max <= 2:
        return 1.0
    return 2.0


def _trial_index_from_report(trial: dict[str, Any], fallback: int) -> int:
    value = trial.get("trial_index", fallback)
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    return value


def _blinded_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    blinded = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        text = message.get("text") or message.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        role = message.get("role") if isinstance(message.get("role"), str) else "agent"
        blinded.append({"role": role, "text": text})
    return blinded


def _blinded_audio_variant(variant: Any) -> dict[str, Any] | None:
    if not isinstance(variant, dict):
        return None
    return {
        "track": variant.get("track"),
        "transcript": variant.get("transcript"),
        "perturbations": variant.get("perturbations", []),
    }


def validate_judge_annotations(
    annotations: list[dict[str, Any]],
    rubric: dict[str, Any],
    report: dict[str, Any],
) -> list[JudgeIssue]:
    """Validate annotations against a rubric and report scenario IDs."""
    issues: list[JudgeIssue] = []
    dimensions = {dimension["id"] for dimension in rubric.get("dimensions", [])}
    min_score = rubric.get("scale", {}).get("min", 1)
    max_score = rubric.get("scale", {}).get("max", 5)
    scenario_ids = {
        result["id"]
        for result in report.get("results", [])
        if isinstance(result, dict) and result.get("id")
    }

    for index, annotation in enumerate(annotations):
        item_id = str(annotation.get("item_id") or f"<annotation-{index}>")
        if not isinstance(annotation, dict):
            issues.append(JudgeIssue(item_id, f"annotations[{index}]", "must be an object"))
            continue
        for field in ("item_id", "scenario_id", "rater_id", "scores"):
            if field not in annotation:
                issues.append(
                    JudgeIssue(
                        item_id,
                        f"annotations[{index}].{field}",
                        "missing required field",
                    )
                )
        if annotation.get("scenario_id") not in scenario_ids:
            issues.append(
                JudgeIssue(
                    item_id,
                    f"annotations[{index}].scenario_id",
                    "unknown scenario id",
                )
            )
        scores = annotation.get("scores")
        if not isinstance(scores, dict):
            issues.append(JudgeIssue(item_id, f"annotations[{index}].scores", "must be an object"))
            continue
        missing = dimensions - set(scores)
        for dimension_id in sorted(missing):
            issues.append(
                JudgeIssue(
                    item_id,
                    f"annotations[{index}].scores.{dimension_id}",
                    "missing score",
                )
            )
        for dimension_id, value in scores.items():
            if dimension_id not in dimensions:
                issues.append(
                    JudgeIssue(
                        item_id,
                        f"annotations[{index}].scores.{dimension_id}",
                        "unknown dimension",
                    )
                )
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                issues.append(
                    JudgeIssue(
                        item_id,
                        f"annotations[{index}].scores.{dimension_id}",
                        "must be numeric",
                    )
                )
            elif value < min_score or value > max_score:
                issues.append(
                    JudgeIssue(
                        item_id,
                        f"annotations[{index}].scores.{dimension_id}",
                        f"must be between {min_score} and {max_score}",
                    )
                )
    return issues


def build_judge_report(
    report: dict[str, Any],
    annotations: list[dict[str, Any]],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate subjective judge annotations for a benchmark report."""
    issues = validate_judge_rubric(rubric) + validate_judge_annotations(
        annotations,
        rubric,
        report,
    )
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS judge validation failed:\n{formatted}")

    scale_min = rubric["scale"]["min"]
    scale_max = rubric["scale"]["max"]
    dimension_weights = {
        dimension["id"]: float(dimension["weight"])
        for dimension in rubric["dimensions"]
    }
    annotations_by_item: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for annotation in annotations:
        key = (str(annotation["scenario_id"]), str(annotation["item_id"]))
        annotations_by_item.setdefault(key, []).append(annotation)

    item_scores = []
    for (scenario_id, item_id), item_annotations in sorted(annotations_by_item.items()):
        dimension_means = {
            dimension_id: statistics.mean([
                float(annotation["scores"][dimension_id])
                for annotation in item_annotations
            ])
            for dimension_id in dimension_weights
        }
        raw_score = sum(
            dimension_means[dimension_id] * weight
            for dimension_id, weight in dimension_weights.items()
        )
        normalized = (raw_score - scale_min) / (scale_max - scale_min)
        item_scores.append({
            "scenario_id": scenario_id,
            "item_id": item_id,
            "num_raters": len({annotation["rater_id"] for annotation in item_annotations}),
            "raw_score": round(raw_score, 6),
            "normalized_score": round(normalized, 6),
            "dimension_scores": {
                dimension_id: round(value, 6)
                for dimension_id, value in dimension_means.items()
            },
        })

    min_raters = int(rubric.get("minimum_raters_per_item", 1))
    agreement = _agreement_summary(annotations, rubric)
    return {
        "benchmark": "OpenVoiceCS-Bench Judge Report",
        "generated_at": time.strftime("%Y-%m-%d"),
        "source_benchmark": report.get("benchmark"),
        "source_benchmark_version": report.get("benchmark_version"),
        "model_metadata": report.get("model_metadata", {}),
        "rubric": {
            "name": rubric.get("name"),
            "version": rubric.get("version"),
            "dimension_weights": dimension_weights,
            "scale": rubric.get("scale"),
        },
        "num_annotations": len(annotations),
        "num_items": len(item_scores),
        "num_raters": len({annotation["rater_id"] for annotation in annotations}),
        "coverage": {
            "minimum_raters_per_item": min_raters,
            "items_meeting_minimum_raters": sum(
                1 for item in item_scores
                if item["num_raters"] >= min_raters
            ),
            "items_below_minimum_raters": [
                item["item_id"]
                for item in item_scores
                if item["num_raters"] < min_raters
            ],
        },
        "overall_subjective_score": round(
            statistics.mean([item["normalized_score"] for item in item_scores])
            if item_scores else 0.0,
            6,
        ),
        "dimension_scores": _dimension_report_scores(item_scores, dimension_weights),
        "agreement": agreement,
        "items": item_scores,
    }


def build_judge_report_from_files(
    report_path: str | Path,
    annotations_path: str | Path,
    *,
    rubric_path: str | Path = DEFAULT_JUDGE_RUBRIC_PATH,
) -> dict[str, Any]:
    """Load report, annotations, and rubric files and aggregate judge scores."""
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    rubric = load_judge_rubric(rubric_path)
    annotations = load_judge_annotations(annotations_path)
    return build_judge_report(report, annotations, rubric)


def validate_judge_report_file(path: str | Path) -> list[JudgeIssue]:
    """Validate a saved aggregated judge report."""
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    return validate_judge_report(report)


def validate_judge_report(report: dict[str, Any]) -> list[JudgeIssue]:
    """Validate aggregated judge-report structure and release-quality gates."""
    issues: list[JudgeIssue] = []
    if not isinstance(report, dict):
        return [JudgeIssue("<judge-report>", "<root>", "must be an object")]

    required = {
        "benchmark",
        "generated_at",
        "rubric",
        "num_annotations",
        "num_items",
        "num_raters",
        "coverage",
        "overall_subjective_score",
        "dimension_scores",
        "agreement",
        "items",
    }
    for field in sorted(required - set(report)):
        issues.append(JudgeIssue("<judge-report>", field, "missing required field"))
    if issues:
        return issues

    if report.get("benchmark") != "OpenVoiceCS-Bench Judge Report":
        issues.append(
            JudgeIssue(
                "<judge-report>",
                "benchmark",
                "must be OpenVoiceCS-Bench Judge Report",
            )
        )

    _validate_nonnegative_int(issues, report, "num_annotations")
    _validate_nonnegative_int(issues, report, "num_items")
    _validate_nonnegative_int(issues, report, "num_raters")
    _validate_unit_score(issues, report.get("overall_subjective_score"), "overall_subjective_score")
    _validate_judge_report_rubric(issues, report.get("rubric"))
    _validate_judge_report_coverage(issues, report.get("coverage"), report.get("num_items"))
    _validate_judge_report_agreement(issues, report.get("agreement"))
    _validate_judge_report_dimensions(issues, report)
    _validate_judge_report_items(issues, report)
    return issues


def apply_judge_report(
    report: dict[str, Any],
    judge_report: dict[str, Any],
) -> dict[str, Any]:
    """Attach an aggregated judge report to an OpenVoiceCS report.

    Item IDs may be scenario-level (``<scenario_id>``) or trial-specific via
    ``<scenario_id>:<trial_index>`` / ``<scenario_id>#trial_<trial_index>``.
    Scenario-level items are applied to every trial for that scenario.
    """
    updated = deepcopy(report)
    items = judge_report.get("items", [])
    items_by_key = {
        (str(item.get("scenario_id")), str(item.get("item_id"))): item
        for item in items
        if isinstance(item, dict)
    }
    assigned_scores = []
    assigned_count = 0
    total_trials = 0
    for result in updated.get("results", []):
        scenario_id = str(result.get("id"))
        for trial in result.get("trials", []):
            total_trials += 1
            trial_index = trial.get("trial_index", 0)
            item = _find_judge_item(items_by_key, scenario_id, trial_index)
            if not item:
                continue
            judgment = _experience_judgment_from_item(judge_report, item)
            trial["experience_judgment"] = judgment
            assigned_scores.append(judgment["score"])
            assigned_count += 1

    updated["conversation_experience_score"] = round(
        statistics.mean(assigned_scores),
        6,
    ) if assigned_scores else None
    updated["conversation_experience"] = {
        "score": updated["conversation_experience_score"],
        "coverage": round(assigned_count / total_trials, 4) if total_trials else 0.0,
        "num_judged_trials": assigned_count,
        "judge_counts": {
            _judge_key(judge_report): assigned_count,
        } if assigned_count else {},
        "source_judge_report": {
            "rubric": judge_report.get("rubric", {}),
            "num_annotations": judge_report.get("num_annotations"),
            "num_raters": judge_report.get("num_raters"),
            "agreement": judge_report.get("agreement"),
        },
    }
    return updated


def apply_judge_report_from_files(
    report_path: str | Path,
    judge_report_path: str | Path,
) -> dict[str, Any]:
    """Load a report and judge report, then attach judged experience."""
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    with open(judge_report_path, encoding="utf-8") as f:
        judge_report = json.load(f)
    return apply_judge_report(report, judge_report)


def _dimension_report_scores(
    item_scores: list[dict[str, Any]],
    dimension_weights: dict[str, float],
) -> dict[str, float]:
    scores = {}
    for dimension_id in dimension_weights:
        values = [
            item["dimension_scores"][dimension_id]
            for item in item_scores
        ]
        scores[dimension_id] = round(statistics.mean(values), 6) if values else 0.0
    return scores


def _validate_judge_report_rubric(
    issues: list[JudgeIssue],
    rubric: Any,
) -> None:
    if not isinstance(rubric, dict):
        issues.append(JudgeIssue("<judge-report>", "rubric", "must be an object"))
        return
    for field in ("name", "version", "dimension_weights", "scale"):
        if field not in rubric:
            issues.append(JudgeIssue("<judge-report>", f"rubric.{field}", "missing required field"))
    weights = rubric.get("dimension_weights")
    if not isinstance(weights, dict) or not weights:
        issues.append(
            JudgeIssue(
                "<judge-report>",
                "rubric.dimension_weights",
                "must be a non-empty object",
            )
        )
    elif abs(sum(_number(value) for value in weights.values()) - 1.0) > 1e-6:
        issues.append(
            JudgeIssue(
                "<judge-report>",
                "rubric.dimension_weights",
                "weights must sum to 1.0",
            )
        )
    scale = rubric.get("scale")
    if not isinstance(scale, dict):
        issues.append(JudgeIssue("<judge-report>", "rubric.scale", "must be an object"))
    elif not isinstance(scale.get("min"), int) or not isinstance(scale.get("max"), int):
        issues.append(JudgeIssue("<judge-report>", "rubric.scale", "min/max must be integers"))


def _validate_judge_report_coverage(
    issues: list[JudgeIssue],
    coverage: Any,
    num_items: Any,
) -> None:
    if not isinstance(coverage, dict):
        issues.append(JudgeIssue("<judge-report>", "coverage", "must be an object"))
        return
    for field in (
        "minimum_raters_per_item",
        "items_meeting_minimum_raters",
        "items_below_minimum_raters",
    ):
        if field not in coverage:
            issues.append(
                JudgeIssue(
                    "<judge-report>",
                    f"coverage.{field}",
                    "missing required field",
                )
            )
    _validate_nonnegative_int(issues, coverage, "minimum_raters_per_item", prefix="coverage")
    _validate_nonnegative_int(issues, coverage, "items_meeting_minimum_raters", prefix="coverage")
    below = coverage.get("items_below_minimum_raters")
    if not isinstance(below, list):
        issues.append(
            JudgeIssue(
                "<judge-report>",
                "coverage.items_below_minimum_raters",
                "must be a list",
            )
        )
    elif below:
        issues.append(
            JudgeIssue(
                "<judge-report>",
                "coverage.items_below_minimum_raters",
                "all judged items must meet minimum rater coverage for release",
            )
        )
    if (
        isinstance(num_items, int)
        and isinstance(coverage.get("items_meeting_minimum_raters"), int)
        and coverage["items_meeting_minimum_raters"] != num_items
    ):
        issues.append(
            JudgeIssue(
                "<judge-report>",
                "coverage.items_meeting_minimum_raters",
                "must equal num_items for release",
            )
        )


def _validate_judge_report_agreement(
    issues: list[JudgeIssue],
    agreement: Any,
) -> None:
    if not isinstance(agreement, dict):
        issues.append(JudgeIssue("<judge-report>", "agreement", "must be an object"))
        return
    for field in ("method", "overall", "by_dimension", "minimum_alpha_for_release"):
        if field not in agreement:
            issues.append(
                JudgeIssue(
                    "<judge-report>",
                    f"agreement.{field}",
                    "missing required field",
                )
            )
    overall = agreement.get("overall")
    minimum = agreement.get("minimum_alpha_for_release")
    if overall is None:
        issues.append(
            JudgeIssue(
                "<judge-report>",
                "agreement.overall",
                "must be available for release",
            )
        )
    elif isinstance(overall, bool) or not isinstance(overall, (int, float)):
        issues.append(JudgeIssue("<judge-report>", "agreement.overall", "must be numeric or null"))
    elif overall < -1 or overall > 1:
        issues.append(JudgeIssue("<judge-report>", "agreement.overall", "must be between -1 and 1"))
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        issues.append(
            JudgeIssue(
                "<judge-report>",
                "agreement.minimum_alpha_for_release",
                "must be numeric",
            )
        )
    elif isinstance(overall, (int, float)) and not isinstance(overall, bool) and overall < minimum:
        issues.append(
            JudgeIssue(
                "<judge-report>",
                "agreement.overall",
                "below minimum_alpha_for_release",
            )
        )
    by_dimension = agreement.get("by_dimension")
    if not isinstance(by_dimension, dict) or not by_dimension:
        issues.append(
            JudgeIssue(
                "<judge-report>",
                "agreement.by_dimension",
                "must be a non-empty object",
            )
        )
        return
    for dimension_id, value in by_dimension.items():
        if value is None:
            issues.append(
                JudgeIssue(
                    str(dimension_id),
                    "agreement.by_dimension",
                    "missing paired ratings",
                )
            )
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < -1
            or value > 1
        ):
            issues.append(
                JudgeIssue(
                    str(dimension_id),
                    "agreement.by_dimension",
                    "must be between -1 and 1",
                )
            )


def _validate_judge_report_dimensions(
    issues: list[JudgeIssue],
    report: dict[str, Any],
) -> None:
    dimensions = report.get("dimension_scores")
    weights = report.get("rubric", {}).get("dimension_weights", {})
    if not isinstance(dimensions, dict) or not dimensions:
        issues.append(
            JudgeIssue(
                "<judge-report>",
                "dimension_scores",
                "must be a non-empty object",
            )
        )
        return
    if isinstance(weights, dict):
        missing = set(weights) - set(dimensions)
        extra = set(dimensions) - set(weights)
        for dimension_id in sorted(missing):
            issues.append(JudgeIssue(str(dimension_id), "dimension_scores", "missing dimension"))
        for dimension_id in sorted(extra):
            issues.append(JudgeIssue(str(dimension_id), "dimension_scores", "unknown dimension"))
    scale = report.get("rubric", {}).get("scale", {})
    min_score = scale.get("min", 1) if isinstance(scale, dict) else 1
    max_score = scale.get("max", 5) if isinstance(scale, dict) else 5
    for dimension_id, value in dimensions.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            issues.append(JudgeIssue(str(dimension_id), "dimension_scores", "must be numeric"))
        elif value < min_score or value > max_score:
            issues.append(JudgeIssue(str(dimension_id), "dimension_scores", "outside rubric scale"))


def _validate_judge_report_items(
    issues: list[JudgeIssue],
    report: dict[str, Any],
) -> None:
    items = report.get("items")
    if not isinstance(items, list):
        issues.append(JudgeIssue("<judge-report>", "items", "must be a list"))
        return
    if isinstance(report.get("num_items"), int) and len(items) != report["num_items"]:
        issues.append(JudgeIssue("<judge-report>", "items", "length must equal num_items"))
    seen_items = set()
    weights = report.get("rubric", {}).get("dimension_weights", {})
    for index, item in enumerate(items):
        path = f"items[{index}]"
        if not isinstance(item, dict):
            issues.append(JudgeIssue(f"<item-{index}>", path, "must be an object"))
            continue
        item_id = str(item.get("item_id") or f"<item-{index}>")
        key = (item.get("scenario_id"), item.get("item_id"))
        if key in seen_items:
            issues.append(JudgeIssue(item_id, path, "duplicate judged item"))
        seen_items.add(key)
        for field in (
            "scenario_id",
            "item_id",
            "num_raters",
            "raw_score",
            "normalized_score",
            "dimension_scores",
        ):
            if field not in item:
                issues.append(JudgeIssue(item_id, f"{path}.{field}", "missing required field"))
        _validate_nonnegative_int(issues, item, "num_raters", prefix=path, item_id=item_id)
        _validate_unit_score(
            issues,
            item.get("normalized_score"),
            f"{path}.normalized_score",
            item_id=item_id,
        )
        dimensions = item.get("dimension_scores")
        if not isinstance(dimensions, dict):
            issues.append(JudgeIssue(item_id, f"{path}.dimension_scores", "must be an object"))
        elif isinstance(weights, dict):
            for dimension_id in sorted(set(weights) - set(dimensions)):
                issues.append(
                    JudgeIssue(
                        item_id,
                        f"{path}.dimension_scores.{dimension_id}",
                        "missing score",
                    )
                )


def _find_judge_item(
    items_by_key: dict[tuple[str, str], dict[str, Any]],
    scenario_id: str,
    trial_index: Any,
) -> dict[str, Any] | None:
    trial_index = str(trial_index)
    keys = (
        (scenario_id, f"{scenario_id}:{trial_index}"),
        (scenario_id, f"{scenario_id}#trial_{trial_index}"),
        (scenario_id, trial_index),
        (scenario_id, scenario_id),
    )
    for key in keys:
        if key in items_by_key:
            return items_by_key[key]
    return None


def _experience_judgment_from_item(
    judge_report: dict[str, Any],
    item: dict[str, Any],
) -> dict[str, Any]:
    rubric = judge_report.get("rubric", {})
    return {
        "score": item["normalized_score"],
        "judge": {
            "type": "offline_aggregate",
            "rubric_name": rubric.get("name"),
            "rubric_version": rubric.get("version"),
            "method": "human_or_model_annotations",
        },
        "dimensions": {
            dimension_id: {"score": score}
            for dimension_id, score in item.get("dimension_scores", {}).items()
        },
        "notes": {
            "scenario_id": item.get("scenario_id"),
            "item_id": item.get("item_id"),
            "num_raters": item.get("num_raters"),
            "raw_score": item.get("raw_score"),
        },
    }


def _judge_key(judge_report: dict[str, Any]) -> str:
    rubric = judge_report.get("rubric", {})
    name = rubric.get("name") or "judge"
    version = rubric.get("version") or "unknown"
    return f"{name}@{version}"


def _agreement_summary(
    annotations: list[dict[str, Any]],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    dimensions = [dimension["id"] for dimension in rubric.get("dimensions", [])]
    per_dimension = {
        dimension_id: _krippendorff_alpha_interval(annotations, dimension_id)
        for dimension_id in dimensions
    }
    values = [
        value for value in per_dimension.values()
        if value is not None
    ]
    return {
        "method": "krippendorff_alpha_interval",
        "overall": round(statistics.mean(values), 6) if values else None,
        "by_dimension": {
            key: round(value, 6) if value is not None else None
            for key, value in per_dimension.items()
        },
        "minimum_alpha_for_release": rubric.get("agreement", {}).get("minimum_alpha_for_release"),
    }


def _krippendorff_alpha_interval(
    annotations: list[dict[str, Any]],
    dimension_id: str,
) -> float | None:
    values_by_item: dict[str, list[float]] = {}
    for annotation in annotations:
        if dimension_id in annotation.get("scores", {}):
            values_by_item.setdefault(str(annotation["item_id"]), []).append(
                float(annotation["scores"][dimension_id])
            )
    paired_items = {
        item_id: values
        for item_id, values in values_by_item.items()
        if len(values) >= 2
    }
    if not paired_items:
        return None

    observed_numer = 0.0
    observed_denom = 0
    all_values = []
    for values in paired_items.values():
        all_values.extend(values)
        for i, left in enumerate(values):
            for right in values[i + 1:]:
                observed_numer += (left - right) ** 2
                observed_denom += 1
    if observed_denom == 0 or len(all_values) < 2:
        return None
    observed = observed_numer / observed_denom

    expected_numer = 0.0
    expected_denom = 0
    for i, left in enumerate(all_values):
        for right in all_values[i + 1:]:
            expected_numer += (left - right) ** 2
            expected_denom += 1
    if expected_denom == 0:
        return None
    expected = expected_numer / expected_denom
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return max(-1.0, min(1.0, 1.0 - (observed / expected)))


def _validate_nonnegative_int(
    issues: list[JudgeIssue],
    data: dict[str, Any],
    field: str,
    *,
    prefix: str | None = None,
    item_id: str = "<judge-report>",
) -> None:
    value = data.get(field)
    path = f"{prefix}.{field}" if prefix else field
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        issues.append(JudgeIssue(item_id, path, "must be a nonnegative integer"))


def _validate_unit_score(
    issues: list[JudgeIssue],
    value: Any,
    path: str,
    *,
    item_id: str = "<judge-report>",
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(JudgeIssue(item_id, path, "must be numeric"))
    elif value < 0 or value > 1:
        issues.append(JudgeIssue(item_id, path, "must be between 0 and 1"))


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
