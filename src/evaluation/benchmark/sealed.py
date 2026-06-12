"""Sealed-test operations validation for OpenVoiceCS releases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.splits import (
    DEFAULT_SPLIT_COMMITMENT_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
)


DEFAULT_SEALED_OPS_PATH = Path("data/openvoicecs/sealed_ops_v0.1.json")
DEFAULT_SEALED_QUEUE_PATH = Path("data/openvoicecs/sealed_evaluator_queue_v0.1.json")
QUEUE_STATUSES = {"reference_fixture", "queued", "running", "completed", "rejected", "withdrawn", "retired"}
ATTEMPT_STATUSES = {"not_started", "running", "completed", "failed", "cancelled"}


@dataclass(frozen=True)
class SealedIssue:
    """Structured sealed-operations validation issue."""

    item_id: str
    path: str
    message: str


@dataclass(frozen=True)
class SealedQueueIssue:
    """Structured sealed-evaluator queue validation issue."""

    item_id: str
    path: str
    message: str


def load_sealed_ops_manifest(
    path: str | Path = DEFAULT_SEALED_OPS_PATH,
) -> dict[str, Any]:
    """Load and validate the sealed-test operations manifest."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    issues = validate_sealed_ops_manifest(manifest)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS sealed ops validation failed:\n{formatted}")
    return manifest


def load_sealed_queue_manifest(
    path: str | Path = DEFAULT_SEALED_QUEUE_PATH,
) -> dict[str, Any]:
    """Load and validate the sealed-evaluator queue manifest."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    issues = validate_sealed_queue_manifest_file(path)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS sealed queue validation failed:\n{formatted}")
    return manifest


def validate_sealed_ops_manifest_file(
    path: str | Path = DEFAULT_SEALED_OPS_PATH,
    *,
    split_manifest_path: str | Path | None = DEFAULT_SPLIT_MANIFEST_PATH,
    split_commitment_path: str | Path | None = DEFAULT_SPLIT_COMMITMENT_PATH,
) -> list[SealedIssue]:
    """Validate a saved sealed-test operations manifest."""
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    split_manifest = _load_json(split_manifest_path) if split_manifest_path else None
    split_commitments = _load_json(split_commitment_path) if split_commitment_path else None
    return validate_sealed_ops_manifest(
        manifest,
        split_manifest=split_manifest,
        split_commitments=split_commitments,
    )


def validate_sealed_queue_manifest_file(
    path: str | Path = DEFAULT_SEALED_QUEUE_PATH,
    *,
    base_dir: str | Path = ".",
    sealed_ops_path: str | Path | None = DEFAULT_SEALED_OPS_PATH,
    split_commitment_path: str | Path | None = DEFAULT_SPLIT_COMMITMENT_PATH,
) -> list[SealedQueueIssue]:
    """Validate a saved sealed-evaluator queue manifest."""
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    sealed_ops = _load_json(sealed_ops_path) if sealed_ops_path else None
    split_commitments = _load_json(split_commitment_path) if split_commitment_path else None
    return validate_sealed_queue_manifest(
        manifest,
        base_dir=base_dir,
        sealed_ops=sealed_ops,
        split_commitments=split_commitments,
    )


def validate_sealed_ops_manifest(
    manifest: dict[str, Any],
    *,
    split_manifest: dict[str, Any] | None = None,
    split_commitments: dict[str, Any] | None = None,
) -> list[SealedIssue]:
    """Return all sealed-test operations contract issues."""
    issues: list[SealedIssue] = []
    if not isinstance(manifest, dict):
        return [SealedIssue("<sealed_ops>", "<root>", "must be an object")]

    required = (
        "name",
        "version",
        "benchmark_version",
        "split_manifest",
        "split_commitments",
        "custody",
        "access_policy",
        "evaluation_protocol",
        "disclosure_policy",
    )
    for field in required:
        if field not in manifest:
            issues.append(SealedIssue("<sealed_ops>", field, "missing required field"))
    if issues:
        return issues

    for field in ("name", "version", "benchmark_version", "split_manifest", "split_commitments"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            issues.append(SealedIssue("<sealed_ops>", field, "must be a non-empty string"))

    custody = _object_field(issues, manifest, "custody")
    if custody is not None:
        _require_false(issues, custody, "sealed_ids_revealed", "custody.sealed_ids_revealed")
        _require_true(issues, custody, "commitment_hash_published", "custody.commitment_hash_published")
        _require_true(issues, custody, "access_log_required", "custody.access_log_required")
        _require_non_empty_string(issues, custody, "storage", "custody.storage")

    access_policy = _object_field(issues, manifest, "access_policy")
    if access_policy is not None:
        roles = access_policy.get("allowed_roles")
        if not isinstance(roles, list) or not roles:
            issues.append(SealedIssue("<sealed_ops>", "access_policy.allowed_roles", "must be a non-empty list"))
        elif any(not isinstance(role, str) or not role.strip() for role in roles):
            issues.append(
                SealedIssue(
                    "<sealed_ops>",
                    "access_policy.allowed_roles",
                    "all entries must be non-empty strings",
                )
            )
        _require_false(issues, access_policy, "pre_submission_access", "access_policy.pre_submission_access")
        _require_true(issues, access_policy, "conflict_attestation_required", "access_policy.conflict_attestation_required")
        _require_non_empty_string(
            issues,
            access_policy,
            "reviewer_conflict_policy",
            "access_policy.reviewer_conflict_policy",
        )

    evaluation_protocol = _object_field(issues, manifest, "evaluation_protocol")
    if evaluation_protocol is not None:
        _require_non_empty_string(
            issues,
            evaluation_protocol,
            "submission_queue",
            "evaluation_protocol.submission_queue",
        )
        attempts = evaluation_protocol.get("max_attempts_per_system")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            issues.append(
                SealedIssue(
                    "<sealed_ops>",
                    "evaluation_protocol.max_attempts_per_system",
                    "must be an integer >= 1",
                )
            )
        _require_true(issues, evaluation_protocol, "audit_log_required", "evaluation_protocol.audit_log_required")
        _require_true(
            issues,
            evaluation_protocol,
            "fixed_evaluation_environment",
            "evaluation_protocol.fixed_evaluation_environment",
        )
        _require_non_empty_string(
            issues,
            evaluation_protocol,
            "result_release_delay",
            "evaluation_protocol.result_release_delay",
        )

    disclosure_policy = _object_field(issues, manifest, "disclosure_policy")
    if disclosure_policy is not None:
        _require_false(
            issues,
            disclosure_policy,
            "publish_prompts_before_evaluation",
            "disclosure_policy.publish_prompts_before_evaluation",
        )
        _require_true(
            issues,
            disclosure_policy,
            "publish_aggregate_counts",
            "disclosure_policy.publish_aggregate_counts",
        )
        _require_true(
            issues,
            disclosure_policy,
            "publish_post_eval_errata",
            "disclosure_policy.publish_post_eval_errata",
        )

    _validate_split_evidence(issues, split_manifest, split_commitments)
    return issues


def validate_sealed_queue_manifest(
    manifest: dict[str, Any],
    *,
    base_dir: str | Path = ".",
    sealed_ops: dict[str, Any] | None = None,
    split_commitments: dict[str, Any] | None = None,
) -> list[SealedQueueIssue]:
    """Return all sealed-evaluator queue contract issues."""
    issues: list[SealedQueueIssue] = []
    if not isinstance(manifest, dict):
        return [SealedQueueIssue("<sealed_queue>", "<root>", "must be an object")]
    required = (
        "name",
        "version",
        "benchmark_version",
        "sealed_ops",
        "split_commitments",
        "queue_policy",
        "submissions",
        "audit_log",
    )
    for field in required:
        if field not in manifest:
            issues.append(SealedQueueIssue("<sealed_queue>", field, "missing required field"))
    if issues:
        return issues
    if manifest.get("name") != "OpenVoiceCS Sealed Evaluator Queue":
        issues.append(
            SealedQueueIssue(
                "<sealed_queue>",
                "name",
                "must be OpenVoiceCS Sealed Evaluator Queue",
            )
        )
    for field in ("version", "benchmark_version"):
        if not _non_empty_string(manifest.get(field)):
            issues.append(SealedQueueIssue("<sealed_queue>", field, "must be a non-empty string"))

    base_path = Path(base_dir)
    _load_queue_file_entry(
        issues,
        manifest.get("sealed_ops"),
        "sealed_ops",
        "<sealed_queue>",
        base_path,
    )
    _load_queue_file_entry(
        issues,
        manifest.get("split_commitments"),
        "split_commitments",
        "<sealed_queue>",
        base_path,
    )
    _validate_queue_policy(issues, manifest.get("queue_policy"), sealed_ops=sealed_ops)
    _validate_queue_audit_log(issues, manifest.get("audit_log"))

    submissions = manifest.get("submissions")
    if not isinstance(submissions, list):
        issues.append(SealedQueueIssue("<sealed_queue>", "submissions", "must be a list"))
    else:
        seen_queue_ids: set[str] = set()
        seen_submission_ids: set[str] = set()
        for index, submission in enumerate(submissions):
            _validate_queue_submission(
                issues,
                submission,
                index=index,
                seen_queue_ids=seen_queue_ids,
                seen_submission_ids=seen_submission_ids,
                base_dir=base_path,
                sealed_ops=sealed_ops,
            )
    _validate_queue_split_privacy(issues, split_commitments)
    return issues


def sealed_ops_stats(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return compact stats for a sealed-test operations manifest."""
    return {
        "version": manifest.get("version"),
        "benchmark_version": manifest.get("benchmark_version"),
        "storage": manifest.get("custody", {}).get("storage"),
        "max_attempts_per_system": manifest.get("evaluation_protocol", {}).get(
            "max_attempts_per_system"
        ),
        "pre_submission_access": manifest.get("access_policy", {}).get(
            "pre_submission_access"
        ),
    }


def sealed_queue_stats(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Return compact stats for a sealed-evaluator queue manifest."""
    if not isinstance(manifest, dict):
        return {"present": False, "num_submissions": 0}
    submissions = manifest.get("submissions", [])
    submissions = submissions if isinstance(submissions, list) else []
    by_status: dict[str, int] = {}
    num_attempts = 0
    official_candidates = 0
    for submission in submissions:
        if not isinstance(submission, dict):
            continue
        status = str(submission.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        attempts = submission.get("attempts", [])
        if isinstance(attempts, list):
            num_attempts += len(attempts)
        if submission.get("official_candidate") is True:
            official_candidates += 1
    return {
        "present": True,
        "version": manifest.get("version"),
        "benchmark_version": manifest.get("benchmark_version"),
        "num_submissions": len(submissions),
        "num_attempts": num_attempts,
        "by_status": by_status,
        "official_candidates": official_candidates,
        "reference_fixtures": by_status.get("reference_fixture", 0),
    }


def _validate_queue_policy(
    issues: list[SealedQueueIssue],
    policy: Any,
    *,
    sealed_ops: dict[str, Any] | None,
) -> None:
    if not isinstance(policy, dict):
        issues.append(SealedQueueIssue("<sealed_queue>", "queue_policy", "must be an object"))
        return
    for field in (
        "sealed_ids_revealed",
        "raw_prompts_exported_to_submitter",
        "pre_submission_access_allowed",
    ):
        if policy.get(field) is not False:
            issues.append(SealedQueueIssue("<sealed_queue>", f"queue_policy.{field}", "must be false"))
    for field in (
        "append_only_audit_log",
        "conflict_attestation_required",
        "fixed_evaluation_environment",
    ):
        if policy.get(field) is not True:
            issues.append(SealedQueueIssue("<sealed_queue>", f"queue_policy.{field}", "must be true"))
    attempts = policy.get("max_attempts_per_system")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        issues.append(
            SealedQueueIssue(
                "<sealed_queue>",
                "queue_policy.max_attempts_per_system",
                "must be an integer >= 1",
            )
        )
    ops_attempts = None
    if isinstance(sealed_ops, dict):
        ops_attempts = sealed_ops.get("evaluation_protocol", {}).get("max_attempts_per_system")
    if isinstance(attempts, int) and isinstance(ops_attempts, int) and attempts > ops_attempts:
        issues.append(
            SealedQueueIssue(
                "<sealed_queue>",
                "queue_policy.max_attempts_per_system",
                "must not exceed sealed ops policy",
            )
        )
    ordering = policy.get("ordering")
    if ordering not in {"first_in_first_out", "submission_window_batch"}:
        issues.append(
            SealedQueueIssue(
                "<sealed_queue>",
                "queue_policy.ordering",
                "must be first_in_first_out or submission_window_batch",
            )
        )


def _validate_queue_audit_log(
    issues: list[SealedQueueIssue],
    audit_log: Any,
) -> None:
    if not isinstance(audit_log, dict):
        issues.append(SealedQueueIssue("<sealed_queue>", "audit_log", "must be an object"))
        return
    if audit_log.get("hash_algorithm") != "sha256":
        issues.append(SealedQueueIssue("<sealed_queue>", "audit_log.hash_algorithm", "must be sha256"))
    if audit_log.get("append_only") is not True:
        issues.append(SealedQueueIssue("<sealed_queue>", "audit_log.append_only", "must be true"))
    for field in ("retention", "public_disclosure"):
        if not _non_empty_string(audit_log.get(field)):
            issues.append(SealedQueueIssue("<sealed_queue>", f"audit_log.{field}", "must be a non-empty string"))


def _validate_queue_submission(
    issues: list[SealedQueueIssue],
    submission: Any,
    *,
    index: int,
    seen_queue_ids: set[str],
    seen_submission_ids: set[str],
    base_dir: Path,
    sealed_ops: dict[str, Any] | None,
) -> None:
    path = f"submissions[{index}]"
    if not isinstance(submission, dict):
        issues.append(SealedQueueIssue(path, path, "must be an object"))
        return
    queue_id = str(submission.get("queue_id") or path)
    for field in ("queue_id", "submission_id", "system_id", "status", "submitted_at", "intake", "attempt_limit", "attempts"):
        if field not in submission:
            issues.append(SealedQueueIssue(queue_id, f"{path}.{field}", "missing required field"))
    for field in ("queue_id", "submission_id", "system_id", "submitted_at"):
        if field in submission and not _non_empty_string(submission.get(field)):
            issues.append(SealedQueueIssue(queue_id, f"{path}.{field}", "must be a non-empty string"))
    if queue_id in seen_queue_ids:
        issues.append(SealedQueueIssue(queue_id, f"{path}.queue_id", "duplicate queue id"))
    seen_queue_ids.add(queue_id)
    submission_id = submission.get("submission_id")
    if isinstance(submission_id, str):
        if submission_id in seen_submission_ids:
            issues.append(SealedQueueIssue(queue_id, f"{path}.submission_id", "duplicate submission id"))
        seen_submission_ids.add(submission_id)
    if submission.get("status") not in QUEUE_STATUSES:
        issues.append(
            SealedQueueIssue(
                queue_id,
                f"{path}.status",
                f"must be one of: {', '.join(sorted(QUEUE_STATUSES))}",
            )
        )
    if not isinstance(submission.get("official_candidate"), bool):
        issues.append(SealedQueueIssue(queue_id, f"{path}.official_candidate", "must be boolean"))
    if submission.get("status") == "reference_fixture" and submission.get("official_candidate") is not False:
        issues.append(
            SealedQueueIssue(
                queue_id,
                f"{path}.official_candidate",
                "reference fixtures cannot be official candidates",
            )
        )
    attempt_limit = submission.get("attempt_limit")
    if isinstance(attempt_limit, bool) or not isinstance(attempt_limit, int) or attempt_limit < 1:
        issues.append(SealedQueueIssue(queue_id, f"{path}.attempt_limit", "must be an integer >= 1"))
        attempt_limit = None
    ops_attempts = None
    if isinstance(sealed_ops, dict):
        ops_attempts = sealed_ops.get("evaluation_protocol", {}).get("max_attempts_per_system")
    if isinstance(attempt_limit, int) and isinstance(ops_attempts, int) and attempt_limit > ops_attempts:
        issues.append(SealedQueueIssue(queue_id, f"{path}.attempt_limit", "must not exceed sealed ops policy"))

    intake = _load_queue_file_entry(issues, submission.get("intake"), f"{path}.intake", queue_id, base_dir)
    if intake is not None:
        _validate_queue_intake_link(issues, queue_id, path, submission, intake)
    attempts = submission.get("attempts")
    if not isinstance(attempts, list):
        issues.append(SealedQueueIssue(queue_id, f"{path}.attempts", "must be a list"))
    else:
        if isinstance(attempt_limit, int) and len(attempts) > attempt_limit:
            issues.append(SealedQueueIssue(queue_id, f"{path}.attempts", "must not exceed attempt_limit"))
        seen_attempt_ids: set[str] = set()
        for attempt_index, attempt in enumerate(attempts):
            _validate_queue_attempt(
                issues,
                attempt,
                path=f"{path}.attempts[{attempt_index}]",
                queue_id=queue_id,
                seen_attempt_ids=seen_attempt_ids,
                base_dir=base_dir,
            )


def _validate_queue_intake_link(
    issues: list[SealedQueueIssue],
    queue_id: str,
    path: str,
    submission: dict[str, Any],
    intake: dict[str, Any],
) -> None:
    from src.evaluation.benchmark.submission import validate_submission_intake

    issues.extend(
        SealedQueueIssue(queue_id, f"{path}.intake.{issue.path}", issue.message)
        for issue in validate_submission_intake(intake)
    )
    if intake.get("submission_id") != submission.get("submission_id"):
        issues.append(SealedQueueIssue(queue_id, f"{path}.submission_id", "must match intake submission_id"))
    if intake.get("system_id") != submission.get("system_id"):
        issues.append(SealedQueueIssue(queue_id, f"{path}.system_id", "must match intake system_id"))
    if intake.get("status") == "reference_fixture" and submission.get("official_candidate") is True:
        issues.append(
            SealedQueueIssue(
                queue_id,
                f"{path}.official_candidate",
                "reference fixture intake cannot be an official candidate",
            )
        )


def _validate_queue_attempt(
    issues: list[SealedQueueIssue],
    attempt: Any,
    *,
    path: str,
    queue_id: str,
    seen_attempt_ids: set[str],
    base_dir: Path,
) -> None:
    if not isinstance(attempt, dict):
        issues.append(SealedQueueIssue(queue_id, path, "must be an object"))
        return
    attempt_id = attempt.get("attempt_id")
    if not _non_empty_string(attempt_id):
        issues.append(SealedQueueIssue(queue_id, f"{path}.attempt_id", "must be a non-empty string"))
        attempt_id = path
    elif attempt_id in seen_attempt_ids:
        issues.append(SealedQueueIssue(queue_id, f"{path}.attempt_id", "duplicate attempt id"))
    seen_attempt_ids.add(str(attempt_id))
    index = attempt.get("attempt_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        issues.append(SealedQueueIssue(queue_id, f"{path}.attempt_index", "must be an integer >= 1"))
    if attempt.get("status") not in ATTEMPT_STATUSES:
        issues.append(
            SealedQueueIssue(
                queue_id,
                f"{path}.status",
                f"must be one of: {', '.join(sorted(ATTEMPT_STATUSES))}",
            )
        )
    for field in ("started_at", "completed_at"):
        if attempt.get(field) is not None and not isinstance(attempt.get(field), str):
            issues.append(SealedQueueIssue(queue_id, f"{path}.{field}", "must be a string or null"))
    exposure = attempt.get("sealed_split_exposure")
    if not isinstance(exposure, dict):
        issues.append(SealedQueueIssue(queue_id, f"{path}.sealed_split_exposure", "must be an object"))
    else:
        for field in ("sealed_ids_revealed", "raw_prompts_exported", "expected_state_exported"):
            if exposure.get(field) is not False:
                issues.append(SealedQueueIssue(queue_id, f"{path}.sealed_split_exposure.{field}", "must be false"))
        if exposure.get("served_by") != "hosted_evaluator":
            issues.append(
                SealedQueueIssue(
                    queue_id,
                    f"{path}.sealed_split_exposure.served_by",
                    "must be hosted_evaluator",
                )
            )
    environment = attempt.get("environment")
    if not isinstance(environment, dict):
        issues.append(SealedQueueIssue(queue_id, f"{path}.environment", "must be an object"))
    else:
        for field in ("region", "network", "hardware_profile", "transport"):
            if not _non_empty_string(environment.get(field)):
                issues.append(SealedQueueIssue(queue_id, f"{path}.environment.{field}", "must be a non-empty string"))
    artifacts = attempt.get("artifacts", {})
    if artifacts is not None and not isinstance(artifacts, dict):
        issues.append(SealedQueueIssue(queue_id, f"{path}.artifacts", "must be an object"))
        return
    for artifact_name, entry in (artifacts or {}).items():
        if artifact_name not in {"report", "run_manifest", "release_bundle", "judge_annotation_package", "leaderboard_claims"}:
            issues.append(SealedQueueIssue(queue_id, f"{path}.artifacts.{artifact_name}", "unknown artifact key"))
            continue
        _load_queue_file_entry(
            issues,
            entry,
            f"{path}.artifacts.{artifact_name}",
            queue_id,
            base_dir,
            parse_json=False,
        )


def _validate_queue_split_privacy(
    issues: list[SealedQueueIssue],
    split_commitments: dict[str, Any] | None,
) -> None:
    if not isinstance(split_commitments, dict):
        return
    privacy = split_commitments.get("privacy", {})
    if privacy.get("sealed_test_ids_revealed") is not False:
        issues.append(
            SealedQueueIssue(
                "sealed_test",
                "split_commitments.privacy.sealed_test_ids_revealed",
                "must be false",
            )
        )
    root_hash = split_commitments.get("root_hash")
    if not _looks_sha256(root_hash):
        issues.append(SealedQueueIssue("sealed_test", "split_commitments.root_hash", "must publish a sha256 root hash"))


def _load_queue_file_entry(
    issues: list[SealedQueueIssue],
    entry: Any,
    path: str,
    item_id: str,
    base_dir: Path,
    *,
    parse_json: bool = True,
) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        issues.append(SealedQueueIssue(item_id, path, "must be an object"))
        return None
    for field in ("path", "sha256", "bytes"):
        if field not in entry:
            issues.append(SealedQueueIssue(item_id, f"{path}.{field}", "missing required field"))
    raw_path = entry.get("path")
    if not _non_empty_string(raw_path):
        issues.append(SealedQueueIssue(item_id, f"{path}.path", "must be a non-empty string"))
        return None
    resolved = Path(str(raw_path))
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    if not resolved.exists():
        issues.append(SealedQueueIssue(item_id, f"{path}.path", "file does not exist"))
        return None
    data = resolved.read_bytes()
    if entry.get("sha256") != hashlib.sha256(data).hexdigest():
        issues.append(SealedQueueIssue(item_id, f"{path}.sha256", "does not match file contents"))
    if entry.get("bytes") != len(data):
        issues.append(SealedQueueIssue(item_id, f"{path}.bytes", "does not match file size"))
    if not parse_json:
        return None
    try:
        loaded = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(SealedQueueIssue(item_id, path, f"invalid JSON: {exc.msg}"))
        return None
    return loaded if isinstance(loaded, dict) else None


def _validate_split_evidence(
    issues: list[SealedIssue],
    split_manifest: dict[str, Any] | None,
    split_commitments: dict[str, Any] | None,
) -> None:
    if split_manifest is not None:
        sealed = split_manifest.get("splits", {}).get("sealed_test")
        if not isinstance(sealed, dict):
            issues.append(SealedIssue("sealed_test", "split_manifest.splits.sealed_test", "missing sealed split"))
        elif not sealed.get("scenario_ids") and not sealed.get("audio_variant_ids"):
            issues.append(
                SealedIssue(
                    "sealed_test",
                    "split_manifest.splits.sealed_test",
                    "sealed split must contain scenarios or audio variants",
                )
            )
    if split_commitments is not None:
        privacy = split_commitments.get("privacy", {})
        if privacy.get("sealed_test_ids_revealed") is not False:
            issues.append(
                SealedIssue(
                    "sealed_test",
                    "split_commitments.privacy.sealed_test_ids_revealed",
                    "must be false",
                )
            )
        root_hash = split_commitments.get("root_hash")
        if not isinstance(root_hash, str) or len(root_hash) != 64:
            issues.append(
                SealedIssue(
                    "sealed_test",
                    "split_commitments.root_hash",
                    "must publish a sha256 root hash",
                )
            )


def _object_field(
    issues: list[SealedIssue],
    obj: dict[str, Any],
    field: str,
) -> dict[str, Any] | None:
    value = obj.get(field)
    if not isinstance(value, dict):
        issues.append(SealedIssue("<sealed_ops>", field, "must be an object"))
        return None
    return value


def _require_true(
    issues: list[SealedIssue],
    obj: dict[str, Any],
    field: str,
    path: str,
) -> None:
    if obj.get(field) is not True:
        issues.append(SealedIssue("<sealed_ops>", path, "must be true"))


def _require_false(
    issues: list[SealedIssue],
    obj: dict[str, Any],
    field: str,
    path: str,
) -> None:
    if obj.get(field) is not False:
        issues.append(SealedIssue("<sealed_ops>", path, "must be false"))


def _require_non_empty_string(
    issues: list[SealedIssue],
    obj: dict[str, Any],
    field: str,
    path: str,
) -> None:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        issues.append(SealedIssue("<sealed_ops>", path, "must be a non-empty string"))


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _looks_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}
