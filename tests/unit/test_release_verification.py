"""Tests for OpenVoiceCS release verification."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.benchmark.release_verification import (
    verify_openvoicecs_release,
)


def test_verify_openvoicecs_seed_release_passes():
    verification = verify_openvoicecs_release()

    assert verification["passed"] is True
    assert verification["num_issues"] == 0
    assert verification["release_gates"]["passed"] is True
    check_names = {check["name"] for check in verification["checks"]}
    assert {
        "scenario_suite",
        "audio_manifest",
        "pricing_manifest",
        "split_manifest",
        "split_commitments",
        "provenance_manifest",
        "changelog",
        "reference_baselines",
        "scenario_reviews",
        "datasheet",
        "judge_protocol",
        "judge_study",
        "judge_annotation_package",
        "sealed_ops",
        "sealed_queue",
        "external_systems",
        "leaderboard_claims",
        "submission_intake",
        "release_audit",
        "release_gates",
        "saved_release_audit",
        "readiness:seed",
    }.issubset(check_names)
    assert verification["readiness_profile"] == "seed"


def test_verify_openvoicecs_release_detects_stale_saved_audit(tmp_path: Path):
    stale_audit = tmp_path / "release_audit.json"
    with open("data/openvoicecs/release_audit.json", encoding="utf-8") as f:
        audit = json.load(f)
    audit["release_gates"]["has_reference_baselines"] = False
    stale_audit.write_text(json.dumps(audit), encoding="utf-8")

    verification = verify_openvoicecs_release(release_audit_path=stale_audit)

    messages = {
        (issue["check"], issue["path"], issue["message"])
        for issue in verification["issues"]
    }
    assert verification["passed"] is False
    assert (
        "saved_release_audit",
        "release_gates",
        "does not match freshly computed release audit",
    ) in messages


def test_verify_openvoicecs_release_can_require_audio_assets():
    verification = verify_openvoicecs_release(require_audio_assets=True)

    check = next(item for item in verification["checks"] if item["name"] == "audio_assets")

    assert verification["passed"] is True
    assert check["passed"] is True
    assert check["num_issues"] == 0


def test_verify_openvoicecs_release_reports_public_readiness_gaps():
    verification = verify_openvoicecs_release(readiness_profile="public_beta")

    messages = {
        (issue["check"], issue["path"], issue["message"])
        for issue in verification["issues"]
    }
    check = next(item for item in verification["checks"] if item["name"] == "readiness:public_beta")

    assert verification["passed"] is False
    assert check["passed"] is False
    assert (
        "readiness:public_beta",
        "frontier_artifact",
        "frontier release profiles require a generated frontier report",
    ) in messages
