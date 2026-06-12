"""Tests for OpenVoiceCS sealed-test operations manifests."""

from __future__ import annotations

from copy import deepcopy

from src.evaluation.benchmark.sealed import (
    load_sealed_ops_manifest,
    load_sealed_queue_manifest,
    sealed_ops_stats,
    sealed_queue_stats,
    validate_sealed_ops_manifest,
    validate_sealed_ops_manifest_file,
    validate_sealed_queue_manifest,
    validate_sealed_queue_manifest_file,
)
from src.evaluation.benchmark.splits import load_split_manifest


def test_sealed_ops_manifest_validates():
    manifest = load_sealed_ops_manifest()

    assert validate_sealed_ops_manifest_file() == []
    assert validate_sealed_ops_manifest(manifest, split_manifest=load_split_manifest()) == []
    assert sealed_ops_stats(manifest)["pre_submission_access"] is False


def test_sealed_ops_manifest_rejects_weak_access_policy():
    manifest = load_sealed_ops_manifest()
    manifest["custody"]["sealed_ids_revealed"] = True
    manifest["access_policy"]["pre_submission_access"] = True
    manifest["evaluation_protocol"]["max_attempts_per_system"] = 0
    manifest["disclosure_policy"]["publish_prompts_before_evaluation"] = True

    messages = {
        (issue.path, issue.message)
        for issue in validate_sealed_ops_manifest(manifest)
    }

    assert ("custody.sealed_ids_revealed", "must be false") in messages
    assert ("access_policy.pre_submission_access", "must be false") in messages
    assert (
        "evaluation_protocol.max_attempts_per_system",
        "must be an integer >= 1",
    ) in messages
    assert (
        "disclosure_policy.publish_prompts_before_evaluation",
        "must be false",
    ) in messages


def test_sealed_ops_manifest_rejects_sealed_id_leakage():
    manifest = load_sealed_ops_manifest()
    split_manifest = {
        "splits": {
            "public_dev": {"scenario_ids": ["public"], "audio_variant_ids": []},
            "sealed_test": {"scenario_ids": ["sealed"], "audio_variant_ids": []},
        }
    }
    split_commitments = {
        "privacy": {"sealed_test_ids_revealed": True},
        "root_hash": "abc",
    }

    issues = validate_sealed_ops_manifest(
        deepcopy(manifest),
        split_manifest=split_manifest,
        split_commitments=split_commitments,
    )
    messages = {(issue.path, issue.message) for issue in issues}

    assert (
        "split_commitments.privacy.sealed_test_ids_revealed",
        "must be false",
    ) in messages
    assert ("split_commitments.root_hash", "must publish a sha256 root hash") in messages


def test_sealed_evaluator_queue_validates():
    queue = load_sealed_queue_manifest()

    assert validate_sealed_queue_manifest_file() == []
    stats = sealed_queue_stats(queue)
    assert stats["num_submissions"] == 1
    assert stats["num_attempts"] == 1
    assert stats["reference_fixtures"] == 1
    assert stats["official_candidates"] == 0


def test_sealed_evaluator_queue_rejects_exposure_and_attempt_overage():
    queue = load_sealed_queue_manifest()
    submission = queue["submissions"][0]
    submission["official_candidate"] = True
    submission["attempt_limit"] = 1
    submission["attempts"].append(deepcopy(submission["attempts"][0]))
    submission["attempts"][0]["sealed_split_exposure"]["sealed_ids_revealed"] = True
    submission["attempts"][0]["sealed_split_exposure"]["served_by"] = "manual_export"
    queue["queue_policy"]["raw_prompts_exported_to_submitter"] = True

    messages = {
        (issue.path, issue.message)
        for issue in validate_sealed_queue_manifest(queue)
    }

    assert ("queue_policy.raw_prompts_exported_to_submitter", "must be false") in messages
    assert (
        "submissions[0].official_candidate",
        "reference fixtures cannot be official candidates",
    ) in messages
    assert ("submissions[0].attempts", "must not exceed attempt_limit") in messages
    assert (
        "submissions[0].attempts[0].sealed_split_exposure.sealed_ids_revealed",
        "must be false",
    ) in messages
    assert (
        "submissions[0].attempts[0].sealed_split_exposure.served_by",
        "must be hosted_evaluator",
    ) in messages
